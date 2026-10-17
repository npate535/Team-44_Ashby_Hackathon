"""Step 04b: nucleus segmentation + classification on the selected tiles (CELLS_SPEC §2-§5; BUILD_PLAN §F, Task 8).
GPU job (cells.sbatch / cells_spike.sbatch) in the `cells` venv (pipeline/jobs/cells_env.sh).

Backends (settings cells.backend, override --backend), BUILD_PLAN §F order:
  tiatoolbox  TIAToolbox 2.1.3 MultiTaskSegmentor("hovernet_fast-pannuke") with the explicit local weights file
              cells.checkpoint (no network), rcParam torch_compile_mode = "disable", tile mode (patch_mode=False,
              baseline resolution 1.0, auto_get_mask=False), one zarr per tile (docs/research/cells_backends.md §1).
  hovernet    vqdang/hover_net tile mode (`run_infer.py --model_mode=fast --nr_types=6 tile`, imgaug import removed)
              from cells.hovernet_repo with cells.hovernet_checkpoint (§2).
  hovernext   not wired: GPL-3.0, its own conda env (torch 2.1) and an unverified output layout; documented as the
              last resort in cells_backends.md §3 and reported as such if selected.
CellViT is skipped (§F).

Every tile is pre-resized bicubically by measured mpp / cells.model_mpp (inventory.json measured_mpp via
core.config.effective_mpp; ~512 -> 860 px), segmented, converted by pipeline.cells_convert to CELL_JSON (µm units,
contours in original tile pixels) and drawn as an overlay on the original tile (settings cells.overlay_colors).
Outputs per tile: cells/<case_id>/<tile_idx>_overlay.png, then cells/<case_id>/<tile_idx>.json (written last: its
existence marks the tile done when its backend, model and source mpp match the current run, so reruns resume;
--spike always recomputes). Tiles are processed in chunks of cells.chunk_size; a crash loses
at most one chunk; a failing chunk is retried tile by tile, failing tiles are logged and the step exits 1 at the end.

Modes:
  default   tiles of cell_tiles.csv (04a), sharded (--shard k --nshards n: rows k, k+n, ... of the case_id/tile_idx
            order), meta/04b_cells_shard<k>.json (versions, weights path + sha256 + license, seconds per tile).
  --spike   cells.spike_tiles tiles from different cases (one per class while possible; labels pick cases only and
            are never written), no embeddings needed; writes artifacts/eval/cells_spike.md as a review template.
  --verify  every cell_tiles.csv tile has its JSON and overlay; writes meta/04b_cells.json (totals, throughput);
            exit 1 when anything is missing.
--limit N keeps the tiles of the first N cases (sorted case_id). With cells.enabled false the step exits 0 with a
"disabled" meta record.

Portable (Delta-side): numpy, pandas, Pillow, core; backend packages (tiatoolbox, zarr) are imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from core import cli, config, io, schemas
from pipeline import cells_convert as cc

STEP = "04b_cells"


# ---- backends -----------------------------------------------------------------------------------------------------
class Raw:
    """Model-pixel instances of one tile from any backend."""

    def __init__(self, contours: Sequence[Any], centroids: Sequence[Any], types: Sequence[Any], probs: Sequence[Any]):
        self.contours, self.centroids, self.types, self.probs = contours, centroids, types, probs


class TiatoolboxBackend:
    name = "tiatoolbox"

    def __init__(self, cfg: Mapping[str, Any]):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from tiatoolbox import rcParam

        rcParam["torch_compile_mode"] = "disable"  # must precede engine construction (A100 would compile)
        self.cfg = cfg
        self.weights = Path(cfg["cells"]["checkpoint"])
        if not self.weights.is_file():
            raise FileNotFoundError(f"weights missing: {self.weights} (pipeline/jobs/cells_env.sh downloads them)")
        self.engine = self._engine()
        self.rebuilt = False

    def _engine(self) -> Any:
        from tiatoolbox.models import MultiTaskSegmentor

        return MultiTaskSegmentor(model=self.cfg["cells"]["model"], weights=str(self.weights),
                                  batch_size=int(self.cfg["cells"]["batch_size"]), num_workers=0, device="cuda",
                                  verbose=False)

    def version(self) -> str:
        return f"tiatoolbox=={metadata.version('tiatoolbox')}"

    def segment(self, pngs: Sequence[Path], scratch: Path) -> dict[Path, Raw]:
        import zarr

        save_dir = scratch / f"tia_{uuid.uuid4().hex[:8]}"  # must not exist (TIAToolbox mkdir without exist_ok)
        kwargs = dict(images=list(pngs), patch_mode=False, save_dir=save_dir, output_type="zarr",
                      input_resolutions=[{"units": "baseline", "resolution": 1.0}], auto_get_mask=False,
                      overwrite=False)
        try:
            out = self.engine.run(**kwargs)
        except Exception:  # run() re-entrancy is unverified: rebuild the engine once per process and retry
            shutil.rmtree(save_dir, ignore_errors=True)
            if self.rebuilt:
                raise
            self.rebuilt = True
            self.engine = self._engine()
            out = self.engine.run(**kwargs)
        result = {}
        for png in pngs:
            g = zarr.open(str(out[Path(png)] if Path(png) in out else out[str(png)]), mode="r")
            if "type" in g and g["type"].shape[0] > 0:
                result[Path(png)] = Raw(g["contours"][:], g["centroid"][:], g["type"][:], g["prob"][:])
            else:
                result[Path(png)] = Raw([], [], [], [])
        shutil.rmtree(save_dir, ignore_errors=True)
        return result


class HovernetBackend:
    name = "hovernet"

    def __init__(self, cfg: Mapping[str, Any]):
        self.repo = Path(os.path.expanduser(cfg["cells"]["hovernet_repo"]))
        self.ckpt = Path(cfg["cells"]["hovernet_checkpoint"])
        self.batch = int(cfg["cells"]["batch_size"])  # 4 shards share one GPU
        for p in (self.repo / "run_infer.py", self.ckpt):
            if not p.exists():
                raise FileNotFoundError(f"hover_net fallback asset missing: {p} (see cells_backends.md §2)")

    def version(self) -> str:
        try:
            sha = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                 check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            sha = "unknown"
        return f"vqdang/hover_net@{sha}"

    def segment(self, pngs: Sequence[Path], scratch: Path) -> dict[Path, Raw]:
        out_dir = scratch / f"hn_{uuid.uuid4().hex[:8]}"
        in_dir = Path(pngs[0]).parent  # the chunk's input dir holds only these images
        cmd = [sys.executable, "run_infer.py", f"--gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}", "--nr_types=6",
               "--type_info_path=type_info.json", f"--batch_size={self.batch}", "--model_mode=fast",
               f"--model_path={self.ckpt}", "--nr_inference_workers=4", "--nr_post_proc_workers=4",
               "tile", f"--input_dir={in_dir}", f"--output_dir={out_dir}", "--mem_usage=0.1"]
        subprocess.run(cmd, cwd=self.repo, check=True)
        result = {}
        for png in pngs:
            nuc = list(io.read_json(out_dir / "json" / f"{Path(png).stem}.json")["nuc"].values())
            result[Path(png)] = Raw([n["contour"] for n in nuc], [n["centroid"] for n in nuc],
                                    [n["type"] for n in nuc], [n["type_prob"] for n in nuc])
        shutil.rmtree(out_dir, ignore_errors=True)
        return result


class HovernextBackend:
    name = "hovernext"

    def __init__(self, cfg: Mapping[str, Any]):
        raise NotImplementedError("hovernext is not wired (GPL-3.0, own torch 2.1 env, unverified output layout); "
                                  "use tiatoolbox or hovernet, or disable cells (CELLS_SPEC §11)")


BACKENDS: dict[str, Any] = {"tiatoolbox": TiatoolboxBackend, "hovernet": HovernetBackend, "hovernext": HovernextBackend}


# ---- tile lists ---------------------------------------------------------------------------------------------------
def spike_tiles(cases: pd.DataFrame, tiles: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """cells.spike_tiles tiles from different cases, one per class first (labels pick cases only), each the case's
    QC contract-grid tile with the most tissue. Columns: case_id, tile_idx, path."""
    k = int(cfg["cells"]["spike_tiles"])
    rng = np.random.default_rng(int(cfg["seed"]))
    chosen: list[str] = []
    for cls in cfg["classes"]:
        pool = sorted(cases.loc[cases["label"] == cls, "case_id"])
        if pool and len(chosen) < k:
            chosen.append(pool[int(rng.integers(len(pool)))])
    rest = sorted(set(cases["case_id"]) - set(chosen))
    while len(chosen) < k and rest:
        chosen.append(rest.pop(int(rng.integers(len(rest)))))
    t = tiles[(tiles["grid"] == cfg["tiles"]["contract_grid"]) & tiles["case_id"].isin(chosen)].copy()
    t = t[t["qc_pass"].astype(str).str.lower().isin(["1", "true", "1.0"])]
    t["tissue_frac"] = t["tissue_frac"].astype(float)
    t["tile_idx"] = t["tile_idx"].astype(int)
    t = t.sort_values(["case_id", "tissue_frac", "tile_idx"], ascending=[True, False, True]).groupby("case_id").head(1)
    return t[["case_id", "tile_idx", "path"]].reset_index(drop=True)


def tile_done(root: Path, case_id: str, tile_idx: int, identity: Mapping[str, Any]) -> bool:
    """JSON + overlay exist and the JSON came from the same backend, model and source mpp (a backend switch after a
    failed spike review must not keep the old outputs)."""
    jpath = schemas.cell_json(root, case_id, tile_idx)
    if not (jpath.is_file() and schemas.cell_overlay_png(root, case_id, tile_idx).is_file()):
        return False
    try:
        doc = io.read_json(jpath)
    except (OSError, ValueError):
        return False
    return all(doc.get(k) == v for k, v in identity.items())


def todo_tiles(rows: pd.DataFrame, root: Path, shard: int, nshards: int, force: bool,
               identity: Mapping[str, Any]) -> pd.DataFrame:
    rows = rows.sort_values(["case_id", "tile_idx"]).reset_index(drop=True).iloc[shard::nshards]
    if force:
        return rows.reset_index(drop=True)
    done = [tile_done(root, c, int(i), identity) for c, i in zip(rows["case_id"], rows["tile_idx"])]
    return rows[~np.asarray(done, dtype=bool)].reset_index(drop=True)


def run_identity(cfg: Mapping[str, Any], backend_name: str, mpp_src: float) -> dict[str, Any]:
    return {"backend_name": backend_name, "model": cfg["cells"]["model"], "mpp_source": mpp_src}


# ---- processing ---------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def process_chunk(chunk: pd.DataFrame, backend: Any, root: Path, scratch: Path, mpp_src: float,
                  cfg: Mapping[str, Any], fcfg: Mapping[str, Any], meta_common: Mapping[str, Any]) -> dict[str, int]:
    """Resize -> segment -> convert -> overlay, then JSON last per tile. Returns nucleus counts by type."""
    tmp_in = scratch / f"in_{uuid.uuid4().hex[:8]}"
    tmp_in.mkdir(parents=True)
    names = cc.type_names(fcfg)
    rules = fcfg["rules"]
    counts: dict[str, int] = {t: 0 for t in cfg["cells"]["types"]}
    try:
        geo, pngs = {}, []
        for t in chunk.itertuples(index=False):
            png = tmp_in / f"{t.case_id}_{int(t.tile_idx)}.png"  # unique stem -> unique backend output
            geo[png] = cc.resize_to_model(schemas.resolve(root, t.path), png, mpp_src, float(cfg["cells"]["model_mpp"]))
            pngs.append(png)
        raw = backend.segment(pngs, scratch)
        for t, png in zip(chunk.itertuples(index=False), pngs):
            g, r = geo[png], raw[png]
            recs, n_untyped = cc.nuclei_records(r.contours, r.centroids, r.types, r.probs, mpp_model=g["mpp_model"],
                                                scale=g["scale"], hw_model=g["hw_model"], names=names,
                                                simplify_px=float(rules["simplify_px"]), edge_px=int(rules["edge_px"]))
            with Image.open(schemas.resolve(root, t.path)) as tile:
                overlay = cc.render_overlay(tile, recs, cfg["cells"]["overlay_colors"])
            io.save_image(schemas.cell_overlay_png(root, t.case_id, int(t.tile_idx)), overlay)
            meta = {**meta_common, "case_id": t.case_id, "tile_idx": int(t.tile_idx), "mpp_model": g["mpp_model"],
                    "mpp_source": mpp_src, "scale": g["scale"], "tile_px": g["tile_px"],
                    "contour_frame": "original_tile_px", "type_prob_kind": fcfg["type_prob_kind"]}
            cc.write_tile_json(schemas.cell_json(root, t.case_id, int(t.tile_idx)), meta, recs, n_untyped)
            for rec in recs:
                counts[rec["type"]] += 1
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
    return counts


def spike_markdown(rows: pd.DataFrame, root: Path, info: Mapping[str, Any]) -> str:
    lines = [
        "# Cells spike — overlay review (CELLS_SPEC §2, C0)",
        "",
        "Status: **TEMPLATE — not yet reviewed.** When filled, mark it **\"reviewed by Claude (AI), not a pathologist\"**.",
        "Research prototype — not for clinical use. Pass if nuclear outlines are mostly on nuclei and tumor vs lymphocyte",
        "labels look plausible. Overlay colors: " + ", ".join(f"{k} {v}" for k, v in info["colors"].items()) + ".",
        "",
        f"- backend: {info['backend']} · model: {info['model']} · weights: {info['weights']} (sha256 {info['weights_sha256']})",
        f"- source mpp {info['mpp_source']} → model mpp {info['model_mpp']} (scale ≈ {info['scale']:.3f})",
        f"- engine setup {info['setup_s']:.1f} s; {info['s_per_tile']:.2f} s/tile over {info['n_tiles']} tiles",
        f"- gate (cells_backends.md §1.5): 1,200 tiles × s/tile / {info['shards']} shards = {info['gate_min']:.1f} min "
        f"({'NOT MEASURED (no tile segmented)' if info['gate_min'] != info['gate_min'] else 'OK' if info['gate_min'] <= 35 else 'TOO SLOW: raise shards or cut tiles'})",
        "",
        "| # | case_id | tile_idx | overlay | nuclei | per type | outlines on nuclei? | tumor vs lymphocyte plausible? | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(rows.itertuples(index=False), start=1):
        jpath = schemas.cell_json(root, t.case_id, int(t.tile_idx))
        if jpath.is_file():
            doc = io.read_json(jpath)
            per = {}
            for n in doc["nuclei"]:
                per[n["type"]] = per.get(n["type"], 0) + 1
            n_nuc, per_txt = doc["n_nuclei"], ", ".join(f"{k} {v}" for k, v in sorted(per.items()))
        else:
            n_nuc, per_txt = "missing", "missing"
        ov = schemas.cell_overlay_png(root, t.case_id, int(t.tile_idx)).relative_to(root).as_posix()
        lines.append(f"| {i} | {t.case_id} | {int(t.tile_idx)} | {ov} | {n_nuc} | {per_txt} |  |  |  |")
    lines += ["", "Reviewer:  · Date:  · Overall: pass / fail · Notes:", ""]
    return "\n".join(lines)


def verify(root: Path, cfg: Mapping[str, Any], ctx: cli.StepContext, backend_name: str) -> int:
    sel = io.read_csv(schemas.cell_tiles_csv(root), dtype={"case_id": str}, keep_default_na=False)
    keep = cli.limit_case_ids(sel["case_id"], ctx.limit)
    sel = sel[sel["case_id"].isin(keep)]
    identity = run_identity(cfg, backend_name, config.effective_mpp(cfg, schemas.inventory_json(root)))
    missing = [(c, int(i)) for c, i in zip(sel["case_id"], sel["tile_idx"])
               if not tile_done(root, c, int(i), identity)]
    shards = [io.read_json(p) for p in sorted((root / "meta").glob(f"{STEP}_shard*.json"))]
    secs = sum(s.get("summary", {}).get("seconds_segmenting", 0.0) for s in shards)
    n_done = sum(s.get("summary", {}).get("tiles_done", 0) for s in shards)
    n_nuclei = 0
    for c, i in zip(sel["case_id"], sel["tile_idx"]):
        p = schemas.cell_json(root, c, int(i))
        if p.is_file():
            n_nuclei += int(io.read_json(p)["n_nuclei"])
    summary = {"tiles_selected": int(len(sel)), "tiles_missing": len(missing), "missing_examples": missing[:5],
               "nuclei_total": n_nuclei, "shards": len(shards),
               "seconds_per_tile_process": round(secs / n_done, 3) if n_done else None,
               "backends": sorted({s.get("summary", {}).get("backend", "?") for s in shards})}
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg, inputs=[schemas.cell_tiles_csv(root)],
                  outputs=[root / "cells"], args={"verify": True, "limit": ctx.limit}, synthetic=ctx.synthetic,
                  extra={"summary": summary})
    print(json.dumps(summary, indent=1))
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Nucleus segmentation on selected tiles (CELLS_SPEC §5).")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=None, help="override settings cells.backend")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--scratch", default=None, help="scratch dir (default $TMPDIR or <data_root>/.cells_scratch)")
    parser.add_argument("--spike", action="store_true", help="segment cells.spike_tiles tiles + write cells_spike.md")
    parser.add_argument("--verify", action="store_true", help="check every selected tile has JSON + overlay")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    if not cfg["cells"]["enabled"]:
        io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg, inputs=[], outputs=[],
                      args={"limit": ctx.limit}, synthetic=ctx.synthetic, extra={"status": "disabled"})
        print(json.dumps({"status": "disabled"}))
        return 0
    if args.verify:
        return verify(root, cfg, ctx, args.backend or cfg["cells"]["backend"])
    if not 0 <= args.shard < args.nshards:
        print(f"ERROR {STEP}: --shard must be in [0, --nshards)", file=sys.stderr)
        return 2
    scratch = Path(args.scratch or os.environ.get("TMPDIR") or root / ".cells_scratch") / f"04b_{os.getpid()}_{args.shard}"
    try:
        fcfg = cc.load_feature_config(cfg)
        mpp_src = config.effective_mpp(cfg, schemas.inventory_json(root))
        if args.spike:
            cases = io.read_csv(schemas.cases_csv(root), dtype=str, keep_default_na=False)
            tiles = io.read_csv(schemas.tiles_csv(root), dtype=str, keep_default_na=False)
            rows = spike_tiles(cases, tiles, cfg)
        else:
            rows = io.read_csv(schemas.cell_tiles_csv(root), dtype={"case_id": str, "path": str}, keep_default_na=False)
            schemas.validate_frame(schemas.CELL_TILES, rows)
        keep = cli.limit_case_ids(rows["case_id"], ctx.limit)
        rows = rows[rows["case_id"].isin(keep)]
        name = args.backend or cfg["cells"]["backend"]
        identity = run_identity(cfg, name, mpp_src)
        # --spike always re-measures (a rerun must never report a throughput gate from nothing)
        todo = todo_tiles(rows, root, args.shard, args.nshards, ctx.force or args.spike, identity)
        weights = Path(cfg["cells"]["checkpoint"]) if name == "tiatoolbox" else Path(cfg["cells"]["hovernet_checkpoint"])
        weights_sha = sha256_file(weights) if name == "tiatoolbox" and len(todo) else None
        if weights_sha is not None and weights_sha != cfg["cells"]["weights_sha256"]:
            raise ValueError(f"{weights} sha256 {weights_sha} != settings cells.weights_sha256")
        t0 = time.time()
        backend = BACKENDS[name](cfg) if len(todo) else None
        setup_s = time.time() - t0
    except (ValueError, FileNotFoundError, KeyError, NotImplementedError, ImportError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2
    meta_common = {**identity, "backend": backend.version() if backend else name, "model": cfg["cells"]["model"],
                   "weights": str(weights), "weights_sha256": cfg["cells"]["weights_sha256"] if name == "tiatoolbox" else None}
    counts = {t: 0 for t in cfg["cells"]["types"]}
    failures, done, seg_s = [], 0, 0.0
    chunk = int(cfg["cells"]["chunk_size"])
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk]
        t1 = time.time()
        try:
            for k, v in process_chunk(part, backend, root, scratch, mpp_src, cfg, fcfg, meta_common).items():
                counts[k] += v
            done += len(part)
            seg_s += time.time() - t1
            print(f"[shard {args.shard}] {done}/{len(todo)} tiles, {(time.time() - t1) / len(part):.2f} s/tile", flush=True)
        except Exception as exc:  # noqa: BLE001 - retry tile by tile so one bad tile never sinks its chunk
            print(f"[shard {args.shard}] chunk failed ({type(exc).__name__}: {exc}); retrying its tiles one by one",
                  file=sys.stderr, flush=True)
            for i in range(len(part)):
                one = part.iloc[i:i + 1]
                try:
                    for k, v in process_chunk(one, backend, root, scratch, mpp_src, cfg, fcfg, meta_common).items():
                        counts[k] += v
                    done += 1
                except Exception as tile_exc:  # noqa: BLE001 - logged; exit 1 at the end
                    failures.append({"tile": f"{one.iloc[0]['case_id']}:{int(one.iloc[0]['tile_idx'])}",
                                     "error": f"{type(tile_exc).__name__}: {tile_exc}"})
            seg_s += time.time() - t1
    shutil.rmtree(scratch, ignore_errors=True)
    summary = {"backend": meta_common["backend"], "shard": args.shard, "nshards": args.nshards,
               "tiles_selected_in_shard": int(len(rows.iloc[args.shard::args.nshards])), "tiles_todo": int(len(todo)),
               "tiles_done": done, "failed_tiles": failures, "nuclei_by_type": counts, "setup_seconds": round(setup_s, 1),
               "seconds_segmenting": round(seg_s, 1), "seconds_per_tile": round(seg_s / done, 3) if done else None,
               "mpp_source": mpp_src, "model_mpp": cfg["cells"]["model_mpp"]}
    step = f"{STEP}_spike" if args.spike else f"{STEP}_shard{args.shard}"
    io.write_meta(schemas.step_meta_json(root, step), step=step, cfg=cfg,
                  inputs=[schemas.cell_tiles_csv(root) if not args.spike else schemas.tiles_csv(root), str(weights)],
                  outputs=[root / "cells"], args={"limit": ctx.limit, "force": ctx.force, "backend": name,
                                                  "shard": args.shard, "nshards": args.nshards, "spike": args.spike},
                  synthetic=ctx.synthetic,
                  extra={"summary": summary, "weights": str(weights), "weights_sha256_expected": meta_common["weights_sha256"],
                         "weights_sha256_actual": weights_sha,
                         "license": cfg["cells"]["weights_license"]})
    if args.spike:
        spt = summary["seconds_per_tile"] or float("nan")
        info = {**meta_common, "colors": cfg["cells"]["overlay_colors"], "mpp_source": mpp_src,
                "model_mpp": cfg["cells"]["model_mpp"], "scale": mpp_src / float(cfg["cells"]["model_mpp"]),
                "setup_s": setup_s, "s_per_tile": spt, "n_tiles": done, "shards": int(cfg["cells"]["shards"]),
                "gate_min": 1200 * spt / int(cfg["cells"]["shards"]) / 60}
        io.atomic_write_text(schemas.eval_asset(ctx.artifacts_root, "cells_spike.md"), spike_markdown(rows, root, info))
    print(json.dumps(summary, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
