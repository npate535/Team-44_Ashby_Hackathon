"""GPU embedding engine: embed_manifest.csv -> emb/<encoder>/<set>_<head>.npy + _index.csv + meta.json (Task 3, §A6-§A7).

    python -m pipeline.02_embed --sets bach512,nct --encoders conch,musk [--limit 5] [--force]
    python -m pipeline.02_embed --data-root <dir> --smoke-manifest      # smoke.sbatch: build a tiny manifest, exit

Per (encoder, set): manifest rows sorted by (item_id, view) are cut into fixed chunks of engine.chunk_rows; each
chunk is embedded by a DataLoader (open `path`, crop x,y,w,h, degrade.apply for degraded rows, dihedral `view`,
encoder preprocessing) and saved as a float16 npz named by a hash of its row IDs, so a killed job resumes where it
stopped. When every chunk exists the final files are written atomically (core.schemas EMB_INDEX / EMB_NPY):

    <set>_<head>.npy + _index.csv               one row per item_id: mean over its views (TTA); align rows are
                                                re-normalized after averaging, probe rows are raw means
    <set>_<head>_views.npy + _views_index.csv   one row per (item, view) for every split when items have > 1 view
                                                (ruling M9: cal/test view-0 rows are the clean baseline for degraded
                                                view-0 reads; train views are the augmentation copies)

Order inside one tile: crop -> degrade -> view. Degrading the cropped tile (not the whole image) keeps the work per
row small; JPEG blocks still align because tile edges are multiples of 8, and blur differs only within ~3 sigma of
the tile border.

Aborting checks (exit 2 with a clear message): head dims, no NaN/Inf, align L2 norms ~1, align (and CONCH probe)
== the official adapter on the first batch (parity; a failing one-pass CONCH/MUSK path falls back to the adapter's
own calls, logged), optional autocast-vs-fp32 cosine (--precision-check), and NCT zero-shot accuracy above the
encoders.yaml floor (prompts_nct.json, untemplated) for text-aligned encoders. Throughput (images/s) is logged per
(encoder, set) and stored in meta.json for sizing later jobs.

Delta-portable: Python >= 3.10 with the pytorch-conda/2.8 packages (+ organizer installs). Never on a login node.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from core import cli, config, io, schemas
from pipeline import encoders as E

STEP = "pipeline.02_embed"
META_STEP = "02_embed"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_STR_COLS = ("row_id", "item_id", "set", "case_id", "split", "grid", "path", "degrade_type", "label", "group")
_INT_COLS = ("tile_idx", "x", "y", "w", "h", "view", "degrade_severity", "degrade_seed")
_INT_DEFAULT = {"tile_idx": -1, "degrade_severity": 0, "degrade_seed": 0}
_INDEX_COLS = [c.name for c in schemas.EMB_INDEX.columns]
BACH_PIXEL_SETS = ("bach", "deg", "comp", "wsi")


class CheckFailed(RuntimeError):
    """An aborting check failed (dims, norms, parity, NaN/Inf, NCT zero-shot)."""


# ---- manifest ----------------------------------------------------------------------------------------------------------
def read_manifest(path: str | Path) -> pd.DataFrame:
    """embed_manifest.csv with empty strings kept as '' and integer columns as ints (empty -> -1 / 0)."""
    df = pd.read_csv(path, dtype={c: str for c in _STR_COLS}, keep_default_na=False)
    schemas.validate_frame(schemas.EMBED_MANIFEST, df)
    for col in _INT_COLS:
        values = df[col].replace("", np.nan) if df[col].dtype == object else df[col]
        if col in _INT_DEFAULT:
            values = values.fillna(_INT_DEFAULT[col])
        elif pd.isna(values).any():
            raise ValueError(f"{path}: column {col} has empty values")
        df[col] = pd.to_numeric(values).astype(int)
    if ((df["view"] < 0) | (df["view"] > 7)).any():
        raise ValueError(f"{path}: view must be 0-7")
    if df.duplicated(["item_id", "view"]).any():
        raise ValueError(f"{path}: an item_id lists the same view twice")
    return df


def select_rows(df: pd.DataFrame, set_name: str, limit: int | None, max_views: int | None = None) -> pd.DataFrame:
    rows = df[df["set"] == set_name]
    if max_views is not None:
        rows = rows[rows["view"] < max_views]
    if limit is not None:
        rows = rows[rows["case_id"].isin(cli.limit_case_ids(rows["case_id"], limit))]
    return rows.sort_values(["item_id", "view"], kind="stable").reset_index(drop=True)


ROW_KEY_COLS = ["row_id", "path", "x", "y", "w", "h", "view", "degrade_type", "degrade_severity", "degrade_seed"]


def encoder_fingerprint(name: str, spec: Mapping[str, Any]) -> str:
    """What produced an embedding, known before the model loads: resolved revision, loader, preprocessing, heads."""
    info = {"encoder": name, "revision": E._revision(spec), "loader": spec["loader"], "spec": spec.get("spec"),
            "repo": spec.get("repo"), "preprocess": E.preprocess_info(spec), "heads": spec["heads"],
            "autocast": spec["autocast"], "output": spec.get("output"), "timm_kwargs": spec.get("timm_kwargs")}
    return hashlib.sha1(json.dumps(info, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _rows_digest(rows: pd.DataFrame, fingerprint: str) -> str:
    digest = hashlib.sha1(fingerprint.encode("utf-8"))
    for record in rows[ROW_KEY_COLS].itertuples(index=False, name=None):
        digest.update(("\x1f".join(map(str, record)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def plan_key(rows: pd.DataFrame, fingerprint: str) -> str:
    """Identity of a full (encoder, set) plan: every row's pixels-defining fields + the encoder fingerprint."""
    return _rows_digest(rows, fingerprint)[:20]


def plan_chunks(rows: pd.DataFrame, chunk_rows: int, fingerprint: str = "") -> list[tuple[int, str, pd.DataFrame]]:
    """[(k, hash, rows)] in order. The hash covers the chunk's rows (ids, path, crop, view, degradation) and the
    encoder fingerprint, so a changed manifest, revision or preprocessing never reuses a stale chunk."""
    out = []
    for k, start in enumerate(range(0, len(rows), chunk_rows)):
        part = rows.iloc[start:start + chunk_rows]
        out.append((k, _rows_digest(part, fingerprint)[:12], part))
    return out


def resolve_device(requested: str | None) -> str:
    """cuda when available. Inside a Slurm GPU allocation a missing CUDA device is fatal (never a silent CPU run)."""
    import torch

    gpu_job = bool(os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE"))
    if requested is None:
        if torch.cuda.is_available():
            return "cuda"
        if gpu_job:
            raise CheckFailed("this Slurm job has GPUs allocated but torch sees no CUDA device")
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise CheckFailed(f"--device {requested} requested but torch sees no CUDA device")
    return requested


# ---- dataset -----------------------------------------------------------------------------------------------------------
class TileDataset:
    """Map-style dataset over manifest rows -> preprocessed tensors. Decoded source images are cached per worker
    (rows are sorted, so the tiles and views of one image are adjacent)."""

    def __init__(self, rows: pd.DataFrame, root: Path, preprocess: Any, cfg: Mapping[str, Any], cache_size: int):
        self.records = list(rows[["path", "x", "y", "w", "h", "view", "degrade_type", "degrade_severity",
                                  "degrade_seed"]].itertuples(index=False, name=None))
        self.root = Path(root)
        self.preprocess = preprocess
        self.cfg = cfg
        self.cache_size = max(1, int(cache_size))
        self._cache: collections.OrderedDict[str, Image.Image] = collections.OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _open(self, stored: str) -> Image.Image:
        if stored in self._cache:
            self._cache.move_to_end(stored)
            return self._cache[stored]
        path = schemas.resolve(self.root, stored)
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except Exception as exc:  # unreadable input is fatal: skipping a row would misalign every array
            raise RuntimeError(f"cannot read image {path}: {exc}") from exc
        self._cache[stored] = image
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image

    def image(self, i: int) -> Image.Image:
        path, x, y, w, h, view, dtype, severity, seed = self.records[i]
        tile = E.crop_box(self._open(path), int(x), int(y), int(w), int(h))
        if dtype:
            from pipeline import degrade

            tile = degrade.apply(tile, dtype, int(severity), int(seed), self.cfg)
        return E.dihedral(tile, int(view))

    def __getitem__(self, i: int) -> Any:
        return self.preprocess(self.image(i))


def make_loader(dataset: TileDataset, batch: int, workers: int, pin: bool) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin,
                      persistent_workers=False)


# ---- checks ------------------------------------------------------------------------------------------------------------
def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def check_finite(name: str, out: Mapping[str, np.ndarray]) -> None:
    for head, arr in out.items():
        if not np.isfinite(arr).all():
            raise CheckFailed(f"{name}/{head}: NaN or Inf in the embeddings")


def check_norms(name: str, spec: Mapping[str, Any], out: Mapping[str, np.ndarray], tol: float) -> None:
    for head, info in spec["heads"].items():
        if info["normalized"]:
            dev = np.abs(np.linalg.norm(out[head].astype(np.float64), axis=1) - 1.0)
            if dev.max() > tol:
                raise CheckFailed(f"{name}/{head}: L2 norms deviate from 1 by up to {dev.max():.4f} (tol {tol})")


def parity(enc: E.Encoder, batch: Any, ours: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Minimum row cosine of our heads vs the official adapter's outputs on the same batch."""
    ref = enc.reference(batch) if enc.reference is not None else {}
    return {head: float(rowwise_cosine(ours[head], np.asarray(r, np.float32)).min()) for head, r in ref.items()}


def run_parity_check(enc: E.Encoder, batch: Any, ours: dict[str, np.ndarray], min_cos: float,
                     log: list[str]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Parity on the first batch. A failing one-pass CONCH/MUSK path switches to the adapter's own calls and is
    re-checked; failing that, CheckFailed."""
    scores = parity(enc, batch, ours)
    if scores and min(scores.values()) < min_cos and enc.set_one_pass is not None:
        log.append(f"{enc.name}: one-pass parity {scores} below {min_cos}; falling back to the adapter's two calls")
        enc.set_one_pass(False)
        ours = enc.embed(batch)
        scores = parity(enc, batch, ours)
    bad = {h: s for h, s in scores.items() if s < min_cos}
    if bad:
        raise CheckFailed(f"{enc.name}: align/probe != official adapter (min cosine {bad}, need >= {min_cos})")
    return ours, scores


def nct_prompts() -> dict[str, str]:
    with open(E.PROMPTS_NCT, encoding="utf-8") as handle:
        return json.load(handle)


def zero_shot_accuracy(align: np.ndarray, labels: Sequence[str], text: np.ndarray, classes: Sequence[str]) -> float:
    pred = np.asarray(classes)[np.argmax(np.asarray(align, np.float32) @ np.asarray(text, np.float32).T, axis=1)]
    return float(np.mean(pred == np.asarray(labels)))


def nct_zero_shot(enc: E.Encoder, align: np.ndarray, labels: Sequence[str], ecfg: Mapping[str, Any]) -> dict[str, Any]:
    prompts = nct_prompts()
    keep = np.isin(np.asarray(labels), list(prompts))
    n = int(keep.sum())
    floor = float(enc.spec["nct_floor"])
    result: dict[str, Any] = {"n": n, "floor": floor, "chance": 1.0 / len(prompts)}
    if n < int(ecfg["checks"]["nct_min_items"]):
        return {**result, "status": "skipped", "reason": f"only {n} labeled NCT items"}
    classes = list(prompts)
    text = E._l2(enc.encode_texts([prompts[c] for c in classes]))
    acc = zero_shot_accuracy(align[keep], np.asarray(labels)[keep], text, classes)
    return {**result, "accuracy": round(acc, 4), "status": "pass" if acc >= floor else "fail"}


# ---- reduction and final files ---------------------------------------------------------------------------------------------
def reduce_views(rows: pd.DataFrame, arrays: Mapping[str, np.ndarray], spec: Mapping[str, Any]
                 ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-item TTA means; rows must be sorted by item_id (contiguous). Align heads are re-normalized."""
    item = rows["item_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, item[1:] != item[:-1]])
    if len(np.unique(item)) != len(starts):
        raise ValueError("rows are not grouped by item_id")
    counts = np.diff(np.r_[starts, len(item)])
    reduced: dict[str, np.ndarray] = {}
    for head, arr in arrays.items():
        mean = np.add.reduceat(arr.astype(np.float32), starts, axis=0) / counts[:, None]
        if spec["heads"][head]["normalized"]:
            mean = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
        reduced[head] = mean.astype(np.float16)
    first = rows.iloc[starts].reset_index(drop=True)
    index = first.assign(view=-1, n_views=counts)
    return _index_frame(index), reduced


def _index_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "n_views" not in out:
        out["n_views"] = 1
    out.insert(0, "emb_row", np.arange(len(out)))
    out = out[_INDEX_COLS]
    schemas.validate_frame(schemas.EMB_INDEX, out)
    return out


def set_done(root: Path, encoder: str, set_name: str, heads: Sequence[str], rows: pd.DataFrame, key: str,
             fingerprint: str, limited: bool) -> bool:
    """Finished outputs for this exact plan (meta plan_key), or — for a --limit run — finished outputs of the same
    encoder fingerprint that already cover the limited items (a smoke run never shrinks a full set). A set whose
    recorded NCT zero-shot check failed is never done."""
    path = schemas.emb_meta_json(root, encoder)
    meta = io.read_json(path) if path.is_file() else {}
    info = meta.get("sets", {}).get(set_name)
    if not info or meta.get("checks", {}).get(set_name, {}).get("nct_zero_shot", {}).get("status") == "fail":
        return False
    items = existing_items(root, encoder, set_name, heads)
    if items is None:
        return False
    if info.get("plan_key") == key:
        return True
    return limited and info.get("fingerprint") == fingerprint and set(rows["item_id"]) <= items


def existing_items(root: Path, encoder: str, set_name: str, heads: Sequence[str]) -> set[str] | None:
    """Item IDs of finished per-item outputs (all heads present and consistent), else None."""
    found: set[str] | None = None
    for head in heads:
        npy, idx = schemas.emb_npy(root, encoder, set_name, head), schemas.emb_index_csv(root, encoder, set_name, head)
        if not io.outputs_exist(npy, idx):
            return None
        items = set(pd.read_csv(idx, dtype={"item_id": str}, keep_default_na=False)["item_id"])
        if found is not None and items != found:
            return None
        found = items
    return found


def update_meta(root: Path, enc_info: Mapping[str, Any], cfg: Mapping[str, Any], *, synthetic: bool,
                set_name: str, set_info: Mapping[str, Any], checks: Mapping[str, Any]) -> Path:
    path = schemas.emb_meta_json(root, enc_info["encoder"])
    meta = io.read_json(path) if path.is_file() else {}
    prov = io.provenance(cfg, synthetic=synthetic)
    sets = dict(meta.get("sets", {}))
    sets[set_name] = dict(set_info)
    all_checks = dict(meta.get("checks", {}))
    all_checks[set_name] = dict(checks)
    preprocess = dict(enc_info["preprocess"])
    preprocess["mpp"] = config.effective_mpp(cfg, schemas.inventory_json(root))
    sets[set_name].update(revision=enc_info["revision"], one_pass=enc_info.get("one_pass"),
                          compute_dtype=enc_info["compute_dtype"], git_sha=prov["git_sha"],
                          created_utc=prov["created_utc"])
    revisions = sorted({str(v.get("revision")) for v in sets.values() if v.get("revision")})
    meta.update({
        "encoder": enc_info["encoder"], "model_card": enc_info["model_card"], "revision": enc_info["revision"],
        "repo": enc_info["repo"], "license": enc_info["license"], "loader": enc_info["loader"],
        "heads": enc_info["heads"], "sets": sets, "preprocess": preprocess, "dtype": "float16",
        "compute_dtype": enc_info["compute_dtype"], "one_pass": enc_info.get("one_pass"),
        "testing_only": enc_info["testing_only"], "checks": all_checks, "revisions_in_sets": revisions,
        "created_utc": prov["created_utc"], "git_sha": prov["git_sha"], "config_hash": prov["config_hash"],
        "versions": prov["versions"], "synthetic": bool(synthetic or meta.get("synthetic", False)),
    })
    problems = schemas.EMB_META.problems(meta)
    if problems:
        raise ValueError("; ".join(problems))
    return io.write_json(path, meta)


# ---- engine --------------------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, ctx: cli.StepContext, ecfg: Mapping[str, Any], device: str, *, workers: int | None,
                 batch: int | None, precision_check: bool, log: Any = print,
                 max_views: Mapping[str, int] | None = None):
        self.ctx, self.ecfg, self.device = ctx, ecfg, device
        self.max_views = dict(max_views or {})
        self.workers = int(ecfg["engine"]["num_workers"]) if workers is None else workers
        self.batch = batch
        self.precision_check = precision_check
        self.log = log
        self.notes: list[str] = []
        self.outputs: list[Path] = []

    def run_encoder(self, name: str, manifest: pd.DataFrame, sets: Sequence[str]) -> dict[str, Any]:
        spec = E.encoder_spec(self.ecfg, name)
        heads = list(spec["heads"])
        report: dict[str, Any] = {}
        enc: E.Encoder | None = None
        first_batch_checked = False
        for set_name in sets:
            rows = select_rows(manifest, set_name, self.ctx.limit, self.max_views.get(set_name))
            if rows.empty:
                report[set_name] = {"status": "no rows"}
                continue
            fingerprint = encoder_fingerprint(name, spec)
            key = plan_key(rows, fingerprint)
            if not self.ctx.force and set_done(self.ctx.data_root, name, set_name, heads, rows, key, fingerprint,
                                               self.ctx.limit is not None):
                self.log(f"[{name}/{set_name}] done: {rows['item_id'].nunique()} items already embedded for this "
                         f"plan (use --force to redo)")
                report[set_name] = {"status": "skipped (done)"}
                continue
            if enc is None:
                t0 = time.time()
                enc = E.load_encoder(name, self.ecfg, self.device)
                self.log(f"[{name}] loaded in {time.time() - t0:.1f}s: revision {enc.info['revision']}, "
                         f"compute {enc.info['compute_dtype']}, heads {enc.info['heads']}")
                for note in enc.notes:
                    self.log(f"[{name}] note: {note}")
            checks, first_batch_checked = self._embed_set(enc, set_name, rows, first_batch_checked, fingerprint, key)
            report[set_name] = {"status": "embedded", **checks}
        return report

    def _embed_set(self, enc: E.Encoder, set_name: str, rows: pd.DataFrame, checked: bool, fingerprint: str,
                   key: str) -> tuple[dict[str, Any], bool]:
        root, ecfg, name = self.ctx.data_root, self.ecfg, enc.name
        tol = float(ecfg["checks"]["align_norm_tol"])
        chunk_dir = schemas.emb_dir(root, name) / ".chunks" / set_name
        chunks = plan_chunks(rows, int(ecfg["engine"]["chunk_rows"]), fingerprint)
        bsz = self.batch or E.batch_size(ecfg, name)
        pin = bool(ecfg["engine"]["pin_memory"]) and self.device.startswith("cuda")
        checks: dict[str, Any] = {}
        n_done, t_embed = 0, 0.0
        for k, digest, part in chunks:
            path = chunk_dir / f"{k:05d}_{digest}.npz"
            if path.is_file() and not self.ctx.force:
                continue
            ds = TileDataset(part, root, enc.preprocess, self.ctx.cfg, int(ecfg["engine"]["image_cache"]))
            outs: dict[str, list[np.ndarray]] = {h: [] for h in enc.heads}
            t0 = time.time()
            for batch in make_loader(ds, bsz, self.workers, pin):
                out = enc.embed(batch)
                if not checked:
                    out, scores = run_parity_check(enc, batch, out, float(ecfg["checks"]["parity_min_cosine"]),
                                                   self.notes)
                    checks["parity_min_cosine"] = scores
                    if self.precision_check and enc.spec["autocast"] and self.device.startswith("cuda"):
                        fp32 = enc.embed(batch, use_autocast=False)
                        prec = {h: float(rowwise_cosine(out[h], fp32[h]).min()) for h in out}
                        checks["precision_min_cosine"] = prec
                        need = float(ecfg["checks"]["precision_min_cosine"])
                        if min(prec.values()) < need:
                            raise CheckFailed(f"{name}: {enc.spec['autocast']} autocast vs fp32 cosine {prec} < {need}")
                    checked = True
                    self.log(f"[{name}/{set_name}] first-batch checks: {checks}")
                check_finite(name, out)
                check_norms(name, enc.spec, out, tol)
                for h, arr in out.items():
                    outs[h].append(arr.astype(np.float16))
            t_embed += time.time() - t0
            n_done += len(part)
            io.save_npz(path, row_id=np.asarray(part["row_id"].tolist(), dtype=str),
                        one_pass=np.asarray(bool(enc.info.get("one_pass", True))),
                        **{h: np.concatenate(v) for h, v in outs.items()})
            self.log(f"[{name}/{set_name}] chunk {k + 1}/{len(chunks)}: {len(part)} rows, "
                     f"{len(part) / max(time.time() - t0, 1e-9):.1f} img/s")
        throughput = n_done / t_embed if t_embed > 0 else None
        if throughput is not None:
            self.log(f"[{name}/{set_name}] THROUGHPUT {throughput:.1f} images/s over {n_done} rows "
                     f"(batch {bsz}, {self.workers} workers, {self.device})")
        arrays = self._load_chunks(chunks, chunk_dir, list(enc.heads))
        self._write_final(enc, set_name, rows, arrays)
        if set_name == "nct" and enc.text_aligned:
            align = np.load(schemas.emb_npy(root, name, set_name, "align")).astype(np.float32)
            idx = pd.read_csv(schemas.emb_index_csv(root, name, set_name, "align"), dtype={"label": str},
                              keep_default_na=False)
            checks["nct_zero_shot"] = nct_zero_shot(enc, align, idx["label"].tolist(), ecfg)
            self.log(f"[{name}/nct] zero-shot: {checks['nct_zero_shot']}")
        model_mpp = None
        if set_name.startswith(BACH_PIXEL_SETS):  # source pixels at the BACH (or resampled WSI) µm/px
            model_mpp = round(config.effective_mpp(self.ctx.cfg, schemas.inventory_json(root)) * float(
                np.median(rows["w"])) / float(enc.spec["input_px"]), 4)
        n_views = int(rows.groupby("item_id")["view"].nunique().max())
        set_info = {"n_items": int(rows["item_id"].nunique()), "n_views": n_views, "n_rows": int(len(rows)),
                    "views_file": bool(self._has_views(rows)), "model_mpp": model_mpp,
                    "throughput_img_s": None if throughput is None else round(throughput, 2),
                    "finished_utc": io.utc_now(), "plan_key": key, "fingerprint": fingerprint}
        update_meta(root, enc.info, self.ctx.cfg, synthetic=self.ctx.synthetic, set_name=set_name,
                    set_info=set_info, checks=checks)
        if not bool(self.ecfg["engine"]["keep_chunks"]):
            shutil.rmtree(chunk_dir, ignore_errors=True)
        self.notes.extend(n for n in enc.notes if n not in self.notes)
        zs = checks.get("nct_zero_shot", {})
        if zs.get("status") == "fail":
            raise CheckFailed(f"{name}: NCT zero-shot accuracy {zs['accuracy']:.3f} < floor {zs['floor']} (chance "
                              f"{zs['chance']:.3f}): image and text spaces look misaligned")
        return checks, checked

    @staticmethod
    def _has_views(rows: pd.DataFrame) -> bool:
        """Ruling M9: a _views file for every split whenever items carry more than one view (bach512/bach256)."""
        return bool(len(rows)) and bool((rows.groupby("item_id")["view"].nunique() > 1).any())

    @staticmethod
    def _load_chunks(chunks: Sequence[tuple[int, str, pd.DataFrame]], chunk_dir: Path, heads: Sequence[str]
                     ) -> dict[str, np.ndarray]:
        parts: dict[str, list[np.ndarray]] = {h: [] for h in heads}
        modes = set()
        for k, digest, part in chunks:
            data = io.load_npz(chunk_dir / f"{k:05d}_{digest}.npz")
            if data["row_id"].tolist() != part["row_id"].tolist():
                raise RuntimeError(f"chunk {k} rows do not match the plan; rerun with --force")
            modes.add(bool(data["one_pass"]))
            for h in heads:
                parts[h].append(data[h])
        if len(modes) > 1:
            raise CheckFailed("chunks mix the one-pass and the adapter two-call paths (a resumed run fell back); "
                              "rerun this set with --force")
        return {h: np.concatenate(v) for h, v in parts.items()}

    def _write_final(self, enc: E.Encoder, set_name: str, rows: pd.DataFrame, arrays: Mapping[str, np.ndarray]) -> None:
        root = self.ctx.data_root
        index, reduced = reduce_views(rows, arrays, enc.spec)
        for head in enc.heads:
            npy = schemas.emb_npy(root, enc.name, set_name, head)
            io.save_npy(npy, reduced[head])
            io.write_csv(schemas.emb_index_csv(root, enc.name, set_name, head), index)
            self.outputs.append(npy)
        if self._has_views(rows):
            vindex = _index_frame(rows.reset_index(drop=True).assign(n_views=1))
            for head in enc.heads:
                io.save_npy(schemas.emb_npy(root, enc.name, set_name, head, views=True),
                            arrays[head].astype(np.float16))
                io.write_csv(schemas.emb_index_csv(root, enc.name, set_name, head, views=True), vindex)
        self.log(f"[{enc.name}/{set_name}] wrote {len(index)} items x heads {list(enc.heads)}")


# ---- smoke manifest --------------------------------------------------------------------------------------------------------
def _safe_id(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "_-" else "-" for ch in text)
    return schemas.check_case_id(out.lstrip("_-") or "x")


def _images_under(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _degrade_available(cfg: Mapping[str, Any]) -> bool:
    try:
        from pipeline import degrade

        degrade.apply(Image.new("RGB", (16, 16), (200, 120, 160)), cfg["degrade"]["types"][0], 1, 0, cfg)
        return True
    except NotImplementedError:
        return False


def build_smoke_manifest(cfg: Mapping[str, Any], ecfg: Mapping[str, Any], nct_dir: Path, bach_dirs: Sequence[Path]
                         ) -> tuple[pd.DataFrame, list[str]]:
    """A tiny schema-exact manifest that needs no Agent 1 output: nct_per_class NCT tiles per class folder
    (seeded), plus up to smoke.bach_images readable BACH images tiled on the g512 grid as 'test' items with
    smoke.bach_views views (and one degraded row per type when pipeline.degrade is implemented)."""
    sm, notes, rows = ecfg["smoke"], [], []
    rng = np.random.default_rng(int(sm["seed"]))

    def row(set_name: str, case_id: str, split: str, grid: str, tile: int | None, path: Path, box: tuple[int, ...],
            view: int, label: str = "", dtype: str = "", severity: int = 0) -> dict[str, Any]:
        item = schemas.make_item_id(set_name, case_id, tile, dtype or None, severity or None)
        return {"row_id": f"{item}|v{view}", "item_id": item, "set": set_name, "case_id": case_id, "split": split,
                "grid": grid, "tile_idx": -1 if tile is None else tile, "path": str(path), "x": box[0], "y": box[1],
                "w": box[2], "h": box[3], "view": view, "degrade_type": dtype, "degrade_severity": severity,
                "degrade_seed": int(cfg["degrade"]["seed"]) if dtype else 0, "label": label, "group": ""}

    for cls in cfg["nct"]["classes"]:
        files = _images_under(nct_dir / cls) if (nct_dir / cls).is_dir() else []
        if not files:
            notes.append(f"NCT class folder {nct_dir / cls} missing or empty")
            continue
        pick = sorted(rng.choice(len(files), size=min(int(sm["nct_per_class"]), len(files)), replace=False))
        for i in pick:
            path = files[int(i)]
            with Image.open(path) as im:
                w, h = im.size
            rows.append(row("nct", _safe_id(path.stem), "", "", None, path, (0, 0, w, h), 0, label=cls))
    grid = cfg["tiles"]["grids"]["g512"]
    size, deg_ok, n_bach = int(grid["size"]), _degrade_available(cfg), 0
    if not deg_ok:
        notes.append("pipeline.degrade not implemented yet: smoke has no degraded rows")
    for folder in bach_dirs:
        if n_bach >= int(sm["bach_images"]) or not folder.is_dir():
            continue
        for path in _images_under(folder):
            if n_bach >= int(sm["bach_images"]):
                break
            try:
                with Image.open(path) as im:
                    w, h = im.size
            except Exception as exc:
                notes.append(f"unreadable BACH image {path}: {exc}")
                continue
            n_rows, n_cols = min(int(grid["rows"]), h // size), min(int(grid["cols"]), w // size)
            if not n_rows or not n_cols:
                continue
            case_id = f"SMOKE_{n_bach:04d}"
            for r in range(n_rows):
                for c in range(n_cols):
                    tile, box = r * n_cols + c, (c * size, r * size, size, size)
                    for view in range(int(sm["bach_views"])):
                        rows.append(row("bach512", case_id, "test", "g512", tile, path, box, view))
                    if deg_ok and tile == 0:
                        for dtype in cfg["degrade"]["types"]:
                            rows.append(row("deg512", case_id, "test", "g512", tile, path, box, 0, dtype=dtype,
                                            severity=int(cfg["degrade"]["severities"][-1])))
            n_bach += 1
    if not n_bach:
        notes.append(f"no readable BACH images under {[str(d) for d in bach_dirs]}")
    df = pd.DataFrame(rows, columns=[c.name for c in schemas.EMBED_MANIFEST.columns])
    schemas.validate_frame(schemas.EMBED_MANIFEST, df)
    return df, notes


# ---- CLI ------------------------------------------------------------------------------------------------------------------
def parse_list(text: str | None) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def parse_max_views(text: str | None) -> dict[str, int]:
    """'bach512=1,bach256=4' -> {set: n}: keep only views 0..n-1 of that set (e.g. bach512 view 0 for 224 encoders)."""
    out: dict[str, int] = {}
    for part in parse_list(text):
        set_name, _, n = part.partition("=")
        if set_name not in schemas.EMBED_SETS or not n.isdigit() or int(n) < 1:
            raise ValueError(f"--max-views entries look like bach512=1, got {part!r}")
        out[set_name] = int(n)
    return out


def build_parser() -> Any:
    p = cli.step_parser(STEP, "Embed manifest tiles with foundation encoders (BUILD_PLAN Task 3).")
    p.add_argument("--sets", default=None, help="comma list of embed sets (default: every set in the manifest)")
    p.add_argument("--encoders", default=None, help="comma list of encoders (default: settings encoders.primary)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default engine.num_workers)")
    p.add_argument("--batch-size", type=int, default=None, help="override engine.batch_size")
    p.add_argument("--max-views", default=None, help="per-set view cap, e.g. bach512=1 (views 0..n-1 only)")
    p.add_argument("--manifest", default=None, help="manifest path (default <data-root>/embed_manifest.csv)")
    p.add_argument("--encoders-config", default=None, help="config/encoders.yaml path")
    p.add_argument("--precision-check", action="store_true", help="autocast vs fp32 cosine on the first batch")
    p.add_argument("--keep-going", action="store_true", help="continue with the next encoder after a failure")
    p.add_argument("--skip-unavailable", action="store_true", help="skip encoders whose weights are not cached")
    p.add_argument("--smoke-manifest", action="store_true", help="write a tiny smoke manifest into the data root, exit")
    p.add_argument("--nct-dir", default=None, help="smoke: NCT folder (default settings paths.delta.shared.nct_crc)")
    p.add_argument("--bach-dir", action="append", default=None, help="smoke: BACH image folder(s)")
    return p


def _default_bach_dirs(cfg: Mapping[str, Any]) -> list[Path]:
    official = Path(cfg["paths"]["delta"]["bach_official"]) / cfg["dataset"]["official"]["photos_subdir"]
    return [official, Path(cfg["paths"]["delta"]["shared"]["pathmmu_bach"])]


def main(argv: Sequence[str] | None = None, log: Any = print) -> int:
    args = build_parser().parse_args(argv)
    ctx = cli.context(args)
    ecfg = E.load_encoders_config(args.encoders_config)
    E.check_native_grids(ecfg, ctx.cfg)
    root = ctx.data_root
    manifest_path = Path(args.manifest) if args.manifest else schemas.embed_manifest_csv(root)

    if args.smoke_manifest:
        nct_dir = Path(args.nct_dir or ctx.cfg["paths"]["delta"]["shared"]["nct_crc"])
        bach = [Path(d) for d in args.bach_dir] if args.bach_dir else _default_bach_dirs(ctx.cfg)
        df, notes = build_smoke_manifest(ctx.cfg, ecfg, nct_dir, bach)
        io.write_csv(manifest_path, df)
        for note in notes:
            log(f"smoke manifest note: {note}")
        log(f"smoke manifest: {len(df)} rows -> {manifest_path}; sets {df['set'].value_counts().to_dict()}")
        return 0 if (df["set"] == "nct").any() else 2

    if not manifest_path.is_file():
        log(f"ABORT: manifest not found: {manifest_path}")
        return 2
    manifest = read_manifest(manifest_path)
    sets = parse_list(args.sets) or [s for s in schemas.EMBED_SETS if s in set(manifest["set"])]
    unknown = [s for s in sets if s not in schemas.EMBED_SETS]
    if unknown:
        log(f"ABORT: unknown sets {unknown}")
        return 2
    names = parse_list(args.encoders) or [ctx.cfg["encoders"]["primary"]]
    for name in names:
        E.encoder_spec(ecfg, name)
    try:
        device = resolve_device(args.device)
    except CheckFailed as exc:
        log(f"ABORT: {exc}")
        return 2
    engine = Engine(ctx, ecfg, device, workers=args.workers, batch=args.batch_size,
                    precision_check=args.precision_check, log=log, max_views=parse_max_views(args.max_views))
    summary: dict[str, Any] = {}
    failed = False
    for name in names:
        try:
            summary[name] = engine.run_encoder(name, manifest, sets)
        except E.EncoderUnavailable as exc:
            if not args.skip_unavailable:
                log(f"ABORT: {exc}")
                return 2
            summary[name] = {"status": f"SKIPPED (unavailable): {exc}"}
            log(f"SKIP {name}: {exc}")
        except CheckFailed as exc:
            log(f"ABORT: check failed: {exc}")
            summary[name] = {"status": f"FAILED: {exc}"}
            failed = True
            if not args.keep_going:
                break
        except Exception as exc:  # --keep-going (smoke): record the real error, try the next encoder, exit 2
            if not args.keep_going:
                raise
            log(f"ERROR {name}: {type(exc).__name__}: {exc}")
            summary[name] = {"status": f"FAILED: {type(exc).__name__}: {exc}"}
            failed = True
    for note in engine.notes:
        log(f"note: {note}")
    log("SUMMARY " + json.dumps(summary, default=str))
    if not failed and all(str(r.get("status", "")).startswith("SKIPPED") for r in summary.values()):
        log("ABORT: no encoder was available (all skipped): nothing was embedded")
        failed = True
    io.write_meta(schemas.step_meta_json(root, META_STEP), step=META_STEP, cfg=ctx.cfg,
                  inputs=[manifest_path], outputs=engine.outputs, synthetic=ctx.synthetic,
                  args={"sets": sets, "encoders": names, "limit": ctx.limit, "force": ctx.force, "device": device,
                        "max_views": args.max_views},
                  extra={"summary": summary, "notes": engine.notes})
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
