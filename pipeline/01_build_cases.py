"""Step 01: BACH cases, THUNDER split, lanes, QC'd tiles, display JPEGs, NCT sample and the core embedding manifest
(plan §5.3, §6.1; BUILD_PLAN §A2, Task 2). CPU job (prep_a.sbatch); runs on the laptop for tests.

Listing and split (BUILD_PLAN §A2):
  classes in split.class_order, '.tif' files sorted, 100 per class asserted; global source_index = class_idx * 100 +
  position; split from the THUNDER half-open ranges in split.ranges (sizes 218/50/132 asserted). The micro patient
  workbook gives patient IDs for 145 images; zero patient overlap between splits is asserted (unidentified images
  come from other patients, per the organizers). group = 'p<patient_id>' when known, else
  'b<source_index // split.cv.block_size>'.
Case IDs: '<split.ids.prefix>_<n:04d>', n = 1 + a seeded permutation of source_index (opaque: never reveals the
  label or the class-sorted position).
Lanes: lanes.surgical_fraction of the lanes.surgical_source_split cases are surgical (seeded), with at least one
  case from every group in lanes.surgical_must_cover; everything else is biopsy.
Reading: PIL .convert("RGB") decodes the 381 16-bit and 19 8-bit TIFFs to 8-bit RGB; mode and size are asserted.
QC: per tile, the fraction of pixels with HSV saturation > qc.saturation_min and value < qc.brightness_max;
  qc_pass = tissue_frac >= qc.min_tissue_frac. Every tile of both grids is written and listed, QC pass or not.
Displays: display/<case_id>.jpg (long side tiles.display.long_side) for every case, plus degraded displays for
  test-surgical cases x degrade.types at degrade.display_severity (display/<variant_case_id>.jpg): degraded at
  source resolution, then downscaled.
NCT: nct.per_class files per class, seeded per class, read in place (absolute paths).
Manifest (embed_manifest.csv, core sets; other sets already present are kept untouched):
  bach512 / bach256  every case x every tile x encoders.views[grid] dihedral views (path = the tile PNG)
  nct                whole 224-px tiles, view 0
  deg512             calibration + test cases x degrade.types x degrade.severities x g512 tiles x the same views
  deg256             test-surgical cases x degrade.types x degrade.display_severity x g256 tiles x views
  Degraded rows point at the clean tile PNG; 02_embed applies pipeline.degrade.apply on the fly with degrade_seed
  (one seed per case, type and severity, as for the display JPEG). Their case_id is the variant ID
  (schemas.variant_case_id) so per-case pooling never mixes variants; split is the source case's split.
  Every tile is listed (QC pass or not); consumers filter with tiles.csv qc_pass.

Resumable: tiles and displays that exist are skipped unless --force; CSVs, manifest and meta are rewritten on
every run (cheap). --limit N keeps the first N cases by case_id (after the full listing, split, ID and lane
assignment) and caps the NCT sample at N per class.

Run: python -m pipeline.01_build_cases [--limit 5] [--workers 16] [--bach-root ...] [--nct-root ...]
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import cli, config, io, schemas
from pipeline import common, degrade

STEP = "01_build_cases"
CORE_SETS = ("bach512", "bach256", "nct", "deg512", "deg256")


# ---- listing, split, patients, IDs, lanes -----------------------------------------------------------------------------
def build_listing(photos_dir: str | Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    rows = common.list_bach_photos(photos_dir, cfg)
    df = pd.DataFrame(rows)
    per_class = int(cfg["dataset"]["per_class"])
    counts = df["label"].value_counts().to_dict() if len(df) else {}
    bad = {c: counts.get(c, 0) for c in cfg["split"]["class_order"] if counts.get(c, 0) != per_class}
    if bad:
        raise ValueError(f"expected {per_class} .tif photos per class, found {bad}")
    if len(df) != int(cfg["dataset"]["n_images"]):
        raise ValueError(f"expected {cfg['dataset']['n_images']} photos, found {len(df)}")
    return df


def assign_splits(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    splits = config.split_assignments(cfg)
    if len(splits) != len(df):
        raise ValueError(f"split ranges cover {len(splits)} indices but {len(df)} photos were listed")
    out = df.copy()
    out["split"] = [splits[i] for i in out["source_index"]]
    counts = out["split"].value_counts().to_dict()
    expected = dict(cfg["split"]["expected_counts"])
    if {s: counts.get(s, 0) for s in expected} != expected:
        raise ValueError(f"split sizes {counts} differ from split.expected_counts {expected}")
    return out


def assign_patients(df: pd.DataFrame, pmap: Mapping[str, int] | None, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    pmap = dict(pmap or {})
    unknown = sorted(set(pmap) - set(out["file"]))
    if unknown:
        raise ValueError(f"patient workbook lists files that are not in the photo listing: {unknown[:5]}")
    out["patient_id"] = [str(pmap[f]) if f in pmap else "" for f in out["file"]]
    block = int(cfg["split"]["cv"]["block_size"])
    out["group"] = [f"p{p}" if p else f"b{i // block}" for p, i in zip(out["patient_id"], out["source_index"])]
    return out


def patient_overlap(df: pd.DataFrame) -> dict[str, list[str]]:
    """{'train&test': [shared patient ids], ...} over known patient IDs; empty lists mean disjoint."""
    known = df[df["patient_id"] != ""]
    by_split = {s: set(known.loc[known["split"] == s, "patient_id"]) for s in schemas.SPLITS}
    pairs = [("train", "calibration"), ("train", "test"), ("calibration", "test")]
    return {f"{a}&{b}": sorted(by_split[a] & by_split[b]) for a, b in pairs}


def assert_patient_disjoint(df: pd.DataFrame) -> None:
    shared = {k: v for k, v in patient_overlap(df).items() if v}
    if shared:
        raise ValueError(f"patient overlap between splits (leakage): {shared}")


def assign_case_ids(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    ids = cfg["split"]["ids"]
    perm = np.random.default_rng(int(ids["seed"])).permutation(len(df))
    out = df.copy()
    out["case_id"] = [schemas.check_case_id(f"{ids['prefix']}_{perm[i] + 1:04d}") for i in out["source_index"]]
    return out


def assign_lanes(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    lanes = cfg["lanes"]
    pool = df[df["split"] == lanes["surgical_source_split"]].sort_values("case_id")
    must = [list(g) for g in lanes["surgical_must_cover"]]
    n_surgical = max(int(round(float(lanes["surgical_fraction"]) * len(pool))), len(must))
    rng = np.random.default_rng(int(lanes["seed"]))
    chosen: list[str] = []
    for group in must:
        candidates = sorted(pool.loc[pool["label"].isin(group) & ~pool["case_id"].isin(chosen), "case_id"])
        if not candidates:
            raise ValueError(f"no {lanes['surgical_source_split']} case of {group} available for the surgical lane")
        chosen.append(candidates[int(rng.integers(len(candidates)))])
    rest = sorted(set(pool["case_id"]) - set(chosen))
    if n_surgical > len(chosen):
        chosen += [str(c) for c in rng.choice(rest, size=n_surgical - len(chosen), replace=False)]
    out = df.copy()
    out["lane"] = np.where(out["case_id"].isin(chosen), "surgical", "biopsy")
    return out


# ---- per-case image work (runs in worker processes) ------------------------------------------------------------------
def process_case(task: Mapping[str, Any]) -> dict[str, Any]:
    """Read one photo, QC every tile of every grid, write missing tile PNGs and display JPEGs."""
    cfg, root, case_id, force = task["cfg"], Path(task["data_root"]), task["case_id"], task["force"]
    img = common.read_rgb(task["image_path"], cfg["dataset"]["image_size"])
    mask = common.tissue_mask(img, cfg)
    min_frac = float(cfg["qc"]["min_tissue_frac"])
    png_level = int(cfg["tiles"]["png_compress_level"])
    rows, written = [], 0
    for grid in cfg["tiles"]["grids"]:
        for box in common.grid_boxes(cfg, grid):
            x, y, s = box["x"], box["y"], box["size"]
            frac = common.tissue_fraction(mask, x, y, s, s)
            path = schemas.tile_png(root, grid, case_id, box["row"], box["col"])
            if force or not path.exists():
                io.save_image(path, img.crop((x, y, x + s, y + s)), compress_level=png_level)
                written += 1
            rows.append({"case_id": case_id, "grid": grid, **box, "tissue_frac": round(frac, 4),
                         "qc_pass": int(frac >= min_frac), "path": path.relative_to(root).as_posix()})
    display = cfg["tiles"]["display"]
    out = schemas.display_jpg(root, case_id)
    if force or not out.exists():
        io.save_image(out, common.to_display(img, int(display["long_side"])), quality=int(display["jpeg_quality"]))
        written += 1
    if task["surgical"]:
        sev = int(cfg["degrade"]["display_severity"])
        for kind in cfg["degrade"]["types"]:
            out = schemas.display_jpg(root, schemas.variant_case_id(case_id, kind, sev))
            if force or not out.exists():
                deg = degrade.apply(img, kind, sev, common.degrade_seed(cfg, case_id, kind, sev), cfg)
                io.save_image(out, common.to_display(deg, int(display["long_side"])),
                              quality=int(display["jpeg_quality"]))
                written += 1
    return {"case_id": case_id, "width": img.width, "height": img.height, "tiles": rows, "written": written}


def run_cases(tasks: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(tasks) <= 1:
        return [process_case(t) for t in tasks]
    with multiprocessing.get_context().Pool(workers) as pool:
        return list(pool.imap_unordered(process_case, tasks, chunksize=2))


# ---- NCT sample --------------------------------------------------------------------------------------------------------
def nct_sample(nct_root: str | Path, cfg: Mapping[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    from PIL import Image

    nct = cfg["nct"]
    k = int(nct["per_class"]) if limit is None else min(int(nct["per_class"]), limit)
    rows: list[dict[str, Any]] = []
    for cls in nct["classes"]:
        folder = Path(nct_root) / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"NCT class folder missing: {folder}")
        files = common.image_files(folder)
        if len(files) < k:
            raise ValueError(f"NCT class {cls} has {len(files)} images, nct.per_class needs {k}")
        rng = np.random.default_rng(common.stable_seed(nct["seed"], cls))
        for i in sorted(rng.choice(len(files), size=k, replace=False)):
            path = files[int(i)].resolve()
            with Image.open(path) as im:
                w, h = im.size
            rows.append(common.manifest_row(set_name="nct", case_id=schemas.check_case_id(path.stem), split="",
                                            grid="", tile_idx=None, path=str(path), x=0, y=0, w=w, h=h, view=0,
                                            label=cls))
    return rows


# ---- manifest ----------------------------------------------------------------------------------------------------------
def core_manifest(cases: pd.DataFrame, tiles: pd.DataFrame, nct_rows: list[dict[str, Any]],
                  cfg: Mapping[str, Any]) -> pd.DataFrame:
    views = cfg["encoders"]["views"]
    dcfg = cfg["degrade"]
    by_case_grid: dict[tuple[str, str], list[tuple[int, str, int]]] = {}
    for t in tiles.itertuples(index=False):
        by_case_grid.setdefault((t.case_id, t.grid), []).append((int(t.tile_idx), t.path, int(t.size)))
    rows: list[dict[str, Any]] = []

    def tile_rows(set_name: str, case: Any, grid: str, kind: str = "", sev: int = 0) -> None:
        case_id = case.case_id
        item_case = schemas.variant_case_id(case_id, kind, sev) if kind else case_id
        seed = common.degrade_seed(cfg, case_id, kind, sev) if kind else 0
        for tile_idx, path, size in by_case_grid[(case_id, grid)]:
            first = common.manifest_row(
                set_name=set_name, case_id=item_case, split=case.split, grid=grid, tile_idx=tile_idx, path=path,
                x=0, y=0, w=size, h=size, view=0, degrade_type=kind, degrade_severity=sev, degrade_seed=seed)
            rows.append(first)
            for view in range(1, int(views[grid])):
                rows.append({**first, "row_id": f"{first['item_id']}|v{view}", "view": view})

    for case in cases.itertuples(index=False):
        tile_rows("bach512", case, "g512")
        tile_rows("bach256", case, "g256")
    for case in cases[cases["split"].isin(["calibration", "test"])].itertuples(index=False):
        for kind in dcfg["types"]:
            for sev in dcfg["severities"]:
                tile_rows("deg512", case, "g512", kind, int(sev))
    surgical = cases[(cases["split"] == "test") & (cases["lane"] == "surgical")]
    for case in surgical.itertuples(index=False):
        for kind in dcfg["types"]:
            tile_rows("deg256", case, "g256", kind, int(dcfg["display_severity"]))
    rows += nct_rows
    return pd.DataFrame(rows, columns=common.MANIFEST_COLUMNS)


# ---- main --------------------------------------------------------------------------------------------------------------
CASE_COLUMNS = [c.name for c in schemas.CASES.columns]
TILE_COLUMNS = [c.name for c in schemas.TILES.columns]


def assemble_cases(photos_dir: Path, pmap: Mapping[str, int] | None, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Full listing -> split -> patients (disjointness asserted) -> case IDs -> lanes, all 400 photos."""
    df = build_listing(photos_dir, cfg)
    df = assign_splits(df, cfg)
    df = assign_patients(df, pmap, cfg)
    assert_patient_disjoint(df)
    df = assign_case_ids(df, cfg)
    df = assign_lanes(df, cfg)
    df["source"] = "bach_official"
    return df.sort_values("case_id").reset_index(drop=True)


def summarize(cases: pd.DataFrame, tiles: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    qc = tiles.groupby("grid")["qc_pass"].mean().round(4).to_dict()
    return {
        "n_cases": int(len(cases)),
        "per_split_class": {s: g["label"].value_counts().sort_index().to_dict()
                            for s, g in cases.groupby("split")},
        "per_split_lane": {s: g["lane"].value_counts().to_dict() for s, g in cases.groupby("split")},
        "n_patients_per_split": {s: int(g.loc[g["patient_id"] != "", "patient_id"].nunique())
                                 for s, g in cases.groupby("split")},
        "n_groups_per_split": {s: int(g["group"].nunique()) for s, g in cases.groupby("split")},
        "qc_pass_rate": qc,
        "insufficient_tissue_cases": None,
        "manifest_rows_per_set": manifest["set"].value_counts().sort_index().to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "BACH cases, THUNDER split, tiles, displays, core manifest.")
    common.add_source_args(parser, ("bach_root", "nct_root"))
    parser.add_argument("--workers", type=int, default=None, help="processes (default $SLURM_CPUS_PER_TASK or 1)")
    parser.add_argument("--allow-missing-patients", action="store_true",
                        help="proceed without the micro patient workbook (no patient check; block groups only)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    try:
        bach_root = common.source_root(cfg, args, "bach_root")
        nct_root = common.source_root(cfg, args, "nct_root")
        photos = common.bach_photos_dir(bach_root, cfg)
        pfile = common.patient_file(bach_root, cfg, "micro")
        if pfile.is_file():
            pmap = common.load_micro_patients(pfile)
            patient_check = "asserted disjoint"
        elif args.allow_missing_patients:
            pmap, patient_check = None, "skipped: patient workbook missing (--allow-missing-patients)"
        else:
            raise FileNotFoundError(f"patient workbook missing: {pfile} (prep_a fetches it; or pass "
                                    "--allow-missing-patients to proceed without the leakage check)")
        all_cases = assemble_cases(photos, pmap, cfg)
        keep = cli.limit_case_ids(all_cases["case_id"], ctx.limit)
        cases = all_cases[all_cases["case_id"].isin(keep)].reset_index(drop=True)

        common.check_grid_fits(cfg, *cfg["dataset"]["image_size"])
        tasks = [{"cfg": cfg, "data_root": str(root), "case_id": c.case_id, "image_path": c.image_path,
                  "surgical": c.lane == "surgical", "force": ctx.force} for c in cases.itertuples(index=False)]
        results = sorted(run_cases(tasks, common.worker_count(args.workers)), key=lambda r: r["case_id"])
        size = {r["case_id"]: (r["width"], r["height"]) for r in results}
        cases["width"] = [size[c][0] for c in cases["case_id"]]
        cases["height"] = [size[c][1] for c in cases["case_id"]]
        cases["display_path"] = [schemas.display_jpg(Path(""), c).as_posix() for c in cases["case_id"]]
        cases = cases[CASE_COLUMNS]
        tiles = pd.DataFrame([t for r in results for t in r["tiles"]], columns=TILE_COLUMNS)
        schemas.validate_frame(schemas.CASES, cases)
        schemas.validate_frame(schemas.TILES, tiles)

        nct_rows = nct_sample(nct_root, cfg, ctx.limit)
        manifest = core_manifest(cases, tiles, nct_rows, cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2

    io.write_csv(schemas.cases_csv(root), cases, columns=CASE_COLUMNS)
    io.write_csv(schemas.tiles_csv(root), tiles, columns=TILE_COLUMNS)
    full_manifest = common.replace_manifest_sets(root, CORE_SETS, manifest)
    summary = summarize(cases, tiles, manifest)
    contract = tiles[(tiles["grid"] == cfg["tiles"]["contract_grid"])].groupby("case_id")["qc_pass"].sum()
    summary["insufficient_tissue_cases"] = int((contract < int(cfg["qc"]["min_tiles_kept"])).sum())
    summary["files_written"] = int(sum(r["written"] for r in results))
    summary["patient_check"] = patient_check
    summary["patient_overlap"] = patient_overlap(all_cases)
    summary["manifest_rows_total"] = int(len(full_manifest))
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg,
                  inputs=[str(photos), str(pfile), str(nct_root)],
                  outputs=[schemas.cases_csv(root), schemas.tiles_csv(root), schemas.embed_manifest_csv(root),
                           root / "tiles", root / "display"],
                  args={"limit": ctx.limit, "force": ctx.force, "workers": common.worker_count(args.workers),
                        "bach_root": str(bach_root), "nct_root": str(nct_root)},
                  synthetic=ctx.synthetic, extra={"summary": summary})
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
