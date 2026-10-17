"""Step 04c: nuclei -> per-tile and per-case cell features (CELLS_SPEC §6; BUILD_PLAN Task 8). CPU; runs at the end
of cells.sbatch and on the laptop.

Feature ids, definitions, units and min-count rules: config/cell_features.yaml (settings cells.features_file).
  * Per tile: counts per PanNuke type and every feature from the tile's own nuclei (display only, no minimum).
  * Per case: nuclei pooled over the case's selected tiles (cell_tiles.csv) = nucleus-count-weighted mean of the
    tile ratios; densities = pooled counts / pooled tissue area (tile area x tiles.csv tissue_frac at the measured
    source mpp). Morphology (nuc_*) uses non-edge neoplastic nuclei only and needs >= cells.min_neoplastic_nuclei of
    them; ratios/densities need >= cells.min_total_nuclei typed nuclei; til_proximity also needs >= 1 inflammatory
    nucleus. Otherwise the value is missing (empty) and f__missing = 1. insufficient_tumor_nuclei = 1 exactly when the
    nuc_* features are missing for lack of non-edge neoplastic nuclei (< cells.min_neoplastic_nuclei); n_neoplastic
    counts all neoplastic nuclei.
  * f__ref_pct: mid-rank percentile of the case value among train cases labeled cells.reference_classes (Normal,
    Benign), in [0, 1]; missing when the value is missing.
  * Train medians (imputation for the risk model, which also adds f__missing) are recorded in the meta file.
Labels are read only to define the benign reference set; they are never written to the outputs.

Writes cell_features.csv (CELL_FEATURES + dynamic columns) and cell_features_tile.csv (CELL_FEATURES_TILE + per-tile
feature values), meta/04c_cell_features.json. With cells.enabled false: a "disabled" meta record, exit 0, no files.
Tiles whose JSON is missing are skipped and counted (04b --verify is the gate). --limit N: first N cases.

Portable (Delta-side): numpy, pandas, PyYAML, core.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from core import cli, config, io, schemas
from pipeline import cells_convert as cc

STEP = "04c_cell_features"


def _cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean = values.mean() if values.size else np.nan
    return float(values.std() / mean) if values.size and mean > 0 else np.nan


def til_counts(nuclei: Sequence[Mapping[str, Any]], radius_um: float) -> tuple[int, int]:
    """(inflammatory nuclei with a neoplastic centroid within radius_um, inflammatory nuclei) for one tile."""
    infl = np.asarray([n["centroid_um"] for n in nuclei if n["type"] == "inflammatory"], dtype=float).reshape(-1, 2)
    neo = np.asarray([n["centroid_um"] for n in nuclei if n["type"] == "neoplastic"], dtype=float).reshape(-1, 2)
    if not len(infl):
        return 0, 0
    if not len(neo):
        return 0, len(infl)
    d2 = ((infl[:, None, :] - neo[None, :, :]) ** 2).sum(axis=2)
    return int((d2.min(axis=1) <= radius_um ** 2).sum()), len(infl)


def compute_features(nuclei: Sequence[Mapping[str, Any]], tissue_mm2: float, til: tuple[int, int],
                     cfg: Mapping[str, Any], fcfg: Mapping[str, Any], *, enforce_min: bool) -> dict[str, float]:
    """Feature values (NaN = missing) from a pool of nuclei (one tile or one case)."""
    types = [n["type"] for n in nuclei]
    n_total = len(types)
    count = {t: types.count(t) for t in cfg["cells"]["types"]}
    neo_core = [n for n in nuclei if n["type"] == "neoplastic" and not n["edge"]]
    min_neo, min_total = int(cfg["cells"]["min_neoplastic_nuclei"]), int(cfg["cells"]["min_total_nuclei"])
    ok = {"morphology": len(neo_core) >= min_neo if enforce_min else len(neo_core) >= 2,
          "ratio": n_total >= min_total if enforce_min else n_total > 0,
          "density": (n_total >= min_total if enforce_min else True) and tissue_mm2 > 0}
    area = np.asarray([n["area_um2"] for n in neo_core], dtype=float)
    raw = {
        "neo_fraction": count["neoplastic"] / n_total if n_total else np.nan,
        "neo_density": count["neoplastic"] / tissue_mm2 if tissue_mm2 > 0 else np.nan,
        "nuc_area_mean": float(area.mean()) if area.size else np.nan,
        "nuc_area_cv": _cv(area),
        "nuc_shape_cv": float(np.nanmean([_cv([n["eccentricity"] for n in neo_core]),
                                          _cv([n["solidity"] for n in neo_core])])) if len(neo_core) >= 2 else np.nan,
        "infl_fraction": count["inflammatory"] / n_total if n_total else np.nan,
        "til_proximity": til[0] / til[1] if til[1] else np.nan,
        "dead_fraction": count["dead"] / n_total if n_total else np.nan,
        "epi_fraction": count["epithelial"] / n_total if n_total else np.nan,
    }
    out = {}
    for f in fcfg["features"]:
        if f["id"] not in raw:
            raise ValueError(f"no implementation for cell feature {f['id']!r} (config/cell_features.yaml)")
        value = raw[f["id"]]
        out[f["id"]] = float(value) if ok[f["min_count"]] and np.isfinite(value) else np.nan
    return out


def ref_percentile(values: pd.Series, reference: np.ndarray) -> pd.Series:
    """Mid-rank empirical percentile of each value among the reference values (NaN stays NaN)."""
    ref = np.sort(reference[np.isfinite(reference)])
    if not len(ref):
        return pd.Series(np.nan, index=values.index)
    v = values.to_numpy(dtype=float)
    less = np.searchsorted(ref, v, side="left")
    leq = np.searchsorted(ref, v, side="right")
    pct = (less + 0.5 * (leq - less)) / len(ref)
    return pd.Series(np.where(np.isfinite(v), pct, np.nan), index=values.index)


def build(root: Any, cfg: Mapping[str, Any], fcfg: Mapping[str, Any], limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sel = io.read_csv(schemas.cell_tiles_csv(root), dtype={"case_id": str}, keep_default_na=False)
    schemas.validate_frame(schemas.CELL_TILES, sel)
    cases = io.read_csv(schemas.cases_csv(root), dtype=str, keep_default_na=False)
    tiles = io.read_csv(schemas.tiles_csv(root), dtype={"case_id": str, "grid": str}, keep_default_na=False)
    keep = cli.limit_case_ids(sel["case_id"], limit)
    sel = sel[sel["case_id"].isin(keep)].sort_values(["case_id", "rank"])
    grid = cfg["tiles"]["contract_grid"]
    t = tiles[tiles["grid"] == grid].copy()
    t["tile_idx"] = t["tile_idx"].astype(int)
    tinfo = t.set_index(["case_id", "tile_idx"])
    mpp = config.effective_mpp(cfg, schemas.inventory_json(root))
    radius = float(cfg["cells"]["til_radius_um"])
    ids = [f["id"] for f in fcfg["features"]]
    tile_rows, case_rows, missing_tiles = [], [], []
    for case_id, group in sel.groupby("case_id", sort=True):
        pool: list[dict[str, Any]] = []
        tissue_mm2, til_near, til_n, n_tiles = 0.0, 0, 0, 0
        for r in group.itertuples(index=False):
            path = schemas.cell_json(root, case_id, int(r.tile_idx))
            if not path.is_file():
                missing_tiles.append(f"{case_id}:{int(r.tile_idx)}")
                continue
            doc = io.read_json(path)
            nuclei = doc["nuclei"]
            info = tinfo.loc[(case_id, int(r.tile_idx))]
            area_mm2 = (int(info["size"]) * mpp / 1000.0) ** 2 * float(info["tissue_frac"])
            near, n_infl = til_counts(nuclei, radius)
            feats = compute_features(nuclei, area_mm2, (near, n_infl), cfg, fcfg, enforce_min=False)
            counts = {f"n_{ty}": sum(n["type"] == ty for n in nuclei) for ty in cfg["cells"]["types"]}
            tile_rows.append({"case_id": case_id, "tile_idx": int(r.tile_idx), "n_nuclei": len(nuclei), **counts,
                              "overlay": schemas.cell_overlay_png(root, case_id, int(r.tile_idx)).relative_to(root).as_posix(),
                              **feats})
            pool += nuclei
            tissue_mm2 += area_mm2
            til_near, til_n, n_tiles = til_near + near, til_n + n_infl, n_tiles + 1
        feats = compute_features(pool, tissue_mm2, (til_near, til_n), cfg, fcfg, enforce_min=True)
        n_neo = sum(n["type"] == "neoplastic" for n in pool)
        n_neo_core = sum(n["type"] == "neoplastic" and not n["edge"] for n in pool)
        # flag == the rule that blanks the nuc_* features (non-edge neoplastic nuclei below the minimum)
        case_rows.append({"case_id": case_id, "n_tiles": n_tiles, "n_nuclei": len(pool), "n_neoplastic": n_neo,
                          "insufficient_tumor_nuclei": int(n_neo_core < int(cfg["cells"]["min_neoplastic_nuclei"])),
                          **feats})
    per_case = pd.DataFrame(case_rows)
    per_case = per_case.merge(cases[["case_id", "split", "label"]], on="case_id", how="left")
    if per_case["split"].isna().any():
        raise ValueError("cell_tiles.csv lists cases missing from cases.csv")
    ref_mask = (per_case["split"] == "train") & per_case["label"].isin(cfg["cells"]["reference_classes"])
    medians: dict[str, float | None] = {}
    for f in ids:
        per_case[f"{f}__missing"] = per_case[f].isna().astype(int)
        per_case[f"{f}__ref_pct"] = ref_percentile(per_case[f], per_case.loc[ref_mask, f].to_numpy(dtype=float)).round(4)
        med = per_case.loc[per_case["split"] == "train", f].median()
        medians[f] = None if pd.isna(med) else float(med)
    cols = ["case_id", "split", "n_tiles", "n_nuclei", "n_neoplastic", "insufficient_tumor_nuclei"]
    cols += [c for f in ids for c in (f, f"{f}__missing", f"{f}__ref_pct")]
    per_case = per_case[cols]  # label dropped: never written
    per_tile = pd.DataFrame(tile_rows, columns=["case_id", "tile_idx", "n_nuclei"]
                            + [f"n_{ty}" for ty in cfg["cells"]["types"]] + ["overlay"] + ids)
    schemas.validate_frame(schemas.CELL_FEATURES, per_case, extra_required=cols)
    schemas.validate_frame(schemas.CELL_FEATURES_TILE, per_tile, extra_required=ids)
    info = {"mpp_source": mpp, "n_reference_cases": int(ref_mask.sum()), "train_medians": medians,
            "tiles_missing": len(missing_tiles), "missing_examples": missing_tiles[:5],
            "missing_rate": {f: round(float(per_case[f"{f}__missing"].mean()), 4) for f in ids} if len(per_case) else {},
            "insufficient_tumor_nuclei_cases": int(per_case["insufficient_tumor_nuclei"].sum()) if len(per_case) else 0}
    return per_case, per_tile, info


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Per-tile and per-case cell features (CELLS_SPEC §6).")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    meta_path = schemas.step_meta_json(root, STEP)
    if not cfg["cells"]["enabled"]:
        io.write_meta(meta_path, step=STEP, cfg=cfg, inputs=[], outputs=[], args={"limit": ctx.limit},
                      synthetic=ctx.synthetic, extra={"status": "disabled"})
        print(json.dumps({"status": "disabled"}))
        return 0
    try:
        fcfg = cc.load_feature_config(cfg)
        per_case, per_tile, info = build(root, cfg, fcfg, ctx.limit)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2
    io.write_csv(schemas.cell_features_csv(root), per_case)
    io.write_csv(schemas.cell_features_csv(root, per_tile=True), per_tile)
    io.write_meta(meta_path, step=STEP, cfg=cfg,
                  inputs=[schemas.cell_tiles_csv(root), root / "cells", schemas.tiles_csv(root)],
                  outputs=[schemas.cell_features_csv(root), schemas.cell_features_csv(root, per_tile=True)],
                  args={"limit": ctx.limit, "force": ctx.force}, synthetic=ctx.synthetic,
                  extra={"status": "ok", "summary": {"n_cases": int(len(per_case)), "n_tiles": int(len(per_tile)), **info}})
    print(json.dumps({"n_cases": len(per_case), "n_tiles": len(per_tile), **{k: v for k, v in info.items()
                                                                             if k != "train_medians"}}, indent=1))
    if info["tiles_missing"]:
        print(f"WARNING {STEP}: {info['tiles_missing']} selected tiles have no cell JSON (run 04b / --verify)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
