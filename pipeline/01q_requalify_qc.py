"""Step 01q: re-run tile QC on the laptop from the display JPEGs (fix for the 2026-10-18 tissue-mask bug).

    python -m pipeline.01q_requalify_qc [--data-root D] [--limit N] [--force]

Why: 01_build_cases (Delta) marked a pixel as tissue when HSV saturation > qc.saturation_min AND HSV value <
qc.brightness_max (0.92). HSV value is the max RGB channel, so pale eosin/hematoxylin pixels in BACH sit at V ~ 0.95-1.0
and the value cut rejected real tissue: 51 of 400 images (32 of 100 Invasive) kept fewer than qc.min_tiles_kept
contract tiles and were flagged "insufficient tissue", and their concept scores / pooled embeddings fell back to
whatever tiles survived. Saturation alone separates BACH's grey glass background (S ~ 0.00-0.05) from tissue.

What: recompute every tile's tissue fraction with common.tissue_mask (current settings) on the case's display JPEG
(display/<case_id>.jpg, long side tiles.display.long_side), mapping each source-pixel tile box onto display pixels.
Validation recorded in meta/01q_requalify_qc_summary.json: the OLD rule recomputed on display JPEGs reproduces the Delta
full-resolution tissue fractions (Pearson r and mean |diff| are measured on every tile and written there).
Every tile was embedded on Delta whether or not it passed QC (embed_manifest.csv), so steps 04-08 can be re-run on the
laptop from the new qc_pass column without new GPU work. The previous tissue_frac / qc_pass columns are kept in
meta/tiles_qc_v1.csv for provenance (written once; re-runs read their baseline from it). Laptop only (PIL + numpy).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from core import cli, io, schemas
from pipeline import common
from pipeline import riskmodels as rm

STEP = "01q_requalify_qc"


def old_rule_cfg(cfg: dict[str, Any], brightness_max: float) -> dict[str, Any]:
    return {**cfg, "qc": {**cfg["qc"], "brightness_max": brightness_max}}


def tile_fracs(mask: np.ndarray, tiles: pd.DataFrame, scale_x: float, scale_y: float) -> np.ndarray:
    """Tissue fraction of each tile box (source pixels) on a display-resolution mask."""
    out = np.empty(len(tiles))
    for i, t in enumerate(tiles.itertuples()):
        x0, y0 = int(round(t.x * scale_x)), int(round(t.y * scale_y))
        x1, y1 = int(round((t.x + t.size) * scale_x)), int(round((t.y + t.size) * scale_y))
        out[i] = common.tissue_fraction(mask, x0, y0, x1 - x0, y1 - y0)
    return out


def run(ctx: cli.StepContext, old_brightness_max: float) -> dict[str, Any]:
    cfg, root = ctx.cfg, ctx.data_root
    cases = rm.read_cases(root)
    tiles = rm.read_csv(schemas.tiles_csv(root))
    schemas.validate_frame(schemas.TILES, tiles)
    v1_path = root / "meta" / "tiles_qc_v1.csv"
    if v1_path.is_file():                        # re-run: the Delta-era values stay the baseline
        v1_cols = rm.read_csv(v1_path).set_index(["case_id", "grid", "tile_idx"])
        key = pd.MultiIndex.from_frame(tiles[["case_id", "grid", "tile_idx"]])
        tiles["tissue_frac"] = v1_cols.loc[key, "tissue_frac"].to_numpy()
        tiles["qc_pass"] = v1_cols.loc[key, "qc_pass"].to_numpy()
    ids = cli.limit_case_ids(cases["case_id"], ctx.limit)
    min_frac = float(cfg["qc"]["min_tissue_frac"])
    old_cfg = old_rule_cfg(cfg, old_brightness_max)
    new = tiles["tissue_frac"].astype(float).to_numpy().copy()
    old_display = np.full(len(tiles), np.nan)
    by_case = tiles.groupby("case_id").indices
    for r in cases[cases.case_id.isin(ids)].itertuples():
        with Image.open(root / r.display_path) as handle:
            img = handle.convert("RGB")
        sx, sy = img.size[0] / float(r.width), img.size[1] / float(r.height)
        rows = by_case[r.case_id]
        part = tiles.iloc[rows]
        new[rows] = tile_fracs(common.tissue_mask(img, cfg), part, sx, sy)
        old_display[rows] = tile_fracs(common.tissue_mask(img, old_cfg), part, sx, sy)
    done = ~np.isnan(old_display)
    v1 = tiles["tissue_frac"].astype(float).to_numpy()
    out = tiles.copy()
    out["tissue_frac"] = np.round(new, 4)
    out["qc_pass"] = (out["tissue_frac"] >= min_frac).astype(int)
    contract = cfg["tiles"]["contract_grid"]
    min_kept = int(cfg["qc"]["min_tiles_kept"])

    def kept(frame: pd.DataFrame) -> pd.Series:
        c = frame[frame.grid == contract]
        return c.assign(p=c.qc_pass.astype(str).str.lower().isin(["1", "true", "1.0"])).groupby("case_id").p.sum()

    k_old, k_new = kept(tiles), kept(out)
    label = cases.set_index("case_id").label
    summary = {
        "rule_old": {"saturation_min": cfg["qc"]["saturation_min"], "brightness_max": old_brightness_max},
        "rule_new": {"saturation_min": cfg["qc"]["saturation_min"], "brightness_max": cfg["qc"]["brightness_max"]},
        "n_cases": int(len(ids)), "n_tiles": int(done.sum()),
        "validation_old_rule_display_vs_fullres": {
            "pearson_r": float(np.corrcoef(old_display[done], v1[done])[0, 1]),
            "mean_abs_diff": float(np.mean(np.abs(old_display[done] - v1[done]))),
        },
        "qc_pass_rate": {"old": {g: float(tiles[tiles.grid == g].qc_pass.astype(int).mean()) for g in sorted(tiles.grid.unique())},
                         "new": {g: float(out[out.grid == g].qc_pass.mean()) for g in sorted(out.grid.unique())}},
        "insufficient_tissue_cases": {
            "old": int((k_old < min_kept).sum()), "new": int((k_new < min_kept).sum()),
            "old_by_class": {c: int(((k_old < min_kept) & (label.reindex(k_old.index) == c)).sum()) for c in cfg["classes"]},
            "new_by_class": {c: int(((k_new < min_kept) & (label.reindex(k_new.index) == c)).sum()) for c in cfg["classes"]},
        },
        "mean_contract_tiles_kept": {"old": float(k_old.mean()), "new": float(k_new.mean())},
    }
    if not v1_path.is_file():
        io.write_csv(v1_path, tiles[["case_id", "grid", "tile_idx", "tissue_frac", "qc_pass"]])
    schemas.validate_frame(schemas.TILES, out)
    io.write_csv(schemas.tiles_csv(root), out)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Re-run tile QC from display JPEGs with the corrected tissue mask.")
    parser.add_argument("--old-brightness-max", type=float, default=0.92,
                        help="value cut of the previous rule, recomputed only to validate display vs full resolution")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    meta = schemas.step_meta_json(ctx.data_root, STEP)
    if meta.is_file() and not ctx.force:
        print(f"{STEP}: already applied (meta exists); use --force to recompute")
        return 0
    summary = run(ctx, args.old_brightness_max)
    io.write_json(ctx.data_root / "meta" / f"{STEP}_summary.json", summary)
    io.write_json(schemas.eval_json(ctx.artifacts_root, "qc"),       # full EVAL_ENVELOPE (config_hash included)
                  {"name": "qc", **io.provenance(ctx.cfg, synthetic=bool(ctx.synthetic)), **summary})
    io.write_meta(meta, step=STEP, cfg=ctx.cfg, inputs=[schemas.tiles_csv(ctx.data_root), "display/"],
                  outputs=[schemas.tiles_csv(ctx.data_root), ctx.data_root / "meta" / "tiles_qc_v1.csv"],
                  args={"limit": ctx.limit, "force": ctx.force}, synthetic=ctx.synthetic)
    it = summary["insufficient_tissue_cases"]
    print(f"{STEP}: insufficient-tissue cases {it['old']} -> {it['new']}; validation r = "
          f"{summary['validation_old_rule_display_vs_fullres']['pearson_r']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
