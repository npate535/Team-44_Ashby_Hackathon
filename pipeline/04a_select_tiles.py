"""Step 04a: pick the tiles to segment per case (CELLS_SPEC §4; BUILD_PLAN §A3, Task 8). Runs inside cells.sbatch.

For every case of every split, the top cells.tiles_per_case QC-passing contract-grid (g512) tiles ranked by the
microscopy-only out-of-fold tile risk: pipeline.tile_probe.oof_tile_risk (Agent 3) on the on-disk
emb/<cells.probe_encoder>/<cells.probe_set>_<cells.probe_head> TTA-mean embeddings, with cells.oof_folds grouped
folds over train cases. Train tiles are scored out of fold (a case's own label never chooses its own tiles);
calibration/test tiles by the full-train fit. Ties break by tile_idx; cases with fewer QC tiles take what exists
(n_tiles records it).

Writes tile_risk_oof.csv (TILE_RISK_OOF) if absent (reused unless --force; Agent 3's 05 reads the same file) and
cell_tiles.csv (CELL_TILES), meta/04a_select_tiles.json. With cells.enabled false it writes only a "disabled" meta
record and exits 0. --limit N selects tiles for the first N cases only (the probe always fits on all train cases).

Portable (Delta-side): numpy, pandas, scikit-learn (via tile_probe), core.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from core import cli, io, schemas
from pipeline import tile_probe

STEP = "04a_select_tiles"
COLUMNS = [c.name for c in schemas.CELL_TILES.columns]


def select_tiles(risk: pd.DataFrame, tiles: pd.DataFrame, cases: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """CELL_TILES frame: top-k QC contract-grid tiles per case by risk (desc), ties by tile_idx."""
    grid = cfg["tiles"]["contract_grid"]
    k = int(cfg["cells"]["tiles_per_case"])
    t = tiles[tiles["grid"] == grid].copy()
    t["tile_idx"] = t["tile_idx"].astype(int)
    t = t[t["qc_pass"].astype(str).str.lower().isin(["1", "true", "1.0"])]
    r = risk[risk["grid"] == grid][["case_id", "tile_idx", "risk"]].copy()
    r["tile_idx"] = r["tile_idx"].astype(int)
    merged = t.merge(r, on=["case_id", "tile_idx"], how="left", validate="one_to_one")
    if merged["risk"].isna().any():
        missing = merged.loc[merged["risk"].isna(), ["case_id", "tile_idx"]].head(3).values.tolist()
        raise ValueError(f"QC tiles without a tile risk (e.g. {missing}); rerun with --force to recompute the probe")
    merged = merged.merge(cases[["case_id", "split"]], on="case_id", how="inner")
    merged = merged.sort_values(["case_id", "risk", "tile_idx"], ascending=[True, False, True])
    top = merged.groupby("case_id", sort=True).head(k).copy()
    top["rank"] = top.groupby("case_id").cumcount() + 1
    top["n_tiles"] = top.groupby("case_id")["tile_idx"].transform("size")
    top = top.rename(columns={"risk": "tile_risk"})
    out = top[COLUMNS].reset_index(drop=True)
    schemas.validate_frame(schemas.CELL_TILES, out)
    return out


def compute_risk(root: Any, cases: pd.DataFrame, tiles: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    c = cfg["cells"]
    emb_path = schemas.emb_npy(root, c["probe_encoder"], c["probe_set"], c["probe_head"])
    idx_path = schemas.emb_index_csv(root, c["probe_encoder"], c["probe_set"], c["probe_head"])
    if not emb_path.is_file() or not idx_path.is_file():
        raise FileNotFoundError(f"probe embeddings missing: {emb_path} / {idx_path} (embed_core must run first)")
    emb = np.load(emb_path, allow_pickle=False).astype(np.float32)
    index = io.read_csv(idx_path, dtype={"case_id": str, "grid": str, "split": str}, keep_default_na=False)
    schemas.validate_frame(schemas.EMB_INDEX, index)
    grid_tiles = tiles[tiles["grid"] == cfg["tiles"]["contract_grid"]]
    return tile_probe.oof_tile_risk(emb, index, cases, cfg, probe_name=c["probe_name"], folds=int(c["oof_folds"]),
                                    qc_tiles=grid_tiles)


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Top QC tiles per case by out-of-fold tile risk (CELLS_SPEC §4).")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    meta_path = schemas.step_meta_json(root, STEP)
    if not cfg["cells"]["enabled"]:
        io.write_meta(meta_path, step=STEP, cfg=cfg, inputs=[], outputs=[], args={"limit": ctx.limit},
                      synthetic=ctx.synthetic, extra={"status": "disabled"})
        print(json.dumps({"status": "disabled"}))
        return 0
    risk_path = schemas.tile_risk_csv(root, oof=True)
    try:
        cases = io.read_csv(schemas.cases_csv(root), dtype=str, keep_default_na=False)
        tiles = io.read_csv(schemas.tiles_csv(root), dtype=str, keep_default_na=False)
        schemas.validate_frame(schemas.CASES, cases)
        schemas.validate_frame(schemas.TILES, tiles)
        reused = risk_path.is_file() and not ctx.force
        if reused:
            risk = io.read_csv(risk_path, dtype={"case_id": str, "grid": str, "split": str, "probe": str},
                               keep_default_na=False)
            schemas.validate_frame(schemas.TILE_RISK_OOF, risk)
        else:
            risk = compute_risk(root, cases, tiles, cfg)
        keep = cli.limit_case_ids(cases["case_id"], ctx.limit)
        sel = select_tiles(risk, tiles, cases[cases["case_id"].isin(keep)], cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2
    if not reused:
        io.write_csv(risk_path, risk)
    io.write_csv(schemas.cell_tiles_csv(root), sel, columns=COLUMNS)
    train = risk[risk["split"] == "train"]
    summary = {"n_cases": int(sel["case_id"].nunique()), "n_tiles": int(len(sel)),
               "tiles_per_split": sel["split"].value_counts().to_dict(),
               "cases_with_fewer_tiles": int((sel.groupby("case_id")["n_tiles"].first() < int(cfg["cells"]["tiles_per_case"])).sum()),
               "tile_risk_reused": bool(reused), "probe": str(risk["probe"].iloc[0]) if len(risk) else None,
               "train_tiles_out_of_fold": bool((train["fold"].astype(int) >= 0).all()) if len(train) else None}
    io.write_meta(meta_path, step=STEP, cfg=cfg,
                  inputs=[schemas.cases_csv(root), schemas.tiles_csv(root), risk_path],
                  outputs=[schemas.cell_tiles_csv(root)] + ([] if reused else [risk_path]),
                  args={"limit": ctx.limit, "force": ctx.force}, synthetic=ctx.synthetic, extra={"summary": summary})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
