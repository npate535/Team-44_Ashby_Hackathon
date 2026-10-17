"""Composite margin-localization fields (BUILD_PLAN §A12, Task 2; CPU job prep_b).

Each composite is one source-size field built from TEST images only: one half from a carcinoma case
(settings carcinoma_classes) and the other half from a benign case (settings benign_classes), both halves taken
from the same position in their source image. Orientation k % composites.orientations puts the carcinoma half on
the left, right, top or bottom (in that order). Pairing is seeded (composites.seed): carcinoma and benign cases are
drawn from seeded permutations and cycled, so every eligible case is used about equally often.

Per-margin ground truth (margin names from settings margin.names: top=superior, bottom=inferior, left=medial,
right=lateral):
  tumor_side      the margin lies entirely in the carcinoma half (it may still touch stroma: "carcinoma field",
                  not "tumor at ink")
  exact_negative  the margin lies entirely in the benign half: every segment there is a true negative inside a
                  carcinoma case, so a reader that merely echoes the case diagnosis scores 0% clear on it
  mixed           the margin crosses both halves
Outputs (data root): composites/<case_id>.png (full resolution), display/<case_id>.jpg, composites.csv (COMPOSITES),
manifest sets comp512 and comp256 (crop boxes on the composite PNG, encoders.views[grid] dihedral views; split =
test; labels live in composites.csv only). Composite case IDs: schemas.composite_case_id(carcinoma_case_id, k).

Composites are simulated variants: never part of headline metrics and no conformal coverage claim (§A5).
Resumable (existing images skipped unless --force); --limit N builds composites k < N only.

Run: python -m pipeline.composites [--limit 5]
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from core import cli, io, schemas
from pipeline import common

STEP = "composites"
SETS = ("comp512", "comp256")
SIDES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
COLUMNS = [c.name for c in schemas.COMPOSITES.columns]


def margin_truth(carcinoma_side: str, names: Mapping[str, str]) -> dict[str, str]:
    """{margin name: truth} for a composite whose carcinoma half is on `carcinoma_side`."""
    if carcinoma_side not in SIDES:
        raise ValueError(f"carcinoma_side must be one of {SIDES}, got {carcinoma_side!r}")
    out = {}
    for side in SIDES:
        if side == carcinoma_side:
            out[names[side]] = "tumor_side"
        elif side == OPPOSITE[carcinoma_side]:
            out[names[side]] = "exact_negative"
        else:
            out[names[side]] = "mixed"
    return out


def half_mask(width: int, height: int, carcinoma_side: str) -> np.ndarray:
    """Boolean HxW mask, True where the composite takes the carcinoma image."""
    mask = np.zeros((height, width), dtype=bool)
    if carcinoma_side == "left":
        mask[:, : width // 2] = True
    elif carcinoma_side == "right":
        mask[:, width // 2:] = True
    elif carcinoma_side == "top":
        mask[: height // 2, :] = True
    elif carcinoma_side == "bottom":
        mask[height // 2:, :] = True
    else:
        raise ValueError(f"carcinoma_side must be one of {SIDES}, got {carcinoma_side!r}")
    return mask


def compose(carcinoma: np.ndarray, benign: np.ndarray, carcinoma_side: str) -> np.ndarray:
    if carcinoma.shape != benign.shape:
        raise ValueError(f"source shapes differ: {carcinoma.shape} vs {benign.shape}")
    mask = half_mask(carcinoma.shape[1], carcinoma.shape[0], carcinoma_side)
    return np.where(mask[..., None], carcinoma, benign)


def plan_composites(cases: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """One row per composite k: carcinoma/benign source cases, orientation, side, truth columns (no images)."""
    ccfg = cfg["composites"]
    n, n_orient = int(ccfg["n"]), int(ccfg["orientations"])
    if not 1 <= n_orient <= len(SIDES):
        raise ValueError(f"composites.orientations must be 1..{len(SIDES)}, got {n_orient}")
    pool = cases[cases["split"] == ccfg["source_split"]].sort_values("case_id")
    carc = pool.loc[pool["label"].isin(cfg["carcinoma_classes"]), "case_id"].tolist()
    ben = pool.loc[pool["label"].isin(cfg["benign_classes"]), "case_id"].tolist()
    if not carc or not ben:
        raise ValueError(f"composites need carcinoma and benign {ccfg['source_split']} cases; got {len(carc)} / {len(ben)}")
    rng = np.random.default_rng(int(ccfg["seed"]))
    carc_order = [carc[i] for i in rng.permutation(len(carc))]
    ben_order = [ben[i] for i in rng.permutation(len(ben))]
    names = cfg["margin"]["names"]
    rows = []
    for k in range(n):
        orientation = k % n_orient
        side = SIDES[orientation]
        carcinoma_id = carc_order[k % len(carc_order)]
        case_id = schemas.composite_case_id(carcinoma_id, k)
        truth = margin_truth(side, names)
        rows.append({
            "case_id": case_id, "carcinoma_case_id": carcinoma_id, "benign_case_id": ben_order[k % len(ben_order)],
            "orientation": orientation, "carcinoma_side": side,
            "path": f"composites/{case_id}.png", "display_path": schemas.display_jpg(Path(""), case_id).as_posix(),
            **{f"truth_{m}": truth[m] for m in schemas.MARGINS},
        })
    return pd.DataFrame(rows)


def manifest_rows(comp: pd.DataFrame, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    views = cfg["encoders"]["views"]
    grid_of = {"comp512": "g512", "comp256": "g256"}
    rows: list[dict[str, Any]] = []
    for c in comp.itertuples(index=False):
        for set_name in SETS:
            grid = grid_of[set_name]
            for box in common.grid_boxes(cfg, grid):
                first = common.manifest_row(set_name=set_name, case_id=c.case_id, split=cfg["composites"]["source_split"],
                                            grid=grid, tile_idx=box["tile_idx"], path=c.path, x=box["x"], y=box["y"],
                                            w=box["size"], h=box["size"], view=0)
                rows.append(first)
                rows += [{**first, "row_id": f"{first['item_id']}|v{v}", "view": v} for v in range(1, int(views[grid]))]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Composite margin-localization fields from test images (§A12).")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    try:
        cases = io.read_csv(schemas.cases_csv(root), dtype=str, keep_default_na=False)
        schemas.validate_frame(schemas.CASES, cases)
        comp = plan_composites(cases, cfg)
        if ctx.limit is not None:
            comp = comp.iloc[: ctx.limit].copy()
        by_id = cases.set_index("case_id")
        size = tuple(cfg["dataset"]["image_size"])
        display = cfg["tiles"]["display"]
        written = 0
        for c in comp.itertuples(index=False):
            png, jpg = root / c.path, root / c.display_path
            if ctx.force or not (png.exists() and jpg.exists()):
                carc = np.asarray(common.read_rgb(by_id.loc[c.carcinoma_case_id, "image_path"], size))
                ben = np.asarray(common.read_rgb(by_id.loc[c.benign_case_id, "image_path"], size))
                img = Image.fromarray(compose(carc, ben, c.carcinoma_side), "RGB")
                io.save_image(png, img, compress_level=int(cfg["tiles"]["png_compress_level"]))
                io.save_image(jpg, common.to_display(img, int(display["long_side"])), quality=int(display["jpeg_quality"]))
                written += 1
        comp["width"], comp["height"] = size
        comp = comp[COLUMNS]
        schemas.validate_frame(schemas.COMPOSITES, comp)
        rows = manifest_rows(comp, cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2
    io.write_csv(schemas.composites_csv(root), comp, columns=COLUMNS)
    common.replace_manifest_sets(root, SETS, rows)
    summary = {"n_composites": int(len(comp)), "files_written": written,
               "per_side": comp["carcinoma_side"].value_counts().to_dict(),
               "carcinoma_labels": cases.set_index("case_id").loc[comp["carcinoma_case_id"], "label"].value_counts().to_dict(),
               "benign_labels": cases.set_index("case_id").loc[comp["benign_case_id"], "label"].value_counts().to_dict(),
               "manifest_rows": {s: sum(r["set"] == s for r in rows) for s in SETS}}
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg, inputs=[schemas.cases_csv(root)],
                  outputs=[schemas.composites_csv(root), root / "composites", schemas.embed_manifest_csv(root)],
                  args={"limit": ctx.limit, "force": ctx.force}, synthetic=ctx.synthetic,
                  extra={"summary": summary, "note": "simulated composite fields; never in headline metrics"})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
