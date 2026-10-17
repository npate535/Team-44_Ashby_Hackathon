"""Step 05b: ABMIL attention heatmaps (tile_attention.csv) when ABMIL is used (BUILD_PLAN Task 10; cut order §I item 4).

    python -m pipeline.05b_abmil [--data-root D] [--artifacts-root A] [--limit N] [--force]

05_risk_model trains and evaluates `abmil` (gated-attention MIL on the best single encoder) inside the same
patient-grouped CV protocol as every other candidate and exports its weights to models/abmil.npz. This step loads
those weights (never refits) and writes per-tile attention for calibration/test cases and their degraded/composite
variants, but only when eval/model_choice.json shows ABMIL is used (production model or model A). Otherwise it
removes a stale tile_attention.csv and exits 0, so 08 omits tiles[].attention (core.schemas.TILE_ATTENTION).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from core import cli, io, schemas
from pipeline import riskmodels as rm

STEP = "05b_abmil"
NAME = "abmil"


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "ABMIL attention heatmaps from the exported 05 weights.")
    ctx = cli.context(parser.parse_args(argv))
    cfg, root = ctx.cfg, ctx.data_root
    out_path = schemas.tile_attention_csv(root)
    choice_path = schemas.eval_json(ctx.artifacts_root, "model_choice")
    if not choice_path.is_file():
        print(f"{STEP}: run pipeline.05_risk_model first ({choice_path} missing)", file=sys.stderr)
        return 2
    choice = io.read_json(choice_path)
    weights = root / cfg["risk_model"]["models_dir"] / f"{NAME}.npz"
    used = NAME in (choice.get("production"), choice.get("model_a"))
    if not used or not weights.is_file():
        if out_path.is_file():
            out_path.unlink()
        why = "ABMIL is neither the production model nor model A" if weights.is_file() else "ABMIL was not trained"
        print(f"{STEP}: {why}; no tile_attention.csv")
        return 0
    if out_path.is_file() and not ctx.force:
        print(f"{STEP}: outputs exist; use --force to recompute")
        return 0
    z = io.load_npz(weights)
    enc, set_name, head = str(z["encoder"]), str(z["set"]), str(z["head"])
    model = rm.ABMIL.from_npz(weights, cfg["abmil"])
    cases = rm.read_cases(root)
    tiles = rm.read_tiles(root)
    split = dict(zip(cases.case_id, cases.split))
    keep = None
    if ctx.limit is not None:
        keep = set(cli.limit_case_ids(cases[cases.split.isin(["calibration", "test"])].case_id, ctx.limit))
    rows = []
    sources = [(set_name, False), (rm.DEG_SETS.get(set_name), False), (rm.COMP_SETS.get(set_name), True)]
    for sname, comp in sources:
        if not sname or not rm.emb_exists(root, enc, sname, head):
            continue
        x, idx = rm.load_emb(root, enc, sname, head)
        ids, bags, tile_idx = rm.tile_bags(x, idx, tiles, composite=comp)
        grid = str(idx["grid"].iloc[0])
        sel = []
        for i, rid in enumerate(ids):
            src = rid.split("_FSIM_")[0].split("_COMP_")[0]
            if split.get(src) in ("calibration", "test") and (keep is None or src in keep):
                sel.append(i)
        if not sel:
            continue
        _, att = model.predict([bags[i] for i in sel])
        for i, a in zip(sel, att):
            src = ids[i].split("_FSIM_")[0].split("_COMP_")[0]
            for t, w in zip(tile_idx[i], a):
                rows.append({"case_id": ids[i], "split": split[src], "grid": grid, "tile_idx": int(t),
                             "attention": float(w), "model": f"{NAME}_{enc}"})
    df = pd.DataFrame(rows).sort_values(["case_id", "grid", "tile_idx"]).reset_index(drop=True)
    schemas.validate_frame(schemas.TILE_ATTENTION, df)
    sums = df.groupby("case_id").attention.sum()
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise RuntimeError("attention weights do not sum to 1 per case")
    io.write_csv(out_path, df)
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg, inputs=[weights, f"emb/{enc}/"],
                  outputs=[out_path], args={"limit": ctx.limit, "force": ctx.force}, synthetic=ctx.synthetic)
    print(f"{STEP}: attention for {df.case_id.nunique()} reads ({NAME}_{enc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
