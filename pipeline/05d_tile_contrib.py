"""Step 05d: exact region contributions of each read's own linear model (docs/VALIDATION_ROUND3.md §3).

    python -m pipeline.05d_tile_contrib [--data-root D] [--artifacts-root A] [--force]

The production read is a multinomial logistic regression on the mean of a read's QC-passing tile embeddings
(pipeline.riskmodels.pool_tiles). With x_r the read's n pooled embedding rows, standardized s_r = (x_r - mean) / scale,
centered weights w = W_target - mean_k W_k and centered intercept b = b_target - mean_k b_k, the calibrated centered
logit of the target class is

    (logit_target(z) - mean_k logit_k(z)) / T = b / T + sum_r (s_r . w) / (n T),    z = mean_r x_r,

so each row adds (s_r . w) / (n T) and the constant is b / T. A row's term is zero when it equals the training mean.
Rows are summed per native tile, and native tiles (256-px for the 256-px fallback model) are summed into the contract
grid regions that contain them. Because pcal = softmax(logits / T), the constant plus the region sum equals
log pcal_target - mean_k log pcal_k; the step checks that identity for every read against predictions.csv.

Reads: every calibration/test row of predictions.csv, each with its own `model` and `temperature`. The model's
feature set comes from settings risk_model.embedding_candidates (the table 05_risk_model uses); clean reads use its
set, degraded reads riskmodels.DEG_SETS[set] with the source case's QC, composites riskmodels.COMP_SETS[set] with
every tile. A read whose model is not a mean-pooled embedding probe (concept models, ABMIL, fusion, precomputed
logits) or that has no per-tile embeddings gets no rows: its contributions are null. Train rows are out-of-fold
averages of CV fold models, not reads of the exported model, so they are not decomposed.

Writes tile_contrib.csv (data root; schemas.TILE_CONTRIB) and eval/tile_contrib.json (identity check, null reads).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import cli, io, schemas
from pipeline import riskmodels as rm

STEP = "05d_tile_contrib"
COLUMNS = [c.name for c in schemas.TILE_CONTRIB.columns]
TOLERANCE = 1e-6           # acceptance criterion (VALIDATION_ROUND3 §3)
PROB_FLOOR = 1e-12         # reads with any pcal at or below this are not checked (log of a rounded probability)
SERVED_SPLITS = ("calibration", "test")


# ---- the decomposition ------------------------------------------------------------------------------------------------
def read_contributions(model: rm.LinearModel, rows: np.ndarray, tile_of_row: np.ndarray, temperature: float,
                       target: int) -> tuple[pd.Series, float]:
    """Per-tile terms (index = tile key) and the constant of the read pooled from `rows`:
    constant + terms.sum() == (logit_target(z) - mean_k logit_k(z)) / temperature for z = rows.mean(0)."""
    w = model.coef[target] - model.coef.mean(axis=0)
    b = float(model.intercept[target] - model.intercept.mean())
    per_row = model.standardize(np.asarray(rows, dtype=np.float64)) @ w / (len(rows) * temperature)
    terms = pd.Series(per_row).groupby(np.asarray(tile_of_row)).sum()
    terms.index.name = "tile_idx"
    return terms, b / temperature


def grid_tiles(spec: Mapping[str, Any]) -> pd.DataFrame:
    """Tile geometry of a settings tiles.grids entry: tile_idx, x, y, size (row-major, as 01 and 08 number them)."""
    size, rows, cols = int(spec["size"]), int(spec["rows"]), int(spec["cols"])
    idx = np.arange(rows * cols)
    return pd.DataFrame({"tile_idx": idx, "x": (idx % cols) * size, "y": (idx // cols) * size, "size": size})


def to_contract_grid(contrib: pd.Series, native_tiles: pd.DataFrame, contract_tiles: pd.DataFrame) -> pd.Series:
    """Sum native-tile terms into the contract regions that contain each native tile's center."""
    nat = native_tiles.set_index("tile_idx").loc[contrib.index]
    cx = (nat["x"] + nat["size"] / 2).to_numpy()[:, None]
    cy = (nat["y"] + nat["size"] / 2).to_numpy()[:, None]
    con = contract_tiles
    x0, y0, s = (con[c].to_numpy()[None, :] for c in ("x", "y", "size"))
    inside = (cx >= x0) & (cx < x0 + s) & (cy >= y0) & (cy < y0 + s)
    if not (inside.sum(1) == 1).all():
        raise ValueError("native tiles outside the contract grid or in more than one region")
    region = con["tile_idx"].to_numpy()[inside.argmax(1)]
    out = pd.Series(contrib.to_numpy(), index=region).groupby(level=0).sum()
    out.index.name = "tile_idx"
    return out


def set_contributions(model: rm.LinearModel, x: np.ndarray, index: pd.DataFrame, qc: pd.DataFrame | None,
                      reads: Mapping[str, float], target: int, cfg: Mapping[str, Any], *,
                      composite: bool = False) -> pd.DataFrame:
    """tile_contrib rows (without `model`/`target`) for the reads of one embedding set: read_id -> temperature.
    Pools exactly like riskmodels.pool_tiles (QC tiles, all-tiles fallback; composites keep every tile)."""
    ids = rm.read_ids(index)
    sel = np.isin(ids.astype(str), list(reads))
    if not sel.any():
        return pd.DataFrame(columns=["read_id", "tile_idx", "contribution", "constant", "temperature", "n_rows"])
    sub = index.loc[sel].reset_index(drop=True)
    grids = cfg["tiles"]["grids"]
    native = grid_tiles(grids[str(sub["grid"].iloc[0])])
    contract = grid_tiles(grids[cfg["tiles"]["contract_grid"]])
    read_list, bags, tiles = rm.tile_bags(x[sel], sub, qc, composite=composite)
    parts = []
    for rid, bag, tile_ids in zip(read_list, bags, tiles):
        t = float(reads[rid])
        terms, constant = read_contributions(model, bag, tile_ids, t, target)
        region = to_contract_grid(terms, native, contract)
        parts.append(pd.DataFrame({"read_id": rid, "tile_idx": region.index.astype(int), "contribution": region.values,
                                   "constant": constant, "temperature": t, "n_rows": len(bag)}))
    return pd.concat(parts, ignore_index=True)


def region_values(table: pd.DataFrame, read_id: str, tile_ids: Iterable[int]) -> list[float | None]:
    """Contribution per contract region of one read; None where the region has no row (QC-failed, or no data)."""
    part = table[table["read_id"] == read_id]
    vals = dict(zip(part["tile_idx"].astype(int), part["contribution"].astype(float)))
    return [vals.get(int(t)) for t in tile_ids]


# ---- the step -------------------------------------------------------------------------------------------------------------
def _model_sets(spec: Mapping[str, str]) -> list[tuple[str, bool]]:
    """(embedding set, composite) pairs a model reads: clean, degraded, composite."""
    out = [(spec["set"], False)]
    if spec["set"] in rm.DEG_SETS:
        out.append((rm.DEG_SETS[spec["set"]], False))
    if spec["set"] in rm.COMP_SETS:
        out.append((rm.COMP_SETS[spec["set"]], True))
    return out


def compute(root: Path, cfg: Mapping[str, Any], pred: pd.DataFrame, tiles: pd.DataFrame
            ) -> tuple[pd.DataFrame, dict[str, str]]:
    """(tile_contrib table, {read_id: why its contributions are null}) for the served rows of predictions.csv."""
    target_name = cfg["aggressive_class"]
    specs = cfg["risk_model"]["embedding_candidates"]
    models_dir = Path(root) / cfg["risk_model"]["models_dir"]
    served = pred[pred["split"].isin(SERVED_SPLITS)]
    null: dict[str, str] = {}
    parts = []
    for model_id, rows in served.groupby("model", sort=True):
        reads = dict(zip(rows["case_id"].astype(str), rows["temperature"].astype(float)))
        if model_id not in specs:
            null.update({r: f"{model_id} is not a mean-pooled embedding probe" for r in reads})
            continue
        try:
            model = rm.LinearModel.from_npz(models_dir / f"{model_id}.npz")
        except (FileNotFoundError, ValueError) as err:
            null.update({r: f"{model_id}: no linear model export ({err})" for r in reads})
            continue
        if list(model.classes) != list(cfg["classes"]):
            raise ValueError(f"{model_id}: classes {model.classes} differ from settings classes")
        target = list(model.classes).index(target_name)
        spec = specs[model_id]
        found = []
        for set_name, composite in _model_sets(spec):
            if not rm.emb_exists(root, spec["encoder"], set_name, spec["head"]):
                continue
            x, index = rm.load_emb(root, spec["encoder"], set_name, spec["head"])
            part = set_contributions(model, x, index, None if composite else tiles, reads, target, cfg,
                                     composite=composite)
            found.append(part)
        got = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["read_id"])
        if len(got) and got.duplicated(["read_id", "tile_idx"]).any():
            raise ValueError(f"{model_id}: a read appears in more than one embedding set")
        null.update({r: f"{model_id}: no per-tile embeddings for this read" for r in reads if r not in set(got["read_id"])})
        if len(got):
            parts.append(got.assign(model=model_id, target=target_name))
    if parts:
        table = pd.concat(parts, ignore_index=True)[COLUMNS]
        table = table.sort_values(["read_id", "tile_idx"]).reset_index(drop=True)
    else:
        table = pd.DataFrame(columns=COLUMNS)
    return table, dict(sorted(null.items()))


def identity_check(table: pd.DataFrame, pred: pd.DataFrame, classes: Sequence[str], target: str) -> dict[str, Any]:
    """Max |constant + sum(contributions) - (log pcal_target - mean_k log pcal_k)| over reads with every pcal > floor."""
    p = pred.set_index("case_id")[[schemas.cal_prob_col(c) for c in classes]].astype(float)
    t = list(classes).index(target)
    errors, saturated = {}, []
    for rid, part in table.groupby("read_id", sort=True):
        probs = p.loc[rid].to_numpy()
        if probs.min() <= PROB_FLOOR:
            saturated.append(rid)
            continue
        want = np.log(probs[t]) - np.log(probs).mean()
        errors[rid] = abs(float(part["contribution"].sum() + part["constant"].iloc[0]) - want)
    worst = max(errors, key=errors.get) if errors else None
    return {"tolerance": TOLERANCE, "prob_floor": PROB_FLOOR, "n_reads_checked": len(errors),
            "max_abs_error": errors[worst] if worst else None, "worst_read": worst,
            "n_reads_over_tolerance": sum(e > TOLERANCE for e in errors.values()),
            "skipped_saturated": saturated, "passed": bool(errors) and max(errors.values()) <= TOLERANCE}


def run(ctx: cli.StepContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg, root = ctx.cfg, ctx.data_root
    pred = rm.read_csv(schemas.predictions_csv(root))
    table, null = compute(root, cfg, pred, rm.read_tiles(root))
    schemas.validate_frame(schemas.TILE_CONTRIB, table)
    reasons: dict[str, int] = {}
    for why in null.values():
        reasons[why] = reasons.get(why, 0) + 1
    served = pred[pred["split"].isin(SERVED_SPLITS)]
    body = {
        "target": cfg["aggressive_class"], "unit": "centered calibrated logit",
        "n_reads_served": int(len(served)),
        "n_reads_with_contributions": int(table["read_id"].nunique()),
        "n_reads_null": len(null), "null_reasons": reasons,
        "reads_by_model": {str(m): int(n) for m, n in table.groupby("model")["read_id"].nunique().items()},
        "n_train_rows_not_decomposed": int((pred["split"] == "train").sum()),
        "identity": identity_check(table, pred, cfg["classes"], cfg["aggressive_class"]),
        "method": __doc__.split("\n\n")[1].strip(),
    }
    return table, body


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Exact region contributions of each read's linear model.")
    ctx = cli.context(parser.parse_args(argv))
    out_csv, out_json = schemas.tile_contrib_csv(ctx.data_root), schemas.eval_json(ctx.artifacts_root, "tile_contrib")
    if io.outputs_exist(out_csv, out_json) and not ctx.force:
        print(f"{STEP}: outputs exist; use --force to recompute")
        return 0
    if not schemas.predictions_csv(ctx.data_root).is_file():
        print(f"{STEP}: run pipeline.05_risk_model first", file=sys.stderr)
        return 2
    table, body = run(ctx)
    ident = body["identity"]
    print(f"{STEP}: {body['n_reads_with_contributions']} reads decomposed, {body['n_reads_null']} null; identity "
          f"max |error| = {ident['max_abs_error']} over {ident['n_reads_checked']} reads "
          f"({len(ident['skipped_saturated'])} saturated reads skipped)")
    for why, n in body["null_reasons"].items():
        print(f"{STEP}:   null: {n} x {why}")
    if table["read_id"].nunique() and not ident["passed"]:
        print(f"{STEP}: identity check FAILED (tolerance {TOLERANCE}); nothing written", file=sys.stderr)
        return 1
    io.write_csv(out_csv, table)
    prov = io.provenance(ctx.cfg, synthetic=ctx.synthetic)
    doc = {"name": "tile_contrib", "created_utc": prov["created_utc"], "config_hash": prov["config_hash"],
           "git_sha": prov["git_sha"], "synthetic": bool(ctx.synthetic), **body}
    io.write_json(out_json, rm.jsonable(doc))
    print(f"{STEP}: wrote {out_csv} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
