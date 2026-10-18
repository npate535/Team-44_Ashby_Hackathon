"""Deferral (risk-coverage) curve: accuracy on the reads Sepia keeps as the least reliable reads are deferred first.

    python -m eval.deferral

Every analysis choice is fixed by docs/VALIDATION_ROUND3.md section 2 (pre-registered before any result):
  * Reads. Population (a) clean_test: the clean test images. Population (b) all_test_reads: clean test reads plus the
    simulated degraded variants (blur, JPEG, stain shift, resampling); composites are excluded. Each read is scored
    with the model that produced it in predictions.csv (`top`, as in the app); its true label is its source case's
    label in cases.csv. Reads are joined to reliability.csv on read ID; any unmatched read is an error.
  * Rankings (what is deferred first). reliability: lowest reliability score first (primary). top_prob: lowest
    calibrated top-class probability first. set_size: largest conformal tier set first (the set the app shows), ties
    by lowest top-class probability. random: mean over 1,000 random orders (seed 20261018). Remaining ties: read ID.
  * Fractions 0%..50% in steps of 5%; floor(fraction x n) reads are deferred.
  * Per fraction: four-class accuracy on kept reads; share of all errors that were deferred; invasive reads called
    non-invasive among kept reads; in situ reads called invasive among kept reads.
  * Headline: accuracy on kept reads at 20% deferral in population (b) under the reliability ranking, with a 95%
    patient-grouped bootstrap interval (settings.bootstrap: 2,000 resamples of `group`), against 0% deferral and
    against random deferral at 20%. In each resample the 20% rule is re-applied to the resampled reads.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import cli, io, schemas
from eval.metrics import ci, load_cases, load_predictions, write_eval

STEP = "eval_deferral"
FRACTIONS = tuple(round(0.05 * i, 2) for i in range(11))
RANDOM_ORDERS = 1000
RANDOM_SEED = 20261018
HEADLINE_FRACTION = 0.2
HEADLINE_POPULATION = "all_test_reads"
RANKINGS = ("reliability", "top_prob", "set_size")
FLAGS = ("invasive_missed", "insitu_overcalled")


def n_deferred(n: int, fraction: float) -> int:
    """Reads deferred at `fraction` of n: floor(fraction * n), robust to float noise (0.35 * 100 -> 35)."""
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError(f"deferral fraction must be in [0, 1], got {fraction!r}")
    return int(math.floor(float(fraction) * n + 1e-9))


def risk_coverage(correct: np.ndarray, defer_first: np.ndarray, fractions: Sequence[float],
                  flags: Mapping[str, np.ndarray] | None = None) -> list[dict[str, float]]:
    """One row per fraction. `defer_first` is the sort key (smallest deferred first; a stable sort, so equal keys
    keep the upstream row order, i.e. read ID). `flags` maps a name to a boolean per read; each row then counts
    `<name>_kept` among the kept reads."""
    correct = np.asarray(correct).astype(bool)
    key = np.asarray(defer_first, dtype=float)
    n = len(correct)
    if n == 0 or len(key) != n:
        raise ValueError(f"correct and defer_first must be non-empty and aligned, got {n} and {len(key)}")
    if np.isnan(key).any():
        raise ValueError("defer_first has NaN values")
    flags = {k: np.asarray(v).astype(bool) for k, v in (flags or {}).items()}
    if any(len(v) != n for v in flags.values()):
        raise ValueError("every flag array must align with correct")
    order = np.argsort(key, kind="stable")
    c = correct[order]
    ordered = {k: v[order] for k, v in flags.items()}
    total_errors = int(n - c.sum())
    rows = []
    for f in fractions:
        k = n_deferred(n, f)
        kept = c[k:]
        row: dict[str, float] = {
            "fraction": float(f), "kept": int(kept.size),
            "accuracy": float(kept.mean()) if kept.size else math.nan,
            "errors_deferred_share": float((~c[:k]).sum() / total_errors) if total_errors else math.nan,
        }
        for name, v in ordered.items():
            row[f"{name}_kept"] = int(v[k:].sum())
        rows.append(row)
    return rows


def random_risk_coverage(correct: np.ndarray, fractions: Sequence[float], flags: Mapping[str, np.ndarray],
                         n_orders: int = RANDOM_ORDERS, seed: int = RANDOM_SEED) -> list[dict[str, float]]:
    """Mean of risk_coverage over `n_orders` uniformly random deferral orders (seeded)."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    runs = [risk_coverage(correct, rng.permutation(n).astype(float), fractions, flags) for _ in range(int(n_orders))]
    out = []
    for i, f in enumerate(fractions):
        row: dict[str, float] = {"fraction": float(f), "kept": runs[0][i]["kept"]}
        for key in runs[0][i]:
            if key not in row:
                row[key] = float(np.mean([r[i][key] for r in runs]))
        out.append(row)
    return out


def _ordinal(*keys: np.ndarray) -> np.ndarray:
    """Unique rank (0 = deferred first) by keys in priority order, then row order (read ID)."""
    n = len(keys[0])
    order = np.lexsort((np.arange(n),) + tuple(reversed([np.asarray(k, dtype=float) for k in keys])))
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n)
    return rank


def ranking_keys(reads: pd.DataFrame) -> dict[str, np.ndarray]:
    """defer_first keys for the pre-registered rankings; rows must already be in read-ID order."""
    rel = reads["reliability"].to_numpy(float)
    top = reads["top_prob"].to_numpy(float)
    size = reads["set_size"].to_numpy(float)
    return {"reliability": _ordinal(rel), "top_prob": _ordinal(top), "set_size": _ordinal(-size, top)}


def _sample(ids: Sequence[str]) -> str:
    ids = sorted(ids)
    return ", ".join(ids[:3]) + (" ..." if len(ids) > 3 else "")


def build_reads(cases: pd.DataFrame, preds: pd.DataFrame, rel: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Test reads (clean + simulated degraded, composites excluded) in read-ID order, with truth, prediction, scores
    and the error flags. Raises on any unmatched read."""
    classes = list(cfg["classes"])
    aggressive = cfg["aggressive_class"]
    insitu = [c for c in cfg["carcinoma_classes"] if c != aggressive]
    if len(insitu) != 1:
        raise ValueError(f"expected one non-aggressive carcinoma class in settings, got {insitu}")
    insitu = insitu[0]
    rel_test = rel[rel["split"] == "test"]
    composite = set(rel_test.loc[rel_test["variant"] == "composite", "case_id"])
    rel_test = rel_test[rel_test["variant"] != "composite"]
    if rel_test["case_id"].duplicated().any():
        raise ValueError("reliability.csv has duplicate test read IDs")
    preds_test = preds[(preds["split"] == "test") & ~preds["case_id"].isin(composite)]
    no_pred = set(rel_test["case_id"]) - set(preds_test["case_id"])
    if no_pred:
        raise ValueError(f"{len(no_pred)} test reads are missing from predictions.csv: {_sample(no_pred)}")
    no_rel = set(preds_test["case_id"]) - set(rel_test["case_id"])
    if no_rel:
        raise ValueError(f"{len(no_rel)} test predictions are missing from reliability.csv: {_sample(no_rel)}")
    test_cases = cases[cases["split"] == "test"].set_index("case_id")
    clean = rel_test[rel_test["variant"] == "clean"]
    if (clean["case_id"] != clean["source_case_id"]).any():
        raise ValueError("reliability.csv clean rows must have source_case_id == case_id")
    no_clean = set(test_cases.index) - set(clean["case_id"])
    if no_clean:
        raise ValueError(f"{len(no_clean)} test cases have no clean read in reliability.csv: {_sample(no_clean)}")
    bad_src = set(rel_test["source_case_id"]) - set(test_cases.index)
    if bad_src:
        raise ValueError(f"{len(bad_src)} reads derive from cases that are not test cases: {_sample(bad_src)}")
    if "set_tier_size" not in preds_test.columns:
        raise ValueError("predictions.csv has no set_tier_size column (the conformal tier set the app shows)")
    df = rel_test[["case_id", "source_case_id", "variant", "reliability"]].merge(
        preds_test[["case_id", "model", "top", "set_tier_size"] + [schemas.cal_prob_col(c) for c in classes]],
        on="case_id", how="inner", validate="one_to_one")
    unknown = sorted(set(df["top"]) - set(classes))
    if unknown:
        raise ValueError(f"predictions.csv top classes {unknown} are not settings classes")
    probs = df[[schemas.cal_prob_col(c) for c in classes]].to_numpy(float)
    df["top_prob"] = probs[np.arange(len(df)), df["top"].map(classes.index).to_numpy()]
    df["truth"] = df["source_case_id"].map(test_cases["label"])
    df["group"] = df["source_case_id"].map(test_cases["group"]).astype(str)
    df["set_size"] = pd.to_numeric(df["set_tier_size"])
    df["reliability"] = pd.to_numeric(df["reliability"])
    if df[["reliability", "top_prob", "set_size"]].isna().any().any():
        raise ValueError("a test read has a blank reliability, probability or set size")
    df["clean"] = df["variant"] == "clean"
    df["correct"] = df["top"] == df["truth"]
    df["invasive_missed"] = (df["truth"] == aggressive) & (df["top"] != aggressive)
    df["insitu_overcalled"] = (df["truth"] == insitu) & (df["top"] == aggressive)
    df = df.rename(columns={"case_id": "read_id"}).sort_values("read_id", kind="stable").reset_index(drop=True)
    return df[["read_id", "source_case_id", "variant", "clean", "model", "group", "truth", "top", "top_prob",
               "set_size", "reliability", "correct", *FLAGS]]


def load_reads(cfg: Mapping[str, Any], root: Path) -> pd.DataFrame:
    rel_path = schemas.reliability_csv(root)
    if not rel_path.is_file():
        raise FileNotFoundError(f"missing input reliability.csv: {rel_path}")
    return build_reads(load_cases(cfg, root), load_predictions(cfg, root), io.read_csv(rel_path), cfg)


def population_block(reads: pd.DataFrame, fractions: Sequence[float] = FRACTIONS) -> dict[str, Any]:
    correct = reads["correct"].to_numpy(bool)
    flags = {k: reads[k].to_numpy(bool) for k in FLAGS}
    keys = ranking_keys(reads)
    rankings = {name: risk_coverage(correct, keys[name], fractions, flags) for name in RANKINGS}
    rankings["random"] = random_risk_coverage(correct, fractions, flags)
    return {"n_reads": int(len(reads)), "n_groups": int(reads["group"].nunique()), "n_errors": int((~correct).sum()),
            "models": {str(k): int(v) for k, v in reads["model"].value_counts().sort_index().items()},
            "variants": {str(k): int(v) for k, v in reads["variant"].value_counts().sort_index().items()},
            "rankings": rankings}


def headline(reads: pd.DataFrame, cfg: Mapping[str, Any], random_row: Mapping[str, float]) -> dict[str, Any]:
    correct = reads["correct"].to_numpy(float)
    key = ranking_keys(reads)["reliability"]
    groups = reads["group"].to_numpy(str)
    f = HEADLINE_FRACTION

    def kept(c: np.ndarray, k: np.ndarray) -> float:
        return risk_coverage(c, k, [f])[0]["accuracy"]

    def gain(c: np.ndarray, k: np.ndarray) -> float:
        return kept(c, k) - float(np.mean(c))

    # the same seed offset for all three -> the same resamples (paired)
    return {"population": HEADLINE_POPULATION, "fraction": f, "ranking": "reliability",
            "deferred": n_deferred(len(reads), f), "kept": int(len(reads) - n_deferred(len(reads), f)),
            "accuracy_kept": ci(kept, [correct, key], groups, cfg),
            "accuracy_all": ci(lambda c, k: float(np.mean(c)), [correct, key], groups, cfg),
            "accuracy_random_20": float(random_row["accuracy"]),
            "gain_vs_all": ci(gain, [correct, key], groups, cfg),
            "gain_vs_all_note": "accuracy_kept minus accuracy_all on the same resamples; reported in addition to the "
                                "pre-registered headline, not part of it"}


def production_model(art: Path, reads: pd.DataFrame) -> str:
    """Production model from eval/model_choice.json; else the single model that produced the clean test reads."""
    path = schemas.eval_json(art, "model_choice")
    clean_models = sorted(set(reads.loc[reads["clean"], "model"]))
    if path.is_file():
        prod = io.read_json(path).get("production")
        if prod and clean_models != [prod]:
            raise ValueError(f"clean test reads come from {clean_models}, but model_choice.json names {prod!r}")
        if prod:
            return str(prod)
    if len(clean_models) != 1:
        raise ValueError(f"clean test reads come from several models {clean_models}")
    return clean_models[0]


def run(ctx: cli.StepContext) -> Path:
    cfg = ctx.cfg
    reads = load_reads(cfg, ctx.data_root)
    prod = production_model(ctx.artifacts_root, reads)
    pops = {"clean_test": population_block(reads[reads["clean"]].reset_index(drop=True)),
            "all_test_reads": population_block(reads)}
    i20 = list(FRACTIONS).index(HEADLINE_FRACTION)
    head = headline(reads, cfg, pops[HEADLINE_POPULATION]["rankings"]["random"][i20])
    payload = {
        "pre_registration": "docs/VALIDATION_ROUND3.md section 2",
        "production_model": prod,
        "fractions": list(FRACTIONS),
        "protocol": {
            "populations": {"clean_test": "the clean held-out test images",
                            "all_test_reads": "clean test reads plus the simulated degraded variants (blur, JPEG, "
                                              "stain shift, resampling); composites excluded"},
            "scoring": "each read's predicted class is `top` in predictions.csv, from the model that produced it "
                       "(production model for clean reads, fallback model for simulated variants), as in the app; "
                       "truth is the source case's label",
            "rankings": {"reliability": "lowest reliability score (reliability.csv) deferred first",
                         "top_prob": "lowest calibrated top-class probability deferred first",
                         "set_size": "largest conformal tier set (set_tier_size) deferred first, ties by lowest "
                                     "top-class probability",
                         "random": f"mean over {RANDOM_ORDERS} random orders, seed {RANDOM_SEED}",
                         "ties": "read ID"},
            "deferred_count": "floor(fraction x n_reads)",
            "statistics": {"accuracy": "four-class accuracy on kept reads",
                           "errors_deferred_share": "share of all errors in the population that were deferred "
                                                    "(null when there are no errors)",
                           "invasive_missed_kept": "kept reads truly Invasive and called non-invasive",
                           "insitu_overcalled_kept": "kept reads truly InSitu and called Invasive"},
            "interval": "95% patient-grouped percentile bootstrap (settings.bootstrap; groups = cases.csv group); "
                        "the 20% deferral rule is re-applied within each resample",
        },
        "populations": pops,
        "headline": head,
        "notes": ["Degraded variants are simulated frozen-section artifacts applied to microscopy images; no real "
                  "frozen sections were available.",
                  "Deferred reads are sent to closer review; this measures accuracy on the reads kept, not a "
                  "clinical outcome."],
    }
    out = write_eval(ctx.artifacts_root, "deferral", payload, cfg, ctx.synthetic)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=cfg,
                  inputs=[schemas.cases_csv(ctx.data_root), schemas.predictions_csv(ctx.data_root),
                          schemas.reliability_csv(ctx.data_root)],
                  outputs=[out], args={}, synthetic=ctx.synthetic)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.deferral", "Deferral (risk-coverage) curve on test reads.")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    try:
        out = run(ctx)
    except (FileNotFoundError, ValueError) as exc:
        print(f"eval.deferral: {exc}", file=sys.stderr)
        return 1
    print(f"eval.deferral: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
