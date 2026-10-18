"""Conformal coverage across random patient-grouped re-splits (docs/VALIDATION_ROUND3.md §1, pre-registered).

    python -m eval.conformal_resplit [--splits N]

The production model's calibrated probabilities (predictions.csv `pcal__*`, its fixed temperature; nothing is refit)
on the 182 held-out clean images (calibration + test; training images never used) are re-partitioned 1,000 times
(seed 20261018). Patient groups (cases.csv `group`) stay whole: groups are shuffled and added to the calibration part
until it holds at least 50 images; the remaining groups form the test part. Each partition refits the four methods of
eval/conformal.json at the same alpha and settings: LAC (marginal), APS (randomized as configured), Mondrian
(class-wise LAC) and the tier set (settings conformal.method x mode, top class added to empty sets when
conformal.nonempty). Per method: test coverage (mean and 2.5th / 50th / 97.5th percentiles across partitions), share of
partitions with coverage at or above 1 - alpha, mean set size, share of empty sets, share of invasive test images with
Invasive in the set and of non-invasive test images with Invasive in the set (means across partitions). The original
THUNDER partition is reported alongside from the sets stored in predictions.csv, i.e. the sets the worklist uses and
conformal.json reports. Randomized APS draws its U values from a per-partition seed (fit with that seed, sets with
seed + 1, as step 05 does with conformal.seed). Writes eval/conformal_resplit.json.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import cli, conformal, io, schemas
from eval.metrics import load_cases, load_predictions, set_matrix, write_eval

NAME = "conformal_resplit"
# Pre-registered in docs/VALIDATION_ROUND3.md §1 (written before any result); not settings, so they cannot drift.
N_SPLITS = 1000
SEED = 20261018
N_CAL = 50
HELD_OUT = ("calibration", "test")
STATS = ("coverage", "mean_set_size", "empty_share", "invasive_in_set_given_invasive",
         "invasive_in_set_given_not_invasive")
MEAN_STATS = STATS[1:]
STORED = {"lac_marginal": "set_lac", "aps": "set_aps", "mondrian": "set_mondrian"}


# ---- partitions ------------------------------------------------------------------------------------------------------
def _resplit_orders(groups: np.ndarray, n_cal: int, n_splits: int,
                    seed: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(calibration indices, test indices, shuffled group order) per partition; indices sorted ascending."""
    groups = np.asarray(groups).astype(str)
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if counts.sum() <= n_cal:
        raise ValueError(f"need more than {n_cal} images to leave a test part, got {counts.sum()}")
    rng = np.random.default_rng(seed)
    for _ in range(int(n_splits)):
        perm = rng.permutation(len(names))
        reached = np.cumsum(counts[perm]) >= n_cal
        stop = int(np.argmax(reached))                       # first group at which calibration holds >= n_cal images
        in_cal = np.zeros(len(names), dtype=bool)
        in_cal[perm[: stop + 1]] = True
        mask = in_cal[inverse]
        if mask.all():
            raise ValueError("a partition left no test images; the largest group is too large for this n_cal")
        yield np.flatnonzero(mask), np.flatnonzero(~mask), names[perm]


def grouped_resplits(groups: np.ndarray, n_cal: int, n_splits: int,
                     seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """`n_splits` calibration/test partitions that keep every group whole: shuffled groups are added to calibration
    until it holds at least `n_cal` images (it may overshoot); the remaining groups are the test part."""
    for cal, test, _order in _resplit_orders(groups, n_cal, n_splits, seed):
        yield cal, test


# ---- statistics ------------------------------------------------------------------------------------------------------
def _share(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def set_stats(sets: np.ndarray, labels: np.ndarray, target: int) -> dict[str, float | None]:
    """Statistics of one method's test sets; None where a statistic is undefined (no images of that kind)."""
    sets = np.asarray(sets, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    inv = labels == target
    return {"coverage": conformal.coverage(sets, labels), "mean_set_size": conformal.mean_set_size(sets),
            "empty_share": float(np.mean(sets.sum(axis=1) == 0)),
            "invasive_in_set_given_invasive": _share(sets[inv, target]),
            "invasive_in_set_given_not_invasive": _share(sets[~inv, target])}


def method_names(cfg: Mapping[str, Any]) -> list[str]:
    """Method keys exactly as eval/conformal.json names them (eval.metrics.conformal_payload)."""
    names = list(STORED)
    tier = conformal.tier_set_name(cfg)
    if tier not in ("lac_marginal", "aps_marginal"):
        names.append(tier)
    return names


def _tier_sets(cfg: Mapping[str, Any], by_source: Mapping[str, np.ndarray], probs: np.ndarray) -> np.ndarray:
    """The tier-driving set as step 05 builds it: the configured method's sets, top class added to empty rows."""
    sets = by_source[conformal.tier_set_source(cfg)]
    return conformal.nonempty_sets(sets, probs) if bool(cfg["conformal"].get("nonempty", False)) else sets


def split_stats(probs: np.ndarray, labels: np.ndarray, cal: np.ndarray, test: np.ndarray, alpha: float, target: int,
                seed: int, cfg: Mapping[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """Fit every method on `cal`, build sets on `test`, return {method: statistics}. Randomized APS: fit with `seed`,
    sets with `seed + 1`. Mondrian also reports `n_unbounded_classes` (classes with too few calibration images for a
    finite threshold, so they are in every set). `target` is the index of the aggressive (Invasive) class."""
    if cfg is None:
        from core import config

        cfg = config.load_settings()
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pc, yc, pt, yt = probs[cal], labels[cal], probs[test], labels[test]
    randomized = bool(cfg["conformal"]["aps_randomized"])
    q_mond = conformal.mondrian_quantiles(pc, yc, alpha)
    by_source = {
        "set_lac": conformal.lac_sets(pt, conformal.fit_lac(pc, yc, alpha)),
        "set_aps": conformal.aps_sets(pt, conformal.fit_aps(pc, yc, alpha, seed, randomized), seed + 1, randomized),
        "set_mondrian": conformal.mondrian_sets(pt, q_mond),
    }
    sets = {m: by_source[STORED[m]] for m in STORED}
    for m in method_names(cfg):
        if m not in sets:
            sets[m] = _tier_sets(cfg, by_source, pt)
    out = {m: set_stats(s, yt, target) for m, s in sets.items()}
    out["mondrian"]["n_unbounded_classes"] = int(np.sum(~np.isfinite(q_mond)))
    return out


def _mean_defined(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def aggregate(per_split: Sequence[Mapping[str, Mapping[str, Any]]], target_coverage: float) -> dict[str, Any]:
    """Distribution of each method's statistics across partitions. Coverage: mean and 2.5/50/97.5th percentiles,
    share of partitions at or above the target; other statistics: mean over the partitions where they are defined
    (`undefined_splits` counts the rest; null when undefined everywhere)."""
    out: dict[str, Any] = {}
    for m in per_split[0]:
        cov = np.array([s[m]["coverage"] for s in per_split], dtype=float)
        block: dict[str, Any] = {
            "coverage": {"mean": float(cov.mean()), "p2_5": float(np.percentile(cov, 2.5)),
                         "p50": float(np.percentile(cov, 50)), "p97_5": float(np.percentile(cov, 97.5))},
            # tolerance: a coverage of exactly 1 - alpha (e.g. 117/130) must count as reaching the target
            "share_at_or_above_target": float(np.mean(cov >= target_coverage - 1e-12)),
        }
        for k in MEAN_STATS:
            block[k] = _mean_defined([s[m][k] for s in per_split])
        block["undefined_splits"] = {k: int(sum(s[m][k] is None for s in per_split)) for k in MEAN_STATS}
        if "n_unbounded_classes" in per_split[0][m]:
            unb = np.array([s[m]["n_unbounded_classes"] for s in per_split])
            block["unbounded_class_share"] = float(np.mean(unb > 0))
            block["mean_unbounded_classes"] = float(unb.mean())
        out[m] = block
    return out


def resplit_summary(probs: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, n_cal: int, n_splits: int,
                    seed: int, alpha: float, target: int, methods: Sequence[str],
                    cfg: Mapping[str, Any]) -> dict[str, Any]:
    """aggregate() over `n_splits` grouped partitions; the per-partition APS seeds come from their own stream."""
    aps_seeds = np.random.default_rng([seed, 1]).integers(0, 2**31 - 2, size=n_splits)
    per_split, n_cals = [], []
    for (cal, test), s in zip(grouped_resplits(groups, n_cal, n_splits, seed), aps_seeds):
        stats = split_stats(probs, labels, cal, test, alpha, target, int(s), cfg)
        per_split.append({m: stats[m] for m in methods})
        n_cals.append(len(cal))
    out = aggregate(per_split, 1.0 - alpha)
    n = len(labels)
    cal_sizes = np.array(n_cals)
    out["_partition_sizes"] = {"n_cal": {"min": int(cal_sizes.min()), "p50": float(np.median(cal_sizes)),
                                         "max": int(cal_sizes.max()), "mean": float(cal_sizes.mean())},
                               "n_test": {"min": int(n - cal_sizes.max()), "p50": float(n - np.median(cal_sizes)),
                                          "max": int(n - cal_sizes.min()), "mean": float(n - cal_sizes.mean())}}
    return out


# ---- data ------------------------------------------------------------------------------------------------------------
def held_out(cfg: Mapping[str, Any], root: Path) -> pd.DataFrame:
    """Clean calibration + test cases (cases.csv) joined one-to-one with their production-model predictions."""
    cases = load_cases(cfg, root)
    preds = load_predictions(cfg, root)
    keep = cases[cases["split"].isin(HELD_OUT)][["case_id", "label", "split", "group"]]
    df = keep.merge(preds.drop(columns=["split"]), on="case_id", how="left", validate="one_to_one")
    missing = df.loc[df[schemas.cal_prob_col(cfg["classes"][0])].isna(), "case_id"].tolist()
    if missing:
        raise ValueError(f"{len(missing)} held-out cases have no predictions, e.g. {missing[:3]}")
    index = {c: i for i, c in enumerate(cfg["classes"])}
    unknown = sorted(set(df["label"]) - set(index))
    if unknown:
        raise ValueError(f"cases.csv labels {unknown} are not settings classes")
    df["y"] = df["label"].map(index).astype(int)
    df["group"] = df["group"].astype(str)
    return df.sort_values("case_id").reset_index(drop=True)


def original_split(df: pd.DataFrame, cfg: Mapping[str, Any], methods: Sequence[str], target: int) -> dict[str, Any]:
    """Statistics of the sets stored in predictions.csv on the original test split (what conformal.json reports)."""
    classes = list(cfg["classes"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    y = test["y"].to_numpy()
    stored = {col: set_matrix(test, col, classes) for col in STORED.values()}
    probs = test[[schemas.cal_prob_col(c) for c in classes]].to_numpy(dtype=float)
    out = {}
    for m in methods:
        if m in STORED:
            sets = stored[STORED[m]]
        elif "set_tier" in test.columns and test["set_tier"].notna().any():
            sets = set_matrix(test, "set_tier", classes)
        else:
            sets = _tier_sets(cfg, stored, probs)
        out[m] = set_stats(sets, y, target)
    return out


def check_against_conformal(orig: Mapping[str, Any], art: Path) -> dict[str, Any]:
    path = schemas.eval_json(art, "conformal")
    if not path.is_file():
        return {"matches_conformal_json": None, "note": "eval/conformal.json not present at run time"}
    conf = io.read_json(path).get("methods") or {}
    rows = {}
    for m, s in orig.items():
        ref = ((conf.get(m) or {}).get("coverage") or {}).get("estimate")
        rows[m] = {"original_split": s["coverage"], "conformal_json": ref,
                   "equal": None if ref is None else bool(abs(float(ref) - s["coverage"]) < 1e-9)}
    compared = [r["equal"] for r in rows.values() if r["equal"] is not None]
    return {"matches_conformal_json": bool(compared) and all(compared), "coverage": rows}


def run(ctx: cli.StepContext, n_splits: int | None = None) -> Path:
    cfg = ctx.cfg
    n_splits = N_SPLITS if n_splits is None else int(n_splits)
    alpha = float(cfg["alpha"])
    classes = list(cfg["classes"])
    target = classes.index(cfg["aggressive_class"])
    df = held_out(cfg, ctx.data_root)
    models = sorted(set(df["model"].astype(str)))
    if len(models) != 1:
        raise ValueError(f"held-out rows carry several production models {models}")
    choice_path = schemas.eval_json(ctx.artifacts_root, "model_choice")
    chosen = io.read_json(choice_path).get("production") if choice_path.is_file() else None
    if chosen is not None and chosen != models[0]:
        raise ValueError(f"predictions.csv model {models[0]!r} is not model_choice.json production {chosen!r}")
    probs = df[[schemas.cal_prob_col(c) for c in classes]].to_numpy(dtype=float)
    y, groups = df["y"].to_numpy(), df["group"].to_numpy()
    methods = method_names(cfg)
    summary = resplit_summary(probs, y, groups, n_cal=N_CAL, n_splits=n_splits, seed=SEED, alpha=alpha,
                              target=target, methods=methods, cfg=cfg)
    sizes = summary.pop("_partition_sizes")
    orig = original_split(df, cfg, methods, target)
    payload = {
        "n_images": int(len(df)), "n_groups": int(df["group"].nunique()), "n_splits": n_splits, "n_cal_target": N_CAL,
        "alpha": alpha, "target_coverage": 1.0 - alpha, "seed": SEED, "model": models[0],
        "drives_tiers": conformal.tier_set_name(cfg), "aps_randomized": bool(cfg["conformal"]["aps_randomized"]),
        "partition_sizes": sizes,
        "original": {"n_cal": int((df["split"] == "calibration").sum()), "n_test": int((df["split"] == "test").sum()),
                     "n_test_groups": int(df.loc[df["split"] == "test", "group"].nunique())},
        "methods": summary, "original_split": orig,
        "original_split_check": check_against_conformal(orig, ctx.artifacts_root),
        "pre_registration": "docs/VALIDATION_ROUND3.md §1",
        "notes": ["Coverage percentiles describe variation across calibration/test partitions of the same 182 "
                  "held-out images; they are not a confidence interval for a new cohort.",
                  f"{conformal.tier_set_name(cfg)} drives the worklist tiers and was chosen after viewing test "
                  "coverage on the original split; the re-split distribution is reported for every method alike.",
                  "Mondrian thresholds are infinite for a class with fewer calibration images than ceil((n+1)(1-alpha)) "
                  "needs; that class is then in every set (unbounded_class_share)."],
        "method": __doc__.split("\n\n")[1].strip(),
    }
    # no step meta file: this check only reads the data root (like eval.deletion_test)
    return write_eval(ctx.artifacts_root, NAME, payload, cfg, ctx.synthetic)


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.conformal_resplit", "Conformal coverage across random patient-grouped re-splits.")
    parser.add_argument("--splits", type=int, default=None, help=f"partitions (default {N_SPLITS}, pre-registered)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    try:
        out = run(ctx, args.splits)
    except (FileNotFoundError, ValueError) as exc:
        print(f"eval.conformal_resplit: {exc}", file=sys.stderr)
        return 1
    print(f"eval.conformal_resplit: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
