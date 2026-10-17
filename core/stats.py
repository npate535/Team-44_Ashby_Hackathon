"""Evaluation statistics: cluster/stratified bootstrap CIs and classification metrics (plan §12, BUILD_PLAN §A5).

FROZEN INTERFACE (Task 1 stub; Task 4 implements; tests/test_contract.py pins every signature).
Callers: eval/*.py, pipeline/04_concepts.py, 04_cav.py, 05_risk_model.py, 06_reliability.py, 07_retrieval.py.

Conventions:
  * y_true: int (n,) class indices (or 0/1 for binary metrics); probs: float (n, C); scores: float (n,).
  * Bootstrap (settings.bootstrap): n_boot = 2000, ci = 0.95, seeded. When `groups` is given, whole groups
    (patient / pseudo-patient IDs) are resampled (cluster bootstrap, §A5); `strata` resamples within each
    stratum (e.g. class) so every replicate keeps every class. Replicates where the metric is undefined
    (e.g. one class only) are skipped and counted.
  * Percentile intervals. Laptop only (numpy + scikit-learn).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata


def _labels(values: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.asarray(values)
    if out.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {out.shape}")
    out = out.astype(int)
    if out.size and (out.min() < 0 or out.max() >= n_classes):
        raise ValueError(f"labels must be class indices in [0, {n_classes})")
    return out


def _safe(metric: Callable[..., float], arrays: Sequence[np.ndarray]) -> float:
    """metric(*arrays) as float for one bootstrap replicate; NaN when the metric is undefined on it (returns NaN/inf
    or raises ValueError, e.g. a resample with a single class). bootstrap_ci calls the metric once on the full data
    first without this guard, so input misuse raises instead of turning into NaN."""
    try:
        value = float(metric(*arrays))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _percentiles(reps: Sequence[float], ci: float) -> tuple[float, float]:
    if not reps:
        return math.nan, math.nan
    tail = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(np.asarray(reps, dtype=float), [tail, 1.0 - tail])
    return float(lo), float(hi)


def _resampler(n: int, groups: np.ndarray | None, strata: np.ndarray | None) -> Callable[[np.random.Generator], np.ndarray]:
    """Return draw(rng) -> row indices of one bootstrap replicate.

    No groups: rows with replacement (within each stratum when strata is given). Groups: whole groups with
    replacement, each drawn group contributing all its rows; with strata, each group takes the most common stratum
    of its rows (ties -> the smallest) and groups are resampled within their stratum, keeping per-stratum group counts.
    """
    strata_arr = None if strata is None else np.asarray(strata)
    if strata_arr is not None and len(strata_arr) != n:
        raise ValueError(f"strata must have {n} entries, got {len(strata_arr)}")
    if groups is None:
        units = [np.array([i]) for i in range(n)]
        unit_stratum = None if strata_arr is None else strata_arr
    else:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != n:
            raise ValueError(f"groups must have {n} entries, got {len(groups_arr)}")
        keys, inverse = np.unique(groups_arr.astype(str), return_inverse=True)
        units = [np.flatnonzero(inverse == g) for g in range(len(keys))]
        unit_stratum = None
        if strata_arr is not None:
            unit_stratum = []
            for rows in units:
                vals, counts = np.unique(strata_arr[rows].astype(str), return_counts=True)
                unit_stratum.append(vals[np.argmax(counts)])
            unit_stratum = np.asarray(unit_stratum)
    if unit_stratum is None:
        pools = [np.arange(len(units))]
    else:
        labels = np.asarray(unit_stratum).astype(str)
        pools = [np.flatnonzero(labels == v) for v in np.unique(labels)]

    def draw(rng: np.random.Generator) -> np.ndarray:
        chosen = np.concatenate([rng.choice(pool, size=pool.size, replace=True) for pool in pools])
        return np.concatenate([units[u] for u in chosen])

    return draw


def bootstrap_ci(
    metric: Callable[..., float],
    arrays: Sequence[np.ndarray],
    groups: np.ndarray | None,
    strata: np.ndarray | None,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    """Percentile bootstrap CI of metric(*arrays) with row-aligned arrays.

    Returns {"estimate": metric on the full data, "lo", "hi", "ci", "n_boot", "n_valid", "n", "n_groups"}.
    """
    arrays = [np.asarray(a) for a in arrays]
    if not arrays:
        raise ValueError("bootstrap_ci needs at least one array")
    n = len(arrays[0])
    if n == 0 or any(len(a) != n for a in arrays):
        raise ValueError(f"bootstrap arrays must be non-empty and row-aligned, got lengths {[len(a) for a in arrays]}")
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if not 0.0 < float(ci) < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    estimate = float(metric(*arrays))                      # misuse (bad labels, shapes) raises here, before resampling
    estimate = estimate if math.isfinite(estimate) else math.nan
    draw = _resampler(n, groups, strata)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(int(n_boot)):
        idx = draw(rng)                                     # one draw shared by every array (row alignment)
        value = _safe(metric, [a[idx] for a in arrays])
        if not math.isnan(value):
            reps.append(value)
    lo, hi = _percentiles(reps, ci)
    return {"estimate": estimate, "lo": lo, "hi": hi, "ci": float(ci), "n_boot": int(n_boot),
            "n_valid": len(reps), "n_skipped": int(n_boot) - len(reps), "n": int(n),
            "n_groups": int(n if groups is None else len(np.unique(np.asarray(groups))))}


def paired_difference_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray | None,
    strata: np.ndarray | None,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    """CI of metric(y_true, score_a) - metric(y_true, score_b), resampling the same rows for both (paired).

    Returns {"estimate", "lo", "hi", "ci", "n_boot", "n_valid", "a", "b"}.
    """
    def diff(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float(metric(y, a)) - float(metric(y, b))

    out = bootstrap_ci(diff, [np.asarray(y_true), np.asarray(score_a), np.asarray(score_b)], groups, strata,
                       n_boot, ci, seed)
    out["a"] = float(metric(np.asarray(y_true), np.asarray(score_a)))
    out["b"] = float(metric(np.asarray(y_true), np.asarray(score_b)))
    return out


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC (ties count half). NaN when y_true has a single class."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError(f"y_true and scores must be 1-D of equal length, got {y.shape} and {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    y = y.astype(int) if y.dtype == bool else y
    values = set(np.unique(y).tolist())
    if not values <= {0, 1}:
        raise ValueError(f"auroc needs binary 0/1 labels, got {sorted(values)}")
    n_pos = int(np.sum(y == 1))
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(s)                                     # average ranks: ties count half (Mann-Whitney U)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_ovr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> list[float]:
    """One-vs-rest AUROC per class (NaN for a class absent or always present)."""
    y, probs = _labels(y_true, n_classes), np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape != (len(y), n_classes):
        raise ValueError(f"probs must have shape ({len(y)}, {n_classes}), got {probs.shape}")
    return [auroc((y == k).astype(int), probs[:, k]) for k in range(n_classes)]


def macro_auroc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of the defined one-vs-rest AUROCs."""
    defined = [v for v in auroc_ovr(y_true, probs, n_classes) if not math.isnan(v)]
    return float(np.mean(defined)) if defined else math.nan


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over n_classes (a class with no support and no predictions contributes 0)."""
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)                # 2TP + FP + FN = predicted + support
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall over the classes present in y_true."""
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    support = cm.sum(axis=1)
    present = support > 0
    return float(np.mean(np.diag(cm)[present] / support[present]))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts (n_classes, n_classes): rows = true class, columns = predicted class."""
    y, p = _labels(y_true, n_classes), _labels(y_pred, n_classes)
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_pred must have the same length, got {y.shape} and {p.shape}")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y, p), 1)
    return cm
