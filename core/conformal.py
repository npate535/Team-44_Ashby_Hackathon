"""Split-conformal prediction sets: LAC (default), APS and Mondrian (plan §6.6, BUILD_PLAN §A5).

FROZEN INTERFACE (Task 1 stub; Task 4 implements; tests/test_contract.py pins every signature).
Callers: pipeline/05_risk_model.py, pipeline/06_reliability.py, pipeline/08_build_case_json.py, eval/metrics.py.

Conventions:
  * probs: float array (n, C) of calibrated class probabilities, columns in settings.classes order.
  * labels: int array (n,) of class indices into that order.
  * A prediction set is a boolean mask (n, C); set_labels() turns one row into class names.
  * alpha comes from settings.alpha (0.1); seeds come from settings.conformal.seed. No defaults hide config.
  * Finite-sample quantile: k = ceil((n + 1) * (1 - alpha)); q-hat = k-th smallest calibration score;
    k > n => q-hat = +inf (every class is in every set). Empty sets are allowed (reported, never "fixed").
  * Randomized APS draws one U per row from np.random.default_rng(seed) on every call, so U depends on the seed and
    the row position. Fit on calibration with one seed and build test sets ONCE on the whole test matrix with a
    different seed (05_risk_model uses conformal.seed and conformal.seed + 1); never call aps_sets row by row.
Laptop only (numpy).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _check_alpha(alpha: float) -> None:
    if not (isinstance(alpha, (int, float)) and 0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"probs must be a 2-D array (n, C>=2), got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs must be finite")
    return probs


def _check_labels(labels: np.ndarray, n: int, n_classes: int) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.shape[0] != n:
        raise ValueError(f"labels must be 1-D with {n} entries, got shape {raw.shape}")
    if n and (not np.issubdtype(raw.dtype, np.integer)):
        if not np.all(np.isfinite(raw.astype(float))) or not np.all(raw.astype(float) == np.round(raw.astype(float))):
            raise ValueError("labels must be integer class indices")
    out = raw.astype(int)
    if n and (out.min() < 0 or out.max() >= n_classes):
        raise ValueError(f"labels must be class indices in [0, {n_classes})")
    return out


def _check_probs_labels(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = _check_probs(probs)
    if probs.shape[0] == 0:
        raise ValueError("need at least one row")
    return probs, _check_labels(labels, probs.shape[0], probs.shape[1])


def _check_sets_labels(sets: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sets = np.asarray(sets, dtype=bool)
    if sets.ndim != 2 or sets.shape[0] == 0:
        raise ValueError(f"sets must be a non-empty 2-D boolean mask, got shape {sets.shape}")
    return sets, _check_labels(labels, sets.shape[0], sets.shape[1])


def _aps_parts(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per (row, class): mass of the classes ranked strictly above it, and its own mass. Ranking is descending
    probability with a stable tie-break on class index (lower index ranks first)."""
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    above_sorted = np.cumsum(sorted_p, axis=1) - sorted_p
    above = np.empty_like(probs)
    np.put_along_axis(above, order, above_sorted, axis=1)
    return above, probs


def _aps_u(n: int, seed: int | None, randomized: bool) -> np.ndarray:
    """One U ~ Uniform(0, 1) per row from default_rng(seed) when randomized, else ones (full own mass)."""
    if not randomized:
        return np.ones(n)
    return np.random.default_rng(seed).uniform(size=n)


def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """LAC nonconformity score of the true class: 1 - p_hat[i, labels[i]]. Returns shape (n,)."""
    probs, labels = _check_probs_labels(probs, labels)
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal threshold: the k-th smallest score, k = ceil((n+1)(1-alpha)); +inf if k > n."""
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("conformal_quantile needs at least one calibration score")
    if not np.all(np.isfinite(scores)):
        raise ValueError("conformal scores must be finite")
    _check_alpha(alpha)
    n = scores.size
    k = math.ceil((n + 1) * (1.0 - alpha) - 1e-12)   # guard against (n+1)(1-alpha) landing a hair above an integer
    if k > n:
        return math.inf
    return float(np.sort(scores, kind="stable")[k - 1])


def lac_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """LAC sets: mask[i, y] = (1 - probs[i, y] <= qhat). Returns bool (n, C); rows may be empty."""
    probs = _check_probs(probs)
    return (1.0 - probs) <= qhat


def fit_lac(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    """Fit marginal LAC on the calibration split: conformal_quantile(lac_scores(cal_probs, cal_labels), alpha)."""
    return conformal_quantile(lac_scores(cal_probs, cal_labels), alpha)


def aps_scores(probs: np.ndarray, labels: np.ndarray, seed: int | None, randomized: bool) -> np.ndarray:
    """APS score: probability mass of classes ranked above the true class plus (U x) its own mass when randomized.

    Sorting is by descending probability with a stable tie-break on class index; U ~ Uniform(0, 1) from
    np.random.default_rng(seed). randomized=False uses the full own mass (conservative). Returns shape (n,).
    """
    probs, labels = _check_probs_labels(probs, labels)
    above, own = _aps_parts(probs)
    rows = np.arange(len(labels))
    return above[rows, labels] + _aps_u(len(labels), seed, randomized) * own[rows, labels]


def aps_sets(probs: np.ndarray, qhat: float, seed: int | None, randomized: bool) -> np.ndarray:
    """APS sets: include classes in descending-probability order while the (randomized) cumulative mass <= qhat."""
    probs = _check_probs(probs)
    above, own = _aps_parts(probs)
    return (above + _aps_u(len(probs), seed, randomized)[:, None] * own) <= qhat


def fit_aps(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float, seed: int | None, randomized: bool) -> float:
    """Fit APS on the calibration split: conformal_quantile(aps_scores(...), alpha)."""
    return conformal_quantile(aps_scores(cal_probs, cal_labels, seed, randomized), alpha)


def mondrian_quantiles(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> np.ndarray:
    """Class-conditional (Mondrian) LAC thresholds: one q-hat per class from that class's calibration cases.

    Returns float array (C,). A class with too few calibration cases gets +inf (always included).
    """
    cal_probs, cal_labels = _check_probs_labels(cal_probs, cal_labels)
    _check_alpha(alpha)
    scores = lac_scores(cal_probs, cal_labels)
    qhats = np.full(cal_probs.shape[1], math.inf)
    for y in range(cal_probs.shape[1]):
        own = scores[cal_labels == y]
        if own.size:
            qhats[y] = conformal_quantile(own, alpha)    # +inf when k > n_y (too few cases)
    return qhats


def mondrian_sets(probs: np.ndarray, qhats: np.ndarray) -> np.ndarray:
    """Mondrian LAC sets: mask[i, y] = (1 - probs[i, y] <= qhats[y]). Returns bool (n, C)."""
    probs = _check_probs(probs)
    qhats = np.asarray(qhats, dtype=float).ravel()
    if qhats.shape != (probs.shape[1],):
        raise ValueError(f"mondrian_sets needs one q-hat per class ({probs.shape[1]}), got {qhats.shape}")
    return (1.0 - probs) <= qhats[None, :]


def set_labels(mask: np.ndarray, classes: Sequence[str]) -> list[str]:
    """Class names of one set row (bool (C,)), in classes order. [] for an empty set."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != len(classes):
        raise ValueError(f"set row has {mask.size} entries for {len(classes)} classes")
    return [str(c) for c, m in zip(classes, mask) if m]


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Empirical coverage: share of rows whose set contains the true label."""
    sets, labels = _check_sets_labels(sets, labels)
    return float(sets[np.arange(len(labels)), labels].mean())


def mean_set_size(sets: np.ndarray) -> float:
    """Mean number of classes per set (empty sets count as 0)."""
    sets = np.asarray(sets, dtype=bool)
    if sets.ndim != 2 or sets.shape[0] == 0:
        raise ValueError(f"sets must be a non-empty 2-D boolean mask, got shape {sets.shape}")
    return float(sets.sum(axis=1).mean())
