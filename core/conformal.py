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
Laptop only (numpy).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """LAC nonconformity score of the true class: 1 - p_hat[i, labels[i]]. Returns shape (n,)."""
    raise NotImplementedError("Task 4 implements core.conformal.lac_scores")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal threshold: the k-th smallest score, k = ceil((n+1)(1-alpha)); +inf if k > n."""
    raise NotImplementedError("Task 4 implements core.conformal.conformal_quantile")


def lac_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """LAC sets: mask[i, y] = (1 - probs[i, y] <= qhat). Returns bool (n, C); rows may be empty."""
    raise NotImplementedError("Task 4 implements core.conformal.lac_sets")


def fit_lac(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    """Fit marginal LAC on the calibration split: conformal_quantile(lac_scores(cal_probs, cal_labels), alpha)."""
    raise NotImplementedError("Task 4 implements core.conformal.fit_lac")


def aps_scores(probs: np.ndarray, labels: np.ndarray, seed: int | None, randomized: bool) -> np.ndarray:
    """APS score: probability mass of classes ranked above the true class plus (U x) its own mass when randomized.

    Sorting is by descending probability with a stable tie-break on class index; U ~ Uniform(0, 1) from
    np.random.default_rng(seed). randomized=False uses the full own mass (conservative). Returns shape (n,).
    """
    raise NotImplementedError("Task 4 implements core.conformal.aps_scores")


def aps_sets(probs: np.ndarray, qhat: float, seed: int | None, randomized: bool) -> np.ndarray:
    """APS sets: include classes in descending-probability order while the (randomized) cumulative mass <= qhat."""
    raise NotImplementedError("Task 4 implements core.conformal.aps_sets")


def fit_aps(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float, seed: int | None, randomized: bool) -> float:
    """Fit APS on the calibration split: conformal_quantile(aps_scores(...), alpha)."""
    raise NotImplementedError("Task 4 implements core.conformal.fit_aps")


def mondrian_quantiles(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> np.ndarray:
    """Class-conditional (Mondrian) LAC thresholds: one q-hat per class from that class's calibration cases.

    Returns float array (C,). A class with too few calibration cases gets +inf (always included).
    """
    raise NotImplementedError("Task 4 implements core.conformal.mondrian_quantiles")


def mondrian_sets(probs: np.ndarray, qhats: np.ndarray) -> np.ndarray:
    """Mondrian LAC sets: mask[i, y] = (1 - probs[i, y] <= qhats[y]). Returns bool (n, C)."""
    raise NotImplementedError("Task 4 implements core.conformal.mondrian_sets")


def set_labels(mask: np.ndarray, classes: Sequence[str]) -> list[str]:
    """Class names of one set row (bool (C,)), in classes order. [] for an empty set."""
    raise NotImplementedError("Task 4 implements core.conformal.set_labels")


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Empirical coverage: share of rows whose set contains the true label."""
    raise NotImplementedError("Task 4 implements core.conformal.coverage")


def mean_set_size(sets: np.ndarray) -> float:
    """Mean number of classes per set (empty sets count as 0)."""
    raise NotImplementedError("Task 4 implements core.conformal.mean_set_size")
