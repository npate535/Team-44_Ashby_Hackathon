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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


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
    raise NotImplementedError("Task 4 implements core.stats.bootstrap_ci")


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
    raise NotImplementedError("Task 4 implements core.stats.paired_difference_ci")


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC (ties count half). NaN when y_true has a single class."""
    raise NotImplementedError("Task 4 implements core.stats.auroc")


def auroc_ovr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> list[float]:
    """One-vs-rest AUROC per class (NaN for a class absent or always present)."""
    raise NotImplementedError("Task 4 implements core.stats.auroc_ovr")


def macro_auroc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of the defined one-vs-rest AUROCs."""
    raise NotImplementedError("Task 4 implements core.stats.macro_auroc")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over n_classes (a class with no support and no predictions contributes 0)."""
    raise NotImplementedError("Task 4 implements core.stats.macro_f1")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall over the classes present in y_true."""
    raise NotImplementedError("Task 4 implements core.stats.balanced_accuracy")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts (n_classes, n_classes): rows = true class, columns = predicted class."""
    raise NotImplementedError("Task 4 implements core.stats.confusion_matrix")
