"""Probability calibration: temperature scaling and calibration metrics (plan §6.6, BUILD_PLAN §A3/§A5).

FROZEN INTERFACE (Task 1 stub; Task 4 implements; tests/test_contract.py pins every signature).
Callers: pipeline/05_risk_model.py (temperature fit on out-of-fold TRAIN logits, §A3), eval/metrics.py.

Conventions: logits/probs are float arrays (n, C) in settings.classes order; labels are int (n,).
bounds come from settings.calibration.temperature_bounds; n_bins from settings.calibration.ece_bins
(equal-mass bins, settings.calibration.ece_binning). Temperature scaling never changes the argmax.
Laptop only (numpy + scipy).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax(logits / temperature) along the last axis."""
    raise NotImplementedError("Task 4 implements core.calibration.softmax")


def fit_temperature(logits: np.ndarray, labels: np.ndarray, bounds: Sequence[float]) -> float:
    """Temperature T in [bounds[0], bounds[1]] minimizing the NLL of softmax(logits / T) (bounded scalar search)."""
    raise NotImplementedError("Task 4 implements core.calibration.fit_temperature")


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (probabilities clipped away from 0)."""
    raise NotImplementedError("Task 4 implements core.calibration.nll")


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean over rows of sum_k (p_k - 1[y = k])^2."""
    raise NotImplementedError("Task 4 implements core.calibration.brier")


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Top-label expected calibration error with n_bins equal-mass bins (biased at small n; say so in reports)."""
    raise NotImplementedError("Task 4 implements core.calibration.ece")


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Reliability-diagram data for plotting/JSON: equal-mass bins of top-label confidence.

    Returns {"bins": [{"lo", "hi", "count", "confidence", "accuracy"}...], "ece": float, "n": int}.
    """
    raise NotImplementedError("Task 4 implements core.calibration.reliability_diagram")
