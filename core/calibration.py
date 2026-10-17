"""Probability calibration: temperature scaling and calibration metrics (plan §6.6, BUILD_PLAN §A3/§A5).

FROZEN INTERFACE (Task 1 stub; Task 4 implements; tests/test_contract.py pins every signature).
Callers: pipeline/05_risk_model.py (temperature fit on out-of-fold TRAIN logits, §A3), eval/metrics.py.

Conventions: logits/probs are float arrays (n, C) in settings.classes order; labels are int (n,).
bounds come from settings.calibration.temperature_bounds; n_bins from settings.calibration.ece_bins
(equal-mass bins, settings.calibration.ece_binning). Temperature scaling never changes the argmax.
Laptop only (numpy + scipy).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import optimize

_EPS = 1e-12   # probability floor inside log() so a confidently wrong row costs a large but finite NLL


def _check(probs: np.ndarray, labels: np.ndarray, what: str = "probs") -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError(f"{what} must be a non-empty 2-D array (n, C>=2), got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{what} must be finite")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError(f"labels must be 1-D with {probs.shape[0]} entries, got shape {labels.shape}")
    labels = labels.astype(int)
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError(f"labels must be class indices in [0, {probs.shape[1]})")
    return probs, labels


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax(logits / temperature) along the last axis."""
    z = np.asarray(logits, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    if not (np.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be a positive finite number, got {temperature!r}")
    z = z / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, bounds: Sequence[float]) -> float:
    """Temperature T in [bounds[0], bounds[1]] minimizing the NLL of softmax(logits / T) (bounded scalar search)."""
    z, labels = _check(logits, labels, "logits")
    if len(bounds) != 2 or not (0 < float(bounds[0]) < float(bounds[1])):
        raise ValueError(f"temperature bounds must be [lo, hi] with 0 < lo < hi, got {list(bounds)}")
    lo, hi = math.log(float(bounds[0])), math.log(float(bounds[1]))

    def objective(log_t: float) -> float:
        return nll(softmax(z, math.exp(log_t)), labels)

    # NLL is unimodal in log T; also evaluate the bounds so a boundary optimum is returned exactly.
    res = optimize.minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    candidates = [(objective(lo), lo), (objective(hi), hi)]
    if res.success:
        candidates.append((float(res.fun), float(res.x)))
    return float(math.exp(min(candidates)[1]))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (probabilities clipped away from 0)."""
    probs, labels = _check(probs, labels)
    p = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p, _EPS, 1.0))))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean over rows of sum_k (p_k - 1[y = k])^2."""
    probs, labels = _check(probs, labels)
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Top-label expected calibration error with n_bins equal-mass bins (biased at small n; say so in reports)."""
    return float(reliability_diagram(probs, labels, n_bins)["ece"])


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Reliability-diagram data for plotting/JSON: equal-mass bins of top-label confidence.

    Returns {"bins": [{"lo", "hi", "count", "confidence", "accuracy"}...], "ece": float, "n": int}.
    """
    probs, labels = _check(probs, labels)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    order = np.argsort(conf, kind="stable")
    n = len(labels)
    bins = []
    total = 0.0
    for idx in np.array_split(order, int(n_bins)):       # equal-mass: bin sizes differ by at most one
        if idx.size == 0:
            continue
        c, a = float(conf[idx].mean()), float(correct[idx].mean())
        bins.append({"lo": float(conf[idx].min()), "hi": float(conf[idx].max()), "count": int(idx.size),
                     "confidence": c, "accuracy": a})
        total += idx.size * abs(a - c)
    return {"bins": bins, "ece": total / n, "n": int(n)}
