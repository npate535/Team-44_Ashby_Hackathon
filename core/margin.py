"""Simulated surgical margin read: strip geometry, threshold, call (plan §6.9, BUILD_PLAN §A12).

FROZEN INTERFACE (Task 1 stub; Task 4 implements; tests/test_contract.py pins every signature).
Callers: pipeline/05_risk_model.py (threshold on benign calibration images), pipeline/08_build_case_json.py
(contract `margin` block), eval/margin_eval.py.

Geometry (settings.margin): the outermost strip_px band of a width x height field is cut into
segment_px x segment_px squares: 8 per long edge and 6 per short edge for 2048x1536 at 256 px. Corner squares
belong to both adjacent margins, so a 2048x1536 field has 24 distinct squares and 28 (margin, square)
memberships. Image edges map to margin names via settings.margin.names (top=superior, bottom=inferior,
left=medial, right=lateral); this is a 2D simulation (anterior/posterior are not representable).
A segment dict is {"margin", "x", "y", "size"} in source pixels (the contract MarginSegment minus "p").
Laptop only (numpy).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def strip_segments(width: int, height: int, segment_px: int, names: Mapping[str, str]) -> list[dict[str, Any]]:
    """Border squares of the field, one dict per (margin, square) membership, in margin order then position.

    names maps image edges {"top", "bottom", "left", "right"} to margin names. Raises if width/height are not
    multiples of segment_px.
    """
    raise NotImplementedError("Task 4 implements core.margin.strip_segments")


def per_margin_scores(segments: Sequence[Mapping[str, Any]], probs: Sequence[float]) -> dict[str, float]:
    """Per-margin score = max segment probability over that margin's segments (probs aligned with segments)."""
    raise NotImplementedError("Task 4 implements core.margin.per_margin_scores")


def fit_threshold(benign_per_margin: np.ndarray, max_fpr: float) -> float:
    """Threshold = the (1 - max_fpr) quantile (method settings.margin.threshold_quantile_method) of per-margin
    maxima pooled over benign calibration images, so the false 'tumor at margin' rate is controlled."""
    raise NotImplementedError("Task 4 implements core.margin.fit_threshold")


def involved_margins(per_margin: Mapping[str, float], threshold: float) -> list[str]:
    """Margins with score strictly above threshold, in contract margin order (superior, inferior, medial, lateral)."""
    raise NotImplementedError("Task 4 implements core.margin.involved_margins")


def involved_length_um(
    segments: Sequence[Mapping[str, Any]], probs: Sequence[float], threshold: float, mpp: float
) -> dict[str, float]:
    """Per involved margin: total edge length (µm) of its segments with p > threshold (segment size x mpp each)."""
    raise NotImplementedError("Task 4 implements core.margin.involved_length_um")


def suggest_call(per_margin: Mapping[str, float], threshold: float, reliability_flag: bool) -> tuple[str, list[str]]:
    """Suggested call: ("defer", []) if reliability_flag; else ("tumor_at_margin", involved) if any margin is
    involved; else ("clear", []). The pathologist always makes the final call."""
    raise NotImplementedError("Task 4 implements core.margin.suggest_call")
