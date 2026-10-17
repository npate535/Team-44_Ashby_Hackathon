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
from functools import lru_cache
from typing import Any

import numpy as np

from core import schemas
from core.config import load_settings

MARGIN_ORDER: tuple[str, ...] = schemas.MARGINS   # contract order (= core.contract.MARGIN_NAMES)
_EDGES = ("top", "bottom", "left", "right")


@lru_cache(maxsize=1)
def _quantile_method() -> str:
    """settings.margin.threshold_quantile_method from the default settings file (read once per process)."""
    return str(load_settings()["margin"]["threshold_quantile_method"])


def _probs(probs: Sequence[float], n: int) -> np.ndarray:
    out = np.asarray(probs, dtype=float).ravel()
    if out.size != n:
        raise ValueError(f"need one probability per segment ({n}), got {out.size}")
    if not np.all(np.isfinite(out)) or out.min(initial=0.0) < 0.0 or out.max(initial=0.0) > 1.0:
        raise ValueError("segment probabilities must be finite and in [0, 1]")
    return out


def strip_segments(width: int, height: int, segment_px: int, names: Mapping[str, str]) -> list[dict[str, Any]]:
    """Border squares of the field, one dict per (margin, square) membership, in margin order then position.

    names maps image edges {"top", "bottom", "left", "right"} to margin names. Raises if width/height are not
    multiples of segment_px.
    """
    if sorted(names) != sorted(_EDGES) or len(set(names.values())) != 4 or set(names.values()) != set(MARGIN_ORDER):
        raise ValueError(f"names must map {list(_EDGES)} one-to-one onto {list(MARGIN_ORDER)}, got {dict(names)}")
    for what, value in (("width", width), ("height", height), ("segment_px", segment_px)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{what} must be a positive integer, got {value!r}")
    if width % segment_px or height % segment_px:
        raise ValueError(f"field {width}x{height} is not a multiple of segment_px={segment_px}")
    if width // segment_px < 2 or height // segment_px < 2:
        raise ValueError(f"field {width}x{height} needs at least 2 segments of {segment_px} px per edge")
    xs, ys = range(0, width, segment_px), range(0, height, segment_px)
    squares = {
        "top": [(x, 0) for x in xs],
        "bottom": [(x, height - segment_px) for x in xs],
        "left": [(0, y) for y in ys],
        "right": [(width - segment_px, y) for y in ys],
    }
    edge_of = {name: edge for edge, name in names.items()}
    return [{"margin": m, "x": int(x), "y": int(y), "size": int(segment_px)}
            for m in MARGIN_ORDER for x, y in squares[edge_of[m]]]


def per_margin_scores(segments: Sequence[Mapping[str, Any]], probs: Sequence[float]) -> dict[str, float]:
    """Per-margin score = max segment probability over that margin's segments (probs aligned with segments)."""
    probs = _probs(probs, len(segments))
    out: dict[str, float] = {}
    for m in MARGIN_ORDER:
        ps = [p for seg, p in zip(segments, probs) if seg["margin"] == m]
        if not ps:
            raise ValueError(f"no segment for margin {m!r}")
        out[m] = float(max(ps))
    return out


def fit_threshold(benign_per_margin: np.ndarray, max_fpr: float) -> float:
    """Threshold = the (1 - max_fpr) quantile (method settings.margin.threshold_quantile_method) of per-margin
    maxima pooled over benign calibration images, so the false 'tumor at margin' rate is controlled."""
    values = np.asarray(benign_per_margin, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("fit_threshold needs at least one benign per-margin maximum")
    if not np.all(np.isfinite(values)):
        raise ValueError("benign per-margin maxima must be finite")
    if not 0.0 < float(max_fpr) < 1.0:
        raise ValueError(f"max_fpr must be in (0, 1), got {max_fpr!r}")
    method = _quantile_method()
    return float(np.quantile(values, 1.0 - float(max_fpr), method=method))


def involved_margins(per_margin: Mapping[str, float], threshold: float) -> list[str]:
    """Margins with score strictly above threshold, in contract margin order (superior, inferior, medial, lateral)."""
    missing = [m for m in MARGIN_ORDER if m not in per_margin]
    if missing:
        raise ValueError(f"per_margin is missing margins {missing}")
    return [m for m in MARGIN_ORDER if float(per_margin[m]) > float(threshold)]


def involved_length_um(
    segments: Sequence[Mapping[str, Any]], probs: Sequence[float], threshold: float, mpp: float
) -> dict[str, float]:
    """Per involved margin: total edge length (µm) of its segments with p > threshold (segment size x mpp each)."""
    probs = _probs(probs, len(segments))
    if not (np.isfinite(mpp) and mpp > 0):
        raise ValueError(f"mpp must be positive, got {mpp!r}")
    involved = involved_margins(per_margin_scores(segments, probs), threshold)
    return {m: float(sum(seg["size"] * mpp for seg, p in zip(segments, probs) if seg["margin"] == m and p > threshold))
            for m in involved}


def suggest_call(per_margin: Mapping[str, float], threshold: float, reliability_flag: bool) -> tuple[str, list[str]]:
    """Suggested call: ("defer", []) if reliability_flag; else ("tumor_at_margin", involved) if any margin is
    involved; else ("clear", []). The pathologist always makes the final call."""
    if reliability_flag:
        return "defer", []
    involved = involved_margins(per_margin, threshold)
    return ("tumor_at_margin", involved) if involved else ("clear", [])
