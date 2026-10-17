"""Worklist tier assignment: the one function plan §10 requires, shared by step 08, backend and queue sim.

Rules (plan §1; BUILD_PLAN §A5 "tier safety", §A21 aging guard):

    tier 1  lane == "surgical" (simulated frozen section; timer running)
    tier 2  argmax class == aggressive class  OR  aggressive class in the conformal set.
            Safety rule: this holds even when the conformal set is empty (an empty set only adds
            reason "empty_set"). Ties for the top probability resolve toward the aggressive class.
    tier 3  conformal set size != 1 (empty or > 1), OR quality.reliability_flag, OR insufficient
            tissue (quality.qc_tiles_kept < cfg["qc"]["min_tiles_kept"])
    tier 4  everything else (unflagged)

    Aging guard (optional, cfg["aging"]["enabled"], needs a clock `now`): a tier-3 or tier-4 case whose
    elapsed time is >= promote_at_fraction x turnaround_target_min[its tier] moves up exactly one tier
    and gains reason "aging_promoted". Tiers 1 and 2 are never promoted; nothing is promoted above 2.

Reasons are every applicable flag (not only the one that decided the tier), in REASON_ORDER, which is
the contract's `flag.reasons` enum order (tests/test_contract.py keeps the two identical).

Sort order within the worklist: tier ascending, risk_score descending, arrival ascending, then case_id
(a deterministic final tie-break).

Clock convention: `now` and `case["arrived_offset_s"]` are seconds on the same clock (the simulator
clock in the backend, the simulated day in the queue sim). Turnaround targets are minutes.

Cases may be plain dicts in the Case JSON shape or pydantic models from core.contract. This module
imports neither pydantic nor numpy, so the queue simulation can call it in tight loops.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AGGRESSIVE_TOP = "aggressive_top"
AGGRESSIVE_IN_CONFORMAL_SET = "aggressive_in_conformal_set"
SET_SIZE_GT_1 = "set_size_gt_1"
EMPTY_SET = "empty_set"
LOW_RELIABILITY = "low_reliability"
INSUFFICIENT_TISSUE = "insufficient_tissue"
AGING_PROMOTED = "aging_promoted"
SURGICAL_FROZEN_SECTION = "surgical_frozen_section"

REASON_ORDER: tuple[str, ...] = (
    AGGRESSIVE_TOP,
    AGGRESSIVE_IN_CONFORMAL_SET,
    SET_SIZE_GT_1,
    EMPTY_SET,
    LOW_RELIABILITY,
    INSUFFICIENT_TISSUE,
    AGING_PROMOTED,
    SURGICAL_FROZEN_SECTION,
)

# Reasons raised by the model/quality read itself (what "flagged" means); aging is about waiting time.
MODEL_FLAG_REASONS = frozenset(
    {AGGRESSIVE_TOP, AGGRESSIVE_IN_CONFORMAL_SET, SET_SIZE_GT_1, EMPTY_SET, LOW_RELIABILITY, INSUFFICIENT_TISSUE}
)

LANES = ("surgical", "biopsy")
AGING_CAP_TIER = 2  # §A21: aging never promotes a case above tier 2 (tier 1 is reserved for frozen sections)


def _as_mapping(case: Any) -> Mapping[str, Any]:
    if isinstance(case, Mapping):
        return case
    dump = getattr(case, "model_dump", None)
    if callable(dump):
        return dump()
    raise TypeError(f"case must be a mapping or a pydantic model, got {type(case).__name__}")


def _read_flags(case: Mapping[str, Any], cfg: Mapping[str, Any]) -> set[str]:
    lane = case["lane"]
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")

    pred = case["prediction"]
    classes = list(pred["classes"])
    probs = [float(p) for p in pred["probs"]]
    if len(probs) != len(classes):
        raise ValueError(f"prediction.probs has {len(probs)} values for {len(classes)} classes")
    aggressive = cfg["aggressive_class"]
    if aggressive not in classes:
        raise ValueError(f"aggressive class {aggressive!r} is not among prediction.classes {classes}")
    conformal_set = list(pred["conformal_set"])
    unknown = [c for c in conformal_set if c not in classes]
    if unknown:
        raise ValueError(f"conformal_set contains unknown classes {unknown}; classes are {classes}")

    flags: set[str] = set()
    if lane == "surgical":
        flags.add(SURGICAL_FROZEN_SECTION)
    if probs[classes.index(aggressive)] >= max(probs):
        flags.add(AGGRESSIVE_TOP)
    if aggressive in conformal_set:
        flags.add(AGGRESSIVE_IN_CONFORMAL_SET)
    if len(conformal_set) > 1:
        flags.add(SET_SIZE_GT_1)
    if len(conformal_set) == 0:
        flags.add(EMPTY_SET)

    quality = case["quality"]
    if quality["reliability_flag"]:
        flags.add(LOW_RELIABILITY)
    if int(quality["qc_tiles_kept"]) < int(cfg["qc"]["min_tiles_kept"]):
        flags.add(INSUFFICIENT_TISSUE)
    return flags


def _base_tier(flags: set[str]) -> int:
    if SURGICAL_FROZEN_SECTION in flags:
        return 1
    if flags & {AGGRESSIVE_TOP, AGGRESSIVE_IN_CONFORMAL_SET}:
        return 2
    if flags & {SET_SIZE_GT_1, EMPTY_SET, LOW_RELIABILITY, INSUFFICIENT_TISSUE}:
        return 3
    return 4


def _target_minutes(targets: Mapping[Any, Any], tier: int) -> float:
    # YAML gives int keys; a JSON round trip (e.g. /api/config) gives str keys.
    for key in (tier, str(tier)):
        if key in targets:
            return float(targets[key])
    raise ValueError(f"aging.turnaround_target_min has no target for tier {tier}")


def _aging_promotes(tier: int, case: Mapping[str, Any], cfg: Mapping[str, Any], now: float | None) -> bool:
    aging = cfg.get("aging") or {}
    if now is None or not aging.get("enabled", False) or tier <= AGING_CAP_TIER:
        return False
    elapsed_s = float(now) - float(case["arrived_offset_s"])
    threshold_s = float(aging["promote_at_fraction"]) * _target_minutes(aging["turnaround_target_min"], tier) * 60.0
    return elapsed_s >= threshold_s


def assign_tier(case: Any, cfg: Mapping[str, Any], now: float | None = None) -> tuple[int, list[str]]:
    """Return (tier, reasons) for one case.

    Args:
        case: Case JSON dict (plan §7 shape) or a core.contract model. Reads lane, arrived_offset_s,
            prediction.{classes, probs, conformal_set}, quality.{reliability_flag, qc_tiles_kept}.
        cfg: settings mapping (config/settings.yaml). Reads aggressive_class, qc.min_tiles_kept and the
            aging block {enabled, promote_at_fraction, turnaround_target_min{tier: minutes}}.
        now: current time in seconds on the clock of arrived_offset_s; None disables the aging guard.

    Returns:
        tier in 1..4 and the applicable reasons in REASON_ORDER.

    Raises:
        ValueError: unknown lane, probs/classes length mismatch, aggressive class missing from the
            prediction classes, or conformal-set classes outside the prediction classes.
    """
    case = _as_mapping(case)
    flags = _read_flags(case, cfg)
    tier = _base_tier(flags)
    if _aging_promotes(tier, case, cfg, now):
        tier -= 1
        flags.add(AGING_PROMOTED)
    return tier, [reason for reason in REASON_ORDER if reason in flags]


def sort_key(case: Any, cfg: Mapping[str, Any], now: float | None = None) -> tuple[int, float, float, str]:
    """Worklist sort key: (tier asc, risk_score desc, arrived_offset_s asc, case_id asc).

    Use as ``sorted(cases, key=lambda c: sort_key(c, cfg, now))``; pass the same `now` as for
    assign_tier so aging promotions are reflected.
    """
    case = _as_mapping(case)
    tier, _ = assign_tier(case, cfg, now)
    return tier, -float(case["prediction"]["risk_score"]), float(case["arrived_offset_s"]), str(case["case_id"])


def flag_for(case: Any, cfg: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    """The contract's `flag` block: {"flagged", "tier", "reasons"}.

    flagged is True only for biopsy cases with a model/quality flag (base tier 2 or 3: likely aggressive
    or low confidence) -- the cases the agent drafts orders for. Surgical cases are never "flagged"
    (they are tier 1 by lane), and an aging promotion raises the tier without setting flagged.
    Step 08 calls this with now=None (static flag); the backend may pass its clock.
    """
    case = _as_mapping(case)
    tier, reasons = assign_tier(case, cfg, now)
    flagged = case["lane"] == "biopsy" and bool(MODEL_FLAG_REASONS.intersection(reasons))
    return {"flagged": flagged, "tier": tier, "reasons": reasons}
