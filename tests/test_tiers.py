"""Worklist tier assignment (plan §1; BUILD_PLAN §A5 tier safety, §A21 aging guard).

Expected tiers and reasons are written out by hand from the rules:
  surgical -> 1; aggressive top OR aggressive in conformal set -> 2 (even when the set is empty);
  set size != 1, reliability flag or insufficient tissue -> 3; else 4.
  Aging: >= promote_at_fraction x the tier's turnaround target -> one tier up, never above tier 2.
Reasons are every applicable flag, in the contract's enum order.
"""

import pytest

from core.tiers import assign_tier, flag_for, sort_key

CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
MIN = 60  # seconds per minute, for readable elapsed times


def make_cfg(*, aging_enabled=True, fraction=0.75, targets=None, min_tiles=3):
    return {
        "classes": CLASSES,
        "aggressive_class": "Invasive",
        "qc": {"min_tiles_kept": min_tiles},
        "aging": {
            "enabled": aging_enabled,
            "promote_at_fraction": fraction,
            "turnaround_target_min": targets or {1: 20, 2: 1440, 3: 1440, 4: 2880},
        },
    }


def make_case(
    *,
    case_id="c1",
    lane="biopsy",
    probs=(0.10, 0.85, 0.03, 0.02),
    conformal_set=("Benign",),
    reliability_flag=False,
    qc_tiles_kept=12,
    arrived_offset_s=0.0,
    risk=None,
):
    return {
        "case_id": case_id,
        "lane": lane,
        "arrived_offset_s": arrived_offset_s,
        "quality": {
            "qc_tiles_kept": qc_tiles_kept,
            "ood_percentile": 0.5,
            "reliability": 0.9,
            "reliability_flag": reliability_flag,
        },
        "prediction": {
            "model": "concept_bottleneck",
            "classes": CLASSES,
            "probs": list(probs),
            "risk_score": probs[3] if risk is None else risk,
            "conformal_alpha": 0.1,
            "conformal_set": list(conformal_set),
        },
    }


# Probability vectors are in CLASSES order: Normal, Benign, InSitu, Invasive.
TIER_TABLE = [
    (
        "surgical frozen section is always tier 1",
        dict(lane="surgical", probs=(0.70, 0.20, 0.05, 0.05), conformal_set=("Normal",)),
        1,
        ["surgical_frozen_section"],
    ),
    (
        "surgical keeps tier 1 and still reports low reliability",
        dict(lane="surgical", probs=(0.70, 0.20, 0.05, 0.05), conformal_set=("Normal",), reliability_flag=True),
        1,
        ["low_reliability", "surgical_frozen_section"],
    ),
    (
        "surgical with an invasive read is still tier 1",
        dict(lane="surgical", probs=(0.02, 0.03, 0.15, 0.80), conformal_set=("Invasive",)),
        1,
        ["aggressive_top", "aggressive_in_conformal_set", "surgical_frozen_section"],
    ),
    (
        "invasive top with a singleton set",
        dict(probs=(0.02, 0.03, 0.15, 0.80), conformal_set=("Invasive",)),
        2,
        ["aggressive_top", "aggressive_in_conformal_set"],
    ),
    (
        "safety rule: invasive in the set but not the top prediction",
        dict(probs=(0.02, 0.08, 0.60, 0.30), conformal_set=("InSitu", "Invasive")),
        2,
        ["aggressive_in_conformal_set", "set_size_gt_1"],
    ),
    (
        "empty set with argmax invasive stays tier 2",
        dict(probs=(0.05, 0.05, 0.35, 0.55), conformal_set=()),
        2,
        ["aggressive_top", "empty_set"],
    ),
    (
        "empty set without an invasive top is low confidence",
        dict(probs=(0.30, 0.40, 0.20, 0.10), conformal_set=()),
        3,
        ["empty_set"],
    ),
    (
        "benign versus in situ ambiguity is low confidence",
        dict(probs=(0.05, 0.50, 0.40, 0.05), conformal_set=("Benign", "InSitu")),
        3,
        ["set_size_gt_1"],
    ),
    (
        "reliability flag on a confident normal read",
        dict(probs=(0.85, 0.10, 0.03, 0.02), conformal_set=("Normal",), reliability_flag=True),
        3,
        ["low_reliability"],
    ),
    (
        "fewer QC tiles than the minimum is insufficient tissue",
        dict(probs=(0.85, 0.10, 0.03, 0.02), conformal_set=("Normal",), qc_tiles_kept=2),
        3,
        ["insufficient_tissue"],
    ),
    (
        "exactly the minimum QC tiles is sufficient",
        dict(probs=(0.85, 0.10, 0.03, 0.02), conformal_set=("Normal",), qc_tiles_kept=3),
        4,
        [],
    ),
    (
        "confident benign read is unflagged",
        dict(probs=(0.10, 0.85, 0.03, 0.02), conformal_set=("Benign",)),
        4,
        [],
    ),
    (
        "a tie at the top resolves toward the aggressive class",
        dict(probs=(0.0, 0.0, 0.5, 0.5), conformal_set=("InSitu",)),
        2,
        ["aggressive_top"],
    ),
]


@pytest.mark.parametrize("name, case_kwargs, want_tier, want_reasons", TIER_TABLE, ids=[row[0] for row in TIER_TABLE])
def test_assign_tier_table(name, case_kwargs, want_tier, want_reasons):
    assert assign_tier(make_case(**case_kwargs), make_cfg()) == (want_tier, want_reasons)


# ---- aging guard (BUILD_PLAN §A21) -------------------------------------------------------------
# Default targets: tier 3 = 1440 min, tier 4 = 2880 min; promotion at 0.75 -> 1080 min and 2160 min.
ROUTINE = dict(probs=(0.10, 0.85, 0.03, 0.02), conformal_set=("Benign",))  # base tier 4
AMBIGUOUS = dict(probs=(0.05, 0.50, 0.40, 0.05), conformal_set=("Benign", "InSitu"))  # base tier 3
INVASIVE = dict(probs=(0.02, 0.03, 0.15, 0.80), conformal_set=("Invasive",))  # base tier 2


def test_aging_promotes_routine_case_at_the_threshold():
    case = make_case(**ROUTINE)
    assert assign_tier(case, make_cfg(), now=2160 * MIN) == (3, ["aging_promoted"])


def test_aging_does_not_promote_just_below_the_threshold():
    case = make_case(**ROUTINE)
    assert assign_tier(case, make_cfg(), now=2160 * MIN - 1) == (4, [])


def test_aging_promotes_low_confidence_case_to_tier_2():
    case = make_case(**AMBIGUOUS)
    assert assign_tier(case, make_cfg(), now=1080 * MIN) == (2, ["set_size_gt_1", "aging_promoted"])


def test_aging_never_promotes_above_tier_2():
    case = make_case(**INVASIVE)
    assert assign_tier(case, make_cfg(), now=100_000 * MIN) == (2, ["aggressive_top", "aggressive_in_conformal_set"])


def test_aging_never_touches_surgical_cases():
    case = make_case(lane="surgical", **ROUTINE)
    assert assign_tier(case, make_cfg(), now=100_000 * MIN) == (1, ["surgical_frozen_section"])


def test_aging_promotes_a_single_tier_only():
    # 3000 min is past both the tier-4 (2160) and tier-3 (1080) thresholds: still one step, 4 -> 3.
    case = make_case(**ROUTINE)
    assert assign_tier(case, make_cfg(), now=3000 * MIN) == (3, ["aging_promoted"])


def test_aging_measures_elapsed_time_from_arrival():
    case = make_case(arrived_offset_s=1000.0, **ROUTINE)
    assert assign_tier(case, make_cfg(), now=1000.0 + 2160 * MIN) == (3, ["aging_promoted"])
    assert assign_tier(case, make_cfg(), now=2160 * MIN) == (4, [])


def test_aging_uses_the_configured_fraction():
    case = make_case(**ROUTINE)
    assert assign_tier(case, make_cfg(fraction=0.5), now=1440 * MIN) == (3, ["aging_promoted"])
    assert assign_tier(case, make_cfg(fraction=0.5), now=1440 * MIN - 1) == (4, [])


def test_aging_disabled_never_promotes():
    case = make_case(**ROUTINE)
    assert assign_tier(case, make_cfg(aging_enabled=False), now=100_000 * MIN) == (4, [])


def test_no_clock_means_no_aging():
    case = make_case(**ROUTINE)
    assert assign_tier(case, make_cfg(), now=None) == (4, [])


def test_case_not_yet_arrived_is_not_promoted():
    case = make_case(arrived_offset_s=500_000.0, **ROUTINE)
    assert assign_tier(case, make_cfg(), now=1000.0) == (4, [])


# ---- sorting: tier ascending, risk descending, arrival ascending -------------------------------
def _ids(cases, cfg, now=None):
    return [c["case_id"] for c in sorted(cases, key=lambda c: sort_key(c, cfg, now))]


def test_sort_puts_lower_tiers_first_regardless_of_risk():
    tier4_high_risk = make_case(case_id="t4", probs=(0.10, 0.85, 0.03, 0.02), conformal_set=("Benign",), risk=0.99)
    tier2_low_risk = make_case(case_id="t2", probs=(0.02, 0.08, 0.60, 0.30), conformal_set=("InSitu", "Invasive"))
    surgical = make_case(case_id="s1", lane="surgical", probs=(0.70, 0.20, 0.05, 0.05), conformal_set=("Normal",))
    assert _ids([tier4_high_risk, tier2_low_risk, surgical], make_cfg()) == ["s1", "t2", "t4"]


def test_sort_within_tier_by_risk_descending_then_arrival():
    common = dict(probs=(0.02, 0.03, 0.15, 0.80), conformal_set=("Invasive",))
    a = make_case(case_id="a", risk=0.70, arrived_offset_s=10.0, **common)
    b = make_case(case_id="b", risk=0.90, arrived_offset_s=50.0, **common)
    c = make_case(case_id="c", risk=0.70, arrived_offset_s=5.0, **common)
    assert _ids([a, b, c], make_cfg()) == ["b", "c", "a"]


def test_sort_full_ties_break_on_case_id():
    common = dict(probs=(0.10, 0.85, 0.03, 0.02), conformal_set=("Benign",), arrived_offset_s=7.0)
    assert _ids([make_case(case_id="z", **common), make_case(case_id="m", **common)], make_cfg()) == ["m", "z"]


def test_sort_reflects_aging_promotion_when_a_clock_is_given():
    cfg = make_cfg()
    aged = make_case(case_id="aged", probs=(0.05, 0.60, 0.10, 0.25), conformal_set=("Benign",))  # tier 4, risk 0.25
    amb = make_case(case_id="amb", arrived_offset_s=2100 * MIN, **AMBIGUOUS)  # tier 3, risk 0.05
    inv = make_case(case_id="inv", arrived_offset_s=2150 * MIN, **INVASIVE)  # tier 2, risk 0.80
    cases = [aged, amb, inv]
    assert _ids(cases, cfg) == ["inv", "amb", "aged"]
    # 2200 min after arrival the routine case is promoted 4 -> 3 (amb has waited only 100 min) and then
    # outranks amb on risk inside tier 3; it never passes the tier-2 case.
    assert _ids(cases, cfg, now=2200 * MIN) == ["inv", "aged", "amb"]


# ---- flag block -----------------------------------------------------------------------------------
def test_flag_for_flags_biopsy_tiers_2_and_3():
    cfg = make_cfg()
    assert flag_for(make_case(**INVASIVE), cfg) == {
        "flagged": True, "tier": 2, "reasons": ["aggressive_top", "aggressive_in_conformal_set"],
    }
    assert flag_for(make_case(**AMBIGUOUS), cfg) == {"flagged": True, "tier": 3, "reasons": ["set_size_gt_1"]}
    assert flag_for(make_case(**ROUTINE), cfg) == {"flagged": False, "tier": 4, "reasons": []}


def test_flag_for_does_not_flag_surgical_cases():
    assert flag_for(make_case(lane="surgical", **INVASIVE), make_cfg()) == {
        "flagged": False,
        "tier": 1,
        "reasons": ["aggressive_top", "aggressive_in_conformal_set", "surgical_frozen_section"],
    }


def test_flag_for_aging_raises_tier_but_not_the_model_flag():
    assert flag_for(make_case(**ROUTINE), make_cfg(), now=2160 * MIN) == {
        "flagged": False, "tier": 3, "reasons": ["aging_promoted"],
    }


# ---- malformed input ------------------------------------------------------------------------------
def test_probs_length_must_match_classes():
    case = make_case()
    case["prediction"]["probs"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="probs"):
        assign_tier(case, make_cfg())


def test_conformal_set_must_use_known_classes():
    case = make_case(conformal_set=("Benign", "Carcinoma"))
    with pytest.raises(ValueError, match="Carcinoma"):
        assign_tier(case, make_cfg())


def test_aggressive_class_must_be_among_prediction_classes():
    case = make_case()
    case["prediction"]["classes"] = ["Normal", "Benign", "InSitu", "Malignant"]
    with pytest.raises(ValueError, match="Invasive"):
        assign_tier(case, make_cfg())


def test_unknown_lane_is_rejected():
    with pytest.raises(ValueError, match="lane"):
        assign_tier(make_case(lane="frozen"), make_cfg())
