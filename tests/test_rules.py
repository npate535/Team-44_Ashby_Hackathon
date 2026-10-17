"""agent/rules.py: each stain rule fires exactly when specified and never otherwise; the lane guard; no invented items."""

from __future__ import annotations

import copy
import itertools
import json

import pytest

from agent import rules as R
from core import config, contract

CFG = config.load_settings()
RULES = R.load_rules(CFG)
CLASSES = list(CFG["classes"])
MIN_KEPT = int(CFG["qc"]["min_tiles_kept"])


def make_case(top="Invasive", conformal_set=("InSitu", "Invasive"), reliability_flag=False, qc=12, lane="biopsy"):
    probs = [0.1] * len(CLASSES)
    probs[CLASSES.index(top)] = 1.0 - 0.1 * (len(CLASSES) - 1)
    return {"lane": lane,
            "prediction": {"classes": CLASSES, "probs": probs, "conformal_set": list(conformal_set)},
            "quality": {"qc_tiles_kept": qc, "reliability_flag": reliability_flag}}


def fired(case):
    return [h.rule.id for h in R.fired_rules(case, CFG, RULES)]


def expected(top, conformal_set, reliability_flag, qc):
    """Independent restatement of the six rules (BUILD_PLAN §A14)."""
    out = []
    if {"InSitu", "Invasive"} <= set(conformal_set):
        out.append("breast_insitu_vs_invasive")
    if top == "Invasive":
        out.append("breast_invasive")
    if top == "InSitu":
        out.append("breast_insitu")
    if {"Benign", "InSitu"} <= set(conformal_set):
        out += ["uncertain_benign_atypia", "uncertain_benign_atypia_ihc"]
    if reliability_flag:
        out += ["low_quality", "low_quality_levels"]
    if qc < MIN_KEPT:
        out.append("insufficient")
    return out


ALL_SETS = [tuple(c for i, c in enumerate(CLASSES) if mask >> i & 1) for mask in range(2 ** len(CLASSES))]


def test_rule_file_is_marked_illustrative_and_has_the_six_rules():
    assert "clinical review" in RULES.status.lower()
    assert [r.id for r in RULES.rules] == ["breast_insitu_vs_invasive", "breast_invasive", "breast_insitu",
                                          "uncertain_benign_atypia", "uncertain_benign_atypia_ihc", "low_quality",
                                          "low_quality_levels", "insufficient"]
    text = R.rules_path(CFG).read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("# ILLUSTRATIVE; REQUIRES CLINICAL REVIEW")
    for citation in ("ASCO/CAP", "J Clin Oncol 2020", "Ki67", "SMMHC"):
        assert citation in text


@pytest.mark.parametrize("top,conformal_set,reliability_flag,qc", [
    (top, s, flag, qc)
    for top, s, flag, qc in itertools.product(CLASSES, ALL_SETS, [False, True], [0, MIN_KEPT - 1, MIN_KEPT, 12])
])
def test_each_rule_fires_exactly_when_specified(top, conformal_set, reliability_flag, qc):
    case = make_case(top, conformal_set, reliability_flag, qc)
    assert fired(case) == expected(top, conformal_set, reliability_flag, qc)


def test_rule_items_match_the_spec():
    items = {r.id: [(o.type, list(o.items)) for o in r.orders] for r in RULES.rules}
    assert items["breast_insitu_vs_invasive"] == [("stain", ["p63", "calponin"])]
    assert items["breast_invasive"] == [("stain", ["ER", "PR", "HER2", "Ki-67"])]
    assert items["breast_insitu"] == [("stain", ["ER"])]
    assert items["uncertain_benign_atypia"] == [("deeper_levels", ["deeper levels"])]
    assert items["uncertain_benign_atypia_ihc"] == [("stain", ["CK5/6", "ER"])]
    assert items["low_quality"] == [("recut", ["recut H&E section"])]
    assert items["low_quality_levels"] == [("deeper_levels", ["deeper levels"])]
    assert all(len(r.orders) == 1 for r in RULES.rules)
    assert items["insufficient"] == [("new_sample", ["new sample"])]


def test_demo_fixture_orders_are_reproduced():
    for name in ("demo_0001", "demo_0002"):
        case = json.loads((config.fixtures_root(CFG) / f"{name}.json").read_text(encoding="utf-8"))
        assert R.draft_orders(case, CFG, RULES) == {**{"blocked": []}, **case["agent"]}


def test_surgical_lane_blocks_stains_visibly():
    agent = R.draft_orders(make_case(lane="surgical"), CFG, RULES)
    assert agent["status"] == "none" and agent["orders"] == []
    assert agent["blocked"] == [{"action": "draft_stains", "reason": RULES.lane_guard.reason}]
    contract.Agent.model_validate(agent)


def test_surgical_lane_keeps_non_stain_orders():
    agent = R.draft_orders(make_case(lane="surgical", reliability_flag=True), CFG, RULES)
    assert agent["status"] == "drafted"
    assert {o["type"] for o in agent["orders"]} == {"recut", "deeper_levels"}
    assert agent["blocked"]


def test_surgical_case_without_stain_rules_has_no_blocked_action():
    agent = R.draft_orders(make_case(top="Normal", conformal_set=("Normal",), lane="surgical"), CFG, RULES)
    assert agent == {"status": "none", "orders": [], "blocked": []}


def test_items_are_deduped_across_rules():
    # top InSitu with {Benign, InSitu}: breast_insitu drafts ER, so uncertain_benign_atypia keeps only CK5/6.
    agent = R.draft_orders(make_case(top="InSitu", conformal_set=("Benign", "InSitu")), CFG, RULES)
    stains = [o for o in agent["orders"] if o["type"] == "stain"]
    assert [o["items"] for o in stains] == [["ER"], ["CK5/6"]]
    all_items = [it for o in agent["orders"] for it in o["items"]]
    assert len(all_items) == len(set(all_items))


@pytest.mark.parametrize("top,conformal_set,reliability_flag,qc,lane", [
    (top, s, flag, qc, lane) for top, s, flag, qc, lane in
    itertools.product(CLASSES, ALL_SETS, [False, True], [1, 12], ["biopsy", "surgical"])
])
def test_output_validates_and_never_invents_items(top, conformal_set, reliability_flag, qc, lane):
    agent = R.draft_orders(make_case(top, conformal_set, reliability_flag, qc, lane), CFG, RULES)
    contract.Agent.model_validate(agent)
    by_id = RULES.by_id()
    for order in agent["orders"]:
        allowed = {it for spec in by_id[order["rule_id"]].orders if spec.type == order["type"] for it in spec.items}
        assert set(order["items"]) <= allowed
        if lane == "surgical":
            assert order["type"] != "stain"


def test_no_rule_fires_for_a_clean_benign_read():
    agent = R.draft_orders(make_case(top="Benign", conformal_set=("Benign",)), CFG, RULES)
    assert agent == {"status": "none", "orders": [], "blocked": []}


def test_insufficient_rationale_names_the_counts():
    agent = R.draft_orders(make_case(top="Normal", conformal_set=("Normal",), qc=1), CFG, RULES)
    (order,) = agent["orders"]
    assert order["rule_id"] == "insufficient" and f"fewer than {MIN_KEPT}" in order["rationale"]
    assert "Only 1 tissue" in order["rationale"]


def test_tie_resolves_toward_the_aggressive_class_like_core_tiers():
    case = make_case()
    case["prediction"]["probs"] = [0.0, 0.0, 0.5, 0.5]
    assert R.top_class(case["prediction"], CFG["aggressive_class"]) == "Invasive"
    assert "breast_invasive" in fired(case)


def _raw():
    import yaml
    return yaml.safe_load(R.rules_path(CFG).read_text(encoding="utf-8"))


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d["rules"][0]["when"].update({"top_class": "Melanoma"}), "top_class"),
    (lambda d: d["rules"][0]["when"].update({"colour": "red"}), "unknown condition"),
    (lambda d: d["rules"][0]["order"].update({"type": "biopsy"}), "type must be one of"),
    (lambda d: d["rules"][0]["order"].update({"items": []}), "items"),
    (lambda d: d["rules"][0].update({"order": [{"type": "stain", "items": ["p63"]}]}), "one order per rule"),
    (lambda d: d["rules"][1].update({"id": d["rules"][0]["id"]}), "unique"),
    (lambda d: d["rules"][0].update({"rationale": "uses {unknown}"}), "unknown fields"),
    (lambda d: d.update({"status": "approved"}), "clinical review"),
    (lambda d: d["lane_guard"].update({"blocked_types": ["surgery"]}), "blocked_types"),
])
def test_invalid_rule_files_are_rejected(mutate, message):
    data = copy.deepcopy(_raw())
    mutate(data)
    with pytest.raises(ValueError, match=message):
        R.parse_rules(data, CFG)


def test_rule_metadata_is_json_ready():
    meta = R.rule_metadata(RULES)
    json.dumps(meta)
    assert meta[0]["rule_id"] == "breast_insitu_vs_invasive" and meta[0]["citations"] == ["CAP-IHC"]


@pytest.mark.parametrize("top,conformal_set,reliability_flag,qc,lane", [
    (top, s, flag, qc, lane) for top, s, flag, qc, lane in
    itertools.product(CLASSES, ALL_SETS, [False, True], [1, 12], ["biopsy", "surgical"])
])
def test_rule_ids_are_unique_per_case_backend_addresses_orders_by_rule_id(top, conformal_set, reliability_flag, qc, lane):
    agent = R.draft_orders(make_case(top, conformal_set, reliability_flag, qc, lane), CFG, RULES)
    ids = [o["rule_id"] for o in agent["orders"]]
    assert len(ids) == len(set(ids))
