"""Case JSON contract (plan §7 + CELLS_SPEC §7.2 + BUILD_PLAN §A13), fixtures, the TypeScript mirror and the
frozen Task-1 interfaces that later tasks implement or consume.

Case builders below are hand-written; every expected rejection names the broken rule.
"""

import copy
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import get_args

import pytest
from PIL import Image
from pydantic import ValidationError

from core import config, contract, schemas, tiers

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]


def biopsy_case():
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "synthetic": True,
        "case_id": "case_b1",
        "lane": "biopsy",
        "simulated": False,
        "arrived_offset_s": 340,
        "image": {"display": "images/case_b1.jpg", "width": 2048, "height": 1536, "tile_size": 512, "grid": [3, 4]},
        "quality": {"qc_tiles_kept": 3, "ood_percentile": 0.42, "reliability": 0.81, "reliability_flag": False},
        "prediction": {
            "model": "concept_bottleneck", "classes": CLASSES, "probs": [0.02, 0.05, 0.21, 0.72],
            "risk_score": 0.72, "conformal_alpha": 0.1, "conformal_set": ["InSitu", "Invasive"],
        },
        "concepts": [
            {"id": "invasion", "label": "Tumor infiltrating stroma", "value": 1.9, "contribution": 0.44,
             "direction": "up", "spread": 0.06, "top_tiles": [5, 6], "source": "cav",
             "components": {"text_z": 1.7, "cav_z": 2.1}, "caption_consistent": True},
            {"id": "duct_confined", "label": "Cells confined within ducts", "value": 0.8, "contribution": -0.12,
             "direction": "down", "spread": 0.09, "top_tiles": [2]},
        ],
        "tiles": [
            {"idx": 2, "row": 0, "col": 2, "x": 1024, "y": 0, "risk": 0.35, "concepts": {"invasion": 0.4, "duct_confined": 1.1}},
            {"idx": 5, "row": 1, "col": 1, "x": 512, "y": 512, "risk": 0.88, "concepts": {"invasion": 2.3}},
            {"idx": 6, "row": 1, "col": 2, "x": 1024, "y": 512, "risk": 0.81, "concepts": {"invasion": 2.0}, "attention": 0.4},
        ],
        "similar_cases": [{"case_id": "ref_0007", "label": "Invasive", "similarity": 0.93, "image": "images/ref_0007.jpg"}],
        "contrast_case": {"case_id": "ref_0044", "label": "InSitu", "similarity": 0.88, "differs_most_in": ["invasion"]},
        "agreement": {"conch_musk_same_top": True, "neighbor_agreement": 0.8},
        "margin": None,
        "cells": {"status": "disabled"},
        "flag": {"flagged": True, "tier": 2, "reasons": ["aggressive_top", "aggressive_in_conformal_set", "set_size_gt_1"]},
        "findings": [
            {"id": "F1", "kind": "prediction", "data": {"top": "Invasive", "p": 0.72, "set": ["InSitu", "Invasive"]}},
            {"id": "F2", "kind": "concept", "data": {"concept": "invasion", "contribution": 0.44, "tiles": [5, 6]}},
            {"id": "F3", "kind": "retrieval", "data": {"agree": 4, "of": 5, "label": "Invasive"}},
            {"id": "F4", "kind": "quality", "data": {"reliability_flag": False}},
        ],
        "summary": {"generator": "template", "sentences": [
            {"text": "Suggested category: invasive carcinoma, with in situ carcinoma also in the confidence set.",
             "findings": ["F1"]},
            {"text": "Tumor cells infiltrating stroma is the feature most associated with the suggestion.", "findings": ["F2"]},
        ]},
        "agent": {"status": "drafted", "orders": [
            {"type": "stain", "items": ["p63", "calponin"], "rule_id": "breast_insitu_vs_invasive",
             "rationale": "Myoepithelial markers help distinguish in situ from invasive disease."},
            {"type": "stain", "items": ["ER", "PR", "HER2", "Ki-67"], "rule_id": "breast_invasive",
             "rationale": "Standard biomarker panel if invasion is confirmed."},
        ]},
        "eval_only": {"true_label": "Invasive", "split": "test"},
    }


def ring_segments(p_at):
    """28 margin memberships of the 24 border squares of a 2048x1536 field (256-px squares, corners shared)."""
    segs = [{"margin": "superior", "x": x, "y": 0, "size": 256, "p": p_at(x, 0)} for x in range(0, 2048, 256)]
    segs += [{"margin": "inferior", "x": x, "y": 1280, "size": 256, "p": p_at(x, 1280)} for x in range(0, 2048, 256)]
    segs += [{"margin": "medial", "x": 0, "y": y, "size": 256, "p": p_at(0, y)} for y in range(0, 1536, 256)]
    segs += [{"margin": "lateral", "x": 1792, "y": y, "size": 256, "p": p_at(1792, y)} for y in range(0, 1536, 256)]
    return segs


HOT = {(768, 0): 0.81, (1024, 1280): 0.07, (0, 512): 0.12, (1792, 768): 0.04}


def surgical_case():
    case = biopsy_case()
    case.update(case_id="case_s1", lane="surgical", simulated=True)
    case["image"]["display"] = "images/case_s1.jpg"
    case["margin"] = {
        "suggested_call": "tumor_at_margin",
        "involved": ["superior"],
        "per_margin": {"superior": 0.81, "inferior": 0.07, "medial": 0.12, "lateral": 0.04},
        "strip_um": 107.52,
        "threshold": 0.5,
        "segments": ring_segments(lambda x, y: HOT.get((x, y), 0.03)),
        "involved_length_um": {"superior": 107.52},
    }
    case["flag"] = {"flagged": False, "tier": 1,
                    "reasons": ["aggressive_top", "aggressive_in_conformal_set", "set_size_gt_1", "surgical_frozen_section"]}
    case["findings"].append({"id": "F5", "kind": "margin", "data": {"suggested_call": "tumor_at_margin", "involved": ["superior"]}})
    case["summary"]["sentences"].append({"text": "Suggested margin call: tumor at the superior margin.", "findings": ["F5"]})
    case["agent"] = {"status": "none", "orders": [], "blocked": [
        {"action": "draft_stains", "reason": "Stains are not drafted for simulated frozen sections."}]}
    return case


def rejects(case, match):
    with pytest.raises(ValidationError, match=match):
        contract.Case.model_validate(case)


# ---- valid cases ---------------------------------------------------------------------------------
def test_biopsy_case_validates_and_round_trips():
    case = contract.Case.model_validate(biopsy_case())
    again = contract.Case.model_validate(json.loads(case.model_dump_json()))
    assert again == case


def test_surgical_case_validates():
    case = contract.Case.model_validate(surgical_case())
    assert case.margin.suggested_call == "tumor_at_margin"
    assert len(case.margin.segments) == 28


def test_tiers_flag_of_builder_cases_matches_core_tiers():
    cfg = config.load_settings()
    for build in (biopsy_case, surgical_case):
        case = build()
        assert tiers.flag_for(contract.Case.model_validate(case), cfg) == case["flag"]


# ---- eval_only never leaves --------------------------------------------------------------------------
def test_case_public_rejects_eval_only():
    with pytest.raises(ValidationError, match="eval_only"):
        contract.CasePublic.model_validate(biopsy_case())


def test_to_public_strips_ground_truth():
    public = contract.to_public(biopsy_case())
    text = json.dumps(public)
    assert "eval_only" not in public and "true_label" not in text
    contract.CasePublic.model_validate(public)


def test_public_method_drops_eval_only():
    public = contract.Case.model_validate(biopsy_case()).public()
    assert isinstance(public, contract.CasePublic)
    assert "eval_only" not in public.model_dump()


# ---- lane, simulation and flag rules ------------------------------------------------------------------
def test_surgical_case_needs_a_margin_read():
    case = surgical_case()
    case["margin"] = None
    rejects(case, "surgical")


def test_surgical_case_must_be_marked_simulated():
    case = surgical_case()
    case["simulated"] = False
    rejects(case, "simulated")


def test_biopsy_case_has_no_margin():
    case = biopsy_case()
    case["margin"] = surgical_case()["margin"]
    rejects(case, "biopsy")


def test_tier_1_is_reserved_for_the_surgical_lane():
    case = biopsy_case()
    case["flag"]["tier"] = 1
    rejects(case, "tier")


def test_flag_reasons_use_the_enum():
    case = biopsy_case()
    case["flag"]["reasons"].append("looks_bad")
    rejects(case, "reasons")


def test_flag_reason_enum_matches_core_tiers_order():
    assert contract.FLAG_REASONS == tiers.REASON_ORDER


def test_degraded_variant_names_its_degradation_and_source():
    case = surgical_case()
    case["simulated_variant"] = "degraded"
    case["source_case_ids"] = ["case_b1"]
    rejects(case, "degradation")
    case["degradation"] = {"type": "blur", "severity": 3}
    contract.Case.model_validate(case)
    case["source_case_ids"] = []
    rejects(case, "source_case_ids")


def test_composite_variant_lists_both_sources():
    case = surgical_case()
    case.update(simulated_variant="composite", source_case_ids=["case_b1", "case_b2"])
    contract.Case.model_validate(case)


# ---- prediction --------------------------------------------------------------------------------------
def test_probs_must_match_classes():
    case = biopsy_case()
    case["prediction"]["probs"] = [0.5, 0.5]
    rejects(case, "probs")


def test_probs_must_sum_to_one():
    case = biopsy_case()
    case["prediction"]["probs"] = [0.1, 0.1, 0.1, 0.1]
    rejects(case, "sum")


def test_conformal_set_must_use_prediction_classes():
    case = biopsy_case()
    case["prediction"]["conformal_set"] = ["Carcinoma"]
    rejects(case, "Carcinoma")


def test_unknown_fields_are_rejected_everywhere():
    case = biopsy_case()
    case["prediction"]["temperature"] = 1.3
    rejects(case, "temperature")


def test_schema_version_must_match():
    case = biopsy_case()
    case["schema_version"] = "0.9"
    rejects(case, "schema_version")


# ---- tiles, concepts, findings, summary ----------------------------------------------------------------
def test_tile_geometry_must_match_the_grid():
    case = biopsy_case()
    case["tiles"][1]["idx"] = 4  # row 1, col 1 on a 3x4 grid is idx 5
    rejects(case, "tile")


def test_qc_tiles_kept_counts_the_listed_tiles():
    case = biopsy_case()
    case["quality"]["qc_tiles_kept"] = 11
    rejects(case, "qc_tiles_kept")


def test_concept_top_tiles_must_be_listed_tiles():
    case = biopsy_case()
    case["concepts"][0]["top_tiles"] = [5, 11]
    rejects(case, "top_tiles")


def test_summary_may_only_cite_existing_findings():
    case = biopsy_case()
    case["summary"]["sentences"][0]["findings"] = ["F9"]
    rejects(case, "F9")


def test_summary_sentence_must_cite_a_finding():
    case = biopsy_case()
    case["summary"]["sentences"][0]["findings"] = []
    rejects(case, "findings")


def test_summary_generator_is_template_or_lumen_model():
    case = biopsy_case()
    case["summary"]["generator"] = "gpt"
    rejects(case, "generator")
    case["summary"]["generator"] = "lumen:ornith-1.0-35b"
    contract.Case.model_validate(case)


def test_finding_data_carries_its_kind_keys():
    case = biopsy_case()
    del case["findings"][0]["data"]["set"]
    rejects(case, "set")


def test_finding_ids_are_F_numbers():
    case = biopsy_case()
    case["findings"][3]["id"] = "Q4"
    rejects(case, "id")


# ---- margin ------------------------------------------------------------------------------------------------
def test_per_margin_is_the_max_over_its_segments():
    case = surgical_case()
    case["margin"]["per_margin"]["medial"] = 0.5
    rejects(case, "medial")


def test_involved_margins_follow_the_threshold():
    case = surgical_case()
    case["margin"]["involved"] = ["superior", "medial"]
    rejects(case, "involved")


def test_clear_call_has_no_involved_margin():
    case = surgical_case()
    case["margin"]["suggested_call"] = "clear"
    rejects(case, "clear")


def test_segments_sit_on_their_own_edge():
    case = surgical_case()
    case["margin"]["segments"][0]["y"] = 256  # a superior segment must touch the top edge
    rejects(case, "superior")


def test_per_margin_needs_all_four_margins():
    case = surgical_case()
    del case["margin"]["per_margin"]["lateral"]
    rejects(case, "lateral")


# ---- cells ----------------------------------------------------------------------------------------------------
def cells_ok():
    return {
        "status": "ok", "model": "hovernet_fast-pannuke", "mpp_model": 0.25,
        "tiles": [{"idx": 5, "overlay": "cells/case_b1_5_overlay.png",
                   "counts": {"neoplastic": 212, "inflammatory": 31, "connective": 18, "dead": 4, "epithelial": 2}}],
        "features": {"nuc_area_cv": {"value": 0.48, "unit": "", "ref_percentile": 0.95, "contribution": 0.21}},
        "flags": {"insufficient_tumor_nuclei": False},
    }


def test_cells_ok_block_validates():
    case = biopsy_case()
    case["cells"] = cells_ok()
    contract.Case.model_validate(case)


def test_cells_ok_needs_model_and_resolution():
    case = biopsy_case()
    case["cells"] = cells_ok()
    del case["cells"]["model"]
    rejects(case, "model")


def test_cell_counts_use_pannuke_types():
    case = biopsy_case()
    case["cells"] = cells_ok()
    case["cells"]["tiles"][0]["counts"]["mitotic"] = 3
    rejects(case, "mitotic")


def test_disabled_cells_block_carries_no_measurements():
    case = biopsy_case()
    case["cells"] = {**cells_ok(), "status": "disabled"}
    rejects(case, "disabled")


def test_cells_block_may_be_absent():
    case = biopsy_case()
    del case["cells"]
    contract.Case.model_validate(case)


# ---- agent ---------------------------------------------------------------------------------------------------
def test_drafted_agent_has_orders():
    case = biopsy_case()
    case["agent"]["orders"] = []
    rejects(case, "drafted")


def test_order_types_use_the_enum():
    case = biopsy_case()
    case["agent"]["orders"][0]["type"] = "biopsy"
    rejects(case, "type")


# ---- fixtures ----------------------------------------------------------------------------------------------------
FIXTURES = sorted((REPO / "fixtures").glob("*.json"))


def test_both_demo_fixtures_exist():
    assert [p.name for p in FIXTURES] == ["demo_0001.json", "demo_0002.json"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_validates_and_is_marked_synthetic(path):
    data = json.loads(path.read_text())
    case = contract.Case.model_validate(data)
    assert case.synthetic is True
    assert case.case_id == path.stem


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_flag_matches_core_tiers(path):
    data = json.loads(path.read_text())
    assert tiers.flag_for(data, config.load_settings()) == data["flag"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_media_exist_in_the_flat_fixtures_folder(path):
    data = json.loads(path.read_text())
    refs = [data["image"]["display"], *(s["image"] for s in data["similar_cases"])]
    refs += [t["overlay"] for t in (data.get("cells") or {}).get("tiles", [])]
    for ref in refs:
        media = schemas.fixture_media(REPO / "fixtures", ref)
        assert media.is_file(), ref
        with Image.open(media) as im:
            im.verify()


def test_demo_0001_is_the_biopsy_demo_case():
    case = contract.Case.model_validate_json((REPO / "fixtures" / "demo_0001.json").read_text())
    assert case.lane == "biopsy" and case.flag.tier == 2
    assert case.prediction.classes[case.prediction.probs.index(max(case.prediction.probs))] == "Invasive"
    assert case.prediction.conformal_set == ["InSitu", "Invasive"]
    ordered = {item for order in case.agent.orders for item in order.items}
    assert {"p63", "calponin", "ER", "PR", "HER2", "Ki-67"} <= ordered
    assert case.agent.status == "drafted" and case.cells.status == "ok"


def test_demo_0002_is_the_simulated_surgical_case():
    case = contract.Case.model_validate_json((REPO / "fixtures" / "demo_0002.json").read_text())
    assert case.lane == "surgical" and case.simulated and case.flag.tier == 1
    assert case.margin.suggested_call == "tumor_at_margin" and case.margin.segments


# ---- TypeScript mirror (frontend/src/types.ts) --------------------------------------------------------------------
def ts_interfaces(src):
    out = {}
    for block in re.finditer(r"export interface (\w+) \{(.*?)\n\}", src, re.S):
        fields = {}
        for line in block.group(2).splitlines():
            m = re.match(r"\s+(\w+)(\?)?:\s*(.+?);\s*(//.*)?$", line)
            if m:
                fields[m.group(1)] = bool(m.group(2))
        out[block.group(1)] = fields
    return out


def ts_unions(src):
    return {m.group(1): set(re.findall(r'"([^"]+)"', m.group(2)))
            for m in re.finditer(r"export type (\w+) =([^;]+);", src)}


def test_types_ts_mirrors_case_public_field_for_field():
    src = (REPO / "frontend" / "src" / "types.ts").read_text()
    interfaces = ts_interfaces(src)
    for model in contract.PUBLIC_MODELS:
        want = {name: not field.is_required() for name, field in model.model_fields.items()}
        assert interfaces.get(model.__name__) == want, model.__name__


def test_types_ts_mirrors_contract_enums():
    src = (REPO / "frontend" / "src" / "types.ts").read_text()
    unions = ts_unions(src)
    for name, literal in contract.ENUMS.items():
        assert unions.get(name) == set(get_args(literal)), name
    assert f'SCHEMA_VERSION = "{contract.SCHEMA_VERSION}"' in src


# ---- frozen interfaces (Task 1 stubs; Task 2/4 implement, Waves 1-2 consume) ----------------------------------------------
FROZEN = {
    "core.conformal": {
        "lac_scores": ["probs", "labels"],
        "conformal_quantile": ["scores", "alpha"],
        "lac_sets": ["probs", "qhat"],
        "fit_lac": ["cal_probs", "cal_labels", "alpha"],
        "aps_scores": ["probs", "labels", "seed", "randomized"],
        "aps_sets": ["probs", "qhat", "seed", "randomized"],
        "fit_aps": ["cal_probs", "cal_labels", "alpha", "seed", "randomized"],
        "mondrian_quantiles": ["cal_probs", "cal_labels", "alpha"],
        "mondrian_sets": ["probs", "qhats"],
        "set_labels": ["mask", "classes"],
        "coverage": ["sets", "labels"],
        "mean_set_size": ["sets"],
    },
    "core.calibration": {
        "softmax": ["logits", "temperature"],
        "fit_temperature": ["logits", "labels", "bounds"],
        "nll": ["probs", "labels"],
        "brier": ["probs", "labels"],
        "ece": ["probs", "labels", "n_bins"],
        "reliability_diagram": ["probs", "labels", "n_bins"],
    },
    "core.stats": {
        "bootstrap_ci": ["metric", "arrays", "groups", "strata", "n_boot", "ci", "seed"],
        "paired_difference_ci": ["metric", "y_true", "score_a", "score_b", "groups", "strata", "n_boot", "ci", "seed"],
        "auroc": ["y_true", "scores"],
        "auroc_ovr": ["y_true", "probs", "n_classes"],
        "macro_auroc": ["y_true", "probs", "n_classes"],
        "macro_f1": ["y_true", "y_pred", "n_classes"],
        "balanced_accuracy": ["y_true", "y_pred", "n_classes"],
        "confusion_matrix": ["y_true", "y_pred", "n_classes"],
    },
    "core.margin": {
        "strip_segments": ["width", "height", "segment_px", "names"],
        "per_margin_scores": ["segments", "probs"],
        "fit_threshold": ["benign_per_margin", "max_fpr"],
        "involved_margins": ["per_margin", "threshold"],
        "involved_length_um": ["segments", "probs", "threshold", "mpp"],
        "suggest_call": ["per_margin", "threshold", "reliability_flag"],
    },
    "pipeline.degrade": {
        "severity_params": ["cfg", "degrade_type", "severity"],
        "gaussian_blur": ["image", "sigma_px"],
        "jpeg_compress": ["image", "quality"],
        "stain_shift": ["image", "alpha", "beta", "seed"],
        "down_up_sample": ["image", "factor"],
        "apply": ["image", "degrade_type", "severity", "seed", "cfg"],
    },
}


@pytest.mark.parametrize("module_name", sorted(FROZEN))
def test_frozen_interfaces_keep_their_signatures(module_name):
    module = importlib.import_module(module_name)
    got = {name: list(inspect.signature(getattr(module, name)).parameters) for name in FROZEN[module_name]}
    assert got == FROZEN[module_name]
    assert all(getattr(module, name).__doc__ for name in FROZEN[module_name])


def test_contract_dump_is_json_serialisable_for_the_api():
    public = contract.to_public(copy.deepcopy(surgical_case()))
    assert json.loads(json.dumps(public)) == public
