"""backend/summary.py: validator rules, template determinism, LLM path with mocked client, cache invalidation."""

from __future__ import annotations

import copy
import json
import re

import httpx
import pytest

from agent import lumen
from backend import summary as S
from core import config, contract

CFG = config.load_settings()


def fixture(name="demo_0001"):
    return json.loads((config.fixtures_root(CFG) / f"{name}.json").read_text(encoding="utf-8"))


FINDINGS = fixture()["findings"]


def sentence(text, *ids):
    return {"text": text, "findings": list(ids)}


# ---- validator -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad,reason", [
    (sentence("Most likely invasive carcinoma."), "cites no finding"),
    (sentence("Most likely invasive carcinoma.", "F9"), "unknown findings"),
    (sentence("Calibrated probability 0.81.", "F1"), "number 0.81"),
    (sentence("Calibrated probability 0.72 in 3 regions.", "F1"), "number 3"),
    (sentence("4 of 6 similar cases agree.", "F4"), "number 6"),
    (sentence("Regions 5 and 6 show invasion.", "F1"), "number 5"),          # tiles are in F2, not F1
    (sentence("The diagnosis is invasive carcinoma.", "F1"), "banned"),
    (sentence("Invasion causes the high risk.", "F2"), "banned"),
    ({"text": "", "findings": ["F1"]}, "not a {text, findings}"),
    ({"findings": ["F1"]}, "not a {text, findings}"),
    ("just a string", "not a {text, findings}"),
    ({"text": "ok", "findings": "F1"}, "list of finding IDs"),
])
def test_validator_drops_bad_sentences(bad, reason):
    kept, dropped = S.validate([bad], FINDINGS, CFG)
    assert kept == [] and len(dropped) == 1 and reason in dropped[0]["reason"]


@pytest.mark.parametrize("good", [
    sentence("Suggested category: invasive carcinoma (calibrated probability 0.72); in situ carcinoma cannot be "
             "excluded.", "F1"),
    sentence("Suggested category: invasive carcinoma (calibrated probability 72%); in situ carcinoma cannot be "
             "excluded.", "F1"),
    sentence("Suggested category: invasive carcinoma (probability 0.7); in situ carcinoma cannot be excluded.",
             "F1"),                                                                         # rounding of 0.72
    sentence("In the concept model, tumor infiltrating stroma in regions 5 and 6 pushes the score up by 0.44.", "F2"),
    sentence("Variation in nuclear size is higher than in 95% of benign or normal reference cases (measured in regions 5 "
             "and 6).", "F6"),
    sentence("4 of 5 similar reference cases share the suggested category.", "F4"),
    sentence("Ki-67, CK5/6 and p63 are named in F1 but are not numbers.", "F2"),
])
def test_validator_keeps_grounded_sentences(good):
    kept, dropped = S.validate([good], FINDINGS, CFG)
    assert dropped == [] and kept == [good]


def test_signs_are_ignored_but_uncited_values_are_not():
    assert S.validate([sentence("In the concept model it pushes the score up (-0.44).", "F2")], FINDINGS, CFG)[1] == []
    assert S.validate([sentence("Duct confinement pushes it down (-0.12).", "F2")], FINDINGS, CFG)[0] == []


def test_number_words_inside_a_cited_label_are_allowed():
    findings = FINDINGS + [{"id": "F9", "kind": "concept", "data": {"concept": "normal_lobules", "contribution": -0.2,
                                                                    "tiles": [3], "label": "Normal two-layered lobules"}}]
    ok = sentence("Normal two-layered lobules (region 3) lower the score in the concept model.", "F9")
    assert S.validate([ok], findings, CFG)[0] == [ok]
    assert S.validate([sentence("Two regions show lobules.", "F9")], findings, CFG)[0] == []


def test_numbers_across_multiple_cited_findings_are_pooled():
    s = sentence("Invasive carcinoma is suggested (probability 0.72; in situ carcinoma cannot be excluded), with "
                 "tumor infiltrating stroma in regions 5 and 6 noted by the concept model.", "F1", "F2")
    assert S.validate([s], FINDINGS, CFG)[0] == [s]


def test_percent_and_decimal_forms_are_consistent():
    assert S.numbers_in_text("72% and 0.72 and 3,150 and F12 and Ki-67 and .5", codes=["Ki-67"]) == [
        (72.0, 0, True), (0.72, 2, False), (3150.0, 0, False), (0.5, 1, False)]
    v = [(0.72, False)]
    assert S.number_supported(72, 0, True, v)
    assert S.number_supported(72, 0, True, [(72.0, False)])
    assert not S.number_supported(72, 0, False, v)        # "72" without % is not 0.72
    assert S.number_supported(0.7, 1, False, v)
    assert not S.number_supported(0.70, 2, False, v)
    assert not S.number_supported(1, 0, False, v)         # a whole number never rounds a probability
    assert S.number_supported(3150, 0, False, [(3150.4, False)])
    assert not S.number_supported(5, 0, True, [(5.0, True)])     # "5%" is not tile index 5
    assert S.number_supported(5, 0, False, [(5.0, True)])
    assert not S.number_supported(0.5, 1, False, [(0.5, True)])  # counts never support decimals


@pytest.mark.parametrize("text", [
    "Probability of .99 here.", "probability-0.99 here.", "ER-90% expected.", "Grade-3 features.", "pT3 disease.",
    "ninety-nine percent likely.", "Probability 1 for invasive carcinoma.", "5% of the tissue.", "The 95th percentile.",
])
def test_reviewer_bypasses_are_rejected(text):
    kept, dropped = S.validate([sentence(text, "F1", "F2")], FINDINGS, CFG)
    assert kept == [], f"{text!r} should be dropped"


def test_validator_caps_sentences_and_dedupes_citations():
    many = [sentence("Invasive carcinoma is suggested; in situ carcinoma cannot be excluded.", "F1", "F1")] * 6
    kept, dropped = S.validate(many, FINDINGS, CFG)
    assert len(kept) == CFG["summary"]["max_sentences"] and kept[0]["findings"] == ["F1"]
    assert all(d["reason"] == "beyond summary.max_sentences" for d in dropped)


def test_validator_rejects_a_non_list_reply():
    kept, dropped = S.validate({"text": "x"}, FINDINGS, CFG)
    assert kept == [] and "not a list" in dropped[0]["reason"]


# ---- template ------------------------------------------------------------------------------------------------------
def test_template_is_deterministic_grounded_and_valid():
    for name in ("demo_0001", "demo_0002"):
        case = fixture(name)
        a = S.template(case["findings"], CFG)
        b = S.template(copy.deepcopy(case["findings"]), CFG)
        assert a == b and 2 <= len(a) <= CFG["summary"]["max_sentences"]
        kept, dropped = S.validate(a, case["findings"], CFG)
        assert dropped == [] and kept == a
        case["summary"] = S.template_summary(case["findings"], CFG)
        contract.Case.model_validate(case)            # contract citation rule


def test_template_covers_prediction_concept_cell_and_retrieval():
    texts = [s["text"] for s in S.template(FINDINGS, CFG)]
    assert texts[0].startswith("Suggested category: invasive carcinoma (calibrated probability 72%)")
    assert "in situ carcinoma is also in the confidence set" in texts[0]
    assert any(t.startswith("Features associated with invasive carcinoma in the concept model: invasion (regions 5 "
                            "and 6), then pleomorphism") for t in texts)
    assert any("higher than in 95% of benign or normal reference cases (measured in regions 5 and 6)" in t for t in texts)


def test_template_margin_sentence_for_surgical_cases():
    texts = [s["text"] for s in S.template(fixture("demo_0002")["findings"], CFG)]
    assert "Suggested margin call (simulated frozen section): tumor at the superior margin." in texts


@pytest.mark.parametrize("call,involved,expect", [
    ("clear", [], "margins clear"),
    ("defer", [], "defer to permanent sections"),
    ("tumor_at_margin", ["lateral", "superior"], "tumor at the superior and lateral margins"),
])
def test_template_margin_calls(call, involved, expect):
    findings = [{"id": "F1", "kind": "prediction", "data": {"top": "Benign", "p": 0.6, "set": ["Benign"]}},
                {"id": "F2", "kind": "margin", "data": {"suggested_call": call, "involved": involved}}]
    texts = [s["text"] for s in S.template(findings, CFG)]
    assert any(expect in t for t in texts)


@pytest.mark.parametrize("members,expect", [
    ([], "is empty"), (["Benign"], "holds this category only"), (["InSitu"], "holds in situ carcinoma instead"),
    (["Benign", "InSitu", "Invasive"], "in situ carcinoma and invasive carcinoma are also"),
])
def test_template_prediction_set_phrasing(members, expect):
    findings = [{"id": "F1", "kind": "prediction", "data": {"top": "Benign", "p": 0.512, "set": members,
                                                            "confidence": 0.9}},
                {"id": "F2", "kind": "quality", "data": {"reliability_flag": False}}]
    out = S.template(findings, CFG)
    assert expect in out[0]["text"] and "90% confidence set" in out[0]["text"] and "(calibrated probability 51%)" in out[0]["text"]
    assert S.validate(out, findings, CFG)[1] == []


def test_template_low_percentile_cells_and_quality_concerns():
    findings = [
        {"id": "F1", "kind": "prediction", "data": {"top": "Normal", "p": 0.9, "set": ["Normal"]}},
        {"id": "F2", "kind": "cells", "data": {"feature": "neo_density", "label": "Tumor cell density", "value": 10.0,
                                               "ref_percentile": 0.04, "tiles": [3]}},
        {"id": "F3", "kind": "quality", "data": {"reliability_flag": True, "qc_tiles_kept": 2, "min_tiles_kept": 3}},
    ]
    out = S.template(findings, CFG)
    texts = " ".join(s["text"] for s in out)
    assert "Only 4% of benign or normal reference cases have a lower tumor cell density (measured in region 3)" in texts
    assert "Only 2 tissue regions passed quality control (fewer than 3)" in texts
    assert S.validate(out, findings, CFG)[1] == []


def test_template_respects_max_sentences_by_priority():
    case = fixture()
    findings = case["findings"] + [{"id": "F9", "kind": "concept", "data": {"concept": "tubules", "label": "Tubules",
                                                                            "contribution": -0.3, "tiles": [0]}}]
    out = S.template(findings, CFG)
    assert len(out) == CFG["summary"]["max_sentences"]
    assert out[0]["findings"] == ["F1"]


def test_template_concept_down_sentence():
    findings = [{"id": "F1", "kind": "prediction", "data": {"top": "Normal", "p": 0.8, "set": ["Normal"]}},
                {"id": "F2", "kind": "concept", "data": {"concept": "normal_lobules", "label": "Normal two-layered lobules",
                                                         "contribution": -0.4, "tiles": [3]}}]
    texts = [s["text"] for s in S.template(findings, CFG)]
    assert ("Feature associated with a lower likelihood of invasive carcinoma in the concept model: normal two-layered "
            "lobules (region 3).") in texts


# ---- generate (LLM path mocked) ---------------------------------------------------------------------------------------
class Server:
    def __init__(self, contents):
        self.contents = list(contents)
        self.chat_calls = 0

    def __call__(self, request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "ornith-1.0-35b"}]})
        self.chat_calls += 1
        content = self.contents.pop(0)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {}})


def client_for(server, tmp_path, key="k", max_calls=3):
    return lumen.LumenClient(base_url="https://lumen.example/v1", api_key=key, model_preference=["ornith-1.0-35b"],
                             cache_dir=tmp_path / "cache", max_calls=max_calls, retries=0, timeout_s=5,
                             temperature=0.2, max_tokens=100, transport=httpx.MockTransport(server),
                             sleep=lambda s: None)


GOOD_REPLY = json.dumps({"sentences": [
    sentence("Invasive carcinoma is suggested (probability 0.72); in situ carcinoma cannot be excluded.", "F1"),
    sentence("Tumor infiltrating stroma in regions 5 and 6 is associated with invasive carcinoma in the concept model.", "F2"),
    sentence("This sentence cites nothing."),
]})


def test_generate_without_client_is_template_and_zero_calls():
    out, info = S.generate(FINDINGS, CFG)
    assert out["generator"] == "template" and info["llm_calls"] == 0 and info["path"] == "template"
    contract.Summary.model_validate(out)


def test_generate_with_llm_validates_and_badges_the_model(tmp_path):
    server = Server([GOOD_REPLY])
    out, info = S.generate(FINDINGS, CFG, client=client_for(server, tmp_path))
    assert out["generator"] == "lumen:ornith-1.0-35b" and len(out["sentences"]) == 2
    assert info["path"] == "llm" and info["llm_calls"] == 1 and info["dropped"][0]["reason"] == "cites no finding"
    contract.Summary.model_validate(out)


def test_generate_prompt_contains_findings_only(tmp_path):
    messages = S.build_messages(FINDINGS, CFG)
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"prompt_version", "classes", "aggressive_class", "findings"}
    text = json.dumps(messages)
    assert "true_label" not in text and "eval_only" not in text and ".jpg" not in text


def test_prompt_refuses_ground_truth_keys():
    leaky = copy.deepcopy(FINDINGS)
    leaky[0]["data"]["true_label"] = "Invasive"
    with pytest.raises(ValueError, match="ground truth"):
        S.build_messages(leaky, CFG)


def test_generate_falls_back_when_too_few_valid_sentences(tmp_path):
    reply = json.dumps([sentence("Probability 0.99.", "F1"), sentence("Uncited.")])
    out, info = S.generate(FINDINGS, CFG, client=client_for(Server([reply]), tmp_path))
    assert out["generator"] == "template" and "only 0 valid" in info["reason"] and info["llm_calls"] == 1


def test_generate_missing_key_takes_template_path(tmp_path):
    server = Server([GOOD_REPLY])
    client = client_for(server, tmp_path, key=None)
    client._model = "ornith-1.0-35b"
    out, info = S.generate(FINDINGS, CFG, client=client)
    assert out["generator"] == "template" and "NoApiKey" in info["reason"] and server.chat_calls == 0


def test_generate_unparseable_reply_takes_template_path(tmp_path):
    out, info = S.generate(FINDINGS, CFG, client=client_for(Server(["I cannot help"]), tmp_path))
    assert out["generator"] == "template" and "ValueError" in info["reason"]


def test_cache_hit_reuses_text_and_changed_findings_fall_back_to_template(tmp_path):
    server = Server([GOOD_REPLY])
    first, _ = S.generate(FINDINGS, CFG, client=client_for(server, tmp_path, max_calls=1))
    again, info = S.generate(FINDINGS, CFG, client=client_for(server, tmp_path, max_calls=0))
    assert again == first and info["cached"] and info["llm_calls"] == 0 and server.chat_calls == 1
    changed = copy.deepcopy(FINDINGS)
    changed[0]["data"]["p"] = 0.65
    out, info = S.generate(changed, CFG, client=client_for(server, tmp_path, max_calls=0))
    assert out["generator"] == "template" and "BudgetExceeded" in info["reason"]
    assert S.findings_hash(changed) != S.findings_hash(FINDINGS)


# ---- semantic consistency + required facts (2026-10-18) ------------------------------------------------------------
from backend import summary as _S  # noqa: E402

_MARGIN_DEFER = {"id": "F7", "kind": "margin", "data": {"suggested_call": "defer", "involved": []}}
_MARGIN_TUMOR = {"id": "F7", "kind": "margin", "data": {"suggested_call": "tumor_at_margin", "involved": ["superior", "medial"]}}
_FLAGGED = {"id": "F6", "kind": "quality", "data": {"reliability_flag": True, "qc_tiles_kept": 12, "min_tiles_kept": 3}}
_LYMPH_LOW = {"id": "F3", "kind": "concept", "data": {"concept": "lymphocytes", "label": "Dense lymphocytic infiltrate (TILs)",
                                                      "contribution": 0.6, "tiles": [1], "level": "below_typical"}}
_MIT_HIGH = {"id": "F2", "kind": "concept", "data": {"concept": "mitoses", "label": "Frequent mitotic figures",
                                                     "contribution": 0.3, "tiles": [2], "level": "above_typical"}}
_TUB_LOW = {"id": "F4", "kind": "concept", "data": {"concept": "tubules", "label": "Gland / tubule formation",
                                                    "contribution": 0.2, "tiles": [3], "level": "below_typical"}}


def test_defer_margin_never_reads_as_clear():
    assert _S.semantic_problem("Margins are clear.", [_MARGIN_DEFER])
    assert _S.semantic_problem("Defer to permanent sections; the read is unreliable.", [_MARGIN_DEFER]) is None


def test_tumor_at_margin_names_every_involved_margin():
    assert _S.semantic_problem("Tumor is suggested at the superior margin.", [_MARGIN_TUMOR])
    assert _S.semantic_problem("Tumor is suggested at the superior and medial margins.", [_MARGIN_TUMOR]) is None


def test_flagged_read_is_not_called_reliable():
    assert _S.semantic_problem("The read has high reliability.", [_FLAGGED])
    assert _S.semantic_problem("The read was flagged as unreliable.", [_FLAGGED]) is None


def test_concept_presence_follows_its_level_not_its_push():
    assert _S.semantic_problem("A dense lymphocytic infiltrate is present.", [_LYMPH_LOW])
    assert _S.semantic_problem("Scarce lymphocytes favour invasive carcinoma in the concept model.", [_LYMPH_LOW]) is None
    # the label's own wording ("Frequent", "Dense") is a name, not a claim; mixed levels may share a sentence
    assert _S.semantic_problem("Frequent mitotic figures and scarce gland or tubule formation are associated in the "
                               "concept model.",
                               [_MIT_HIGH, _TUB_LOW]) is None


def test_required_margin_and_quality_sentences_are_appended(cfg=None):
    from core import config
    cfg = config.load_settings()
    findings = [{"id": "F1", "kind": "prediction", "data": {"top": "Invasive", "p": 0.95, "set": ["Invasive"]}},
                _FLAGGED, _MARGIN_DEFER]
    kept = [{"text": "Invasive carcinoma is suggested.", "findings": ["F1"]}]
    out, added = _S.ensure_required(kept, findings, cfg)
    assert added == 2 and {"F6", "F7"} <= {f for s in out for f in s["findings"]}
    assert len(out) <= int(cfg["summary"]["max_sentences"])
    assert all(_S.semantic_problem(s["text"], [f for f in findings if f["id"] in s["findings"]]) is None for s in out)


def test_features_pushing_away_are_not_presented_as_support():
    away = {"id": "F4", "kind": "concept", "data": {"concept": "duct_confined", "label": "Atypical cells confined within ducts",
                                                    "contribution": -0.23, "tiles": [2], "level": "above_typical"}}
    assert _S.semantic_problem("Prominent atypical cells confined within ducts are associated with the finding.",
                               [away], aggressive_top=True)
    assert _S.semantic_problem("In the concept model, prominent atypical cells confined within ducts weigh against "
                               "invasion.", [away], aggressive_top=True) is None
    assert _S.semantic_problem("In the concept model, prominent atypical cells confined within ducts favour a "
                               "non-invasive process.",
                               [away], aggressive_top=True) is None


def test_margin_calls_are_hedged():
    assert _S.semantic_problem("Tumor is present at the superior and medial margins.", [_MARGIN_TUMOR])
    assert _S.semantic_problem("Tumor is suggested at the superior and medial margins.", [_MARGIN_TUMOR]) is None


# ---- clinical audit rules (2026-10-18: every sentence of the 328 demo summaries was audited) ---------------------------
def _c(fid, concept, label, contribution, level, tiles=(1,)):
    return {"id": fid, "kind": "concept", "data": {"concept": concept, "label": label, "contribution": contribution,
                                                   "tiles": list(tiles), "level": level}}


def _pred(top, members, fid="F1"):
    return {"id": fid, "kind": "prediction", "data": {"top": top, "p": 0.7, "set": list(members), "confidence": 0.9}}


_INV_LOW_AWAY = _c("F2", "invasion", "Tumor infiltrating stroma", -0.58, "below_typical")
_INV_HIGH_UP = _c("F2", "invasion", "Tumor infiltrating stroma", 0.66, "above_typical")
_TUB_HIGH_AWAY = _c("F3", "tubules", "Gland / tubule formation", -0.6, "above_typical")
_TUB_LOW_UP = _c("F3", "tubules", "Gland / tubule formation", 0.9, "below_typical")
_LYMPH_HIGH_AWAY = _c("F4", "lymphocytes", "Dense lymphocytic infiltrate (TILs)", -0.8, "above_typical")
_MIT_LOW_UP = _c("F5", "mitoses", "Frequent mitotic figures", 0.4, "below_typical")
_INV_TYPICAL = _c("F6", "invasion", "Tumor infiltrating stroma", -0.23, "typical")


@pytest.mark.parametrize("text,ok", [
    ("A benign lesion is suggested; in situ carcinoma cannot be excluded.", True),
    ("A benign lesion is suggested.", False),                                                   # omits a set class
    ("A benign lesion is suggested; in situ carcinoma and a benign lesion cannot be excluded.", False),   # top again
    ("A benign lesion is suggested; invasive carcinoma cannot be excluded.", False),            # not in the set
    ("A benign lesion; in situ carcinoma cannot be excluded.", False),                          # stated as fact
    ("In situ carcinoma cannot be excluded.", False),                                           # top not named
])
def test_prediction_sentence_names_the_set_exactly(text, ok):
    problem = _S.semantic_problem(text, [_pred("Benign", ["Benign", "InSitu"])], aggressive_top=False)
    assert (problem is None) == ok, problem


@pytest.mark.parametrize("text,cited,ok", [
    ("Scarce tumor infiltrating stroma pushes the concept model toward invasive carcinoma.", [_INV_LOW_AWAY], False),
    ("Scarce tumor infiltrating stroma pushes the concept model away from invasive carcinoma.", [_INV_LOW_AWAY], True),
    ("Scarce gland / tubule formation is associated with a lower likelihood of invasive carcinoma in the concept model.",
     [_TUB_LOW_UP], False),
    ("Scarce gland / tubule formation pushes the concept model toward invasive carcinoma.", [_TUB_LOW_UP], True),
    ("Prominent tumor infiltrating stroma pushes the concept model toward invasive carcinoma, while a prominent "
     "lymphocytic infiltrate pushes it away.", [_INV_HIGH_UP, _LYMPH_HIGH_AWAY], True),                    # per-clause attribution
    ("Prominent tumor infiltrating stroma pushes the concept model away from invasive carcinoma, while a prominent "
     "lymphocytic infiltrate pushes it toward invasive carcinoma.", [_INV_HIGH_UP, _LYMPH_HIGH_AWAY], False),  # crossed
    ("Prominent tumor infiltrating stroma pushes the concept model toward invasive carcinoma, while scarce gland / "
     "tubule formation is associated with a lower likelihood of invasive carcinoma.", [_INV_HIGH_UP, _TUB_LOW_UP], False),
    ("Prominent tumor infiltrating stroma pushes the read toward invasive carcinoma.", [_INV_HIGH_UP], False),  # the read
])
def test_concept_direction_follows_the_sign_of_the_contribution(text, cited, ok):
    problem = _S.semantic_problem(text, cited, aggressive_top=True)
    assert (problem is None) == ok, problem


@pytest.mark.parametrize("text,cited,aggressive_top,ok", [
    ("Prominent gland / tubule formation is associated with this result.", [_TUB_HIGH_AWAY], False, False),
    ("Prominent tumor infiltrating stroma is associated with this finding.", [_INV_HIGH_UP], False, False),
    ("Prominent tumor infiltrating stroma is the strongest associated feature.", [_INV_HIGH_UP], False, False),
    ("Scarce gland / tubule formation and frequent mitotic figures further support the result.", [_TUB_LOW_UP], False,
     False),
    ("Prominent tumor infiltrating stroma is associated with this category in the concept model.", [_INV_HIGH_UP], True,
     True),
    ("Prominent tumor infiltrating stroma is associated with this finding.", [_INV_HIGH_UP], True, False),  # vague
    ("Prominent gland / tubule formation is associated with a lower likelihood of invasive carcinoma in the concept "
     "model.", [_TUB_HIGH_AWAY], False, True),
])
def test_concept_support_is_never_retargeted_at_the_suggested_category(text, cited, aggressive_top, ok):
    problem = _S.semantic_problem(text, cited, aggressive_top=aggressive_top)
    assert (problem is None) == ok, problem


@pytest.mark.parametrize("text,cited,ok", [
    ("Tumor infiltrating stroma is present.", [_INV_TYPICAL], False),         # typical is neither present nor scarce
    ("Typical tumor infiltrating stroma pushes the concept model away from invasive carcinoma.", [_INV_TYPICAL], True),
    ("Scarce frequent mitotic figures push the concept model toward invasive carcinoma.", [_MIT_LOW_UP], False),
    ("Frequent mitotic figures are scarce.", [_MIT_LOW_UP], False),
    ("Scarce mitotic figures push the concept model toward invasive carcinoma.", [_MIT_LOW_UP], True),
])
def test_concept_level_wording(text, cited, ok):
    problem = _S.semantic_problem(text, cited, aggressive_top=False)
    assert (problem is None) == ok, problem


_RETR_LOW = {"id": "F5", "kind": "retrieval", "data": {"agree": 1, "of": 5, "label": "Benign"}}
_RETR_HIGH = {"id": "F5", "kind": "retrieval", "data": {"agree": 4, "of": 5, "label": "Benign"}}


@pytest.mark.parametrize("text,finding,ok", [
    ("The case is similar to benign reference cases.", _RETR_LOW, False),
    ("Similar reference cases favor a benign lesion.", _RETR_LOW, False),
    ("The case is similar to 1 of 5 reference cases.", _RETR_LOW, False),       # which category?
    ("The case is similar to 5 reference cases.", _RETR_LOW, False),            # the total quoted as agreeing
    ("1 of 5 most similar reference cases shares the suggested category.", _RETR_LOW, True),
    ("Only 1 of 5 similar reference cases is benign.", _RETR_LOW, True),
    ("The case is similar to 5 benign reference cases.", _RETR_HIGH, False),
    ("4 of 5 most similar reference cases share the suggested category.", _RETR_HIGH, True),
])
def test_retrieval_agreement_is_quoted_not_inflated(text, finding, ok):
    problem = _S.semantic_problem(text, [finding])
    assert (problem is None) == ok, problem


def test_margin_hedge_and_exact_margins():
    clear = {"id": "F7", "kind": "margin", "data": {"suggested_call": "clear", "involved": []}}
    assert _S.semantic_problem("Margins are clear and the read is consistent with similar cases.", [clear])
    assert _S.semantic_problem("Suggested margin call (simulated frozen section): margins clear.", [clear]) is None
    assert _S.semantic_problem("Suggested margin call: tumor at the superior, medial and lateral margins.",
                               [_MARGIN_TUMOR])                            # lateral is not involved


_NUC = {"id": "F7", "kind": "cells", "data": {"feature": "nuc_area_mean", "label": "Nuclear size", "value": 43.88,
                                              "unit": "um2", "ref_percentile": 0.0, "tiles": [1]}}
_NUC_CV = {"id": "F8", "kind": "cells", "data": {"feature": "nuc_area_cv", "label": "Variation in nuclear size",
                                                 "value": 0.57, "unit": "", "ref_percentile": 0.95, "tiles": [1]}}
_NEO = {"id": "F9", "kind": "cells", "data": {"feature": "neo_density", "label": "Tumor cell density", "value": 2471.9977,
                                              "unit": "per mm2", "ref_percentile": 0.96, "tiles": [1]}}


@pytest.mark.parametrize("text,cited,aggressive_top,ok", [
    ("Nuclear size is higher than in most benign reference cases.", [_NUC], False, False),
    ("No benign reference case has a lower nuclear size, while variation in nuclear size is higher than in 95% of "
     "benign reference cases.", [_NUC, _NUC_CV], False, True),            # "nuclear size" inside the longer label
    ("Nuclear size is 43.88.", [_NUC], False, False),                     # a bare value means nothing
    ("Tumor cell density is higher than in 96% of benign reference cases.", [_NEO], False, True),
    ("The case is supported by a high tumor cell density of 2472.", [_NEO], False, False),   # cells are not support
    ("A high tumor cell density of 2472 is in keeping with the suggestion.", [_NEO], True, True),
])
def test_cell_wording_follows_the_benign_reference_percentile(text, cited, aggressive_top, ok):
    problem = _S.semantic_problem(text, cited, aggressive_top=aggressive_top)
    assert (problem is None) == ok, problem


def test_numbers_are_rounded_for_reading():
    findings = [_pred("Benign", ["Benign"]), dict(_NEO, id="F2"),
                {"id": "F3", "kind": "cells", "data": {"feature": "dead_fraction", "label": "Dead cell fraction",
                                                       "value": 0.0245, "ref_percentile": 0.004, "tiles": [1]}}]
    for text, ids in (("Tumor cell density of 2471.9977 is higher than in 96% of benign reference cases.", ["F2"]),
                      ("Only 0% of benign reference cases have a lower dead cell fraction.", ["F3"])):
        kept, dropped = S.validate([sentence(text, *ids)], findings, CFG)
        assert kept == [], text


def test_required_prediction_first_and_strongest_concept_backfilled():
    findings = [_pred("Benign", ["Benign", "InSitu"]), _INV_HIGH_UP, _TUB_HIGH_AWAY, _RETR_HIGH, _MARGIN_DEFER,
                _FLAGGED]
    kept = [{"text": "4 of 5 most similar reference cases share the suggested category.", "findings": ["F5"]}]
    out, added = S.ensure_required(kept, findings, CFG)
    assert out[0]["findings"] == ["F1"] and out[1]["findings"] == ["F2"]          # strongest concept: invasion (+)
    assert out[1]["text"].startswith("Feature associated with invasive carcinoma in the concept model: prominent tumor")
    assert added == 4 and len(out) == CFG["summary"]["max_sentences"]
    assert {"F6", "F7"} <= {i for s in out for i in s["findings"]}
    assert S.validate(out, findings, CFG)[1] == []


def test_template_concept_phrases_follow_level():
    findings = [_pred("Invasive", ["Invasive"]), _TUB_LOW_UP, _MIT_LOW_UP, _LYMPH_HIGH_AWAY, _INV_TYPICAL]
    texts = [s["text"] for s in S.template(findings, CFG)]
    assert ("Features associated with invasive carcinoma in the concept model: scarce gland / tubule formation "
            "(region 1), then scarce mitotic figures.") in texts
    assert ("Features associated with a lower likelihood of invasive carcinoma in the concept model: dense lymphocytic "
            "infiltrate (TILs) and tumor infiltrating stroma (typical level).") in texts
    assert S.validate(S.template(findings, CFG), findings, CFG)[1] == []


# ---- AI spot check: a failed concept is never given as a reason (judge feedback T4b) ---------------------------------
def _rc(fid, concept, label, contribution, level, ok, n=20, tiles=(1,)):
    f = _c(fid, concept, label, contribution, level, tiles)
    f["data"]["spotcheck"] = None if ok is None else {"acceptable": ok, "reviewed": n}
    f["data"]["reliable"] = contract.spotcheck_passed(f["data"]["spotcheck"])
    return f


_MIT_FAIL = _rc("F2", "mitoses", "Frequent mitotic figures", 0.77, "above_typical", 0, tiles=(2, 5, 6))
_LOB_FAIL = _rc("F3", "normal_lobules", "Normal lobules with two-layered epithelium", -0.6, "above_typical", 0,
                tiles=(3, 7, 10))
_FAT_OK = _rc("F4", "benign_stroma_fat", "Benign fibrous stroma or fat", -0.48, "above_typical", 20, tiles=(7, 8, 11))
_INV_OK = _rc("F2", "invasion", "Tumor infiltrating stroma", 0.66, "above_typical", 19, tiles=(6, 8, 11))
_PLEO_OK = _rc("F3", "pleomorphism", "Marked nuclear pleomorphism", 0.64, "above_typical", 18)
_LYM_FAIL_UP = _rc("F4", "lymphocytes", "Dense lymphocytic infiltrate (TILs)", 0.5, "below_typical", 8, tiles=(11,))
_RETR = {"id": "F5", "kind": "retrieval", "data": {"agree": 5, "of": 5, "label": "Invasive"}}
_OK_Q = {"id": "F6", "kind": "quality", "data": {"reliability_flag": False, "qc_tiles_kept": 12, "min_tiles_kept": 3}}
_CELL = {"id": "F7", "kind": "cells", "data": {"feature": "neo_fraction", "label": "Tumor cell fraction", "value": 0.8,
                                               "unit": "", "ref_percentile": 1.0, "tiles": [1, 2, 5]}}
MIT_CAVEAT = ("The score for frequent mitotic figures was above typical in this case, but the feature failed its AI spot "
              "check (0/20 regions acceptable), so it is not given as a reason.")
OLD_MIT_CAVEAT = ("Frequent mitotic figures pushed this read toward invasive carcinoma but failed its AI spot check "
                  "(0/20 regions acceptable), so it is not given as a reason.")


def _cites(out, fid):
    return [s for s in out if fid in s["findings"]]


def test_failed_strongest_push_becomes_a_caveat_not_a_reason():
    # BACH_0087: mitoses (0/20) was the strongest push; nothing reliable pushes toward invasive carcinoma
    findings = [_pred("InSitu", ["Normal", "InSitu", "Invasive"]), _MIT_FAIL, _LOB_FAIL, _FAT_OK, _RETR,
                dict(_FLAGGED, id="F6"), _CELL]
    out = S.template(findings, CFG)
    texts = [s["text"] for s in out]
    assert not any(t.startswith(("Feature associated with invasive", "Features associated with invasive")) for t in texts)
    assert _cites(out, "F2") == [{"text": MIT_CAVEAT, "findings": ["F2"]}]
    assert _cites(out, "F3") == []                     # the lowering sentence was not written, so no caveat for it
    assert [s["findings"][0] for s in out] == ["F1", "F2", "F7", "F6"]   # caveat in the push slot (priority 2.5)
    assert S.validate(out, findings, CFG)[1] == []


def test_reliable_push_keeps_only_reliable_features_and_adds_one_caveat():
    # BACH_0084: invasion 19/20 and pleomorphism 18/20 pass; lymphocytes 8/20 fails
    findings = [_pred("Invasive", ["Invasive"]), _INV_OK, _PLEO_OK, _LYM_FAIL_UP, _RETR, _OK_Q, _CELL]
    out = S.template(findings, CFG)
    assert out[1] == {"text": "Features associated with invasive carcinoma in the concept model: prominent tumor "
                              "infiltrating stroma (regions 6, 8 and 11), then marked nuclear pleomorphism.",
                      "findings": ["F2", "F3"]}
    assert out[2] == {"text": "The score for lymphocytic infiltrate (TILs) was below typical in this case, but the "
                              "feature failed its AI spot check (8/20 regions acceptable), so it is not given as a "
                              "reason.",
                      "findings": ["F4"]}
    assert len(out) == CFG["summary"]["max_sentences"] and S.validate(out, findings, CFG)[1] == []


def test_failed_lowering_feature_is_named_only_when_the_lowering_sentence_is_written():
    findings = [_pred("Invasive", ["Invasive"]), _INV_OK, _LOB_FAIL, _FAT_OK]
    out = S.template(findings, CFG)
    assert [s["findings"] for s in out] == [["F1"], ["F2"], ["F4"], ["F3"]]
    assert out[2]["text"] == ("Feature associated with a lower likelihood of invasive carcinoma in the concept model: "
                              "prominent benign fibrous stroma or fat (regions 7, 8 and 11).")
    assert out[3]["text"] == ("The score for normal lobules with two-layered epithelium was above typical in this case, "
                              "but the feature failed its AI spot check (0/20 regions acceptable), so it is not given "
                              "as a reason.")
    assert S.validate(out, findings, CFG)[1] == []


def test_several_failed_features_and_no_spot_check_share_one_caveat():
    nec = _rc("F3", "necrosis", "Tumor necrosis", 0.37, "above_typical", 3)
    unreviewed = _rc("F4", "tubules", "Gland / tubule formation", 0.2, "below_typical", None)
    findings = [_pred("Invasive", ["InSitu", "Invasive"]), _MIT_FAIL, nec, unreviewed]
    out = S.template(findings, CFG)
    assert out[1] == {"text": "Scores were above typical for frequent mitotic figures (0/20 regions acceptable) and tumor "
                              "necrosis (3/20 regions acceptable), and below typical for gland / tubule formation (no AI "
                              "spot check) in this case, but these features did not pass an AI spot check, so they are "
                              "not given as reasons.",
                      "findings": ["F2", "F3", "F4"]}
    assert len(out) == 2 and S.validate(out, findings, CFG)[1] == []
    alone = S.template([_pred("Invasive", ["Invasive"]), unreviewed], CFG)[1]["text"]
    assert alone == ("The score for gland / tubule formation was below typical in this case, but the feature has no AI "
                     "spot check, so it is not given as a reason.")


@pytest.mark.parametrize("text,cited,ok", [
    ("Frequent mitotic figures push the read toward invasive carcinoma.", [_MIT_FAIL], False),
    ("Strongest push toward invasive carcinoma in this read: frequent mitotic figures (regions 2, 5 and 6).",
     [_MIT_FAIL], False),
    ("Frequent mitotic figures failed the AI spot check but push the read toward invasive carcinoma.", [_MIT_FAIL],
     False),                                                                 # names the check, still a reason
    (MIT_CAVEAT, [_MIT_FAIL], True),
    (OLD_MIT_CAVEAT, [_MIT_FAIL], False),                                   # a push on the read (BACH_0004)
    ("Prominent tumor infiltrating stroma pushes the read toward invasive carcinoma; frequent mitotic figures failed "
     "the AI spot check and are not given as a reason.", [_INV_OK, _MIT_FAIL], False),   # mixed with a reason
    ("Prominent tumor infiltrating stroma pushes the read toward invasive carcinoma.", [_INV_OK], False),  # not the read
    ("Prominent tumor infiltrating stroma is associated with invasive carcinoma in the concept model.", [_INV_OK], True),
])
def test_validator_rejects_failed_concepts_outside_the_caveat(text, cited, ok):
    problem = _S.semantic_problem(text, cited, aggressive_top=True)
    assert (problem is None) == ok, problem


def test_llm_sentence_citing_a_failed_concept_is_dropped_and_backfilled_with_the_caveat():
    findings = [_pred("InSitu", ["Normal", "InSitu", "Invasive"]), _MIT_FAIL, _LOB_FAIL, _FAT_OK, _RETR,
                dict(_FLAGGED, id="F6"), _CELL]
    llm = [sentence("In situ carcinoma is suggested; normal breast tissue and invasive carcinoma cannot be excluded.",
                    "F1"),
           sentence("Frequent mitotic figures push the read toward invasive carcinoma.", "F2")]
    kept, dropped = S.validate(llm, findings, CFG)
    assert [d["text"] for d in dropped] == [llm[1]["text"]] and "spot check" in dropped[0]["reason"]
    out, added = S.ensure_required(kept, findings, CFG)
    assert {"text": MIT_CAVEAT, "findings": ["F2"]} in out and added == 2      # quality concern + caveat backfill
    assert S.validate(out, findings, CFG)[1] == []


def test_findings_without_spot_check_keys_are_reasons():
    out = S.template([_pred("Invasive", ["Invasive"]), _INV_HIGH_UP], CFG)
    assert out[1]["text"].startswith("Feature associated with invasive carcinoma in the concept model: prominent tumor")


# ---- QA pass 2026-10-19: UI number formats, concept-model attribution, wording ------------------------------------------
@pytest.mark.parametrize("p,shown", [(0.6985, "70%"), (0.8048, "80%"), (0.9949, "99%"), (0.995, ">99%"),
                                     (0.9999, ">99%"), (0.0049, "<1%"), (0.005, "1%")])
def test_probabilities_are_quoted_as_the_ui_shows_them(p, shown):
    assert S._fmt_pct(p) == shown


def test_top_p_quotes_the_case_probability_and_must_match_the_finding():
    findings = [{"id": "F1", "kind": "prediction", "data": {"top": "Benign", "p": 0.805, "set": ["Benign"],
                                                            "confidence": 0.9}}]
    assert "(calibrated probability 81%)" in S.template(findings, CFG)[0]["text"]               # 3-dp finding
    out = S.template(findings, CFG, top_p=0.8048)                                              # the UI shows 80%
    assert "(calibrated probability 80%)" in out[0]["text"] and S.validate(out, findings, CFG)[1] == []
    with pytest.raises(ValueError, match="top_p"):
        S.template(findings, CFG, top_p=0.81)


@pytest.mark.parametrize("text,p,ok", [
    ("Suggested category: benign lesion (calibrated probability >99%).", 0.999, True),
    ("Suggested category: benign lesion (calibrated probability >99%).", 0.97, False),       # the bound must hold
    ("Suggested category: benign lesion (calibrated probability over 99%).", 0.999, True),
    ("Suggested category: benign lesion (calibrated probability 100%).", 0.999, False),     # reads as certain
    ("Suggested category: benign lesion (calibrated probability 1.00).", 0.999, False),
])
def test_bounded_and_certain_probabilities(text, p, ok):
    findings = [{"id": "F1", "kind": "prediction", "data": {"top": "Benign", "p": p, "set": ["Benign"]}}]
    kept, dropped = S.validate([sentence(text, "F1")], findings, CFG)
    assert (kept != []) == ok, dropped


@pytest.mark.parametrize("pct,expect", [
    (1.0, "No benign or normal reference case has a higher tumor cell fraction (measured in regions 1 and 2)."),
    (0.996, "No benign or normal reference case has a higher tumor cell fraction (measured in regions 1 and 2)."),
    (0.925, "Tumor cell fraction is higher than in 93% of benign or normal reference cases (measured in regions 1 "
            "and 2)."),                                                      # half up, as the UI's 93rd percentile
    (0.004, "No benign or normal reference case has a lower tumor cell fraction (measured in regions 1 and 2)."),
])
def test_cell_percentiles_follow_the_ui_and_never_read_100_percent(pct, expect):
    findings = [_pred("Normal", ["Normal"]),
                {"id": "F2", "kind": "cells", "data": {"feature": "neo_fraction", "label": "Tumor cell fraction",
                                                       "value": 0.5, "ref_percentile": pct, "tiles": [1, 2]}}]
    out = S.template(findings, CFG)
    assert expect in [s["text"] for s in out] and S.validate(out, findings, CFG)[1] == []


def test_plural_cell_label_reads_as_a_share():
    findings = [_pred("Invasive", ["Invasive"]),
                {"id": "F2", "kind": "cells", "data": {"feature": "til_proximity", "label": "Immune cells near tumor",
                                                       "value": 0.3, "ref_percentile": 0.92, "tiles": [0]}}]
    out = S.template(findings, CFG)
    assert out[1]["text"] == ("The share of immune cells near tumor is higher than in 92% of benign or normal "
                              "reference cases (measured in region 0).")
    assert S.validate(out, findings, CFG)[1] == []


@pytest.mark.parametrize("agree,expect", [
    (0, "None of the 5 most similar reference cases shares the suggested category (benign lesion)."),
    (1, "1 of the 5 most similar reference cases shares the suggested category (benign lesion)."),
    (3, "3 of the 5 most similar reference cases share the suggested category (benign lesion)."),
    (5, "All 5 most similar reference cases share the suggested category (benign lesion)."),
])
def test_retrieval_sentence_wording(agree, expect):
    findings = [_pred("Benign", ["Benign"]), {"id": "F5", "kind": "retrieval",
                                              "data": {"agree": agree, "of": 5, "label": "Benign"}}]
    out = S.template(findings, CFG)
    assert expect in [s["text"] for s in out] and S.validate(out, findings, CFG)[1] == []


def test_typical_level_merges_into_the_label_parentheses_and_flag_wording():
    findings = [_pred("Invasive", ["Invasive"]),
                _c("F2", "lymphocytes", "Dense lymphocytic infiltrate (TILs)", 0.4, "typical", tiles=(3,)),
                dict(_FLAGGED, id="F3")]
    texts = [s["text"] for s in S.template(findings, CFG)]
    assert ("Feature associated with invasive carcinoma in the concept model: lymphocytic infiltrate (TILs; typical "
            "level; region 3).") in texts
    assert ("The reliability check flagged this read (image quality or an unusual appearance), so its probabilities "
            "should be interpreted with caution.") in texts
    assert S.validate(S.template(findings, CFG), findings, CFG)[1] == []


def test_concept_wording_never_claims_a_push_on_the_read():
    # BACH_0004: suggested in situ carcinoma, P(invasive) < 1%; the concept model (not the classifier) puts mitoses
    # toward invasive carcinoma, and that feature failed its spot check
    for f in (_MIT_FAIL, _LOB_FAIL):
        for s in S.template([_pred("InSitu", ["Normal", "InSitu"]), f], CFG)[1:]:
            assert not re.search(r"\bpush|\bthis read\b|toward|away from|lower likelihood", s["text"]), s
    for f in (_INV_OK, _FAT_OK):
        for s in S.template([_pred("Invasive", ["Invasive"]), f], CFG)[1:]:
            assert "in the concept model" in s["text"] and not re.search(r"\bpush|\bthis read\b", s["text"]), s


CASES_DIR = config.REPO_ROOT / "artifacts" / "cases"


@pytest.mark.skipif(not CASES_DIR.is_dir(), reason="no committed case files")
def test_committed_case_summaries_are_the_current_template_and_valid():
    """Every committed case summary is today's template (regenerate with step 08 after a wording change) and passes
    the validator; its prediction sentence quotes the percent the UI shows."""
    paths = sorted(CASES_DIR.glob("*.json"))
    assert paths
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        pred = case["prediction"]
        top = next(f["data"]["top"] for f in case["findings"] if f["kind"] == "prediction")
        top_p = pred["probs"][pred["classes"].index(top)]
        if case["summary"]["generator"] == CFG["summary"]["template_generator"]:
            assert case["summary"]["sentences"] == S.template(case["findings"], CFG, top_p=top_p), path.name
        assert S.validate(case["summary"]["sentences"], case["findings"], CFG)[1] == [], path.name
        assert f"(calibrated probability {S._fmt_pct(top_p)})" in case["summary"]["sentences"][0]["text"], path.name
