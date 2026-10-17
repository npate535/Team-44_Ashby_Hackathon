"""backend/summary.py: validator rules, template determinism, LLM path with mocked client, cache invalidation."""

from __future__ import annotations

import copy
import json

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
    sentence("Suggested category: invasive carcinoma (calibrated probability 0.72).", "F1"),
    sentence("Suggested category: invasive carcinoma (calibrated probability 72%).", "F1"),
    sentence("Suggested category: invasive carcinoma (probability 0.7).", "F1"),            # rounding of 0.72
    sentence("Tumor infiltrating stroma in regions 5 and 6 pushes the read up by 0.44.", "F2"),
    sentence("Variation in nuclear size is higher than in 95% of benign reference cases (regions 5 and 6).", "F6"),
    sentence("4 of 5 similar reference cases share the suggested category.", "F4"),
    sentence("Ki-67, CK5/6 and p63 are named in F1 but are not numbers.", "F1"),
])
def test_validator_keeps_grounded_sentences(good):
    kept, dropped = S.validate([good], FINDINGS, CFG)
    assert dropped == [] and kept == [good]


def test_signs_are_ignored_but_uncited_values_are_not():
    assert S.validate([sentence("It pushes the read up (-0.44).", "F2")], FINDINGS, CFG)[1] == []
    assert S.validate([sentence("Duct confinement pushes it down (-0.12).", "F2")], FINDINGS, CFG)[0] == []


def test_numbers_across_multiple_cited_findings_are_pooled():
    s = sentence("Probability 0.72; regions 5 and 6.", "F1", "F2")
    assert S.validate([s], FINDINGS, CFG)[0] == [s]


def test_percent_and_decimal_forms_are_consistent():
    assert S.numbers_in_text("72% and 0.72 and 3,150 and F12 and Ki-67") == [(72.0, 0, True), (0.72, 2, False),
                                                                             (3150.0, 0, False)]
    assert S.number_supported(72, 0, True, [0.72])
    assert S.number_supported(72, 0, True, [72.0])
    assert not S.number_supported(72, 0, False, [0.72])        # "72" without % is not 0.72
    assert S.number_supported(0.7, 1, False, [0.72])
    assert not S.number_supported(0.70, 2, False, [0.72])
    assert S.number_supported(3150, 0, False, [3150.0])


def test_validator_caps_sentences_and_dedupes_citations():
    many = [sentence("Suggested category: invasive carcinoma.", "F1", "F1")] * 6
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
    assert texts[0].startswith("Suggested category: invasive carcinoma (calibrated probability 0.72)")
    assert "in situ carcinoma is also in the confidence set" in texts[0]
    assert any("most associated with invasive carcinoma" in t and "regions 5 and 6" in t for t in texts)
    assert any("higher than in 95% of benign reference cases (regions 5 and 6)" in t for t in texts)


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
    assert expect in out[0]["text"] and "90% confidence set" in out[0]["text"] and "0.51" in out[0]["text"]
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
    assert "Only 4% of benign reference cases have a lower tumor cell density (region 3)" in texts
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
    assert "Normal two-layered lobules (region 3) is a feature associated with a lower likelihood of invasive carcinoma." in texts


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
    sentence("Tumor infiltrating stroma in regions 5 and 6 is the feature most associated with the suggestion.", "F2"),
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
