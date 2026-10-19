"""backend/interpret: cache lookup across generating models, and the second-model agreement wording."""

from __future__ import annotations

import json

from backend import interpret as I


def _case():
    return {"case_id": "X1", "prediction": {"classes": ["Normal", "Benign", "InSitu", "Invasive"],
                                            "probs": [0.1, 0.1, 0.2, 0.6], "conformal_set": ["Invasive"]},
            "quality": {"reliability_flag": False}, "agreement": {"conch_musk_same_top": True}, "concepts": []}


def test_cached_text_from_another_model_is_served(tmp_path, monkeypatch):
    monkeypatch.setenv(I.ENV_MODEL, "serving-model")
    monkeypatch.setenv(I.ENV_CACHE_MODELS, "batch-model")
    interp = I.Interpreter(tmp_path)
    case = _case()
    assert interp.case(case)["status"] == "none"                       # nothing cached yet

    # what a batch run under the other model writes: same payload, key built with that model
    evidence = I.case_evidence(case)
    payload = (json.dumps(evidence, indent=1, sort_keys=True, ensure_ascii=False) + "\n\nVALIDATION JSON:\n"
               + json.dumps({}, indent=1, sort_keys=True, ensure_ascii=False))
    key = I._cache_key("case", payload, "batch-model")
    path = interp._path("case", "X1", key)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "ok", "model": "batch-model", "markdown": "## Bottom line"}))

    got = interp.case(case)
    assert got["status"] == "ok" and got["cached"] is True and got["model"] == "batch-model"


def test_serving_model_is_tried_first(monkeypatch):
    monkeypatch.setenv(I.ENV_CACHE_MODELS, "b, a")
    assert I._cache_models("a") == ["a", "b"]
    monkeypatch.delenv(I.ENV_CACHE_MODELS)
    assert I._cache_models("a") == ["a", *I.DEFAULT_CACHE_MODELS]


def test_agreement_is_not_described_as_conch_vs_musk():
    md = ("## Bottom line\n**Suggests Invasive (60%)** · Confident · Careful review\n\n"
          "| Signal | Value | Reading |\n|---|---|---|\n| CONCH–MUSK agreement | yes | models agree |\n")
    problems = I.check_claims(md, _case(), {})
    assert any("CONCH is not part" in p for p in problems)
    ok = md.replace("CONCH–MUSK agreement", "Second-model (MUSK) agreement")
    assert not any("CONCH is not part" in p for p in I.check_claims(ok, _case(), {}))


def test_prompt_describes_the_real_checks():
    assert "CONCH is not part of this check" in I.CASE_INSTRUCTIONS
    assert "separate concept model" in I.CASE_INSTRUCTIONS
    assert "CONCH–MUSK agreement" not in I.CASE_INSTRUCTIONS


def test_serve_only_mode_never_starts_a_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv(I.ENV_ON_DEMAND, "0")
    called = []
    monkeypatch.setattr(I, "_run", lambda *a, **k: called.append(a) or ("x", {}))
    out = I.Interpreter(tmp_path).case(_case(), generate=True, force=True)
    assert out == {"status": "none", "model": I._model(), "on_demand": False} and not called
