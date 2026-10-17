"""eval/metrics.py on the eval fixture: test split only, variants excluded, CIs everywhere, envelope + synthetic."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from core import config, io, schemas
from eval import metrics as M
from tests import downstream_fixture as D

CFG = copy.deepcopy(config.load_settings())
CFG["bootstrap"]["n"] = 200            # fast tests; the real run uses settings (2000)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("evalroot")
    info = D.write_eval_root(tmp / "data", CFG)
    ctx = D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_metrics")
    paths = M.run(ctx)
    return {name: io.read_json(p) for name, p in paths.items()}, info, tmp


def _is_ci(block):
    return {"estimate", "lo", "hi", "ci", "n", "n_groups", "n_boot", "n_valid"} <= set(block)


def test_envelopes_and_synthetic_flag(outputs):
    out, _, _ = outputs
    for name, obj in out.items():
        assert obj["name"] == name and obj["synthetic"] is True
        assert all(k in obj for k in schemas.EVAL_ENVELOPE)


def test_metrics_test_split_only_variants_excluded(outputs):
    out, info, _ = outputs
    m = out["metrics"]
    assert m["n"] == len(info["test"]) and m["variants_excluded"] == 1 and m["split"] == "test"
    assert m["n_groups"] < m["n"]
    for key in ("invasive_vs_rest_auroc", "macro_auroc", "macro_f1", "balanced_accuracy", "accuracy"):
        assert _is_ci(m["production"][key]) and m["production"][key]["lo"] <= m["production"][key]["hi"]
    assert m["production"]["invasive_vs_rest_auroc"]["estimate"] > 0.7      # planted signal


def test_a_vs_b_paired_difference(outputs):
    out, _, _ = outputs
    m = out["metrics"]
    assert m["models"]["a"] == "conch_probe_lr" and m["models"]["b"] == "cb_text"
    diff = m["a_vs_b"]["difference_a_minus_b"]
    assert _is_ci(diff)
    a, b = m["a_vs_b"]["auroc_a"]["estimate"], m["a_vs_b"]["auroc_b"]["estimate"]
    assert diff["estimate"] == pytest.approx(a - b)


def test_confusion_and_insitu_invasive_counts(outputs):
    out, info, _ = outputs
    conf = out["metrics"]["production"]["confusion"]
    assert sum(map(sum, conf["matrix"])) == len(info["test"])
    ii = conf["insitu_invasive"]
    i, v = CFG["classes"].index("InSitu"), CFG["classes"].index("Invasive")
    assert ii["insitu_called_invasive"] == conf["matrix"][i][v] and ii["invasive_called_insitu"] == conf["matrix"][v][i]


def test_calibration_before_after(outputs):
    out, _, _ = outputs
    c = out["calibration"]
    for side in ("before", "after"):
        for key in ("ece", "brier", "nll"):
            assert _is_ci(c[side][key])
        assert c[side]["reliability_diagram"]["bins"]
    assert c["after"]["nll"]["estimate"] < c["before"]["nll"]["estimate"]   # B was overconfident before temperature
    assert c["temperature"] > 1.0 and "biased" in c["notes"][0]


def test_conformal_methods_per_class_and_beta_note(outputs):
    out, _, _ = outputs
    c = out["conformal"]
    assert set(c["methods"]) == {"lac_marginal", "aps", "mondrian"}
    lac = c["methods"]["lac_marginal"]
    assert _is_ci(lac["coverage"]) and _is_ci(lac["mean_set_size"])
    assert set(lac["per_class"]) == set(CFG["classes"])
    fs = c["finite_sample"]
    assert fs["n_calibration"] == 40 and fs["k"] == 37 and fs["beta"] == [37, 4]
    assert 0.0 < fs["sd"] < 0.1


def test_beta_note_matches_the_spec_example():
    note = M.beta_note(50, 0.1)
    assert note["k"] == 46 and note["beta"] == [46, 5] and note["sd"] == pytest.approx(0.04, abs=0.005)


def test_missing_predictions_for_a_test_case_is_an_error(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    path = schemas.predictions_csv(tmp_path / "data")
    df = io.read_csv(path)
    io.write_csv(path, df[df["case_id"] != df[df["split"] == "test"]["case_id"].iloc[0]])
    with pytest.raises(ValueError, match="no predictions"):
        M.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))


def test_main_returns_nonzero_on_missing_input(tmp_path):
    assert M.main(["--data-root", str(tmp_path / "nothing"), "--artifacts-root", str(tmp_path / "art")]) == 1


def test_thunder_comparison(outputs):
    from eval import benchmark_thunder as T

    _, _, tmp = outputs
    T.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_benchmark_thunder"))
    th = io.read_json(schemas.eval_json(tmp / "art", "thunder"))
    assert th["label"].startswith("same split, different protocol") and th["synthetic"] is True
    assert th["published"]["linear_probe"] == {"conch": 84.4, "uni2h": 84.7, "musk": 69.9}
    assert th["published"]["zero_shot"] == {"conch": 56.1, "musk": 62.5}
    prod = th["ours"][0]
    assert prod["role"].startswith("production") and 0 <= prod["macro_f1"]["estimate"] <= 100
    assert "87%" in " ".join(th["notes"]) and "different, hidden test set" in " ".join(th["notes"])


def test_model_choice_naming_absent_models_is_an_error(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    io.write_json(schemas.eval_json(tmp_path / "art", "model_choice"), {"model_a": "uni2h_lr", "model_b": "cb_text"})
    with pytest.raises(ValueError, match="uni2h_lr"):
        M.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))


def test_model_choice_names_are_used(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    io.write_json(schemas.eval_json(tmp_path / "art", "model_choice"), {"model_a": "cb_text", "model_b": "conch_probe_lr"})
    out = io.read_json(M.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))["metrics"])
    assert out["models"]["a"] == "cb_text" and out["models"]["source"] == "eval/model_choice.json"


def test_calibration_variants_do_not_count_toward_n_calibration(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    path = schemas.predictions_csv(tmp_path / "data")
    df = io.read_csv(path)
    extra = df[df["split"] == "calibration"].head(3).copy()
    extra["case_id"] = [schemas.variant_case_id(c, "jpeg", 2) for c in extra["case_id"]]
    io.write_csv(path, pd.concat([df, extra]))
    out = M.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))
    assert io.read_json(out["conformal"])["finite_sample"]["n_calibration"] == 40
    assert io.read_json(out["metrics"])["variants_excluded"] == 4
