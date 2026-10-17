"""eval/margin_eval.py on the eval fixture: benign FPR, composite per-segment test, echo statistics, WSI windows."""

from __future__ import annotations

import copy

import pytest

from core import config, io, schemas
from eval import margin_eval as ME
from tests import downstream_fixture as D

CFG = copy.deepcopy(config.load_settings())
CFG["bootstrap"]["n"] = 200


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("margin")
    info = D.write_eval_root(tmp / "data", CFG)
    out = ME.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_margin_eval"))
    return io.read_json(out), info, tmp


def test_envelope_and_threshold(result):
    m, _, _ = result
    assert m["name"] == "margin" and m["synthetic"] is True and m["threshold"] == 0.5
    assert all(k in m for k in schemas.EVAL_ENVELOPE)


def test_benign_fpr_counts_false_tumor_calls(result):
    m, _, tmp = result
    b = m["benign_fpr"]
    assert b["n_benign_test"] == 24 and b["n_scored"] == 24
    # fixture: benign test case i (0-based among all test cases) gets a hot superior margin when i % 11 == 0
    margins = io.read_json(schemas.margins_json(tmp / "data"))["cases"]
    cases = io.read_csv(schemas.cases_csv(tmp / "data"))
    benign = cases[(cases["split"] == "test") & cases["label"].isin(CFG["benign_classes"])]["case_id"]
    expected = sum(margins[c]["per_margin"]["superior"] > 0.5 for c in benign) / len(benign)
    assert b["per_margin"]["superior"]["estimate"] == pytest.approx(expected)
    assert b["any_margin"]["n"] == 24 and b["any_margin"]["lo"] <= b["any_margin"]["estimate"] <= b["any_margin"]["hi"]
    pooled = sum(margins[c]["per_margin"][m] > 0.5 for c in benign for m in schemas.MARGINS) / (4 * len(benign))
    assert b["pooled_per_margin"]["n"] == 96 and b["pooled_per_margin"]["estimate"] == pytest.approx(pooled)


def test_composite_exact_negatives_and_echo_field(result):
    m, info, _ = result
    c = m["composite"]
    assert c["n_composites"] == len(info["composites"])
    # 5 of 6 fields localize; the last one also lights its exact-negative margin (echo-like)
    assert c["exact_negative_margin_clear_rate"]["estimate"] == pytest.approx(5 / 6)
    assert c["tumor_side_margin_sensitivity"]["estimate"] == pytest.approx(1.0)
    assert c["n_expected"] == c["n_composites"] and c["n_missing"] == 0
    # 2048x1536 at 256 px splits on segment boundaries: no straddling squares; 12 of 24 border squares are benign-half
    # for left/right splits, 12 for top/bottom; corners counted once
    assert c["n_straddling_segments"] == 0
    assert c["benign_half_segment_clear_rate"]["n"] == 6 * 12
    assert 0 < c["carcinoma_half_segment_hit_rate"]["estimate"] <= 1


def test_echo_statistics(result):
    m, _, _ = result
    e = m["echo"]
    assert e["n_scored"] == 24 and 0 <= e["p_at_least_one_clear_margin"]["estimate"] <= 1
    assert e["p_at_least_one_clear_margin"]["estimate"] > 0.5             # fixture localizes (one hot margin)
    assert "estimate" in e["spearman_max_margin_vs_risk"]


def test_wsi_windows_sensitivity_specificity(result):
    m, info, _ = result
    w = m["wsi_windows"]
    assert w["n_windows"] == len(info["windows"]) and w["n_patients"] == 4
    assert 0 <= w["sensitivity"]["estimate"] <= 1 and w["specificity"]["estimate"] > 0.8
    assert set(w["per_margin"]) == set(schemas.MARGINS)


def test_missing_margins_is_an_error(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    schemas.margins_json(tmp_path / "data").unlink()
    assert ME.main(["--data-root", str(tmp_path / "data"), "--artifacts-root", str(tmp_path / "art")]) == 1


def test_segment_half_classification():
    from pipeline.composites import half_mask

    mask = half_mask(2048, 1536, "left")
    assert ME.segment_half({"x": 0, "y": 0, "size": 256}, mask) == "carcinoma"
    assert ME.segment_half({"x": 1024, "y": 0, "size": 256}, mask) == "benign"
    assert ME.segment_half({"x": 896, "y": 0, "size": 256}, mask) == "straddle"


def test_missing_composite_scores_fail_a_real_run(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    path = schemas.margins_json(tmp_path / "data")
    data = io.read_json(path)
    comp = [k for k in data["cases"] if k.startswith("COMP_")][0]
    del data["cases"][comp]
    io.write_json(path, data)
    synthetic = ME.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))
    assert io.read_json(synthetic)["incomplete"] == {"composite": 1}
    (tmp_path / "data" / "SYNTHETIC.json").unlink()
    with pytest.raises(ValueError, match="lacks scores"):
        ME.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))
