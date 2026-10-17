"""pipeline/06_reliability.py on the synthetic root after 05 (ruling M9 like-for-like reads)."""

import importlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from core import config, schemas
from tests.a3_synthetic import after_05, clone_root

step06 = importlib.import_module("pipeline.06_reliability")
CFG = config.load_settings()


@pytest.fixture(scope="module")
def run06(tmp_path_factory):
    root0, art0, cfg_path = after_05()
    work = tmp_path_factory.mktemp("r06")
    root, art = work / "data", work / "art"
    clone_root(root0, root)
    shutil.copytree(art0, art)
    assert step06.main(["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]) == 0
    return root, art, config.load_settings(cfg_path)


def test_unit_helpers():
    assert step06.recall_threshold(np.array([0.9, 0.8, 0.1, 0.7, 0.2]), np.array([1, 1, 0, 1, 1]), 0.8) == 0.2
    assert step06.recall_threshold(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), np.array([1, 1, 1, 1, 1]), 0.6) == 0.7
    np.testing.assert_allclose(step06.percentile(np.array([0.0, 2.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0])),
                               [0.0, 0.5, 1.0])
    ref = np.eye(3)
    d = step06.knn_distance(np.array([[1.0, 0, 0]]), ref, 1)
    assert d[0] == pytest.approx(0.0)


def test_reliability_csv_rows_and_schema(run06):
    root, _, cfg = run06
    rel = pd.read_csv(schemas.reliability_csv(root), keep_default_na=False, na_values=[""])
    schemas.validate_frame(schemas.RELIABILITY, rel)
    pred = pd.read_csv(schemas.predictions_csv(root), keep_default_na=False, na_values=[""])
    assert set(rel.case_id) == set(pred[pred.split != "train"].case_id)
    assert rel.ood_percentile.between(0, 1).all() and rel.p_error.between(0, 1).all()
    np.testing.assert_allclose(rel.reliability, 1 - rel.p_error)
    assert ((rel.p_error >= rel.threshold) == (rel.reliability_flag == 1)).all()
    assert set(rel.variant) >= {"clean", "composite"} | {f"{t}{s}" for t in cfg["degrade"]["types"]
                                                         for s in cfg["degrade"]["severities"]}


def test_degradation_raises_ood_and_flags(run06):
    root, _, _ = run06
    rel = pd.read_csv(schemas.reliability_csv(root), keep_default_na=False, na_values=[""])
    rel["sev"] = rel.variant.str.extract(r"(\d)$").fillna(0).astype(int)
    by = rel[rel.variant != "composite"].groupby("sev")
    assert by.ood_distance.mean().is_monotonic_increasing
    assert by.reliability_flag.mean()[3] > by.reliability_flag.mean()[0]


def test_eval_json(run06):
    _, art, cfg = run06
    doc = json.loads(schemas.eval_json(art, "reliability").read_text())
    for key in ("name", "created_utc", "config_hash", "git_sha", "synthetic"):
        assert key in doc
    assert "No coverage claim" in doc["coverage_note"]
    a = doc["error_auroc_test"]
    assert a["lo"] <= a["estimate"] <= a["hi"] and a["estimate"] > 0.6 and a["n_groups"] > 10
    curve = doc["accuracy_vs_degradation"]["curve"]
    for t in cfg["degrade"]["types"]:
        assert set(curve[t]) == {"0", "1", "2", "3"}
        assert curve[t]["3"]["estimate"] < curve[t]["0"]["estimate"]
    assert doc["error_recall_calibration"] >= cfg["reliability"]["target_error_recall"]
    if doc["accuracy_vs_degradation"]["clean_read"] != "view0":     # concept production without view-0 features
        assert doc["model_a_curve"]["clean_read"] == "view0"
