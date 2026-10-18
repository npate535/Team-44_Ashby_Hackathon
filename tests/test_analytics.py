"""eval/analytics.py on the eval fixture: test split only, curves agree with metrics.json, aggregates only."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from core import config, io, schemas, stats
from eval import analytics as A
from eval import metrics as M
from eval import queue_sim as Q
from tests import downstream_fixture as D

CFG = copy.deepcopy(config.load_settings())
CFG["bootstrap"]["n"] = 200            # fast tests; the real run uses settings (2000)
REPS = 25


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analytics")
    info = D.write_eval_root(tmp / "data", CFG)
    M.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_metrics"))
    Q.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_queue_sim"), reps=REPS)
    path = A.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_analytics"), reps=REPS)
    metrics = io.read_json(schemas.eval_json(tmp / "art", "metrics"))
    return io.read_json(path), metrics, info, tmp, path


def test_envelope_population_and_registration(run):
    a, m, info, tmp, path = run
    assert path == schemas.eval_json(tmp / "art", "analytics") and "analytics" in schemas.EVAL_FILES
    assert all(k in a for k in schemas.EVAL_ENVELOPE) and a["name"] == "analytics" and a["synthetic"] is True
    assert a["split"] == "test" and a["n"] == len(info["test"]) == m["n"]
    assert a["variants_excluded"] == m["variants_excluded"] == 1            # same exclusion as eval/metrics.py
    assert sum(a["n_by_class"].values()) == a["n"] and a["n_groups"] == m["n_groups"]
    assert a["classes"] == CFG["classes"] and a["aggressive_class"] == CFG["aggressive_class"]
    assert a["alpha"] == CFG["alpha"] and a["unavailable"] == {}


def test_roc_auroc_matches_metrics_and_the_stored_points(run):
    a, m, _, _, _ = run
    curves = a["roc"]["invasive_vs_rest"]
    assert set(curves) == {"production", *m["per_model"]}
    for name, c in curves.items():
        ref = m["production"] if name == "production" else m["per_model"][name]
        assert c["auroc"] == pytest.approx(ref["invasive_vs_rest_auroc"]["estimate"])
        assert c["auroc_trapezoid"] == pytest.approx(c["auroc"], abs=1e-9)       # the plotted points carry the area
        area = float(np.sum(np.diff(c["fpr"]) * (np.array(c["tpr"][1:]) + np.array(c["tpr"][:-1])) / 2))
        assert area == pytest.approx(c["auroc"], abs=2e-3)                       # after rounding to 4 decimals
        assert c["fpr"][0] == 0 and c["tpr"][-1] == 1 and c["threshold"][0] is None
        assert c["fpr"] == sorted(c["fpr"]) and len(c["threshold"]) == len(c["fpr"])
    ovr = a["roc"]["production_one_vs_rest"]
    assert list(ovr) == CFG["classes"]
    assert np.mean([c["auroc"] for c in ovr.values()]) == pytest.approx(m["production"]["macro_auroc"]["estimate"])


def test_pr_confusion_and_histogram_counts(run):
    a, m, info, _, _ = run
    pr = a["pr"]["models"]["production"]
    assert pr["n_pos"] == a["n_by_class"][CFG["aggressive_class"]]
    assert 0 < pr["average_precision"] <= 1 and pr["recall"] == sorted(pr["recall"])
    conf = a["confusion"]
    assert conf["counts"] == m["production"]["confusion"]["matrix"] and conf["n"] == a["n"]
    for row, total, norm in zip(conf["counts"], conf["row_totals"], conf["row_normalized"]):
        assert sum(row) == total and sum(norm) == pytest.approx(1.0)
    hist = a["risk_histogram"]
    assert len(hist["bin_edges"]) == A.RISK_HIST_BINS + 1
    assert {c: sum(v) for c, v in hist["by_class"].items()} == a["n_by_class"]


def test_reliability_bins_are_equal_mass_and_match_calibration(run):
    a, _, _, tmp, _ = run
    cal = io.read_json(schemas.eval_json(tmp / "art", "calibration"))
    rel = a["reliability"]
    assert rel["n_bins"] == CFG["calibration"]["ece_bins"]
    for side in ("before", "after"):
        assert rel["top_label"][side]["bins"] == cal[side]["reliability_diagram"]["bins"]
        counts = [b["count"] for b in rel["aggressive"][side]["bins"]]
        assert sum(counts) == a["n"] and max(counts) - min(counts) <= 1
        assert all(0 <= b["observed"] <= 1 and b["lo"] <= b["mean_predicted"] <= b["hi"]
                   for b in rel["aggressive"][side]["bins"])


def test_conformal_counts_agree_with_conformal_json(run):
    a, _, _, tmp, _ = run
    conf = io.read_json(schemas.eval_json(tmp / "art", "conformal"))
    for name, block in a["conformal"]["methods"].items():
        ref = conf["methods"][name]
        assert block["coverage"] == pytest.approx(ref["coverage"]["estimate"])
        assert block["mean_set_size"] == pytest.approx(ref["mean_set_size"]["estimate"])
        assert block["set_size"] == ref["set_size_histogram"]
        assert sum(x["count"] for x in block["composition"]) == a["n"]
        for cls, pc in block["per_class"].items():
            assert pc["coverage"] == pytest.approx(ref["per_class"][cls]["estimate"])
            assert sum(pc["set_size"].values()) == pc["n"] == a["n_by_class"][cls]
    assert a["conformal"]["target_coverage"] == pytest.approx(1 - CFG["alpha"])


def test_triage_and_queue_distributions(run):
    a, _, _, tmp, _ = run
    qs = io.read_json(schemas.eval_json(tmp / "art", "queue_sim"))
    tri = a["triage"]
    assert {t: sum(by[t] for by in tri["tiers_by_class"].values()) for t in "1234"} == qs["test_case_tiers"]
    assert tri["aggressive_not_flagged"] == qs["false_negatives_in_pool"]
    queue = a["queue"]
    assert queue["reps"] == REPS and queue["consistency"]["matches_queue_sim"] is True
    tiers = queue["distributions"]["tiers"]["aggressive_mean"]
    assert tiers["n_days"] == REPS and tiers["p2_5"] <= tiers["p50"] <= tiers["p97_5"]
    diff = queue["vs_fifo"]["tiers"]["non_flagged_mean"]
    assert diff["mean"] == pytest.approx(qs["results"]["tiers"]["vs_fifo"]["non_flagged_mean"]["mean"])
    assert 0 <= diff["share_days_higher"] <= 1


def test_aggregates_only_no_case_ids_or_label_keys(run):
    a, _, info, _, path = run
    text = path.read_text()
    assert not any(cid in text for cid in info["test"])
    assert "true_label" not in text and "eval_only" not in text and "case_id" not in text
    json.loads(text)                                                          # strict JSON (no NaN)


def test_missing_optional_inputs_degrade_to_unavailable(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    schemas.reliability_csv(tmp_path / "data").unlink()
    a = io.read_json(A.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"), reps=5))
    assert {"triage", "queue"} <= set(a["unavailable"]) and "roc" in a
    assert "reliability.csv" not in a["sources"]["data"]


def test_main_returns_nonzero_on_missing_input(tmp_path):
    assert A.main(["--data-root", str(tmp_path / "nothing"), "--artifacts-root", str(tmp_path / "art")]) == 1


def test_roc_curve_single_class_is_empty():
    out = A.roc_curve(np.zeros(5, dtype=int), np.linspace(0, 1, 5))
    assert out["fpr"] == [] and out["auroc"] is None
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    assert A.roc_curve(y, s)["auroc"] == pytest.approx(stats.auroc(y, s)) == pytest.approx(0.75)


def test_api_metrics_serves_analytics(run, tmp_path):
    from tests.app_helpers import dev_root, find_keys, make_client

    a, _, _, _, path = run
    root = dev_root(tmp_path)
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "analytics.json").write_text(path.read_text())
    client, _ = make_client(tmp_path, artifacts_root=root)
    body = client.get("/api/metrics").json()
    assert body["eval"]["analytics"]["roc"]["invasive_vs_rest"]["production"]["auroc"] == pytest.approx(
        a["roc"]["invasive_vs_rest"]["production"]["auroc"])
    assert "analytics" in body["expected"] and find_keys(body) == []


def test_tier_set_drives_conformal_block_and_post_hoc_flag(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["conformal"] = {**cfg["conformal"], "method": "aps", "mode": "marginal", "nonempty": True}
    D.write_eval_root(tmp_path / "data", cfg)
    path = schemas.predictions_csv(tmp_path / "data")
    df = io.read_csv(path)
    classes = cfg["classes"]
    tier = []
    for _, row in df.iterrows():                  # APS set with the argmax added when empty (core.conformal.nonempty_sets)
        members = schemas.parse_set(row["set_aps"]) or [row["top"]]
        tier.append(schemas.format_set(members, classes))
    df["set_tier"] = tier
    io.write_csv(path, df)
    a = io.read_json(A.run(D.make_ctx(cfg, tmp_path / "data", tmp_path / "art"), reps=5))
    conf = a["conformal"]
    assert conf["drives_tiers"] == "aps_marginal_nonempty" == conf["tier_method_key"]
    assert conf["pre_registered"] == "lac_marginal" and conf["post_hoc_switch"] is True
    assert "after viewing test results" in conf["selection_note"]
    assert {"lac_marginal", "aps", "mondrian", "aps_marginal_nonempty"} == set(conf["methods"])
    assert conf["methods"]["aps_marginal_nonempty"]["empty_sets"] == 0



def test_pr_points_walk_in_decreasing_threshold_order():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    pr = A.pr_curve(y, s)
    assert pr["recall"] == sorted(pr["recall"])
    thr = [t for t in pr["threshold"] if t is not None]
    assert thr == sorted(thr, reverse=True)                      # one walk from strict to lenient, ties included
    # at recall 1/3: precision 1.0 (threshold 0.9) comes before 0.5 (threshold 0.8)
    third = [(p, t) for r, p, t in zip(pr["recall"], pr["precision"], pr["threshold"]) if r == round(1 / 3, 4)]
    assert third == [(1.0, 0.9), (0.5, 0.8)]
