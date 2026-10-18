"""Conformal coverage across random patient-grouped re-splits (docs/VALIDATION_ROUND3.md §1, Task B1)."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from core import config, io, schemas
from eval import conformal_resplit as R
from eval import metrics as M
from tests import downstream_fixture as D

CFG = copy.deepcopy(config.load_settings())
CFG["bootstrap"]["n"] = 50


def _groups(sizes):
    return np.concatenate([np.full(s, f"g{i}") for i, s in enumerate(sizes)])


def test_groups_never_straddle():
    rng = np.random.default_rng(0)
    groups = _groups([36, 18, 14, 11, 9] + list(rng.integers(1, 6, size=30)))
    rng.shuffle(groups)
    n = len(groups)
    count = 0
    for cal, test in R.grouped_resplits(groups, 50, 200, 1):
        count += 1
        assert len(set(groups[cal]) & set(groups[test])) == 0
        assert len(np.intersect1d(cal, test)) == 0
        assert np.array_equal(np.sort(np.concatenate([cal, test])), np.arange(n))
    assert count == 200


def test_calibration_reaches_target():
    groups = _groups([36, 18, 14, 11, 9, 5, 5, 5, 5, 5, 4, 3, 3, 2, 1, 1, 1])
    sizes = {g: int((groups == g).sum()) for g in np.unique(groups)}
    overshoot = False
    for cal, test in R.grouped_resplits(groups, 50, 300, 3):
        assert len(cal) >= 50 and len(test) > 0
        overshoot |= len(cal) > 50
    assert overshoot                                       # a 36-image group must be able to push past 50
    # stops at the first group that reaches the target: calibration = the shuffled groups up to and including it
    for cal, _test, order in R._resplit_orders(groups, 50, 50, 3):
        counts = np.cumsum([sizes[g] for g in order])
        first = int(np.argmax(counts >= 50))
        assert set(groups[cal]) == set(order[: first + 1])


def test_deterministic_with_seed():
    groups = _groups([5] * 40)
    a = [(c.tolist(), t.tolist()) for c, t in R.grouped_resplits(groups, 50, 20, 20261018)]
    b = [(c.tolist(), t.tolist()) for c, t in R.grouped_resplits(groups, 50, 20, 20261018)]
    c = [(c.tolist(), t.tolist()) for c, t in R.grouped_resplits(groups, 50, 20, 7)]
    assert a == b and a != c
    probs, labels = _dirichlet(200, 0)
    cal, test = next(iter(R.grouped_resplits(_groups([2] * 100), 50, 1, 1)))
    s1 = R.split_stats(probs, labels, cal, test, 0.1, 3, 11)
    s2 = R.split_stats(probs, labels, cal, test, 0.1, 3, 11)
    assert s1 == s2


def _dirichlet(n, seed):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(4, 0.7), size=n)
    labels = np.array([rng.choice(4, p=p) for p in probs])           # calibrated by construction: exchangeable
    return probs, labels


def test_coverage_near_target_on_exchangeable_data():
    probs, labels = _dirichlet(400, 5)
    groups = _groups([2] * 200)
    out = R.resplit_summary(probs, labels, groups, n_cal=50, n_splits=200, seed=9, alpha=0.1, target=3,
                            methods=R.method_names(CFG), cfg=CFG)
    for m in ("aps", "lac_marginal"):
        assert abs(out[m]["coverage"]["mean"] - 0.9) <= 0.03, (m, out[m]["coverage"])
        q = out[m]["coverage"]
        assert q["p2_5"] <= q["p50"] <= q["p97_5"]
    assert out["aps_marginal_nonempty"]["empty_share"] == 0.0
    assert out["aps_marginal_nonempty"]["coverage"]["mean"] >= out["aps"]["coverage"]["mean"]


def test_mondrian_sparse_class_is_reported():
    probs, labels = _dirichlet(120, 2)
    labels[:] = np.where(labels == 3, 2, labels)                  # no Invasive anywhere ...
    labels[0], labels[1] = 3, 3                                    # ... except two calibration images
    cal = np.arange(0, 50)
    test = np.arange(50, 120)
    s = R.split_stats(probs, labels, cal, test, 0.1, 3, 4)
    assert set(s) == {"lac_marginal", "aps", "mondrian", "aps_marginal_nonempty"}
    mond = s["mondrian"]
    assert mond["n_unbounded_classes"] >= 1                         # Invasive: 2 calibration images -> q-hat = inf
    assert mond["invasive_in_set_given_invasive"] is None           # no invasive test image: undefined, not 0
    assert mond["invasive_in_set_given_not_invasive"] == 1.0       # unbounded class is in every set
    for m in s.values():
        for v in m.values():
            assert v is None or math.isfinite(v)
    # aggregation keeps the undefined statistic null instead of crashing
    agg = R.aggregate([s, s], 0.9)
    assert agg["mondrian"]["invasive_in_set_given_invasive"] is None
    assert agg["mondrian"]["undefined_splits"]["invasive_in_set_given_invasive"] == 2
    assert agg["mondrian"]["unbounded_class_share"] == 1.0


def test_statistics_match_their_definitions():
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])
    labels = np.array([0, 3, 1])
    sets = np.array([[True, False, False, False], [False, False, True, True], [False, False, False, False]])
    st = R.set_stats(sets, labels, 3)
    assert st == {"coverage": pytest.approx(2 / 3), "mean_set_size": 1.0, "empty_share": pytest.approx(1 / 3),
                  "invasive_in_set_given_invasive": 1.0, "invasive_in_set_given_not_invasive": 0.0}


def test_run_on_fixture_writes_valid_json_matching_conformal(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    ctx = D.make_ctx(CFG, tmp_path / "data", tmp_path / "art", step="eval_conformal_resplit")
    M.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art", step="eval_metrics"))
    path = R.run(ctx, n_splits=30)
    assert path == schemas.eval_json(tmp_path / "art", "conformal_resplit")
    doc = io.read_json(path)
    json.dumps(doc, allow_nan=False)
    for k in ("name", "created_utc", "config_hash", "git_sha", "synthetic", "n_images", "n_groups", "n_splits",
              "n_cal_target", "alpha", "seed", "model", "methods", "original_split"):
        assert k in doc, k
    assert doc["name"] == "conformal_resplit" and doc["synthetic"] is True and doc["n_splits"] == 30
    assert doc["model"] == "cb_text" and doc["alpha"] == 0.1 and doc["n_cal_target"] == 50
    conf = io.read_json(schemas.eval_json(tmp_path / "art", "conformal"))
    for m in conf["methods"]:
        assert m in doc["methods"] and m in doc["original_split"]
        assert doc["original_split"][m]["coverage"] == pytest.approx(conf["methods"][m]["coverage"]["estimate"])
        b = doc["methods"][m]
        assert set(b["coverage"]) == {"mean", "p2_5", "p50", "p97_5"}
        for k in ("share_at_or_above_target", "mean_set_size", "empty_share", "invasive_in_set_given_invasive",
                  "invasive_in_set_given_not_invasive"):
            assert k in b
    assert doc["original_split_check"]["matches_conformal_json"] is True


def test_results_md_section_is_built_from_the_json(tmp_path):
    from eval import results_md

    D.write_eval_root(tmp_path / "data", CFG)
    M.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art", step="eval_metrics"))
    doc = io.read_json(R.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"), n_splits=25))
    text = "\n".join(results_md.section_conformal_resplit(doc))
    assert text.startswith("## Conformal coverage across 25 random re-splits")
    assert "eval/conformal_resplit.json" in text and "docs/VALIDATION_ROUND3.md" in text
    for m, b in doc["methods"].items():
        assert f"| {m} |" in text
        assert f"{100 * b['coverage']['p50']:.1f}%" in text
        assert f"{100 * doc['original_split'][m]['coverage']:.1f}%" in text
    assert "NaN" not in text and "| nan" not in text and "None" not in text
    out = results_md.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"))
    assert "## Conformal coverage across 25 random re-splits" in out.read_text(encoding="utf-8")
