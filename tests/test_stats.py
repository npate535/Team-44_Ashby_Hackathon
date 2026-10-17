"""core.stats: cluster/stratified bootstrap CIs and classification metrics (plan §12, BUILD_PLAN §A5)."""

import math

import numpy as np
import pytest
from sklearn import metrics as skm

from core import stats
from core.config import load_settings

CFG = load_settings()
B = CFG["bootstrap"]


# ---- metrics vs scikit-learn ---------------------------------------------------------------------------
def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    s = np.round(rng.normal(size=300) + y, 1)              # rounding creates ties
    assert stats.auroc(y, s) == pytest.approx(skm.roc_auc_score(y, s))


def test_auroc_ties_count_half():
    assert stats.auroc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert stats.auroc(np.array([0, 0, 1]), np.array([0.1, 0.5, 0.5])) == pytest.approx(0.75)


def test_auroc_nan_on_single_class_and_bool_input():
    assert math.isnan(stats.auroc(np.ones(5, dtype=int), np.arange(5.0)))
    assert stats.auroc(np.array([False, True]), np.array([0.1, 0.9])) == 1.0


def test_auroc_rejects_non_binary_and_mismatch():
    with pytest.raises(ValueError):
        stats.auroc(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        stats.auroc(np.array([0, 1]), np.array([0.1]))


def test_auroc_ovr_and_macro_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, size=400)
    probs = rng.dirichlet(np.ones(4), size=400) + 0.5 * np.eye(4)[y]
    probs /= probs.sum(1, keepdims=True)
    ovr = stats.auroc_ovr(y, probs, 4)
    for k in range(4):
        assert ovr[k] == pytest.approx(skm.roc_auc_score(y == k, probs[:, k]))
    assert stats.macro_auroc(y, probs, 4) == pytest.approx(skm.roc_auc_score(y, probs, multi_class="ovr", average="macro"))


def test_auroc_ovr_absent_class_nan_and_macro_ignores_it():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])
    ovr = stats.auroc_ovr(y, probs, 3)
    assert math.isnan(ovr[2])
    assert stats.macro_auroc(y, probs, 3) == pytest.approx(np.mean(ovr[:2]))


def test_macro_f1_matches_sklearn_including_absent_classes():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, size=200)
    pred = np.where(rng.uniform(size=200) < 0.7, y, rng.integers(0, 4, size=200))
    assert stats.macro_f1(y, pred, 4) == pytest.approx(
        skm.f1_score(y, pred, labels=range(4), average="macro", zero_division=0))


def test_macro_f1_class_without_support_or_predictions_counts_zero():
    assert stats.macro_f1(np.array([0, 1]), np.array([0, 1]), 3) == pytest.approx(2 / 3)


def test_balanced_accuracy_matches_sklearn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, size=200)
    pred = np.where(rng.uniform(size=200) < 0.6, y, rng.integers(0, 4, size=200))
    assert stats.balanced_accuracy(y, pred, 4) == pytest.approx(skm.balanced_accuracy_score(y, pred))


def test_balanced_accuracy_only_over_present_classes():
    assert stats.balanced_accuracy(np.array([0, 0, 1]), np.array([0, 2, 1]), 3) == pytest.approx((0.5 + 1.0) / 2)


def test_confusion_matrix_rows_true_cols_pred():
    cm = stats.confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
    rng = np.random.default_rng(4)
    y, p = rng.integers(0, 4, 100), rng.integers(0, 4, 100)
    np.testing.assert_array_equal(stats.confusion_matrix(y, p, 4), skm.confusion_matrix(y, p, labels=range(4)))


def test_confusion_matrix_rejects_out_of_range():
    with pytest.raises(ValueError):
        stats.confusion_matrix(np.array([0, 3]), np.array([0, 1]), 3)


# ---- bootstrap -------------------------------------------------------------------------------------------
def mean_metric(x):
    return float(np.mean(x))


def test_bootstrap_ci_returns_documented_keys_and_contains_estimate():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    r = stats.bootstrap_ci(mean_metric, [x], None, None, 500, 0.95, 1)
    for key in ("estimate", "lo", "hi", "ci", "n_boot", "n_valid", "n", "n_groups"):
        assert key in r
    assert r["estimate"] == pytest.approx(x.mean())
    assert r["lo"] < r["estimate"] < r["hi"]
    assert r["n"] == 200 and r["n_groups"] == 200 and r["n_valid"] == 500 and r["ci"] == 0.95
    se = x.std(ddof=1) / math.sqrt(200)
    assert (r["hi"] - r["lo"]) == pytest.approx(2 * 1.96 * se, rel=0.2)


def test_bootstrap_ci_is_seeded():
    x = np.random.default_rng(6).normal(size=50)
    a = stats.bootstrap_ci(mean_metric, [x], None, None, 200, 0.95, 7)
    b = stats.bootstrap_ci(mean_metric, [x], None, None, 200, 0.95, 7)
    c = stats.bootstrap_ci(mean_metric, [x], None, None, 200, 0.95, 8)
    assert a == b and a != c


def correlated_groups(n_groups=20, per=20, seed=7):
    rng = np.random.default_rng(seed)
    effect = rng.normal(scale=1.0, size=n_groups)
    groups = np.repeat(np.arange(n_groups), per)
    x = effect[groups] + rng.normal(scale=0.2, size=groups.size)
    return x, groups


def test_cluster_bootstrap_wider_than_row_bootstrap_when_groups_correlated():
    x, groups = correlated_groups()
    row = stats.bootstrap_ci(mean_metric, [x], None, None, int(B["n"]), float(B["ci"]), int(B["seed"]))
    clus = stats.bootstrap_ci(mean_metric, [x], groups, None, int(B["n"]), float(B["ci"]), int(B["seed"]))
    assert clus["n_groups"] == 20 and clus["n"] == 400
    assert (clus["hi"] - clus["lo"]) > 2.5 * (row["hi"] - row["lo"])


def test_cluster_bootstrap_resamples_whole_groups():
    """Each group is constant, so every replicate mean is a mean of n_groups group values (multiples of 1/n_groups)."""
    groups = np.repeat(np.arange(4), [1, 2, 3, 4])
    x = groups.astype(float)                                # value = group id
    seen = []

    def metric(v, g):
        # within a replicate, every drawn group appears with all of its rows
        for gid in np.unique(g):
            assert np.all(v[g == gid] == gid)
            counts = np.sum(g == gid)
            assert counts % [1, 2, 3, 4][gid] == 0
        seen.append(len(v))
        return float(v.mean())

    r = stats.bootstrap_ci(metric, [x, groups], groups, None, 100, 0.9, 3)
    assert r["n_valid"] == 100 and r["n_groups"] == 4


def test_undefined_replicates_are_skipped_and_counted():
    y = np.array([0] * 3 + [1] * 1)
    s = np.array([0.1, 0.2, 0.3, 0.9])
    r = stats.bootstrap_ci(stats.auroc, [y, s], None, None, 400, 0.95, 2)
    assert 0 < r["n_valid"] < 400                          # replicates without the single positive are skipped
    assert r["n_boot"] == 400


def test_metric_raising_value_error_is_skipped():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ValueError("undefined")
        return float(np.mean(x))

    r = stats.bootstrap_ci(flaky, [np.arange(10.0)], None, None, 100, 0.95, 1)
    assert r["n_valid"] < 100


def test_stratified_bootstrap_keeps_every_class():
    y = np.array([0] * 30 + [1] * 2)
    s = np.random.default_rng(8).normal(size=32) + y
    r = stats.bootstrap_ci(stats.auroc, [y, s], None, y, 300, 0.95, 4)
    assert r["n_valid"] == 300


def test_stratified_cluster_bootstrap_keeps_strata():
    y = np.repeat([0, 0, 0, 1, 1], 4)
    groups = np.repeat(np.arange(5), 4)
    s = y + np.random.default_rng(9).normal(scale=0.5, size=20)
    r = stats.bootstrap_ci(stats.auroc, [y, s], groups, y, 300, 0.95, 5)
    assert r["n_valid"] == 300 and r["n_groups"] == 5


def test_all_replicates_undefined_gives_nan_interval():
    r = stats.bootstrap_ci(stats.auroc, [np.zeros(5, dtype=int), np.arange(5.0)], None, None, 50, 0.95, 1)
    assert math.isnan(r["estimate"]) and math.isnan(r["lo"]) and math.isnan(r["hi"]) and r["n_valid"] == 0


def test_bootstrap_input_validation():
    with pytest.raises(ValueError):
        stats.bootstrap_ci(mean_metric, [np.arange(3.0), np.arange(4.0)], None, None, 10, 0.95, 1)
    with pytest.raises(ValueError):
        stats.bootstrap_ci(mean_metric, [np.arange(3.0)], np.arange(4), None, 10, 0.95, 1)
    with pytest.raises(ValueError):
        stats.bootstrap_ci(mean_metric, [np.arange(3.0)], None, None, 0, 0.95, 1)
    with pytest.raises(ValueError):
        stats.bootstrap_ci(mean_metric, [np.arange(3.0)], None, None, 10, 1.5, 1)
    with pytest.raises(ValueError):
        stats.bootstrap_ci(mean_metric, [], None, None, 10, 0.95, 1)


def test_paired_difference_ci():
    rng = np.random.default_rng(10)
    y = rng.integers(0, 2, size=300)
    good = y + rng.normal(scale=0.5, size=300)
    bad = y + rng.normal(scale=3.0, size=300)
    r = stats.paired_difference_ci(stats.auroc, y, good, bad, None, y, 500, 0.95, 3)
    assert r["a"] == pytest.approx(stats.auroc(y, good)) and r["b"] == pytest.approx(stats.auroc(y, bad))
    assert r["estimate"] == pytest.approx(r["a"] - r["b"])
    assert r["lo"] > 0 and r["n_valid"] == 500
    same = stats.paired_difference_ci(stats.auroc, y, good, good, None, y, 200, 0.95, 3)
    assert same["lo"] == same["hi"] == 0.0


def test_paired_difference_narrower_than_unpaired_when_scores_correlated():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=200)
    base = y + rng.normal(scale=1.0, size=200)
    a, b = base + rng.normal(scale=0.1, size=200), base + rng.normal(scale=0.1, size=200)
    paired = stats.paired_difference_ci(stats.auroc, y, a, b, None, None, 500, 0.95, 1)
    single = stats.bootstrap_ci(stats.auroc, [y, a], None, None, 500, 0.95, 1)
    assert (paired["hi"] - paired["lo"]) < (single["hi"] - single["lo"])


def test_caller_misuse_raises_instead_of_nan():
    """A 4-class y passed to the binary auroc is a caller error, not an undefined replicate."""
    y = np.array([0, 1, 2, 3] * 5)
    with pytest.raises(ValueError):
        stats.bootstrap_ci(stats.auroc, [y, np.arange(20.0)], None, None, 50, 0.95, 1)
