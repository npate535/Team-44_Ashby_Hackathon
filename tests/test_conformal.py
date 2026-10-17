"""core.conformal: LAC, APS and Mondrian split-conformal sets (plan §6.6, BUILD_PLAN §A5)."""

import math

import numpy as np
import pytest
from scipy import stats as sps

from core import conformal
from core.config import load_settings

CFG = load_settings()
CLASSES = list(CFG["classes"])
ALPHA = float(CFG["alpha"])


def dirichlet_data(n, c, seed, conc=1.0):
    """Exchangeable (probs, labels) with labels drawn from the probs themselves (perfectly calibrated)."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(c, conc), size=n)
    labels = np.array([rng.choice(c, p=p) for p in probs])
    return probs, labels


# ---- LAC ------------------------------------------------------------------------------------------
def test_lac_scores_are_one_minus_true_class_prob():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_allclose(conformal.lac_scores(probs, np.array([0, 1])), [0.3, 0.7])


def test_conformal_quantile_finite_sample_k_at_n50_alpha01_is_46():
    scores = np.arange(1, 51) / 100.0                    # sorted 0.01 .. 0.50
    assert math.ceil(51 * 0.9) == 46
    assert conformal.conformal_quantile(scores[::-1].copy(), 0.1) == pytest.approx(0.46)


def test_conformal_quantile_k_greater_than_n_is_inf():
    # n = 5, alpha = 0.1: k = ceil(6 * 0.9) = 6 > 5
    assert conformal.conformal_quantile(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.1) == math.inf


def test_conformal_quantile_k_equal_n_is_max():
    # n = 9, alpha = 0.1: k = ceil(10 * 0.9) = 9 = n
    scores = np.linspace(0.1, 0.9, 9)
    assert conformal.conformal_quantile(scores, 0.1) == pytest.approx(0.9)


def test_conformal_quantile_rejects_bad_inputs():
    with pytest.raises(ValueError):
        conformal.conformal_quantile(np.array([]), 0.1)
    with pytest.raises(ValueError):
        conformal.conformal_quantile(np.array([0.1, np.nan]), 0.1)
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            conformal.conformal_quantile(np.array([0.1, 0.2]), bad_alpha)


def test_lac_sets_threshold_inclusive_and_empty_allowed():
    probs = np.array([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]])
    sets = conformal.lac_sets(probs, 0.4)                 # include y when 1 - p_y <= 0.4, i.e. p_y >= 0.6
    assert sets.dtype == bool
    assert sets.tolist() == [[True, False, False], [False, False, False]]
    assert conformal.lac_sets(probs, math.inf).all()


def test_lac_sets_inclusive_at_exact_boundary():
    probs = np.array([[0.75, 0.25]])
    assert conformal.lac_sets(probs, 0.25).tolist() == [[True, False]]


def test_fit_lac_matches_quantile_of_scores():
    probs, labels = dirichlet_data(50, 4, seed=1)
    expected = conformal.conformal_quantile(conformal.lac_scores(probs, labels), ALPHA)
    assert conformal.fit_lac(probs, labels, ALPHA) == expected


def beta_band(n_cal, alpha, n_test, level=0.999):
    """Coverage of split conformal, marginal over the calibration draw: Beta(k, n+1-k) mixed with binomial test noise.
    We use the Beta mean/var plus binomial test variance for a conservative normal band."""
    k = math.ceil((n_cal + 1) * (1 - alpha))
    a, b = k, n_cal + 1 - k
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1)) + mean * (1 - mean) / n_test
    z = sps.norm.ppf(0.5 + level / 2)
    return mean - z * math.sqrt(var), mean + z * math.sqrt(var)


@pytest.mark.parametrize("n_cal", [50, 500])
def test_lac_marginal_coverage_inside_beta_band(n_cal):
    probs, labels = dirichlet_data(n_cal + 20000, 4, seed=7)
    qhat = conformal.fit_lac(probs[:n_cal], labels[:n_cal], ALPHA)
    cov = conformal.coverage(conformal.lac_sets(probs[n_cal:], qhat), labels[n_cal:])
    lo, hi = beta_band(n_cal, ALPHA, 20000)
    assert lo <= cov <= hi


def test_lac_average_coverage_over_many_calibration_draws_is_at_least_1_minus_alpha():
    covs = []
    for seed in range(200):
        probs, labels = dirichlet_data(50 + 400, 4, seed=1000 + seed)
        qhat = conformal.fit_lac(probs[:50], labels[:50], ALPHA)
        covs.append(conformal.coverage(conformal.lac_sets(probs[50:], qhat), labels[50:]))
    k = math.ceil(51 * (1 - ALPHA))
    expected = k / 51                                     # exact expected coverage of split conformal
    se = np.std(covs) / math.sqrt(len(covs))
    assert abs(np.mean(covs) - expected) < 4 * se + 1e-3
    assert np.mean(covs) >= 1 - ALPHA - 4 * se


# ---- APS ------------------------------------------------------------------------------------------
def test_aps_nonrandomized_score_is_cumulative_mass_through_true_class():
    probs = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    s = conformal.aps_scores(probs, np.array([1, 0]), seed=None, randomized=False)
    np.testing.assert_allclose(s, [0.8, 0.5])


def test_aps_randomized_score_is_mass_above_plus_u_times_own():
    probs = np.array([[0.5, 0.3, 0.2]] * 3)
    labels = np.array([1, 1, 1])
    u = np.random.default_rng(3).uniform(size=3)
    s = conformal.aps_scores(probs, labels, seed=3, randomized=True)
    np.testing.assert_allclose(s, 0.5 + u * 0.3)


def test_aps_scores_seeded_determinism():
    probs, labels = dirichlet_data(100, 4, seed=2)
    a = conformal.aps_scores(probs, labels, seed=11, randomized=True)
    b = conformal.aps_scores(probs, labels, seed=11, randomized=True)
    c = conformal.aps_scores(probs, labels, seed=12, randomized=True)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_aps_ties_break_stably_on_class_index():
    probs = np.array([[0.4, 0.4, 0.2]])
    s0 = conformal.aps_scores(probs, np.array([0]), seed=None, randomized=False)
    s1 = conformal.aps_scores(probs, np.array([1]), seed=None, randomized=False)
    np.testing.assert_allclose([s0[0], s1[0]], [0.4, 0.8])
    sets = conformal.aps_sets(probs, 0.5, seed=None, randomized=False)
    assert sets.tolist() == [[True, False, False]]


def test_aps_sets_nonrandomized_include_while_cumulative_le_qhat():
    probs = np.array([[0.5, 0.3, 0.2]])
    assert conformal.aps_sets(probs, 0.8, seed=None, randomized=False).tolist() == [[True, True, False]]
    assert conformal.aps_sets(probs, 0.79, seed=None, randomized=False).tolist() == [[True, False, False]]
    assert conformal.aps_sets(probs, 0.49, seed=None, randomized=False).tolist() == [[False, False, False]]
    assert conformal.aps_sets(probs, math.inf, seed=None, randomized=False).all()


def test_aps_sets_consistent_with_scores():
    """y is in the set iff its own (same-U) score <= qhat: the defining property of the APS set."""
    probs, _ = dirichlet_data(300, 4, seed=5)
    qhat = 0.7
    for randomized in (False, True):
        sets = conformal.aps_sets(probs, qhat, seed=9, randomized=randomized)
        for y in range(4):
            s = conformal.aps_scores(probs, np.full(len(probs), y), seed=9, randomized=randomized)
            np.testing.assert_array_equal(sets[:, y], s <= qhat)


def test_aps_coverage_inside_beta_band():
    n_cal = 500
    probs, labels = dirichlet_data(n_cal + 20000, 4, seed=8)
    seed = int(CFG["conformal"]["seed"])
    qhat = conformal.fit_aps(probs[:n_cal], labels[:n_cal], ALPHA, seed, True)
    sets = conformal.aps_sets(probs[n_cal:], qhat, seed + 1, True)
    lo, hi = beta_band(n_cal, ALPHA, 20000)
    assert lo <= conformal.coverage(sets, labels[n_cal:]) <= hi


# ---- Mondrian -------------------------------------------------------------------------------------
def test_mondrian_quantiles_per_class_and_inf_for_tiny_classes():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=40)
    labels = np.array([0] * 30 + [1] * 8 + [2] * 2)
    q = conformal.mondrian_quantiles(probs, labels, 0.1)
    assert q.shape == (3,)
    scores = conformal.lac_scores(probs, labels)
    assert q[0] == conformal.conformal_quantile(scores[labels == 0], 0.1)
    assert q[1] == math.inf                       # n=8: k = ceil(9*0.9) = 9 > 8
    assert q[2] == math.inf


def test_mondrian_absent_class_is_inf():
    probs = np.array([[0.9, 0.1, 0.0]] * 20)
    q = conformal.mondrian_quantiles(probs, np.zeros(20, dtype=int), 0.1)
    assert q[1] == math.inf and q[2] == math.inf


def test_mondrian_sets_use_per_class_threshold():
    probs = np.array([[0.6, 0.4], [0.2, 0.8]])
    sets = conformal.mondrian_sets(probs, np.array([0.5, 0.1]))
    assert sets.tolist() == [[True, False], [False, False]]


def test_mondrian_per_class_coverage_near_target():
    n_cal = 3000
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 3, size=n_cal + 30000)
    logits = rng.normal(size=(len(labels), 3)) + 2.0 * np.eye(3)[labels] * np.array([1.5, 0.5, 1.0])
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    q = conformal.mondrian_quantiles(probs[:n_cal], labels[:n_cal], ALPHA)
    sets = conformal.mondrian_sets(probs[n_cal:], q)
    for y in range(3):
        m = labels[n_cal:] == y
        cov = conformal.coverage(sets[m], labels[n_cal:][m])
        assert abs(cov - (1 - ALPHA)) < 0.03


# ---- helpers --------------------------------------------------------------------------------------
def test_set_labels_in_class_order_and_empty():
    assert conformal.set_labels(np.array([False, False, True, True]), CLASSES) == ["InSitu", "Invasive"]
    assert conformal.set_labels(np.zeros(4, dtype=bool), CLASSES) == []
    with pytest.raises(ValueError):
        conformal.set_labels(np.array([True, False]), CLASSES)


def test_coverage_and_mean_set_size():
    sets = np.array([[True, False, False], [False, True, True], [False, False, False]])
    assert conformal.coverage(sets, np.array([0, 0, 2])) == pytest.approx(1 / 3)
    assert conformal.mean_set_size(sets) == pytest.approx(1.0)


def test_input_validation():
    with pytest.raises(ValueError):
        conformal.lac_scores(np.array([0.5, 0.5]), np.array([0]))          # not 2-D
    with pytest.raises(ValueError):
        conformal.lac_scores(np.array([[0.5, 0.5]]), np.array([0, 1]))     # length mismatch
    with pytest.raises(ValueError):
        conformal.lac_scores(np.array([[0.5, 0.5]]), np.array([2]))        # label out of range
    with pytest.raises(ValueError):
        conformal.lac_scores(np.array([[0.5, np.nan]]), np.array([0]))
    with pytest.raises(ValueError):
        conformal.coverage(np.zeros((2, 3), dtype=bool), np.array([0]))
    with pytest.raises(ValueError):
        conformal.mondrian_sets(np.array([[0.5, 0.5]]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        conformal.coverage(np.zeros((0, 3), dtype=bool), np.array([], dtype=int))
