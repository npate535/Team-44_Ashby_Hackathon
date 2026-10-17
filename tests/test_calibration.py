"""core.calibration: temperature scaling and calibration metrics (plan §6.6, BUILD_PLAN §A3/§A5)."""

import numpy as np
import pytest

from core import calibration
from core.config import load_settings

CFG = load_settings()
BOUNDS = CFG["calibration"]["temperature_bounds"]
N_BINS = int(CFG["calibration"]["ece_bins"])


def logits_with_temperature(n, c, true_t, seed):
    """Labels drawn from softmax(z / true_t): fitting T on z should recover true_t."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=(n, c))
    p = calibration.softmax(z, true_t)
    labels = np.array([rng.choice(c, p=row) for row in p])
    return z, labels


def test_softmax_rows_sum_to_one_and_are_stable_for_huge_logits():
    z = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    p = calibration.softmax(z, 1.0)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p[1], np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum())


def test_softmax_temperature_flattens_and_sharpens():
    z = np.array([[2.0, 0.0]])
    assert calibration.softmax(z, 2.0)[0, 0] < calibration.softmax(z, 1.0)[0, 0] < calibration.softmax(z, 0.5)[0, 0]


def test_softmax_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        calibration.softmax(np.zeros((1, 2)), 0.0)
    with pytest.raises(ValueError):
        calibration.softmax(np.zeros((1, 2)), -1.0)


@pytest.mark.parametrize("true_t", [0.5, 2.5])
def test_fit_temperature_recovers_known_temperature(true_t):
    z, labels = logits_with_temperature(6000, 4, true_t, seed=3)
    t = calibration.fit_temperature(z, labels, BOUNDS)
    assert t == pytest.approx(true_t, rel=0.1)


def test_fit_temperature_reduces_nll_and_never_changes_argmax():
    z, labels = logits_with_temperature(2000, 4, 2.0, seed=4)
    t = calibration.fit_temperature(z, labels, BOUNDS)
    before = calibration.nll(calibration.softmax(z, 1.0), labels)
    after = calibration.nll(calibration.softmax(z, t), labels)
    assert after < before
    np.testing.assert_array_equal(calibration.softmax(z, t).argmax(1), z.argmax(1))


def test_fit_temperature_respects_bounds():
    z, labels = logits_with_temperature(2000, 3, 8.0, seed=5)
    t = calibration.fit_temperature(z, labels, [0.5, 2.0])
    assert 0.5 <= t <= 2.0
    assert t == pytest.approx(2.0, abs=1e-3)


def test_fit_temperature_rejects_bad_bounds():
    z, labels = logits_with_temperature(50, 3, 1.0, seed=6)
    for bad in ([0.0, 2.0], [2.0, 1.0], [1.0]):
        with pytest.raises(ValueError):
            calibration.fit_temperature(z, labels, bad)


def test_nll_and_brier_known_values():
    probs = np.array([[0.8, 0.2], [0.4, 0.6]])
    labels = np.array([0, 0])
    assert calibration.nll(probs, labels) == pytest.approx(-(np.log(0.8) + np.log(0.4)) / 2)
    assert calibration.brier(probs, labels) == pytest.approx(((0.2**2 + 0.2**2) + (0.6**2 + 0.6**2)) / 2)


def test_nll_clips_zero_probability():
    v = calibration.nll(np.array([[1.0, 0.0]]), np.array([1]))
    assert np.isfinite(v) and v > 10


def test_ece_of_perfectly_calibrated_data_is_near_zero():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=50000)
    labels = np.array([rng.choice(4, p=p) for p in probs])
    assert calibration.ece(probs, labels, N_BINS) < 0.01


def test_ece_of_overconfident_data_is_large():
    rng = np.random.default_rng(1)
    n = 5000
    labels = rng.integers(0, 2, size=n)
    probs = np.tile([0.99, 0.01], (n, 1))                  # always 0.99 on class 0, right half the time
    assert calibration.ece(probs, labels, N_BINS) == pytest.approx(0.99 - (labels == 0).mean(), abs=1e-9)


def test_reliability_diagram_equal_mass_bins_bookkeeping():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=103)
    labels = rng.integers(0, 3, size=103)
    d = calibration.reliability_diagram(probs, labels, 10)
    assert d["n"] == 103
    counts = [b["count"] for b in d["bins"]]
    assert sum(counts) == 103 and len(counts) == 10
    assert max(counts) - min(counts) <= 1                  # equal mass
    conf = probs.max(1)
    for b in d["bins"]:
        assert b["lo"] <= b["confidence"] <= b["hi"]
        assert 0.0 <= b["accuracy"] <= 1.0
    los = [b["lo"] for b in d["bins"]]
    assert los == sorted(los)
    assert d["bins"][0]["lo"] == pytest.approx(conf.min())
    assert d["bins"][-1]["hi"] == pytest.approx(conf.max())
    weighted = sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in d["bins"]) / 103
    assert d["ece"] == pytest.approx(weighted)
    assert d["ece"] == pytest.approx(calibration.ece(probs, labels, 10))


def test_reliability_diagram_fewer_rows_than_bins_drops_empty_bins():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
    d = calibration.reliability_diagram(probs, np.array([0, 1, 1]), 10)
    assert sum(b["count"] for b in d["bins"]) == 3
    assert all(b["count"] > 0 for b in d["bins"])


def test_metric_input_validation():
    with pytest.raises(ValueError):
        calibration.nll(np.array([[0.5, 0.5]]), np.array([0, 1]))
    with pytest.raises(ValueError):
        calibration.ece(np.array([[0.5, 0.5]]), np.array([0]), 0)
    with pytest.raises(ValueError):
        calibration.brier(np.zeros((0, 2)), np.array([], dtype=int))
