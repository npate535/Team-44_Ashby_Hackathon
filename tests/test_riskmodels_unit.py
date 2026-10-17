"""pipeline.riskmodels unit tests: pooling, linear model export, contributions, CV, one-SE, A-vs-B, ABMIL."""

import numpy as np
import pandas as pd
import pytest

from core import calibration
from core.config import load_settings
from pipeline import riskmodels as rm

CFG = load_settings()
CLASSES = list(CFG["classes"])


def index_frame(rows):
    base = {"emb_row": 0, "item_id": "x", "set": "bach512", "split": "test", "grid": "g512", "view": -1, "n_views": 1,
            "degrade_type": "", "degrade_severity": 0, "label": "", "group": ""}
    out = pd.DataFrame([{**base, **r} for r in rows])
    out["emb_row"] = np.arange(len(out))
    return out


# ---- pooling ---------------------------------------------------------------------------------------------------------------
def test_pool_tiles_uses_qc_tiles_and_falls_back_when_none_pass():
    idx = index_frame([{"case_id": "A", "tile_idx": 0}, {"case_id": "A", "tile_idx": 1},
                       {"case_id": "B", "tile_idx": 0}, {"case_id": "B", "tile_idx": 1}])
    x = np.array([[1.0], [3.0], [10.0], [20.0]], dtype=np.float32)
    qc = pd.DataFrame({"case_id": ["A", "A", "B", "B"], "grid": "g512", "tile_idx": [0, 1, 0, 1],
                       "qc_pass": [True, False, False, False]})
    p = rm.pool_tiles(x, idx, qc)
    f = p.frame()
    assert f.loc["A", 0] == 1.0                      # only the QC tile
    assert f.loc["B", 0] == 15.0                     # no QC tile: all tiles, recorded
    assert p.qc_fallback == ["B"]


def test_pool_tiles_degraded_rows_get_variant_ids_and_source_qc():
    idx = index_frame([{"case_id": "A", "tile_idx": 0, "degrade_type": "blur", "degrade_severity": 2},
                       {"case_id": "A", "tile_idx": 1, "degrade_type": "blur", "degrade_severity": 2},
                       {"case_id": "A", "tile_idx": 0, "degrade_type": "jpeg", "degrade_severity": 1}])
    x = np.array([[2.0], [100.0], [5.0]], dtype=np.float32)
    qc = pd.DataFrame({"case_id": ["A", "A"], "grid": "g512", "tile_idx": [0, 1], "qc_pass": [True, False]})
    p = rm.pool_tiles(x, idx, qc)
    f = p.frame()
    assert f.loc["A_FSIM_blur2", 0] == 2.0 and f.loc["A_FSIM_jpeg1", 0] == 5.0
    assert set(p.keys.variant) == {"blur2", "jpeg1"} and set(p.keys.source_case_id) == {"A"}


def test_pool_tiles_by_view_and_composite():
    idx = index_frame([{"case_id": "A", "tile_idx": t, "view": v} for v in range(2) for t in range(2)])
    x = np.arange(4, dtype=np.float32)[:, None]
    p = rm.pool_tiles(x, idx, None, by_view=True)
    assert p.keys.view.tolist() == [0, 1] and p.x[:, 0].tolist() == [0.5, 2.5]
    c = rm.pool_tiles(x, idx.assign(view=-1, case_id="A_COMP_0"), None, composite=True)
    assert c.keys.variant.tolist() == ["composite"]


def test_pool_tiles_missing_tiles_raise():
    idx = index_frame([{"case_id": "A", "tile_idx": 5}])
    qc = pd.DataFrame({"case_id": ["A"], "grid": "g512", "tile_idx": [0], "qc_pass": [True]})
    with pytest.raises(ValueError):
        rm.pool_tiles(np.ones((1, 2), dtype=np.float32), idx, qc)


# ---- linear model ----------------------------------------------------------------------------------------------------------
def blobs(n=200, d=6, seed=0, k=4):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    x = rng.normal(size=(n, d)) + 2.0 * np.eye(k, d)[y]
    return x, y


def test_linear_model_npz_roundtrip_and_imputation(tmp_path):
    x, y = blobs()
    x[3, 2] = np.nan
    m = rm.LinearModel(CLASSES, 0.5, 2000, None).fit(x, y, [f"f{i}" for i in range(6)])
    assert np.isfinite(m.logits(x)).all()
    path = m.to_npz(tmp_path / "m.npz", model="test")
    m2 = rm.LinearModel.from_npz(path)
    np.testing.assert_allclose(m.logits(x), m2.logits(x))
    assert m2.names == m.names and m2.classes == CLASSES
    assert (m.logits(x).argmax(1) == y).mean() > 0.8


def test_linear_model_requires_every_class():
    x, y = blobs()
    with pytest.raises(ValueError):
        rm.LinearModel(CLASSES, 1.0, 1000, None).fit(x[y != 3], y[y != 3])


def test_centered_contributions_sum_to_centered_logit():
    x, y = blobs()
    m = rm.LinearModel(CLASSES, 1.0, 2000, None).fit(x, y)
    agg = CLASSES.index(CFG["aggressive_class"])
    contrib = rm.centered_contributions(m, x, agg)
    logits = m.logits(x)
    centered = logits[:, agg] - logits.mean(1) - (m.intercept[agg] - m.intercept.mean())
    np.testing.assert_allclose(contrib.sum(1), centered, atol=1e-9)


# ---- CV, one-SE, A-vs-B --------------------------------------------------------------------------------------------------
def train_frame(n=60, group_size=3):
    return pd.DataFrame({"case_id": [f"C{i:03d}" for i in range(n)], "label": [CLASSES[i % 4] for i in range(n)],
                         "group": [f"g{i // group_size}" for i in range(n)]})


def test_cv_splits_are_grouped_and_cover_every_case_each_repeat():
    tr = train_frame()
    splits = rm.cv_splits(tr, CFG)
    assert len(splits) == CFG["split"]["cv"]["repeats"]
    for folds in splits:
        assert len(folds) == CFG["split"]["cv"]["folds"]
        allpos = np.concatenate(folds)
        assert sorted(allpos.tolist()) == list(range(len(tr)))
        for held in folds:
            held_groups = set(tr.group.iloc[held])
            rest = set(tr.group.drop(tr.index[held]))
            assert not held_groups & rest
    assert not all(np.array_equal(a[0], b[0]) for a, b in zip(splits[0:1], splits[1:2]))


def test_run_cv_never_fits_on_held_out_rows():
    tr = train_frame()
    x, y = blobs(len(tr))
    splits = rm.cv_splits(tr, CFG)

    def fit_predict(fit, held):
        assert not set(fit) & set(held)
        m = rm.LinearModel(CLASSES, 1.0, 2000, None).fit(x[fit], y[fit])
        return m.logits(x[held])

    res = rm.run_cv("lr", "black_box", {"C": 1.0}, len(tr), y, splits, fit_predict, 3, 6, 4)
    assert res.oof_logits.shape == (3, len(tr), 4)
    assert len(res.fold_losses) == 15
    assert res.oof_probs().shape == (len(tr), 4)


def fake_result(name, loss, se_spread, complexity, C=None):
    losses = [loss - se_spread, loss + se_spread] * 3 + [loss] * 9
    cfg = {} if C is None else {"C": C}
    return rm.CVResult(name, "black_box", cfg, np.zeros((1, 2, 4)), losses, complexity, 3)


def test_one_se_picks_simplest_within_one_se():
    best = fake_result("big", 0.50, 0.2, complexity=8, C=1.0)
    simple = fake_result("small", 0.52, 0.2, complexity=1, C=0.1)
    far = fake_result("bad", 0.90, 0.01, complexity=0, C=0.01)
    chosen, info = rm.one_se_choice([best, simple, far], n_folds=5)
    assert chosen.name == "small"
    assert info["best"].startswith("big") and "bad@C=0.01" not in info["eligible"]


def test_one_se_prefers_smaller_C_within_candidate():
    a = fake_result("lr", 0.50, 0.2, 3, C=1.0)
    b = fake_result("lr", 0.51, 0.2, 3, C=0.01)
    chosen, _ = rm.one_se_choice([a, b], 5)
    assert chosen.config["C"] == 0.01


def test_a_vs_b_rule():
    def m(auc, bacc):
        return {"auroc_invasive": {"estimate": auc}, "balanced_accuracy": {"estimate": bacc}}

    assert rm.a_vs_b(m(0.95, 0.80), m(0.93, 0.77), 0.03, 0.05)[0] == "B"
    assert rm.a_vs_b(m(0.95, 0.80), m(0.90, 0.79), 0.03, 0.05)[0] == "A"
    assert rm.a_vs_b(m(0.95, 0.80), m(0.95, 0.70), 0.03, 0.05)[0] == "A"
    assert rm.a_vs_b(m(0.90, 0.70), m(0.95, 0.80), 0.03, 0.05)[0] == "B"


# ---- ABMIL ---------------------------------------------------------------------------------------------------------------------
def mil_data(n=120, seed=0):
    """Class = which 'key' tile direction appears in a bag of noise tiles; attention should find it."""
    rng = np.random.default_rng(seed)
    bags, y, key_pos = [], [], []
    for i in range(n):
        c = i % 4
        t = int(rng.integers(4, 10))
        bag = rng.normal(size=(t, 8)).astype(np.float32)
        p = int(rng.integers(t))
        bag[p, c] += 4.0
        bags.append(bag)
        y.append(c)
        key_pos.append(p)
    return bags, np.array(y), key_pos


ABMIL_PARAMS = {**CFG["abmil"], "epochs": 30}


def test_abmil_learns_and_attends_to_key_tile(tmp_path):
    bags, y, key = mil_data()
    m = rm.ABMIL(CLASSES, ABMIL_PARAMS).fit(bags[:80], y[:80])
    logits, att = m.predict(bags[80:])
    assert (logits.argmax(1) == y[80:]).mean() > 0.7
    assert all(abs(a.sum() - 1) < 1e-5 for a in att)
    hit = np.mean([a.argmax() == k for a, k in zip(att, key[80:])])
    assert hit > 0.5
    path = m.to_npz(tmp_path / "abmil.npz")
    m2 = rm.ABMIL.from_npz(path, ABMIL_PARAMS)
    np.testing.assert_allclose(m2.predict(bags[80:])[0], logits, atol=1e-5)


def test_abmil_deterministic():
    bags, y, _ = mil_data(40)
    a = rm.ABMIL(CLASSES, {**ABMIL_PARAMS, "epochs": 3}).fit(bags, y).predict(bags)[0]
    b = rm.ABMIL(CLASSES, {**ABMIL_PARAMS, "epochs": 3}).fit(bags, y).predict(bags)[0]
    np.testing.assert_allclose(a, b)


def test_cv_metrics_have_cluster_cis():
    tr = train_frame()
    x, y = blobs(len(tr))
    splits = rm.cv_splits(tr, CFG)
    res = rm.run_cv("lr", "black_box", {"C": 1.0}, len(tr), y, splits,
                    lambda f, h: rm.LinearModel(CLASSES, 1.0, 2000, None).fit(x[f], y[f]).logits(x[h]), 3, 6, 4)
    cfg = {**CFG, "bootstrap": {**CFG["bootstrap"], "n": 200}}
    met = rm.cv_metrics(res, y, tr.group.to_numpy(), cfg, CLASSES.index(CFG["aggressive_class"]))
    for key in ("log_loss", "auroc_invasive", "balanced_accuracy", "macro_f1", "macro_auroc"):
        assert met[key]["lo"] <= met[key]["estimate"] <= met[key]["hi"]
        assert met[key]["n_groups"] == 20
    assert met["log_loss"]["estimate"] == pytest.approx(calibration.nll(res.oof_probs(), y))


def test_fold_calibrated_losses_are_out_of_sample_and_fix_overconfidence():
    tr = train_frame()
    x, y = blobs(len(tr))
    splits = rm.cv_splits(tr, CFG)[:1]
    good = np.log(np.eye(4)[y] * 0.6 + 0.1)                  # calibrated-ish logits
    over = 10 * good                                         # same ranking, wildly overconfident
    raw = rm.fold_calibrated_losses(over[None], y, splits, None)
    cal = rm.fold_calibrated_losses(over[None], y, splits, CFG["calibration"]["temperature_bounds"])
    assert np.mean(cal) < np.mean(raw)
    assert rm.fold_calibrated_losses(good[None], y, splits, None) == pytest.approx(
        [calibration.nll(calibration.softmax(good[h], 1.0), y[h]) for h in splits[0]])
