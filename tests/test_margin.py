"""core.margin: strip geometry, benign-FPR threshold and suggested call (plan §6.9, BUILD_PLAN §A12)."""

import copy

import numpy as np
import pytest

from core import contract, margin
from core.config import load_settings
from tests.test_contract import ring_segments, surgical_case

CFG = load_settings()
M = CFG["margin"]
W, H = CFG["dataset"]["image_size"]
SEG = int(M["segment_px"])
NAMES = M["names"]


def segs():
    return margin.strip_segments(W, H, SEG, NAMES)


def test_margin_order_is_contract_order():
    assert margin.MARGIN_ORDER == contract.MARGIN_NAMES


def test_strip_geometry_24_squares_28_memberships():
    s = segs()
    assert len(s) == 28
    assert len({(d["x"], d["y"]) for d in s}) == 24
    by = {m: [d for d in s if d["margin"] == m] for m in contract.MARGIN_NAMES}
    assert [len(by[m]) for m in contract.MARGIN_NAMES] == [8, 8, 6, 6]
    assert all(set(d) == {"margin", "x", "y", "size"} and d["size"] == SEG for d in s)


def test_strip_order_matches_contract_ring_builder():
    expected = [{k: v for k, v in d.items() if k != "p"} for d in ring_segments(lambda x, y: 0.0)]
    assert segs() == expected


def test_corner_squares_belong_to_both_margins():
    s = segs()
    corners = {(0, 0): {"superior", "medial"}, (W - SEG, 0): {"superior", "lateral"},
               (0, H - SEG): {"inferior", "medial"}, (W - SEG, H - SEG): {"inferior", "lateral"}}
    for xy, names in corners.items():
        assert {d["margin"] for d in s if (d["x"], d["y"]) == xy} == names


def test_strip_segments_order_follows_contract_even_with_remapped_names():
    names = {"top": "lateral", "bottom": "medial", "left": "superior", "right": "inferior"}
    s = margin.strip_segments(W, H, SEG, names)
    order = [d["margin"] for d in s]
    assert order == sorted(order, key=contract.MARGIN_NAMES.index)
    assert all(d["x"] == 0 for d in s if d["margin"] == "superior")


def test_strip_segments_rejects_bad_geometry_and_names():
    with pytest.raises(ValueError):
        margin.strip_segments(2000, 1536, 256, NAMES)
    with pytest.raises(ValueError):
        margin.strip_segments(256, 1536, 256, NAMES)          # needs at least 2 squares per edge
    with pytest.raises(ValueError):
        margin.strip_segments(W, H, 256, {"top": "superior", "bottom": "superior", "left": "medial", "right": "lateral"})
    with pytest.raises(ValueError):
        margin.strip_segments(W, H, 256, {"top": "superior", "bottom": "inferior", "left": "medial"})


def test_per_margin_scores_is_max_per_margin():
    s = segs()
    probs = [0.01 * i for i in range(len(s))]
    pm = margin.per_margin_scores(s, probs)
    assert list(pm) == list(contract.MARGIN_NAMES)
    for m in contract.MARGIN_NAMES:
        assert pm[m] == pytest.approx(max(p for d, p in zip(s, probs) if d["margin"] == m))


def test_per_margin_scores_validates():
    s = segs()
    with pytest.raises(ValueError):
        margin.per_margin_scores(s, [0.1] * 3)
    with pytest.raises(ValueError):
        margin.per_margin_scores(s[:8], [0.1] * 8)              # missing margins
    with pytest.raises(ValueError):
        margin.per_margin_scores(s, [1.5] * len(s))


def test_fit_threshold_controls_benign_false_positive_rate():
    rng = np.random.default_rng(0)
    benign = rng.beta(2, 8, size=(12, 4))                       # 12 benign cal images x 4 margins
    t = margin.fit_threshold(benign, float(M["max_fpr"]))
    assert np.mean(benign.ravel() > t) <= float(M["max_fpr"])
    assert t in benign.ravel()                                   # method "higher" returns an observed maximum


def test_fit_threshold_uses_settings_quantile_method():
    x = np.arange(1, 21) / 20.0                                  # 20 pooled maxima
    t = margin.fit_threshold(x, 0.05)
    assert t == pytest.approx(float(np.quantile(x, 0.95, method=M["threshold_quantile_method"])))


def test_fit_threshold_validates():
    with pytest.raises(ValueError):
        margin.fit_threshold(np.array([]), 0.05)
    with pytest.raises(ValueError):
        margin.fit_threshold(np.array([0.1, 0.2]), 0.0)
    with pytest.raises(ValueError):
        margin.fit_threshold(np.array([0.1, np.nan]), 0.05)


def test_involved_margins_strictly_above_in_contract_order():
    pm = {"lateral": 0.9, "superior": 0.5, "inferior": 0.51, "medial": 0.2}
    assert margin.involved_margins(pm, 0.5) == ["inferior", "lateral"]
    assert margin.involved_margins(pm, 0.95) == []


def test_involved_length_um_sums_hot_segments_of_involved_margins():
    s = segs()
    hot = {(768, 0), (1024, 0), (0, 512)}
    probs = [0.9 if (d["x"], d["y"]) in hot else 0.1 for d in s]
    mpp = 0.42
    out = margin.involved_length_um(s, probs, 0.5, mpp)
    assert out == {"superior": pytest.approx(2 * SEG * mpp), "medial": pytest.approx(SEG * mpp)}


def test_suggest_call_defer_tumor_clear():
    pm = {"superior": 0.8, "inferior": 0.1, "medial": 0.1, "lateral": 0.6}
    assert margin.suggest_call(pm, 0.5, True) == ("defer", [])
    assert margin.suggest_call(pm, 0.5, False) == ("tumor_at_margin", ["superior", "lateral"])
    assert margin.suggest_call(pm, 0.9, False) == ("clear", [])


@pytest.mark.parametrize("flag,threshold", [(False, 0.5), (False, 0.95), (True, 0.5)])
def test_outputs_validate_against_the_contract(flag, threshold):
    s = segs()
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, size=len(s)).round(4).tolist()
    pm = margin.per_margin_scores(s, probs)
    call, involved = margin.suggest_call(pm, threshold, flag)
    lengths = margin.involved_length_um(s, probs, threshold, CFG["dataset"]["mpp"])
    block = {
        "suggested_call": call,
        "involved": involved,
        "per_margin": pm,
        "strip_um": M["strip_px"] * CFG["dataset"]["mpp"],
        "segments": [{**d, "p": p} for d, p in zip(s, probs)],
        "involved_length_um": {m: v for m, v in lengths.items() if m in involved} or None,
    }
    if call != "defer":            # a deferred read lists no involved margins, so it cannot carry the threshold
        block["threshold"] = threshold
    contract.Margin.model_validate(block)
    case = copy.deepcopy(surgical_case())
    case["margin"] = block
    case["findings"][-1]["data"] = {"suggested_call": call, "involved": involved}
    contract.Case.model_validate(case)
