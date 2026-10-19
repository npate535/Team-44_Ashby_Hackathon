"""eval/attention_comparison: which features ABMIL used, and paired test differences."""

from __future__ import annotations

import numpy as np
import pytest

from eval import attention_comparison as ac


def test_best_single_ignores_abmil_and_fusion():
    mc = {"candidates": [
        {"model": "abmil", "family": "black_box", "chosen_within_candidate": True, "cv_log_loss": 0.10},
        {"model": "fusion_top3", "family": "black_box", "chosen_within_candidate": True, "cv_log_loss": 0.11},
        {"model": "uni2h512_lr", "family": "black_box", "chosen_within_candidate": True, "cv_log_loss": 0.18},
        {"model": "uni2h512_lr", "family": "black_box", "chosen_within_candidate": False, "cv_log_loss": 0.16},
        {"model": "conch_probe_lr", "family": "black_box", "chosen_within_candidate": True, "cv_log_loss": 0.28},
        {"model": "cb_text", "family": "concept", "chosen_within_candidate": True, "cv_log_loss": 0.01}]}
    assert ac.best_single(mc) == "uni2h512_lr"


def test_identical_models_have_zero_difference():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 10)
    p = rng.dirichlet(np.ones(4), size=40)
    p[np.arange(40), y] += 1.0
    groups = np.repeat(np.arange(8), 5)
    out = ac.compare(y, p, p.copy(), groups, {"n_boot": 200, "ci": 0.95, "seed": 1}, 4, 3)
    assert out["macro_f1"]["estimate"] == pytest.approx(0.0)
    assert out["macro_f1"]["lo"] == pytest.approx(0.0) and out["macro_f1"]["hi"] == pytest.approx(0.0)
    assert out["disagreements"] == 0
