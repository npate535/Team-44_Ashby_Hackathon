"""eval/human_baseline: reader figures are parsed from the verified literature file, never typed."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from eval import human_baseline as hb


def _lit():
    return yaml.safe_load(hb.LITERATURE.read_text())


def test_bach_readers_parse_from_literature():
    r = hb.parse_bach_readers(hb._figures(_lit(), hb.BACH_KEY))
    assert r["accuracy"] == [0.96, 0.94, 0.78, 0.73]
    assert r["sensitivity"]["InSitu"] == [1.0, 1.0, 0.56, 0.76]
    assert r["sensitivity"]["Normal"][3] == pytest.approx(0.40)


def test_bpath_categories_parse_from_literature():
    b = hb.parse_bpath(hb._figures(_lit(), hb.BPATH_KEY))
    assert b["invasive"]["concordance"] == pytest.approx(0.96)
    assert b["atypia"] == {"concordance": 0.48, "lo": 0.44, "hi": 0.52}


def test_missing_figure_is_an_error():
    with pytest.raises(ValueError):
        hb.parse_bach_readers([{"value": "Accuracy 96% (E1, BACH annotator), 94%, 78%, 73% (external)"}])


def test_sepia_recall_counts():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])
    out = hb.sepia_rates(y, pred, np.arange(8).astype(str), ["Normal", "Benign", "InSitu", "Invasive"],
                         {"n_boot": 50, "ci": 0.95, "seed": 1})
    assert out["recall"]["Normal"]["estimate"] == pytest.approx(0.5)
    assert out["recall"]["Invasive"]["correct"] == 2 and out["recall"]["Invasive"]["n_class"] == 2
