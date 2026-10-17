"""pipeline.cells_convert on synthetic nuclei: geometry, µm units, original-pixel contours, edge flag, padding,
untyped instances, CELL_JSON keys, overlay colors, resize geometry."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from core import config, schemas
from pipeline import cells_convert as cc

CFG = config.load_settings()
FCFG = cc.load_feature_config(CFG)
NAMES = cc.type_names(FCFG)


def ellipse(cx, cy, a, b, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], axis=1).round().astype(np.int32)


def _recs(contours, types=None, probs=None, hw=(200, 200), mpp=0.25, scale=1.68):
    types = types or [1] * len(contours)
    probs = probs or [0.9] * len(contours)
    cents = [np.asarray(c).mean(axis=0) for c in contours]
    return cc.nuclei_records(contours, cents, types, probs, mpp_model=mpp, scale=scale, hw_model=hw, names=NAMES,
                             simplify_px=1.0, edge_px=2)


def test_feature_config_matches_settings():
    assert [NAMES[k] for k in sorted(NAMES)] == list(schemas.CELL_TYPES)
    assert [f["id"] for f in FCFG["features"]] == ["neo_fraction", "neo_density", "nuc_area_mean", "nuc_area_cv",
                                                   "nuc_shape_cv", "infl_fraction", "til_proximity", "dead_fraction",
                                                   "epi_fraction"]
    assert all(f["display_name"] for f in FCFG["features"])


def test_circle_area_units_eccentricity_solidity():
    (rec,), n_untyped = _recs([ellipse(100, 100, 20, 20)])
    assert n_untyped == 0
    assert rec["area_um2"] == pytest.approx(np.pi * 20 ** 2 * 0.25 ** 2, rel=0.08)
    assert rec["perimeter_um"] == pytest.approx(2 * np.pi * 20 * 0.25, rel=0.05)
    assert rec["eccentricity"] < 0.2 and rec["solidity"] > 0.95
    assert rec["centroid_um"] == pytest.approx([25.0, 25.0], abs=0.3)
    assert rec["type"] == "neoplastic" and rec["edge"] is False
    # contour stored in ORIGINAL tile pixels: model px / scale
    xs = np.asarray(rec["contour_px"])[:, 0]
    assert xs.max() == pytest.approx(120 / 1.68, abs=1.0)


def test_ellipse_eccentricity_and_concave_solidity():
    (rec,), _ = _recs([ellipse(100, 100, 30, 10)])
    assert rec["eccentricity"] == pytest.approx(np.sqrt(1 - (10 / 30) ** 2), abs=0.03)
    c_shape = np.array([[0, 0], [40, 0], [40, 10], [10, 10], [10, 30], [40, 30], [40, 40], [0, 40]]) + 50
    (rec,), _ = _recs([c_shape])
    assert rec["solidity"] < 0.75


def test_edge_flag_padding_untyped_and_degenerate():
    pad = np.iinfo(np.int32).min
    padded = np.vstack([ellipse(100, 100, 10, 10), [[pad, pad]] * 5])
    recs, n_untyped = _recs([ellipse(12, 100, 10, 10), padded, ellipse(50, 50, 5, 5), np.array([[1, 1], [2, 2]])],
                            types=[2, 3, 0, 1])
    assert n_untyped == 1 and len(recs) == 2
    assert recs[0]["edge"] is True and recs[0]["type"] == "inflammatory"
    assert recs[1]["edge"] is False and recs[1]["type"] == "connective"
    assert all(np.isfinite(np.asarray(r["contour_px"])).all() for r in recs)


def test_simplify_keeps_shape_and_reduces_points():
    c = ellipse(100, 100, 30, 20, n=200)
    s = cc.simplify(c, 1.0)
    assert 8 <= len(s) < len(c)
    a, b = 30, 20
    true = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))  # Ramanujan
    assert abs(cc.perimeter(s) - true) / true < 0.03
    (rec,), _ = _recs([c], scale=1.0, mpp=1.0)
    assert rec["perimeter_um"] == pytest.approx(true, rel=0.03)


def test_tile_document_and_json_roundtrip(tmp_path):
    recs, n_untyped = _recs([ellipse(100, 100, 12, 9)])
    meta = {"case_id": "BACH_0001", "tile_idx": 5, "backend": "fake", "model": "m", "mpp_model": 0.25,
            "mpp_source": 0.42, "scale": 1.68}
    path = cc.write_tile_json(schemas.cell_json(tmp_path, "BACH_0001", 5), meta, recs, n_untyped)
    doc = json.loads(path.read_text())
    assert schemas.CELL_JSON.problems(doc) == [] and doc["n_nuclei"] == 1
    with pytest.raises(ValueError):
        cc.tile_document({"case_id": "x"}, recs, 0)


def test_overlay_draws_type_colors():
    tile = Image.new("RGB", (120, 120), (255, 255, 255))
    recs, _ = _recs([ellipse(100, 100, 20, 20), ellipse(40, 40, 15, 15)], types=[1, 2], scale=1.0)
    out = cc.render_overlay(tile, recs, CFG["cells"]["overlay_colors"])
    px = {tuple(v) for v in np.asarray(out).reshape(-1, 3)}
    hexrgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))  # noqa: E731
    assert hexrgb(CFG["cells"]["overlay_colors"]["neoplastic"]) in px
    assert hexrgb(CFG["cells"]["overlay_colors"]["inflammatory"]) in px
    assert np.asarray(tile).min() == 255  # input untouched


def test_resize_to_model_geometry(tmp_path):
    src = tmp_path / "t.png"
    Image.new("RGBA", (512, 512), (200, 100, 150, 255)).save(src)
    geo = cc.resize_to_model(src, tmp_path / "m.png", 0.42, 0.25)
    with Image.open(tmp_path / "m.png") as im:
        assert im.size == (860, 860) and im.mode == "RGB"
    assert geo["scale"] == pytest.approx(860 / 512) and geo["mpp_model"] == pytest.approx(0.42 * 512 / 860)
    assert geo["hw_model"] == [860, 860] and geo["tile_px"] == [512, 512]


def test_cells_convert_is_portable():
    from tests import synthetic_tree as st

    assert st.delta_import_problems("pipeline/cells_convert.py", st.DELTA_ALLOWED - {"sklearn", "scipy"}) == []
