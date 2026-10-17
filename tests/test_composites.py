"""pipeline.composites: truth table per orientation, geometry, determinism, test images only, manifest rows."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from core import config, schemas
from tests import synthetic_tree as st

comp = importlib.import_module("pipeline.composites")
CFG = config.load_settings()
NAMES = CFG["margin"]["names"]

SPECS = [("C_01", "InSitu", "test", "biopsy"), ("C_02", "Invasive", "test", "surgical"),
         ("C_03", "Invasive", "test", "biopsy"), ("C_04", "Normal", "test", "biopsy"),
         ("C_05", "Benign", "test", "surgical"), ("C_06", "Benign", "test", "biopsy"),
         ("C_07", "Invasive", "train", "biopsy"), ("C_08", "Normal", "calibration", "biopsy")]


@pytest.mark.parametrize("side,expected", [
    ("left", {"medial": "tumor_side", "lateral": "exact_negative", "superior": "mixed", "inferior": "mixed"}),
    ("right", {"lateral": "tumor_side", "medial": "exact_negative", "superior": "mixed", "inferior": "mixed"}),
    ("top", {"superior": "tumor_side", "inferior": "exact_negative", "medial": "mixed", "lateral": "mixed"}),
    ("bottom", {"inferior": "tumor_side", "superior": "exact_negative", "medial": "mixed", "lateral": "mixed"}),
])
def test_truth_table_for_every_orientation(side, expected):
    assert comp.margin_truth(side, NAMES) == expected


def test_unknown_side_raises():
    with pytest.raises(ValueError):
        comp.margin_truth("diagonal", NAMES)


@pytest.mark.parametrize("side", comp.SIDES)
def test_compose_geometry(side):
    h, w = 6, 8
    carc = np.full((h, w, 3), 200, np.uint8)
    ben = np.full((h, w, 3), 10, np.uint8)
    out = comp.compose(carc, ben, side)
    is_carc = out[..., 0] == 200
    assert is_carc.sum() == h * w // 2
    edge = {"left": is_carc[:, 0], "right": is_carc[:, -1], "top": is_carc[0, :], "bottom": is_carc[-1, :]}
    assert edge[side].all() and not edge[comp.OPPOSITE[side]].any()
    # each margin's truth agrees with the pixels along it
    truth = comp.margin_truth(side, NAMES)
    for s, pixels in edge.items():
        t = truth[NAMES[s]]
        assert (t == "tumor_side") == bool(pixels.all())
        assert (t == "exact_negative") == bool(not pixels.any())


@pytest.fixture()
def built(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path, **{"composites.n": 10})
    data = tmp_path / "data"
    cases = st.write_cases_csv(data, cfg, SPECS)
    code = comp.main(["--config", str(cfg_path), "--data-root", str(data)])
    assert code == 0
    table = pd.read_csv(schemas.composites_csv(data), dtype=str, keep_default_na=False)
    man = pd.read_csv(schemas.embed_manifest_csv(data), dtype=str, keep_default_na=False)
    return cfg, cfg_path, data, cases, table, man


def test_plan_uses_test_images_only_and_balances_orientations(built):
    cfg, _, _, cases, table, _ = built
    schemas.validate_frame(schemas.COMPOSITES, table)
    lab = cases.set_index("case_id")
    assert set(lab.loc[table["carcinoma_case_id"], "split"]) == {"test"}
    assert set(lab.loc[table["benign_case_id"], "split"]) == {"test"}
    assert set(lab.loc[table["carcinoma_case_id"], "label"]) <= set(cfg["carcinoma_classes"])
    assert set(lab.loc[table["benign_case_id"], "label"]) <= set(cfg["benign_classes"])
    assert "C_07" not in set(table["carcinoma_case_id"]) and "C_08" not in set(table["benign_case_id"])
    assert table["carcinoma_side"].value_counts().to_dict() == {"left": 3, "right": 3, "top": 2, "bottom": 2}
    assert list(table["orientation"].astype(int)) == [k % 4 for k in range(10)]
    assert all(cid == schemas.composite_case_id(c, k)
               for k, (cid, c) in enumerate(zip(table["case_id"], table["carcinoma_case_id"])))
    # cycling: each of the 3 carcinoma test cases used 3-4 times
    assert sorted(table["carcinoma_case_id"].value_counts().tolist()) == [3, 3, 4]


def test_composite_pixels_come_from_the_right_halves(built):
    cfg, _, data, cases, table, _ = built
    src = {c: np.asarray(Image.open(p).convert("RGB")) for c, p in zip(cases["case_id"], cases["image_path"])}
    for row in table.itertuples(index=False):
        img = np.asarray(Image.open(data / row.path))
        mask = comp.half_mask(img.shape[1], img.shape[0], row.carcinoma_side)
        np.testing.assert_array_equal(img[mask], src[row.carcinoma_case_id][mask])
        np.testing.assert_array_equal(img[~mask], src[row.benign_case_id][~mask])
        with Image.open(data / row.display_path) as d:
            assert max(d.size) == cfg["tiles"]["display"]["long_side"]


def test_manifest_rows(built):
    cfg, _, data, _, table, man = built
    schemas.validate_frame(schemas.EMBED_MANIFEST, man)
    n = {g: s["rows"] * s["cols"] for g, s in cfg["tiles"]["grids"].items()}
    v = cfg["encoders"]["views"]
    assert (man["set"] == "comp512").sum() == 10 * n["g512"] * v["g512"]
    assert (man["set"] == "comp256").sum() == 10 * n["g256"] * v["g256"]
    c = man[man["set"] == "comp256"]
    assert set(c["split"]) == {"test"} and (c["label"] == "").all()
    assert set(c["path"]) == set(table["path"])
    row = c[(c["case_id"] == table["case_id"].iloc[0]) & (c["tile_idx"] == "9") & (c["view"] == "0")].iloc[0]
    assert (int(row["x"]), int(row["y"]), int(row["w"])) == (8, 8, 8)  # g256 small grid: row 1, col 1


def test_determinism_resume_and_limit(built, tmp_path):
    cfg, cfg_path, data, _, table, _ = built
    probe = data / table["path"].iloc[0]
    mtime = probe.stat().st_mtime_ns
    assert comp.main(["--config", str(cfg_path), "--data-root", str(data)]) == 0
    assert probe.stat().st_mtime_ns == mtime
    again = pd.read_csv(schemas.composites_csv(data), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(again, table)
    assert comp.main(["--config", str(cfg_path), "--data-root", str(data), "--limit", "3"]) == 0
    small = pd.read_csv(schemas.composites_csv(data), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(small, table.iloc[:3])


def test_missing_cases_csv_fails(tmp_path):
    cfg_path, _ = st.small_settings(tmp_path)
    assert comp.main(["--config", str(cfg_path), "--data-root", str(tmp_path / "empty")]) == 2


def test_composites_module_is_portable():
    assert st.delta_import_problems("pipeline/composites.py", st.DELTA_ALLOWED - {"sklearn", "scipy"}) == []
