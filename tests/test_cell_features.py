"""pipeline.04c_cell_features: minimum counts, µm units, pooling, density, TIL proximity, edge rule, benign
reference percentiles, labels never written, disabled flag, end to end after the fake 04b."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from core import schemas
from pipeline import cells_convert as cc
from tests import fake_cells
from tests import synthetic_tree as st

feat = importlib.import_module("pipeline.04c_cell_features")
cells = importlib.import_module("pipeline.04b_cells")
select = importlib.import_module("pipeline.04a_select_tiles")


def nuc(t, x=10.0, y=10.0, area=40.0, ecc=0.5, sol=0.9, edge=False):
    return {"centroid_um": [x, y], "area_um2": area, "perimeter_um": 20.0, "eccentricity": ecc, "solidity": sol,
            "type": t, "type_prob": 0.9, "edge": edge, "contour_px": [[0, 0], [1, 0], [1, 1]]}


def write_tile(root, case_id, tile_idx, nuclei):
    cc.write_tile_json(schemas.cell_json(root, case_id, tile_idx),
                       {"case_id": case_id, "tile_idx": tile_idx, "backend": "t", "model": "m", "mpp_model": 0.25,
                        "mpp_source": 0.42, "scale": 1.68}, nuclei, 0)


@pytest.fixture()
def tree(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    data = tmp_path / "data"
    specs = [("C_0001", "Normal", "train", "biopsy"), ("C_0002", "Benign", "train", "biopsy"),
             ("C_0003", "Invasive", "train", "biopsy"), ("C_0004", "Invasive", "test", "biopsy"),
             ("C_0005", "Normal", "calibration", "biopsy")]
    cases = st.write_cases_csv(data, cfg, specs)
    tiles, _ = st.write_tiles_and_embeddings(data, cfg, cases, blank_tiles=())
    (data / "inventory.json").write_text(json.dumps({"measured_mpp": 0.5}))
    rows = []
    for c in cases["case_id"]:
        for rank, idx in enumerate((3, 4), start=1):
            rows.append({"case_id": c, "tile_idx": idx, "split": cases.set_index("case_id").loc[c, "split"],
                         "rank": rank, "tile_risk": 0.5, "path": f"tiles/g512/{c}/0_{idx}.png", "n_tiles": 2})
    pd.DataFrame(rows).to_csv(schemas.cell_tiles_csv(data), index=False)
    return cfg_path, cfg, data


def _run(cfg_path, data):
    assert feat.main(["--config", str(cfg_path), "--data-root", str(data)]) == 0
    case = pd.read_csv(schemas.cell_features_csv(data), dtype={"case_id": str}).set_index("case_id")
    tile = pd.read_csv(schemas.cell_features_csv(data, per_tile=True), dtype={"case_id": str})
    return case, tile


def test_min_counts_units_pooling_and_edge_rule(tree):
    cfg_path, cfg, data = tree
    # C_0001: 19 non-edge neoplastic + 5 edge neoplastic + 30 connective = 54 nuclei over two tiles
    write_tile(data, "C_0001", 3, [nuc("neoplastic", area=a) for a in range(30, 49)] + [nuc("connective")] * 15)
    write_tile(data, "C_0001", 4, [nuc("neoplastic", area=500, edge=True)] * 5 + [nuc("connective")] * 15)
    # C_0002: 20 non-edge neoplastic (areas 30..49) + 10 inflammatory = 30 nuclei (< 50: ratios missing)
    write_tile(data, "C_0002", 3, [nuc("neoplastic", area=a) for a in range(30, 50)])
    write_tile(data, "C_0002", 4, [nuc("inflammatory")] * 10)
    for c in ("C_0003", "C_0004", "C_0005"):
        write_tile(data, c, 3, [nuc("neoplastic", area=60 + i) for i in range(40)] + [nuc("dead")] * 20)
        write_tile(data, c, 4, [nuc("epithelial")] * 10)
    case, tile = _run(cfg_path, data)
    schemas.validate_frame(schemas.CELL_FEATURES, case.reset_index())
    a = case.loc["C_0001"]
    # 24 neoplastic but only 19 non-edge: morphology missing AND flagged (flag == the blanking rule)
    assert a["n_nuclei"] == 54 and a["n_neoplastic"] == 24 and a["insufficient_tumor_nuclei"] == 1
    assert np.isnan(a["nuc_area_mean"]) and a["nuc_area_mean__missing"] == 1  # 19 non-edge < 20
    assert a["neo_fraction"] == pytest.approx(24 / 54) and a["neo_fraction__missing"] == 0
    b = case.loc["C_0002"]
    assert b["nuc_area_mean"] == pytest.approx(np.mean(range(30, 50)))  # µm² straight from the JSON
    assert b["insufficient_tumor_nuclei"] == 0
    assert b["nuc_area_cv"] == pytest.approx(np.std(range(30, 50)) / np.mean(range(30, 50)))
    assert np.isnan(b["neo_fraction"]) and b["infl_fraction__missing"] == 1  # 30 < 50 total
    # density: pooled neoplastic / pooled tissue (two 16-px tiles at 0.5 µm/px, tissue_frac 0.9)
    tissue_mm2 = 2 * (16 * 0.5 / 1000) ** 2 * 0.9
    assert a["neo_density"] == pytest.approx(24 / tissue_mm2)
    d = case.loc["C_0003"]
    assert d["dead_fraction"] == pytest.approx(20 / 70) and d["epi_fraction"] == pytest.approx(10 / 70)
    # per-tile rows: counts per type and the overlay path
    t = tile[(tile["case_id"] == "C_0001") & (tile["tile_idx"] == 4)].iloc[0]
    assert t["n_neoplastic"] == 5 and t["n_connective"] == 15 and t["n_nuclei"] == 20
    assert t["overlay"] == "cells/C_0001/4_overlay.png"


def test_til_proximity_and_reference_percentiles(tree):
    cfg_path, cfg, data = tree
    r = cfg["cells"]["til_radius_um"]
    base = [nuc("connective")] * 50
    write_tile(data, "C_0001", 3, base + [nuc("neoplastic", 0, 0), nuc("inflammatory", r - 1, 0),
                                          nuc("inflammatory", r + 10, 0)])
    write_tile(data, "C_0001", 4, [nuc("inflammatory", 0, 0)])  # other tile: no neoplastic there
    write_tile(data, "C_0002", 3, base + [nuc("neoplastic")] * 10)
    write_tile(data, "C_0002", 4, [])
    for c, n_neo in (("C_0003", 30), ("C_0004", 5), ("C_0005", 40)):
        write_tile(data, c, 3, base + [nuc("neoplastic")] * n_neo)
        write_tile(data, c, 4, [])
    case, _ = _run(cfg_path, data)
    assert case.loc["C_0001", "til_proximity"] == pytest.approx(1 / 3)
    assert np.isnan(case.loc["C_0002", "til_proximity"])  # no inflammatory nuclei
    # reference = train Normal/Benign = C_0001 (1/53) and C_0002 (10/60) neo_fraction
    ref = sorted([1 / 53, 10 / 60])
    assert case.loc["C_0003", "neo_fraction__ref_pct"] == 1.0  # above both (Invasive train, not in reference)
    assert case.loc["C_0004", "neo_fraction__ref_pct"] == pytest.approx(0.5)  # 5/55 between the two
    assert case.loc["C_0001", "neo_fraction__ref_pct"] == pytest.approx(0.25)  # mid-rank tie with itself
    assert ref[0] < 5 / 55 < ref[1]


def test_labels_never_written_and_meta(tree):
    cfg_path, cfg, data = tree
    for c in ("C_0001", "C_0002", "C_0003", "C_0004", "C_0005"):
        write_tile(data, c, 3, [nuc("neoplastic")] * 30 + [nuc("connective")] * 30)
    case, tile = _run(cfg_path, data)  # tile 4 JSONs missing -> skipped and counted
    text = schemas.cell_features_csv(data).read_text() + schemas.cell_features_csv(data, per_tile=True).read_text()
    for cls in cfg["classes"]:
        assert cls not in text
    meta = json.loads(schemas.step_meta_json(data, "04c_cell_features").read_text())
    assert meta["summary"]["tiles_missing"] == 5 and meta["summary"]["n_reference_cases"] == 2
    assert meta["summary"]["train_medians"]["neo_fraction"] == pytest.approx(0.5)
    assert (case["n_tiles"] == 1).all()


def test_disabled_writes_only_a_record(tmp_path):
    cfg_path, _ = st.small_settings(tmp_path, **{"cells.enabled": False})
    assert feat.main(["--config", str(cfg_path), "--data-root", str(tmp_path / "d")]) == 0
    assert not schemas.cell_features_csv(tmp_path / "d").exists()
    assert json.loads(schemas.step_meta_json(tmp_path / "d", "04c_cell_features").read_text())["status"] == "disabled"


def test_end_to_end_after_fake_segmentation(tmp_path, monkeypatch):
    cfg_path, cfg = st.small_settings(tmp_path, **{"cells.min_total_nuclei": 10, "cells.min_neoplastic_nuclei": 5})
    data = tmp_path / "data"
    cases = st.write_cases_csv(data, cfg, st.case_specs(20, 4, 4))
    st.write_tiles_and_embeddings(data, cfg, cases)
    monkeypatch.setitem(cells.BACKENDS, "tiatoolbox", fake_cells.FakeBackend)
    fake_cells.FakeBackend.fail_on = None
    base = ["--config", str(cfg_path), "--data-root", str(data)]
    assert select.main(base) == 0
    assert cells.main(base + ["--scratch", str(tmp_path / "s")]) == 0
    case, tile = _run(cfg_path, data)
    assert len(case) == len(cases) and (case["n_tiles"] == cfg["cells"]["tiles_per_case"]).all()
    assert (case["n_nuclei"] == 8 * cfg["cells"]["tiles_per_case"]).all()
    assert case["neo_fraction"].between(0, 1).all() and case["nuc_area_mean"].notna().any()
    assert (case["nuc_area_mean"].isna() == (case["nuc_area_mean__missing"] == 1)).all()
    assert case["nuc_area_mean__ref_pct"].dropna().between(0, 1).all()


def test_cell_features_module_is_portable():
    assert st.delta_import_problems("pipeline/04c_cell_features.py", st.DELTA_ALLOWED - {"sklearn", "scipy"}) == []
