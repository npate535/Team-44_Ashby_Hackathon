"""pipeline.04a_select_tiles: top-k QC tiles per case by out-of-fold tile risk, all splits, contract call to
pipeline.tile_probe.oof_tile_risk, reuse of tile_risk_oof.csv, disabled flag."""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from core import schemas
from tests import synthetic_tree as st

sel = importlib.import_module("pipeline.04a_select_tiles")


@pytest.fixture()
def tree(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    data = tmp_path / "data"
    cases = st.write_cases_csv(data, cfg, st.case_specs())
    tiles, _ = st.write_tiles_and_embeddings(data, cfg, cases, blank_tiles=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    return cfg_path, cfg, data, cases, tiles


def _argv(cfg_path, data, *extra):
    return ["--config", str(cfg_path), "--data-root", str(data), *extra]


def test_select_real_probe_all_splits(tree):
    cfg_path, cfg, data, cases, tiles = tree
    assert sel.main(_argv(cfg_path, data)) == 0
    out = pd.read_csv(schemas.cell_tiles_csv(data), dtype={"case_id": str})
    schemas.validate_frame(schemas.CELL_TILES, out)
    assert set(out["split"]) == {"train", "calibration", "test"}
    assert out["case_id"].nunique() == len(cases)
    # only tiles 10 and 11 pass QC: 2 per case, n_tiles records it
    assert (out.groupby("case_id").size() == 2).all() and (out["n_tiles"] == 2).all()
    assert set(out["tile_idx"]) == {10, 11}
    risk = pd.read_csv(schemas.tile_risk_csv(data, oof=True), dtype={"case_id": str})
    schemas.validate_frame(schemas.TILE_RISK_OOF, risk)
    assert (risk.loc[risk["split"] == "train", "fold"] >= 0).all()
    assert (risk.loc[risk["split"] != "train", "fold"] == -1).all()
    assert set(risk["probe"]) == {cfg["cells"]["probe_name"]}
    for cid, g in out.groupby("case_id"):
        assert list(g["rank"]) == [1, 2] and g["tile_risk"].is_monotonic_decreasing
    meta = json.loads(schemas.step_meta_json(data, "04a_select_tiles").read_text())
    assert meta["summary"]["train_tiles_out_of_fold"] is True


def test_contract_call_and_reuse(tree, monkeypatch):
    cfg_path, cfg, data, cases, tiles = tree
    calls = []

    def fake(emb, index, cases_, cfg_, *, probe_name, folds, qc_tiles=None):
        calls.append({"n": len(emb), "probe_name": probe_name, "folds": folds,
                      "grids": set(qc_tiles["grid"]) if qc_tiles is not None else None})
        r = index[["case_id", "split", "grid", "tile_idx"]].copy()
        r["risk"] = r["tile_idx"].astype(int) / 100.0
        r["probe"] = probe_name
        r["fold"] = [0 if s == "train" else -1 for s in r["split"]]
        return r

    monkeypatch.setattr(sel.tile_probe, "oof_tile_risk", fake)
    assert sel.main(_argv(cfg_path, data)) == 0
    assert calls == [{"n": len(tiles), "probe_name": cfg["cells"]["probe_name"], "folds": cfg["cells"]["oof_folds"],
                      "grids": {cfg["tiles"]["contract_grid"]}}]
    out = pd.read_csv(schemas.cell_tiles_csv(data), dtype={"case_id": str})
    assert (out.loc[out["rank"] == 1, "tile_idx"] == 11).all()
    assert sel.main(_argv(cfg_path, data, "--limit", "3")) == 0  # reuses tile_risk_oof.csv: no new probe call
    assert len(calls) == 1
    assert pd.read_csv(schemas.cell_tiles_csv(data))["case_id"].nunique() == 3
    assert sel.main(_argv(cfg_path, data, "--force")) == 0
    assert len(calls) == 2


def test_select_tiles_ties_and_k():
    from core import config

    cfg = config.load_settings()
    tiles = pd.DataFrame({"case_id": ["A_1"] * 5, "grid": ["g512"] * 5, "tile_idx": range(5), "qc_pass": [1, 1, 1, 1, 0],
                          "path": [f"p{i}" for i in range(5)]})
    risk = pd.DataFrame({"case_id": ["A_1"] * 5, "split": ["test"] * 5, "grid": ["g512"] * 5, "tile_idx": range(5),
                         "risk": [0.5, 0.9, 0.5, 0.1, 0.99], "probe": "p", "fold": -1})
    cases = pd.DataFrame({"case_id": ["A_1"], "split": ["test"]})
    out = sel.select_tiles(risk, tiles, cases, cfg)
    assert list(out["tile_idx"]) == [1, 0, 2]  # tile 4 fails QC; tie 0.5 broken by tile_idx
    assert list(out["path"]) == ["p1", "p0", "p2"]


def test_missing_embeddings_fail(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    data = tmp_path / "data"
    cases = st.write_cases_csv(data, cfg, st.case_specs(8, 2, 2))
    st.write_tiles_and_embeddings(data, cfg, cases)
    schemas.emb_npy(data, "conch", "bach512", "probe").unlink()
    assert sel.main(_argv(cfg_path, data)) == 2


def test_disabled_flag(tmp_path):
    cfg_path, _ = st.small_settings(tmp_path, **{"cells.enabled": False})
    assert sel.main(_argv(cfg_path, tmp_path / "data")) == 0
    meta = json.loads(schemas.step_meta_json(tmp_path / "data", "04a_select_tiles").read_text())
    assert meta["status"] == "disabled"


def test_select_tiles_is_portable():
    assert st.delta_import_problems("pipeline/04a_select_tiles.py") == []


def test_stale_tile_risk_is_recomputed(tree, monkeypatch):
    import os

    cfg_path, cfg, data, cases, tiles = tree
    assert sel.main(_argv(cfg_path, data)) == 0
    risk_path = schemas.tile_risk_csv(data, oof=True)
    calls = []
    real = sel.tile_probe.oof_tile_risk
    monkeypatch.setattr(sel.tile_probe, "oof_tile_risk", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert sel.main(_argv(cfg_path, data)) == 0 and calls == []  # fresh: reused
    emb = schemas.emb_npy(data, "conch", "bach512", "probe")
    t = risk_path.stat().st_mtime + 10
    os.utime(emb, (t, t))  # embeddings recomputed after the risks
    assert sel.main(_argv(cfg_path, data)) == 0 and calls == [1]
