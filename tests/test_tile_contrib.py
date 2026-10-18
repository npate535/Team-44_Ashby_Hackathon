"""pipeline/05d_tile_contrib.py: exact additive region contributions of the production linear read."""

import importlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from core import config, schemas
from pipeline import riskmodels as rm
from tests.a3_synthetic import after_05, clone_root

tc = importlib.import_module("pipeline.05d_tile_contrib")
CFG = config.load_settings()
CLASSES = list(CFG["classes"])
TARGET = CLASSES.index(CFG["aggressive_class"])
GRIDS = CFG["tiles"]["grids"]


def random_model(dim: int, seed: int = 0) -> rm.LinearModel:
    rng = np.random.default_rng(seed)
    m = rm.LinearModel(CLASSES, 1.0, 1, None)
    m.median, m.mean = rng.normal(size=dim), rng.normal(size=dim)
    m.scale = rng.uniform(0.5, 2.0, size=dim)
    m.coef, m.intercept = rng.normal(size=(len(CLASSES), dim)), rng.normal(size=len(CLASSES))
    return m


def centered(m: rm.LinearModel, rows: np.ndarray, temperature: float) -> float:
    lg = m.logits(rows.mean(0, keepdims=True))[0]
    return float((lg[TARGET] - lg.mean()) / temperature)


def index_frame(case: str, grid: str, tile_ids, **extra) -> pd.DataFrame:
    n = len(tile_ids)
    base = {"item_id": "x", "set": "s", "case_id": case, "split": "test", "grid": grid, "view": -1, "n_views": 1,
            "degrade_type": "", "degrade_severity": 0, "label": "", "group": ""}
    out = pd.DataFrame([{**base, **extra, "tile_idx": int(t)} for t in tile_ids])
    out.insert(0, "emb_row", np.arange(n))
    return out


def test_sum_matches_centered_logit():
    rng = np.random.default_rng(1)
    m = random_model(16)
    rows = rng.normal(size=(24, 16))                   # 12 tiles x 2 views
    tile_of_row = np.repeat(np.arange(12), 2)
    contrib, constant = tc.read_contributions(m, rows, tile_of_row, 0.7, TARGET)
    assert sorted(contrib.index.tolist()) == list(range(12))
    assert abs(constant + contrib.sum() - centered(m, rows, 0.7)) < 1e-9


def test_variant_contributions_aggregate_to_contract_grid():
    rng = np.random.default_rng(2)
    native = tc.grid_tiles(GRIDS["g256"])
    contract = tc.grid_tiles(GRIDS["g512"])
    assert len(native) == 48 and len(contract) == 12
    contrib = pd.Series(rng.normal(size=48), index=native["tile_idx"].to_numpy())
    regions = tc.to_contract_grid(contrib, native, contract)
    assert sorted(regions.index.tolist()) == list(range(12))
    assert regions.sum() == pytest.approx(contrib.sum(), abs=1e-12)
    cols = int(GRIDS["g256"]["cols"])
    for region in range(12):
        r, c = divmod(region, int(GRIDS["g512"]["cols"]))
        subs = [(2 * r + dr) * cols + 2 * c + dc for dr in (0, 1) for dc in (0, 1)]
        assert regions[region] == pytest.approx(contrib[subs].sum(), abs=1e-12)


def test_variant_reads_are_summed_on_the_contract_grid_and_stay_exact():
    """A 256-px variant read (fallback model) yields 512-px region rows whose sum plus constant is exact."""
    rng = np.random.default_rng(3)
    m = random_model(8, seed=3)
    idx = index_frame("A", "g256", range(48), degrade_type="blur", degrade_severity=3)
    x = rng.normal(size=(48, 8)).astype(np.float32)
    qc = pd.DataFrame({"case_id": "A", "grid": "g256", "tile_idx": range(48), "qc_pass": True})
    rows = tc.set_contributions(m, x, idx, qc, {"A_FSIM_blur3": 0.5}, TARGET, CFG)
    assert sorted(rows.tile_idx) == list(range(12)) and set(rows.read_id) == {"A_FSIM_blur3"}
    total = rows.contribution.sum() + rows.constant.iloc[0]
    assert total == pytest.approx(centered(m, x.astype(np.float64), 0.5), abs=1e-9)
    assert (rows.n_rows == 48).all()


def test_qc_failed_tiles_are_null():
    rng = np.random.default_rng(4)
    m = random_model(8, seed=4)
    x = rng.normal(size=(12, 8)).astype(np.float32)
    idx = index_frame("B", "g512", range(12))
    failed = [3, 7]
    qc = pd.DataFrame({"case_id": "B", "grid": "g512", "tile_idx": range(12),
                       "qc_pass": [t not in failed for t in range(12)]})
    rows = tc.set_contributions(m, x, idx, qc, {"B": 0.9}, TARGET, CFG)
    assert set(rows.tile_idx) == set(range(12)) - set(failed)       # excluded from pooling: no row -> null
    assert tc.region_values(rows, "B", range(12))[3] is None
    assert tc.region_values(rows, "B", range(12))[7] is None
    passing = np.array([t not in failed for t in range(12)])
    total = rows.contribution.sum() + rows.constant.iloc[0]
    assert total == pytest.approx(centered(m, x[passing].astype(np.float64), 0.9), abs=1e-9)


def test_missing_embeddings_give_null(tmp_path):
    """A read whose model is not an embedding probe, or whose embeddings are absent, gets no rows (null)."""
    m = random_model(4, seed=5)
    (tmp_path / "models").mkdir()
    m.to_npz(tmp_path / "models" / "ghost_lr.npz")
    cfg = json.loads(json.dumps(CFG))
    cfg["risk_model"]["embedding_candidates"]["ghost_lr"] = {"encoder": "ghost", "set": "bach512", "head": "probe"}
    pred = pd.DataFrame({"case_id": ["C1", "C2", "C3"], "split": ["test", "test", "train"],
                         "model": ["ghost_lr", "cb_text", "ghost_lr"], "temperature": [1.0, 1.0, 1.0]})
    tiles = pd.DataFrame(columns=["case_id", "grid", "tile_idx", "qc_pass"])
    table, null = tc.compute(tmp_path, cfg, pred, tiles)
    assert table.empty
    assert set(null) == {"C1", "C2"}                                     # train OOF rows are not served reads
    assert "embeddings" in null["C1"] and "embedding probe" in null["C2"]
    assert tc.region_values(table, "C1", range(12)) == [None] * 12


# ---- the step on the synthetic root -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run_step(tmp_path_factory):
    root0, art0, cfg_path = after_05()
    work = tmp_path_factory.mktemp("contrib")
    root, art = work / "data", work / "art"
    clone_root(root0, root)
    shutil.copytree(art0, art)
    assert tc.main(["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]) == 0
    return root, art


def test_step_reproduces_predictions_within_tolerance(run_step):
    root, art = run_step
    table = pd.read_csv(schemas.tile_contrib_csv(root), dtype={"read_id": str})
    assert list(table.columns) == ["read_id", "model", "tile_idx", "contribution", "constant", "target",
                                   "temperature", "n_rows"]
    assert len(table) and set(table.target) == {CFG["aggressive_class"]}
    doc = json.loads(schemas.eval_json(art, "tile_contrib").read_text())
    for key in ("name", "created_utc", "config_hash", "git_sha", "synthetic"):
        assert key in doc
    ident = doc["identity"]
    assert ident["n_reads_checked"] > 0 and ident["max_abs_error"] < 1e-6
    pred = pd.read_csv(schemas.predictions_csv(root), dtype={"case_id": str}).set_index("case_id")
    for rid, part in table.groupby("read_id"):
        p = pred.loc[rid, [schemas.cal_prob_col(c) for c in CLASSES]].to_numpy(float)
        if p.min() > 1e-12:
            want = np.log(p[TARGET]) - np.log(p).mean()
            assert abs(part.contribution.sum() + part.constant.iloc[0] - want) < 1e-6
        assert (part.model == pred.loc[rid, "model"]).all()
