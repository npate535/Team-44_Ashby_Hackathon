"""Shared test helpers (Agent 3): one small synthetic data root per test session, plus read helpers.

Import as `from tests.a3_synthetic import synthetic_root, pooled` inside test modules. The root is generated once
per pytest process (pipeline.make_synthetic --small --with-cells) into a temp dir, reused read-only by every test
module (a module-level cache, since tests/conftest.py is master-owned), and removed at exit.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import config, schemas
from pipeline import make_synthetic

CFG = config.load_settings()


_CACHE: dict[str, Path] = {}


def build_root() -> Path:
    if "root" not in _CACHE:
        tmp = Path(tempfile.mkdtemp(prefix="a3syn_"))
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
        root = tmp / "data_synthetic"
        assert make_synthetic.main(["--out", str(root), "--small", "--with-cells"]) == 0
        _CACHE["root"] = root
    return _CACHE["root"]


@pytest.fixture
def synthetic_root() -> Path:
    """Read-only small synthetic root shared by all tests in the process (copy it before writing)."""
    return build_root()


def qc_tiles(root: Path) -> pd.DataFrame:
    tiles = pd.read_csv(schemas.tiles_csv(root))
    return tiles[tiles.qc_pass == 1][["case_id", "grid", "tile_idx"]]


def pooled(root: Path, encoder: str, set_name: str, head: str) -> pd.DataFrame:
    """Mean-pooled QC-passing tile embeddings per case (rows indexed by case_id)."""
    x = np.load(schemas.emb_npy(root, encoder, set_name, head)).astype(float)
    idx = pd.read_csv(schemas.emb_index_csv(root, encoder, set_name, head)).reset_index()
    idx = idx.merge(qc_tiles(root), on=["case_id", "grid", "tile_idx"])
    df = pd.DataFrame(x[idx["index"].to_numpy()])
    df["case_id"] = idx["case_id"].to_numpy()
    return df.groupby("case_id").mean()


def add_fake_concepts(root: Path) -> None:
    """TEST-ONLY stand-in for Agent 2's 04_concepts on a synthetic root: concept z per CONCH align tile (mean cosine
    to positive prompts minus contrasts, z-scored on train QC tiles), case features c__top<k> / c__frac for every
    case and every deg512/comp512 variant (ruling M5), and concepts_tile.npz. Never used outside tests."""
    from pipeline import riskmodels as rm

    cfg = CFG
    z = np.load(schemas.text_npz(root, "conch"))
    ids = list(z["ids"])
    concepts = list(cfg["concepts"]["ids"])
    pos = np.stack([z["emb"][[i for i, s in enumerate(ids) if s.startswith(f"concept:{c}:pos:")]].mean(0) for c in concepts])
    con = np.stack([z["emb"][[i for i, s in enumerate(ids) if s.startswith(f"concept:{c}:con:")]].mean(0) for c in concepts])
    tiles = rm.read_tiles(root)
    cases = rm.read_cases(root).set_index("case_id")
    frames = []
    for set_name in ("bach512", "deg512", "comp512"):
        x, idx = rm.load_emb(root, "conch", set_name, "align")
        raw = x @ pos.T - x @ con.T
        idx = idx.copy()
        idx["read_id"] = [rm.read_id(c, d, s) for c, d, s in zip(idx.case_id, idx.degrade_type.fillna(""),
                                                                 idx.degrade_severity.fillna(0))]
        if set_name == "comp512":
            idx["qc_pass"] = True
        else:
            q = tiles[["case_id", "grid", "tile_idx", "qc_pass"]]
            idx = idx.merge(q, on=["case_id", "grid", "tile_idx"], how="left")
        frames.append((set_name, idx, raw))
    _, idx0, raw0 = frames[0]
    train = ((idx0.split == "train") & idx0.qc_pass).to_numpy()
    mu, sd = raw0[train].mean(0), raw0[train].std(0)
    k = int(cfg["concepts"]["top_k"])
    thr = float(cfg["concepts"]["z_threshold"])
    rows, tile_case, tile_idx, tile_z = [], [], [], []
    for set_name, idx, raw in frames:
        zz = (raw - mu) / sd
        for rid, part in idx.groupby("read_id"):
            use = part[part.qc_pass.astype(bool)] if part.qc_pass.astype(bool).any() else part
            zc = zz[use.index.to_numpy()]
            row = {"case_id": rid, "split": cases.loc[part.case_id.iloc[0], "split"] if set_name != "comp512" else "test"}
            for j, c in enumerate(concepts):
                col = np.sort(zc[:, j])[::-1]
                row[f"{c}__top{k}"] = float(col[:k].mean())
                row[f"{c}__frac"] = float((col > thr).mean())
                row[f"{c}__spread"] = 0.1
            rows.append(row)
            if set_name == "bach512":
                tile_case += [rid] * len(use)
                tile_idx += use.tile_idx.astype(int).tolist()
                tile_z.append(zc)
    df = pd.DataFrame(rows)
    schemas.validate_frame(schemas.CASE_FEATURES, df, extra_required=schemas.concept_feature_columns(cfg))
    df.to_csv(schemas.case_features_csv(root), index=False)
    zt = np.concatenate(tile_z).astype(np.float32)
    np.savez(schemas.concepts_tile_npz(root), case_id=np.asarray(tile_case), tile_idx=np.asarray(tile_idx, np.int32),
             grid=np.asarray("g512"), concept_ids=np.asarray(concepts), z=zt, z_text=zt,
             z_cav=np.full_like(zt, np.nan), spread=np.full_like(zt, 0.1), encoder=np.asarray("conch"),
             synthetic=np.asarray(True))


_WORK: dict[str, Path] = {}


def analysis_root() -> Path:
    """A writable copy of the synthetic root with fake concept features, shared by the CP3 step tests (built once)."""
    if "root" not in _WORK:
        tmp = Path(tempfile.mkdtemp(prefix="a3work_"))
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
        root = tmp / "data"
        shutil.copytree(build_root(), root)
        add_fake_concepts(root)
        _WORK["root"] = root
    return _WORK["root"]


def fast_config(directory: Path) -> Path:
    """A settings.yaml copy with reduced compute for step tests (fewer C values, one CV repeat, short ABMIL,
    smaller bootstrap). Same keys, so every strict read still works."""
    import yaml

    data = yaml.safe_load((config.SETTINGS_PATH).read_text(encoding="utf-8"))
    data["risk_model"]["C_grid"] = [0.01, 0.1]
    data["split"]["cv"]["repeats"] = 1
    data["abmil"]["epochs"] = 8
    data["bootstrap"]["n"] = 200
    path = Path(directory) / "settings_fast.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config.load_settings(path)
    return path
