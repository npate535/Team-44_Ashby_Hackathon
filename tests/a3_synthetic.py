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
