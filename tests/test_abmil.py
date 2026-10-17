"""pipeline/05b_abmil.py: attention export only when ABMIL is used (weights from 05, never refit)."""

import importlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from core import schemas
from tests.a3_synthetic import after_05, clone_root

step05b = importlib.import_module("pipeline.05b_abmil")


@pytest.fixture
def roots(tmp_path):
    root0, art0, cfg_path = after_05()
    root, art = tmp_path / "data", tmp_path / "art"
    clone_root(root0, root)
    shutil.copytree(art0, art)
    return root, art, cfg_path


def args(root, art, cfg_path):
    return ["--data-root", str(root), "--artifacts-root", str(art), "--config", str(cfg_path)]


def test_no_attention_file_when_abmil_unused(roots):
    root, art, cfg_path = roots
    choice = json.loads(schemas.eval_json(art, "model_choice").read_text())
    assert "abmil" not in (choice["production"], choice["model_a"])
    (root / "tile_attention.csv").write_text("stale")
    assert step05b.main(args(root, art, cfg_path)) == 0
    assert not schemas.tile_attention_csv(root).exists()


def test_attention_written_when_abmil_is_model_a(roots):
    root, art, cfg_path = roots
    path = schemas.eval_json(art, "model_choice")
    choice = json.loads(path.read_text())
    choice["model_a"] = "abmil"
    path.write_text(json.dumps(choice))
    assert step05b.main(args(root, art, cfg_path)) == 0
    att = pd.read_csv(schemas.tile_attention_csv(root), dtype={"case_id": str})
    schemas.validate_frame(schemas.TILE_ATTENTION, att)
    np.testing.assert_allclose(att.groupby("case_id").attention.sum(), 1.0, atol=1e-4)
    assert set(att.split) <= {"calibration", "test"}
    assert att.case_id.str.contains("_FSIM_").any()
    tiles = pd.read_csv(schemas.tiles_csv(root), dtype={"case_id": str})
    clean = att[~att.case_id.str.contains("_FSIM_|_COMP_")].merge(tiles, on=["case_id", "grid", "tile_idx"])
    assert (clean.qc_pass == 1).all()
