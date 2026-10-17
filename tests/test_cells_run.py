"""pipeline.04b_cells with a fake backend: sharding, resume, chunk-failure isolation, --spike template, --verify,
--limit, disabled flag. (The conversion layer itself is tested in test_cells_convert.py.)"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from core import schemas
from tests import fake_cells
from tests import synthetic_tree as st

cells = importlib.import_module("pipeline.04b_cells")
select = importlib.import_module("pipeline.04a_select_tiles")


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    cfg_path, cfg = st.small_settings(tmp_path, **{"cells.chunk_size": 4, "cells.spike_tiles": 5})
    data = tmp_path / "data"
    cases = st.write_cases_csv(data, cfg, st.case_specs(20, 4, 4))
    st.write_tiles_and_embeddings(data, cfg, cases)
    (data / "inventory.json").write_text(json.dumps({"measured_mpp": 0.42}))
    assert select.main(["--config", str(cfg_path), "--data-root", str(data)]) == 0
    fake_cells.FakeBackend.fail_on = None
    fake_cells.FakeBackend.calls = []
    monkeypatch.setitem(cells.BACKENDS, "tiatoolbox", fake_cells.FakeBackend)
    return cfg_path, cfg, data, tmp_path


def _argv(cfg_path, data, tmp, *extra):
    return ["--config", str(cfg_path), "--data-root", str(data), "--artifacts-root", str(tmp / "art"),
            "--scratch", str(tmp / "scratch"), *extra]


def test_shards_cover_every_selected_tile_and_verify_passes(tree):
    cfg_path, cfg, data, tmp = tree
    assert cells.main(_argv(cfg_path, data, tmp, "--verify")) == 1  # nothing segmented yet
    for k in range(2):
        assert cells.main(_argv(cfg_path, data, tmp, "--shard", str(k), "--nshards", "2")) == 0
    assert cells.main(_argv(cfg_path, data, tmp, "--verify")) == 0
    sel = pd.read_csv(schemas.cell_tiles_csv(data), dtype={"case_id": str})
    for c, i in zip(sel["case_id"], sel["tile_idx"]):
        doc = json.loads(schemas.cell_json(data, c, int(i)).read_text())
        assert schemas.CELL_JSON.problems(doc) == []
        assert doc["n_nuclei"] == 8 and doc["n_untyped"] == 1 and doc["backend"] == "fake==0"
        assert doc["mpp_source"] == 0.42 and doc["scale"] == pytest.approx(0.42 / 0.25, rel=0.02)
        assert schemas.cell_overlay_png(data, c, int(i)).is_file()
    meta = json.loads(schemas.step_meta_json(data, "04b_cells").read_text())
    assert meta["summary"]["tiles_missing"] == 0 and meta["summary"]["shards"] == 2
    assert meta["summary"]["nuclei_total"] == 8 * len(sel)


def test_resume_force_and_chunking(tree):
    cfg_path, cfg, data, tmp = tree
    assert cells.main(_argv(cfg_path, data, tmp)) == 0
    n = len(pd.read_csv(schemas.cell_tiles_csv(data)))
    assert sum(fake_cells.FakeBackend.calls) == n and max(fake_cells.FakeBackend.calls) <= 4
    fake_cells.FakeBackend.calls = []
    assert cells.main(_argv(cfg_path, data, tmp)) == 0
    assert fake_cells.FakeBackend.calls == []  # everything done: backend never constructed/called
    assert cells.main(_argv(cfg_path, data, tmp, "--force", "--limit", "2")) == 0
    assert sum(fake_cells.FakeBackend.calls) == 2 * cfg["cells"]["tiles_per_case"]


def test_failing_chunk_is_isolated(tree):
    cfg_path, cfg, data, tmp = tree
    sel = pd.read_csv(schemas.cell_tiles_csv(data), dtype={"case_id": str}).sort_values(["case_id", "tile_idx"])
    fake_cells.FakeBackend.fail_on = sel["case_id"].iloc[0]
    assert cells.main(_argv(cfg_path, data, tmp)) == 1
    meta = json.loads(schemas.step_meta_json(data, "04b_cells_shard0").read_text())
    n_bad = int((sel["case_id"] == sel["case_id"].iloc[0]).sum())
    assert len(meta["summary"]["failed_tiles"]) == n_bad  # per-tile retry: only the bad case's tiles are lost
    assert meta["summary"]["tiles_done"] == len(sel) - n_bad
    fake_cells.FakeBackend.fail_on = None
    assert cells.main(_argv(cfg_path, data, tmp)) == 0  # resume finishes the lost chunk
    assert cells.main(_argv(cfg_path, data, tmp, "--verify")) == 0


def test_spike_writes_review_template_without_labels(tree):
    cfg_path, cfg, data, tmp = tree
    assert cells.main(_argv(cfg_path, data, tmp, "--spike")) == 0
    md = (tmp / "art" / "eval" / "cells_spike.md").read_text()
    assert "reviewed by Claude (AI), not a pathologist" in md and "TEMPLATE" in md
    rows = [line for line in md.splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert len(rows) == cfg["cells"]["spike_tiles"]
    assert len({r.split("|")[2] for r in rows}) == len(rows)  # different cases
    for cls in cfg["classes"]:
        assert cls not in md  # labels pick cases but are never written
    meta = json.loads(schemas.step_meta_json(data, "04b_cells_spike").read_text())
    assert meta["summary"]["tiles_done"] == cfg["cells"]["spike_tiles"]


def test_spike_tiles_one_per_class_first(tree):
    cfg_path, cfg, data, tmp = tree
    cases = pd.read_csv(schemas.cases_csv(data), dtype=str, keep_default_na=False)
    tiles = pd.read_csv(schemas.tiles_csv(data), dtype=str, keep_default_na=False)
    rows = cells.spike_tiles(cases, tiles, cfg)
    labels = cases.set_index("case_id").loc[rows["case_id"], "label"]
    assert set(labels.iloc[:4]) == set(cfg["classes"]) and rows["case_id"].is_unique


def test_unknown_backend_assets_and_disabled(tree, tmp_path):
    cfg_path, cfg, data, tmp = tree
    assert cells.main(_argv(cfg_path, data, tmp, "--backend", "hovernext")) == 2
    assert cells.main(_argv(cfg_path, data, tmp, "--backend", "hovernet")) == 2  # repo/checkpoint absent
    (tmp_path / "off").mkdir()
    off, _ = st.small_settings(tmp_path / "off", **{"cells.enabled": False})
    assert cells.main(["--config", str(off), "--data-root", str(tmp_path / "d2")]) == 0


def test_cells_run_module_is_portable():
    assert st.delta_import_problems("pipeline/04b_cells.py", st.DELTA_ALLOWED - {"sklearn", "scipy"} | {"tiatoolbox", "zarr"}) == []
    src = (st.REPO / "pipeline" / "04b_cells.py").read_text()
    assert "\nimport tiatoolbox" not in src and "\nimport zarr" not in src


def test_spike_rerun_remeasures_and_backend_switch_redoes_tiles(tree):
    cfg_path, cfg, data, tmp = tree
    assert cells.main(_argv(cfg_path, data, tmp, "--spike")) == 0
    fake_cells.FakeBackend.calls = []
    assert cells.main(_argv(cfg_path, data, tmp, "--spike")) == 0  # rerun: measured again, never "OK from nothing"
    assert sum(fake_cells.FakeBackend.calls) == cfg["cells"]["spike_tiles"]
    assert "NOT MEASURED" not in (tmp / "art" / "eval" / "cells_spike.md").read_text()
    assert cells.main(_argv(cfg_path, data, tmp)) == 0
    assert cells.main(_argv(cfg_path, data, tmp, "--verify")) == 0
    # outputs from another backend do not count as done
    sel = pd.read_csv(schemas.cell_tiles_csv(data), dtype={"case_id": str})
    p = schemas.cell_json(data, sel["case_id"].iloc[0], int(sel["tile_idx"].iloc[0]))
    doc = json.loads(p.read_text())
    doc["backend_name"] = "hovernet"
    p.write_text(json.dumps(doc))
    assert cells.main(_argv(cfg_path, data, tmp, "--verify")) == 1
    fake_cells.FakeBackend.calls = []
    assert cells.main(_argv(cfg_path, data, tmp)) == 0
    assert sum(fake_cells.FakeBackend.calls) == 1
    assert json.loads(p.read_text())["backend_name"] == "tiatoolbox"
