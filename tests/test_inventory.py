"""pipeline.00_inventory on fake trees: required keys, gate pass/fail, pathmmu comparison, NCT/Quilt/PathGen."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from core import schemas
from tests import synthetic_tree as st

inventory = importlib.import_module("pipeline.00_inventory")


def _trees(base: Path, cfg, *, n_per_class=100, with_patches=False, with_patients=True):
    roots = {
        "bach": st.make_bach_tree(base / "bach", cfg, n_per_class=n_per_class, with_patients=with_patients),
        "nct": st.make_nct_tree(base / "nct", cfg["nct"]["classes"], per_class=4),
        "pathmmu": st.make_pathmmu_bach(base / "pathmmu_bach"),
        "quilt": st.make_quilt_tree(base / "quilt", [
            {"caption": "invasive ductal carcinoma", "pathology": "['Breast', 'Gynecologic']"},
            {"caption": "colon adenoma", "pathology": "['Gastrointestinal']"},
            {"caption": "lobular carcinoma", "pathology": "['Breast pathology']"},
            {"caption": "skin", "pathology": "['Dermatopathology']"},
        ]),
        "pathgen": st.make_pathgen_tree(base / "pathgen", [
            {"wsi_id": "TCGA-AA-3844-01Z-00-DX1.abc", "position": ["35136", "33344"], "caption": "breast tissue"},
            {"wsi_id": "TCGA-BH-0001-01Z-00-DX1.def", "position": ["10", "20"], "caption": "colon"},
        ], with_patches=with_patches),
    }
    return roots


def _argv(cfg_path, data_root, roots):
    return ["--config", str(cfg_path), "--data-root", str(data_root), "--bach-root", str(roots["bach"]),
            "--nct-root", str(roots["nct"]), "--pathmmu-bach-root", str(roots["pathmmu"]),
            "--quilt-root", str(roots["quilt"]), "--pathgen-root", str(roots["pathgen"])]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("inv")
    cfg_path, cfg = st.small_settings(base)
    roots = _trees(base, cfg)
    data_root = base / "data"
    code = inventory.main(_argv(cfg_path, data_root, roots))
    inv = json.loads(schemas.inventory_json(data_root).read_text())
    return code, inv, cfg, data_root


def test_inventory_has_every_required_key_and_passes_the_gate(full_run):
    code, inv, cfg, data_root = full_run
    assert code == 0
    assert schemas.INVENTORY.problems(inv) == []
    assert inv["gate_passed"] is True
    assert inv["per_class_counts"] == {c: 100 for c in cfg["classes"]}
    assert inv["n_images"] == 400
    assert inv["image_size"] == list(st.SMALL_SIZE)
    assert inv[schemas.INVENTORY_MPP_KEY] == cfg["dataset"]["mpp"]
    assert "meaningless" in inv["mpp_source"]
    assert (data_root / "meta" / "00_inventory.json").is_file()


def test_bach_details_ground_truth_bit_depth_wsi_patients(full_run):
    _, inv, _, _ = full_run
    bach = inv["bach"]
    assert bach["ground_truth_csv"]["agrees"] is True
    assert bach["ignored_files"] == {"Thumbs.db": 4, ".DS_Store": 4}
    assert bach["bit_depth_by_file_size"] == {"16": len(st.SIXTEEN_BIT), "8": 400 - len(st.SIXTEEN_BIT)}
    sixteen = [s for s in bach["header_sample"] if Path(s["path"]).name in st.SIXTEEN_BIT]
    assert sixteen and all(s["bits_per_sample"] == [16, 16, 16] for s in sixteen)
    assert inv["wsi"]["n_svs"] == 4 and inv["wsi"]["n_xml"] == 2
    assert inv["wsi"]["annotated"] == ["A01", "A02"]
    assert inv["wsi"]["xml_microns_per_pixel"] == pytest.approx(0.466667)
    assert inv["patient_file"]["present"] and inv["patient_file"]["n_identified"] > 0
    assert inv["patient_file"]["md5_ok"] is False  # generated workbook != the real file's md5
    assert bach["wsi_patient_file"]["n_slides"] == 30 and bach["wsi_patient_file"]["n_patients"] == 10


def test_pathmmu_is_recorded_as_not_iciar(full_run):
    _, inv, _, _ = full_run
    pm = inv["pathmmu_bach"]
    assert pm["present"] and pm["is_iciar2018_bach"] is False
    assert "dataset of record" in pm["note"]
    assert pm["zips"]["BACH.zip"]["n_files"] == 12
    assert pm["label_source"] == "class folders inside zip"


def test_nct_quilt_pathgen_sections(full_run):
    _, inv, cfg, _ = full_run
    assert inv["nct"]["per_class_counts"] == {c: 4 for c in cfg["nct"]["classes"]}
    assert inv["nct"]["tile_size"] == [12, 12]
    assert inv["nct"]["enough_for_sample"] is True  # small settings sample 3 per class
    q = inv["quilt"]
    assert q["csv"] == "quilt_1M_lookup-002.csv"
    assert "caption" in q["columns"] and "pathology" in q["columns"]
    assert q["peek_breast_fraction"] == 0.5
    assert q["image_parts"] == ["images_part_1"]
    assert inv["pathgen"]["has_patch_images"] is False
    assert inv["pathgen"]["n_tcga_svs"] == 2
    assert "wsi_id" in inv["pathgen"]["json_files"][0]["keys_in_head"]


def test_pathgen_with_patch_images(tmp_path):
    _, cfg = st.small_settings(tmp_path)
    root = st.make_pathgen_tree(tmp_path / "pg", [
        {"wsi_id": "TCGA-AA-3844-01Z-00-DX1.abc", "position": ["1", "2"], "caption": "x"}], with_patches=True)
    out = inventory.inventory_pathgen(root, cfg)
    assert out["has_patch_images"] is True and out["n_patch_named_images"] == 1


def test_gate_fails_with_too_few_images(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    roots = _trees(tmp_path, cfg, n_per_class=50, with_patients=False)
    code = inventory.main(_argv(cfg_path, tmp_path / "data", roots))
    inv = json.loads(schemas.inventory_json(tmp_path / "data").read_text())
    assert code == 1
    assert inv["gate_passed"] is False
    assert inv["patient_file"]["present"] is False


def test_missing_bach_tree_fails_gate_but_writes_inventory(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    roots = {k: tmp_path / f"missing_{k}" for k in ("bach", "nct", "pathmmu", "quilt", "pathgen")}
    code = inventory.main(_argv(cfg_path, tmp_path / "data", roots))
    inv = json.loads(schemas.inventory_json(tmp_path / "data").read_text())
    assert code == 1 and inv["n_images"] == 0
    assert inv["nct"]["present"] is False and inv["quilt"]["present"] is False


def test_micro_patient_parser_reads_wide_layout(tmp_path):
    _, cfg = st.small_settings(tmp_path)
    pmap = {"b032.tif": 11, "is040.tif": 22, "n068.tif": 39}
    path = tmp_path / "p.xlsx"
    st.write_micro_patient_xlsx(path, pmap, cfg)
    from pipeline import common

    assert common.load_micro_patients(path) == pmap


def test_wsi_patient_parser(tmp_path):
    path = tmp_path / "w.xlsx"
    st.write_wsi_patient_xlsx(path, n_labeled=3)
    from pipeline import common

    got = common.load_wsi_patients(path)
    assert got["A03"] == {"type": "labeled", "patient": 3}
    assert got["05"]["patient"] == 3 and got["06"]["patient"] == 3 and got["01"]["patient"] == 1
