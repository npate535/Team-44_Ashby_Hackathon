"""THUNDER split reproduction, patient disjointness, opaque case IDs and lane coverage (BUILD_PLAN §A2)."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from core import config
from pipeline import common
from tests import synthetic_tree as st

build = importlib.import_module("pipeline.01_build_cases")
CFG = config.load_settings()

# Published THUNDER test files (data_formats.md §2): b042-059, b061-090, is040-069, iv041-058, iv061-073, n046-068.
THUNDER_TEST = ([f"b{i:03d}.tif" for i in list(range(42, 60)) + list(range(61, 91))]
                + [f"is{i:03d}.tif" for i in range(40, 70)]
                + [f"iv{i:03d}.tif" for i in list(range(41, 59)) + list(range(61, 74))]
                + [f"n{i:03d}.tif" for i in range(46, 69)])
THUNDER_CAL = ([f"b{i:03d}.tif" for i in range(16, 26)] + [f"is{i:03d}.tif" for i in range(11, 21)]
               + [f"iv{i:03d}.tif" for i in range(1, 11)] + [f"n{i:03d}.tif" for i in list(range(6, 16)) + list(range(81, 91))])


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    base = tmp_path_factory.mktemp("split")
    cfg_path, cfg = st.small_settings(base)
    root = st.make_bach_tree(base / "bach", cfg, with_wsi=False)
    return cfg, root


def _listing(cfg, root):
    return build.build_listing(common.bach_photos_dir(root, cfg), cfg)


def test_listing_order_and_junk_filtering(tree):
    cfg, root = tree
    df = _listing(cfg, root)
    assert len(df) == 400 and list(df["source_index"]) == list(range(400))
    assert df.loc[0, "file"] == "b001.tif" and df.loc[100, "file"] == "is001.tif"
    assert df.loc[200, "file"] == "iv001.tif" and df.loc[300, "file"] == "n001.tif"
    assert df.loc[0, "source_file"] == "Photos/Benign/b001.tif"
    assert not df["file"].str.contains("Thumbs|DS_Store").any()


def test_thunder_split_reproduced_exactly(tree):
    cfg, root = tree
    df = build.assign_splits(_listing(cfg, root), cfg)
    assert df["split"].value_counts().to_dict() == {"train": 218, "test": 132, "calibration": 50}
    assert sorted(df.loc[df["split"] == "test", "file"]) == sorted(THUNDER_TEST)
    assert sorted(df.loc[df["split"] == "calibration", "file"]) == sorted(THUNDER_CAL)
    per_class = df.groupby(["split", "label"]).size().unstack().loc[:, ["Benign", "InSitu", "Invasive", "Normal"]]
    assert per_class.loc["train"].tolist() == [42, 60, 59, 57]
    assert per_class.loc["calibration"].tolist() == [10, 10, 10, 20]
    assert per_class.loc["test"].tolist() == [48, 30, 31, 23]


def test_wrong_class_count_raises(tmp_path):
    _, cfg = st.small_settings(tmp_path)
    root = st.make_bach_tree(tmp_path / "bach", cfg, n_per_class=99, with_patients=False, with_wsi=False)
    with pytest.raises(ValueError, match="per class"):
        _listing(cfg, root)


def test_patient_groups_and_disjointness(tree):
    cfg, root = tree
    pmap = common.load_micro_patients(root / cfg["dataset"]["patients"]["micro_file"])
    df = build.assign_patients(build.assign_splits(_listing(cfg, root), cfg), pmap, cfg)
    build.assert_patient_disjoint(df)
    row = df[df["file"] == "b032.tif"].iloc[0]
    assert row["patient_id"] == str(pmap["b032.tif"]) and row["group"] == f"p{pmap['b032.tif']}"
    row = df[df["file"] == "b001.tif"].iloc[0]
    assert row["patient_id"] == "" and row["group"] == "b0"
    row = df[df["file"] == "is001.tif"].iloc[0]  # source_index 100 -> block 20 with block_size 5
    assert row["group"] == f"b{100 // cfg['split']['cv']['block_size']}"


def test_overlapping_patient_workbook_fires_the_assertion(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    pmap = st.default_patients(cfg)
    pmap["b001.tif"] = pmap["b042.tif"]  # b001 is train, b042 is test: same patient across splits
    root = st.make_bach_tree(tmp_path / "bach", cfg, patients=pmap, with_wsi=False)
    pm = common.load_micro_patients(root / cfg["dataset"]["patients"]["micro_file"])
    df = build.assign_patients(build.assign_splits(_listing(cfg, root), cfg), pm, cfg)
    assert build.patient_overlap(df)["train&test"] == [str(pmap["b042.tif"])]
    with pytest.raises(ValueError, match="leakage"):
        build.assert_patient_disjoint(df)
    code = build.main(["--config", str(cfg_path), "--data-root", str(tmp_path / "data"), "--bach-root", str(root),
                       "--nct-root", str(tmp_path / "nonexistent")])
    assert code == 2
    assert not (tmp_path / "data" / "cases.csv").exists()


def test_patient_workbook_listing_unknown_file_raises(tree):
    cfg, root = tree
    df = build.assign_splits(_listing(cfg, root), cfg)
    with pytest.raises(ValueError, match="not in the photo listing"):
        build.assign_patients(df, {"zz999.tif": 1}, cfg)


def test_case_ids_are_opaque_unique_and_seeded(tree):
    cfg, root = tree
    df = build.assign_case_ids(build.assign_splits(_listing(cfg, root), cfg), cfg)
    ids = df["case_id"]
    assert ids.is_unique and ids.str.fullmatch(r"BACH_\d{4}").all()
    nums = ids.str[-4:].astype(int)
    assert sorted(nums) == list(range(1, 401))
    # opaque: the number is not the sorted index, and class blocks are scrambled
    assert (nums.values != df["source_index"].values + 1).mean() > 0.9
    first_benign = sorted(nums[df["label"] == "Benign"])[:10]
    assert first_benign != list(range(1, 11))
    again = build.assign_case_ids(build.assign_splits(_listing(cfg, root), cfg), cfg)
    assert list(again["case_id"]) == list(ids)


def test_lanes_fraction_coverage_and_determinism(tree):
    cfg, root = tree
    df = build.assign_lanes(build.assign_case_ids(build.assign_splits(_listing(cfg, root), cfg), cfg), cfg)
    surg = df[df["lane"] == "surgical"]
    assert set(surg["split"]) == {"test"}
    assert len(surg) == round(cfg["lanes"]["surgical_fraction"] * 132)
    for group in cfg["lanes"]["surgical_must_cover"]:
        assert surg["label"].isin(group).any()
    again = build.assign_lanes(build.assign_case_ids(build.assign_splits(_listing(cfg, root), cfg), cfg), cfg)
    assert list(again["lane"]) == list(df["lane"])


def test_lanes_must_cover_is_enforced_even_with_tiny_fraction(tree):
    cfg, root = tree
    tiny = {**cfg, "lanes": {**cfg["lanes"], "surgical_fraction": 0.0}}
    df = build.assign_lanes(build.assign_case_ids(build.assign_splits(_listing(cfg, root), cfg), cfg), tiny)
    surg = df[df["lane"] == "surgical"]
    assert len(surg) == len(cfg["lanes"]["surgical_must_cover"])
    for group in cfg["lanes"]["surgical_must_cover"]:
        assert surg["label"].isin(group).sum() >= 1


def test_lanes_raise_when_a_group_is_absent():
    cfg = CFG
    df = pd.DataFrame({"case_id": ["A_1", "A_2"], "split": ["test", "test"], "label": ["Normal", "Benign"]})
    with pytest.raises(ValueError, match="surgical lane"):
        build.assign_lanes(df, cfg)
