"""pipeline.01_build_cases end to end on a scaled synthetic tree: CSV schemas, tile geometry, QC, displays,
manifest sets/views/crop boxes, --limit, resume/idempotence, aux-set preservation."""

from __future__ import annotations

import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from core import schemas
from pipeline import common, degrade
from tests import synthetic_tree as st

build = importlib.import_module("pipeline.01_build_cases")


def _argv(cfg_path, data_root, bach, nct, *extra):
    return ["--config", str(cfg_path), "--data-root", str(data_root), "--bach-root", str(bach),
            "--nct-root", str(nct), *extra]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("b01")
    cfg_path, cfg = st.small_settings(base)
    bach = st.make_bach_tree(base / "bach", cfg, with_wsi=False)
    nct = st.make_nct_tree(base / "nct", cfg["nct"]["classes"], per_class=5)
    data = base / "data"
    code = build.main(_argv(cfg_path, data, bach, nct, "--workers", "4"))
    assert code == 0
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    return {"cfg": cfg, "cfg_path": cfg_path, "bach": bach, "nct": nct, "data": data,
            "cases": read(schemas.cases_csv(data)), "tiles": read(schemas.tiles_csv(data)),
            "manifest": read(schemas.embed_manifest_csv(data))}


def test_cases_csv_schema_and_content(run):
    cases, cfg = run["cases"], run["cfg"]
    schemas.validate_frame(schemas.CASES, cases)
    assert len(cases) == 400 and list(cases.columns) == [c.name for c in schemas.CASES.columns]
    assert list(cases["case_id"]) == sorted(cases["case_id"])
    assert set(cases["source"]) == {"bach_official"}
    assert (cases["width"].astype(int) == st.SMALL_SIZE[0]).all()
    assert all(os.path.isabs(p) and os.path.isfile(p) for p in cases["image_path"])
    assert (cases["display_path"] == "display/" + cases["case_id"] + ".jpg").all()
    assert set(cases.loc[cases["lane"] == "surgical", "split"]) == {"test"}
    assert cases.groupby("split").size().to_dict() == cfg["split"]["expected_counts"]


def test_tiles_geometry_and_qc(run):
    tiles, cfg, data = run["tiles"], run["cfg"], run["data"]
    schemas.validate_frame(schemas.TILES, tiles)
    n = {g: s["rows"] * s["cols"] for g, s in cfg["tiles"]["grids"].items()}
    assert tiles.groupby("grid").size().to_dict() == {g: 400 * k for g, k in n.items()}
    for grid, spec in cfg["tiles"]["grids"].items():
        t = tiles[tiles["grid"] == grid].astype({"tile_idx": int, "row": int, "col": int, "x": int, "y": int,
                                                 "size": int})
        assert (t["tile_idx"] == t["row"] * spec["cols"] + t["col"]).all()
        assert (t["x"] == t["col"] * spec["size"]).all() and (t["y"] == t["row"] * spec["size"]).all()
        assert (t["size"] == spec["size"]).all()
        assert (t["path"] == "tiles/" + grid + "/" + t["case_id"] + "/" + t["row"].astype(str) + "_"
                + t["col"].astype(str) + ".png").all()
    # synthetic photos: the top-left contract-tile area is white -> g512 tile 0 and g256 tiles 0,1,8,9 fail QC
    g512 = tiles[tiles["grid"] == "g512"]
    assert set(g512.loc[g512["qc_pass"] == "0", "tile_idx"]) == {"0"}
    g256 = tiles[tiles["grid"] == "g256"]
    assert set(g256.loc[g256["qc_pass"] == "0", "tile_idx"]) == {"0", "1", "8", "9"}
    assert g512.loc[g512["qc_pass"] == "1", "tissue_frac"].astype(float).min() >= cfg["qc"]["min_tissue_frac"]
    # a tile PNG is the exact crop of the decoded photo (16-bit source included)
    case = run["cases"][run["cases"]["image_path"].str.endswith("b001.tif")].iloc[0]
    src = np.asarray(common.read_rgb(case["image_path"]))
    row = g512[(g512["case_id"] == case["case_id"]) & (g512["tile_idx"] == "5")].iloc[0]
    x, y, s = int(row["x"]), int(row["y"]), int(row["size"])
    np.testing.assert_array_equal(np.asarray(Image.open(data / row["path"])), src[y:y + s, x:x + s])


def test_display_jpegs_and_degraded_variants(run):
    cases, cfg, data = run["cases"], run["cfg"], run["data"]
    long_side = cfg["tiles"]["display"]["long_side"]
    for p in cases["display_path"]:
        with Image.open(data / p) as im:
            assert max(im.size) == long_side and im.mode == "RGB"
    sev = cfg["degrade"]["display_severity"]
    surg = cases[cases["lane"] == "surgical"]
    for cid in surg["case_id"]:
        for kind in cfg["degrade"]["types"]:
            assert schemas.display_jpg(data, schemas.variant_case_id(cid, kind, sev)).is_file()
    n_variants = len(list((data / "display").glob("*_FSIM_*.jpg")))
    assert n_variants == len(surg) * len(cfg["degrade"]["types"])


def test_degraded_display_matches_degrade_apply_with_manifest_seed(run):
    cases, cfg, data, man = run["cases"], run["cfg"], run["data"], run["manifest"]
    sev = cfg["degrade"]["display_severity"]
    case = cases[cases["lane"] == "surgical"].iloc[0]
    vid = schemas.variant_case_id(case["case_id"], "stain", sev)
    seed = int(man.loc[(man["set"] == "deg256") & (man["case_id"] == vid), "degrade_seed"].iloc[0])
    want = common.to_display(degrade.apply(common.read_rgb(case["image_path"]), "stain", sev, seed, cfg),
                             cfg["tiles"]["display"]["long_side"])
    got = np.asarray(Image.open(schemas.display_jpg(data, vid)).convert("RGB"), float)
    err = lambda s: np.abs(got - np.asarray(common.to_display(degrade.apply(  # noqa: E731
        common.read_rgb(case["image_path"]), "stain", sev, s, cfg), cfg["tiles"]["display"]["long_side"]), float)).mean()
    assert np.abs(got - np.asarray(want, float)).mean() == err(seed)
    assert err(seed) < min(err(seed + k) for k in range(1, 6))  # the manifest seed is the one the display used


def test_manifest_sets_views_and_crop_boxes(run):
    man, cfg, cases = run["manifest"], run["cfg"], run["cases"]
    schemas.validate_frame(schemas.EMBED_MANIFEST, man)
    assert list(man.columns) == [c.name for c in schemas.EMBED_MANIFEST.columns]
    views = cfg["encoders"]["views"]
    n = {g: s["rows"] * s["cols"] for g, s in cfg["tiles"]["grids"].items()}
    counts = man["set"].value_counts().to_dict()
    n_caltest = int(cases["split"].isin(["calibration", "test"]).sum())
    n_surg = int((cases["lane"] == "surgical").sum())
    types, sevs = cfg["degrade"]["types"], cfg["degrade"]["severities"]
    assert counts["bach512"] == 400 * n["g512"] * views["g512"]
    assert counts["bach256"] == 400 * n["g256"] * views["g256"]
    dv = cfg["degrade"]["embed_views"]
    assert counts["deg512"] == n_caltest * len(types) * len(sevs) * n["g512"] * dv["g512"]
    assert counts["deg256"] == n_surg * len(types) * n["g256"] * dv["g256"]
    assert counts["nct"] == len(cfg["nct"]["classes"]) * cfg["nct"]["per_class"]
    assert man["row_id"].is_unique
    assert (man["row_id"] == man["item_id"] + "|v" + man["view"]).all()
    b = man[man["set"] == "bach512"]
    assert sorted(b["view"].astype(int).unique()) == list(range(views["g512"]))
    assert (b["x"] == "0").all() and (b["w"] == "16").all() and (b["label"] == "").all()
    assert set(b["split"]) == set(schemas.SPLITS)
    # each item has exactly the configured number of views
    assert (b.groupby("item_id").size() == views["g512"]).all()
    # every referenced path exists
    assert all((run["data"] / p).is_file() for p in man.loc[man["set"].str.startswith(("bach", "deg")), "path"].unique())


def test_manifest_degraded_rows(run):
    man, cfg, cases = run["manifest"], run["cfg"], run["cases"]
    d = man[man["set"] == "deg512"]
    assert set(d["split"]) == {"calibration", "test"}
    assert set(d["degrade_type"]) == set(cfg["degrade"]["types"])
    assert set(d["degrade_severity"].astype(int)) == set(cfg["degrade"]["severities"])
    assert d["case_id"].str.contains("_FSIM_").all()
    # clean path reused, degraded on the fly
    assert d["path"].str.startswith("tiles/g512/").all()
    # one seed per (case, type, severity), shared by all its tiles and views
    assert (d.groupby("case_id")["degrade_seed"].nunique() == 1).all()
    d256 = man[man["set"] == "deg256"]
    surg = set(cases.loc[cases["lane"] == "surgical", "case_id"])
    assert {c.split("_FSIM_")[0] for c in d256["case_id"]} == surg
    assert set(d256["degrade_severity"].astype(int)) == {cfg["degrade"]["display_severity"]}
    nct = man[man["set"] == "nct"]
    assert (nct["tile_idx"] == "-1").all() and (nct["view"] == "0").all() and (nct["grid"] == "").all()
    assert set(nct["label"]) == set(cfg["nct"]["classes"])
    assert all(os.path.isabs(p) for p in nct["path"]) and (nct["w"] == "12").all()


def test_meta_written(run):
    meta = json.loads(schemas.step_meta_json(run["data"], "01_build_cases").read_text())
    assert schemas.STEP_META.problems(meta) == []
    s = meta["summary"]
    assert s["patient_check"] == "asserted disjoint"
    assert all(v == [] for v in s["patient_overlap"].values())
    assert s["qc_pass_rate"]["g512"] == pytest.approx(11 / 12, abs=1e-3)


def test_resume_skips_existing_and_force_rewrites(tmp_path, run):
    cfg_path, bach, nct = run["cfg_path"], run["bach"], run["nct"]
    data = tmp_path / "data"
    argv = _argv(cfg_path, data, bach, nct, "--limit", "6")
    assert build.main(argv) == 0
    tiles = pd.read_csv(schemas.tiles_csv(data), dtype=str, keep_default_na=False)
    probe = data / tiles["path"].iloc[0]
    mtime = probe.stat().st_mtime_ns
    first = pd.read_csv(schemas.cases_csv(data), dtype=str, keep_default_na=False)
    assert build.main(argv) == 0
    assert probe.stat().st_mtime_ns == mtime
    meta = json.loads(schemas.step_meta_json(data, "01_build_cases").read_text())
    assert meta["summary"]["files_written"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(schemas.cases_csv(data), dtype=str, keep_default_na=False), first)
    probe.unlink()
    assert build.main(argv) == 0
    assert probe.is_file()
    assert build.main(argv + ["--force"]) == 0
    meta = json.loads(schemas.step_meta_json(data, "01_build_cases").read_text())
    assert meta["summary"]["files_written"] > 6 * 60


def test_limit_and_aux_sets_preserved(tmp_path, run):
    cfg_path, bach, nct = run["cfg_path"], run["bach"], run["nct"]
    data = tmp_path / "data"
    aux = pd.DataFrame([common.manifest_row(set_name="quilt", case_id="img0", split="", grid="", tile_idx=None,
                                            path="/abs/img0.jpg", x=0, y=0, w=10, h=10, view=0,
                                            label="invasion:pos", group="yt:abc")])
    common.replace_manifest_sets(data, ["quilt"], aux)
    assert build.main(_argv(cfg_path, data, bach, nct, "--limit", "5")) == 0
    cases = pd.read_csv(schemas.cases_csv(data), dtype=str, keep_default_na=False)
    assert list(cases["case_id"]) == sorted(run["cases"]["case_id"])[:5]
    man = pd.read_csv(schemas.embed_manifest_csv(data), dtype=str, keep_default_na=False)
    assert (man["set"] == "quilt").sum() == 1
    assert set(man.loc[man["set"] == "bach512", "case_id"]) == set(cases["case_id"])
    assert (man["set"] == "nct").sum() == len(run["cfg"]["nct"]["classes"]) * min(5, run["cfg"]["nct"]["per_class"])
    # the limited cases keep the IDs, splits and lanes of the full assignment
    full = run["cases"].set_index("case_id")
    assert (cases.set_index("case_id")[["split", "lane", "group"]] == full.loc[cases["case_id"], ["split", "lane", "group"]]).all().all()


def test_missing_patient_file_fails_unless_allowed(tmp_path, run):
    cfg_path, cfg = st.small_settings(tmp_path)
    bach = st.make_bach_tree(tmp_path / "bach", cfg, with_patients=False, with_wsi=False)
    data = tmp_path / "data"
    assert build.main(_argv(cfg_path, data, bach, run["nct"], "--limit", "2")) == 2
    assert build.main(_argv(cfg_path, data, bach, run["nct"], "--limit", "2", "--allow-missing-patients")) == 0
    cases = pd.read_csv(schemas.cases_csv(data), dtype=str, keep_default_na=False)
    assert (cases["patient_id"] == "").all() and cases["group"].str.startswith("b").all()


def test_wrong_image_size_is_rejected(tmp_path, run):
    cfg_path, cfg = st.small_settings(tmp_path, **{"dataset.image_size": [60, 48]})
    assert build.main(_argv(cfg_path, tmp_path / "data", run["bach"], run["nct"], "--limit", "1")) != 0


def test_parallel_workers_match_serial(tmp_path, run):
    data = tmp_path / "data"
    assert build.main(_argv(run["cfg_path"], data, run["bach"], run["nct"], "--limit", "4", "--workers", "2")) == 0
    tiles = pd.read_csv(schemas.tiles_csv(data), dtype=str, keep_default_na=False)
    ref = run["tiles"][run["tiles"]["case_id"].isin(tiles["case_id"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(tiles, ref)


@pytest.mark.parametrize("relpath", ["pipeline/common.py", "pipeline/00_inventory.py", "pipeline/01_build_cases.py",
                                     "pipeline/degrade.py"])
def test_delta_side_modules_are_portable(relpath):
    assert st.delta_import_problems(relpath, st.DELTA_ALLOWED - {"sklearn", "scipy"}) == []


def test_existing_cases_csv_with_other_photos_is_refused(tmp_path, run):
    data = tmp_path / "data"
    argv = _argv(run["cfg_path"], data, run["bach"], run["nct"], "--limit", "2")
    assert build.main(argv) == 0
    cases = pd.read_csv(schemas.cases_csv(data), dtype=str, keep_default_na=False)
    cases.loc[0, "source_file"] = "Photos/Benign/b999.tif"
    cases.to_csv(schemas.cases_csv(data), index=False)
    assert build.main(argv) == 2
    assert build.main(argv + ["--force"]) == 0


def test_degraded_view_count_is_configurable(tmp_path, run):
    cfg_path, cfg = st.small_settings(tmp_path, **{"degrade.embed_views": {"g512": 1, "g256": 1}})
    data = tmp_path / "data"
    assert build.main(_argv(cfg_path, data, run["bach"], run["nct"], "--limit", "40")) == 0
    man = pd.read_csv(schemas.embed_manifest_csv(data), dtype=str, keep_default_na=False)
    assert set(man.loc[man["set"].isin(["deg512", "deg256"]), "view"]) == {"0"}
    assert set(man.loc[man["set"] == "bach512", "view"].astype(int)) == set(range(cfg["encoders"]["views"]["g512"]))


def test_manifest_only_rebuilds_without_reading_images(tmp_path, run):
    import shutil

    data = tmp_path / "data"
    for name in ("cases.csv", "tiles.csv", "embed_manifest.csv"):
        shutil.copy(run["data"] / name, data / name) if data.exists() else (data.mkdir(), shutil.copy(run["data"] / name, data / name))
    cfg_path, cfg = st.small_settings(tmp_path, **{"degrade.embed_views": {"g512": 2, "g256": 1}})
    # no tiles/ or photos needed: point --bach-root nowhere
    assert build.main(["--config", str(cfg_path), "--data-root", str(data), "--bach-root", str(tmp_path / "x"),
                       "--nct-root", str(run["nct"]), "--manifest-only"]) == 0
    man = pd.read_csv(schemas.embed_manifest_csv(data), dtype=str, keep_default_na=False)
    assert set(man.loc[man["set"] == "deg512", "view"]) == {"0", "1"}
    ref = run["manifest"]
    pd.testing.assert_frame_equal(man[man["set"] == "bach512"].reset_index(drop=True),
                                  ref[ref["set"] == "bach512"].reset_index(drop=True))
    assert not (data / "tiles").exists()
