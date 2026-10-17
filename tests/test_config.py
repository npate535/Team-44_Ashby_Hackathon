"""Settings, paths, the concepts.yaml schema (ruling R1), atomic IO, inter-task file specs, portability.

Expected values are hand-derived from plan.md, CELLS_SPEC.md and BUILD_PLAN §A/§D (e.g. the THUNDER
ranges give 218/50/132 cases and, per class, test = Benign 48, InSitu 30, Invasive 31, Normal 23).
"""

import ast
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core import config, io, schemas
from core.tiers import REASON_ORDER, assign_tier

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return config.load_settings()


def problems_after(cfg, mutate):
    bad = copy.deepcopy(cfg)
    mutate(bad)
    return config.settings_problems(bad)


# ---- settings.yaml ------------------------------------------------------------------------------
def test_shipped_settings_validate(cfg):
    assert config.settings_problems(cfg) == []


def test_class_order_and_aggressive_class(cfg):
    assert config.class_index(cfg) == {"Normal": 0, "Benign": 1, "InSitu": 2, "Invasive": 3}
    assert config.aggressive_index(cfg) == 3


def test_concept_ids_come_from_config_in_plan_order(cfg):
    assert config.concept_ids(cfg) == [
        "tubules", "pleomorphism", "mitoses", "invasion", "duct_confined",
        "normal_lobules", "benign_stroma_fat", "lymphocytes", "necrosis",
    ]


def test_thunder_split_sizes(cfg):
    assignments = config.split_assignments(cfg)
    assert len(assignments) == 400
    assert Counter(assignments) == {"train": 218, "calibration": 50, "test": 132}


def test_thunder_split_boundaries(cfg):
    a = config.split_assignments(cfg)
    want = {
        0: "train", 14: "train", 15: "calibration", 24: "calibration", 25: "train", 40: "train",
        41: "test", 58: "test", 59: "train", 60: "test", 89: "test", 90: "train", 109: "train",
        110: "calibration", 119: "calibration", 120: "train", 139: "test", 168: "test", 169: "train",
        200: "calibration", 210: "train", 240: "test", 258: "train", 259: "train", 260: "test",
        272: "test", 273: "train", 305: "calibration", 315: "train", 345: "test", 367: "test",
        368: "train", 380: "calibration", 389: "calibration", 390: "train", 399: "train",
    }
    assert {i: a[i] for i in want} == want


def test_thunder_test_split_class_mix(cfg):
    # Indices run over class folders sorted alphabetically, 100 images each.
    folder_order = ["Benign", "InSitu", "Invasive", "Normal"]
    a = config.split_assignments(cfg)
    test_mix = Counter(folder_order[i // 100] for i, s in enumerate(a) if s == "test")
    cal_mix = Counter(folder_order[i // 100] for i, s in enumerate(a) if s == "calibration")
    assert test_mix == {"Benign": 48, "InSitu": 30, "Invasive": 31, "Normal": 23}
    assert cal_mix == {"Benign": 10, "InSitu": 10, "Invasive": 10, "Normal": 20}


def test_overlapping_split_ranges_are_rejected(cfg):
    probs = problems_after(cfg, lambda c: c["split"]["ranges"]["calibration"].append([10, 16]))
    assert any("overlap" in p for p in probs)


def test_gap_in_split_ranges_is_rejected(cfg):
    probs = problems_after(cfg, lambda c: c["split"]["ranges"]["test"].remove([345, 368]))
    assert any("not assigned" in p for p in probs)


def test_split_counts_must_match_expected(cfg):
    probs = problems_after(cfg, lambda c: c["split"]["expected_counts"].update(test=133))
    assert any("test" in p and "132" in p for p in probs)


def test_aggressive_class_must_be_a_class(cfg):
    probs = problems_after(cfg, lambda c: c.update(aggressive_class="Malignant"))
    assert any("aggressive_class" in p for p in probs)


def test_alpha_must_be_in_unit_interval(cfg):
    assert any("alpha" in p for p in problems_after(cfg, lambda c: c.update(alpha=1.5)))
    assert any("alpha" in p for p in problems_after(cfg, lambda c: c.update(alpha=0)))


def test_conformal_method_and_mode_are_checked(cfg):
    assert any("conformal.method" in p for p in problems_after(cfg, lambda c: c["conformal"].update(method="raps")))
    assert any("conformal.mode" in p for p in problems_after(cfg, lambda c: c["conformal"].update(mode="local")))


def test_concept_groups_must_cover_ids_exactly(cfg):
    probs = problems_after(cfg, lambda c: c["concepts"]["groups"]["microenvironment"]["ids"].remove("necrosis"))
    assert any("necrosis" in p for p in probs)


def test_timer_amber_must_precede_red(cfg):
    probs = problems_after(cfg, lambda c: c["margin"]["timer"].update(amber_min=25))
    assert any("timer" in p for p in probs)


def test_aging_fraction_must_be_in_unit_interval(cfg):
    probs = problems_after(cfg, lambda c: c["aging"].update(promote_at_fraction=1.2))
    assert any("promote_at_fraction" in p for p in probs)


def test_cells_reference_classes_must_be_classes(cfg):
    probs = problems_after(cfg, lambda c: c["cells"].update(reference_classes=["Normal", "Healthy"]))
    assert any("Healthy" in p for p in probs)


def test_validate_settings_raises_with_every_problem(cfg):
    bad = copy.deepcopy(cfg)
    bad["alpha"] = 2
    bad["aggressive_class"] = "Malignant"
    with pytest.raises(config.ConfigError) as err:
        config.validate_settings(bad)
    assert "alpha" in str(err.value) and "aggressive_class" in str(err.value)


def test_load_settings_rejects_an_invalid_file(tmp_path):
    text = (REPO / "config" / "settings.yaml").read_text().replace("alpha: 0.1", "alpha: 7")
    path = tmp_path / "settings.yaml"
    path.write_text(text)
    with pytest.raises(config.ConfigError, match="alpha"):
        config.load_settings(path)


def test_team_dir_expands_into_every_delta_path(tmp_path):
    text = (REPO / "config" / "settings.yaml").read_text().replace("dir: team44", "dir: team99")
    path = tmp_path / "settings.yaml"
    path.write_text(text)
    cfg = config.load_settings(path)
    assert cfg["paths"]["delta"]["data_root"] == "/work/nvme/bfqi/teams/team99/pathqueue"
    assert cfg["paths"]["delta"]["bach_official"] == "/work/nvme/bfqi/teams/team99/bach"
    assert cfg["cells"]["checkpoint"].startswith("/work/nvme/bfqi/teams/team99/pathqueue/weights/")
    assert "{team_dir}" not in json.dumps(cfg)


def test_ui_has_a_label_for_every_flag_reason(cfg):
    assert set(cfg["ui"]["reason_labels"]) == set(REASON_ORDER)


def test_shipped_settings_drive_the_tier_function(cfg):
    case = {
        "case_id": "x", "lane": "biopsy", "arrived_offset_s": 0,
        "quality": {"qc_tiles_kept": 12, "ood_percentile": 0.4, "reliability": 0.9, "reliability_flag": False},
        "prediction": {"model": "m", "classes": cfg["classes"], "probs": [0.1, 0.85, 0.03, 0.02],
                       "risk_score": 0.02, "conformal_alpha": cfg["alpha"], "conformal_set": ["Benign"]},
    }
    assert assign_tier(case, cfg) == (4, [])
    target_s = cfg["aging"]["turnaround_target_min"][4] * 60 * cfg["aging"]["promote_at_fraction"]
    assert assign_tier(case, cfg, now=target_s) == (3, ["aging_promoted"])


# ---- config hash, site, roots, mpp ----------------------------------------------------------------
def test_config_hash_ignores_key_order_and_tracks_values():
    a = {"alpha": 0.1, "classes": ["A", "B"], "nested": {"x": 1, "y": 2}}
    b = {"nested": {"y": 2, "x": 1}, "classes": ["A", "B"], "alpha": 0.1}
    assert config.config_hash(a) == config.config_hash(b)
    assert config.config_hash(a) != config.config_hash({**a, "alpha": 0.2})
    assert len(config.config_hash(a)) == 64


def test_site_override_from_env(cfg):
    assert config.detect_site(cfg, env={"PATHQUEUE_SITE": "delta"}) == "delta"
    assert config.detect_site(cfg, env={"PATHQUEUE_SITE": "local"}) == "local"
    with pytest.raises(ValueError, match="PATHQUEUE_SITE"):
        config.detect_site(cfg, env={"PATHQUEUE_SITE": "deltaai"})


def test_site_detected_from_the_delta_work_root(cfg, tmp_path):
    here = copy.deepcopy(cfg)
    here["paths"]["delta"]["work_root"] = str(tmp_path)
    assert config.detect_site(here, env={}) == "delta"
    here["paths"]["delta"]["work_root"] = str(tmp_path / "missing")
    assert config.detect_site(here, env={}) == "local"


def test_data_root_resolution(cfg, tmp_path):
    local = copy.deepcopy(cfg)
    local["paths"]["delta"]["work_root"] = str(tmp_path / "missing")
    assert config.data_root(local, env={}) == REPO / "data"
    assert config.data_root(local, env={"PATHQUEUE_SITE": ""}) == REPO / "data"
    assert config.data_root(local, env={"PATHQUEUE_DATA_ROOT": "data_synthetic"}) == REPO / "data_synthetic"
    assert config.data_root(local, env={"PATHQUEUE_DATA_ROOT": str(tmp_path)}) == tmp_path
    assert config.data_root(local, env={"PATHQUEUE_SITE": "delta"}) == Path("/work/nvme/bfqi/teams/team44/pathqueue")


def test_artifacts_root_resolution(cfg, tmp_path):
    local = copy.deepcopy(cfg)
    local["paths"]["delta"]["work_root"] = str(tmp_path / "missing")
    assert config.artifacts_root(local, env={}) == REPO / "artifacts"
    assert config.artifacts_root(local, env={"PATHQUEUE_ARTIFACTS_ROOT": "fixtures"}) == REPO / "fixtures"
    assert config.fixtures_root(local) == REPO / "fixtures"


def test_effective_mpp_prefers_the_measured_inventory_value(cfg, tmp_path):
    inv = tmp_path / "inventory.json"
    assert config.effective_mpp(cfg, inv) == 0.42  # no inventory yet
    inv.write_text(json.dumps({"measured_mpp": 0.4}))
    assert config.effective_mpp(cfg, inv) == 0.4
    inv.write_text(json.dumps({"measured_mpp": None}))
    assert config.effective_mpp(cfg, inv) == 0.42
    inv.write_text(json.dumps({"measured_mpp": -1}))
    with pytest.raises(ValueError, match="measured_mpp"):
        config.effective_mpp(cfg, inv)


# ---- concepts.yaml schema (ruling R1) ---------------------------------------------------------------
GROUP_OF = {
    "tubules": "nottingham", "pleomorphism": "nottingham", "mitoses": "nottingham",
    "invasion": "insitu_vs_invasive", "duct_confined": "insitu_vs_invasive",
    "normal_lobules": "benign_normal", "benign_stroma_fat": "benign_normal",
    "lymphocytes": "microenvironment", "necrosis": "microenvironment",
}


def valid_bank():
    return {
        "templates": ["an H&E image of {}.", "a photomicrograph showing {}."],
        "concepts": [
            {
                "id": cid, "group": group, "label": cid.replace("_", " ").capitalize(),
                "direction": "context" if cid == "lymphocytes" else "up", "weak": cid == "mitoses",
                "positives": [f"{cid} phrase one", f"{cid} phrase two", f"{cid} phrase three"],
                "contrasts": [f"{cid} contrast one", f"{cid} contrast two"],
            }
            for cid, group in GROUP_OF.items()
        ],
        "class_prompts": {c: [f"{c} breast tissue"] for c in ["Normal", "Benign", "InSitu", "Invasive"]},
    }


def test_valid_concept_bank_is_accepted(cfg):
    bank = schemas.validate_concept_bank(valid_bank(), cfg)
    assert bank.ids == config.concept_ids(cfg)
    assert bank.by_id()["mitoses"].weak is True
    assert bank.templates == ("an H&E image of {}.", "a photomicrograph showing {}.")


@pytest.mark.parametrize(
    "defect, mutate",
    [
        ("{}", lambda b: b["templates"].append("an image of tumor")),
        ("unknown", lambda b: b["concepts"][0].update(id="tumour")),
        ("necrosis", lambda b: b["concepts"].pop()),
        ("group", lambda b: b["concepts"][0].update(group="microenvironment")),
        ("direction", lambda b: b["concepts"][0].update(direction="higher")),
        ("weak", lambda b: b["concepts"][0].update(weak="yes")),
        ("positives", lambda b: b["concepts"][0].update(positives=["only one", "two"])),
        ("contrasts", lambda b: b["concepts"][0].update(contrasts=["a", "b", "c", "d"])),
        ("class_prompts", lambda b: b["class_prompts"].pop("Invasive")),
        ("comment", lambda b: b["concepts"][0].update(comment="stray key")),
        ("{", lambda b: b["concepts"][0]["positives"].append("cells {forming} tubules")),
    ],
)
def test_concept_bank_defects_are_reported(cfg, defect, mutate):
    bank = valid_bank()
    mutate(bank)
    with pytest.raises(ValueError, match=defect):
        schemas.validate_concept_bank(bank, cfg)


def test_load_concept_bank_reads_yaml(cfg, tmp_path):
    path = tmp_path / "concepts.yaml"
    path.write_text(yaml.safe_dump(valid_bank()))
    assert config.load_concept_bank(path, cfg).ids == config.concept_ids(cfg)


def test_expand_prompts_enumerates_concepts_and_classes_by_template():
    bank = schemas.ConceptBank(
        templates=("an image of {}.", "{} on H&E."),
        concepts=(
            schemas.Concept(id="necrosis", group="microenvironment", label="Necrosis", direction="up",
                            weak=False, positives=("tumor necrosis",), contrasts=("viable tumor cells",)),
        ),
        class_prompts={"Normal": ("normal breast tissue",)},
    )
    got = [(p.id, p.text) for p in schemas.expand_prompts(bank)]
    assert got == [
        ("concept:necrosis:pos:0:0", "an image of tumor necrosis."),
        ("concept:necrosis:pos:0:1", "tumor necrosis on H&E."),
        ("concept:necrosis:con:0:0", "an image of viable tumor cells."),
        ("concept:necrosis:con:0:1", "viable tumor cells on H&E."),
        ("class:Normal:0:0", "an image of normal breast tissue."),
        ("class:Normal:0:1", "normal breast tissue on H&E."),
    ]


# ---- core.io ---------------------------------------------------------------------------------------
def test_write_json_round_trip_leaves_no_temp_files(tmp_path):
    target = tmp_path / "sub" / "out.json"
    io.write_json(target, {"b": 1, "a": [1.5, None, "x"]})
    assert io.read_json(target) == {"b": 1, "a": [1.5, None, "x"]}
    assert sorted(p.name for p in target.parent.iterdir()) == ["out.json"]


def test_failed_atomic_write_keeps_the_old_file(tmp_path):
    target = tmp_path / "out.json"
    io.write_json(target, {"version": 1})
    with pytest.raises(RuntimeError):
        with io.atomic_path(target) as tmp:
            tmp.write_text('{"version": 2, "trunc')
            raise RuntimeError("crash mid-write")
    assert io.read_json(target) == {"version": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_write_json_rejects_nan_without_touching_disk(tmp_path):
    target = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        io.write_json(target, {"auroc": float("nan")})
    assert list(tmp_path.iterdir()) == []


def test_numpy_writers_keep_the_exact_file_name(tmp_path):
    arr = np.arange(6, dtype=np.float16).reshape(2, 3)
    npy = io.save_npy(tmp_path / "bach512_align.npy", arr)
    npz = io.save_npz(tmp_path / "conch.npz", ids=np.array(["a", "b"]), emb=arr)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bach512_align.npy", "conch.npz"]
    assert np.array_equal(np.load(npy), arr)
    loaded = io.load_npz(npz)
    assert loaded["ids"].tolist() == ["a", "b"] and np.array_equal(loaded["emb"], arr)


def test_write_csv_round_trip_without_index(tmp_path):
    df = pd.DataFrame({"case_id": ["BACH_0001", "BACH_0002"], "risk": [0.25, 0.75]})
    path = io.write_csv(tmp_path / "t.csv", df)
    assert path.read_text().splitlines()[0] == "case_id,risk"
    assert io.read_csv(path)["risk"].tolist() == [0.25, 0.75]


def test_save_image_infers_format_from_the_final_name(tmp_path):
    from PIL import Image

    path = io.save_image(tmp_path / "tile.png", Image.new("RGB", (8, 8), (200, 10, 10)))
    with Image.open(path) as im:
        assert im.format == "PNG" and im.size == (8, 8)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tile.png"]


def test_write_meta_records_provenance(cfg, tmp_path):
    path = io.write_meta(
        tmp_path / "meta" / "01_build_cases.json", step="01_build_cases", cfg=cfg,
        inputs=["cases.csv"], outputs=["tiles.csv"], args={"limit": 5}, synthetic=True,
    )
    meta = io.read_json(path)
    assert meta["step"] == "01_build_cases"
    assert meta["inputs"] == ["cases.csv"] and meta["outputs"] == ["tiles.csv"]
    assert meta["args"] == {"limit": 5} and meta["synthetic"] is True
    assert meta["config_hash"] == config.config_hash(cfg)
    assert "git_sha" in meta and meta["created_utc"].endswith("Z")
    assert meta["versions"]["python"] == ".".join(map(str, sys.version_info[:3]))
    assert meta["versions"]["numpy"] == np.__version__


def test_outputs_exist_requires_every_path(tmp_path):
    (tmp_path / "a").write_text("x")
    assert io.outputs_exist(tmp_path / "a")
    assert not io.outputs_exist(tmp_path / "a", tmp_path / "b")
    assert not io.outputs_exist()


# ---- core.schemas: paths and names -----------------------------------------------------------------
def test_data_root_path_helpers(tmp_path):
    r = tmp_path
    assert schemas.tile_png(r, "g512", "BACH_0123", 1, 2) == r / "tiles/g512/BACH_0123/1_2.png"
    assert schemas.display_jpg(r, "BACH_0123") == r / "display/BACH_0123.jpg"
    assert schemas.emb_npy(r, "conch", "bach512", "align") == r / "emb/conch/bach512_align.npy"
    assert schemas.emb_npy(r, "conch", "bach512", "probe", views=True) == r / "emb/conch/bach512_probe_views.npy"
    assert schemas.emb_index_csv(r, "uni2h", "bach256", "probe") == r / "emb/uni2h/bach256_probe_index.csv"
    assert schemas.emb_index_csv(r, "musk", "bach512", "align", views=True) == r / "emb/musk/bach512_align_views_index.csv"
    assert schemas.emb_meta_json(r, "conch") == r / "emb/conch/meta.json"
    assert schemas.text_npz(r, "musk") == r / "text/musk.npz"
    assert schemas.tile_risk_csv(r) == r / "tile_risk.csv"
    assert schemas.tile_risk_csv(r, oof=True) == r / "tile_risk_oof.csv"
    assert schemas.cell_json(r, "BACH_0123", 5) == r / "cells/BACH_0123/5.json"
    assert schemas.cell_overlay_png(r, "BACH_0123", 5) == r / "cells/BACH_0123/5_overlay.png"
    assert schemas.cell_features_csv(r, per_tile=True) == r / "cell_features_tile.csv"
    assert schemas.step_meta_json(r, "01_build_cases") == r / "meta/01_build_cases.json"


def test_artifact_paths_and_contract_refs(tmp_path):
    a = tmp_path
    assert schemas.case_json(a, "BACH_0123") == a / "cases/BACH_0123.json"
    assert schemas.case_image(a, "BACH_0123") == a / "images/BACH_0123.jpg"
    assert schemas.case_cell_overlay(a, "BACH_0123", 5) == a / "cells/BACH_0123_5_overlay.png"
    assert schemas.eval_json(a, "queue_sim") == a / "eval/queue_sim.json"
    assert schemas.display_ref("BACH_0123") == "images/BACH_0123.jpg"
    assert schemas.cell_overlay_ref("BACH_0123", 5) == "cells/BACH_0123_5_overlay.png"
    assert schemas.fixture_case_json(a, "demo_0001") == a / "demo_0001.json"
    # fixtures/ is flat: contract media refs resolve by basename (team44-triage-frontend ?mock=1 does the same)
    assert schemas.fixture_media(a, "images/demo_0001.jpg") == a / "demo_0001.jpg"
    assert schemas.fixture_media(a, "cells/demo_0001_5_overlay.png") == a / "demo_0001_5_overlay.png"
    with pytest.raises(ValueError):
        schemas.fixture_media(a, "images/../../secret.txt")


@pytest.mark.parametrize(
    "call",
    [
        lambda r: schemas.case_json(r, "../secrets"),
        lambda r: schemas.case_image(r, "a/b"),
        lambda r: schemas.emb_npy(r, "conch/..", "bach512", "align"),
        lambda r: schemas.text_npz(r, "Conch"),
        lambda r: schemas.eval_json(r, "not_a_registered_eval"),
        lambda r: schemas.tile_png(r, "g512", "BACH_0001", -1, 0),
    ],
)
def test_unsafe_or_unknown_names_are_rejected(tmp_path, call):
    with pytest.raises(ValueError):
        call(tmp_path)


def test_derived_case_and_item_ids():
    assert schemas.variant_case_id("BACH_0042", "blur", 3) == "BACH_0042_FSIM_blur3"
    assert schemas.composite_case_id("BACH_0042", 7) == "COMP_007"   # opaque: never names the carcinoma source (M10)
    assert schemas.make_item_id("bach512", "BACH_0042", tile_idx=5) == "bach512:BACH_0042:5"
    assert schemas.make_item_id("deg512", "BACH_0042", tile_idx=5, degrade_type="jpeg", severity=2) == (
        "deg512:BACH_0042:5:jpeg2"
    )
    assert schemas.make_item_id("nct", "TUM-AAAWMSFI") == "nct:TUM-AAAWMSFI"
    with pytest.raises(ValueError):
        schemas.variant_case_id("BACH_0042", "fog", 3)


def test_prediction_column_names_and_set_encoding(cfg):
    assert schemas.prob_col("conch_probe_lr", "InSitu") == "p__conch_probe_lr__InSitu"
    assert schemas.cal_prob_col("Invasive") == "pcal__Invasive"
    with pytest.raises(ValueError):
        schemas.prob_col("bad__name", "InSitu")
    assert schemas.format_set(["Invasive", "InSitu"], cfg["classes"]) == "InSitu|Invasive"
    assert schemas.format_set([], cfg["classes"]) == ""
    assert schemas.parse_set("InSitu|Invasive") == ["InSitu", "Invasive"]
    assert schemas.parse_set("") == [] and schemas.parse_set(float("nan")) == []
    with pytest.raises(ValueError):
        schemas.format_set(["Carcinoma"], cfg["classes"])


def test_concept_feature_columns(cfg):
    cols = schemas.concept_feature_columns(cfg)
    assert len(cols) == 18
    assert cols[:2] == ["tubules__top3", "tubules__frac"]
    assert schemas.concept_spread_columns(cfg)[-1] == "necrosis__spread"


# ---- core.schemas: frame validation ------------------------------------------------------------------
def tiles_frame():
    return pd.DataFrame(
        {
            "case_id": ["BACH_0001", "BACH_0001"], "grid": ["g512", "g512"], "tile_idx": [0, 1],
            "row": [0, 0], "col": [0, 1], "x": [0, 512], "y": [0, 0], "size": [512, 512],
            "tissue_frac": [0.9, 0.2], "qc_pass": [1, 0],
            "path": ["tiles/g512/BACH_0001/0_0.png", "tiles/g512/BACH_0001/0_1.png"],
        }
    )


def test_valid_tiles_frame_passes():
    schemas.validate_frame(schemas.TILES, tiles_frame())


def test_frame_validation_reports_missing_columns():
    with pytest.raises(ValueError, match="qc_pass"):
        schemas.validate_frame(schemas.TILES, tiles_frame().drop(columns=["qc_pass"]))


def test_frame_validation_reports_duplicate_keys():
    df = tiles_frame()
    df.loc[1, "tile_idx"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        schemas.validate_frame(schemas.TILES, df)


def test_frame_validation_reports_values_outside_the_vocabulary():
    df = tiles_frame()
    df.loc[0, "grid"] = "g999"
    with pytest.raises(ValueError, match="g999"):
        schemas.validate_frame(schemas.TILES, df)


def test_frame_validation_reports_non_numeric_values():
    df = tiles_frame()
    df["tissue_frac"] = ["high", "low"]
    with pytest.raises(ValueError, match="tissue_frac"):
        schemas.validate_frame(schemas.TILES, df)


def test_frame_validation_checks_dynamic_columns(cfg):
    df = pd.DataFrame({"case_id": ["BACH_0001"], "split": ["train"]})
    with pytest.raises(ValueError, match="tubules__top3"):
        schemas.validate_frame(schemas.CASE_FEATURES, df, extra_required=schemas.concept_feature_columns(cfg))


def test_every_inter_task_file_has_a_spec():
    tables = {t.filename for t in schemas.TABLES}
    assert {
        "cases.csv", "tiles.csv", "embed_manifest.csv", "emb/<encoder>/<set>_<head>_index.csv",
        "composites.csv", "wsi_windows.csv", "case_features.csv", "tile_risk.csv", "tile_risk_oof.csv",
        "predictions.csv", "reliability.csv", "cell_tiles.csv", "cell_features.csv", "cell_features_tile.csv",
    } <= tables
    others = {s.filename for s in schemas.JSON_SPECS} | {s.filename for s in schemas.ARRAY_SPECS}
    assert {
        "inventory.json", "emb/<encoder>/meta.json", "retrieval.json", "margins.json",
        "cells/<case_id>/<tile_idx>.json", "meta/<step>.json", "text/<model>.npz", "concepts_tile.npz",
        "emb/<encoder>/<set>_<head>.npy",
    } <= others
    assert {"concepts", "concepts_bach", "zeroshot", "model_choice", "metrics", "calibration", "conformal",
            "reliability", "retrieval", "deletion", "queue_sim", "margin", "thunder", "cells"} <= set(schemas.EVAL_FILES)


def test_config_vocabularies_match_schema_vocabularies(cfg):
    assert tuple(cfg["degrade"]["types"]) == schemas.DEGRADE_TYPES
    assert tuple(cfg["cells"]["types"]) == schemas.CELL_TYPES
    assert sorted(cfg["margin"]["names"].values()) == sorted(schemas.MARGINS)
    assert set(cfg["tiles"]["grids"]) == set(schemas.GRIDS)
    assert set(cfg["encoders"]["heads"]) == set(schemas.HEADS)


# ---- portability of Delta-side modules (BUILD_PLAN §C.2) ---------------------------------------------
DELTA_SIDE = ["core/__init__.py", "core/config.py", "core/io.py", "core/schemas.py",
              "pipeline/__init__.py", "pipeline/degrade.py"]
ALLOWED_THIRD_PARTY = {"numpy", "pandas", "yaml", "PIL", "core", "pipeline"}


@pytest.mark.parametrize("relpath", DELTA_SIDE)
def test_delta_side_modules_parse_on_python_310_and_import_only_module_packages(relpath):
    tree = ast.parse((REPO / relpath).read_text(), feature_version=(3, 10))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    outside = imported - set(sys.stdlib_module_names) - ALLOWED_THIRD_PARTY - {"__future__"}
    assert outside == set()
