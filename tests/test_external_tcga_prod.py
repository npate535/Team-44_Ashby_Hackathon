"""Round-2 TCGA-BRCA external test of the production model (docs/EXTERNAL_TCGA_R2.md).

The first three tests guard the plan's review focus items: the TCGA read uses the same views as the stored BACH
production embeddings, the same APS q-hat and set construction as the app, and slide scores are maxima over crops.
`test_tile_api_crop_matches_manifest_crop` talks to the GDC tile API; it is opt-in (PQ_NETWORK=1).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import TCGA_script as T
from core import config, conformal, schemas
from eval import external_tcga_prod as P
from pipeline import common
from pipeline import encoders as E

CFG = config.load_settings()
EMBED = importlib.import_module("pipeline.02_embed")


def real_data_root() -> Path | None:
    """The real pipeline data root (PATHQUEUE_DATA_ROOT or <repo>/data) when it holds predictions.csv."""
    env = os.environ.get(config.ENV_DATA_ROOT)
    root = config.resolve_repo_path(env) if env else config.REPO_ROOT / "data"
    return root if (root / "predictions.csv").is_file() and (root / "cases.csv").is_file() else None


def write_fake_store(root: Path, art: Path, n_views: int) -> None:
    """A minimal data/artifacts root with the stored-embedding metadata production_recipe reads."""
    emb = root / "emb" / "uni2h"
    emb.mkdir(parents=True)
    (emb / "meta.json").write_text(json.dumps({"sets": {"bach512": {"n_views": n_views, "n_rows": 24,
                                                                    "views_file": n_views > 1}}}))
    rows = [{"emb_row": i, "item_id": f"bach512:C{i // 12}:{i % 12}", "set": "bach512", "case_id": f"C{i // 12}",
             "split": "test", "grid": "g512", "tile_idx": i % 12, "view": -1, "n_views": n_views,
             "degrade_type": "", "degrade_severity": 0, "label": "", "group": ""} for i in range(24)]
    pd.DataFrame(rows).to_csv(emb / "bach512_probe_index.csv", index=False)
    (art / "eval").mkdir(parents=True)
    (art / "eval" / "model_choice.json").write_text(json.dumps({"production": "uni2h512_lr"}))


# ---- review focus 2: TTA / views ------------------------------------------------------------------------------------
def test_tcga_views_match_bach_recipe(tmp_path):
    root, art = tmp_path / "data", tmp_path / "art"
    write_fake_store(root, art, n_views=1)
    views = P.stored_views(root, art, CFG)
    assert views == {"model": "uni2h512_lr", "encoder": "uni2h", "set": "bach512", "head": "probe", "grid": "g512",
                     "n_views": 1, "views": [0]}

    # The transforms are 02_embed's: TileDataset (crop -> dihedral view -> encoder preprocessing), row by row.
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (1536, 2048, 3), dtype=np.uint8))
    img.save(tmp_path / "img.png")
    spec = E.encoder_spec(E.load_encoders_config(), "uni2h")
    prep = E.Preprocess.from_spec(spec)
    boxes = common.grid_boxes(CFG, "g512")[:2]
    for n_views in (1, int(CFG["encoders"]["views"]["g512"])):
        manifest = pd.DataFrame([{"path": "img.png", "x": b["x"], "y": b["y"], "w": b["size"], "h": b["size"],
                                  "view": v, "degrade_type": "", "degrade_severity": 0, "degrade_seed": 0}
                                 for b in boxes for v in range(n_views)])
        ds = EMBED.TileDataset(manifest, tmp_path, prep, CFG, cache_size=2)
        want = np.stack([ds[i].numpy() for i in range(len(ds))])
        got = P.region_tensors(img, boxes, n_views, prep).numpy()
        assert got.shape == want.shape
        assert np.array_equal(got, want)

    # A store with 8 views would demand an 8-view read; the meta and the index must agree.
    root8, art8 = tmp_path / "d8", tmp_path / "a8"
    write_fake_store(root8, art8, n_views=8)
    assert P.stored_views(root8, art8, CFG)["views"] == list(range(8))
    (root8 / "emb" / "uni2h" / "meta.json").write_text(json.dumps({"sets": {"bach512": {"n_views": 1}}}))
    with pytest.raises(SystemExit, match="n_views"):
        P.stored_views(root8, art8, CFG)

    real = real_data_root()
    if real is not None:                       # the real store: production uni2h512_lr was embedded at view 0 only
        art_real = config.REPO_ROOT / "artifacts"
        got = P.stored_views(real, art_real, CFG)
        meta = json.loads((real / "emb" / got["encoder"] / "meta.json").read_text())
        assert got["n_views"] == int(meta["sets"][got["set"]]["n_views"]) == 1
        assert got["views"] == [0]


# ---- review focus 3: conformal threshold drift ------------------------------------------------------------------------
def test_build_sets_is_05_construction():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(4), size=40)
    q = 0.85
    want = conformal.nonempty_sets(conformal.aps_sets(probs, q, int(CFG["conformal"]["seed"]) + 1, True), probs)
    assert np.array_equal(P.build_sets(probs, q, CFG), want)
    det = conformal.nonempty_sets(conformal.aps_sets(probs, q, None, False), probs)
    assert np.array_equal(P.build_sets(probs, q, CFG, randomized=False), det)


def test_production_qhat_matches_app():
    root = real_data_root()
    if root is None:
        pytest.skip("real data root with predictions.csv not available (set PATHQUEUE_DATA_ROOT)")
    art = config.REPO_ROOT / "artifacts"
    model = json.loads((art / "eval" / "model_choice.json").read_text())["production"]
    stored = json.loads((art / "eval" / "model_choice.json").read_text())["conformal"][model]["qhat_aps"]
    qhat = P.recompute_qhat(root, model, CFG)
    assert math.isclose(qhat, stored, rel_tol=0, abs_tol=1e-12)
    assert P.production_qhat(root, art, CFG) == stored

    # The set builder used on TCGA reproduces the app's set_tier on every BACH row, test rows included.
    frame = P.app_row_order(root, model)
    probs = frame[[schemas.cal_prob_col(c) for c in CFG["classes"]]].to_numpy()
    sets = P.build_sets(probs, stored, CFG)
    rebuilt = [schemas.format_set(conformal.set_labels(s, CFG["classes"]), CFG["classes"]) for s in sets]
    test = (frame["split"] == "test").to_numpy()
    assert test.sum() >= 132
    assert list(np.asarray(rebuilt)[test]) == list(frame.loc[test, "set_tier"])
    assert rebuilt == list(frame["set_tier"])


# ---- slide-level rule ----------------------------------------------------------------------------------------------------
def crop(pid: str, group: str, j: int, p_inv: float, top: str, tier: int) -> dict:
    return {"crop_id": f"{pid}_{group}_{j:02d}", "patient_id": pid, "group": group,
            "label": "Normal" if group.endswith("normal") else "Invasive", "prep": group.split("_")[0],
            "p_invasive": p_inv, "top": top, "tier": tier}


def test_slide_score_is_max_over_crops():
    crops = [crop("P1", "frozen_tumor", 0, 0.20, "Normal", 4), crop("P1", "frozen_tumor", 1, 0.91, "Invasive", 2),
             crop("P1", "frozen_tumor", 2, 0.40, "InSitu", 3),
             crop("P1", "frozen_normal", 0, 0.05, "Normal", 4), crop("P1", "frozen_normal", 1, 0.45, "InSitu", 2),
             crop("P2", "ffpe_tumor", 0, 0.30, "Benign", 4)]
    slides = {(s["patient_id"], s["group"]): s for s in P.slide_table(crops)}
    ft = slides[("P1", "frozen_tumor")]
    assert ft["slide_invasive_score"] == pytest.approx(0.91) and ft["called_invasive"] and ft["n_crops"] == 3
    assert ft["any_tier_1_2"] and ft["min_tier"] == 2
    fn = slides[("P1", "frozen_normal")]
    assert fn["slide_invasive_score"] == pytest.approx(0.45)
    assert not fn["called_invasive"]           # top class decides "called invasive", not the score
    assert fn["any_tier_1_2"]                  # Invasive inside the set still places it in tier 2
    p2 = slides[("P2", "ffpe_tumor")]
    assert p2["slide_invasive_score"] == pytest.approx(0.30) and not p2["called_invasive"]
    for s in slides.values():
        own = [c["p_invasive"] for c in crops if (c["patient_id"], c["group"]) == (s["patient_id"], s["group"])]
        assert s["slide_invasive_score"] == max(own)


# ---- tile source: offline checks against openslide's own DeepZoom generator ---------------------------------------------
def test_dz_stitch_reproduces_region_offline():
    from openslide import ImageSlide
    from openslide.deepzoom import DeepZoomGenerator

    rng = np.random.default_rng(1)
    src = Image.fromarray(rng.integers(0, 255, (1100, 1500, 3), dtype=np.uint8))
    gen = DeepZoomGenerator(ImageSlide(src), tile_size=512, overlap=1, limit_bounds=False)
    dz = T.DeepZoom(width=1500, height=1100, tile=512, overlap=1,
                    fetch_tile=lambda level, col, row: gen.get_tile(level, (col, row)).convert("RGBA"))
    assert dz.level_count == gen.level_count and dz.dims == list(gen.level_dimensions)
    for x, y, w, h in [(0, 0, 1500, 1100), (300, 200, 700, 600), (511, 511, 3, 3), (1024, 512, 476, 588)]:
        got = dz.read(dz.max_level, x, y, w, h)
        assert got.mode == "RGBA" and got.size == (w, h)
        assert np.array_equal(np.asarray(got.convert("RGB")), np.asarray(src.crop((x, y, x + w, y + h))))


def test_best_level_for_downsample_follows_openslide():
    ds = [1.0, 4.0, 16.0, 32.0]
    assert T.best_level_for_downsample(ds, 0.5) == 0
    assert T.best_level_for_downsample(ds, 3.99) == 0
    assert T.best_level_for_downsample(ds, 4.0) == 1
    assert T.best_level_for_downsample(ds, 52.8) == 3
    assert T.best_level_for_downsample(ds, 20.0) == 2


def tiff_bytes(ifds: list[dict[int, tuple[int, object]]]) -> bytes:
    """Classic little-endian TIFF with the given IFDs (tag -> (type, value)); tag values only, no image data.
    ASCII values (type 2) are stored after the IFDs; numeric values are inline."""
    starts, pos = [], 8
    for tags in ifds:
        starts.append(pos)
        pos += 2 + 12 * len(tags) + 4
    out, blobs = bytearray(b"II*\x00" + struct.pack("<I", starts[0])), bytearray()
    for k, tags in enumerate(ifds):
        out += struct.pack("<H", len(tags))
        for tag, (typ, value) in sorted(tags.items()):
            if typ == 2:
                data = str(value).encode() + b"\x00"
                out += struct.pack("<HHII", tag, typ, len(data), pos + len(blobs))
                blobs += data
            else:
                out += struct.pack("<HHII", tag, typ, 1, int(value))
        out += struct.pack("<I", starts[k + 1] if k + 1 < len(ifds) else 0)
    return bytes(out + blobs)


def test_svs_header_reads_mpp_and_levels_with_ranges():
    desc = ("Aperio Image Library v11.2.1\r\n46000x32914 (256x256) JPEG/RGB Q=70|AppMag = 40|MPP = 0.2485|"
            "Filename = demo")
    blob = tiff_bytes([
        {256: (4, 46000), 257: (4, 32914), 270: (2, desc), 322: (3, 256), 323: (3, 256)},
        {256: (3, 1024), 257: (3, 732), 270: (2, "thumbnail")},                       # stripped: not a level
        {256: (4, 11500), 257: (4, 8228), 322: (3, 256), 323: (3, 256)},
        {256: (3, 2875), 257: (3, 2057), 322: (3, 256), 323: (3, 256)},
    ])
    calls: list[tuple[int, int]] = []

    def fetch(start: int, length: int) -> bytes:
        calls.append((start, length))
        return blob[start:start + length]

    h = T.svs_header(fetch)
    assert (h["width"], h["height"]) == (46000, 32914)
    assert h["levels"] == [(46000, 32914), (11500, 8228), (2875, 2057)]
    assert h["downsamples"][0] == 1.0 and h["downsamples"][1] == pytest.approx(4.0, rel=1e-3)
    assert (h["mpp"], h["mpp_estimated"], h["appmag"]) == (0.2485, False, 40.0)
    no_mpp = T.mpp_from_description(desc.replace("|MPP = 0.2485", ""))
    assert no_mpp == (0.25, True)              # 10 / AppMag, flagged as estimated (as slide_mpp does)
    assert all(length <= 65536 for _, length in calls)


# ---- review focus 1: tile API vs openslide crops (network, opt-in) --------------------------------------------------------
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("PQ_NETWORK") != "1", reason="GDC tile API test is opt-in: PQ_NETWORK=1")
def test_tile_api_crop_matches_manifest_crop():
    root = real_data_root()
    tcga = (root / "tcga_brca_test") if root else None
    if tcga is None or not (tcga / "manifest.csv").is_file():
        pytest.skip("round-1 TCGA crops (data/tcga_brca_test) not available")
    row = pd.read_csv(tcga / "manifest.csv").iloc[0].to_dict()
    import requests

    got = T.tile_crop_from_row(requests.Session(), row)
    with Image.open(tcga / row["path"]) as handle:
        want = handle.convert("RGB")
    assert got.size == want.size == (T.TARGET_W, T.TARGET_H)
    mad = float(np.abs(np.asarray(got, np.int16) - np.asarray(want, np.int16)).mean())
    print(f"tile-vs-openslide crop {row['crop_id']}: mean absolute difference {mad:.4f} (0-255 units)")
    assert mad < 3.0


# ---- docs/RESULTS.md section ------------------------------------------------------------------------------------------------
def ci(est: float, lo: float, hi: float, n: int) -> dict:
    return {"estimate": est, "lo": lo, "hi": hi, "n": n, "n_groups": n}


def served_doc() -> dict:
    primary = {"ffpe_tumor_sensitivity": ci(0.875, 0.625, 1.0, 8), "frozen_tumor_sensitivity": ci(0.75, 0.5, 1.0, 8),
               "ffpe_vs_frozen_tumor_sensitivity_gap": {"estimate": 0.125},
               "frozen_tumor_vs_normal": {"auroc": ci(0.93, 0.8, 1.0, 15), "sensitivity": ci(0.75, 0.5, 1.0, 8),
                                          "specificity": ci(0.857, 0.6, 1.0, 7)},
               "tumor_tier_1_2_rate": ci(0.9375, 0.8125, 1.0, 16)}
    cohort = {"primary": primary, "n_patients": 8, "n_slides": 23, "n_crops": 132, "slides_without_crops": [],
              "slides_short_crops": [{"patient_id": "P", "group": "frozen_normal", "n_crops": 5}], "failed_crops": [],
              "sensitivity_s1_six_crop_slides": {**primary, "n_slides": 20},
              "sensitivity_s2_tta8": {"primary": primary},
              "sensitivity_s3_deterministic_sets": {"tumor_tier_1_2_rate": ci(0.9, 0.8, 1.0, 16)}}
    return {"protocol": "docs/EXTERNAL_TCGA_R2.md", "model": "uni2h512_lr", "n_views": 1,
            "recipe": {"temperature": 0.3415, "qhat_aps": 0.9308, "alpha": 0.1},
            **cohort, "n_patients": 25, "n_slides": 72, "n_crops": 410,
            "cohorts": {"same8": cohort},
            "bach_reference": {"n_images": 132, "unit": "single image", "invasive_sensitivity": ci(1.0, 1.0, 1.0, 132),
                               "invasive_sensitivity_patient_bootstrap": ci(1.0, 1.0, 1.0, 31),
                               "normal_specificity": ci(0.97, 0.9, 1.0, 33), "invasive_vs_normal_auroc": ci(1, 1, 1, 64),
                               "invasive_tier_1_2_rate": ci(1.0, 1.0, 1.0, 31),
                               "reliability_flag_rate_clean": {"estimate": 0.1667}},
            "limitations": ["Slide-level labels."]}


def test_results_md_tcga_section_quotes_the_json():
    from eval import results_md as R

    text = "\n".join(R.section_external_tcga(served_doc()))
    assert "## TCGA-BRCA external test (production model)" in text
    assert "no benign or in situ labels" in text
    assert "| FFPE tumor slides called invasive (sensitivity) | 87.5% [62.5%, 100.0%] (n 8, 8 groups) |" in text
    assert "0.930 [0.800, 1.000] (n 15, 15 groups)" in text           # AUROC on the probability scale
    assert "| Patients / slides / crops | 25 / 72 / 410 | 8 / 23 / 132 |" in text
    assert "### BACH test vs TCGA (25 patients)" in text and "slides with exactly 6 crops" in text
    assert "P frozen_normal (5)" in text and "12.5 points" in text and "- Slide-level labels." in text
    assert "| Invasive / tumor called invasive | 100.0% [100.0%, 100.0%] (n 31, 31 groups) |" in text
    for absent in ("uni2h_ft", "fine-tune", "round 1", "8-view", "deterministic"):   # production only; S2/S3 JSON only
        assert absent not in text


# ---- only the production model is served ---------------------------------------------------------------------------------
def artifact_roots() -> list[Path]:
    roots = [config.REPO_ROOT / "artifacts"]
    env = os.environ.get(config.ENV_ARTIFACTS_ROOT)
    if env and config.resolve_repo_path(env).resolve() != roots[0].resolve():
        roots.append(config.resolve_repo_path(env))
    return roots


def test_served_external_tcga_is_the_production_model():
    art = config.REPO_ROOT / "artifacts"
    production = json.loads((art / "eval" / "model_choice.json").read_text())["production"]
    served = json.loads((art / "eval" / "external_tcga.json").read_text())
    assert served["name"] == "external_tcga" and served["model"] == production
    assert served["synthetic"] is False and served["primary"] and served["crops"]
    assert {c["crop_id"] for c in served["crops"]} and all("uni2h_ft" not in json.dumps(c) for c in served["crops"])
    assert "uni2h_ft" not in json.dumps({k: v for k, v in served.items() if k != "tile_equivalence"})
    for path in (art / "eval").glob("*.json"):                     # nothing served under eval/ reports another model
        doc = json.loads(path.read_text())
        if isinstance(doc, dict) and path.stem.startswith("external_tcga"):
            assert doc.get("model") == production, path.name


def test_tcga_specimens_carry_the_production_model():
    checked = 0
    for art in artifact_roots():
        cases = sorted((art / "tcga" / "cases").glob("*.json"))
        if not cases:
            continue
        production = json.loads((art / "eval" / "model_choice.json").read_text())["production"]
        index = json.loads((art / "tcga" / "index.json").read_text())
        assert index["model"] == production and all(i["model"] == production for i in index["items"])
        assert {p.stem for p in cases} == {i["case_id"] for i in index["items"]}
        for path in cases:
            assert json.loads(path.read_text())["prediction"]["model"] == production, path.name
            checked += 1
    if not checked:
        pytest.skip("no artifacts/tcga/cases built on this checkout")


def test_round1_script_writes_only_the_archive(tmp_path):
    from eval import external_tcga as R1

    out = R1.write_archive(tmp_path, {"model": "uni2h_ft", "primary": {}}, CFG)
    assert out == tmp_path / "archive" / "eval" / "external_tcga_round1.json"
    assert not (tmp_path / "eval").exists()
    assert json.loads(out.read_text())["name"] == "external_tcga_round1"


def test_build_tcga_cases_uses_production_and_drops_stale(tmp_path):
    build = _load_script("build_tcga_cases")
    art, crops_root = tmp_path / "art", tmp_path / "crops"
    (art / "eval").mkdir(parents=True)
    (art / "eval" / "model_choice.json").write_text(json.dumps({"production": "uni2h512_lr"}))
    (crops_root / "frozen_tumor").mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 120, 160)).save(crops_root / "frozen_tumor" / "P1_frozen_tumor_00.png")
    crop = {"crop_id": "P1_frozen_tumor_00", "path": "frozen_tumor/P1_frozen_tumor_00.png", "prep": "frozen",
            "patient_id": "P1", "probs": {"Normal": 0.1, "Benign": 0.1, "InSitu": 0.2, "Invasive": 0.6},
            "p_invasive": 0.6, "top": "Invasive", "conformal_set": ["InSitu", "Invasive"], "tier": 2,
            "tier_reasons": ["aggressive_top"], "qc_tiles_kept": 12, "slide_file_name": "x.svs"}
    doc = {"model": "uni2h512_lr", "crops": [crop], "data": {"tcga_root": str(crops_root)}, "n_patients": 1}
    (art / "eval" / "external_tcga.json").write_text(json.dumps(doc))
    stale = art / "tcga" / "cases" / "OLD_frozen_tumor_00.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    args = ["--eval", str(art / "eval" / "external_tcga.json"), "--model-choice", str(art / "eval" / "model_choice.json"),
            "--out", str(art / "tcga")]
    assert build.main(args) == 0
    assert not stale.exists()
    case = json.loads((art / "tcga" / "cases" / "P1_frozen_tumor_00.json").read_text())
    assert case["prediction"]["model"] == "uni2h512_lr" and case["flag"]["tier"] == 2
    assert json.loads((art / "tcga" / "index.json").read_text())["model"] == "uni2h512_lr"
    doc["model"] = "uni2h_ft"                                       # another model is refused
    (art / "eval" / "external_tcga.json").write_text(json.dumps(doc))
    assert build.main(args) == 1


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, config.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
