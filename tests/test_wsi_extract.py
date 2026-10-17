"""pipeline.wsi_extract on generated slides (fake reader; openslide is not in the laptop env): XML parsing, polygon
rasterization, purity filter, resampling math, window strata, per-margin truth, patient-disjoint folds."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from core import config, schemas
from pipeline import common
from tests import synthetic_tree as st

wsi = importlib.import_module("pipeline.wsi_extract")
CFG = config.load_settings()
INV, INS, BEN = 3, 2, 1  # mask codes in settings wsi.paint_order


@pytest.mark.parametrize("raw,label", [("Invasive carcinoma", "Invasive"), ("Carcinoma invasive", "Invasive"),
                                       ("Carcinoma in situ", "InSitu"), ("In situ carcinoma", "InSitu"),
                                       ("Benign", "Benign"), ("BENIGN", "Benign")])
def test_norm_label(raw, label):
    assert wsi.norm_label(raw, CFG) == label


def test_norm_label_rejects_unknown():
    with pytest.raises(ValueError):
        wsi.norm_label("Normal", CFG)


def test_parse_xml_text_and_attribute_styles(tmp_path):
    st.write_bach_xml(tmp_path / "a.xml", [("Invasive carcinoma", st.rect(0, 0, 10, 10))], mpp=0.466667)
    st.write_bach_xml(tmp_path / "b.xml", [("Carcinoma in situ", st.rect(0, 0, 10, 10)),
                                           ("Benign", [(0, 0), (1, 1)])], use_attribute=True)
    mpp, regs = wsi.parse_bach_xml(tmp_path / "a.xml", CFG)
    assert mpp == pytest.approx(0.466667) and regs[0]["label"] == "Invasive" and len(regs[0]["xy"]) == 4
    _, regs = wsi.parse_bach_xml(tmp_path / "b.xml", CFG)
    assert [r["label"] for r in regs] == ["InSitu"]  # 2-vertex region dropped


def test_label_mask_paint_order_and_overlap():
    regions = [{"label": "Invasive", "xy": st.rect(10, 10, 29, 29)},
               {"label": "Benign", "xy": st.rect(0, 0, 39, 39)},
               {"label": "InSitu", "xy": st.rect(60, 0, 79, 19)}]
    mask, overlaps = wsi.label_mask((100, 50), regions, (100, 50), CFG)
    assert mask[20, 20] == INV and mask[5, 5] == BEN and mask[10, 70] == INS and mask[45, 95] == 0
    assert overlaps == {"Invasive": 400}  # nested invasive overwrites benign (20x20 px)
    half, _ = wsi.label_mask((100, 50), regions, (50, 25), CFG)  # scaled rasterization
    assert half[10, 10] == INV and half.shape == (25, 50)


def test_resampling_math_matches_research_numbers():
    assert wsi.side0(512, 0.42, 0.466667) == 461
    assert wsi.side0(256, 0.42, 0.466667) == 230
    assert wsi.side0(2048, 0.42, 0.466667) == 1843
    assert wsi.side0(1536, 0.42, 0.466667) == 1382


def _win(h=48, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def test_window_truth_strata_and_per_margin_truth():
    frac = (8 / 64, 8 / 48)
    m = _win()
    m[:, :8] = INV  # whole left strip invasive
    stratum, truth, _ = wsi.window_truth(m, [INS, INV], frac, CFG)
    assert stratum == "edge_tumor"
    assert truth == {"medial": "tumor", "lateral": "clear", "superior": "tumor", "inferior": "tumor"}
    m = _win()
    m[20:28, 28:36] = INS  # central blob
    assert wsi.window_truth(m, [INS, INV], frac, CFG)[:2] == ("central_only", {k: "clear" for k in schemas.MARGINS})
    m = _win()
    m[:, :] = BEN  # benign only: no carcinoma
    assert wsi.window_truth(m, [INS, INV], frac, CFG)[0] == "none"
    m = _win()
    m[47, 63] = INV  # one pixel: 0 < fraction < threshold in two strips -> ambiguous
    assert wsi.window_truth(m, [INS, INV], frac, CFG)[0] is None
    m = _win()
    m[0:8, 40:64] = INS  # top-right corner: superior + lateral tumor
    _, truth, fracs = wsi.window_truth(m, [INS, INV], frac, CFG)
    assert truth["superior"] == "tumor" and truth["lateral"] == "tumor" and truth["inferior"] == "clear"
    assert fracs["superior"] == pytest.approx(24 * 8 / (64 * 8))


def test_select_windows_is_non_overlapping_and_capped():
    rng = np.random.default_rng(0)
    cands = [{"x0": x, "y0": 0, "w0": 10, "h0": 10, "stratum": s} for x in range(0, 100, 5) for s in ["none"]]
    cands += [{"x0": 3, "y0": 0, "w0": 10, "h0": 10, "stratum": "edge_tumor"}]
    out = wsi.select_windows(cands, 3, rng)
    assert sum(c["stratum"] == "none" for c in out) == 3 and out[0]["stratum"] == "edge_tumor"
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            assert a["x0"] + 10 <= b["x0"] or b["x0"] + 10 <= a["x0"]


def test_folds_are_patient_disjoint_and_seeded():
    groups = [f"wsi_p{k}" for k in range(1, 11)]
    folds = wsi.assign_folds(groups, CFG)
    assert folds == wsi.assign_folds(list(reversed(groups)), CFG)
    assert sorted(set(folds.values())) == list(range(CFG["wsi"]["folds"]))
    assert all(list(folds.values()).count(f) == 2 for f in range(CFG["wsi"]["folds"]))


def test_purity_filter_excludes_boundary_tiles():
    cfg = CFG
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:, :20] = INV
    tissue = np.ones((40, 40), dtype=bool)
    cands = wsi.tile_candidates(mask, (1.0, 1.0), tissue, (1.0, 1.0), (40, 40), 10, cfg)
    assert {(c["x0"], c["label"]) for c in cands} == {(0, "Invasive"), (10, "Invasive")}
    mask[:, 15:25] = INV  # still pure
    mask[0:2, 10:20] = 0  # tile (10, 0): purity 0.8 < 0.9
    cands = wsi.tile_candidates(mask, (1.0, 1.0), tissue, (1.0, 1.0), (40, 40), 10, cfg)
    assert (10, 0) not in {(c["x0"], c["y0"]) for c in cands}
    tissue[:, :] = False
    assert wsi.tile_candidates(mask, (1.0, 1.0), tissue, (1.0, 1.0), (40, 40), 10, cfg) == []


# ---- end to end -------------------------------------------------------------------------------------------------
def _slide_array(w=1200, h=800, seed=0):
    arr = st.tissue_image((w, h), seed=seed)
    arr[760:, :] = 250  # glass band
    return arr


@pytest.fixture()
def wsi_run(tmp_path, monkeypatch):
    cfg_path, cfg = st.small_settings(tmp_path, **{
        "wsi.window": [64, 48], "margin.strip_px": 8, "margin.segment_px": 8, "wsi.windows_per_stratum": 2,
        "wsi.max_tiles_per_class_per_wsi": {"g512": 20, "g256": 30}})
    bach = tmp_path / "bach"
    wdir = bach / cfg["dataset"]["official"]["wsi_subdir"]
    slides = {"A01": st.FakeSlide(_slide_array(seed=1)), "A02": st.FakeSlide(_slide_array(seed=2))}
    st.write_bach_xml(wdir / "A01.xml", [("Invasive carcinoma", st.rect(0, 0, 300, 250)),
                                         ("Carcinoma in situ", st.rect(650, 480, 670, 495)),
                                         ("Benign", st.rect(900, 50, 1150, 300)),
                                         ("Carcinoma in situ", st.rect(1000, 100, 1050, 150))], use_attribute=True)
    st.write_bach_xml(wdir / "A02.xml", [("Carcinoma invasive", st.rect(600, 300, 1000, 700)),
                                         ("Benign", st.rect(50, 50, 400, 400))])
    for name in ("A01", "A02", "01", "02"):
        (wdir / f"{name}.svs").write_bytes(b"fake")
    st.write_wsi_patient_xlsx(bach / cfg["dataset"]["patients"]["wsi_file"], n_labeled=2)
    # gt thumbnail for A01 drawn from the XML with the official colors (rasterizer check)
    _, regs = wsi.parse_bach_xml(wdir / "A01.xml", cfg)
    m, _ = wsi.label_mask((1200, 800), regs, (120, 80), cfg)
    thumb = np.zeros((80, 120, 3), np.uint8)
    for code, ch in ((BEN, 0), (INS, 1), (INV, 2)):
        thumb[m == code, ch] = 255
    (wdir / "gt_thumbnails").mkdir()
    Image.fromarray(thumb).save(wdir / "gt_thumbnails" / "A01.png")
    monkeypatch.setattr(wsi, "SLIDE_OPENER", lambda p: slides[__import__("pathlib").Path(p).stem])
    data = tmp_path / "data"
    code = wsi.main(["--config", str(cfg_path), "--data-root", str(data), "--bach-root", str(bach)])
    assert code == 0
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    return {"cfg": cfg, "cfg_path": cfg_path, "bach": bach, "data": data, "slides": slides,
            "tiles": read(data / "wsi_tiles.csv"), "windows": read(schemas.wsi_windows_csv(data)),
            "manifest": read(schemas.embed_manifest_csv(data)),
            "meta": json.loads(schemas.step_meta_json(data, "wsi_extract").read_text())}


def test_tiles_are_labeled_pure_resampled_and_capped(wsi_run):
    tiles, cfg, data = wsi_run["tiles"], wsi_run["cfg"], wsi_run["data"]
    assert set(tiles["wsi_id"]) == {"A01", "A02"}
    assert set(tiles["label"]) == {"Invasive", "InSitu", "Benign"}
    assert (tiles["purity"].astype(float) >= cfg["wsi"]["min_label_purity"]).all()
    side = {"g512": wsi.side0(16, 0.42, 0.466667), "g256": wsi.side0(8, 0.42, 0.466667)}
    assert (tiles["side0"].astype(int) == tiles["grid"].map(side)).all()
    counts = tiles.groupby(["wsi_id", "grid", "label"]).size()
    caps = cfg["wsi"]["max_tiles_per_class_per_wsi"]
    assert all(n <= caps[g] for (_, g, _), n in counts.items())
    assert counts.loc[("A01", "g512", "Invasive")] == caps["g512"]  # big region hits the cap
    # nested InSitu inside Benign keeps InSitu
    ins = tiles[(tiles["wsi_id"] == "A01") & (tiles["label"] == "InSitu")]
    assert ((ins["x0"].astype(int) >= 640) & (ins["x0"].astype(int) < 1060)).all() and len(ins) > 0
    t = tiles.iloc[0]
    with Image.open(data / t["path"]) as im:
        assert im.size == (cfg["tiles"]["grids"][t["grid"]]["size"],) * 2
    # every tile lies fully inside a region of its label (level-0 geometry check, invasive A02 box)
    a2 = tiles[(tiles["wsi_id"] == "A02") & (tiles["label"] == "Invasive")]
    x0, y0, s = a2["x0"].astype(int), a2["y0"].astype(int), a2["side0"].astype(int)
    assert ((x0 >= 598) & (x0 + s <= 1002) & (y0 >= 298) & (y0 + s <= 702)).all()


def test_windows_strata_truth_and_folds(wsi_run):
    win, cfg, data, slides = wsi_run["windows"], wsi_run["cfg"], wsi_run["data"], wsi_run["slides"]
    schemas.validate_frame(schemas.WSI_WINDOWS, win)
    assert set(win["stratum"]) == {"edge_tumor", "central_only", "none"}
    assert (win.groupby(["wsi_id", "stratum"]).size() <= cfg["wsi"]["windows_per_stratum"]).all()
    for w in win.itertuples(index=False):
        truths = [getattr(w, f"truth_{m}") for m in schemas.MARGINS]
        assert (w.stratum == "edge_tumor") == ("tumor" in truths)
        with Image.open(data / w.path) as im:
            assert im.size == tuple(cfg["wsi"]["window"])
    # truth recomputed from level-0 geometry for A02's invasive box (600..1000 x 300..700)
    mpp0, target = 0.466667, 0.42
    for w in win[win["wsi_id"] == "A02"].itertuples(index=False):
        x0, y0 = round(int(w.x) * target / mpp0), round(int(w.y) * target / mpp0)
        w0, h0 = wsi.side0(64, target, mpp0), wsi.side0(48, target, mpp0)
        strip_x, strip_y = 8 * w0 / 64, 8 * h0 / 48
        left_hit = (x0 < 1000) and (x0 + strip_x > 600) and (y0 < 700) and (y0 + h0 > 300)
        if w.truth_medial == "tumor":
            assert left_hit
    # folds: one per patient, windows and tiles of a slide share it
    folds = json.loads((data / "wsi_folds.json").read_text())
    assert set(folds) == {"wsi_p1", "wsi_p2"} and len(set(folds.values())) == 2
    assert (win.groupby("wsi_id")["fold"].nunique() == 1).all()
    tiles = wsi_run["tiles"]
    merged = tiles.groupby("wsi_id")["fold"].first().to_dict()
    assert merged == win.groupby("wsi_id")["fold"].first().to_dict()


def test_manifest_rows_and_meta(wsi_run):
    man, tiles, win, cfg = wsi_run["manifest"], wsi_run["tiles"], wsi_run["windows"], wsi_run["cfg"]
    schemas.validate_frame(schemas.EMBED_MANIFEST, man)
    w512 = man[man["set"] == "wsi512"]
    assert len(w512) == (tiles["grid"] == "g512").sum()
    assert (w512["view"] == "0").all() and (w512["tile_idx"] == "-1").all()
    assert set(w512["label"]) == {"Invasive", "InSitu", "Benign"}
    assert set(man.loc[man["set"].str.startswith("wsi"), "group"]) == {"wsi_p1", "wsi_p2"}
    wm = man[man["set"] == "wsi_margin256"]
    n = cfg["tiles"]["grids"]["g256"]["rows"] * cfg["tiles"]["grids"]["g256"]["cols"]
    assert len(wm) == len(win) * n * cfg["encoders"]["views"]["g256"]
    assert set(wm["case_id"]) == set(win["window_id"])
    meta = wsi_run["meta"]
    a01 = [s for s in meta["summary"]["per_slide"] if s["wsi_id"] == "A01"][0]
    assert a01["gt_thumbnail_check"]["pixel_agreement"] >= 0.95
    assert a01["overlap_px"].get("InSitu", 0) > 0
    assert any("unannotated" in c for c in meta["caveats"])


def test_resume_and_limit(wsi_run):
    data, cfg_path, bach = wsi_run["data"], wsi_run["cfg_path"], wsi_run["bach"]
    probe = data / wsi_run["tiles"]["path"].iloc[0]
    mtime = probe.stat().st_mtime_ns
    assert wsi.main(["--config", str(cfg_path), "--data-root", str(data), "--bach-root", str(bach)]) == 0
    assert probe.stat().st_mtime_ns == mtime
    again = pd.read_csv(schemas.wsi_windows_csv(data), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(again, wsi_run["windows"])
    assert wsi.main(["--config", str(cfg_path), "--data-root", str(data), "--bach-root", str(bach), "--limit", "1"]) == 0
    assert set(pd.read_csv(data / "wsi_tiles.csv")["wsi_id"]) == {"A01"}


def test_missing_slides_fail(tmp_path):
    cfg_path, _ = st.small_settings(tmp_path)
    assert wsi.main(["--config", str(cfg_path), "--data-root", str(tmp_path / "d"), "--bach-root",
                     str(tmp_path / "none")]) == 2


def test_wsi_module_is_portable_without_openslide_at_import():
    assert st.delta_import_problems("pipeline/wsi_extract.py", st.DELTA_ALLOWED - {"sklearn", "scipy"} | {"openslide"}) == []
    src = (st.REPO / "pipeline" / "wsi_extract.py").read_text()
    assert "\nimport openslide" not in src  # lazily imported inside open_slide()


def test_fake_slide_reads_like_openslide():
    s = st.FakeSlide(_slide_array(100, 80))
    assert s.level_dimensions[1] == (50, 40) and s.level_downsamples[2] == pytest.approx(100 / 12)
    out = wsi.read_rgb(s, (90, 70), 0, (20, 20))
    assert out.size == (20, 20) and np.asarray(out)[15, 15].tolist() == [255, 255, 255]  # outside = white
    assert common.tissue_mask(out, CFG)[:10, :10].mean() > 0.5
