"""pipeline.mine_captions on a synthetic Quilt CSV / PathGen tree: rules, negation, exclusion, organ and stain
filters, image-level aggregation, caps per group, manifest rows."""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from core import config, schemas
from tests import synthetic_tree as st

mine = importlib.import_module("pipeline.mine_captions")
CFG = config.load_settings()
RULES = mine.load_rules(CFG)


def test_rules_cover_exactly_the_settings_concepts():
    assert sorted(RULES.ids) == sorted(CFG["concepts"]["ids"])
    text = (st.REPO / "config" / "caption_mining.yaml").read_text()
    assert "caption co-mention != tissue truth" in text.lower()


@pytest.mark.parametrize("caption,concept,pol", [
    ("Invasive ductal carcinoma infiltrating the stroma", "invasion", "pos"),
    ("Ductal carcinoma in situ with comedonecrosis", "duct_confined", "pos"),
    ("Ductal carcinoma in situ with comedonecrosis", "invasion", "neg"),
    ("Ductal carcinoma in situ with comedonecrosis", "necrosis", "pos"),
    ("Marked nuclear pleomorphism and prominent nucleoli", "pleomorphism", "pos"),
    ("Bland uniform nuclei", "pleomorphism", "neg"),
    ("Numerous mitotic figures", "mitoses", "pos"),
    ("Dense lymphocytic infiltrate around the tumor", "lymphocytes", "pos"),
    ("Normal breast lobules with two-layered epithelium", "normal_lobules", "pos"),
    ("Benign adipose tissue and fibrous stroma", "benign_stroma_fat", "pos"),
    ("Well-formed tubules in a tubular carcinoma", "tubules", "pos"),
])
def test_classify(caption, concept, pol):
    assert RULES.classify(caption).get(concept) == pol


def test_negation_and_exclusion():
    assert "necrosis" not in {c for c, p in RULES.classify("there is no necrosis here").items() if p == "pos"}
    assert RULES.classify("fat necrosis in the breast").get("necrosis") != "pos"  # excluded
    assert RULES.classify("pleomorphic adenoma").get("pleomorphism") != "pos"
    assert RULES.classify("microinvasive carcinoma").get("invasion") != "pos"
    assert RULES.classify("without tubule formation").get("tubules") != "pos"


def test_breast_and_group_helpers():
    assert mine.pathology_is_breast("['Breast', 'Gynecologic']", ["breast"])
    assert mine.pathology_is_breast("['Breast pathology']", ["breast"])
    assert not mine.pathology_is_breast("['Dermatopathology']", ["breast"])
    assert mine.quilt_group("quilt", "YC5jOhcQGes_image_ede69e4d.jpg") == "yt:YC5jOhcQGes"
    assert mine.quilt_group("openpath", "994701482116173824_0.jpg") == "tw:994701482116173824"
    assert mine.quilt_group("pubmed", "x.jpg") == "pubmed:x.jpg"
    assert mine.safe_image_id("a b.c") == "a-b-c"


def _vid(k):
    return f"{k:011d}"[-11:].replace("0", "A")


def _quilt_rows():
    rows = []
    # 4 frames of video A + 3 of video B with invasion captions -> group cap 2 keeps 2 + 2
    for v, n in (("VIDEOAAAAAA", 4), ("VIDEOBBBBBB", 3)):
        for i in range(n):
            rows.append({"caption": "invasive ductal carcinoma infiltrating stroma", "image_path": f"{v}_image_{i}.jpg"})
    rows += [
        {"caption": "ductal carcinoma in situ, cribriform pattern", "image_path": "VIDEOCCCCCC_image_0.jpg"},
        # same image, second caption: positive wins for duct_confined, image-level aggregation
        {"caption": "DCIS confined to the duct", "image_path": "VIDEOCCCCCC_image_0.jpg"},
        # stain filter drops the whole image even though another caption matches
        {"caption": "invasive carcinoma", "image_path": "VIDEODDDDDD_image_0.jpg"},
        {"caption": "HER2 immunohistochemistry is positive", "image_path": "VIDEODDDDDD_image_0.jpg"},
        # not breast -> ignored
        {"caption": "invasive adenocarcinoma of colon", "image_path": "VIDEOEEEEEE_image_0.jpg",
         "pathology": "['Gastrointestinal']"},
        # breast by caption although pathology is empty (openpath)
        {"caption": "breast: dense lymphocytic infiltrate", "image_path": "123456_0.jpg", "subset": "openpath",
         "pathology": ""},
        # non-histology video flag -> ignored
        {"caption": "invasive carcinoma drawing", "image_path": "VIDEOFFFFFF_image_0.jpg", "not_histology": "1"},
        # file missing on disk -> counted and skipped
        {"caption": "tumor necrosis", "image_path": "VIDEOGGGGGG_image_0.jpg", "_no_file": True},
    ]
    return rows


@pytest.fixture()
def mined(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    rows = _quilt_rows()
    root = st.make_quilt_tree(tmp_path / "quilt", [{k: v for k, v in r.items() if k != "_no_file"} for r in rows])
    for r in rows:
        if r.get("_no_file"):
            (root / "images_part_1" / "quilt_1m" / r["image_path"]).unlink()
    rules_path = tmp_path / "rules.yaml"
    import yaml

    data = yaml.safe_load((st.REPO / "config" / "caption_mining.yaml").read_text())
    data["caps"]["max_images_per_group"] = 2
    data["quilt"]["chunksize"] = 3  # exercise chunking
    rules_path.write_text(yaml.safe_dump(data))
    data_root = tmp_path / "data"
    code = mine.main(["--config", str(cfg_path), "--data-root", str(data_root), "--quilt-root", str(root),
                      "--rules", str(rules_path), "--pathgen-root", str(tmp_path / "nopg")])
    assert code == 0
    man = pd.read_csv(schemas.embed_manifest_csv(data_root), dtype=str, keep_default_na=False)
    meta = json.loads(schemas.step_meta_json(data_root, "mine_captions").read_text())
    return man, meta, root, cfg_path, rules_path, data_root


def test_quilt_manifest_rows(mined):
    man, meta, root, *_ = mined
    schemas.validate_frame(schemas.EMBED_MANIFEST, man)
    q = man[man["set"] == "quilt"]
    inv = q[q["label"] == "invasion:pos"]
    assert len(inv) == 4 and inv["group"].value_counts().to_dict() == {"yt:VIDEOAAAAAA": 2, "yt:VIDEOBBBBBB": 2}
    assert (q.loc[q["label"] == "duct_confined:pos", "case_id"] == "VIDEOCCCCCC_image_0").all()
    assert "VIDEOCCCCCC_image_0" in set(q.loc[q["label"] == "invasion:neg", "case_id"])
    assert not q["case_id"].str.contains("VIDEODDDDDD|VIDEOEEEEEE|VIDEOFFFFFF|VIDEOGGGGGG").any()
    assert set(q.loc[q["label"] == "lymphocytes:pos", "group"]) == {"tw:123456"}
    assert q["row_id"].is_unique and (q["view"] == "0").all() and (q["tile_idx"] == "-1").all()
    assert all(p.startswith("/") for p in q["path"])
    s = meta["summary"]["stats"]
    assert s["quilt_missing_files"] == 1 and s["rows_scanned"] == len(_quilt_rows())
    assert meta["summary"]["pathgen_used"] is False
    assert meta["quilt_report"]["invasion"]["pos"]["examples"]
    assert "caption co-mention" in meta["note"]


def test_limit_caps_each_concept(mined):
    man, meta, root, cfg_path, rules_path, data_root = mined
    assert mine.main(["--config", str(cfg_path), "--data-root", str(data_root), "--quilt-root", str(root),
                      "--rules", str(rules_path), "--limit", "1"]) == 0
    m = pd.read_csv(schemas.embed_manifest_csv(data_root), dtype=str, keep_default_na=False)
    assert (m[m["set"] == "quilt"].groupby("label").size() <= 1).all()


def test_deterministic(mined):
    man, meta, root, cfg_path, rules_path, data_root = mined
    assert mine.main(["--config", str(cfg_path), "--data-root", str(data_root), "--quilt-root", str(root),
                      "--rules", str(rules_path)]) == 0
    again = pd.read_csv(schemas.embed_manifest_csv(data_root), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(again, man)


def test_pathgen_used_only_when_inventory_reports_patches(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    quilt = st.make_quilt_tree(tmp_path / "quilt", [{"caption": "breast tissue, lymphocytes", "image_path": "q.jpg"}])
    pg = st.make_pathgen_tree(tmp_path / "pg", [
        {"wsi_id": "TCGA-AA-3844-01Z-00-DX1.abc", "position": ["1", "2"], "caption": "breast invasive carcinoma"},
        {"wsi_id": "TCGA-BB-0001-01Z-00-DX1.def", "position": ["3", "4"], "caption": "colon adenocarcinoma"}],
        with_patches=True)
    data = tmp_path / "data"
    data.mkdir()
    (data / "inventory.json").write_text(json.dumps({"pathgen": {"has_patch_images": True}}))
    assert mine.main(["--config", str(cfg_path), "--data-root", str(data), "--quilt-root", str(quilt),
                      "--pathgen-root", str(pg)]) == 0
    man = pd.read_csv(schemas.embed_manifest_csv(data), dtype=str, keep_default_na=False)
    p = man[man["set"] == "pathgen"]
    assert "invasion:pos" in set(p["label"]) and "duct_confined:neg" in set(p["label"])
    assert set(p["group"]) == {"tcga:TCGA-AA-3844"}  # patient barcode; the colon entry is not breast


def test_missing_quilt_csv_fails(tmp_path):
    cfg_path, _ = st.small_settings(tmp_path)
    (tmp_path / "q").mkdir()
    assert mine.main(["--config", str(cfg_path), "--data-root", str(tmp_path / "d"), "--quilt-root",
                      str(tmp_path / "q")]) == 2


def test_mine_captions_is_portable():
    assert st.delta_import_problems("pipeline/mine_captions.py", st.DELTA_ALLOWED - {"sklearn", "scipy"}) == []


# ---- review regressions -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("caption,concept,expected", [
    ("necrosis not identified; viable tumor cells", "necrosis", "neg"),      # trailing negation
    ("No atypia. Mitoses are frequent", "mitoses", "pos"),                    # clause break stops the look-back
    ("tumor cells without necrosis but with lymphocytes", "lymphocytes", "pos"),
    ("rare mitoses in a bland tumor", "mitoses", "neg"),                      # excluded positive falls to negative
])
def test_negation_scope(caption, concept, expected):
    assert RULES.classify(caption).get(concept) == expected


def test_organ_filter_rules():
    assert RULES.is_breast("['Breast']", "anything")
    assert not RULES.is_breast("['Liver']", "liver with lobular inflammation and lymphocytes")
    assert RULES.is_breast("['Liver']", "metastatic breast carcinoma in the liver")
    assert RULES.is_breast("", "invasive ductal carcinoma")               # openpath/pubmed: weak term ok
    assert not RULES.is_breast("['Breast', 'Gynecologic']", "endometrial glands")  # multi-organ video needs a term
    assert RULES.is_breast("['Breast', 'Gynecologic']", "lobular carcinoma")


def test_stain_filter_keeps_he_wording():
    assert not RULES.stain.search("cytologic atypia with pleomorphic nuclei, breast")
    assert not RULES.stain.search("grossly the tumor looked firm")
    assert RULES.stain.search("breast FNA cytology smear")
    assert RULES.stain.search("Ki-67 stain shown here")


def test_image_level_breast_and_stain_flags(tmp_path):
    cfg_path, cfg = st.small_settings(tmp_path)
    rows = [
        # image A: breast caption + a non-breast caption mentioning Ki-67 -> dropped (stain is image-level)
        {"caption": "breast tissue with lymphocytes", "image_path": "VIDEOAAAAAA_image_0.jpg", "pathology": ""},
        {"caption": "Ki-67 stain shown here", "image_path": "VIDEOAAAAAA_image_0.jpg", "pathology": ""},
        # image B: second caption has no breast word but carries the necrosis vote
        {"caption": "breast carcinoma", "image_path": "VIDEOBBBBBB_image_0.jpg", "pathology": ""},
        {"caption": "necrosis is seen", "image_path": "VIDEOBBBBBB_image_0.jpg", "pathology": ""},
        # image C: openpath row with empty not_histology is kept
        {"caption": "breast: dense lymphocytic infiltrate", "image_path": "777_0.jpg", "subset": "openpath",
         "pathology": "", "not_histology": ""},
    ]
    root = st.make_quilt_tree(tmp_path / "quilt", rows)
    data = tmp_path / "data"
    assert mine.main(["--config", str(cfg_path), "--data-root", str(data), "--quilt-root", str(root)]) == 0
    man = pd.read_csv(schemas.embed_manifest_csv(data), dtype=str, keep_default_na=False)
    assert not man["case_id"].str.startswith("VIDEOAAAAAA").any()
    assert "necrosis:pos" in set(man.loc[man["case_id"] == "VIDEOBBBBBB_image_0", "label"])
    assert "lymphocytes:pos" in set(man.loc[man["case_id"] == "777_0", "label"])
    meta = json.loads(schemas.step_meta_json(data, "mine_captions").read_text())
    assert meta["summary"]["stats"]["rows_histology_per_subset"]["openpath"] == 1
