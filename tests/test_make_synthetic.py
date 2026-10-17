"""pipeline.make_synthetic: schema-exact synthetic data root with planted structure (Agent 3 CP2)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core import cli, config, schemas
from pipeline import make_synthetic
from tests.a3_synthetic import pooled, synthetic_root  # noqa: F401  (fixture)

CFG = config.load_settings()
CLASSES = list(CFG["classes"])


def read(root, table, path=None):
    return pd.read_csv(path or root / table.filename, keep_default_na=False, na_values=[""], low_memory=False,
                       dtype={"case_id": str, "patient_id": str, "group": str})


# ---- schema exactness ------------------------------------------------------------------------------------------------
def test_marker_and_core_tables_validate(synthetic_root):
    root = synthetic_root
    assert cli.is_synthetic_root(root)
    for table in (schemas.CASES, schemas.TILES, schemas.EMBED_MANIFEST, schemas.COMPOSITES, schemas.WSI_WINDOWS):
        schemas.validate_frame(table, read(root, table))
    for table, path in ((schemas.CELL_TILES, schemas.cell_tiles_csv(root)),
                        (schemas.CELL_FEATURES, schemas.cell_features_csv(root)),
                        (schemas.CELL_FEATURES_TILE, schemas.cell_features_csv(root, per_tile=True))):
        schemas.validate_frame(table, read(root, table, path))


def test_every_json_is_stamped_synthetic_and_specs_validate(synthetic_root):
    root = synthetic_root
    files = list(root.rglob("*.json"))
    assert files
    for f in files:
        data = json.loads(f.read_text())
        assert data.get("synthetic") is True, f
    assert not schemas.INVENTORY.problems(json.loads(schemas.inventory_json(root).read_text()))
    assert not schemas.STEP_META.problems(json.loads(schemas.step_meta_json(root, "make_synthetic").read_text()))
    cell_files = list((root / "cells").rglob("*.json"))
    assert cell_files
    for f in cell_files[:20]:
        data = json.loads(f.read_text())
        assert not schemas.CELL_JSON.problems(data)
        nu = data["nuclei"][0]
        assert set(nu) >= {"centroid_um", "area_um2", "perimeter_um", "eccentricity", "solidity", "type",
                           "type_prob", "edge", "contour_px"}
        assert nu["type"] in schemas.CELL_TYPES


def test_cases_follow_thunder_ranges_lanes_and_groups(synthetic_root):
    cases = read(synthetic_root, schemas.CASES)
    assert len(cases) == CFG["dataset"]["n_images"]
    assert cases.case_id.is_unique and (cases.source == "synthetic").all()
    assert cases.split.value_counts().to_dict() == CFG["split"]["expected_counts"]
    splits = config.split_assignments(CFG)
    assert all(splits[i] == s for i, s in zip(cases.source_index, cases.split))
    order = CFG["split"]["class_order"]
    assert all(order[i // 100] == lab for i, lab in zip(cases.source_index, cases.label))
    # opaque IDs: sorting by ID does not sort by label
    assert cases.sort_values("case_id").label.tolist() != sorted(cases.label)
    # patient groups never cross splits (block groups b<index // block_size> may straddle a range boundary, as in
    # the real schema; they are only used within one split: train CV, test bootstrap)
    assert cases[cases.group.str.startswith("p")].groupby("group").split.nunique().max() == 1
    known = cases.patient_id.notna()
    assert 0.25 < known.mean() < 0.5
    assert (cases.loc[known, "group"] == "p" + cases.loc[known, "patient_id"].astype(str)).all()
    assert cases[known].groupby("patient_id").label.nunique().max() > 1          # patients span classes
    # lanes: surgical only in test, ~25%, covering each must-cover group
    surg = cases[cases.lane == "surgical"]
    assert set(surg.split) == {"test"}
    assert len(surg) == round(CFG["lanes"]["surgical_fraction"] * (cases.split == "test").sum())
    for group in CFG["lanes"]["surgical_must_cover"]:
        assert surg.label.isin(group).any()


def test_tiles_both_grids_with_realistic_qc(synthetic_root):
    tiles = read(synthetic_root, schemas.TILES)
    cases = read(synthetic_root, schemas.CASES)
    for grid, spec in CFG["tiles"]["grids"].items():
        part = tiles[tiles.grid == grid]
        assert len(part) == len(cases) * spec["rows"] * spec["cols"]
        assert (part.x == part.col * spec["size"]).all() and (part.y == part.row * spec["size"]).all()
        assert (part.tile_idx == part.row * spec["cols"] + part.col).all()
    assert 0.03 < 1 - tiles.qc_pass.mean() < 0.3
    kept = tiles[(tiles.grid == "g512") & (tiles.qc_pass == 1)].groupby("case_id").size()
    kept = kept.reindex(cases.case_id, fill_value=0)
    insufficient = kept[kept < CFG["qc"]["min_tiles_kept"]]
    assert 1 <= len(insufficient) <= 5
    assert set(cases.set_index("case_id").loc[insufficient.index, "split"]) >= {"test"}
    assert (synthetic_root / tiles.path.iloc[0]).is_file() and (synthetic_root / tiles.path.iloc[-1]).is_file()
    assert all((synthetic_root / p).is_file() for p in cases.display_path.head(10))


def test_manifest_covers_every_set_with_views(synthetic_root):
    m = read(synthetic_root, schemas.EMBED_MANIFEST)
    assert set(m["set"]) == set(schemas.EMBED_SETS)
    assert m.row_id.is_unique
    views = m.groupby("item_id").view.nunique()
    b512 = m[m["set"] == "bach512"].item_id.unique()
    assert (views.loc[b512] == CFG["encoders"]["views"]["g512"]).all()
    assert (views.loc[m[m["set"] == "nct"].item_id.unique()] == 1).all()
    deg = m[m["set"] == "deg512"]
    assert (deg.view == 0).all()                                     # ruling M9
    assert set(deg.degrade_type) == set(CFG["degrade"]["types"]) and set(deg.split) == {"calibration", "test"}
    assert (m[m["set"] == "quilt"].label.str.match(r"^[a-z_]+:(pos|neg)$")).all()


def test_embeddings_for_every_encoder_head_and_set(synthetic_root):
    root = synthetic_root
    for enc in CFG["encoders"]["priority"]:
        meta = json.loads(schemas.emb_meta_json(root, enc).read_text())
        assert not schemas.EMB_META.problems(meta)
        assert meta["synthetic"] is True
        heads = set(meta["heads"])
        assert ("align" in heads) == (enc in CFG["encoders"]["text_aligned"])
        assert "probe" in heads
        for set_name, info in meta["sets"].items():
            for head in heads:
                x = np.load(schemas.emb_npy(root, enc, set_name, head))
                idx = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, enc, set_name, head))
                schemas.validate_frame(schemas.EMB_INDEX, idx)
                assert x.dtype == np.float16 and x.shape == (len(idx), meta["heads"][head]["dim"])
                assert (idx.emb_row == np.arange(len(idx))).all() and (idx.view == -1).all()
                assert idx.item_id.is_unique and info["n_items"] == len(idx)
                if head == "align":
                    np.testing.assert_allclose(np.linalg.norm(x.astype(float), axis=1), 1.0, atol=1e-2)
                if info["views_file"]:
                    xv = np.load(schemas.emb_npy(root, enc, set_name, head, views=True))
                    vi = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, enc, set_name, head, views=True))
                    assert set_name in ("bach512", "bach256")             # ruling M9: every split
                    assert set(vi.split) == set(schemas.SPLITS) and len(vi) == len(xv) == len(idx) * info["n_views"]
                    v0 = xv[(vi.view == 0).to_numpy()]
                    assert v0.shape == x.shape
    # dims differ between encoders (never comparable)
    dims = {enc: json.loads(schemas.emb_meta_json(root, enc).read_text())["heads"]["probe"]["dim"]
            for enc in ("conch", "musk")}
    assert dims["conch"] != dims["musk"]


def test_deg_and_comp_ids(synthetic_root):
    idx = read(synthetic_root, schemas.EMB_INDEX, schemas.emb_index_csv(synthetic_root, "conch", "deg512", "probe"))
    cases = read(synthetic_root, schemas.CASES).set_index("case_id")
    assert set(idx.case_id) <= set(cases.index)                      # source case IDs + degrade columns
    assert (idx.degrade_severity >= 1).all()
    comp = read(synthetic_root, schemas.COMPOSITES)
    cidx = read(synthetic_root, schemas.EMB_INDEX, schemas.emb_index_csv(synthetic_root, "uni2h", "comp256", "probe"))
    assert set(cidx.case_id) == set(comp.case_id)
    assert (cases.loc[comp.carcinoma_case_id, "label"].isin(CFG["carcinoma_classes"])).all()
    assert (cases.loc[comp.benign_case_id, "label"].isin(CFG["benign_classes"])).all()
    assert (cases.loc[comp.carcinoma_case_id, "split"] == "test").all()


def test_text_npz_keyed_by_expand_prompts(synthetic_root):
    bank, _ = make_synthetic.concept_bank(CFG)
    expected = [p.id for p in schemas.expand_prompts(bank)] + [schemas.nct_prompt_id(c) for c in CFG["nct"]["classes"]]
    for enc in CFG["encoders"]["text_aligned"]:
        z = np.load(schemas.text_npz(synthetic_root, enc))
        assert not schemas.TEXT_NPZ.problems(z)
        assert list(z["ids"]) == expected and len(z["texts"]) == len(expected)
        assert z["emb"].dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(z["emb"], axis=1), 1.0, atol=1e-5)
        assert str(z["model"]) == enc and bool(z["synthetic"])


def test_builtin_bank_is_schema_valid():
    bank = make_synthetic.builtin_bank(CFG)
    assert bank.ids == list(CFG["concepts"]["ids"])


# ---- planted structure ------------------------------------------------------------------------------------------------
def fit_eval(root, enc, set_name, head="probe"):
    p = pooled(root, enc, set_name, head)
    cases = read(root, schemas.CASES).set_index("case_id").loc[p.index]
    tr, te = (cases.split == "train").to_numpy(), (cases.split == "test").to_numpy()
    sc = StandardScaler().fit(p.values[tr])
    m = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(p.values[tr]), cases.label[tr])
    return m, sc, p.index[te], m.predict(sc.transform(p.values[te])), cases.label[te].to_numpy()


def test_classes_separable_but_insitu_invasive_confusable(synthetic_root):
    _, _, _, pred, true = fit_eval(synthetic_root, "conch", "bach512")
    acc = (pred == true).mean()
    assert 0.6 < acc < 0.97
    confusions = ((true == "InSitu") & (pred == "Invasive")).sum() + ((true == "Invasive") & (pred == "InSitu")).sum()
    assert confusions >= 2
    agg = (true == CFG["aggressive_class"])
    assert ((pred == CFG["aggressive_class"]) == agg).mean() > 0.8


def test_musk_correlated_but_not_identical_to_conch(synthetic_root):
    _, _, ids_c, pred_c, _ = fit_eval(synthetic_root, "conch", "bach512")
    _, _, ids_m, pred_m, _ = fit_eval(synthetic_root, "musk", "bach512")
    assert list(ids_c) == list(ids_m)
    agree = (pred_c == pred_m).mean()
    assert 0.55 < agree < 0.98


def concept_scores(root):
    z = np.load(schemas.text_npz(root, "conch"))
    ids = list(z["ids"])
    x = np.load(schemas.emb_npy(root, "conch", "bach512", "align")).astype(float)
    idx = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, "conch", "bach512", "align"))
    labels = idx.case_id.map(read(root, schemas.CASES).set_index("case_id").label).to_numpy()
    out = {}
    for cid in CFG["concepts"]["ids"]:
        pos = [i for i, s in enumerate(ids) if s.startswith(f"concept:{cid}:pos:")]
        con = [i for i, s in enumerate(ids) if s.startswith(f"concept:{cid}:con:")]
        score = x @ z["emb"][pos].mean(0) - x @ z["emb"][con].mean(0)
        out[cid] = pd.Series(score).groupby(labels).mean()
    return out


def test_concepts_highest_in_expected_class_with_weak_mitoses(synthetic_root):
    means = concept_scores(synthetic_root)
    expected = {"invasion": "Invasive", "duct_confined": "InSitu", "normal_lobules": "Normal",
                "benign_stroma_fat": "Benign", "tubules": "Normal", "pleomorphism": "Invasive"}
    for cid, cls in expected.items():
        assert means[cid].idxmax() == cls, (cid, means[cid].to_dict())
    gap = {cid: means[cid].max() - means[cid].min() for cid in means}
    assert gap["mitoses"] < gap["invasion"] and gap["mitoses"] < gap["duct_confined"]


def test_degradation_lowers_accuracy_and_moves_off_manifold(synthetic_root):
    root = synthetic_root
    m, sc, _, _, _ = fit_eval(root, "conch", "bach512")
    x = np.load(schemas.emb_npy(root, "conch", "deg512", "probe")).astype(float)
    idx = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, "conch", "deg512", "probe"))
    df = pd.DataFrame(x)
    df["key"] = idx.case_id + "|" + idx.degrade_severity.astype(str)
    g = df.groupby("key").mean()
    case = [k.split("|")[0] for k in g.index]
    sev = np.array([int(k.split("|")[1]) for k in g.index])
    true = read(root, schemas.CASES).set_index("case_id").loc[case, "label"].to_numpy()
    ok = m.predict(sc.transform(g.values)) == true
    assert ok[sev == 3].mean() < ok[sev == 1].mean()
    # off-manifold: cosine distance of degraded align tiles to their nearest train tiles grows with severity
    xa = np.load(schemas.emb_npy(root, "conch", "deg512", "align")).astype(float)
    ia = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, "conch", "deg512", "align"))
    ref = np.load(schemas.emb_npy(root, "conch", "bach512", "align")).astype(float)
    ri = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, "conch", "bach512", "align"))
    train, cal = ref[(ri.split == "train").to_numpy()], ref[(ri.split == "calibration").to_numpy()]

    def knn_dist(q):
        return (1 - np.sort(q @ train.T, axis=1)[:, -CFG["reliability"]["k"]:]).mean()

    d = {s: knn_dist(xa[(ia.degrade_severity == s).to_numpy()][:400]) for s in (1, 2, 3)}
    assert knn_dist(cal) < d[1] < d[2] < d[3]


def test_composite_carcinoma_signal_only_in_its_half(synthetic_root):
    root = synthetic_root
    comp = read(root, schemas.COMPOSITES).set_index("case_id")
    x = np.load(schemas.emb_npy(root, "uni2h", "comp256", "probe")).astype(float)
    idx = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, "uni2h", "comp256", "probe"))
    tx = np.load(schemas.emb_npy(root, "uni2h", "bach256", "probe")).astype(float)
    tidx = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, "uni2h", "bach256", "probe"))
    labels = tidx.case_id.map(read(root, schemas.CASES).set_index("case_id").label)
    train = (tidx.split == "train").to_numpy()
    y = labels.isin(CFG["carcinoma_classes"]).to_numpy()[train]
    sc = StandardScaler().fit(tx[train])
    probe = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(tx[train]), y)
    p = probe.predict_proba(sc.transform(x))[:, 1]
    cols, rows = CFG["tiles"]["grids"]["g256"]["cols"], CFG["tiles"]["grids"]["g256"]["rows"]
    carc_half, benign_half = [], []
    for k, (cid, tile) in enumerate(zip(idx.case_id, idx.tile_idx)):
        r, c = divmod(int(tile), cols)
        side = comp.loc[cid, "carcinoma_side"]
        inside = {"left": c < cols // 2, "right": c >= cols // 2, "top": r < rows // 2, "bottom": r >= rows // 2}[side]
        (carc_half if inside else benign_half).append(p[k])
    assert np.mean(carc_half) > np.mean(benign_half) + 0.2


def test_composite_truth_table(synthetic_root):
    comp = read(synthetic_root, schemas.COMPOSITES)
    names = CFG["margin"]["names"]
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    for row in comp.itertuples():
        assert getattr(row, f"truth_{names[row.carcinoma_side]}") == "tumor_side"
        assert getattr(row, f"truth_{names[opposite[row.carcinoma_side]]}") == "exact_negative"
    assert comp.orientation.nunique() == CFG["composites"]["orientations"]


def test_wsi_windows_truth_matches_planted_tiles(synthetic_root):
    root = synthetic_root
    win = read(root, schemas.WSI_WINDOWS).set_index("window_id")
    idx = read(root, schemas.EMB_INDEX, schemas.emb_index_csv(root, "uni2h", "wsi_margin256", "probe"))
    cols = CFG["tiles"]["grids"]["g256"]["cols"]
    rows = CFG["tiles"]["grids"]["g256"]["rows"]
    names = CFG["margin"]["names"]
    for wid, part in idx.groupby("case_id"):
        tumor = np.zeros((rows, cols), dtype=bool)
        for t, lab in zip(part.tile_idx, part.label):
            tumor[divmod(int(t), cols)] = lab in CFG["carcinoma_classes"]
        strips = {"top": tumor[0], "bottom": tumor[-1], "left": tumor[:, 0], "right": tumor[:, -1]}
        for edge, margin in names.items():
            assert win.loc[wid, f"truth_{margin}"] == ("tumor" if strips[edge].any() else "clear")
    assert set(win.stratum) == {"edge_tumor", "central_only", "none"}
    assert win.groupby("patient_id").fold.nunique().max() == 1


def test_cell_features_follow_expected_directions(synthetic_root):
    cf = read(synthetic_root, schemas.CELL_FEATURES, schemas.cell_features_csv(synthetic_root))
    cf = cf.merge(read(synthetic_root, schemas.CASES)[["case_id", "label"]], on="case_id")
    carc = cf.label.isin(CFG["carcinoma_classes"])
    for f in ("neo_fraction", "nuc_area_mean", "nuc_area_cv"):
        assert cf.loc[carc, f].mean() > cf.loc[~carc, f].mean(), f
    assert cf.loc[carc, "epi_fraction"].mean() < cf.loc[~carc, "epi_fraction"].mean()
    assert cf["nuc_area_cv__missing"].sum() > 0                    # benign cases often lack 20 neoplastic nuclei
    assert ((cf.nuc_area_cv__missing == 1) == cf.nuc_area_cv.isna()).all()
    pct = cf.neo_fraction__ref_pct.dropna()
    assert pct.between(0, 1).all()


# ---- CLI behavior ------------------------------------------------------------------------------------------------------------
def test_deterministic_for_a_seed(tmp_path, synthetic_root):
    """Same seed => identical files (the cells outputs are only added by --with-cells, so the rest must match)."""
    out = tmp_path / "again"
    assert make_synthetic.main(["--out", str(out), "--small"]) == 0
    for rel in ("cases.csv", "tiles.csv", "composites.csv", "wsi_windows.csv", "emb/conch/bach512_probe.npy",
                "emb/uni2h/comp256_probe.npy", "emb/musk/deg512_align.npy"):
        assert (out / rel).read_bytes() == (synthetic_root / rel).read_bytes(), rel
    a, b = np.load(out / "text/conch.npz"), np.load(synthetic_root / "text/conch.npz")
    np.testing.assert_array_equal(a["emb"], b["emb"])
    # skip when already built with the same arguments
    before = (out / "meta/make_synthetic.json").stat().st_mtime_ns
    assert make_synthetic.main(["--out", str(out), "--small"]) == 0
    assert (out / "meta/make_synthetic.json").stat().st_mtime_ns == before
    # different arguments without --force: refuse, keep downstream outputs
    (out / "downstream.csv").write_text("x")
    assert make_synthetic.main(["--out", str(out), "--small", "--seed", "3"]) == 3
    assert (out / "downstream.csv").exists()


def test_refuses_non_synthetic_non_empty_dir(tmp_path):
    (tmp_path / "precious.txt").write_text("keep me")
    assert make_synthetic.main(["--out", str(tmp_path), "--small"]) == 2
    assert (tmp_path / "precious.txt").read_text() == "keep me"


def test_different_seed_changes_data():
    a, b = make_synthetic.World(CFG, 44, True), make_synthetic.World(CFG, 7, True)
    assert a.cases.label.tolist() != b.cases.label.tolist()      # same ID set, different seeded permutation
    assert not np.allclose(a.lat256, b.lat256)


def test_every_manifest_path_exists_and_crop_fits(synthetic_root):
    from PIL import Image
    m = read(synthetic_root, schemas.EMBED_MANIFEST)
    first = m.drop_duplicates("path")
    sample = first.groupby("set").head(3)
    for row in sample.itertuples():
        path = schemas.resolve(synthetic_root, row.path)
        assert path.is_file(), row.path
        with Image.open(path) as img:
            assert row.x + row.w <= img.width and row.y + row.h <= img.height, (row.path, img.size)
