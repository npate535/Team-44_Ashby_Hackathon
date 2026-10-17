"""04_concepts: score math, train-only z-scoring, top-k / frac aggregation, prompt-id lookup, spread, variants,
schema-exact outputs and non-trivial eval JSONs on a planted fake data root."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from core import config, io, schemas
from tests.concept_fakes import Fake, run

C = importlib.import_module("pipeline.04_concepts")


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setenv("PATHQUEUE_SITE", "local")


@pytest.fixture
def fake(tmp_path):
    f = Fake(tmp_path / "root")
    f.build()
    return f


def _features(f):
    return pd.read_csv(schemas.case_features_csv(f.root), keep_default_na=False)


def test_score_is_mean_cos_to_positives_minus_mean_cos_to_contrasts():
    pos = np.array([[1.0, 0, 0], [0.6, 0.8, 0]])
    con = np.array([[0, 0, 1.0], [0, 1.0, 0]])
    x = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    score, std = C.raw_scores(x, {"c": (pos, con)}, ["c"])
    want = (x @ pos.T).mean(1) - (x @ con.T).mean(1)
    np.testing.assert_allclose(score[:, 0], want, atol=1e-6)
    s_i = x @ pos.T - (x @ con.T).mean(1, keepdims=True)
    np.testing.assert_allclose(std[:, 0], s_i.std(1), atol=1e-6)


def test_phrasing_embedding_is_the_normalized_template_mean(fake):
    bank = config.load_concept_bank(cfg=fake.cfg)
    text = C.load_text(fake.root, "conch")
    phr = C.phrasing_embeddings(bank, text)
    n_t = len(bank.templates)
    rows = [text[schemas.concept_prompt_id("invasion", "con", 1, t)] for t in range(n_t)]
    want = np.mean([r / np.linalg.norm(r) for r in rows], axis=0)
    np.testing.assert_allclose(phr["invasion"][1][1], want / np.linalg.norm(want), atol=1e-6)
    assert phr["invasion"][0].shape == (len(bank.by_id()["invasion"].positives), text[next(iter(text))].shape[0])
    with pytest.raises(KeyError, match="re-run 03_text"):
        C.phrasing_embeddings(bank, {k: v for k, v in text.items() if not k.startswith("concept:necrosis")})


def test_outputs_are_schema_exact(fake):
    code, log = run(C, fake)
    assert code == 0, log
    feats = _features(fake)
    schemas.validate_frame(schemas.CASE_FEATURES, feats, extra_required=schemas.concept_feature_columns(fake.cfg))
    assert len(schemas.concept_feature_columns(fake.cfg)) == 18
    assert set(schemas.concept_spread_columns(fake.cfg)) <= set(feats.columns)
    tile = io.load_npz(schemas.concepts_tile_npz(fake.root))
    assert schemas.CONCEPTS_TILE_NPZ.problems(tile) == []
    assert tile["concept_ids"].tolist() == fake.ids and tile["z"].shape == (len(tile["case_id"]), 9)
    assert str(tile["grid"]) == "g512" and str(tile["encoder"]) == "conch"
    clean = set(fake.cases()["case_id"])
    assert clean <= set(tile["case_id"].tolist())  # every case of every split
    assert np.isnan(tile["z_cav"]).all()            # no concept vectors yet -> text only
    np.testing.assert_allclose(tile["z"], tile["z_text"])
    assert schemas.STEP_META.problems(io.read_json(schemas.step_meta_json(fake.root, "04_concepts"))) == []


def test_z_scoring_uses_clean_train_qc_tiles_only(fake):
    assert run(C, fake)[0] == 0
    tile = io.load_npz(schemas.concepts_tile_npz(fake.root))
    cases = fake.cases()
    train = set(cases.loc[cases["split"] == "train", "case_id"])
    mask = np.isin(tile["case_id"], list(train)) & (tile["variant"] == "clean")  # npz rows = QC tiles used
    np.testing.assert_allclose(tile["z_text"][mask].mean(0), 0.0, atol=1e-4)
    np.testing.assert_allclose(tile["z_text"][mask].std(0), 1.0, atol=1e-3)
    others = ~mask
    assert abs(tile["z_text"][others].mean()) > 1e-3  # non-train rows are not re-standardized


def test_top_k_and_fraction_aggregation_by_hand(fake):
    assert run(C, fake)[0] == 0
    tile = io.load_npz(schemas.concepts_tile_npz(fake.root))
    feats = _features(fake).set_index("case_id")
    k, zt = fake.cfg["concepts"]["top_k"], fake.cfg["concepts"]["z_threshold"]
    case = "BACH_0000"
    sel = tile["case_id"] == case
    assert sel.sum() == 11 and feats.loc[case, "n_tiles"] == 11  # the QC-failing tile is excluded
    for j, cid in enumerate(fake.ids):
        z = tile["z"][sel, j]
        assert feats.loc[case, f"{cid}__top{k}"] == pytest.approx(np.sort(z)[::-1][:k].mean(), abs=1e-5)
        assert feats.loc[case, f"{cid}__frac"] == pytest.approx((z > zt).mean(), abs=1e-6)
        top = np.argsort(-z, kind="stable")[:k]
        assert feats.loc[case, f"{cid}__spread"] == pytest.approx(tile["spread"][sel, j][top].mean(), abs=1e-5)


def test_planted_concepts_rank_their_class_highest(fake):
    assert run(C, fake)[0] == 0
    feats = _features(fake).merge(fake.cases()[["case_id", "label"]], on="case_id")
    means = feats[feats["variant"] == "clean"].groupby("label")[["invasion__top3", "duct_confined__top3",
                                                                 "normal_lobules__top3"]].mean()
    assert means["invasion__top3"].idxmax() == "Invasive"
    assert means["duct_confined__top3"].idxmax() == "InSitu"
    assert means["normal_lobules__top3"].idxmax() == "Normal"


def test_degraded_and_composite_variants_use_variant_ids_and_source_splits(fake):
    assert run(C, fake)[0] == 0
    feats = _features(fake)
    deg = feats[feats["variant"] == "degraded"]
    assert len(deg) == 6 and all(c.count("_FSIM_") == 1 for c in deg["case_id"]) and set(deg["split"]) == {"test"}
    assert set(deg["n_tiles"]) == {11}  # source-case QC flags apply to degraded tiles
    assert set(deg["source_case_id"]) <= set(feats.loc[feats["variant"] == "clean", "case_id"])
    src = deg.iloc[0]["source_case_id"]
    assert deg.iloc[0]["case_id"] in {schemas.variant_case_id(src, "blur", 3), schemas.variant_case_id(src, "jpeg", 1)}
    comp = feats[feats["variant"] == "composite"]
    assert len(comp) == 1 and comp.iloc[0]["case_id"].startswith("COMP_") and comp.iloc[0]["split"] == "test"   # opaque IDs (M10)
    assert comp.iloc[0]["n_tiles"] == 12


def test_eval_jsons_are_non_trivial(fake):
    assert run(C, fake)[0] == 0
    art = fake.root / "artifacts"
    concepts = io.read_json(schemas.eval_json(art, "concepts"))
    for key in schemas.EVAL_ENVELOPE:
        assert key in concepts
    assert "cross-organ" in concepts["nct"]["label"]
    lym = concepts["nct"]["concepts"]["lymphocytes"]["text"]
    assert lym["estimate"] > 0.8 and lym["lo"] <= lym["estimate"] <= lym["hi"]
    assert set(concepts["nct"]["concepts"]) == set(fake.cfg["nct"]["concept_checks"])
    bach = io.read_json(schemas.eval_json(art, "concepts_bach"))
    assert bach["split"] == "test" and bach["concepts"]["invasion"]["highest_class"] == "Invasive"
    assert bach["concepts"]["invasion"]["auroc_Invasive_vs_rest"]["estimate"] > 0.8
    assert bach["concepts"]["invasion"]["direction_consistent"] is True
    assert isinstance(bach["contradictions"], list)
    zs = io.read_json(schemas.eval_json(art, "zeroshot"))
    conch = zs["encoders"]["conch"]
    assert conch["status"] == "ok" and conch["balanced_accuracy"]["estimate"] > 0.5
    assert np.array(conch["confusion"]).sum() == conch["n_cases"]
    assert zs["encoders"]["musk"]["status"].startswith("skipped")


def test_contradictions_are_surfaced(tmp_path):
    f = Fake(tmp_path / "root", seed=3)
    built = f.build()
    f.u["tubules"], f.u["invasion"] = f.u["invasion"], f.u["tubules"]  # prompts swapped, tiles unchanged
    f.text_npz("conch", built["bank"])
    assert run(C, f)[0] == 0
    bach = io.read_json(schemas.eval_json(tmp_path / "root" / "artifacts", "concepts_bach"))
    kinds = {(c["concept"], c["kind"]) for c in bach["contradictions"]}
    assert ("tubules", "direction") in kinds and ("invasion", "highest_class") in kinds


def test_concept_vectors_combine_only_when_caption_consistent(fake):
    dim = 24
    w = np.zeros((9, dim), np.float32)
    w[fake.ids.index("invasion")] = fake.u["invasion"] * 5
    consistent = np.zeros(9, bool)
    consistent[fake.ids.index("invasion")] = True
    np.savez(C.concept_vectors_npz(fake.root), concept_ids=np.asarray(fake.ids), w=w, b=np.zeros(9, np.float32),
             caption_consistent=consistent, auroc=np.full(9, 0.9), encoder=np.asarray("conch"),
             synthetic=np.asarray(True))
    code, log = run(C, fake)
    assert code == 0, log
    tile = io.load_npz(schemas.concepts_tile_npz(fake.root))
    j = fake.ids.index("invasion")
    assert np.isfinite(tile["z_cav"][:, j]).all() and np.isnan(np.delete(tile["z_cav"], j, axis=1)).all()
    np.testing.assert_allclose(tile["z"][:, j], (tile["z_text"][:, j] + tile["z_cav"][:, j]) / 2, atol=1e-5)
    concepts = io.read_json(schemas.eval_json(fake.root / "artifacts", "concepts"))
    assert concepts["concepts"]["invasion"]["source"] == "cav" and concepts["concepts"]["tubules"]["source"] == "text"
    assert concepts["concepts"]["invasion"]["caption_consistent"] is True
    assert concepts["concepts"]["invasion"]["cav_auroc"] == pytest.approx(0.9)
    assert concepts["concepts"]["tubules"]["caption_consistent"] is False
    assert "cav" in concepts["nct"]["concepts"]["pleomorphism"]  # vectors are also scored on NCT when present
    feats = _features(fake)
    assert feats["invasion__cav_top3"].notna().all()


def test_rerun_is_skipped_until_inputs_change(fake):
    assert run(C, fake)[0] == 0
    code, log = run(C, fake)
    assert code == 0 and any("up to date" in line for line in log)
    np.savez(C.concept_vectors_npz(fake.root), concept_ids=np.asarray(fake.ids), w=np.zeros((9, 24), np.float32),
             b=np.zeros(9, np.float32), caption_consistent=np.zeros(9, bool), auroc=np.full(9, 0.5),
             encoder=np.asarray("conch"), synthetic=np.asarray(True))
    code, log = run(C, fake)
    assert code == 0 and not any("up to date" in line for line in log)


def test_limit_and_missing_inputs(fake, tmp_path):
    code, _ = run(C, fake, "--limit", "12")
    assert code == 0
    assert _features(fake)["source_case_id"].nunique() <= 12
    empty = Fake(tmp_path / "empty")
    assert run(C, empty)[0] == 2


def test_chain_02_embed_03_text_04_concepts_on_the_cpu_test_encoder(tmp_path, monkeypatch):
    """Real file formats end to end: 02_embed writes emb/test/*, 03_text writes text/test.npz, 04_concepts reads both."""
    from tests.embed_helpers import build_tree, make_encoders_yaml, run_embed

    monkeypatch.setenv("PATHOLOGY_TEST_ADAPTER_DELAY_S", "0")
    monkeypatch.setattr("pipeline.degrade.apply", lambda image, *a, **k: image)
    root = tmp_path / "root"
    f = Fake(root)
    cases = f.cases().head(4).copy()
    cases["case_id"] = ["BACH_0001", "BACH_0002", "BACH_0003", "BACH_0004"]
    cases["split"] = ["train", "train", "calibration", "test"]
    cases.to_csv(schemas.cases_csv(root), index=False)
    tiles = f.tiles(cases)
    tiles = tiles[tiles["tile_idx"] < 4]
    tiles.to_csv(schemas.tiles_csv(root), index=False)
    build_tree(root)
    ecfg = make_encoders_yaml(tmp_path)
    assert run_embed(root, ecfg, "--sets", "bach512,deg512,nct")[0] == 0
    T = importlib.import_module("pipeline.03_text")
    assert T.main(["--data-root", str(root), "--encoders", "test", "--device", "cpu", "--encoders-config", str(ecfg)],
                  log=lambda s: None) == 0
    code, log = run(C, f, "--encoder", "test")
    assert code == 0, log
    feats = _features(f)
    assert set(feats.loc[feats["variant"] == "clean", "case_id"]) == set(cases["case_id"])
    assert (feats["variant"] == "degraded").sum() == 2


def test_variant_ids_from_the_manifest_are_not_suffixed_twice():
    assert C.split_variant_id("BACH_0001_FSIM_blur3", "blur", 3) == ("BACH_0001", "BACH_0001_FSIM_blur3")
    assert C.split_variant_id("BACH_0001", "jpeg", 1) == ("BACH_0001", "BACH_0001_FSIM_jpeg1")


def test_npz_holds_only_the_tiles_each_case_was_aggregated_over(fake):
    assert run(C, fake)[0] == 0
    tile = io.load_npz(schemas.concepts_tile_npz(fake.root))
    clean = tile["variant"] == "clean"
    assert 11 not in set(tile["tile_idx"][clean].tolist())  # the QC-failing tile is absent
    assert "qc_pass" not in tile and set(tile["variant"].tolist()) == {"clean", "degraded", "composite"}


def _vectors(fake, encoder="conch", w=None):
    np.savez(C.concept_vectors_npz(fake.root), concept_ids=np.asarray(fake.ids),
             w=np.zeros((9, 24), np.float32) if w is None else w, b=np.zeros(9, np.float32),
             caption_consistent=np.ones(9, bool), auroc=np.full(9, 0.9), encoder=np.asarray(encoder),
             synthetic=np.asarray(True))


def test_vectors_from_another_encoder_are_refused(fake):
    _vectors(fake, encoder="pathgenclip")
    with pytest.raises(ValueError, match="pathgenclip"):
        run(C, fake)


def test_a_vector_file_without_fitted_vectors_is_text_only(fake):
    _vectors(fake, w=np.full((9, 1), np.nan, np.float32))  # 04_cav ran before any embedding existed
    assert run(C, fake)[0] == 0
    assert np.isnan(io.load_npz(schemas.concepts_tile_npz(fake.root))["z_cav"]).all()
