"""04_cav: grouped held-out AUROC (no group on both sides), caption-consistency threshold, raw-space vectors,
text-only fallback, WSI region check, and the concepts.json section merge with 04_concepts."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from core import io, schemas
from tests.concept_fakes import D, Fake, run

CAV = importlib.import_module("pipeline.04_cav")
C = importlib.import_module("pipeline.04_concepts")


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setenv("PATHQUEUE_SITE", "local")


def test_grouped_split_never_shares_a_group():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 5))
    y = np.tile([0, 1], 60)
    groups = np.repeat(np.arange(12), 10)
    oof, splits = CAV.grouped_oof(x, y, groups, folds=5, c=1.0, seed=0)
    assert len(splits) == 5 and np.isfinite(oof).all()
    for tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])


def test_vector_is_folded_back_into_raw_space():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 4)) * [1, 10, 0.1, 3] + [5, -2, 0, 1]
    y = (x[:, 0] - 5 + 0.3 * rng.standard_normal(200) > 0).astype(int)
    w, b = CAV.fit_vector(x, y, c=1.0, seed=0)
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0), x.std(0)
    ref = LogisticRegression(C=1.0, max_iter=5000, random_state=0).fit((x - mu) / sd, y).decision_function((x - mu) / sd)
    np.testing.assert_allclose(x @ w + b, ref, atol=1e-4)


def test_separable_concepts_are_caption_consistent_and_used_by_04_concepts(tmp_path):
    f = Fake(tmp_path / "root")
    f.build(quilt=True)
    code, log = run(CAV, f)
    assert code == 0, log
    vec = io.load_npz(C.concept_vectors_npz(f.root))
    assert vec["concept_ids"].tolist() == f.ids and vec["w"].shape == (9, D)
    assert vec["caption_consistent"].all() and (vec["auroc"] > 0.75).all()
    concepts = io.read_json(schemas.eval_json(f.root / "artifacts", "concepts"))
    cav = concepts["cav"]
    assert "caption-consistent" in cav["label"] and "validat" not in cav["label"].replace("not validation", "")
    assert cav["sets_used"] == ["quilt"] and cav["concepts"]["invasion"]["heldout_auroc"]["estimate"] > 0.75
    assert run(C, f)[0] == 0
    concepts = io.read_json(schemas.eval_json(f.root / "artifacts", "concepts"))
    assert "cav" in concepts and "nct" in concepts  # both steps keep each other's sections
    assert concepts["concepts"]["invasion"]["source"] == "cav"
    tile = io.load_npz(schemas.concepts_tile_npz(f.root))
    assert np.isfinite(tile["z_cav"]).all()


def test_unseparable_captions_stay_text_only(tmp_path):
    f = Fake(tmp_path / "root")
    f.build()
    f.quilt(separable=False)
    assert run(CAV, f)[0] == 0
    vec = io.load_npz(C.concept_vectors_npz(f.root))
    assert not vec["caption_consistent"].any() and np.isfinite(vec["w"]).all()
    assert run(C, f)[0] == 0
    assert np.isnan(io.load_npz(schemas.concepts_tile_npz(f.root))["z_cav"]).all()


def test_missing_mined_sets_fall_back_to_text_only_and_say_so(tmp_path):
    f = Fake(tmp_path / "root")
    f.build()
    assert run(CAV, f)[0] == 0
    vec = io.load_npz(C.concept_vectors_npz(f.root))
    assert not vec["caption_consistent"].any() and np.isnan(vec["w"]).all()
    report = io.read_json(schemas.eval_json(f.root / "artifacts", "concepts"))["cav"]
    assert report["sets_used"] == [] and "no mined" in report["concepts"]["tubules"]["status"]


def test_too_few_images_or_groups_are_text_only(tmp_path):
    f = Fake(tmp_path / "root")
    f.build()
    f.quilt(n=10)
    assert run(CAV, f)[0] == 0
    report = io.read_json(schemas.eval_json(f.root / "artifacts", "concepts"))["cav"]["concepts"]
    assert "fewer than 20" in report["invasion"]["status"]
    f.quilt(n=40, n_groups=2)
    assert run(CAV, f, "--force")[0] == 0
    report = io.read_json(schemas.eval_json(f.root / "artifacts", "concepts"))["cav"]["concepts"]
    assert "distinct videos" in report["invasion"]["status"]


def test_wsi_region_check_for_invasion_and_duct_confined(tmp_path):
    f = Fake(tmp_path / "root")
    f.build(quilt=True)
    rows, emb = [], []
    for i in range(30):
        for cls in ("Invasive", "InSitu"):
            rows.append({"item_id": f"wsi512:W{i}{cls}", "set": "wsi512", "case_id": f"WSI-{cls}-{i}", "label": cls,
                         "group": f"A0{i % 5}", "grid": "g512", "tile_idx": 0})
            emb.append(f.tile_vec(cls))
    f.write_index("conch", "wsi512", rows, np.asarray(emb))
    assert run(CAV, f)[0] == 0
    report = io.read_json(schemas.eval_json(f.root / "artifacts", "concepts"))["cav"]["concepts"]
    inv = report["invasion"]["wsi_check"]
    assert inv["status"] == "ok" and inv["n_groups"] == 5 and inv["cav"]["estimate"] > 0.7
    assert report["duct_confined"]["wsi_check"]["text"]["estimate"] > 0.7
    assert "wsi_check" not in report["tubules"]


def test_rerun_skips_when_inputs_unchanged(tmp_path):
    f = Fake(tmp_path / "root")
    f.build(quilt=True)
    assert run(CAV, f)[0] == 0
    code, log = run(CAV, f)
    assert code == 0 and any("up to date" in line for line in log)
