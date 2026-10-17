"""03_text: every expand_prompts prompt + NCT prompt -> text/<model>.npz (TEXT_NPZ), checks, idempotence."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import yaml

from core import config, io, schemas
from tests.embed_helpers import build_tree, make_encoders_yaml, run_embed

text = importlib.import_module("pipeline.03_text")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PATHOLOGY_TEST_ADAPTER_DELAY_S", "0")
    monkeypatch.setenv("PATHQUEUE_SITE", "local")
    monkeypatch.setattr("pipeline.degrade.apply", lambda image, *a, **k: image)


def run_text(root, ecfg, *extra):
    lines = []
    code = text.main(["--data-root", str(root), "--encoders", "test", "--device", "cpu", "--encoders-config",
                      str(ecfg), *extra], log=lines.append)
    return code, lines


def test_npz_has_every_prompt_in_expand_order_plus_nct(tmp_path):
    ecfg = make_encoders_yaml(tmp_path)
    code, log = run_text(tmp_path, ecfg)
    assert code == 0, log
    data = io.load_npz(schemas.text_npz(tmp_path, "test"))
    assert schemas.TEXT_NPZ.problems(data) == []
    bank = config.load_concept_bank()
    prompts = schemas.expand_prompts(bank)
    nct = text.embed.nct_prompts()
    assert data["ids"].tolist() == [p.id for p in prompts] + [schemas.nct_prompt_id(c) for c in nct]
    assert data["texts"].tolist() == [p.text for p in prompts] + list(nct.values())
    assert data["emb"].dtype == np.float32 and data["emb"].shape == (len(prompts) + len(nct), 3)
    np.testing.assert_allclose(np.linalg.norm(data["emb"], axis=1), 1.0, atol=1e-5)
    assert str(data["model"]) == "test" and not bool(data["synthetic"])
    lookup = dict(zip(data["ids"].tolist(), data["emb"]))
    assert lookup["concept:invasion:pos:0:0"].shape == (3,)
    assert schemas.STEP_META.problems(io.read_json(schemas.step_meta_json(tmp_path, "03_text"))) == []


def test_up_to_date_npz_is_skipped_and_prompt_edits_reencode(tmp_path, monkeypatch):
    ecfg = make_encoders_yaml(tmp_path)
    assert run_text(tmp_path, ecfg)[0] == 0
    code, log = run_text(tmp_path, ecfg)
    assert code == 0 and any("up to date" in line for line in log)
    data = yaml.safe_load(config.resolve_repo_path("config/concepts.yaml").read_text())
    data["concepts"][3]["positives"][0] = "invasive carcinoma cells infiltrating fibrous stroma"
    edited = tmp_path / "concepts.yaml"
    edited.write_text(yaml.safe_dump(data))
    code, log = run_text(tmp_path, ecfg, "--concepts", str(edited))
    assert code == 0 and any("encoded" in line for line in log)
    texts = io.load_npz(schemas.text_npz(tmp_path, "test"))["texts"].tolist()
    assert "invasive carcinoma cells infiltrating fibrous stroma." in texts


def test_non_text_aligned_encoder_is_refused_before_loading(tmp_path):
    lines = []
    code = text.main(["--data-root", str(tmp_path), "--encoders", "uni2h", "--device", "cpu"], log=lines.append)
    assert code == 2 and "not text-aligned" in lines[-1]


def test_unnormalized_text_embeddings_abort(tmp_path, monkeypatch):
    from pipeline import encoders as E

    real = E._test_text_fn
    monkeypatch.setattr(E, "_test_text_fn", lambda dim: (lambda texts: 2.0 * real(dim)(texts)))
    code, log = run_text(tmp_path, make_encoders_yaml(tmp_path))
    assert code == 2 and any("not L2-normalized" in line for line in log)
    assert not schemas.text_npz(tmp_path, "test").exists()


def test_same_space_check_uses_stored_nct_image_embeddings(tmp_path):
    build_tree(tmp_path)
    ecfg = make_encoders_yaml(tmp_path, nct_min_items=1, nct_floor=0.0)
    assert run_embed(tmp_path, ecfg, "--sets", "nct")[0] == 0
    code, log = run_text(tmp_path, ecfg)
    assert code == 0
    meta = io.read_json(schemas.step_meta_json(tmp_path, "03_text"))
    assert meta["report"]["test"]["nct_same_space"]["status"] == "pass"
    ecfg_fail = make_encoders_yaml(tmp_path, nct_min_items=1, nct_floor=1.01)
    code, log = run_text(tmp_path, ecfg_fail, "--force")
    assert code == 2 and any("text and image spaces disagree" in line for line in log)


def test_prompt_ids_never_collide_with_nct_ids():
    ids, texts = text.prompt_table(config.load_concept_bank())
    assert len(ids) == len(set(ids)) == len(texts)
    assert sum(i.startswith("nct:") for i in ids) == 9
