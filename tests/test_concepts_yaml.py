"""config/concepts.yaml: R1 schema validation, negation lint, counts, uniqueness, prompt expansion."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from core import config, schemas

REPO = Path(__file__).resolve().parents[1]
BANK_PATH = REPO / "config" / "concepts.yaml"
# CONCH cannot parse negation (BUILD_PLAN §A11): contrasts describe the contrasting tissue affirmatively.
NEGATION = re.compile(r"\b(no|not|non|without|absence|absent|lack|lacks|lacking|free of|negative|devoid|none|nor|"
                      r"never|neither)\b|n't\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def cfg():
    return config.load_settings()


@pytest.fixture(scope="module")
def bank(cfg):
    return config.load_concept_bank(cfg=cfg)


def _all_phrasings(bank):
    for c in bank.concepts:
        yield from ((f"{c.id}.pos", p) for p in c.positives)
        yield from ((f"{c.id}.con", p) for p in c.contrasts)
    for cls, phrases in bank.class_prompts.items():
        yield from ((f"class.{cls}", p) for p in phrases)


def test_bank_validates_against_the_r1_schema_and_settings(bank, cfg):
    assert bank.ids == cfg["concepts"]["ids"]
    assert list(bank.class_prompts) == cfg["classes"]


def test_header_marks_the_bank_as_a_draft_for_pathologist_review():
    head = BANK_PATH.read_text().lower()[:600]
    assert "draft" in head and "requires pathologist review" in head and "not for clinical use" in head


@pytest.mark.parametrize("text", ["no tumor", "tissue without nuclei", "absence of necrosis", "not invasive",
                                  "non-neoplastic", "doesn't show", "free of tumor", "lacking tubules"])
def test_negation_lint_catches_negations(text):
    assert NEGATION.search(text)


@pytest.mark.parametrize("text", ["nottingham grade", "notched nuclei", "knotted stroma",
                                  "nodular lymphoid aggregate", "annotated region"])
def test_negation_lint_ignores_words_that_merely_contain_negations(text):
    assert not NEGATION.search(text)


def test_every_phrasing_and_template_is_affirmative(bank):
    bad = [(where, p) for where, p in _all_phrasings(bank) if NEGATION.search(p)]
    bad += [("template", t) for t in bank.templates if NEGATION.search(t)]
    assert bad == []


def test_counts_follow_plan_6_4(bank, cfg):
    lo_p, hi_p = cfg["concepts"]["positives"]
    lo_c, hi_c = cfg["concepts"]["contrasts"]
    for c in bank.concepts:
        assert lo_p <= len(c.positives) <= hi_p and lo_c <= len(c.contrasts) <= hi_c, c.id
    assert 6 <= len(bank.templates) <= 22
    assert all(len(p) >= 1 for p in bank.class_prompts.values())


def test_weak_flag_only_on_mitoses_and_expected_directions(bank):
    by = bank.by_id()
    assert [c.id for c in bank.concepts if c.weak] == ["mitoses"]
    assert by["invasion"].direction == "up" and by["duct_confined"].direction == "down"
    assert by["tubules"].direction == "down" and by["lymphocytes"].direction == "context"


def test_phrasings_are_unique_across_the_whole_bank(bank):
    phrases = [p.lower() for _, p in _all_phrasings(bank)]
    assert len(phrases) == len(set(phrases))
    templates = [t.lower() for t in bank.templates]
    assert len(templates) == len(set(templates))


def test_templates_have_exactly_one_placeholder_and_fit_the_text_contexts(bank):
    for t in bank.templates:
        assert t.count("{}") == 1
    longest = max((t.replace("{}", p) for t in bank.templates for _, p in _all_phrasings(bank)), key=len)
    assert len(longest.split()) < 60  # PathGen-CLIP 77 BPE tokens, MUSK 100, CONCH 127


def test_labels_use_associated_feature_wording(bank):
    for c in bank.concepts:
        assert c.label and c.label[0].isupper()
        assert not re.search(r"\b(grade|diagnos|cause)", c.label, re.IGNORECASE)


def test_expand_prompts_ids_are_unique_and_cover_every_concept_and_class(bank):
    prompts = schemas.expand_prompts(bank)
    ids = [p.id for p in prompts]
    assert len(ids) == len(set(ids))
    n_t = len(bank.templates)
    want = n_t * (sum(len(c.positives) + len(c.contrasts) for c in bank.concepts)
                  + sum(len(v) for v in bank.class_prompts.values()))
    assert len(prompts) == want
    assert {p.target for p in prompts if p.kind == "concept"} == set(bank.ids)
    assert prompts[0].id == "concept:tubules:pos:0:0" and prompts[0].text == bank.concepts[0].positives[0] + "."


def test_unknown_keys_are_rejected(cfg, tmp_path):
    data = yaml.safe_load(BANK_PATH.read_text())
    data["concepts"][0]["negatives"] = ["x"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unexpected keys"):
        config.load_concept_bank(path, cfg)
