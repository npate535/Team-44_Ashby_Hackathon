"""backend.dev_cases: synthetic dev/e2e cases are contract-valid, cover the demo path and never touch fixtures/."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from backend import dev_cases
from core import contract, schemas, tiers
from tests.app_helpers import CFG, FIXTURES, dev_roles, dev_root


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return dev_root(tmp_path_factory.mktemp("dev"), fill=3)


def _cases(root):
    return [json.loads(p.read_text()) for p in sorted((root / "cases").glob("*.json"))]


def test_every_case_validates_and_is_synthetic(root):
    cases = _cases(root)
    assert len(cases) == len(dev_cases.BIOPSY_SCENARIOS) + 3 + 1 + 4
    for case in cases:
        model = contract.Case.model_validate(case)
        assert model.synthetic is True and case["eval_only"]["split"] == "test"
        assert case["flag"] == tiers.flag_for(case, CFG)
        assert "eval_only" not in contract.to_public(case)


def test_demo_path_is_covered(root):
    roles = dev_roles(root)
    cases = {c["case_id"]: c for c in _cases(root)}
    hero = cases[roles["hero_biopsy"]]
    assert hero["flag"]["tier"] == 2 and hero["prediction"]["conformal_set"] == ["InSitu", "Invasive"]
    items = [i for o in hero["agent"]["orders"] for i in o["items"]]
    assert items == ["p63", "calponin", "ER", "PR", "HER2", "Ki-67"]
    assert hero["cells"]["status"] == "ok"
    biopsies = [c for c in cases.values() if c["lane"] == "biopsy"]
    assert max(biopsies, key=lambda c: c["arrived_offset_s"])["case_id"] == hero["case_id"]   # arrives last
    assert {c["flag"]["tier"] for c in biopsies} == {2, 3, 4}
    surgical = cases[roles["hero_surgical"]]
    assert surgical["margin"]["suggested_call"] == "tumor_at_margin" and surgical["margin"]["involved"] == ["superior"]
    degraded = cases[roles["surgical_degraded"]]
    assert degraded["simulated_variant"] == "degraded" and degraded["quality"]["reliability_flag"]
    assert degraded["margin"]["suggested_call"] == "defer"
    composite = cases[roles["surgical_composite"]]
    assert composite["simulated_variant"] == "composite" and composite["margin"]["suggested_call"] == "tumor_at_margin"
    assert cases[roles["surgical_clear"]]["margin"]["suggested_call"] == "clear"
    assert all(c["agent"]["blocked"] for c in cases.values() if c["lane"] == "surgical")


def test_media_exist_and_are_placeholders(root):
    for case in _cases(root):
        refs = [case["image"]["display"]] + [s["image"] for s in case["similar_cases"]]
        refs += [t["overlay"] for t in (case.get("cells") or {}).get("tiles", [])]
        for ref in refs:
            assert (root / ref).is_file(), ref
    with Image.open(root / "images" / "DEV_0001.jpg") as img:
        assert img.size == (dev_cases.DISPLAY_W, 768)


def test_spotcheck_items_follow_the_schema(root):
    data = json.loads(schemas.spotcheck_items_json(root).read_text())
    assert set(schemas.SPOTCHECK_ITEMS.required_keys) <= set(data) and data["synthetic"] is True
    kinds = {i["kind"] for i in data["items"]}
    assert kinds == {"concept", "cells"}
    for item in data["items"]:
        assert (root / item["image"]).is_file()
        assert "label" in item and item["label"] not in CFG["classes"]


def test_cli_refuses_fixtures_and_skips_existing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        dev_cases.main(["--out", str(FIXTURES)])
    out = tmp_path / "dev"
    assert dev_cases.main(["--out", str(out), "--fill", "0", "--mean-interarrival-s", "5"]) == 0
    assert (out / "meta" / "dev_cases.json").is_file()
    n = len(list((out / "cases").glob("*.json")))
    assert dev_cases.main(["--out", str(out)]) == 0 and "already holds" in capsys.readouterr().out
    assert dev_cases.main(["--out", str(out), "--force", "--fill", "1"]) == 0
    assert len(list((out / "cases").glob("*.json"))) == n + 1
    meta = json.loads((out / "meta" / "dev_cases.json").read_text())
    assert meta["synthetic"] is True


def test_with_eval_writes_synthetic_eval_files(tmp_path):
    out = tmp_path / "dev"
    dev_cases.build_dev_cases(CFG, out, fill=0, mean_interarrival_s=5, seed=44, fixtures=FIXTURES, with_eval=True)
    names = {p.stem for p in (out / "eval").glob("*.json")}
    assert {"metrics", "calibration", "conformal", "concepts", "reliability", "retrieval", "queue_sim"} <= names
    assert names <= set(schemas.EVAL_FILES)
    for p in (out / "eval").glob("*.json"):
        data = json.loads(p.read_text())
        assert data["synthetic"] is True and data["name"] == p.stem
