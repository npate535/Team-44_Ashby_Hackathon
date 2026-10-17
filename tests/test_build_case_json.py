"""pipeline/08_build_case_json.py on the downstream fixture: contract validity, tiers, variants, ground truth only in
eval_only, heroes, cells disabled, LLM off by default, idempotency, limits, missing inputs."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import yaml

from agent import lumen
from backend import summary as S
from core import config, contract, io, schemas, tiers
from tests import downstream_fixture as D

STEP = importlib.import_module("pipeline.08_build_case_json")
CFG = config.load_settings()


@pytest.fixture()
def env(tmp_path):
    root, art = tmp_path / "data", tmp_path / "art"
    info = D.write_downstream(root, CFG)
    bank = D.write_concept_bank(tmp_path / "concepts.yaml", CFG)
    cells = D.write_cell_features_yaml(tmp_path / "cell_features.yaml")
    return {"root": root, "art": art, "info": info, "bank": bank, "cells": cells, "tmp": tmp_path}


def run(env, cfg=CFG, **kw):
    ctx_kw = {k: kw.pop(k) for k in ("limit", "force") if k in kw}
    ctx = D.make_ctx(cfg, env["root"], env["art"], **ctx_kw)
    return STEP.build(ctx, concept_bank_path=env["bank"], cell_features_path=env["cells"], **kw)


def load_cases(art):
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((art / "cases").glob("*.json"))}


def test_every_case_is_written_and_validates(env):
    extra = run(env)
    cases = load_cases(env["art"])
    info = env["info"]
    assert set(cases) == set(info["test"]) | set(info["variants"]) | set(info["composites"])
    assert extra["by_kind"] == {"test": 8, "degraded": 8, "composite": 2} and extra["skipped"] == []
    for case in cases.values():
        model = contract.Case.model_validate(case)
        assert model.synthetic is True
        contract.to_public(case)
        for ref in [case["image"]["display"]] + [s["image"] for s in case["similar_cases"]]:
            assert (env["art"] / ref).is_file()
    meta = io.read_json(schemas.step_meta_json(env["root"], "08_build_case_json"))
    assert meta["synthetic"] is True and meta["llm_calls"] == 0 and meta["summary_paths"] == {"template": 18}


def test_flags_equal_core_tiers_and_lanes_are_consistent(env):
    run(env)
    for case in load_cases(env["art"]).values():
        assert case["flag"] == tiers.flag_for(case, CFG)
        if case["lane"] == "biopsy":
            assert case["margin"] is None and case["simulated"] is False
        else:
            assert case["margin"] is not None and case["flag"]["tier"] == 1 and case["simulated"] is True


def test_variants_carry_source_ids_and_degradation(env):
    run(env)
    cases = load_cases(env["art"])
    for vid in env["info"]["variants"]:
        case = cases[vid]
        source = vid.split("_FSIM_")[0]
        assert case["simulated_variant"] == "degraded" and case["source_case_ids"] == [source]
        assert case["degradation"]["severity"] == CFG["degrade"]["display_severity"]
        assert case["quality"]["reliability_flag"] and case["margin"]["suggested_call"] == "defer"
        assert case["margin"]["involved"] == [] and case["margin"]["threshold"] is not None   # ruling M8
        assert all(o["type"] != "stain" for o in case["agent"]["orders"])
        assert bool(case["agent"]["blocked"]) == source.startswith("BACH_0107")   # only carcinoma reads fire stains
    comp = cases[env["info"]["composites"][0]]
    assert comp["simulated_variant"] == "composite" and comp["source_case_ids"] == ["BACH_0101", "BACH_0107"]  # sorted
    assert comp["eval_only"]["margin_truth_superior"] == "tumor_side"
    assert comp["eval_only"]["carcinoma_case_id"] == "BACH_0107"
    for cid in env["info"]["test"]:
        assert cases[cid]["simulated_variant"] is None and cases[cid]["source_case_ids"] == []


def test_ground_truth_appears_only_under_eval_only(env):
    cfg = copy.deepcopy(CFG)
    cfg["demo"]["prefer_correct"] = False           # hero picking may use labels; switch it off for this check
    run(env, cfg)
    first = load_cases(env["art"])
    cases_csv = schemas.cases_csv(env["root"])
    df = pd.read_csv(cases_csv)
    rotate = {"Normal": "Benign", "Benign": "InSitu", "InSitu": "Invasive", "Invasive": "Normal"}
    df.loc[df["split"] == "test", "label"] = df.loc[df["split"] == "test", "label"].map(rotate)
    io.write_csv(cases_csv, df)
    run(env, cfg, force=True)
    second = load_cases(env["art"])
    for cid, case in first.items():
        a, b = dict(case), dict(second[cid])
        assert a.pop("eval_only")["true_label"] != b.pop("eval_only")["true_label"]
        assert a == b, f"{cid}: something outside eval_only depends on the true label"
        text = json.dumps(a)
        assert "true_label" not in text and "eval_only" not in text


def test_hero_cases_are_picked_and_placed_late(env):
    extra = run(env)
    cases = load_cases(env["art"])
    heroes = extra["heroes"]
    assert heroes["biopsy"] == {"case_id": D.HERO_BIOPSY, "exact": True, "matched": 4, "of": 4}
    assert heroes["surgical"]["case_id"] == "BACH_0107" and heroes["surgical"]["exact"]
    assert heroes["surgical_degraded"]["case_id"].startswith("BACH_0107_FSIM_")
    assert heroes["surgical_composite"]["exact"]
    clean = {c: v["arrived_offset_s"] for c, v in cases.items() if v["simulated_variant"] is None}
    assert max(clean, key=clean.get) == D.HERO_BIOPSY
    others = [t for c, t in clean.items() if c not in (D.HERO_BIOPSY, "BACH_0107")]
    assert clean["BACH_0107"] > max(others)
    variant_times = [v["arrived_offset_s"] for v in cases.values() if v["simulated_variant"] is not None]
    assert min(variant_times) > max(clean.values())
    hero = cases[D.HERO_BIOPSY]
    assert hero["agent"]["status"] == "drafted" and hero["prediction"]["conformal_set"] == ["InSitu", "Invasive"]


def test_arrivals_are_seeded_and_deterministic(env):
    run(env)
    a = {c: v["arrived_offset_s"] for c, v in load_cases(env["art"]).items()}
    run(env, force=True)
    b = {c: v["arrived_offset_s"] for c, v in load_cases(env["art"]).items()}
    assert a == b and len(set(a.values())) == len(a)


def test_findings_and_summary_citations_are_consistent(env):
    run(env)
    for case in load_cases(env["art"]).values():
        ids = [f["id"] for f in case["findings"]]
        assert ids == [f"F{i + 1}" for i in range(len(ids))]
        assert case["findings"][0]["kind"] == "prediction"
        kept, dropped = S.validate(case["summary"]["sentences"], case["findings"], CFG)
        assert dropped == [] and kept == case["summary"]["sentences"]
        kinds = [f["kind"] for f in case["findings"]]
        assert ("margin" in kinds) == (case["lane"] == "surgical")
        assert kinds.count("concept") <= CFG["findings"]["max_concepts"]


def test_cells_block_and_cell_findings(env):
    run(env)
    cases = load_cases(env["art"])
    hero = cases[D.HERO_BIOPSY]
    assert hero["cells"]["status"] == "ok" and [t["idx"] for t in hero["cells"]["tiles"]] == [5, 6]
    assert hero["cells"]["features"]["nuc_area_cv"] == {"value": 0.48, "unit": "", "ref_percentile": 0.95,
                                                        "contribution": 0.21}
    cell_findings = [f for f in hero["findings"] if f["kind"] == "cells"]
    assert {f["data"]["feature"] for f in cell_findings} == {"nuc_area_cv", "neo_density"}   # extreme percentiles
    assert any(c["source"] == "cell" and c["id"] == "nuc_area_cv" for c in hero["concepts"])
    assert cases[D.INSUFFICIENT]["cells"] == {"status": "failed", "model": None, "mpp_model": None, "tiles": [],
                                              "features": {}, "flags": {}}
    assert (env["art"] / "cells" / f"{D.HERO_BIOPSY}_5_overlay.png").is_file()


def test_cells_disabled_path(env):
    cfg = copy.deepcopy(CFG)
    cfg["cells"]["enabled"] = False
    run(env, cfg)
    for case in load_cases(env["art"]).values():
        assert case["cells"]["status"] == "disabled"
        assert all(c["source"] != "cell" for c in case["concepts"])
        assert all(f["kind"] != "cells" for f in case["findings"])


def test_rerun_is_idempotent_and_force_rewrites(env):
    run(env)
    again = run(env)
    assert again["written"] == 0 and again["unchanged"] == again["n_cases"]
    forced = run(env, force=True)
    assert forced["written"] == forced["n_cases"]


def test_limit_restricts_clean_cases_and_their_variants(env):
    extra = run(env, limit=7)          # BACH_0101..0107 (sorted); BACH_0108 and its composite are left out
    cases = load_cases(env["art"])
    assert extra["by_kind"]["test"] == 7 and D.HERO_BIOPSY not in cases
    assert "BACH_0108_COMP_1" not in cases and "BACH_0107_COMP_0" in cases


def test_clean_removes_stale_case_files(env):
    run(env)
    stale = env["art"] / "cases" / "OLD_0001.json"
    stale.write_text("{}", encoding="utf-8")
    extra = run(env, clean=True)
    assert not stale.exists() and extra["removed"] == ["OLD_0001.json"]


def test_missing_required_input_is_an_error(env):
    schemas.reliability_csv(env["root"]).unlink()
    with pytest.raises(STEP.InputError, match="reliability.csv"):
        run(env)


def test_clean_case_missing_from_predictions_is_an_error_but_variants_are_skipped(env):
    path = schemas.predictions_csv(env["root"])
    df = pd.read_csv(path)
    io.write_csv(path, df[df["case_id"] != "BACH_0101_FSIM_blur3"])
    extra = run(env)
    assert [s["case_id"] for s in extra["skipped"]] == ["BACH_0101_FSIM_blur3"]
    io.write_csv(path, df[df["case_id"] != "BACH_0102"])
    with pytest.raises(STEP.InputError, match="BACH_0102"):
        run(env, force=True)


def test_reference_index_and_spotcheck_items(env):
    run(env)
    ref = io.read_json(schemas.reference_index_json(env["art"]))
    assert ref["synthetic"] is True and set(ref["cases"]) <= set(env["info"]["train"])
    items = io.read_json(schemas.spotcheck_items_json(env["art"]))
    assert schemas.SPOTCHECK_ITEMS.problems(items) == []
    concept_items = [i for i in items["items"] if i["kind"] == "concept"]
    per = {}
    for i in concept_items:
        per.setdefault(i["target"], []).append(i)
        assert (env["art"] / i["image"]).is_file() and i["case_id"] in env["info"]["test"]
    assert set(per) == set(CFG["concepts"]["ids"])
    assert all(len(v) == CFG["spotcheck"]["per_concept"] for v in per.values())
    for target, its in per.items():
        scores = [i["score"] for i in its]
        assert scores == sorted(scores, reverse=True)
    cell_items = [i for i in items["items"] if i["kind"] == "cells"]
    assert cell_items and all((env["art"] / i["image"]).is_file() for i in cell_items)
    text = json.dumps(items)
    assert "true_label" not in text and not any(i["label"] in CFG["classes"] for i in items["items"])


class _Server:
    def __init__(self):
        self.calls = 0

    def __call__(self, request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "ornith-1.0-35b"}]})
        self.calls += 1
        body = json.loads(request.content)
        findings = json.loads(body["messages"][1]["content"])["findings"]
        f1 = findings[0]
        sentences = [{"text": f"Suggested category with calibrated probability {f1['data']['p']:.2f}.", "findings": ["F1"]},
                     {"text": "Image quality was reviewed by the reliability check.",
                      "findings": [f["id"] for f in findings if f["kind"] == "quality"]}]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"sentences": sentences})},
                                                      "finish_reason": "stop"}], "usage": {}})


def test_llm_summaries_with_budget_heroes_first(env):
    server = _Server()
    client = lumen.LumenClient(base_url="https://lumen.example/v1", api_key="k", model_preference=["ornith-1.0-35b"],
                               cache_dir=env["tmp"] / "llm", max_calls=2, retries=0, timeout_s=5, temperature=0.2,
                               max_tokens=100, transport=httpx.MockTransport(server), sleep=lambda s: None)
    extra = run(env, client=client, max_llm_calls=2)
    cases = load_cases(env["art"])
    llm = sorted(c for c, v in cases.items() if v["summary"]["generator"].startswith("lumen:"))
    assert len(llm) == 2 and server.calls == 2 and extra["llm_calls"] == 2
    hero_ids = {h["case_id"] for h in extra["heroes"].values()}
    assert set(llm) <= hero_ids
    assert extra["summary_paths"] == {"llm": 2, "template": 16}


def test_cli_main_runs_on_a_config_copy(env, tmp_path, capsys):
    settings = yaml.safe_load(config.SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["concepts"]["file"] = str(env["bank"])
    settings["cells"]["enabled"] = False
    cfg_path = tmp_path / "settings.yaml"
    cfg_path.write_text(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), encoding="utf-8")
    code = STEP.main(["--config", str(cfg_path), "--data-root", str(env["root"]), "--artifacts-root",
                      str(env["art"]), "--limit", "5"])
    assert code == 0 and "08_build_case_json: " in capsys.readouterr().out
    assert len(load_cases(env["art"])) >= 5
    schemas.retrieval_json(env["root"]).unlink()
    assert STEP.main(["--config", str(cfg_path), "--data-root", str(env["root"]), "--artifacts-root",
                      str(env["art"])]) == 1


def _edit_csv(path, fn):
    df = pd.read_csv(path)
    io.write_csv(path, fn(df))


@pytest.mark.parametrize("path_fn,column,case_id,match", [
    (schemas.reliability_csv, "reliability_flag", "BACH_0102", "reliability_flag is missing"),
    (schemas.reliability_csv, "reliability", "BACH_0102", "reliability is missing"),
    (schemas.contributions_csv, "contribution", "BACH_0102", "missing contribution"),
    (schemas.tile_risk_csv, "risk", "BACH_0102", "risk is missing"),
    (schemas.predictions_csv, "risk", "BACH_0102", "risk is missing"),
])
def test_blank_required_values_are_errors_never_false_or_zero(env, path_fn, column, case_id, match):
    def blank(df):
        df[column] = df[column].astype(object)
        df.loc[df["case_id"] == case_id, column] = ""
        return df

    _edit_csv(path_fn(env["root"]), blank)
    with pytest.raises(STEP.InputError, match=match):
        run(env)


def test_blank_values_in_a_variant_skip_it(env):
    def blank(df):
        df["reliability_flag"] = df["reliability_flag"].astype(object)
        df.loc[df["case_id"] == "BACH_0107_FSIM_jpeg3", "reliability_flag"] = ""
        return df

    _edit_csv(schemas.reliability_csv(env["root"]), blank)
    extra = run(env)
    assert [s["case_id"] for s in extra["skipped"]] == ["BACH_0107_FSIM_jpeg3"]


def test_retrieval_references_must_be_train_cases(env):
    path = schemas.retrieval_json(env["root"])
    data = io.read_json(path)
    data["cases"]["BACH_0102"]["neighbors"][0]["case_id"] = "BACH_0105"        # a test case
    io.write_json(path, data)
    with pytest.raises(STEP.InputError, match="not a train-split case"):
        run(env)


def test_clean_case_missing_from_retrieval_is_an_error(env):
    path = schemas.retrieval_json(env["root"])
    data = io.read_json(path)
    del data["cases"]["BACH_0102"]
    io.write_json(path, data)
    with pytest.raises(STEP.InputError, match="retrieval.json"):
        run(env)


def test_inconsistent_prediction_rows_are_errors(env):
    def bad_size(df):
        df.loc[df["case_id"] == "BACH_0102", "set_size"] = 3
        return df

    _edit_csv(schemas.predictions_csv(env["root"]), bad_size)
    with pytest.raises(STEP.InputError, match="set_size"):
        run(env)


def test_margin_geometry_must_match_settings(env):
    path = schemas.margins_json(env["root"])
    data = io.read_json(path)
    data["strip_px"] = 128
    io.write_json(path, data)
    with pytest.raises(STEP.InputError, match="strip_px"):
        run(env)
