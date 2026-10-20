"""The Model evaluation view (team44-triage-frontend/app/js/views/metrics.js) rendered in node from eval JSONs built on
the synthetic eval fixture: every section renders a chart or an explicit 'not available' state, no junk tokens leak
into the page, and missing or malformed files never throw."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import textwrap

import pytest

from backend.app import create_app
from core import config, io, schemas
from eval import analytics as A
from eval import conformal_resplit as CR
from eval import metrics as M
from eval import queue_sim as Q
from tests import downstream_fixture as D
from tests.app_helpers import CFG as APP_CFG, FIXTURES

APP = config.REPO_ROOT / "team44-triage-frontend" / "app"
CFG = copy.deepcopy(config.load_settings())
CFG["bootstrap"]["n"] = 100

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture(scope="module")
def payloads(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("view")
    D.write_eval_root(tmp / "data", CFG)
    M.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_metrics"))
    Q.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_queue_sim"), reps=10)
    A.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_analytics"), reps=10)
    CR.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_conformal_resplit"), n_splits=40)
    ev = {p.stem: io.read_json(p) for p in sorted((tmp / "art" / "eval").glob("*.json"))}
    ev["spotcheck"] = {"name": "spotcheck", "reviewer_kind": "ai", "reviews": [
        {"item_id": "concept:invasion:X:1", "reviewer": "AI reviewer", "reviewer_kind": "ai", "verdict": "acceptable"},
        {"item_id": "concept:invasion:X:2", "reviewer": "AI reviewer", "reviewer_kind": "ai", "verdict": "unsure"},
        {"item_id": "concept:mitoses:X:1", "reviewer": "AI reviewer", "reviewer_kind": "ai", "verdict": "not_acceptable"},
        {"item_id": "cells:overlay:X:0", "reviewer": "AI reviewer", "reviewer_kind": "ai", "verdict": "acceptable"}]}
    app = create_app(cfg=APP_CFG, cases_dir=FIXTURES, artifacts_root=tmp / "x", db_path=tmp / "x.sqlite",
                     frontend_dir=None, env={})
    cfg = app.state.backend.config_payload()
    full = {"eval": ev, "figures": [], "errors": [], "expected": dict(schemas.EVAL_FILES), "synthetic": True}
    broken = copy.deepcopy(full)
    broken["eval"]["metrics"]["production"] = None
    broken["eval"]["conformal"]["methods"] = {}
    broken["eval"]["analytics"]["roc"] = None
    broken["eval"]["calibration"]["after"] = {"ece": "x"}
    return {"full": full, "empty": {**full, "eval": {}}, "broken": broken,
            "nulls": {**full, "eval": {k: None for k in ev}}}, cfg, tmp


def _render(payload, cfg, tmp):
    (tmp / "payload.json").write_text(json.dumps(payload))
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    script = textwrap.dedent(f"""
        import {{readFileSync}} from 'node:fs';
        import {{render, spotVerdicts}} from {json.dumps((APP / 'js/views/metrics.js').as_uri())};
        const metrics = JSON.parse(readFileSync({json.dumps(str(tmp / 'payload.json'))}, 'utf8'));
        const cfg = JSON.parse(readFileSync({json.dumps(str(tmp / 'cfg.json'))}, 'utf8'));
        const html = render({{metrics, cfg}}, {{assetUrl: (p) => p}});
        const loading = render({{metrics: null, cfg}}, {{assetUrl: (p) => p}});
        console.log(JSON.stringify({{html, loading, spot: spotVerdicts(metrics.eval && metrics.eval.spotcheck)}}));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_full_payload_renders_every_section(payloads):
    p, cfg, tmp = payloads
    out = _render(p["full"], cfg, tmp)
    html = out["html"]
    for section in ("summary", "classification", "confidence", "explanations", "reliability", "workflow", "cells",
                    "retrieval", "limitations"):
        assert f'id="mx-{section}"' in html
    # the e2e contract: each of these is either a chart or an explicit missing state
    for tid, key in (("metrics-calibration", "calibration"), ("metrics-queue-sim", "queue_sim"),
                     ("metrics-reliability", "reliability"), ("metrics-concepts", "concepts")):
        assert f'data-testid="{tid}"' in html or f'data-testid="metrics-missing-{key}"' in html
    assert 'data-testid="metrics-calibration"' in html and 'data-testid="metrics-queue-sim"' in html
    assert 'data-testid="metrics-synthetic"' in html and 'data-testid="limitations"' in html
    assert html.count('<svg class="chart cx') >= 10 and 'role="img"' in html
    assert "What this shows" in html and "Source numbers" in html
    for junk in ("NaN", "undefined", "[object Object]", "Infinity"):
        assert junk not in html, junk
    assert "true_label" not in html and "eval_only" not in html
    assert "Loading" in out["loading"]


def test_spot_verdicts_are_counted_per_concept(payloads):
    p, cfg, tmp = payloads
    spot = _render(p["full"], cfg, tmp)["spot"]
    assert spot["concepts"]["invasion"] == {"n": 2, "acceptable": 1, "unsure": 1, "not_acceptable": 0}
    assert spot["concepts"]["mitoses"]["not_acceptable"] == 1 and spot["cells"]["acceptable"] == 1
    assert spot["kinds"] == ["ai"]


@pytest.mark.parametrize("variant", ["empty", "broken", "nulls"])
def test_missing_or_malformed_files_never_throw(payloads, variant):
    p, cfg, tmp = payloads
    html = _render(p[variant], cfg, tmp)["html"]
    assert 'data-testid="metrics"' in html and 'data-testid="limitations"' in html
    assert "Not available in this run" in html
    for junk in ("NaN", "undefined", "[object Object]"):
        assert junk not in html, junk


def test_resplit_figure_renders_from_json(payloads):
    p, cfg, tmp = payloads
    RS = p["full"]["eval"]["conformal_resplit"]
    html = _render(p["full"], cfg, tmp)["html"]
    assert 'data-testid="metrics-conformal-resplit"' in html
    assert "Conformal coverage across 40 random re-splits" in html            # count comes from the JSON
    assert "90% target" in html and 'class="cx-ref"' in html
    assert "Original split" in html and "2.5th–97.5th percentile" in html
    # takeaway is computed from the JSON: the tier set's share of re-splits at or above target
    tier = RS["drives_tiers"] if RS["drives_tiers"] in RS["methods"] else "aps"
    share = round(100 * RS["methods"][tier]["share_at_or_above_target"])
    assert f"{share}% of re-splits" in html
    # confidence section order: re-split figure follows the coverage figure
    assert html.index('id="fig-coverage"') < html.index('id="fig-resplit"')
    for junk in ("NaN", "undefined", "[object Object]", "Infinity"):
        assert junk not in html, junk


def test_resplit_mean_on_the_tier_tile(payloads):
    p, cfg, tmp = payloads
    RS = p["full"]["eval"]["conformal_resplit"]
    html = _render(p["full"], cfg, tmp)["html"]
    headline = html[html.index('data-testid="metrics-headline"'):]
    tile = headline[headline.index("Conformal coverage ("):]
    tile = tile[: tile.index("</article>")]
    conf = p["full"]["eval"]["conformal"]["methods"]
    key = p["full"]["eval"]["conformal"]["drives_tiers"]          # the tile's method, as tierMethod() picks it
    key = key if key in conf else ("aps" if key == "aps_marginal" and "aps" in conf else "lac_marginal")
    mean = RS["methods"][key]["coverage"]["mean"]
    assert f"Mean across {RS['n_splits']} random re-splits: {100 * mean:.1f}%" in tile


def test_resplit_missing_file_gives_standard_card(payloads):
    p, cfg, tmp = payloads
    payload = copy.deepcopy(p["full"])
    del payload["eval"]["conformal_resplit"]
    html = _render(payload, cfg, tmp)["html"]
    assert 'data-testid="metrics-missing-conformal_resplit"' in html
    assert "Not available in this run: <code>eval/conformal_resplit.json</code>" in html
    assert "random re-splits:" not in html                                   # the tile drops the re-split line


DEFERRAL_TITLE = "Accuracy on kept reads as the least reliable reads are deferred"


@pytest.fixture(scope="module")
def deferral_json(tmp_path_factory):
    from eval import deferral as DF
    from tests.test_deferral import write_deferral_root
    tmp = tmp_path_factory.mktemp("deferral")
    write_deferral_root(tmp / "data", CFG)
    return io.read_json(DF.run(D.make_ctx(CFG, tmp / "data", tmp / "art", step="eval_deferral")))


def test_deferral_figure_renders_from_its_json(payloads, deferral_json):
    p, cfg, tmp = payloads
    payload = copy.deepcopy(p["full"])
    payload["eval"]["deferral"] = deferral_json
    html = _render(payload, cfg, tmp)["html"]
    assert 'data-testid="metrics-deferral"' in html and DEFERRAL_TITLE in html
    fig = html[html.index('id="fig-deferral"'):]
    fig = fig[:fig.index("</figure>")]
    assert fig.count("cx-line") >= 4 and "cx-dash" in fig                  # reliability x 2 populations, random dashed
    h = deferral_json["headline"]
    pct = lambda v: f"{100 * v:.1f}%"                                       # noqa: E731
    assert pct(h["accuracy_kept"]["estimate"]) in fig and pct(h["accuracy_all"]["estimate"]) in fig
    assert pct(h["accuracy_random_20"]) in fig and "simulated" in fig and "eval/deferral.json" in fig
    assert str(h["deferred"]) in fig
    from tests.test_deferral import model_lines
    pop_line = fig[fig.index('class="mx-pop"'):]
    for pop in deferral_json["populations"].values():                    # every scoring model with its count
        assert len(pop["models"]) >= 1 and all(line in pop_line for line in model_lines(pop))
    assert len(deferral_json["populations"]["all_test_reads"]["models"]) >= 2
    assert "cb_text (concept-bottleneck model)" in pop_line
    for junk in ("NaN", "undefined", "[object Object]", "Infinity"):
        assert junk not in fig, junk
    # the figure sits in the frozen-section reliability section, after the reliability-score figure
    sec = html[html.index('id="mx-reliability"'):html.index('id="mx-workflow"')]
    assert 'id="fig-deferral"' in sec
    if 'id="fig-relscore"' in sec:                                           # absent when reliability.json is
        assert sec.index('id="fig-relscore"') < sec.index('id="fig-deferral"')


@pytest.mark.parametrize("value", ["absent", None, {"name": "deferral", "populations": {}, "headline": None}])
def test_deferral_figure_degrades_to_the_missing_card(payloads, value):
    p, cfg, tmp = payloads
    payload = copy.deepcopy(p["full"])
    if value != "absent":
        payload["eval"]["deferral"] = value
    html = _render(payload, cfg, tmp)["html"]
    assert 'data-testid="metrics-missing-deferral"' in html and 'data-testid="metrics-deferral"' not in html
    assert "undefined" not in html and "NaN" not in html


def _deletion_doc():
    def blk(e, n=34, g=17):
        return {"estimate": e, "lo": e - 0.01, "hi": e + 0.02, "n": n, "n_groups": g}

    pop = {"n": 34, "n_groups": 17, "mean_drop_random": blk(-0.0012), "mean_drop_risk": blk(0.0237),
           "mean_drop_concept": blk(0.0016), "mean_drop_contribution": blk(0.0324),
           "mean_drop_logit_random": blk(-0.0127), "mean_drop_logit_risk": blk(0.2087),
           "mean_drop_logit_concept": blk(-0.033), "mean_drop_logit_contribution": blk(0.4472),
           "risk_minus_random": blk(0.0249), "concept_minus_random": blk(0.0029),
           "contribution_minus_random": blk(0.0336), "risk_minus_random_logit": blk(0.2213),
           "concept_minus_random_logit": blk(-0.0203), "contribution_minus_random_logit": blk(0.4599)}
    return {"name": "deletion", "synthetic": True, "settings": {"n_tiles": 3, "n_random": 20,
                                                               "primary_subset": "predicted_aggressive"},
            "models": {"uni2h512_lr": {"kind": "lr_embedding", "predicted_aggressive": pop, "all_eligible": pop}}}


def test_deletion_figure_shows_four_arms_in_probability_and_logit_units(payloads):
    p, cfg, tmp = payloads
    full = copy.deepcopy(p["full"])
    full["eval"]["deletion"] = _deletion_doc()
    html = _render(full, cfg, tmp)["html"]
    start = html.index('id="fig-deletion"')
    fig = html[start:html.index("</figure>", start)]
    for arm in ("3 random regions", "Top 3 tile-probe regions", "Top 3 regions of the top concept",
                "Top 3 regions by exact contribution"):
        assert arm in fig, arm
    assert "Drop in P(Invasive)" in fig and "Drop in the centered logit" in fig
    title = fig[fig.index("<h3>") + 4:fig.index("</h3>")]
    assert "?" not in title and "↗" not in fig
    take = fig[fig.index("What this shows"):]
    assert "0.45" in take and "exact contribution" in take and "0.21" in take and "random" in take
    for junk in ("NaN", "undefined", "[object Object]"):
        assert junk not in fig


def test_deletion_figure_without_the_contribution_arm_still_renders(payloads):
    p, cfg, tmp = payloads
    full = copy.deepcopy(p["full"])
    doc = _deletion_doc()
    pop = doc["models"]["uni2h512_lr"]["predicted_aggressive"]
    for key in [k for k in pop if "contribution" in k or "logit" in k]:
        del pop[key]
    full["eval"]["deletion"] = doc
    html = _render(full, cfg, tmp)["html"]
    start = html.index('id="fig-deletion"')
    fig = html[start:html.index("</figure>", start)]
    assert "Top 3 tile-probe regions" in fig and "What this shows" in fig
    for junk in ("NaN", "undefined", "[object Object]"):
        assert junk not in fig


def test_credits_come_from_the_data_and_name_reviewers_plainly(payloads):
    p, cfg, tmp = payloads
    full = copy.deepcopy(p["full"])
    full["eval"]["fact_sheet"] = {"name": "fact_sheet", "components": {"language_models": {
        "case_summaries": {"generators": {"template": 328}},
        "plain_language_tab": {"models": {"claude-sonnet-5-5 (claude)": 328}}}}}
    full["eval"]["spotcheck"] = {"name": "spotcheck", "reviews": [
        {"item_id": f"concept:invasion:X:{i}", "reviewer": "Claude Opus 5.5 (AI, not a pathologist)",
         "reviewer_kind": "ai", "verdict": "acceptable"} for i in range(3)] + [
        {"item_id": "cells:overlay:X:0", "reviewer": "Codex UI verification (AI, no clinical assessment)",
         "reviewer_kind": "ai", "verdict": "unsure"}]}
    html = _render(full, cfg, tmp)["html"]
    cite = html[html.index("Datasets, models, tools and reviewers"):]
    cite = cite[:cite.index("</figure>")]
    assert "Lumen" not in cite and "gemma" not in cite
    assert "<dt>Case summaries</dt><dd>A deterministic template from the structured findings, in all 328 case files; " \
           "no language model writes them.</dd>" in cite
    assert "Claude Sonnet 5.5 (Anthropic) through the Claude Code CLI, 328 case texts" in cite
    assert "AI spot check: Claude Opus 5.5 (Anthropic), 3 reviews; Codex UI test, 1 review." in cite
    assert "No human or pathologist review is recorded." in cite
    assert "Reviewers: Claude Opus 5.5 (AI, 3 reviews); Codex UI test (AI, 1 review)." in html
    for raw in ("5.5 (AI, not a pathologist)", "no clinical assessment", "Codex UI verification"):
        assert raw not in html, raw
    # without the fact sheet the authorship rows are left out, never guessed
    del full["eval"]["fact_sheet"]
    html = _render(full, cfg, tmp)["html"]
    assert "<dt>Case summaries</dt>" not in html and "<dt>Plain-language evidence text</dt>" not in html


def test_narrative_is_plain_language(payloads):
    p, cfg, tmp = payloads
    full = copy.deepcopy(p["full"])
    full["eval"]["qc"] = {"name": "qc", "n_cases": 400, "n_tiles": 1000,
                          "insufficient_tissue_cases": {"old": 51, "new": 0, "old_by_class": {"Invasive": 32, "Normal": 19},
                                                        "new_by_class": {"Invasive": 0, "Normal": 0}},
                          "qc_pass_rate": {"old": {"g256": 0.74}, "new": {"g256": 0.98}}}
    html = _render(full, cfg, tmp)["html"]
    assert "worklist tiers as in the app (tier 1: frozen sections;" in html
    assert "74% → 98% (256-px tiles)" in html and "so the earlier step mostly missed tumor tissue" in html
    for raw in ("core.tiers", "!=", "tumour", "Tumour", "likely aggressive", "the bug", "(g256)", "pcal__", "p__)",
                "predictions.csv", "Lumen"):
        assert raw not in html, raw
