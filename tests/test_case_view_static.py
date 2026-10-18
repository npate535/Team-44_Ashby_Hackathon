"""Browser-free checks of the case-view helpers (why.js, heatmap.js) via node: levels come only from value_std,
the conformal method name and coverage come from eval/conformal.json, margin scores stay readable near 1, AI spot
checks are labelled, and ground truth never reaches the rendered HTML."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap

import pytest

from core import config
from tests.app_helpers import FIXTURES

APP = config.REPO_ROOT / "team44-triage-frontend" / "app"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_node(body: str) -> dict:
    script = textwrap.dedent(f"""
        import * as A from {json.dumps((APP / 'js' / 'adapter.js').as_uri())};
        import * as W from {json.dumps((APP / 'js' / 'views' / 'why.js').as_uri())};
        import * as H from {json.dumps((APP / 'js' / 'views' / 'heatmap.js').as_uri())};
        const cfg = {json.dumps(json.loads((APP / 'mock' / 'config.json').read_text()))};
        const biopsy = {json.dumps(json.loads((FIXTURES / 'demo_0001.json').read_text()))};
        const surg = {json.dumps(json.loads((FIXTURES / 'demo_0002.json').read_text()))};
        const out = {{}};
        {body}
        console.log(JSON.stringify(out));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_level_comes_only_from_value_std():
    out = run_node("""
        const row = (extra) => ({id: 'invasion', label: 'x', value: 1.9, contribution: 0.4, direction: 'up', ...extra});
        out.above = W.levelOf(row({valueStd: 1.6}));
        out.below = W.levelOf(row({valueStd: -0.8}));
        out.typical = W.levelOf(row({valueStd: 0.2}));
        out.none = W.levelOf(row({}));                       // no value_std: no level word, tile z only in the tooltip
        out.cell = W.levelOf({id: 'nuc_area_cv', measured: true}, {features: {nuc_area_cv: {ref_percentile: 0.95}}});
    """)
    assert out["above"]["word"] == "Above typical" and "+1.60 SD vs training cases" in out["above"]["short"]
    assert out["below"]["word"] == "Below typical" and out["typical"]["word"] == "Typical"
    assert out["none"]["word"] == "" and "tile z" in out["none"]["tip"] and "+1.90" in out["none"]["tip"]
    assert out["cell"]["word"] == "High" and "95%" in out["cell"]["long"]


def test_why_panel_groups_spot_checks_and_no_ground_truth():
    out = run_node("""
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        const metrics = {eval: {spotcheck: {summary: {concept: {invasion: {n: 20, acceptable: 19},
                                                                duct_confined: {n: 20, acceptable: 3}}}}}};
        const hm = {sel: 'risk', hl: [], opacity: 0.6, why: null};
        out.html = W.why(v, hm, cfg, metrics);
        out.plain = W.why(v, hm, cfg, null);
    """)
    html = out["html"]
    for group in cfg_groups():
        assert group in html
    assert "AI spot check 19/20" in html and "not pathologist-validated" in html
    assert "AI spot check 3/20 · top regions often did not show this feature" in html
    assert "AI spot check" not in out["plain"]                      # absent eval file: render nothing
    assert "why-no-level" in out["plain"]                           # fixture has no value_std: one note, no words
    assert "eval_only" not in html and "true_label" not in html


def cfg_groups() -> list[str]:
    cfg = json.loads((APP / "mock" / "config.json").read_text())
    return [g["label"] for g in cfg["labels"]["concept_groups"].values()]


def test_conformal_name_and_coverage_follow_drives_tiers():
    out = run_node("""
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        const metrics = (method, cov) => ({eval: {metrics: {models: {production: v.model}},
          conformal: {drives_tiers: method, n: 132, target_coverage: 0.9,
                      methods: {other_method: {coverage: {estimate: 0.5}}, [method]: {coverage: {estimate: cov}}}}}});
        out.aps = W.conformalName(metrics('aps_marginal_nonempty', 0.9), 90);
        out.lac = W.conformalName(metrics('lac_marginal', 0.67), 90);
        out.none = W.conformalName(null, 90);
        out.htmlAps = W.howSure(v, cfg, metrics('aps_marginal_nonempty', 0.91));
        out.htmlLac = W.howSure(v, cfg, metrics('lac_marginal', 0.674));
        out.empty = W.howSure({...v, probs: []}, cfg, null);
    """)
    assert out["aps"] == "APS conformal set (90% target)" and out["lac"] == "LAC conformal set (90% target)"
    assert out["none"] == "Conformal set (90% target)"
    assert "91%" in out["htmlAps"] and "not achieved" not in out["htmlAps"]
    assert "67%" in out["htmlLac"] and "not achieved" in out["htmlLac"]
    assert "In situ carcinoma" in out["htmlAps"] and "in set" in out["htmlAps"]
    assert "no-prediction" in out["empty"]


def test_margin_scores_are_readable_near_one():
    out = run_node("""
        out.p = [H.fmtProb(1.0), H.fmtProb(0.99976), H.fmtProb(0.320838), H.fmtProb(0.00001)];
        out.l = [H.fmtLogit(1.0), H.fmtLogit(0.5), H.logit(1.0), H.logit(0)];
        const m = {involved: ['superior'], perMargin: {superior: 1.0, inferior: 0.32, medial: 0.99976, lateral: 0.2},
                   threshold: 0.999823, segments: [], lengths: {superior: 860}};
        out.st = ['superior', 'inferior', 'medial', 'lateral'].map(n => H.marginStatus(m, n));
        const sv = A.toView({...surg, _fetched_ms: 0}, 0, cfg, null);
        out.bars = H.marginBars(sv);
        out.diagram = H.specimenDiagram(sv, cfg.margin.names);
    """)
    assert out["p"] == [">0.9999", "0.9998", "0.321", "<0.0001"]
    assert out["l"][0].startswith("≥ +12") and out["l"][1] == "±0.0" and out["l"][2] == 12 and out["l"][3] == -12
    assert out["st"] == ["involved", "clear", "near", "clear"]
    assert "log-odds" in out["bars"] and "Simulated" in out["bars"]
    for name in ("Superior", "Inferior", "Medial", "Lateral"):
        assert name in out["diagram"]


def test_region_info_explains_qc_failed_regions():
    out = run_node("""
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        const present = new Set(v.tiles.map(t => t.idx));
        const [rows, cols] = v.image.grid;
        const missing = [...Array(rows * cols).keys()].find(i => !present.has(i));
        const hm = {sel: 'risk', hl: [], opacity: 0.6, why: null};
        out.missing = missing === undefined ? null : H.regionInfo(v, hm, missing).text;
        out.kept = H.regionInfo(v, hm, v.tiles[0].idx).text;
        out.grid = H.regionGrid(v, hm);
    """)
    if out["missing"] is not None:
        assert "failed tissue quality control" in out["missing"]
    assert "Tile P(Invasive)" in out["kept"] and "Passed tissue QC" in out["kept"]
    assert out["grid"].count('tabindex="0"') == 1                     # roving tabindex: a single tab stop
