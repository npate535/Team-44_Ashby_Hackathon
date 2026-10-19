"""Browser-free checks of the case-view helpers (why.js, heatmap.js) via node: levels come only from value_std,
the conformal method name and coverage come from eval/conformal.json, margin scores stay readable near 1, AI spot
checks are labelled, and ground truth never reaches the rendered HTML."""

from __future__ import annotations

import json
import re
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
    # below the fixed 0.75 cut-off: muted row, association only (same rule as the case JSON's `reliable`)
    assert "Spot check failed (3/20); association only" in html and "AI spot check 3/20" not in html
    rows = {m.group(2): m.group(1) for m in re.finditer(r'class="why-row([^"]*)" data-testid="why-bar-([a-z_]+)"', html)}
    assert "unreliable" in rows["duct_confined"] and "unreliable" not in rows["invasion"]
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


# ---- exact region contributions (Workstream A) ------------------------------------------------------------------------
LEGEND = ("Each region's push toward (pink) or away from (blue) invasive carcinoma. The pushes plus a constant add up "
          "exactly to the model's invasive score.")


def _contrib_view(extra: str = "") -> str:
    """JS prelude: a biopsy view whose tiles carry known contributions (region 7 = +0.42, region 2 = -0.21)."""
    return """
        const base = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        const vals = {0: 0.05, 1: -0.1, 2: -0.21, 3: 0.0, 4: 0.12, 5: 0.3, 6: 0.25, 7: 0.42, 8: -0.02, 9: 0.18, 10: null, 11: 0.01};
        const v = {...base, tiles: base.tiles.map(t => ({...t, contribution: vals[t.idx]}))};
        const hm = {sel: null, hl: [], opacity: 0.6, why: null};
    """ + extra


def test_contribution_layer_is_offered_and_selected_by_default():
    out = run_node(_contrib_view("""
        out.attribution = v.attribution;
        out.layers = H.layers(v).map(l => l.label);
        out.default = H.layerOf(v, null).id;
        out.controls = H.heatmapControls(v, hm);
        out.svgKind = /data-kind="([a-z]+)"/.exec(H.overlaySvg(v, hm))[1];
    """))
    assert out["attribution"]["method"] == "exact_linear"
    assert out["layers"][0] == "Region contribution (exact)" and "Overall risk (tile P(Invasive))" in out["layers"]
    assert out["default"] == "contribution" and out["svgKind"] == "contribution"
    assert '<option value="contribution" selected>Region contribution (exact)</option>' in out["controls"]


def test_contribution_colours_are_signed_and_scaled_to_the_case_max():
    out = run_node(_contrib_view("""
        const svg = H.overlaySvg(v, hm);
        const rect = (i) => (svg.match(new RegExp('<rect class="hm-tile ([a-z]+)"[^>]*data-tile="' + i + '" data-v="([0-9.]+)"')) || []).slice(1);
        out.r7 = rect(7); out.r2 = rect(2); out.r10 = rect(10);
        out.legend = H.heatmapLegend(v, hm);
        out.tip7 = H.regionInfo(v, hm, 7);
        out.tip2 = H.regionInfo(v, hm, 2).text;
        out.tip10 = H.regionInfo(v, hm, 10).text;
        out.grid = H.regionGrid(v, hm);
    """))
    assert out["r7"] == ["pos", "1.0000"]                              # the case max |contribution| is fully shaded
    assert out["r2"] == ["neg", "0.5000"]                              # 0.21 / 0.42, blue
    assert out["r10"][1] == "0.0000"                                   # pooled-region value missing: unshaded
    assert LEGEND.replace("'", "&#39;") in out["legend"] and 'data-kind="contribution"' in out["legend"]
    assert "+0.42" in out["legend"] and "−0.42" in out["legend"]
    assert out["tip7"]["text"].startswith("Region 7 · +0.42 toward invasive")
    assert "Region 7 · +0.42 toward invasive" in out["tip7"]["html"]
    assert out["tip2"].startswith("Region 2 · −0.21 away from invasive")
    assert "not available" in out["tip10"]
    assert ">+0.42<" in out["grid"]


def test_qc_failed_regions_stay_hatched_in_the_contribution_layer():
    out = run_node(_contrib_view("""
        const kept = {...v, tiles: v.tiles.filter(t => ![3, 8].includes(t.idx))};
        const svg = H.overlaySvg(kept, hm);
        out.drop3 = svg.includes('class="hm-drop" style="fill:url(#hm-hatch)" data-tile="3"');
        out.drop8 = svg.includes('class="hm-drop" style="fill:url(#hm-hatch)" data-tile="8"');
        out.tile3 = svg.includes('class="hm-tile pos" data-testid="hm-tile" data-tile="3"') || svg.includes('class="hm-tile neg" data-testid="hm-tile" data-tile="3"');
        out.tip3 = H.regionInfo(kept, hm, 3).text;
    """))
    assert out["drop3"] and out["drop8"] and not out["tile3"]
    assert "failed tissue quality control" in out["tip3"]


def test_contribution_layer_absent_without_contributions():
    out = run_node("""
        const raw = {...biopsy, prediction: {...biopsy.prediction, attribution: null},
                     tiles: biopsy.tiles.map(t => ({...t, contribution: null}))};
        const v = A.toView({...raw, _fetched_ms: 0}, 0, cfg, null);
        const hm = {sel: null, hl: [], opacity: 0.6, why: null};
        out.layers = H.layers(v).map(l => l.id);
        out.default = H.layerOf(v, null).id;
        out.why = W.why(v, hm, cfg, null);
        const v2 = A.toView({...biopsy, prediction: (({attribution, ...p}) => p)(biopsy.prediction),
                             tiles: biopsy.tiles.map(({contribution, ...t}) => t), _fetched_ms: 0}, 0, cfg, null);
        out.layers2 = H.layers(v2).map(l => l.id);
    """)
    assert "contribution" not in out["layers"] and out["default"] == "risk"
    assert "contribution" not in out["layers2"]
    assert "Regions with the largest push" not in out["why"]


def test_why_card_lists_the_regions_with_the_largest_push():
    out = run_node(_contrib_view("""
        out.html = W.why(v, hm, cfg, null);
        out.top = W.topRegions(v, 3).map(r => [r.idx, r.contribution, r.concepts.map(c => c.id)]);
    """))
    html = out["html"]
    assert "Regions with the largest push" in html
    assert [r[0] for r in out["top"]] == [7, 5, 6]                    # the three largest positive contributions
    assert all(len(r[2]) == 2 for r in out["top"])                     # each with its two highest concept z-scores
    for idx in (7, 5, 6):
        assert f'data-rg="{idx}" data-rg-scope="case"' in html         # opens the viewer focused on that region
    assert "+0.42" in html and "Region 7" in html
    assert html.index("Region 7") < html.index("Region 5") < html.index("Region 6")


# ---- look-alike contrast detail (judge feedback T4) ----------------------------------------------------------------------
CONTRAST_NOTE = ("Feature scores measure similarity to written descriptions of each feature, standardized on training "
                 "cases. They show association, not a pathologist's reasoning.")
CONTRAST_DETAIL = [
    {"concept": "invasion", "label": "Tumor infiltrating stroma", "query_z": 1.62, "contrast_z": -0.41, "delta": 2.03,
     "spotcheck": {"acceptable": 19, "reviewed": 20}, "reliable": True, "query_region": 5, "contrast_region": 6},
    {"concept": "necrosis", "label": "Tumor necrosis", "query_z": 0.12, "contrast_z": 1.45, "delta": -1.33,
     "spotcheck": {"acceptable": 3, "reviewed": 20}, "reliable": False, "query_region": 2, "contrast_region": None},
]


def _compared(detail, differs=None) -> str:
    out = run_node(f"""
        const detail = {json.dumps(detail)};
        const differs = {json.dumps(differs)};
        const cc = {{...biopsy.contrast_case, detail}};
        if (differs) cc.differs_most_in = differs;
        const v = A.toView({{...biopsy, contrast_case: cc, _fetched_ms: 0}}, 0, cfg, null);
        out.html = W.compared(v, cfg, (p) => p);
    """)
    return out["html"]


def _rows(html: str) -> list[str]:
    body = html[html.index("<tbody"):html.index("</tbody>")]
    return body.split("<tr")[1:]


def test_contrast_table_shows_direction_magnitude_spot_check_and_region():
    html = _compared(CONTRAST_DETAIL, [d["concept"] for d in CONTRAST_DETAIL])
    assert 'data-testid="contrast-detail"' in html and "Differs most in" not in html
    for header in ("Feature", "This case", "Look-alike", "Difference (SD of training cases)", "Spot check"):
        assert f">{header}</th>" in html
    ok, failed = _rows(html)
    assert "Tumor infiltrating stroma" in ok and "+1.62" in ok and "−0.41" in ok and "+2.03" in ok
    assert "19/20" in ok and "AI-reviewed" in ok and "unreliable" not in ok and "not used as a reason" not in ok
    assert 'data-rg="5" data-rg-scope="case"' in ok                  # opens this case at its strongest region
    assert "strongest in region 5" in ok and "strongest in region 6" in ok
    assert "Tumor necrosis" in failed and "−1.33" in failed and "3/20" in failed
    assert 'class="ct-row unreliable"' in failed and "Spot check failed; not used as a reason" in failed
    assert failed.count("strongest in region") == 1                  # look-alike region unknown: no region text
    assert html.count("Spot check failed; not used as a reason") == 1
    assert CONTRAST_NOTE.replace("'", "&#39;") in html or CONTRAST_NOTE in html
    assert 'class="ct-mark" data-ct="0"' in html and 'data-ct="1" style' not in html   # thumbnail outline per region


def test_contrast_table_without_a_spot_check_and_all_rows_unreliable():
    detail = [{**CONTRAST_DETAIL[0], "spotcheck": None, "reliable": False}, CONTRAST_DETAIL[1]]
    html = _compared(detail, [d["concept"] for d in detail])
    first, second = _rows(html)
    assert "No spot check; not used as a reason" in first and 'class="ct-row unreliable"' in first
    assert "Spot check failed; not used as a reason" in second
    assert "No feature in this comparison passed its AI spot check" in html


def test_contrast_falls_back_to_chips_without_detail():
    for detail in (None, []):
        html = _compared(detail)
        assert 'data-testid="contrast-detail"' not in html and "Differs most in" in html
        assert 'class="chip"' in html
