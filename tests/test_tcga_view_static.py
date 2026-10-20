"""Browser-free checks of the Outside BACH page (js/views/tcga.js) via node, and of the TCGA case summaries written by
scripts/build_tcga_cases.py: boundary results get exact intervals over their true denominators, never a collapsed
"100.0%–100.0%" bootstrap; the cohort is stated once; summaries use display labels and the header's percent format."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import textwrap

import pytest

from core import config
from tests.app_helpers import FIXTURES

APP = config.REPO_ROOT / "team44-triage-frontend" / "app"
node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def boot(estimate: float, lo: float, hi: float, n: int, n_groups: int) -> dict:
    return {"estimate": estimate, "lo": lo, "hi": hi, "ci": 0.95, "n_boot": 2000, "n": n, "n_groups": n_groups}


def state(confusion: bool = True) -> dict:
    production = {
        "accuracy": boot(114 / 132, 0.7600, 0.9444, 132, 24),
        "balanced_accuracy": boot(0.8705, 0.6765, 0.9375, 132, 24),
        "invasive_sensitivity": boot(1.0, 1.0, 1.0, 132, 24),      # n counts every test image: not the denominator
        "invasive_vs_rest_auroc": boot(0.9990, 0.9961, 1.0, 132, 24),
    }
    if confusion:
        production["confusion"] = {"classes": ["Normal", "Benign", "InSitu", "Invasive"],
                                   "matrix": [[20, 3, 0, 0], [9, 39, 0, 0], [2, 1, 24, 3], [0, 0, 0, 31]]}
    per_model = {k: v for k, v in production.items() if k != "confusion"}
    external = {
        "model": "m", "quantized": False, "n_patients": 25, "n_slides": 65, "n_crops": 353,
        "groups": {"ffpe_tumor": 23, "frozen_normal": 17, "frozen_tumor": 25},
        "primary": {
            "ffpe_tumor_sensitivity": boot(22 / 23, 0.8696, 1.0, 23, 23),
            "frozen_tumor_sensitivity": boot(0.84, 0.68, 0.96, 25, 25),
            "ffpe_vs_frozen_tumor_sensitivity_gap": {"estimate": 22 / 23 - 0.84, "ffpe": 22 / 23, "frozen": 0.84},
            "frozen_tumor_vs_normal": {"auroc": boot(1.0, 1.0, 1.0, 42, 25),
                                       "sensitivity": boot(0.84, 0.68, 0.96, 25, 25),
                                       "specificity": boot(1.0, 1.0, 1.0, 17, 17)},
            "tumor_tier_1_2_rate": boot(47 / 48, 0.9362, 1.0, 48, 25),
        },
        "bach_reference": {"model": "m", "invasive_sensitivity_patient_bootstrap": boot(1.0, 1.0, 1.0, 31, 17)},
        "limitations": [],
    }
    item = {"case_id": "TCGA-A7-A4SD_frozen_normal_02", "prep": "frozen", "patient_id": "TCGA-A7-A4SD", "tier": 3,
            "top": "Normal", "p_invasive": 0.04, "conformal_set": ["Normal", "InSitu"],
            "tier_reasons": ["set_size_gt_1"], "image": "tcga/x.jpg", "model": "m",
            "probs": {"Normal": 0.6, "Benign": 0.01, "InSitu": 0.35, "Invasive": 0.04}}
    return {
        "metrics": {"eval": {"metrics": {"n": 132, "n_groups": 24, "split": "test", "models": {"production": "m"},
                                         "production": production, "per_model": {"m": per_model}},
                             "external_tcga": external}},
        "tcga": {"available": True, "n": 1, "model": "m", "items": [item]},
        "caseId": None, "cases": {},
    }


def render(st: dict, extra: str = "") -> str:
    """The page's HTML for state `st`; `extra` is JS run after the render, adding keys to `out` (T is the module)."""
    return run(st, extra)["html"]


def run(st: dict, extra: str = "") -> dict:
    script = textwrap.dedent(f"""
        const noop = () => {{}};
        globalThis.document = {{addEventListener: noop, removeEventListener: noop, querySelector: () => null,
                               querySelectorAll: () => [], documentElement: {{dataset: {{}}}}, body: {{}}}};
        globalThis.window = globalThis;
        globalThis.addEventListener = noop;
        globalThis.matchMedia = () => ({{matches: false, addEventListener: noop}});
        const T = await import({json.dumps((APP / 'js' / 'views' / 'tcga.js').as_uri())});
        const ST = {json.dumps(st)};
        ST.cfg = {json.dumps(json.loads((APP / 'mock' / 'config.json').read_text()))};
        const out = {{html: T.render(ST, {{now: 0, assetUrl: (u) => u}})}};
        {extra}
        console.log(JSON.stringify(out));
    """)
    res = subprocess.run(["node", "--no-warnings", "--input-type=module", "-e", script], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))


@node
def test_boundary_results_get_exact_intervals_over_true_denominators():
    html = render(state())
    text = text_of(html)
    assert "100.0%–100.0%" not in text and "1.000–1.000" not in text and "100.0–100.0" not in text
    assert "17 of 17 slides (exact 95% CI 80.5–100%)" in text                # frozen normal specificity
    assert "31 of 31 invasive images (exact 95% CI 88.8–100%)" in text       # Invasive row of the confusion, not 132
    assert "of 132 invasive" not in text
    assert "come from 17 patients, so it is optimistic" in text            # images are clustered within patients
    assert "All 25 frozen tumor slides scored above all 17 frozen normal slides." in text
    assert "21 of 25 slides · 95% CI 68.0–96.0%" in text                     # non-boundary keeps its bootstrap
    assert "114 of 132 images · 95% CI 76.0–94.4%" in text
    assert "47 of 48 slides" in text
    assert "a point estimate only, and the two 95% intervals overlap" in text
    # the cohort is stated once, in fields (not crops)
    assert text.count("25 patients") == 1 and "353 fields" in text and "crop" not in text.lower()
    assert "23 FFPE tumor, 25 frozen tumor, 17 frozen normal" in text
    # the specimen id is split and spelled out, never truncated; the full id stays in the link, title and data
    assert '<b class="tcga-pid">TCGA-A7-A4SD</b><span class="tcga-fld">Frozen · normal · field 2</span>' in html
    assert 'href="#/tcga/TCGA-A7-A4SD_frozen_normal_02" tabindex="-1" title="TCGA-A7-A4SD_frozen_normal_02"' in html
    assert 'data-case-id="TCGA-A7-A4SD_frozen_normal_02"' in html
    assert 'data-testid="tcga-row"' in html


@node
def test_bach_denominator_falls_back_to_the_patient_bootstrap_reference():
    text = text_of(render(state(confusion=False)))
    assert "31 of 31 invasive images (exact 95% CI 88.8–100%)" in text


@node
def test_specimen_list_has_set_filter_tier_toggles_and_one_tab_stop():
    st = state()
    base = st["tcga"]["items"][0]
    ids = ["TCGA-A7-A4SD_ffpe_tumor_00", "TCGA-A7-A4SD_ffpe_tumor_01", "TCGA-A7-A4SD_frozen_tumor_00",
           "TCGA-A7-A4SD_frozen_normal_02"]
    st["tcga"]["items"] = [{**base, "case_id": cid, "tier": 2 if "tumor" in cid else 3} for cid in ids]
    html = render(st)
    # one segmented filter whose counts add up to every field, "All" pressed
    for key, text, n in (("all", "All", 4), ("ffpe_tumor", "FFPE tumor", 2), ("frozen_tumor", "Frozen tumor", 1),
                         ("frozen_normal", "Frozen normal", 1)):
        assert re.search(rf'data-tcga-set="{key}"[^>]*aria-pressed="{str(key == "all").lower()}">{text} '
                         rf'<b class="tcga-n">{n}</b>', html), key
    # tier groups are disclosure buttons with their counts; expanded off a phone
    assert re.search(r'data-tcga-tier="2"[^>]*aria-expanded="true">.*?Tier 2 · Likely invasive</span>'
                     r' <span class="g-n">3</span>', html)
    assert 'data-tcga-tier="3"' in html and "Tier 3 · Low confidence" in html
    # roving tab stop: one row is tabbable, the others are reached with J/K and the arrow keys
    assert len(re.findall(r'data-testid="tcga-row"', html)) == 4
    assert html.count('tabindex="0"') == 1 and html.count('class="row tr-2" tabindex="0"') == 1
    assert 'aria-label="TCGA-A7-A4SD_frozen_normal_02, Frozen normal, field 2, tier 3 Low confidence' in html
    # one page name: the page's heading is "External test (TCGA-BRCA)"; it is an h2 (the top bar holds the one h1)
    assert "Outside BACH" not in text_of(html) and "TCGA-BRCA external test" not in text_of(html)
    assert '<h2 class="tcga-intro">External test (TCGA-BRCA)</h2>' in html and "<h1" not in html


@node
def test_limitations_are_plain_language_and_cover_every_source_entry():
    st = state()
    st["metrics"]["eval"]["external_tcga"]["tile_equivalence"] = {"n_crops_compared": 23, "n_slides": 24,
                                                                   "n_slides_picks_identical": 13}
    out = run(st, "out.covers = T.LIMITATIONS.map((l) => l.covers); out.credit = T.CREDIT;")
    section = re.search(r'data-testid="tcga-limits">(.*?)</section>', out["html"], re.S).group(1)
    text = text_of(section)
    # every limitation of the evaluation file is restated, and every restatement points at one of them
    source = json.loads((config.REPO_ROOT / "artifacts" / "eval" / "external_tcga.json").read_text())["limitations"]
    entries = [s for s in source if not s.startswith("Credit:")]
    for entry in entries:
        assert any(c in entry for c in out["covers"]), f"limitation without a plain-language restatement: {entry}"
    for c in out["covers"]:
        assert any(c in entry for entry in entries), f"restates no current limitation: {c}"
    assert section.count("<li>") == len(entries)
    for caveat in ("not outlines of the tumor", "stroma or fat", "four-class accuracy", "p63 and calponin",
                   "adjacent to the tumor", "not computed", "tier 4", "tiers 1–2", "cohort is small", "are wide",
                   "pixel for pixel", "picked on 13 of 24 slides checked", "indicative, not a matched test",
                   "one laptop", "not whole slides", "not clinically validated"):
        assert caveat in text, caveat
    # no file names, field names, internal cohort names or links
    for raw in ("reliability_flag", "protocol25", "DeepZoom", "tile_equivalence", "openslide", "MPS", "OOD",
                "docs/", ".md", ".json", "this laptop", "http", "<a ", "<code"):
        assert raw not in section, raw
    assert "The protocol was written before the run." in text
    assert "TCGA Research Network" in out["credit"] and "NCI Genomic Data Commons" in out["credit"]
    assert out["credit"] in text


@node
def test_panels_say_loading_while_the_evaluation_is_on_its_way():
    st = state()
    st["metrics"] = None
    st["metricsLoading"] = True
    html = render(st)
    panels = text_of(re.search(r'data-testid="tcga-compare">(.*?)</div>\s*<p class="tcga-compare-note"', html, re.S)[1])
    assert 'data-testid="tcga-bach-loading"' in html and 'data-testid="tcga-external-loading"' in html
    assert panels.count("Loading…") == 2 and "not loaded" not in panels and "not available" not in panels
    st["metricsLoading"] = False                     # the request failed: say the metrics are missing
    html = render(st)
    assert 'data-testid="tcga-bach-missing"' in html and "Loading…" not in text_of(html)


@node
def test_flags_column_shows_compact_tags_like_the_worklist():
    st = state()
    base = st["tcga"]["items"][0]
    st["tcga"]["items"] = [
        {**base, "case_id": "TCGA-A7-A4SD_ffpe_tumor_00", "tier": 2, "conformal_set": ["Invasive"],
         "tier_reasons": ["aggressive_top", "aggressive_in_conformal_set"],
         "probs": {"Normal": 0.0, "Benign": 0.0, "InSitu": 0.01, "Invasive": 0.99}, "p_invasive": 0.99},
        {**base, "case_id": "TCGA-A7-A4SD_frozen_tumor_00", "tier": 2, "conformal_set": ["InSitu", "Invasive"],
         "tier_reasons": ["aggressive_in_conformal_set", "set_size_gt_1"],
         "probs": {"Normal": 0.0, "Benign": 0.0, "InSitu": 0.6, "Invasive": 0.4}, "p_invasive": 0.4},
        base,                                                            # tier 3, {Normal, InSitu}
    ]
    html = render(st)
    flags = dict(re.findall(r'data-case-id="([^"]+)".*?<td class="c-flags">(.*?)</td>', html, re.S))
    assert flags["TCGA-A7-A4SD_ffpe_tumor_00"] == '<span class="none" aria-label="no flags">–</span>'
    # the Confidence set column lists the classes, so set size gets no tag (as on the worklist)
    tags = flags["TCGA-A7-A4SD_frozen_tumor_00"]
    assert 'data-testid="reason-aggressive_in_conformal_set">Invasive in set</span>' in tags
    assert "classes" not in tags and "reason-set_size_gt_1" not in tags
    # a field whose only flag was its set size shows the dash, and says why to a screen reader
    assert flags["TCGA-A7-A4SD_frozen_normal_02"] == (
        '<span class="none" aria-label="no flags besides the confidence set">–</span>')
    assert "classes" not in "".join(flags.values()) and "reason-set_size_gt_1" not in "".join(flags.values())
    assert " · " not in "".join(flags.values()) and "suggested" not in "".join(flags.values())


@node
def test_specimen_page_is_titled_as_the_list_names_it():
    case = json.loads((FIXTURES / "demo_0001.json").read_text())
    cid = "TCGA-A7-A4SD_ffpe_tumor_00"
    case.update(case_id=cid, cohort="tcga", lane="biopsy")
    st = {**state(), "caseId": cid, "cases": {cid: case}, "interp": {}, "events": {}, "tab": "evidence",
          "hm": {"sel": "risk", "hl": [], "opacity": 0.6, "why": None}}
    html = render(st)
    title = re.search(r'<h2 data-testid="case-title">(.*?)</h2>(.*?)<div class="sub">', html, re.S)
    assert text_of(title.group(1)).startswith("TCGA-A7-A4SD · FFPE · tumor · field 0")
    assert cid not in title.group(1)
    assert title.group(2) == f'<div class="tcga-raw-id" data-testid="tcga-raw-id">{cid}</div>'
    assert f'data-case-id="{cid}"' in html                       # the case view keeps the raw id in its data


def _build():
    path = config.REPO_ROOT / "scripts" / "build_tcga_cases.py"
    spec = importlib.util.spec_from_file_location("build_tcga_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tcga_summary_uses_display_labels_and_header_percent():
    build = _build()
    labels = dict(config.load_settings()["ui"]["class_labels"])
    s = build.summary_sentences("Invasive", 0.9937, ["Invasive"], 0.1, 12, 3, labels, "Invasive")
    assert s[0]["text"] == ("Suggested category: invasive carcinoma (calibrated probability of invasive carcinoma "
                            "99%); the 90% confidence set holds this category only.")
    assert s[0]["findings"] == ["F1"] and s[1]["findings"] == ["F2"]
    assert "not computed" in s[1]["text"] and "12 of 12 regions" in s[1]["text"]
    s = build.summary_sentences("Normal", 0.001, ["Normal", "InSitu"], 0.1, 2, 3, labels, "Invasive")
    assert "normal breast tissue (calibrated probability of invasive carcinoma <1%)" in s[0]["text"]
    assert "in situ carcinoma is also in the 90% confidence set" in s[0]["text"]
    assert s[1]["text"].startswith("Only 2 tissue regions passed quality control")
    assert not any(raw in s[0]["text"] for raw in ("InSitu", "P(Invasive)", "0.99"))
    assert build._pct(0.995) == ">99%" and build._pct(0.0049) == "<1%" and build._pct(0.125) == "13%"
