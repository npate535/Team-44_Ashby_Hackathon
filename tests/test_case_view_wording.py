"""Case-view wording (design critique): developer ids leave the reading path but stay in one "Model and provenance"
disclosure, no stray dot after probability rows outside the set, citation chips stay with the word before them, a
surgical case says "simulated" once (tag + Margins card line, no dashed banner), and the frozen-section stain rule and
the defer suggestion are each said once. why.js runs in node; the DOM-bound views are checked in their source."""

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
VIEWS = APP / "js" / "views"
node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_node(body: str) -> dict:
    script = textwrap.dedent(f"""
        import * as A from {json.dumps((APP / 'js' / 'adapter.js').as_uri())};
        import * as W from {json.dumps((VIEWS / 'why.js').as_uri())};
        const cfg = {json.dumps(json.loads((APP / 'mock' / 'config.json').read_text()))};
        const biopsy = {json.dumps(json.loads((FIXTURES / 'demo_0001.json').read_text()))};
        const out = {{}};
        {body}
        console.log(JSON.stringify(out));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


@node
def test_prediction_card_plain_model_name_and_provenance_disclosure():
    out = run_node("""
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        v.model = 'uni2h512_lr';
        v.agent = {...v.agent, orders: [{rule_id: 'breast_invasive', type: 'stain', items: ['p63']}], blocked: []};
        out.html = W.howSure(v, cfg, null);
        out.open = W.howSure(v, cfg, null, {provOpen: true});
        out.name = W.modelName('uni2h512_lr');
        out.other = W.modelName('conch_lr');
        out.unknown = W.modelName('xyz');
    """)
    html = out["html"]
    assert out["name"] == "UNI2-h linear probe" and out["other"] == "CONCH linear probe" and out["unknown"] == "xyz"
    assert "UNI2-h linear probe, calibrated" in html and "temperature-calibrated" not in html and "Stamp" not in html
    prov = re.search(r'<details class="cv-more cv-prov" data-testid="provenance"[^>]*>.*?</details>', html, re.S)
    assert prov and "Model and provenance" in prov.group(0)
    # every id is still there, only inside the disclosure
    assert "uni2h512_lr" in prov.group(0) and "breast_invasive" in prov.group(0) and "contract" in prov.group(0).lower()
    assert "uni2h512_lr" not in html.replace(prov.group(0), "")
    assert ' open' in re.search(r'<details class="cv-more cv-prov"[^>]*>', out["open"]).group(0)


@node
def test_probability_rows_outside_the_set_have_no_visible_dot():
    out = run_node("""
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        v.set = [v.top];
        out.html = W.howSure(v, cfg, null);
    """)
    outs = re.findall(r'<span class="pc-out">(.*?)</span></span>', out["html"])
    assert outs, "a fixture class outside the set"
    for o in outs:
        assert "·" not in o and '<span class="sr-only">outside the set' in o


@node
def test_summary_citation_chips_stay_with_the_last_word():
    out = run_node("""
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        out.html = W.summary(v);
        out.n = v.summary.sentences.filter((s) => (s.findings || []).length).length;
    """)
    html = out["html"]
    groups = re.findall(r'<span class="fref-nb">([^<\s]*)<button[^>]*class="fref"', html)
    assert out["n"] and len(groups) == out["n"] and all(groups)
    assert not re.search(r"</span>\s*<button[^>]*class=\"fref\"", html.replace('<span class="fref-nb">', ""))
    css = (APP / "case.css").read_text()
    assert re.search(r"\.fref-nb\s*\{\s*white-space:\s*nowrap", css)


def test_order_rule_id_is_a_tooltip_not_reading_text():
    src = (VIEWS / "agent.js").read_text()
    assert "rule <code>" not in src and "ord-rule" not in src
    assert re.search(r'title="Drafted by rule \$\{esc\(o\.rule_id\)\}', src)


def test_surgical_stain_rule_and_defer_said_once():
    agent = (VIEWS / "agent.js").read_text()
    assert "<b>Guard</b>" not in agent and "Stains are not ordered from frozen sections." in agent
    assert "b.action === 'draft_stains'" in agent  # the blocked stain draft is folded into that one line
    surg = (VIEWS / "case_surgical.js").read_text()
    assert "suggestion: defer" not in surg and "consider permanent sections" not in surg
    assert "m && !mergedDefer" in surg and "Defer to permanent sections: reliability flag raised" in surg


def test_surgical_says_simulated_once_and_no_banner():
    surg = (VIEWS / "case_surgical.js").read_text()
    biopsy = (VIEWS / "case_biopsy.js").read_text()
    assert "simbanner" not in surg and "sim-tag" not in surg and "(simulated)" not in surg
    assert 'class="cv-simline" data-testid="sim-banner"' in surg and "Not a validated margin assessment." in surg
    assert 'data-testid="sim-tag"' in biopsy and ">Simulated</span>" in biopsy
    assert "arrived sim +" not in biopsy and "(simulated replay)" in biopsy
    # the global prototype notice still says the frozen sections are simulated
    assert "Frozen sections and arrivals are simulated" in (APP / "index.html").read_text()
