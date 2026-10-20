"""Browser-free checks of the Cells tab (cells.js) and the prediction card callout (why.js) via node: what every
feature row shares ("measured", "not used by the prediction model", the percentile axis) is said once above the list,
and the conformal-set statement, its coverage note and the held-out coverage share one callout."""

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
        import * as C from {json.dumps((APP / 'js' / 'views' / 'cells.js').as_uri())};
        const cfg = {json.dumps(json.loads((APP / 'mock' / 'config.json').read_text()))};
        const biopsy = {json.dumps(json.loads((FIXTURES / 'demo_0001.json').read_text()))};
        const out = {{}};
        {body}
        console.log(JSON.stringify(out));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_cells_rows_share_one_note_and_one_axis():
    out = run_node("""
        const ST = {cfg, cellsOverlay: true, cellsTile: null, metrics: null};
        const url = (p) => '/' + p;
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        out.mixed = C.cellsTab(v, ST, url);
        const feats = Object.fromEntries(Object.entries(biopsy.cells.features).map(([k, f]) => [k, {...f, contribution: null}]));
        const unused = A.toView({...biopsy, cells: {...biopsy.cells, features: feats}, _fetched_ms: 0}, 0, cfg, null);
        out.unused = C.cellsTab(unused, ST, url);
    """)
    for html in (out["mixed"], out["unused"]):
        assert html.count('class="pct-ticks"') == 1                     # one shared axis, not one per row
        assert html.count("◆ measured") == 1 and 'data-testid="cells-features-note"' in html
    # every feature unused by the model: said once in the note, not on each row
    assert out["unused"].count("not used by the prediction model") == 1 and "cf-use" not in out["unused"]
    assert "cf-nouse" in out["unused"]
    # some features feed the concept model: the per-row model-use column stays
    assert "concept-model push" in out["mixed"] and "cf-nouse" not in out["mixed"]
    assert out["mixed"].count('class="cf-use') == out["mixed"].count('data-testid="cells-feature-')


def test_prediction_card_has_one_set_callout():
    out = run_node("""
        const v = A.toView({...biopsy, _fetched_ms: 0}, 0, cfg, null);
        const metrics = (cov) => ({eval: {metrics: {models: {production: v.model}},
          conformal: {drives_tiers: 'aps_marginal_nonempty', n: 132, target_coverage: 0.9,
                      methods: {aps_marginal_nonempty: {coverage: {estimate: cov}}}}}});
        out.met = W.howSure(v, cfg, metrics(0.91));
        out.missed = W.howSure(v, cfg, metrics(0.674));
    """)
    for html in (out["met"], out["missed"]):
        box = html.split('data-testid="set-callout"', 1)[1].split("</div>", 1)[0]
        assert html.count('data-testid="set-callout"') == 1
        assert 'data-testid="set-statement"' in box and "exchangeable cases" in box
        assert 'data-testid="coverage-validation"' in box
    assert 'hs-callout warn' in out["missed"] and "not achieved" in out["missed"]
    assert 'hs-callout warn' not in out["met"]
