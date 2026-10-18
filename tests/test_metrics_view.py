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
