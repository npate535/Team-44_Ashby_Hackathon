"""Browser-free checks of the connected teammate frontend (Task 7): mock config in sync with the backend, contract
typedef coverage, no external network references, legacy mock untouched, and the pure adapter (via node, if present)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import textwrap

import pytest

from backend.app import create_app
from core import config
from tests.app_helpers import CFG, FIXTURES

APP = config.REPO_ROOT / "team44-triage-frontend" / "app"
LEGACY_SHA256 = "bbb2d4fac51d55cc362cb49ed99fa83133aed2fed2c96e025d558bd5059bb6dc"   # data/cases.json at b309f0b


def _sources():
    return [p for p in APP.rglob("*") if p.suffix in (".js", ".html", ".css")]


def test_mock_config_matches_backend_config(tmp_path):
    app = create_app(cfg=CFG, cases_dir=FIXTURES, artifacts_root=tmp_path, db_path=tmp_path / "x.sqlite",
                     frontend_dir=None, env={})
    live = app.state.backend.config_payload()
    mock = json.loads((APP / "mock" / "config.json").read_text())
    live.pop("synthetic"), mock.pop("synthetic")
    assert mock == live, "regenerate team44-triage-frontend/app/mock/config.json from GET /api/config"


def test_mock_manifest_points_at_fixtures():
    manifest = json.loads((APP / "mock" / "manifest.json").read_text())
    for entry in manifest["cases"]:
        name = entry["src"].split("/")[-1]
        assert (FIXTURES / name).is_file(), entry["src"]
    assert manifest["asset_base"] == "../../fixtures/"


def test_types_js_mirrors_every_contract_type():
    ts = (config.REPO_ROOT / "frontend" / "src" / "types.ts").read_text()
    js = (APP / "js" / "types.js").read_text()
    names = re.findall(r"export (?:interface|type) (\w+)", ts)
    assert names
    missing = [n for n in names if not re.search(rf"@typedef \{{.*\}} {n}\b", js)]
    assert missing == []


def test_no_external_network_references():
    pattern = re.compile(r"""(src|href)\s*=\s*["']https?://|url\(\s*["']?https?://|from\s+["']https?://|"""
                         r"""import\(\s*["']https?://|fetch\(\s*["']https?://|@import""")
    hits = [f"{p.relative_to(APP)}: {m.group(0)}" for p in _sources() for m in pattern.finditer(p.read_text())]
    assert hits == []


def test_ground_truth_is_never_read_for_display():
    # only the defensive strips in the adapter and the mock may name eval_only; true_label appears nowhere
    mentions = {p.relative_to(APP).as_posix() for p in _sources() if "eval_only" in p.read_text()}
    assert mentions <= {"js/adapter.js", "js/mock.js", "js/types.js"}   # types.js: header comment only
    assert not any("true_label" in p.read_text() for p in _sources())


def test_legacy_mock_kept_untouched():
    data = (APP / "data" / "legacy_mock.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == LEGACY_SHA256
    assert "legacy_mock" not in "".join(p.read_text() for p in _sources() if p.suffix == ".js")


def test_wording_rules_on_user_facing_text():
    text = " ".join(p.read_text() for p in _sources()).lower()
    for banned in ("diagnosed as", "the model diagnoses", "this cell caused", "nottingham grade:"):
        assert banned not in text
    assert "research prototype" in (APP / "index.html").read_text().lower()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_adapter_logic_with_node():
    adapter = (APP / "js" / "adapter.js").as_uri()
    fixture = json.loads((FIXTURES / "demo_0001.json").read_text())
    surgical = json.loads((FIXTURES / "demo_0002.json").read_text())
    cfg = json.loads((APP / "mock" / "config.json").read_text())
    script = textwrap.dedent(f"""
        import * as A from {json.dumps(adapter)};
        const cfg = {json.dumps(cfg)};
        const biopsy = {json.dumps(fixture)};
        const surg = {json.dumps(surgical)};
        const out = {{}};
        const now = 1000000;
        const v = A.toView({{...biopsy, _fetched_ms: now}}, now, cfg, {{running: false, speed: 30}});
        out.risk = v.risk; out.tier = v.tier; out.tierLabel = v.tierLabel; out.set = v.set;
        out.evalOnlyGone = !('eval_only' in A.stripGroundTruth(biopsy));
        out.marginNull = v.margin; out.cells = v.cells.status;
        out.whyFirst = v.why[0].id; out.measured = v.why.filter(r => r.measured).map(r => r.id);
        const timer = s => A.timerNow({{lane: 'surgical', _fetched_ms: now, runtime: {{timer: {{kind: 'surgical',
            elapsed_s: s, state: 'ok', amber_s: cfg.timer.amber_min * 60, red_s: cfg.timer.red_min * 60}}}}}},
            now, cfg, null).state;
        out.states = [timer(899), timer(900), timer(1199), timer(1200)];
        out.stopped = A.timerNow({{_fetched_ms: now, runtime: {{timer: {{kind: 'surgical', elapsed_s: 30,
            state: 'stopped'}}}}}}, now + 60000, cfg, null);
        out.biopsyRun = A.timerNow({{_fetched_ms: now, runtime: {{timer: {{kind: 'biopsy', elapsed_s: 60,
            target_s: 600, state: 'none'}}}}}}, now + 2000, cfg, {{running: true, speed: 30}}).elapsedS;
        const sv = A.toView({{...surg, cells: undefined}}, now, cfg, null);
        out.surgMargin = sv.margin.involved; out.surgCells = sv.cells.status; out.surgTier = sv.tierLabel;
        out.unknownReason = A.reasonLabel('brand_new', cfg);
        out.esc = A.esc('<b onclick="x">&</b>');
        out.allowed = A.allowedActions({{lane: 'biopsy', agent: {{status: 'drafted'}}}});
        out.mmss = [A.mmss(65), A.mmss(3725)];
        console.log(JSON.stringify(out));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["risk"] == 72 and out["tier"] == 2 and out["tierLabel"] == CFG["ui"]["tier_labels"][2]
    assert out["set"] == ["InSitu", "Invasive"] and out["evalOnlyGone"] is True
    assert out["marginNull"] is None and out["cells"] == "ok"
    assert out["whyFirst"] == "invasion"
    assert set(out["measured"]) == {"nuc_area_cv", "neo_density", "til_proximity"}
    assert out["states"] == ["ok", "amber", "amber", "red"]
    assert out["stopped"]["state"] == "stopped" and out["stopped"]["elapsedS"] == 30
    assert out["biopsyRun"] == pytest.approx(60 + 2 * 30)
    assert out["surgMargin"] == ["superior"] and out["surgCells"] == "disabled"
    assert out["surgTier"] == CFG["ui"]["tier_labels"][1]
    assert out["unknownReason"] == "brand_new"
    assert out["esc"] == "&lt;b onclick=&quot;x&quot;&gt;&amp;&lt;/b&gt;"
    assert out["allowed"] == ["approve_orders", "edit_orders", "closer_look", "dismiss_flag"]
    assert out["mmss"] == ["01:05", "1:02:05"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("path", sorted(p.relative_to(APP).as_posix() for p in APP.rglob("*.js")))
def test_every_module_parses(path):
    res = subprocess.run(["node", "--check", str(APP / path)], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_coverage_display_requires_matching_model_and_valid_aggregate():
    script = f"""
    import {{coverageStatus}} from {json.dumps((APP / 'js/adapter.js').as_uri())};
    const payload = {{eval: {{metrics: {{models: {{production: 'actual_model'}}}}, conformal: {{
      drives_tiers: 'lac_marginal', n: 132, target_coverage: 0.9,
      methods: {{lac_marginal: {{coverage: {{estimate: 0.71}}}}}}
    }}}}}};
    const result = coverageStatus(payload, 'actual_model');
    if (!result.belowTarget || result.n !== 132 || result.coverage !== 0.71) throw Error('invalid coverage');
    if (coverageStatus(payload, 'another_model') !== null) throw Error('mismatched model');
    payload.eval.conformal.methods.lac_marginal.coverage.estimate = '0.71';
    if (coverageStatus(payload, 'actual_model') !== null) throw Error('invalid value accepted');
    if (coverageStatus(null, 'actual_model') !== null) throw Error('missing report accepted');
    """
    res = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_full_confidence_set_is_visible_and_uncertainty_includes_empty_sets():
    script = rf"""
    import {{setChips, filterViews}} from {json.dumps((APP / 'js/views/worklist.js').as_uri())};
    const classes = ['Normal', 'Benign', 'InSitu', 'Invasive'];
    const html = setChips({{risk: 25, setSize: 4, set: classes, probs: classes.map(cls => ({{cls, p: .25}}))}});
    const labels = [...html.matchAll(/>([^<]+)<\/span>/g)].map(m => m[1]);
    if (JSON.stringify(labels) !== JSON.stringify(['Normal', 'Benign', 'In situ', 'Invasive']))
      throw Error('A confidence category was hidden');
    const views = [0, 1, 2, 4].map(setSize => ({{setSize, agent: {{}}}}));
    const filtered = filterViews({{status: 'all', flagFilter: 'uncertain'}}, views);
    if (JSON.stringify(filtered.map(v => v.setSize)) !== '[0,2,4]')
      throw Error('Uncertainty filter must include ambiguous and empty sets');
    """
    res = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr


def test_thumbnail_widths_match_the_backend():
    # the adapter only rewrites to /thumbs/<w>/... for widths the backend and the Vercel package build
    from backend.app import STATIC_THUMB_WIDTHS
    m = re.search(r"export const THUMB_WIDTHS = \[([\d, ]+)\]", (APP / "js" / "adapter.js").read_text())
    assert m and tuple(int(x) for x in m.group(1).split(",")) == STATIC_THUMB_WIDTHS


def _static_import_graph(entry):
    """Every module `entry` needs before it runs: static `import … from './x.js'`, followed recursively (dynamic
    import() is not followed: those pages load on first use)."""
    pattern = re.compile(r"""^\s*import\s+(?:[^;]*?\sfrom\s+)?['"](\.{1,2}/[^'"]+)['"]""", re.M)
    seen, stack = set(), [entry]
    while stack:
        path = stack.pop().resolve()
        rel = path.relative_to(APP.resolve()).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        stack += [path.parent / spec for spec in pattern.findall(path.read_text())]
    return seen


def test_index_preloads_exactly_the_modules_app_js_imports():
    # a module missing from the list is fetched one import level later (a round trip each on a slow connection); an
    # extra one fetches a lazily loaded page on every first load
    needed = _static_import_graph(APP / "app.js")
    preloaded = re.findall(r'<link rel="modulepreload" href="([^"]+)"', (APP / "index.html").read_text())
    assert len(preloaded) == len(set(preloaded))
    assert set(preloaded) == needed
    lazy = {"js/views/metrics.js", "js/views/tcga.js", "js/views/spotcheck.js", "js/charts.js"}
    assert not lazy & needed, "the evaluation, TCGA and evidence-review pages load on first use"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_api_client_errors_are_short_and_say_what_happened():
    # a proxy's HTML error page is never shown as it is, long details are cut short, and a request that got no reply
    # in time is told apart from one that could not be sent (app.js reads the case again before calling it failed)
    script = textwrap.dedent(f"""
        import {{createApi, ApiError}} from {json.dumps((APP / 'js' / 'api.js').as_uri())};
        const api = createApi();
        const answer = (fn) => {{ globalThis.fetch = fn; }};
        const out = {{}};
        const err = async (call) => {{ try {{ await call(); return null; }} catch (e) {{
          return {{api: e instanceof ApiError, status: e.status, detail: e.detail, timedOut: e.timedOut}}; }} }};
        answer(async () => new Response('<html><body><h1>502 Bad Gateway</h1></body></html>', {{status: 502}}));
        out.html502 = await err(() => api.health());
        answer(async () => new Response('<!doctype html><title>Sepia</title>', {{status: 200}}));
        out.html200 = await err(() => api.health());
        answer(async () => new Response(JSON.stringify({{detail: 'x'.repeat(400)}}), {{status: 409}}));
        out.long = await err(() => api.decide('A', 'sign_out'));
        answer(async () => {{ const e = new Error('The operation was aborted'); e.name = 'AbortError'; throw e; }});
        out.timeout = await err(() => api.decide('A', 'sign_out'));
        answer(async () => {{ throw new TypeError('Failed to fetch'); }});
        out.offline = await err(() => api.decide('A', 'sign_out'));
        answer(async () => new Response(JSON.stringify({{ok: true}}), {{status: 200}}));
        out.ok = await api.health();
        console.log(JSON.stringify(out));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["html502"] == {"api": True, "status": 502, "detail": "Sepia did not respond (HTTP 502)", "timedOut": False}
    assert out["html200"]["status"] == 502 and out["html200"]["detail"] == "Sepia did not respond (HTTP 200)"
    assert out["long"]["status"] == 409 and len(out["long"]["detail"]) <= 140 and out["long"]["detail"].endswith("…")
    assert out["timeout"] == {"api": True, "status": 0, "detail": "timed out", "timedOut": True}
    assert out["offline"]["status"] == 0 and out["offline"]["timedOut"] is False
    assert out["ok"] == {"ok": True}


def test_app_decodes_case_addresses_safely():
    # a malformed % escape in the address (#/case/BACH_0225%) must not throw while the app starts
    js = (APP / "app.js").read_text()
    bare = [m.start() for m in re.finditer(r"decodeURIComponent\(", js)]
    assert len(bare) == 1, "decode route ids through decodeId (try/catch), not decodeURIComponent directly"
    assert re.search(r"const decodeId = \(s\) => \{\s*try \{\s*return decodeURIComponent\(s\);", js)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_biopsy_timers_stand_still_while_the_replay_clock_is_held():
    # sim.held: the replay clock holds after the last scheduled arrival; a biopsy timer must not run on between polls
    script = textwrap.dedent(f"""
        import {{timerNow}} from {json.dumps((APP / 'js' / 'adapter.js').as_uri())};
        const c = {{_fetched_ms: 1000, runtime: {{timer: {{kind: 'biopsy', elapsed_s: 600, target_s: 86400, state: 'none'}}}}}};
        const at = (sim) => timerNow(c, 3000, null, sim).elapsedS;   // 2 s after the poll, 30x replay speed
        console.log(JSON.stringify({{
          running: at({{running: true, speed: 30, sim_time_s: 1000}}),
          held: at({{running: true, speed: 30, sim_time_s: 1000, held: true, hold_at_s: 1000}}),
          capped: at({{running: true, speed: 30, sim_time_s: 1000, held: false, hold_at_s: 1010}}),
          paused: at({{running: false, speed: 30, sim_time_s: 1000}}),
        }}));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out == {"running": 660, "held": 600, "capped": 610, "paused": 600}
