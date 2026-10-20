"""Browser-free checks (node) of the robustness fixes in the case decision panel, the OR board and the TCGA view:
decisions are disabled while offline, a stale order editor closes, frozen sections are ordered by real arrival, and
a worklist id under #/tcga/ links to its case page."""

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
    # case_ui.js registers window/document listeners at import: minimal stubs, then the modules
    script = textwrap.dedent(f"""
        globalThis.window = {{ addEventListener() {{}}, matchMedia: () => ({{ matches: false }}) }};
        globalThis.document = {{ addEventListener() {{}}, querySelector: () => null, querySelectorAll: () => [] }};
        const A = await import({json.dumps((APP / 'js' / 'adapter.js').as_uri())});
        const AG = await import({json.dumps((APP / 'js' / 'views' / 'agent.js').as_uri())});
        const OR = await import({json.dumps((APP / 'js' / 'views' / 'surgeon.js').as_uri())});
        const T = await import({json.dumps((APP / 'js' / 'views' / 'tcga.js').as_uri())});
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


def buttons(html: str) -> dict[str, bool]:
    """data-testid -> disabled, for every button in the HTML."""
    out = {}
    for tag in re.findall(r"<button\b[^>]*>", html):
        testid = re.search(r'data-testid="([^"]+)"', tag)
        if testid:
            out[testid.group(1)] = bool(re.search(r"\sdisabled(\s|>|$)", tag))
    return out


DRAFTED = """
    const drafted = (allowed) => A.toView({...biopsy, _fetched_ms: 0, runtime: {allowed_actions: allowed,
        decision: {signed_out_at: null, margin_call: null}}}, 0, cfg, null);
    const state = (extra) => ({busy: false, fail: 0, editing: null, editDraft: {}, events: {}, ...extra});
"""


def test_decision_buttons_are_disabled_while_offline():
    out = run_node(DRAFTED + """
        const v = drafted(['approve_orders', 'edit_orders', 'closer_look', 'dismiss_flag']);
        out.online = AG.agentPanel(v, state({fail: 1}));            // one failed poll: not offline yet
        out.offline = AG.agentPanel(v, state({fail: 2}));
        const signable = drafted(['sign_out']);
        out.signOnline = AG.agentPanel({...signable, agent: {...signable.agent, status: 'approved'}}, state({}));
        out.signOffline = AG.agentPanel({...signable, agent: {...signable.agent, status: 'approved'}}, state({fail: 3}));
    """)
    online, offline = buttons(out["online"]), buttons(out["offline"])
    acts = ["action-approve", "action-edit", "action-closer", "action-dismiss"]
    assert all(online[a] is False for a in acts), online
    assert all(offline[a] is True for a in acts), offline
    assert buttons(out["signOnline"])["action-signout"] is False
    assert buttons(out["signOffline"])["action-signout"] is True
    assert "Offline: reconnecting to Sepia" in out["signOffline"]


def test_order_editor_closes_once_editing_is_no_longer_allowed():
    out = run_node(DRAFTED + """
        const v = drafted(['approve_orders', 'edit_orders', 'closer_look', 'dismiss_flag']);
        const seq = v.agent.orders.length ? 0 : null;
        const items = v.agent.orders.length ? v.agent.orders[0].items : [];
        const live = state({editing: 'orders', editDraft: {[seq]: [...items]}});
        out.editing = AG.agentPanel(v, live);
        out.liveState = live.editing;
        // another tab approved meanwhile: the poll brings the new status and no edit_orders in allowed_actions
        const approved = drafted(['sign_out']);
        const stale = state({editing: 'orders', editDraft: {[seq]: [...items]}});
        out.approved = AG.agentPanel({...approved, agent: {...approved.agent, status: 'approved'}}, stale);
        out.staleState = [stale.editing, Object.keys(stale.editDraft).length];
        out.nOrders = v.agent.orders.length;
    """)
    assert out["nOrders"] > 0
    assert 'data-testid="edit-item"' in out["editing"] and out["liveState"] == "orders"
    assert 'data-testid="edit-item"' not in out["approved"]
    assert 'data-testid="action-edit-save"' not in out["approved"]
    assert out["staleState"] == [None, 0]


def test_or_board_orders_same_second_sections_by_real_arrival():
    out = run_node("""
        const now = Date.parse('2026-10-20T10:00:00Z');
        // three sections injected while the replay clock was held: one simulator second, different wall times
        const card = (id, at, elapsed) => ({...surg, case_id: id, lane: 'surgical', _fetched_ms: now, runtime: {
            or_room: 'OR-1', arrived_sim_s: 7200, arrived_at: at, done: false, allowed_actions: ['margin_call'],
            decision: {signed_out_at: null, margin_call: null},
            timer: {kind: 'surgical', clock: 'wall', elapsed_s: elapsed, state: 'ok', amber_s: 900, red_s: 1200}}});
        const ST = {items: [card('SURG_A', '2026-10-20T09:59:00.000Z', 60), card('SURG_C', '2026-10-20T09:58:00.000Z', 120),
                            card('SURG_B', '2026-10-20T09:59:30.000Z', 30)],
                    routed: {surgeon: []}, freshCalls: new Set(), cfg, sim: {running: true, speed: 1, held: true}};
        const html = OR.render(ST, {now});
        out.order = [...html.matchAll(/data-testid="or-card" data-case-id="([^"]+)"/g)].map(m => m[1]);
        // no wall-clock arrival: the longest-running timer first
        const noWall = (id, elapsed, sim) => ({id, arrivedAt: null, arrivedSim: sim, timer: {elapsedS: elapsed}});
        out.fallback = [noWall('X', 10, 5), noWall('Y', 90, 5), noWall('Z', 40, 5)].sort(OR.byArrival).map(v => v.id);
        const wall = (id, at, sim) => ({id, arrivedAt: at, arrivedSim: sim, timer: {elapsedS: 0}});
        out.tie = [wall('Q', '2026-10-20T09:00:00Z', 20), wall('P', '2026-10-20T09:00:00Z', 10)].sort(OR.byArrival)
            .map(v => v.id);
    """)
    assert out["order"] == ["SURG_C", "SURG_A", "SURG_B"]
    assert out["fallback"] == ["Y", "Z", "X"]
    assert out["tie"] == ["P", "Q"]                               # same wall time: simulator time decides


def test_or_board_shows_no_model_read_before_the_pathologists_call():
    out = run_node("""
        const now = Date.parse('2026-10-20T10:00:00Z');
        const runtime = (done) => ({or_room: 'OR-1', arrived_sim_s: 7200, arrived_at: '2026-10-20T09:59:00.000Z', done,
            allowed_actions: done ? [] : ['margin_call'], decision: {signed_out_at: null, margin_call: null},
            timer: {kind: 'surgical', clock: 'wall', elapsed_s: 74, state: done ? 'stopped' : 'ok', amber_s: 900, red_s: 1200}});
        const open = {...surg, case_id: 'SURG_A', lane: 'surgical', _fetched_ms: now, runtime: runtime(false)};
        const base = {freshCalls: new Set(), cfg, sim: {running: true, speed: 1}};
        out.before = OR.render({...base, items: [open], routed: {surgeon: []}}, {now});
        const call = {case_id: 'SURG_A', or_room: 'OR-1', call: 'clear', margins: [], at: '2026-10-20T10:00:00Z',
                      timer_elapsed_s: 521, suggested_call: 'tumor_at_margin',
                      report: {margin: {suggested_call: 'tumor_at_margin', involved: ['superior']}}};
        out.after = OR.render({...base, items: [{...open, runtime: runtime(true)}], routed: {surgeon: [call]}}, {now});
    """)
    before, after = out["before"], out["after"]
    # an open frozen section: timer and "awaiting" only, no model margin read and no model flag
    assert 'data-testid="or-awaiting"' in before
    for text in ("Model read", "Model suggestion", "Tumor at margin (", "reliability flag"):
        assert text not in before, text
    assert "Target 20 min" in before and "target 20:00" not in before
    assert '<span class="or-clock-l">elapsed</span>' in before
    # after the call: the pathologist's call leads, the model's suggestion follows as secondary text
    assert after.index('data-testid="surgeon-result"') < after.index('data-testid="or-model-suggestion"')
    assert "<summary>Model suggestion</summary><span>Tumor at margin (sup)" in after and "Model read" not in after
    assert '<details class="or-ai" data-testid="or-model-suggestion">' in after          # folded until opened
    assert "the pathologist’s call differs" in after
    assert '<b class="timer stopped" data-testid="or-turnaround">8 min 41 s</b>' in after


def test_inbox_leads_with_the_pathologists_sign_out():
    out = run_node(f"""
        const ON = await import({json.dumps((APP / 'js' / 'views' / 'oncologist.js').as_uri())});
        const r = {{case_id: 'demo_0001', at: '2026-10-20T10:00:00Z', agent_status: 'approved',
                   orders: [{{type: 'stain', items: ['ER']}}], summary: biopsy.summary,
                   suggestion: {{classes: biopsy.prediction.classes, probs: biopsy.prediction.probs,
                                conformal_set: ['InSitu', 'Invasive']}}, report: {{top_class: 'InSitu', features: []}}}};
        const metrics = {{eval: {{conformal: {{drives_tiers: 'aps', n: 132, methods: {{aps: {{coverage: {{estimate: 0.917}}}}}}}}}}}};
        out.html = ON.render({{routed: {{oncologist: [r]}}, inboxRead: new Set(), cfg, metrics}});
    """)
    html = out["html"]
    # the sign-out (time and the pathologist's orders) comes before any model output
    assert html.index('data-testid="inbox-signed-out"') < html.index('data-testid="inbox-signout"')
    assert html.index('data-testid="inbox-signout"') < html.index("Decision support") < html.index('data-testid="inbox-category"')
    assert "Signed out by pathologist · " in html
    assert html.index('data-testid="inbox-order"') < html.index("Decision support")
    # method detail sits behind the disclosure, not in the report body
    body, method = html.split('data-testid="inbox-method"', 1)
    for detail in ("conformal set · 90% target", "held-out coverage 91.7% (n = 132)", "BACH"):
        assert detail not in body and detail in method, detail
    assert "Suggested category" in body and body.endswith('<details class="rep-method" ')


def test_tcga_not_found_links_a_worklist_id_to_its_case_page():
    out = run_node("""
        const page = (id) => T.render({caseId: id, notFound: id, cases: {}, tcga: null, metrics: null}, {});
        out.bach = page('BACH_0225');
        out.variant = page('BACH_0141_FSIM_blur3');
        out.comp = page('COMP_009');
        out.tcga = page('TCGA-A7-A4SD_ffpe_tumor_99');
    """)
    for key, cid in (("bach", "BACH_0225"), ("variant", "BACH_0141_FSIM_blur3"), ("comp", "COMP_009")):
        assert f'href="#/case/{cid}"' in out[key] and 'data-testid="worklist-case-link"' in out[key], key
        assert "is not in the TCGA external set" in out[key]
    assert "worklist-case-link" not in out["tcga"] and "is not in the TCGA external set" in out["tcga"]
