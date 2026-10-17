// Team 44 triage worklist, connected to the PathQueue API (docs/API.md). Original app by adas286 (Team 44);
// extended by Agent 5: live data layer (js/api.js, ?mock=1 js/mock.js), adapter (js/adapter.js), views (js/views/).
// Keeps the teammate's pattern: one global state ST, string templates rendered into #app, event delegation.
// Research prototype — not for clinical use. Ground truth is never fetched or rendered.
import {createApi, ApiError} from './js/api.js';
import {esc, fingerprint, mmss, timerNow} from './js/adapter.js';
import * as worklist from './js/views/worklist.js';
import * as biopsy from './js/views/case_biopsy.js';
import * as surgical from './js/views/case_surgical.js';
import * as surgeon from './js/views/surgeon.js';
import * as oncologist from './js/views/oncologist.js';
import * as metrics from './js/views/metrics.js';
import * as spotcheck from './js/views/spotcheck.js';

const params = new URLSearchParams(location.search);
const ST = {
  route: 'worklist', caseId: null, order: 'ai', lane: 'all', mock: params.get('mock') === '1', cfg: null, sim: null,
  items: [], cases: {}, events: {}, routed: {surgeon: [], oncologist: []}, metrics: null, spot: null, spotDraft: {},
  seen: new Set(), firstLoad: true, fail: 0, staleSince: null, editing: null, editDraft: {}, marginPick: [],
  hm: {sel: 'risk', hl: [], opacity: 0.6, why: null}, tab: 'evidence', cellsOverlay: true, cellsTile: null,
  notFound: null, flashTarget: null,
};
window.__PQ_ST__ = ST;     // debugging + e2e introspection (no secrets or ground truth live here)
let api = null;
let lastFp = null;
let pollTimer = null;
let pollSeq = 0;           // a response older than the latest request is dropped (lane/sort clicks race the timer)
let dragging = false;      // heatmap opacity slider held: do not re-render under it

const $ = sel => document.querySelector(sel);
const ctx = () => ({now: Date.now(), assetUrl: p => api.assetUrl(p)});

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 3500);
}

// ---- routing --------------------------------------------------------------------------------------------------------------
function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  const [route, id] = h.split('/');
  if (route === 'case' && id) return {route: 'case', caseId: decodeURIComponent(id)};
  return {route: ['worklist', 'surgeon', 'oncologist', 'metrics', 'spotcheck'].includes(route) ? route : 'worklist', caseId: null};
}
function go(hash) { if (location.hash !== hash) location.hash = hash; else onRoute(); }
function onRoute() {
  const r = parseHash();
  const changed = r.route !== ST.route || r.caseId !== ST.caseId;
  Object.assign(ST, r);
  if (changed) {
    ST.editing = null; ST.notFound = null; ST.tab = 'evidence'; ST.cellsTile = null;
    ST.hm = {sel: 'risk', hl: [], opacity: ST.hm.opacity, why: null};
    window.scrollTo(0, 0);
  }
  document.querySelectorAll('nav [data-nav]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.nav === (ST.route === 'case' ? 'worklist' : ST.route))));
  lastFp = null;
  draw();
  poll(true);
}

// ---- data -----------------------------------------------------------------------------------------------------------------
async function loadCase(id) {
  try {
    ST.cases[id] = await api.getCase(id);
    ST.events[id] = await api.events(id);
    ST.notFound = null;
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) ST.notFound = id; else throw err;
  }
}

async function refresh(force, seq) {
  const wl = await api.worklist({lane: ST.route === 'surgeon' ? 'surgical' : ST.lane, order: ST.order});
  if (seq !== pollSeq) return;
  ST.items = wl.items || [];
  ST.sim = wl.sim || ST.sim;
  ST.synthetic = wl.synthetic || ST.items.some(c => c.synthetic);
  if (ST.route === 'case' && ST.caseId) {
    await loadCase(ST.caseId);       // every poll: live tier (aging), timers and decisions; local and cheap
  } else if (ST.route === 'surgeon') {
    ST.routed.surgeon = await api.routed('surgeon');
  } else if (ST.route === 'oncologist') {
    ST.routed.oncologist = await api.routed('oncologist');
  } else if (ST.route === 'metrics' && (force || !ST.metrics)) {
    ST.metrics = await api.metrics();
  } else if (ST.route === 'spotcheck' && (force || !ST.spot)) {
    ST.spot = await api.spotcheckItems();
  }
}

async function poll(force = false) {
  clearTimeout(pollTimer);
  if (document.hidden && !force) { pollTimer = setTimeout(poll, ST.cfg ? ST.cfg.poll_ms : 2000); return; }
  const seq = ++pollSeq;
  try {
    await refresh(force, seq);
    if (ST.fail >= 2) toast('Backend reachable again');
    ST.fail = 0; ST.staleSince = null;
  } catch (err) {
    ST.fail += 1;
    if (!ST.staleSince) ST.staleSince = new Date();
    if (!(err instanceof ApiError) || err.status === 0 || err.status >= 500) console.warn('poll failed', err);
  }
  showOffline();
  draw();
  const base = ST.cfg ? ST.cfg.poll_ms : 2000;
  pollTimer = setTimeout(poll, ST.fail >= 5 ? base * 2.5 : base);
}

function showOffline() {
  const el = $('#offline');
  if (ST.fail >= 2) {
    el.hidden = false;
    el.innerHTML = `Backend unreachable at ${esc(api.base || location.origin)}. Retrying… Start it with <code>make demo</code>, or open with <code>?mock=1</code>. Data on screen is stale since ${esc(ST.staleSince.toLocaleTimeString())}.`;
  } else el.hidden = true;
}

// ---- rendering ------------------------------------------------------------------------------------------------------------
function view() {
  if (ST.route === 'case') {
    if (ST.notFound === ST.caseId) return `<button class="back" id="bk" data-testid="back">Back to worklist</button><div class="card" data-testid="case-not-found"><h2>Case not found</h2><p>${esc(ST.caseId)} is unknown or has not arrived yet.</p></div>`;
    const c = ST.cases[ST.caseId];
    if (!c) return '<div class="card" data-testid="state-loading"><h2>Loading case…</h2></div>';
    return (c.lane === 'surgical' ? surgical : biopsy).render(ST, ctx());
  }
  return {worklist, surgeon, oncologist, metrics, spotcheck}[ST.route].render(ST, ctx());
}

function currentFp() {
  const sim = ST.sim || {};
  const parts = [ST.route, ST.caseId, ST.order, ST.lane, ST.tab, ST.editing, ST.notFound, ST.busy, fingerprint(ST.items),
                 sim.running, JSON.stringify(sim.pending),
                 JSON.stringify([ST.hm.sel, ST.hm.hl, ST.hm.why]), ST.cellsOverlay, ST.cellsTile, JSON.stringify(ST.marginPick), JSON.stringify(ST.editDraft)];
  if (ST.route === 'case') {
    const c = ST.cases[ST.caseId];
    parts.push(c ? JSON.stringify([c.agent, c.runtime && [c.runtime.decision, c.runtime.orders, c.runtime.tier,
                                                          c.runtime.reasons, c.runtime.timer && c.runtime.timer.state]]) : '');
    parts.push((ST.events[ST.caseId] || []).length);
  }
  if (ST.route === 'surgeon') parts.push(JSON.stringify(ST.routed.surgeon.map(r => r.id)));
  if (ST.route === 'oncologist') parts.push(JSON.stringify(ST.routed.oncologist.map(r => r.id)));
  if (ST.route === 'metrics') parts.push(ST.metrics ? Object.keys(ST.metrics.eval || {}).join() + (ST.metrics.figures || []).length : 'none');
  if (ST.route === 'spotcheck') parts.push(ST.spot ? ST.spot.items.length : 'none', JSON.stringify(ST.spotDraft), ST.spotResult || '');
  return parts.join('|');
}

function draw(forceRender = false) {
  if (!ST.cfg) return;
  $('[data-testid=banner-mock]').hidden = !(ST.mock || ST.synthetic);
  tickClock();
  const active = document.activeElement;
  const typing = active && $('#app').contains(active) && /^(INPUT|SELECT|TEXTAREA)$/.test(active.tagName) && active.type !== 'checkbox' && active.type !== 'range';
  const fp = currentFp();
  if (!forceRender && (fp === lastFp || typing || dragging)) { tick(); return; }
  lastFp = fp;
  const y = window.scrollY;
  $('#app').innerHTML = view();
  window.scrollTo(0, y);
  if (ST.route === 'worklist') {
    ST.items.forEach(c => ST.seen.add(c.case_id));
    ST.firstLoad = false;
  }
  if (ST.flashTarget) {
    const el = $(ST.flashTarget);
    if (el) { el.scrollIntoView({block: 'center', behavior: 'smooth'}); el.classList.add('flash'); }
    ST.flashTarget = null;
  }
  tick();
}

function lookup(id) {
  if (ST.route === 'case' && ST.cases[id]) return ST.cases[id];
  return ST.items.find(c => c.case_id === id);
}

function tick() {
  const now = Date.now();
  document.querySelectorAll('[data-timer]').forEach(el => {
    const c = lookup(el.dataset.timer);
    if (!c) return;
    const t = timerNow(c, now, ST.cfg, ST.sim);
    const text = el.dataset.timerMode === 'countdown' ? (t.state === 'stopped' ? 'Called' : t.countdown.replace(' left', '')) : t.text;
    if (el.textContent !== text) el.textContent = text;
    el.dataset.state = t.state;
    el.className = el.className.replace(/\b(ok|amber|red|stopped|none)\b/g, '').trim() + ' ' + t.state;
  });
  document.querySelectorAll('[data-timer-sub]').forEach(el => {
    const c = lookup(el.dataset.timerSub);
    if (c) el.textContent = timerNow(c, now, ST.cfg, ST.sim).countdown || '';
  });
  document.querySelectorAll('[data-timer-bar]').forEach(el => {
    const c = lookup(el.dataset.timerBar);
    if (!c) return;
    const t = timerNow(c, now, ST.cfg, ST.sim);
    if (t.usedFrac !== null) el.style.width = Math.round(t.usedFrac * 100) + '%';
    el.className = t.state === 'none' ? '' : t.state;
  });
  tickClock();
}

function tickClock() {
  const sim = ST.sim;
  if (!sim) return;
  const since = sim._fetched_ms && sim.running ? (Date.now() - sim._fetched_ms) / 1000 * sim.speed : 0;
  $('#sim').textContent = `simulator +${mmss(sim.sim_time_s + since)} · ×${sim.speed} · ${sim.running ? 'running' : 'paused'}${ST.mock ? ' · mock' : ''}`;
  const pp = $('#pp');
  pp.setAttribute('aria-pressed', String(!sim.running));
  const pending = sim.pending ? sim.pending.surgical : null;
  if (pending) {
    document.querySelectorAll('[data-sim=inject_surgical]').forEach(b => {
      const n = pending[b.dataset.variant] || 0;
      b.disabled = n === 0;
      b.title = n ? `${n} pending` : 'none left (reset to replay)';
    });
  }
}

// ---- actions ----------------------------------------------------------------------------------------------------------------
async function act(fn, okMsg) {
  try {
    const res = await fn();
    if (okMsg) toast(okMsg);
    return res;
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) toast(`Not allowed: ${err.detail}`);
    else if (err instanceof ApiError && err.status === 422) toast(`Rejected: ${err.detail}`);
    else toast(`Request failed: ${err.message || err}`);
    return null;
  } finally {
    await poll(true);
  }
}

async function decide(action, payload) {
  if (ST.busy) return;                  // no double submits (would only produce a confusing 409)
  const id = ST.caseId;
  ST.busy = true; draw(true);
  const res = await act(() => api.decide(id, action, payload));
  ST.busy = false;
  if (res && res.case) { ST.cases[id] = res.case; ST.editing = null; }   // a rejected edit keeps the editor open
  draw(true);
}

async function simCommand(cmd, variant) {
  const res = await act(() => api.sim(cmd, cmd === 'inject_surgical' ? {variant} : undefined));
  if (res && cmd === 'inject_surgical' && res.injected) toast(`Simulated frozen section ${res.injected} received`);
  if (res && cmd === 'reset') {
    ST.cases = {}; ST.events = {}; ST.routed = {surgeon: [], oncologist: []}; ST.seen = new Set(); ST.firstLoad = true;
    if (ST.route === 'case') go('#/worklist');
  }
}

function findingClick(fid) {
  const c = ST.cases[ST.caseId];
  const f = c && (c.findings || []).find(x => x.id === fid);
  if (!f) return;
  if (f.kind === 'concept') {
    if ((c.tiles || []).some(t => t.concepts && f.data.concept in t.concepts)) ST.hm.sel = f.data.concept;
    ST.hm.hl = f.data.tiles || []; ST.hm.why = f.data.concept; ST.flashTarget = '[data-testid=heatmap]';
  } else if (f.kind === 'cells') {
    ST.tab = 'cells'; ST.cellsTile = (f.data.tiles || [])[0] ?? null;
    ST.flashTarget = ST.cellsTile !== null ? `[data-testid=cells-tile][data-tile="${ST.cellsTile}"]` : '[data-testid=cells-tab]';
  } else if (f.kind === 'prediction') ST.flashTarget = '[data-testid=how-sure]';
  else if (f.kind === 'retrieval') ST.flashTarget = '[data-testid=compared]';
  else if (f.kind === 'quality') ST.flashTarget = '[data-testid=how-sure]';
  else if (f.kind === 'margin') ST.flashTarget = '[data-testid=margin-labels]';
  draw(true);
}

function whyClick(id) {
  const c = ST.cases[ST.caseId];
  const concept = (c.concepts || []).find(k => k.id === id);
  const hasTileValues = (c.tiles || []).some(t => t.concepts && id in t.concepts);
  ST.hm.why = id;
  if (concept) {
    if (hasTileValues) ST.hm.sel = id;
    ST.hm.hl = concept.top_tiles || [];
  } else if (c.cells && c.cells.tiles) ST.hm.hl = c.cells.tiles.map(t => t.idx);
  draw(true);
}

document.addEventListener('click', async e => {
  const t = e.target.closest('button, [data-nav], tr.row, a');
  if (!t) return;
  const d = t.dataset;
  if (d.nav) { e.preventDefault(); go('#/' + d.nav); return; }
  if (d.sim) { await simCommand(d.sim, d.variant); return; }
  if (!$('#app').contains(t)) return;
  if (d.sort) { ST.order = d.sort; await poll(true); return; }
  if (d.laneFilter) { ST.lane = d.laneFilter; await poll(true); return; }
  if (t.id === 'bk') { go('#/worklist'); return; }
  if (t.matches('tr.row')) { go('#/case/' + encodeURIComponent(t.dataset.caseId)); return; }
  if (d.tab) { ST.tab = d.tab; draw(true); return; }
  if (d.f) { findingClick(d.f); return; }
  if (d.why) { whyClick(d.why); return; }
  if (d.act) {
    if (d.act === 'edit_orders') {
      const c = ST.cases[ST.caseId];
      ST.editDraft = Object.fromEntries(((c.runtime && c.runtime.orders) || []).map(o => [o.rule_id, [...o.items]]));
      ST.editing = 'orders'; draw(true); return;
    }
    if (d.act === 'edit_cancel') { ST.editing = null; draw(true); return; }
    if (d.act === 'edit_save') {
      const orders = Object.entries(ST.editDraft).map(([rule_id, items]) => ({rule_id, items}));
      await decide('edit_orders', {orders}); return;
    }
    await decide(d.act, {}); return;
  }
  if (d.call) {
    if (d.call === 'tumor') {
      const c = ST.cases[ST.caseId];
      ST.marginPick = [...((c.margin && c.margin.involved) || [])];
      ST.editing = 'margins'; draw(true); return;
    }
    if (d.call === 'cancel') { ST.editing = null; draw(true); return; }
    if (d.call === 'confirm') {
      if (!ST.marginPick.length) { toast('Pick at least one margin'); return; }
      await decide('margin_call', {call: 'tumor_at_margin', margins: ST.marginPick}); return;
    }
    await decide('margin_call', {call: d.call}); return;
  }
  if (d.spot === 'submit') { await spotcheck.submit(ST, api, toast); draw(true); return; }
  if (d.metrics === 'refresh') { ST.metrics = null; await poll(true); }
});

document.addEventListener('keydown', e => {
  const r = e.target.closest && e.target.closest('tr.row');
  if (r && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); go('#/case/' + encodeURIComponent(r.dataset.caseId)); }
});

document.addEventListener('change', e => {
  const t = e.target;
  const d = t.dataset;
  if (d.hm === 'sel') { ST.hm.sel = t.value; ST.hm.why = null; ST.hm.hl = []; draw(true); }
  else if (d.editRule) {
    const list = ST.editDraft[d.editRule] || (ST.editDraft[d.editRule] = []);
    if (t.checked && !list.includes(d.editItem)) list.push(d.editItem);
    if (!t.checked) ST.editDraft[d.editRule] = list.filter(i => i !== d.editItem);
  } else if (d.pick) {
    ST.marginPick = t.checked ? [...new Set([...ST.marginPick, d.pick])] : ST.marginPick.filter(m => m !== d.pick);
  } else if (d.cells === 'overlay') { ST.cellsOverlay = t.checked; draw(true); }
  else if (d.spotVerdict) { ST.spotDraft[d.spotItem] = d.spotVerdict; draw(true); }
  else if (t.id === 'spot-kind') ST.spotKind = t.value;
});

document.addEventListener('input', e => {
  if (e.target.id === 'spot-reviewer') ST.spotReviewer = e.target.value;
  if (e.target.dataset.hm === 'opacity') {      // live update without a re-render (keeps the slider drag)
    ST.hm.opacity = parseFloat(e.target.value);
    document.querySelectorAll('.hm-tile').forEach(r => r.setAttribute('fill-opacity', (parseFloat(r.dataset.v) * ST.hm.opacity).toFixed(3)));
  }
});

document.addEventListener('pointerdown', e => { if (e.target.dataset && e.target.dataset.hm === 'opacity') dragging = true; });
document.addEventListener('pointerup', () => { if (dragging) { dragging = false; draw(); } });
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(true); });
window.addEventListener('hashchange', onRoute);

// ---- boot -------------------------------------------------------------------------------------------------------------------
async function boot() {
  try {
    if (ST.mock) {
      const {createMockApi} = await import('./js/mock.js');
      api = await createMockApi();
    } else {
      api = createApi({base: params.get('api') || ''});
    }
    ST.cfg = await api.config();
    $('[data-testid=banner-research]').textContent = ST.cfg.banner;
  } catch (err) {
    $('#app').innerHTML = `<div class="card" data-testid="state-offline-boot"><h2>Cases did not load</h2><p>The backend did not answer (${esc(err.message || err)}). Start it with <code>make demo</code>, or open this page with <code>?mock=1</code> to use the committed synthetic fixtures. Opening index.html directly blocks the fetch; serve over HTTP (see README).</p></div>`;
    setTimeout(boot, 3000);
    return;
  }
  setInterval(tick, 1000);
  onRoute();
}
boot();
