// Team 44 triage worklist, connected to the PathQueue API (docs/API.md). Original app by adas286 (Team 44);
// extended by Agent 5: live data layer (js/api.js, ?mock=1 js/mock.js), adapter (js/adapter.js), views (js/views/).
// Keeps the teammate's pattern: one global state ST, string templates rendered into #app, event delegation.
// Research prototype — not for clinical use. Ground truth is never fetched or rendered.
import { createApi, ApiError } from './js/api.js';
import { esc, fingerprint, mmss, timerNow } from './js/adapter.js';
import * as worklist from './js/views/worklist.js';
import * as biopsy from './js/views/case_biopsy.js';
import * as surgical from './js/views/case_surgical.js';
import * as surgeon from './js/views/surgeon.js';
import * as oncologist from './js/views/oncologist.js';
import * as metrics from './js/views/metrics.js';
import * as spotcheck from './js/views/spotcheck.js';

const params = new URLSearchParams(location.search);
const ST = {
  route: 'worklist',
  caseId: null,
  order: 'ai',
  lane: 'all',
  mock: params.get('mock') === '1',
  cfg: null,
  sim: null,
  items: [],
  cases: {},
  events: {},
  routed: { surgeon: [], oncologist: [] },
  metrics: null,
  spot: null,
  spotDraft: {},
  seen: new Set(),
  firstLoad: true,
  fail: 0,
  staleSince: null,
  editing: null,
  editDraft: {},
  marginPick: [],
  hm: { sel: 'risk', hl: [], opacity: 0.6, why: null },
  tab: 'evidence',
  cellsOverlay: true,
  cellsTile: null,
  notFound: null,
  flashTarget: null,
  health: null,
  search: '',
  status: 'all',
  page: 0
};
window.__PQ_ST__ = ST; // debugging + e2e introspection (no secrets or ground truth live here)
let api = null;
let lastFp = null;
let pollTimer = null;
let pollSeq = 0; // a response older than the latest request is dropped (lane/sort clicks race the timer)
let dragging = false; // heatmap opacity slider held: do not re-render under it

const $ = (sel) => document.querySelector(sel);
const ctx = () => ({ now: Date.now(), assetUrl: (p) => api.assetUrl(p) });

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.hidden = true;
  }, 3500);
}

// ---- routing --------------------------------------------------------------------------------------------------------------
function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  const [route, id] = h.split('/');
  if (route === 'case' && id) return { route: 'case', caseId: decodeURIComponent(id) };
  return {
    route: ['worklist', 'surgeon', 'oncologist', 'metrics', 'spotcheck'].includes(route) ? route : 'worklist',
    caseId: null
  };
}
function go(hash) {
  if (location.hash !== hash) location.hash = hash;
  else onRoute();
}
function onRoute() {
  const r = parseHash();
  const changed = r.route !== ST.route || r.caseId !== ST.caseId;
  Object.assign(ST, r);
  if (changed) {
    ST.editing = null;
    ST.notFound = null;
    ST.tab = 'evidence';
    ST.cellsTile = null;
    ST.hm = { sel: 'risk', hl: [], opacity: ST.hm.opacity, why: null };
    window.scrollTo(0, 0);
  }
  document.querySelectorAll('nav [data-nav]').forEach((b) => {
    const active = b.dataset.nav === (ST.route === 'case' ? 'worklist' : ST.route);
    if (active) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  });
  lastFp = null;
  if (!api || !ST.cfg) return;
  draw();
  poll(true);
}

// ---- data -----------------------------------------------------------------------------------------------------------------
// Snapshot route and request generation: slow responses cannot overwrite a newer navigation.
async function refresh(force, seq) {
  const route = ST.route,
    caseId = ST.caseId;
  const [wl, health] = await Promise.all([
    api.worklist({ lane: route === 'surgeon' ? 'surgical' : ST.lane, order: ST.order }),
    api.health()
  ]);
  let detail = null,
    events = null,
    missing = false,
    routed = null,
    evaluation = null,
    spot = null;
  if (route === 'case' && caseId) {
    try {
      [detail, events] = await Promise.all([api.getCase(caseId), api.events(caseId)]);
      if (!ST.metrics) evaluation = await api.metrics();
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) missing = true;
      else throw err;
    }
  } else if (route === 'surgeon' || route === 'oncologist') routed = await api.routed(route);
  else if (route === 'metrics' && (force || !ST.metrics)) evaluation = await api.metrics();
  else if (route === 'spotcheck' && (force || !ST.spot)) spot = await api.spotcheckItems();
  if (seq !== pollSeq) return;
  ST.items = wl.items || [];
  ST.sim = wl.sim || ST.sim;
  ST.health = health;
  ST.synthetic = wl.synthetic || health.synthetic || ST.items.some((c) => c.synthetic);
  if (route === 'case') {
    ST.notFound = missing ? caseId : null;
    if (detail) {
      ST.cases[caseId] = detail;
      ST.events[caseId] = events;
    }
  }
  if (routed) ST.routed[route] = routed;
  if (evaluation) ST.metrics = evaluation;
  if (spot) ST.spot = spot;
}

async function poll(force = false) {
  clearTimeout(pollTimer);
  const seq = ++pollSeq;
  try {
    await refresh(force, seq);
    if (seq !== pollSeq) return;
    if (ST.fail >= 2) toast('Backend reachable again');
    ST.fail = 0;
    ST.staleSince = null;
  } catch (err) {
    if (seq !== pollSeq) return;
    ST.fail += 1;
    if (!ST.staleSince) ST.staleSince = new Date();
    if (!(err instanceof ApiError) || err.status === 0 || err.status >= 500) console.warn('poll failed', err);
  }
  showOffline();
  draw();
  const base = ST.cfg ? ST.cfg.poll_ms : 2000;
  // Background screens keep synchronizing, with a lower polling frequency.
  pollTimer = setTimeout(poll, document.hidden ? base * 5 : ST.fail >= 5 ? base * 2.5 : base);
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
    if (ST.notFound === ST.caseId)
      return `<button class="back" id="bk" data-testid="back">Back to worklist</button><div class="card" data-testid="case-not-found"><h2>Case not found</h2><p>${esc(ST.caseId)} is unknown or has not arrived yet.</p></div>`;
    const c = ST.cases[ST.caseId];
    if (!c) return '<div class="card" data-testid="state-loading"><h2>Loading case…</h2></div>';
    return (c.lane === 'surgical' ? surgical : biopsy).render(ST, ctx());
  }
  return { worklist, surgeon, oncologist, metrics, spotcheck }[ST.route].render(ST, ctx());
}

function currentFp() {
  const sim = ST.sim || {};
  const parts = [
    ST.route,
    ST.caseId,
    ST.order,
    ST.lane,
    ST.tab,
    ST.editing,
    ST.notFound,
    ST.busy,
    ST.search,
    ST.status,
    ST.page,
    JSON.stringify(
      ST.health && [
        ST.health.status,
        ST.health.loaded,
        ST.health.synthetic,
        ST.health.skipped,
        ST.health.assets
      ]
    ),
    fingerprint(ST.items),
    sim.running,
    JSON.stringify(sim.pending),
    JSON.stringify([ST.hm.sel, ST.hm.hl, ST.hm.why]),
    ST.cellsOverlay,
    ST.cellsTile,
    JSON.stringify(ST.marginPick),
    JSON.stringify(ST.editDraft)
  ];
  if (ST.route === 'case') {
    const c = ST.cases[ST.caseId];
    parts.push(
      c
        ? JSON.stringify([
            c.agent,
            c.runtime && [
              c.runtime.decision,
              c.runtime.orders,
              c.runtime.tier,
              c.runtime.reasons,
              c.runtime.timer && c.runtime.timer.state
            ]
          ])
        : ''
    );
    parts.push((ST.events[ST.caseId] || []).length, ST.metrics?.eval?.conformal?.created_utc || '');
  }
  if (ST.route === 'surgeon') parts.push(JSON.stringify(ST.routed.surgeon.map((r) => r.id)));
  if (ST.route === 'oncologist') parts.push(JSON.stringify(ST.routed.oncologist.map((r) => r.id)));
  if (ST.route === 'metrics')
    parts.push(
      ST.metrics ? Object.keys(ST.metrics.eval || {}).join() + (ST.metrics.figures || []).length : 'none'
    );
  if (ST.route === 'spotcheck')
    parts.push(
      ST.spot ? ST.spot.items.length : 'none',
      JSON.stringify(ST.spotDraft),
      ST.spotResult || '',
      ST.spotGroup || ''
    );
  return parts.join('|');
}

function draw(forceRender = false) {
  if (!ST.cfg) return;
  $('[data-testid=banner-mock]').hidden = !(ST.mock || ST.synthetic);
  tickClock();
  updateShell();
  const active = document.activeElement;
  const typing =
    active &&
    $('#app').contains(active) &&
    /^(INPUT|SELECT|TEXTAREA)$/.test(active.tagName) &&
    active.id !== 'case-search' &&
    active.type !== 'checkbox' &&
    active.type !== 'range';
  const fp = currentFp();
  if (!forceRender && (fp === lastFp || typing || dragging)) {
    tick();
    return;
  }
  lastFp = fp;
  const y = window.scrollY;
  const searchFocused = active && active.id === 'case-search';
  const caret = searchFocused ? active.selectionStart : null;
  $('#app').innerHTML = view();
  if (searchFocused && $('#case-search')) {
    $('#case-search').focus({ preventScroll: true });
    $('#case-search').setSelectionRange(caret, caret);
  }
  window.scrollTo(0, y);
  if (ST.route === 'worklist') {
    ST.items.forEach((c) => ST.seen.add(c.case_id));
    ST.firstLoad = false;
  }
  if (ST.flashTarget) {
    const el = $(ST.flashTarget);
    if (el) {
      el.scrollIntoView({
        block: 'center',
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
      });
      el.classList.add('flash');
    }
    ST.flashTarget = null;
  }
  tick();
}

const PAGE = {
  worklist: [
    'REVIEW & PRIORITIZE',
    'Every case. A clearer next step.',
    'An evidence-led worklist, with the cases that need you first.',
    'Worklist'
  ],
  case: [
    'CASE REVIEW',
    'Follow the evidence.',
    'Image regions, model confidence, and your next decision in one place.',
    'Case review'
  ],
  surgeon: [
    'SURGICAL COORDINATION',
    'Clarity for the operating room.',
    'Follow simulated frozen sections from arrival to the pathologist’s call.',
    'Operating rooms'
  ],
  oncologist: [
    'RESULTS & HANDOFFS',
    'The next chapter of care.',
    'Signed-out biopsy results and approved follow-up orders, together.',
    'Oncology inbox'
  ],
  metrics: [
    'MODEL EVALUATION',
    'Evidence behind the worklist.',
    'Inspect measured performance, uncertainty, and the limits of this research.',
    'Model evaluation'
  ],
  spotcheck: [
    'QUALITY & REVIEW',
    'Give the evidence a closer look.',
    'Review model-highlighted regions and record an independent assessment.',
    'Evidence review'
  ]
};
function updateShell() {
  const [eyebrow, title, description, crumb] = PAGE[ST.route] || PAGE.worklist;
  $('#page-eyebrow').textContent = eyebrow;
  $('#page-title').textContent = title;
  $('#page-description').textContent = description;
  $('#breadcrumb').textContent = crumb;
  document.title = `${crumb} · PathQueue`;
  $('.workspace').dataset.route = ST.route;
  $('#queue-count').textContent =
    ST.health?.open_cases ?? ST.items.filter((c) => !(c.runtime && c.runtime.done)).length;
  const h = ST.health;
  const offline = ST.fail >= 2;
  const label = offline
    ? 'Connection interrupted'
    : !h
      ? 'Connecting'
      : !h.loaded
        ? 'Awaiting artifacts'
        : h.synthetic
          ? 'Synthetic development data'
          : h.assets?.missing.length
            ? 'Media incomplete'
            : 'Pipeline artifacts loaded';
  $('#data-title').textContent = label;
  $('#data-detail').textContent = !h
    ? 'Checking local service…'
    : `${h.loaded} cases loaded${h.assets?.missing.length ? ` · ${h.assets.missing.length} missing images` : ''}${h.skipped.length ? ` · ${h.skipped.length} rejected` : ''} · ${ST.mock ? 'in-memory demo' : 'local workspace'}`;
  $('.data-status').dataset.state =
    offline || !h || !h.loaded || h.synthetic || h.skipped.length || h.assets?.missing.length
      ? 'pending'
      : 'ready';
  $('#connection').textContent = offline ? 'Reconnecting' : h ? 'Local service connected' : 'Connecting…';
  $('#connection').dataset.state = offline ? 'offline' : h ? 'online' : 'pending';
  $('[data-testid=sim-start]').disabled = !h || !h.loaded || ST.sim?.running;
  $('[data-testid=sim-reset]').disabled = ST.simBusy || !h || !h.loaded;
}

function lookup(id) {
  if (ST.route === 'case' && ST.cases[id]) return ST.cases[id];
  return ST.items.find((c) => c.case_id === id);
}

function tick() {
  const now = Date.now();
  document.querySelectorAll('[data-timer]').forEach((el) => {
    const c = lookup(el.dataset.timer);
    if (!c) return;
    const t = timerNow(c, now, ST.cfg, ST.sim);
    const text =
      el.dataset.timerMode === 'countdown'
        ? t.state === 'stopped'
          ? 'Called'
          : t.countdown.replace(' left', '')
        : t.text;
    if (el.textContent !== text) el.textContent = text;
    el.dataset.state = t.state;
    el.className = el.className.replace(/\b(ok|amber|red|stopped|none)\b/g, '').trim() + ' ' + t.state;
  });
  document.querySelectorAll('[data-timer-sub]').forEach((el) => {
    const c = lookup(el.dataset.timerSub);
    if (c) el.textContent = timerNow(c, now, ST.cfg, ST.sim).countdown || '';
  });
  document.querySelectorAll('[data-timer-bar]').forEach((el) => {
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
  const since = sim._fetched_ms && sim.running ? ((Date.now() - sim._fetched_ms) / 1000) * sim.speed : 0;
  $('#sim').textContent =
    `simulator +${mmss(sim.sim_time_s + since)} · ×${Number(sim.speed.toFixed(1))} · ${sim.running ? 'running' : 'paused'}${ST.mock ? ' · mock' : ''}`;
  const pp = $('#pp');
  pp.setAttribute('aria-pressed', String(!sim.running));
  pp.textContent = sim.running ? 'Pause replay' : sim.sim_time_s > 0 ? 'Resume replay' : 'Pause replay';
  pp.dataset.sim = sim.running ? 'pause' : 'start';
  pp.disabled = !ST.health?.loaded || (!sim.running && sim.sim_time_s === 0);
  const pending = sim.pending ? sim.pending.surgical : null;
  if (pending) {
    document.querySelectorAll('[data-sim=inject_surgical]').forEach((b) => {
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
  if (ST.busy) return; // no double submits (would only produce a confusing 409)
  const id = ST.caseId;
  ST.busy = true;
  draw(true);
  const res = await act(() => api.decide(id, action, payload));
  ST.busy = false;
  if (res && res.case) {
    ST.cases[id] = res.case;
    ST.editing = null;
  } // a rejected edit keeps the editor open
  draw(true);
}

async function simCommand(cmd, variant) {
  if (ST.simBusy) return;
  ST.simBusy = true;
  const res = await act(() => api.sim(cmd, cmd === 'inject_surgical' ? { variant } : undefined));
  ST.simBusy = false;
  updateShell();
  if (res && cmd === 'inject_surgical' && res.injected)
    toast(`Simulated frozen section ${res.injected} received`);
  if (res && cmd === 'reset') {
    ST.page = 0;
    ST.cases = {};
    ST.events = {};
    ST.routed = { surgeon: [], oncologist: [] };
    ST.seen = new Set();
    ST.firstLoad = true;
    if (ST.route === 'case') go('#/worklist');
  }
  return res;
}

function findingClick(fid) {
  const c = ST.cases[ST.caseId];
  const f = c && (c.findings || []).find((x) => x.id === fid);
  if (!f) return;
  if (f.kind === 'concept') {
    if ((c.tiles || []).some((t) => t.concepts && f.data.concept in t.concepts)) ST.hm.sel = f.data.concept;
    ST.hm.hl = f.data.tiles || [];
    ST.hm.why = f.data.concept;
    ST.flashTarget = '[data-testid=heatmap]';
  } else if (f.kind === 'cells') {
    ST.tab = 'cells';
    ST.cellsTile = (f.data.tiles || [])[0] ?? null;
    ST.flashTarget =
      ST.cellsTile !== null
        ? `[data-testid=cells-tile][data-tile="${ST.cellsTile}"]`
        : '[data-testid=cells-tab]';
  } else if (f.kind === 'prediction') ST.flashTarget = '[data-testid=how-sure]';
  else if (f.kind === 'retrieval') ST.flashTarget = '[data-testid=compared]';
  else if (f.kind === 'quality') ST.flashTarget = '[data-testid=how-sure]';
  else if (f.kind === 'margin') ST.flashTarget = '[data-testid=margin-labels]';
  draw(true);
}

function whyClick(id) {
  const c = ST.cases[ST.caseId];
  const concept = (c.concepts || []).find((k) => k.id === id);
  const hasTileValues = (c.tiles || []).some((t) => t.concepts && id in t.concepts);
  ST.hm.why = id;
  if (concept) {
    if (hasTileValues) ST.hm.sel = id;
    ST.hm.hl = concept.top_tiles || [];
  } else if (c.cells && c.cells.tiles) ST.hm.hl = c.cells.tiles.map((t) => t.idx);
  draw(true);
}

document.addEventListener('click', async (e) => {
  const t = e.target.closest('button, [data-nav], tr.row, a');
  if (!t) return;
  const d = t.dataset;
  if (d.resetConfirm !== undefined) {
    t.disabled = true;
    try {
      if (await simCommand('reset')) $('#reset-dialog').close();
    } finally {
      t.disabled = false;
    }
    return;
  }
  if (d.nav) {
    e.preventDefault();
    go('#/' + d.nav);
    return;
  }
  if (d.sim === 'reset') {
    $('#reset-dialog').showModal();
    return;
  }
  if (d.sim) {
    await simCommand(d.sim, d.variant);
    return;
  }
  if (!$('#app').contains(t)) return;
  if (d.page !== undefined) {
    ST.page = Number(d.page);
    draw(true);
    return;
  }
  if (d.statusFilter) {
    ST.page = 0;
    ST.status = d.statusFilter;
    draw(true);
    return;
  }
  if (d.clearSearch !== undefined) {
    ST.search = '';
    ST.status = 'all';
    ST.page = 0;
    draw(true);
    $('#case-search')?.focus();
    return;
  }
  if (d.sort) {
    ST.page = 0;
    ST.order = d.sort;
    await poll(true);
    return;
  }
  if (d.laneFilter) {
    ST.page = 0;
    ST.lane = d.laneFilter;
    await poll(true);
    return;
  }
  if (t.id === 'bk') {
    go('#/worklist');
    return;
  }
  if (t.matches('tr.row')) {
    go('#/case/' + encodeURIComponent(t.dataset.caseId));
    return;
  }
  if (d.tab) {
    ST.tab = d.tab;
    draw(true);
    return;
  }
  if (d.f) {
    findingClick(d.f);
    return;
  }
  if (d.why) {
    whyClick(d.why);
    return;
  }
  if (d.act) {
    if (d.act === 'edit_orders') {
      const c = ST.cases[ST.caseId];
      ST.editDraft = Object.fromEntries(
        ((c.runtime && c.runtime.orders) || []).map((o) => [o.seq, [...o.items]])
      );
      ST.editing = 'orders';
      draw(true);
      return;
    }
    if (d.act === 'edit_cancel') {
      ST.editing = null;
      draw(true);
      return;
    }
    if (d.act === 'edit_save') {
      const orders = Object.entries(ST.editDraft).map(([seq, items]) => ({ seq: Number(seq), items }));
      await decide('edit_orders', { orders });
      return;
    }
    await decide(d.act, {});
    return;
  }
  if (d.call) {
    if (d.call === 'tumor') {
      const c = ST.cases[ST.caseId];
      ST.marginPick = [...((c.margin && c.margin.involved) || [])];
      ST.editing = 'margins';
      draw(true);
      return;
    }
    if (d.call === 'cancel') {
      ST.editing = null;
      draw(true);
      return;
    }
    if (d.call === 'confirm') {
      if (!ST.marginPick.length) {
        toast('Pick at least one margin');
        return;
      }
      await decide('margin_call', { call: 'tumor_at_margin', margins: ST.marginPick });
      return;
    }
    await decide('margin_call', { call: d.call });
    return;
  }
  if (d.spot === 'submit') {
    await spotcheck.submit(ST, api, toast);
    draw(true);
    return;
  }
  if (d.metrics === 'refresh') {
    ST.metrics = null;
    await poll(true);
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    document.querySelectorAll('details[open]').forEach((el) => {
      el.open = false;
    });
  }
  const r = e.target.closest && e.target.closest('tr.row');
  if (r && (e.key === 'Enter' || e.key === ' ')) {
    e.preventDefault();
    go('#/case/' + encodeURIComponent(r.dataset.caseId));
  }
});

document.addEventListener('change', (e) => {
  const t = e.target;
  const d = t.dataset;
  if (d.hm === 'sel') {
    ST.hm.sel = t.value;
    ST.hm.why = null;
    ST.hm.hl = [];
    draw(true);
  } else if (d.editSeq !== undefined) {
    const list = ST.editDraft[d.editSeq] || (ST.editDraft[d.editSeq] = []);
    if (t.checked && !list.includes(d.editItem)) list.push(d.editItem);
    if (!t.checked) ST.editDraft[d.editSeq] = list.filter((i) => i !== d.editItem);
  } else if (d.pick) {
    ST.marginPick = t.checked
      ? [...new Set([...ST.marginPick, d.pick])]
      : ST.marginPick.filter((m) => m !== d.pick);
  } else if (d.cells === 'overlay') {
    ST.cellsOverlay = t.checked;
    draw(true);
  } else if (d.spotVerdict) {
    ST.spotDraft[d.spotItem] = d.spotVerdict;
    draw(true);
  } else if (t.id === 'spot-kind') ST.spotKind = t.value;
  else if (d.spotGroup) {
    ST.spotGroup = t.value;
    draw(true);
  }
});

document.addEventListener('input', (e) => {
  if (e.target.id === 'case-search') {
    ST.page = 0;
    ST.search = e.target.value;
    draw(true);
  }
  if (e.target.id === 'spot-reviewer') ST.spotReviewer = e.target.value;
  if (e.target.dataset.hm === 'opacity') {
    // live update without a re-render (keeps the slider drag)
    ST.hm.opacity = parseFloat(e.target.value);
    document
      .querySelectorAll('.hm-tile')
      .forEach((r) => r.setAttribute('fill-opacity', (parseFloat(r.dataset.v) * ST.hm.opacity).toFixed(3)));
  }
});

document.addEventListener('pointerdown', (e) => {
  if (e.target.dataset && e.target.dataset.hm === 'opacity') dragging = true;
});
document.addEventListener('pointerup', () => {
  if (dragging) {
    dragging = false;
    draw();
  }
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) poll(true);
});
window.addEventListener('hashchange', onRoute);

// ---- boot -------------------------------------------------------------------------------------------------------------------
async function boot() {
  try {
    if (ST.mock) {
      const { createMockApi } = await import('./js/mock.js');
      api = await createMockApi();
    } else {
      api = createApi({ base: params.get('api') || '' });
    }
    ST.cfg = await api.config();
    $('[data-testid=banner-research]').textContent = ST.cfg.banner;
  } catch (err) {
    $('#app').innerHTML =
      `<div class="card" data-testid="state-offline-boot"><h2>Cases did not load</h2><p>The backend did not answer (${esc(err.message || err)}). Start it with <code>make demo</code>, or open this page with <code>?mock=1</code> to use the committed synthetic fixtures. Opening index.html directly blocks the fetch; serve over HTTP (see README).</p></div>`;
    setTimeout(boot, 3000);
    return;
  }
  setInterval(tick, 1000);
  onRoute();
}
boot();
