// PathQueue workspace shell, connected to the PathQueue API (docs/API.md). Original app by adas286 (Team 44);
// live data layer (js/api.js, ?mock=1 js/mock.js), adapter (js/adapter.js) and views (js/views/) by Agent 5;
// clinical-workspace shell (compact header, replay controls, toasts, next-case flow, keyboard) in this file.
// Pattern: one global state ST, string templates rendered into #app, event delegation on document.
// Research prototype — not for clinical use. Ground truth is never fetched or rendered.
import { createApi, ApiError } from './js/api.js';
import { esc, fingerprint, mmss, timerNow, tierInfo } from './js/adapter.js';
import { callText, load, nextCase, openItems, queuePosition, routedKey, save, waitInfo } from './js/flow.js';
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
  page: 0,
  tierFilter: null,
  flagFilter: 'all',
  ordersOnly: false,
  focusId: null,
  inboxRead: new Set(load('pq.inbox.read', [])),
  callsSeen: new Set(load('pq.or.seen', [])),
  freshCalls: new Set(),
  routedPrimed: false,
  justDone: null
};
window.__PQ_ST__ = ST; // debugging + e2e introspection (no secrets or ground truth live here)
let api = null;
let lastFp = null;
let pollTimer = null;
let pollSeq = 0; // a response older than the latest request is dropped (lane/sort clicks race the timer)
let dragging = false; // heatmap opacity slider held: do not re-render under it

const $ = (sel) => document.querySelector(sel);
const ctx = () => ({ now: Date.now(), assetUrl: (p) => api.assetUrl(p) });
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---- toasts ---------------------------------------------------------------------------------------------------------------
// toast('text') or toast('text', {tone: 'ok'|'warn'|'surg', action: {label, href | next: true}, sub})
function toast(msg, opts = {}) {
  const box = $('#toast');
  const el = document.createElement('div');
  el.className = 'toast-item ' + (opts.tone || '');
  el.dataset.testid = 'toast';
  const action = opts.action
    ? opts.action.href
      ? `<a class="toast-act" href="${esc(opts.action.href)}">${esc(opts.action.label)}</a>`
      : `<button class="toast-act" data-toast-next>${esc(opts.action.label)}</button>`
    : '';
  el.innerHTML = `<div class="toast-msg"><b>${esc(msg)}</b>${opts.sub ? `<span>${esc(opts.sub)}</span>` : ''}</div>${action}<button class="toast-x" aria-label="Dismiss notification">×</button>`;
  box.prepend(el);
  while (box.children.length > 2) box.lastElementChild.remove();
  const ttl = opts.ttl || (opts.action ? 6500 : 4000);
  setTimeout(() => {
    el.classList.add('out');
    setTimeout(() => el.remove(), reducedMotion() ? 0 : 220);
  }, ttl);
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
    ST.marginPick = [];
    ST.justDone = null;
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
  if (changed && ST.route !== 'worklist') $('#app').focus({ preventScroll: true });
}

// ---- data -----------------------------------------------------------------------------------------------------------------
function worklistQuery(route) {
  if (route === 'surgeon') return { lane: 'surgical', order: 'ai' };
  if (route === 'worklist') return { lane: ST.lane, order: ST.order };
  return { lane: 'all', order: 'ai' }; // case view + other pages: the global priority order drives "next case"
}

// Snapshot route and request generation: slow responses cannot overwrite a newer navigation.
async function refresh(force, seq) {
  const route = ST.route,
    caseId = ST.caseId;
  const [wl, health, toSurgeon, toOncologist] = await Promise.all([
    api.worklist(worklistQuery(route)),
    api.health(),
    api.routed('surgeon'),
    api.routed('oncologist')
  ]);
  let detail = null,
    events = null,
    missing = false,
    evaluation = null,
    spot = null;
  // After a 404 the case is not asked for again until the route or the replay (arrivals, reset) changes.
  const arrivals = wl.sim ? `${wl.sim.arrived}|${wl.sim.sim_time_s > 0}` : '';
  const skipMissing = route === 'case' && ST.notFound === caseId && ST.notFoundAt === arrivals;
  if (route === 'case' && caseId && skipMissing) missing = true;
  else if (route === 'case' && caseId) {
    try {
      [detail, events] = await Promise.all([api.getCase(caseId), api.events(caseId)]);
      if (!ST.metrics) evaluation = await api.metrics();
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) missing = true;
      else throw err;
    }
  } else if (route === 'metrics' && (force || !ST.metrics)) evaluation = await api.metrics();
  else if (route === 'spotcheck' && (force || !ST.spot)) spot = await api.spotcheckItems();
  else if (!ST.metrics && !ST.metricsAsked && !ST.mock) {
    ST.metricsAsked = true; // once: the conformal method label + coverage shown in worklist/inbox copy
    evaluation = await api.metrics().catch(() => null);
  }
  if (seq !== pollSeq) return;
  ST.items = wl.items || [];
  ST.loadedOnce = true;
  ST.sim = wl.sim || ST.sim;
  ST.health = health;
  ST.synthetic = wl.synthetic || health.synthetic || ST.items.some((c) => c.synthetic);
  if (route === 'case') {
    ST.notFound = missing ? caseId : null;
    ST.notFoundAt = missing ? arrivals : null;
    if (detail) {
      ST.cases[caseId] = detail;
      ST.events[caseId] = events;
    }
  }
  noticeRouted(toSurgeon || [], toOncologist || []);
  ST.routed.surgeon = toSurgeon || [];
  ST.routed.oncologist = toOncologist || [];
  if (evaluation) ST.metrics = evaluation;
  if (spot) ST.spot = spot;
}

// New margin calls light up on the OR board (and a toast elsewhere); results remembered per viewer.
function noticeRouted(toSurgeon, toOncologist) {
  const fresh = toSurgeon.filter((r) => !ST.callsSeen.has(routedKey(r)));
  if (fresh.length) {
    for (const r of fresh) {
      ST.callsSeen.add(routedKey(r));
      if (ST.routedPrimed) ST.freshCalls.add(routedKey(r));
    }
    save('pq.or.seen', [...ST.callsSeen].slice(-400));
    if (ST.freshCalls.size) setTimeout(() => {
      fresh.forEach((r) => ST.freshCalls.delete(routedKey(r)));
      lastFp = null;
      draw();
    }, 20000);
  }
  // first poll after load only primes the "seen" set: nothing restored from storage is announced as new
  ST.routedPrimed = true;
  ST.inboxUnread = toOncologist.filter((r) => !ST.inboxRead.has(routedKey(r))).length;
}

async function poll(force = false) {
  clearTimeout(pollTimer);
  const seq = ++pollSeq;
  try {
    await refresh(force, seq);
    if (seq !== pollSeq) return;
    if (ST.fail >= 2) toast('Connection to the local service restored', { tone: 'ok' });
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
    el.innerHTML = `<b>The local service is not answering</b> at ${esc(api.base || location.origin)}. Retrying every few seconds; what you see is from ${esc(ST.staleSince.toLocaleTimeString())}. Start it with <code>make demo</code>, or open <code>?mock=1</code> for the built-in fixtures.`;
  } else el.hidden = true;
}

// ---- rendering ------------------------------------------------------------------------------------------------------------
function view() {
  if (ST.route === 'case') {
    if (ST.notFound === ST.caseId)
      return `<button class="back" id="bk" data-testid="back">Back to worklist</button><div class="card" data-testid="case-not-found"><h2>Case not found</h2><p>${esc(ST.caseId)} is unknown or has not arrived yet. Arrivals appear as the replay runs; a reset clears earlier cases.</p></div>`;
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
    ST.tierFilter,
    ST.flagFilter,
    ST.ordersOnly,
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
  if (ST.route === 'surgeon')
    parts.push(JSON.stringify(ST.routed.surgeon.map((r) => r.id)), [...ST.freshCalls].join());
  if (ST.route === 'oncologist')
    parts.push(JSON.stringify(ST.routed.oncologist.map((r) => r.id)), [...ST.inboxRead].length, ST.inboxOpen || '');
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

// Stable identity of a focusable element across re-renders (volatile attributes such as data-state are left out).
const FOCUS_ATTRS = [
  'id', 'data-testid', 'data-case-id', 'data-sort', 'data-lane-filter', 'data-status-filter', 'data-tier-filter',
  'data-inbox-key', 'data-inbox-read', 'data-act', 'data-call', 'data-tab', 'data-f', 'data-why', 'data-page',
  'data-pick', 'data-edit-seq', 'data-edit-item', 'data-hm', 'data-cells', 'data-spot-item', 'data-spot-verdict',
  'data-spot-group', 'data-next-case', 'data-mx-open', 'href'
];
function focusKey(el) {
  const app = $('#app');
  if (!el || el === app || !app.contains(el)) return null;
  const tag = el.tagName.toLowerCase();
  const attrs = FOCUS_ATTRS.filter((a) => el.hasAttribute(a)).map((a) => `[${a}="${CSS.escape(el.getAttribute(a))}"]`);
  // anonymous controls (e.g. a <summary>) and repeated links are told apart by their position among the matches
  const sel = tag + attrs.join('');
  try {
    return { sel, index: Math.max(0, [...app.querySelectorAll(sel)].indexOf(el)) };
  } catch (_) {
    return null;
  }
}
function findFocus(key) {
  try {
    const list = $('#app').querySelectorAll(key.sel);
    return list[key.index] || list[0] || null;
  } catch (_) {
    return null;
  }
}
function detailsKey(el) {
  const owner = el.parentElement && el.parentElement.closest('[data-room], [data-case-id], [data-inbox-key], [id]');
  const ownerId = owner ? owner.dataset.room || owner.dataset.caseId || owner.dataset.inboxKey || owner.id : '';
  return `${el.className}|${el.dataset.mxOpen || el.dataset.testid || ''}|${ownerId}`;
}

function draw(forceRender = false) {
  if (!ST.cfg) return;
  $('[data-testid=banner-mock]').hidden = !(ST.mock || ST.synthetic);
  updateShell();
  renderCasebar();
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
  // Keyboard position survives the poll's re-render: the focused control (row, filter, tab, inbox card...) is found
  // again by its identifying attributes, and open <details> stay open.
  const focusSel = focusKey(active);
  const openDetails = [...$('#app').querySelectorAll('details[open]')].map(detailsKey);
  $('#app').innerHTML = view();
  if (searchFocused && $('#case-search')) {
    $('#case-search').focus({ preventScroll: true });
    $('#case-search').setSelectionRange(caret, caret);
  } else if (focusSel) {
    const el = findFocus(focusSel);
    (el || $('#app')).focus({ preventScroll: true });
  }
  if (openDetails.length)
    $('#app')
      .querySelectorAll('details')
      .forEach((el) => {
        if (openDetails.includes(detailsKey(el))) el.open = true;
      });
  window.scrollTo(0, y);
  if (ST.route === 'worklist' && ST.loadedOnce) {
    // arrivals after the first rendered data animate in; whatever was there on load does not
    ST.items.forEach((c) => ST.seen.add(c.case_id));
    ST.firstLoad = false;
  }
  if (ST.flashTarget) {
    const el = $(ST.flashTarget);
    if (el) {
      el.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' });
      el.classList.add('flash');
    }
    ST.flashTarget = null;
  }
  tick();
}

const PAGE = {
  worklist: 'Worklist',
  case: 'Case review',
  surgeon: 'Operating rooms',
  oncologist: 'Oncology inbox',
  metrics: 'Model evaluation',
  spotcheck: 'Evidence review'
};

function pageDescription() {
  const open = openItems(ST.items);
  if (ST.route === 'worklist') {
    const all = ST.health?.open_cases;
    return ST.lane === 'all' || all === undefined ? `${open.length} open` : `${open.length} open here · ${all} in all lanes`;
  }
  if (ST.route === 'case') return ST.caseId || '';
  if (ST.route === 'surgeon') {
    const live = open.length;
    return live ? `${live} awaiting a margin call` : 'No frozen section awaiting a call';
  }
  if (ST.route === 'oncologist') {
    const n = ST.routed.oncologist.length;
    return `${n} result${n === 1 ? '' : 's'}${ST.inboxUnread ? ` · ${ST.inboxUnread} unread` : ''}`;
  }
  if (ST.route === 'metrics') return 'Held-out test split · bootstrap 95% intervals';
  return 'Independent review of model-highlighted regions';
}

function updateShell() {
  const crumb = PAGE[ST.route] || PAGE.worklist;
  $('#breadcrumb').textContent = crumb;
  $('#page-description').textContent = pageDescription();
  document.title = `${ST.route === 'case' && ST.caseId ? ST.caseId + ' · ' : ''}${crumb} · PathQueue`;
  $('.workspace').dataset.route = ST.route;
  $('#queue-count').textContent = ST.health?.open_cases ?? openItems(ST.items).length;
  const orOpen = (ST.health && ST.health.open_surgical) ?? null;
  const orEl = $('#or-count');
  const liveFrozen = orOpen !== null ? orOpen : ST.route === 'surgeon' || ST.lane === 'all' || ST.route === 'case'
    ? openItems(ST.items).filter((c) => c.lane === 'surgical').length
    : 0;
  orEl.hidden = !liveFrozen;
  orEl.textContent = liveFrozen;
  const inbox = $('#inbox-count');
  inbox.hidden = !ST.inboxUnread;
  inbox.textContent = ST.inboxUnread || 0;
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
    : `${h.loaded} cases${h.assets?.missing.length ? ` · ${h.assets.missing.length} missing images` : ''}${h.skipped.length ? ` · ${h.skipped.length} rejected` : ''} · ${ST.mock ? 'in-memory demo' : 'local workspace'}`;
  $('.data-status').dataset.state =
    offline || !h || !h.loaded || h.synthetic || h.skipped.length || h.assets?.missing.length
      ? 'pending'
      : 'ready';
  $('#connection').textContent = offline ? 'Reconnecting' : h ? 'Connected' : 'Connecting…';
  $('#connection').title = offline ? 'The local service is not answering' : 'Local PathQueue service';
  $('#connection').dataset.state = offline ? 'offline' : h ? 'online' : 'pending';
  updateReplay();
}

function updateReplay() {
  const sim = ST.sim;
  const h = ST.health;
  const box = $('.replay');
  const start = $('[data-testid=sim-start]');
  const pp = $('#pp');
  const loaded = !!(h && h.loaded);
  if (!sim) {
    start.disabled = true;
    return;
  }
  const fresh = !sim.running && !(sim.sim_time_s > 0);
  box.dataset.state = sim.running ? 'running' : fresh ? 'ready' : 'paused';
  start.hidden = !fresh;
  start.disabled = !loaded || ST.simBusy;
  pp.hidden = fresh;
  pp.setAttribute('aria-pressed', String(!sim.running));
  pp.textContent = sim.running ? 'Pause replay' : 'Resume replay';
  pp.dataset.sim = sim.running ? 'pause' : 'start';
  pp.disabled = !loaded || ST.simBusy;
  $('[data-testid=sim-reset]').disabled = ST.simBusy || !loaded;
  const pending = sim.pending ? sim.pending.surgical : null;
  if (pending) {
    document.querySelectorAll('[data-sim=inject_surgical]').forEach((b) => {
      const n = pending[b.dataset.variant] || 0;
      b.disabled = n === 0 || ST.simBusy || !loaded;
      b.title = n ? `${n} simulated ${b.dataset.variant === 'normal' ? '' : b.dataset.variant + ' '}frozen sections left` : 'None left: reset the replay to send more';
      const tag = b.querySelector('[data-pending]');
      if (tag) tag.textContent = n;
    });
  }
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
  // compact waiting cells (worklist, OR board): "12.4 h / 24 h", "08:41 / 20:00"
  document.querySelectorAll('[data-wait]').forEach((el) => {
    const c = lookup(el.dataset.wait);
    if (!c) return;
    const w = waitInfo(timerNow(c, now, ST.cfg, ST.sim));
    const b = el.firstElementChild;
    if (b && b.textContent !== w.text) b.textContent = w.text;
    if (el.dataset.state !== w.state) {
      el.dataset.state = w.state;
      el.className = 'wait ' + w.state;
    }
  });
  tickClock();
}

function tickClock() {
  const clock = $('#clock');
  const d = new Date();
  const text = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  if (clock.textContent !== text) {
    clock.textContent = text;
    clock.dateTime = d.toISOString();
  }
  const sim = ST.sim;
  if (!sim) return;
  const since = sim._fetched_ms && sim.running ? ((Date.now() - sim._fetched_ms) / 1000) * (Number(sim.speed) || 1) : 0;
  const fresh = !sim.running && !(sim.sim_time_s > 0);
  const state = sim.running ? 'Replay running' : fresh ? 'Replay ready' : 'Replay paused';
  $('#sim').innerHTML =
    `<i class="rs-dot" aria-hidden="true"></i><b>${state}</b><span class="rs-t">+${mmss((Number(sim.sim_time_s) || 0) + since)}</span><span class="rs-x">×${Number((Number(sim.speed) || 1).toFixed(1))}</span>${ST.mock ? '<span class="rs-x">mock</span>' : ''}` +
    `<span class="sr-only"> · simulator ${sim.running ? 'running' : 'paused'}</span>`;
}

// ---- case navigation bar (shell): position in the priority queue + the next case ------------------------------------
function renderCasebar() {
  const bar = $('#casebar');
  if (ST.route !== 'case' || !ST.caseId || ST.notFound === ST.caseId) {
    if (!bar.hidden) {
      bar.hidden = true;
      bar.innerHTML = '';
    }
    return;
  }
  const c = ST.cases[ST.caseId];
  const done = !!(c && c.runtime && c.runtime.done);
  const pos = queuePosition(ST.items, ST.caseId);
  const open = openItems(ST.items).length;
  const next = nextCase(ST.items, ST.caseId);
  const nextTier = next ? tierInfo(next.runtime ? next.runtime.tier : next.flag.tier, ST.cfg) : null;
  let status;
  if (done && c.lane === 'surgical') {
    const call = c.runtime.decision.margin_call;
    status = `<span class="cb-done" data-testid="casebar-done"><b>Call sent to ${esc(c.runtime.or_room || 'the OR')}</b> ${esc(callText(call))}</span>`;
  } else if (done) {
    status = '<span class="cb-done" data-testid="casebar-done"><b>Signed out</b> Result sent to the oncology inbox</span>';
  } else {
    status = pos
      ? `<span class="cb-pos">Case <b>${pos}</b> of ${open} open in priority order</span>`
      : `<span class="cb-pos">${open} open cases</span>`;
  }
  const nextHtml = next
    ? `<button class="btn ${done ? 'p' : ''} cb-next" data-next-case="${esc(next.case_id)}" data-testid="casebar-next">${done ? 'Open next case' : 'Next case'} <span class="cb-id">${esc(next.case_id)} · ${esc(nextTier.label)}</span><kbd>N</kbd></button>`
    : `<span class="cb-empty">${done ? 'Queue clear: no open cases left' : 'No other open case'}</span>`;
  const html = `<button class="cb-back" id="cb-back" data-testid="casebar-back"><span aria-hidden="true">←</span> Worklist</button>${status}${nextHtml}`;
  bar.dataset.done = String(done);
  if (bar.innerHTML !== html) {
    const hadFocus = bar.contains(document.activeElement) && document.activeElement.dataset.nextCase !== undefined;
    bar.innerHTML = html;
    if (hadFocus && $('#casebar [data-next-case]')) $('#casebar [data-next-case]').focus({ preventScroll: true });
  }
  bar.hidden = false;
  if (ST.justDone === ST.caseId && done) {
    ST.justDone = null;
    const btn = $('#casebar [data-next-case]');
    if (btn) btn.focus({ preventScroll: true });
  }
}

function openNext(fromId = null) {
  let next;
  if (ST.route === 'worklist') {
    const visible = [...document.querySelectorAll('tr.row[data-done=false]')].map((r) => r.dataset.caseId);
    next = visible[0] ? { case_id: visible[0] } : nextCase(ST.items, null);
  } else next = nextCase(ST.items, fromId || ST.caseId);
  if (next) go('#/case/' + encodeURIComponent(next.case_id));
  else toast('No open cases left in the queue', { tone: 'ok' });
}

// ---- actions ----------------------------------------------------------------------------------------------------------------
async function act(fn, okMsg) {
  try {
    const res = await fn();
    if (okMsg) toast(okMsg);
    return res;
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) toast('Not allowed', { tone: 'warn', sub: err.detail });
    else if (err instanceof ApiError && err.status === 422) toast('Request rejected', { tone: 'warn', sub: err.detail });
    else toast('Request failed', { tone: 'warn', sub: String(err.message || err) });
    return null;
  } finally {
    await poll(true);
  }
}

function orderList(c) {
  return ((c.agent && c.agent.orders) || []).map((o) => o.items.join(', ')).join(' · ');
}

function announceDecision(action, res) {
  const c = res.case;
  const id = c.case_id;
  if (action === 'approve_orders') toast(`Orders placed for ${id}`, { tone: 'ok', sub: orderList(c) });
  else if (action === 'edit_orders') toast(`Edited orders placed for ${id}`, { tone: 'ok', sub: orderList(c) });
  else if (action === 'closer_look') toast(`${id} marked for a closer look`, { sub: 'Drafted orders stay unplaced until you decide.' });
  else if (action === 'dismiss_flag') toast(`Flag dismissed for ${id}`, { sub: 'No follow-up orders were placed.' });
  else if (action === 'sign_out')
    toast(`${id} signed out`, { tone: 'ok', sub: 'Result sent to the oncology inbox', action: nextCase(ST.items, id) ? { label: 'Open next case', next: true } : { label: 'View inbox', href: '#/oncologist' } });
  else if (action === 'margin_call') {
    const call = c.runtime && c.runtime.decision && c.runtime.decision.margin_call;
    toast(`Margin call sent to ${(c.runtime && c.runtime.or_room) || 'the OR'}`, { tone: 'surg', sub: callText(call), action: { label: 'View OR board', href: '#/surgeon' } });
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
    if (action === 'margin_call') ST.marginPick = [];
    if (res.case.runtime && res.case.runtime.done) ST.justDone = id;
    announceDecision(action, res);
  } // a rejected edit keeps the editor open
  draw(true);
}

async function simCommand(cmd, variant) {
  if (ST.simBusy) return;
  ST.simBusy = true;
  updateReplay();
  const res = await act(async () => {
    const out = await api.sim(cmd, cmd === 'inject_surgical' ? { variant } : undefined);
    // Leave a case view before the follow-up poll: the reset case no longer exists (no 404 + "not found" flash).
    if (cmd === 'reset') clearReplayState();
    return out;
  });
  ST.simBusy = false;
  updateShell();
  if (res && cmd === 'inject_surgical' && res.injected) {
    const item = ST.items.find((c) => c.case_id === res.injected);
    const room = item && item.runtime && item.runtime.or_room;
    const kind = variant === 'degraded' ? 'Degraded frozen section' : variant === 'composite' ? 'Composite frozen section' : 'Frozen section';
    toast(`${kind} ${res.injected} received${room ? ' from ' + room : ''}`, {
      tone: 'surg',
      sub: 'Simulated · timer started · top of the worklist',
      action: { label: 'Open', href: '#/case/' + encodeURIComponent(res.injected) }
    });
  }
  if (res && cmd === 'start') toast('Replay started', { sub: 'Biopsies arrive over the next minute or two.' });
  if (res && cmd === 'reset') toast('Replay reset', { sub: 'Decisions and routed results from the last replay were cleared.' });
  return res;
}

function clearReplayState() {
  ST.page = 0;
  ST.cases = {};
  ST.events = {};
  ST.routed = { surgeon: [], oncologist: [] };
  ST.items = [];
  ST.seen = new Set();
  ST.firstLoad = true;
  ST.tierFilter = null;
  ST.freshCalls.clear();
  ST.inboxOpen = null;
  ST.justDone = null;
  if (ST.route === 'case') {
    location.hash = '#/worklist';
    onRoute(); // synchronously: the hashchange event fires after the follow-up poll has started
  }
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

function markRead(keys) {
  keys.forEach((k) => ST.inboxRead.add(k));
  save('pq.inbox.read', [...ST.inboxRead].slice(-600));
  ST.inboxUnread = ST.routed.oncologist.filter((r) => !ST.inboxRead.has(routedKey(r))).length;
  lastFp = null;
  draw(true);
}

document.addEventListener('click', async (e) => {
  const t = e.target.closest('button, [data-nav], tr.row, a, [data-inbox-key]');
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
  if (t.classList.contains('toast-x')) {
    t.closest('.toast-item').remove();
    return;
  }
  if (d.toastNext !== undefined) {
    t.closest('.toast-item').remove();
    openNext();
    return;
  }
  if (t.matches('.toast-act')) {
    t.closest('.toast-item').remove();
    return; // anchor navigation proceeds
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
  if (d.nextCase !== undefined) {
    if (d.nextCase) go('#/case/' + encodeURIComponent(d.nextCase));
    else openNext();
    return;
  }
  if (t.id === 'cb-back' || t.id === 'bk') {
    go('#/worklist');
    return;
  }
  if (!$('#app').contains(t)) return;
  if (d.inboxRead !== undefined) {
    markRead([d.inboxRead]);
    return;
  }
  if (d.inboxReadAll !== undefined) {
    markRead(ST.routed.oncologist.map(routedKey));
    return;
  }
  if (d.inboxKey !== undefined && !e.target.closest('a, button')) {
    ST.inboxOpen = ST.inboxOpen === d.inboxKey ? null : d.inboxKey;
    if (!ST.inboxRead.has(d.inboxKey)) markRead([d.inboxKey]);
    else draw(true);
    return;
  }
  if (d.page !== undefined) {
    ST.page = Number(d.page);
    draw(true);
    return;
  }
  if (d.tierFilter) {
    const tier = Number(d.tierFilter);
    ST.tierFilter = ST.tierFilter === tier ? null : tier;
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
    ST.tierFilter = null;
    ST.flagFilter = 'all';
    ST.ordersOnly = false;
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
        toast('Pick at least one margin', { tone: 'warn' });
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

// ---- keyboard ---------------------------------------------------------------------------------------------------------------
function isTyping(el) {
  return !!el && (/^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName) || el.isContentEditable);
}
function moveRow(delta) {
  const rows = [...document.querySelectorAll('tr.row')];
  if (!rows.length) return;
  const i = rows.indexOf(document.activeElement);
  const next = rows[i < 0 ? (delta > 0 ? 0 : rows.length - 1) : Math.max(0, Math.min(rows.length - 1, i + delta))];
  next.focus({ preventScroll: true });
  next.scrollIntoView({ block: 'nearest', behavior: 'auto' });
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    document.querySelectorAll('details[open]').forEach((el) => {
      el.open = false;
    });
    if (e.target.id === 'case-search' && e.target.value) {
      ST.search = '';
      draw(true);
    }
  }
  const r = e.target.closest && e.target.closest('tr.row');
  if (r && (e.key === 'Enter' || e.key === ' ')) {
    e.preventDefault();
    go('#/case/' + encodeURIComponent(r.dataset.caseId));
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target) || $('#reset-dialog').open) return;
  if (ST.route === 'worklist') {
    if (e.key === 'j' || e.key === 'ArrowDown') {
      if (e.key === 'ArrowDown' && !r) return; // plain page scrolling unless a row has focus
      e.preventDefault();
      moveRow(1);
      return;
    }
    if (e.key === 'k' || e.key === 'ArrowUp') {
      if (e.key === 'ArrowUp' && !r) return;
      e.preventDefault();
      moveRow(-1);
      return;
    }
    if (e.key === '/') {
      e.preventDefault();
      $('#case-search')?.focus();
      return;
    }
  }
  if ((e.key === 'n' || e.key === 'N') && (ST.route === 'worklist' || ST.route === 'case')) {
    e.preventDefault();
    openNext();
  }
});

document.addEventListener('focusin', (e) => {
  if (e.target.matches && e.target.matches('tr.row')) ST.focusId = e.target.dataset.caseId;
});

document.addEventListener('change', (e) => {
  const t = e.target;
  const d = t.dataset;
  if (t.id === 'flag-filter') {
    ST.flagFilter = t.value;
    draw(true);
  } else if (t.id === 'orders-only') {
    ST.ordersOnly = t.checked;
    draw(true);
  } else if (d.hm === 'sel') {
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

// Sticky offsets follow the real header height (it wraps onto two rows on narrow screens).
function syncStick() {
  const top = $('#topbar');
  const sticky = getComputedStyle(top).position === 'sticky';
  document.documentElement.style.setProperty('--stick', (sticky ? top.offsetHeight : 0) + 'px');
}
if ('ResizeObserver' in window) new ResizeObserver(syncStick).observe($('#topbar'));
window.addEventListener('resize', syncStick);

// ---- boot -------------------------------------------------------------------------------------------------------------------
async function boot() {
  tickClock();
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
      `<div class="card" data-testid="state-offline-boot"><h2>Cases did not load</h2><p>The local service did not answer (${esc(err.message || err)}). Start it with <code>make demo</code>, or open this page with <code>?mock=1</code> to use the committed synthetic fixtures. Opening index.html directly blocks the requests; serve it over HTTP (see README).</p></div>`;
    setTimeout(boot, 3000);
    return;
  }
  syncStick();
  setInterval(tick, 1000);
  onRoute();
}
boot();
