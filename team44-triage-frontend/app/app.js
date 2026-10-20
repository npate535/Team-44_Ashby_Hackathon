// PathQueue workspace shell, connected to the PathQueue API (docs/API.md). Original app by adas286 (Team 44);
// live data layer (js/api.js, ?mock=1 js/mock.js), adapter (js/adapter.js) and views (js/views/) by Agent 5;
// clinical-workspace shell (compact header, replay controls, toasts, next-case flow, keyboard) in this file.
// Pattern: one global state ST, string templates rendered into #app, event delegation on document.
// Research prototype — not for clinical use. Ground truth is never fetched or rendered.
import { createApi, ApiError } from './js/api.js';
import { esc, fingerprint, mmss, timerNow, tierInfo, toView } from './js/adapter.js';
import { callText, load, nextCase, openItems, queuePosition, routedKey, save, waitInfo } from './js/flow.js';
import * as worklist from './js/views/worklist.js';
import * as biopsy from './js/views/case_biopsy.js';
import * as surgical from './js/views/case_surgical.js';
import * as surgeon from './js/views/surgeon.js';
import * as oncologist from './js/views/oncologist.js';
import { animateScene, syncGlass } from './js/glass.js';

// The evaluation, TCGA and evidence-review pages (with the chart library, about 230 KB) load when one is first opened,
// so a slow connection reaches the worklist without them. Imported once; after a failed import the page says so, and
// opening it again tries again.
const LAZY = {
  metrics: () => import('./js/views/metrics.js'),
  tcga: () => import('./js/views/tcga.js'),
  spotcheck: () => import('./js/views/spotcheck.js')
};
const views = {};
const viewLoads = {};
const viewErrors = {};
function loadView(route) {
  if (views[route]) return Promise.resolve(views[route]);
  if (!viewLoads[route])
    viewLoads[route] = LAZY[route]()
      .then((mod) => {
        views[route] = mod;
        delete viewErrors[route];
        // A page module may watch the route (tcga.js remembers the specimen the list was left for): its hashchange
        // listener has to run before the app redraws, as when every module was imported ahead of this file.
        window.removeEventListener('hashchange', onRoute);
        window.addEventListener('hashchange', onRoute);
        return mod;
      })
      .catch((err) => {
        viewLoads[route] = null;
        viewErrors[route] = err;
        throw err;
      })
      .finally(() => {
        if (ST.route !== route) return;
        lastFp = null; // the page's module (or its failure) replaces the loading state
        draw();
      });
  return viewLoads[route];
}
// Once the first screen is up, the other pages are fetched while the browser is idle, so opening one is immediate
// (not when the viewer has asked the browser to save data).
let prefetched = false;
function prefetchViews() {
  if (prefetched || (navigator.connection && navigator.connection.saveData)) return;
  prefetched = true;
  const run = () => Object.keys(LAZY).forEach((r) => loadView(r).catch(() => {}));
  setTimeout(() => ('requestIdleCallback' in window ? requestIdleCallback(run, { timeout: 8000 }) : run()), 5000);
}

const params = new URLSearchParams(location.search);
const LOCAL_HOSTS = ['localhost', '127.0.0.1', '[::1]', ''];
const localService = LOCAL_HOSTS.includes(new URL(params.get('api') || location.href, location.href).hostname);
// the "make demo" hint only for a copy of the app opened from this machine against a local service
const connectionHelp = localService && LOCAL_HOSTS.includes(location.hostname)
  ? 'Start it with <code>make demo</code>, or open <code>?mock=1</code> for the built-in fixtures.'
  : 'The service may be waking up. This page will reconnect automatically; you can also try again shortly.';
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
  arrived: new Map(), // case id -> ms it first appeared after the initial load (worklist "New" tag + highlight)
  firstLoad: true,
  fail: 0,
  staleSince: null,
  editing: null,
  editDraft: {},
  marginPick: [],
  hm: { sel: null, hl: [], opacity: 0.6, why: null }, // sel null: the case's first layer (heatmap.js layers)
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
  justDone: null,
  interp: {}, // AI interpretation per case id ('__results' = evaluation page): {status, data, error, startedAt}
  interpRev: 0,
  tcga: null // GET /api/tcga index (external specimens; not the arrival queue)
};
window.__PQ_ST__ = ST; // debugging + e2e introspection (no secrets or ground truth live here)
let api = null;
let lastFp = null;
let pollTimer = null;
let pollSeq = 0; // a response older than the latest request is dropped (lane/sort clicks race the timer)
let dragging = false; // heatmap opacity slider held: do not re-render under it
let tickAt = Date.now(); // the instant every timer on screen shows; it advances once a second, never between ticks

const $ = (sel) => document.querySelector(sel);
const ctx = () => ({ now: tickAt, assetUrl: (p) => api.assetUrl(p) }); // views draw timers at the current tick
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---- toasts ---------------------------------------------------------------------------------------------------------------
// toast('text') or toast('text', {tone: 'ok'|'warn'|'surg', action: {label, href | next: true}, sub, key, net})
// net: the notice is about the connection (cleared when Sepia answers again).
function toast(msg, opts = {}) {
  const box = $('#toast');
  const el = document.createElement('div');
  el.className = 'toast-item ' + (opts.tone || '');
  el.dataset.testid = 'toast';
  if (opts.net) el.dataset.net = opts.net === true ? 'yes' : opts.net;
  const action = opts.action
    ? opts.action.href
      ? `<a class="toast-act" href="${esc(opts.action.href)}">${esc(opts.action.label)}</a>`
      : `<button class="toast-act" data-toast-next>${esc(opts.action.label)}</button>`
    : '';
  el.innerHTML = `<div class="toast-msg"><b>${esc(msg)}</b>${opts.sub ? `<span>${esc(opts.sub)}</span>` : ''}</div>${action}<button class="toast-x" aria-label="Dismiss notification">×</button>`;
  // the same notice twice (N pressed again on a clear queue, a repeated rejection) replaces the first one
  el.dataset.key = opts.key || `${msg}|${opts.sub || ''}`;
  el.dataset.born = Date.now();
  for (const old of [...box.children]) if (old.dataset.key === el.dataset.key) old.remove();
  box.prepend(el);
  while (box.children.length > 2) box.lastElementChild.remove();
  // A warning stays until it is dismissed and is announced at once. Other notices are short-lived, so they never sit
  // over the controls for long; the countdown pauses while the pointer or keyboard focus is on the notice, and a
  // click anywhere on it dismisses it.
  if (opts.tone === 'warn') {
    el.setAttribute('role', 'alert');
    return;
  }
  let left = opts.ttl || (opts.action ? 5000 : 3500);
  let timer = null;
  let since = 0;
  const run = () => {
    since = Date.now();
    timer = setTimeout(() => dismissToast(el), left);
  };
  const hold = () => {
    if (timer === null) return;
    clearTimeout(timer);
    timer = null;
    left = Math.max(1200, left - (Date.now() - since));
  };
  el.addEventListener('pointerenter', hold);
  el.addEventListener('focusin', hold);
  el.addEventListener('pointerleave', () => !el.contains(document.activeElement) && timer === null && run());
  el.addEventListener('focusout', (e) => !el.contains(e.relatedTarget) && !el.matches(':hover') && timer === null && run());
  run();
}
function dismissToasts(match) {
  let n = 0;
  document.querySelectorAll('#toast .toast-item').forEach((el) => {
    if (match(el)) {
      dismissToast(el);
      n += 1;
    }
  });
  return n;
}
function dismissToast(el) {
  if (!el.isConnected || el.classList.contains('out')) return;
  el.classList.add('out');
  setTimeout(() => el.remove(), reducedMotion() ? 0 : 200);
}

// ---- routing --------------------------------------------------------------------------------------------------------------
// A malformed % escape in a typed or truncated address (#/case/BACH_0225%) is kept as it is: decodeURIComponent would
// throw, and the app would never finish starting. The id then simply is not found.
const decodeId = (s) => {
  try {
    return decodeURIComponent(s);
  } catch (_) {
    return s;
  }
};
function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  if (h === 'app') return { route: ST.route, caseId: ST.caseId }; // the skip link's target is not a route
  const [route, id] = h.split('/');
  if (route === 'case' && id) return { route: 'case', caseId: decodeId(id) };
  if (route === 'tcga')
    return { route: 'tcga', caseId: id ? decodeId(id) : null };
  return {
    route: ['worklist', 'surgeon', 'oncologist', 'metrics', 'tcga', 'spotcheck'].includes(route) ? route : 'worklist',
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
    // notices from the page being left are cleared; one raised by the action that navigated (a reset) stays
    document.querySelectorAll('#toast .toast-item').forEach((el) => {
      if (Date.now() - Number(el.dataset.born) > 800) dismissToast(el);
    });
    ST.editing = null;
    ST.notFound = null;
    ST.tab = 'evidence';
    ST.cellsTile = null;
    ST.marginPick = [];
    ST.justDone = null;
    ST.hm = { sel: null, hl: [], opacity: ST.hm.opacity, why: null };
    setReplayOpen(false); // phones: a new page opens with the replay controls folded
    if (!ST.metrics) ST.metricsAsked = false; // an evaluation request that failed is tried again on the next page
    delete viewErrors[r.route]; // a page whose code failed to load is tried again when it is opened again
    window.scrollTo(0, 0);
  }
  document.querySelectorAll('nav [data-nav]').forEach((b) => {
    const active =
      b.dataset.nav === (ST.route === 'case' ? 'worklist' : ST.route === 'tcga' ? 'tcga' : ST.route);
    if (active) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  });
  syncGlass(); // also scrolls the current destination into view when the pill scrolls (phones)
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

// The evaluation payload as optional context (case coverage line, TCGA story panel, inbox wording): requested beside
// the page's own data, never awaited by it; the page redraws when it lands. Asked once (it is about 1 MB and the poll
// runs every 2 s); after a failure it is asked again when another page is opened.
let metricsReq = null;
function metricsInBackground(force = false) {
  if (metricsReq || (!force && (ST.metrics || ST.metricsAsked))) return;
  ST.metricsAsked = true;
  ST.metricsLoading = true; // a view can say "loading" rather than "not available" while it is on its way
  metricsReq = api
    .metrics()
    .then((evaluation) => {
      if (evaluation) {
        ST.metrics = evaluation;
        ST.metricsRev = (ST.metricsRev || 0) + 1;
      }
    })
    .catch(() => {})
    .finally(() => {
      metricsReq = null;
      ST.metricsLoading = false;
      draw(); // the views that read the evaluation redraw (ST.metricsRev is part of their fingerprint)
    });
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
    interpretation = null,
    missing = false,
    evaluation = null,
    spot = null;
  // After a 404 the case is not asked for again until the route or the replay (arrivals, reset) changes.
  const arrivals = wl.sim ? `${wl.sim.arrived}|${wl.sim.sim_time_s > 0}` : '';
  const skipMissing = route === 'case' && ST.notFound === caseId && ST.notFoundAt === arrivals;
  if (route === 'case' && caseId && skipMissing) missing = true;
  else if (route === 'case' && caseId) {
    try {
      // Read saved prose alongside the evidence so the initial case screen is complete and stable.
      // This is a cache lookup only; an unavailable explanation never blocks the case evidence.
      [detail, events, interpretation] = await Promise.all([
        api.getCase(caseId),
        api.events(caseId),
        ST.interp[caseId] ? null : api.interpretation(caseId)
          .then((data) => ({ status: data?.status === 'ok' ? 'ok' : 'none', data }))
          .catch((err) => ({ status: 'error', error: err.message }))
      ]);
      // optional context (held-out coverage line): fetched beside the case, never ahead of it; a failing /api/metrics
      // must not mark the case missing or offline
      metricsInBackground();
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) missing = true;
      else throw err;
    }
  } else if (route === 'metrics' && (force || !ST.metrics)) evaluation = await api.metrics();
  else if (route === 'spotcheck' && (force || !ST.spot)) spot = await api.spotcheckItems();
  else if (route === 'tcga') {
    // Story panel needs BACH held-out + external_tcga aggregates alongside the specimen index: the specimens are drawn
    // first and the panel fills in when the (about 1 MB) evaluation payload arrives.
    metricsInBackground(force);
    if (force || !ST.tcga) {
      try {
        ST.tcga = await api.tcga();
      } catch (err) {
        if (!(err instanceof ApiError && err.status === 404)) throw err;
        ST.tcga = { available: false, n: 0, items: [] };
      }
    }
    if (caseId) {
      const skipTcgaMissing = ST.notFound === caseId && ST.notFoundAt === 'tcga';
      if (skipTcgaMissing) missing = true;
      else if (!ST.cases[caseId] || force) {
        try {
          detail = await api.getTcgaCase(caseId);
        } catch (err) {
          if (err instanceof ApiError && err.status === 404) missing = true;
          else throw err;
        }
      }
    }
  } else if (route === 'oncologist' && !ST.mock) {
    // the conformal method label + coverage in the inbox copy. The worklist only names the method in a column tooltip
    // and falls back to "Confidence set"; it does not pull the 1 MB evaluation payload on first load.
    metricsInBackground();
  }
  if (seq !== pollSeq) return;
  ST.items = steadyTimers(wl.items || []);
  ST.itemsFor = JSON.stringify(worklistQuery(route)); // which query these items answer (the OR board asks for one lane)
  ST.loadedOnce = true;
  ST.sim = wl.sim || ST.sim;
  ST.health = health;
  ST.synthetic = wl.synthetic || health.synthetic || ST.items.some((c) => c.synthetic);
  if (route === 'case') {
    ST.notFound = missing ? caseId : null;
    ST.notFoundAt = missing ? arrivals : null;
    if (detail) {
      ST.cases[caseId] = steadyTimers([detail])[0];
      ST.events[caseId] = events;
      if (interpretation && !ST.interp[caseId]) {
        ST.interp[caseId] = interpretation;
        ST.interpRev++;
      }
    }
  } else if (route === 'tcga' && caseId) {
    ST.notFound = missing ? caseId : null;
    ST.notFoundAt = missing ? 'tcga' : null;
    if (detail) {
      ST.cases[caseId] = detail;
      ST.events[caseId] = ST.events[caseId] || [];
    }
  }
  noticeRouted(toSurgeon || [], toOncologist || []);
  ST.routed.surgeon = toSurgeon || [];
  ST.routed.oncologist = toOncologist || [];
  if (evaluation) {
    ST.metrics = evaluation;
    ST.metricsRev = (ST.metricsRev || 0) + 1; // "Reload results" with changed numbers redraws the dashboard
  }
  if (spot) ST.spot = spot;
}

// Frozen-section timers count from one local anchor (receipt time minus elapsed). Each poll's latency differs a little;
// re-anchoring on every response nudged the display back and forth by a second. The anchor moves only for a real
// change (a call that stops the timer, a reset), so the display advances one second per tick and never steps back.
const anchors = new Map();
function steadyTimers(cases) {
  for (const c of cases) {
    const t = c && c.runtime && c.runtime.timer;
    if (!t || t.kind !== 'surgical' || t.state === 'stopped' || !c._fetched_ms) continue;
    let anchor = c._fetched_ms - t.elapsed_s * 1000;
    const prev = anchors.get(c.case_id);
    if (prev !== undefined && Math.abs(anchor - prev) < 1500) anchor = prev;
    else anchors.set(c.case_id, anchor);
    // the same timer, read 2 s earlier: timerNow(c, tick) is then exactly (tick - anchor) even for a response that
    // arrived after the last tick (timerNow never extrapolates backwards from the receipt time)
    t.elapsed_s = Math.max(0, t.elapsed_s - 2);
    c._fetched_ms = anchor + t.elapsed_s * 1000;
  }
  return cases;
}
// The running frozen-section timer the reader is looking at (the open case, else the first on the board), if any.
function leadAnchor() {
  const lead =
    ST.route === 'case'
      ? ST.caseId
      : (ST.items.find((c) => c.lane === 'surgical' && c.runtime && c.runtime.timer && c.runtime.timer.state !== 'stopped') || {}).case_id;
  return lead && anchors.has(lead) ? anchors.get(lead) : null;
}
// One tick per second, phase-locked so a frozen-section timer's second boundaries fall half-way between ticks (else
// on the wall-clock second): each displayed second lasts exactly one tick, however the polls fall.
function steadyTick() {
  tickAt = Date.now();
  tick();
  const lead = leadAnchor();
  const phase = lead === null ? 0 : (((lead + 500) % 1000) + 1000) % 1000;
  const wait = (phase - (Date.now() % 1000) + 2000) % 1000;
  setTimeout(steadyTick, wait < 250 ? wait + 1000 : wait);
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
    if (ST.fail >= 2) {
      const unsent = dismissToasts((el) => el.dataset.net === 'unsent');
      dismissToasts((el) => !!el.dataset.net);
      toast('Connection to Sepia restored', {
        tone: 'ok',
        sub: unsent ? 'A decision that could not be sent has to be made again.' : undefined
      });
    }
    ST.fail = 0;
    ST.lastError = null;
    ST.staleSince = null;
    ST.lastOk = new Date();
    checkUnanswered();
  } catch (err) {
    if (seq !== pollSeq) return;
    ST.fail += 1;
    ST.lastError = { status: err instanceof ApiError ? err.status : 0 };
    if (!ST.staleSince) ST.staleSince = ST.lastOk || new Date(); // the screen shows data from the last good poll
    if (!(err instanceof ApiError) || err.status === 0 || err.status >= 500) console.warn('poll failed', err);
  }
  const base = ST.cfg ? ST.cfg.poll_ms : 2000;
  // Background screens keep synchronizing, with a lower polling frequency. Scheduled before drawing: a view that
  // throws on unexpected data must not stop the poll loop for good.
  pollTimer = setTimeout(poll, document.hidden ? base * 5 : ST.fail >= 5 ? base * 2.5 : base);
  showOffline();
  draw();
  if (ST.loadedOnce) prefetchViews();
}

// Two failed polls in a row: say whether Sepia is not reachable or answers with an error, and how old the screen is.
function showOffline() {
  const el = $('#offline');
  if (ST.fail >= 2) {
    const status = (ST.lastError && ST.lastError.status) || 0;
    const since = `what you see is from ${esc(ST.staleSince.toLocaleTimeString())}`;
    const html = status
      ? `<b>Sepia answered with an error</b> (HTTP ${status}). Retrying every few seconds; ${since}. ${localService ? 'The service log has the details.' : 'This is on the service side; it usually clears on its own.'}`
      : `<b>Reconnecting to Sepia</b>. Retrying every few seconds; ${since}. ${connectionHelp}`;
    if (el.innerHTML !== html) el.innerHTML = html;
    el.hidden = false;
  } else el.hidden = true;
}

// ---- rendering ------------------------------------------------------------------------------------------------------------
function view() {
  if (ST.route === 'case') {
    if (ST.notFound === ST.caseId)
      return `<button class="back" id="bk" data-testid="back">Back to worklist</button><div class="card" data-testid="case-not-found"><h2>Case not found</h2><p>${notFoundText(ST.caseId)}</p></div>`;
    const c = ST.cases[ST.caseId];
    if (!c) return '<div class="card" data-testid="state-loading"><h2>Loading case…</h2></div>';
    return (c.lane === 'surgical' ? surgical : biopsy).render(ST, ctx());
  }
  if (LAZY[ST.route] && !views[ST.route]) {
    if (viewErrors[ST.route])
      return `<div class="card" data-testid="state-render-error"><h2>This page could not be loaded</h2><p>Its code did not arrive (${esc(viewErrors[ST.route].message || viewErrors[ST.route])}). Check the connection, then open the page again or reload.</p></div>`;
    loadView(ST.route).catch((err) => console.warn('page module failed to load', err));
    return lazyPlaceholder();
  }
  return ({ worklist, surgeon, oncologist, ...views })[ST.route].render(ST, ctx());
}

// Why a worklist address shows nothing: an external TCGA specimen has its own page, and the simulated frozen sections
// that are held for the OR controls arrive only when one is sent.
function notFoundText(id) {
  const tcga = /^TCGA-/i.test(id) || !!(ST.tcga && (ST.tcga.items || []).some((it) => it.case_id === id));
  if (tcga)
    return `${esc(id)} is an external TCGA specimen, not a worklist case. <a href="#/tcga/${encodeURIComponent(id)}" data-testid="not-found-tcga">Open it in External test</a>.`;
  const held = ST.sim && ST.sim.hold_surgical_for_injection;
  if (/^COMP_/i.test(id))
    return `${esc(id)} is a simulated composite frozen section.${held ? ' It reaches the worklist only when one is sent with <b>+ Composite</b> in the header, and a reset clears it.' : ' It has not arrived yet, or a reset cleared it.'}`;
  if (/_FSIM_/i.test(id))
    return `${esc(id)} is a simulated degraded frozen section.${held ? ' It reaches the worklist only when one is sent with <b>+ Degraded</b> in the header, and a reset clears it.' : ' It has not arrived yet, or a reset cleared it.'}`;
  return `${esc(id)} is unknown or has not arrived yet. Arrivals appear as the replay runs; a reset clears earlier cases.`;
}

// While a page's module loads: the page's own loading state (the same markup its render() draws before its data
// arrives), so the screen does not change from one placeholder to another.
function lazyPlaceholder() {
  if (ST.route === 'metrics')
    return '<div data-testid="metrics" class="mx"><div class="card" data-testid="state-loading"><h2>Loading evaluation results…</h2></div></div>';
  if (ST.route === 'spotcheck')
    return '<div class="sc card" data-testid="spotcheck"><h2>Evidence review</h2><p class="note" data-testid="state-loading">Loading review items…</p></div>';
  if (ST.caseId) return '<div class="card" data-testid="state-loading"><h2>Loading specimen…</h2></div>';
  return '<div class="mx tcga-page" data-testid="tcga-loading"><header class="mx-head tcga-claim"><h2 class="tcga-intro">External test (TCGA-BRCA)</h2><p class="mx-lede">Loading external generalization results…</p></header></div>';
}

function currentFp() {
  // Pages that do not read the queue redraw only for their own data: a running replay must not re-render the
  // dashboard (charts lose focus and tooltips) or the evidence review (images reload under the reviewer).
  if (ST.route === 'metrics') return ['metrics', ST.metricsRev || 0, ST.interpRev, !!ST.metrics].join('|');
  if (ST.route === 'tcga') {
    if (ST.caseId)
      return [
        'tcga-case',
        ST.caseId,
        ST.tab,
        ST.notFound === ST.caseId ? 'missing' : !!ST.cases[ST.caseId],
        JSON.stringify(ST.hm),
        ST.interpRev,
        ST.metricsRev || 0
      ].join('|');
    return [
      'tcga',
      ST.tcga ? ST.tcga.n : 'none',
      !!(ST.tcga && ST.tcga.available),
      ST.metricsRev || 0,
      !!ST.metricsLoading,
      !!(ST.metrics && ST.metrics.eval && ST.metrics.eval.external_tcga)
    ].join('|');
  }
  if (ST.route === 'spotcheck')
    return [
      'spotcheck',
      ST.spot ? ST.spot.items.length : 'none',
      JSON.stringify(ST.spotDraft),
      ST.spotResult || '',
      ST.spotGroup || '',
      ST.metricsRev || 0
    ].join('|');
  if (ST.route === 'oncologist')
    return [
      'oncologist',
      JSON.stringify(ST.routed.oncologist.map(routedKey)),
      [...ST.inboxRead].length,
      ST.inboxOpen || '',
      ST.metricsRev || 0
    ].join('|');
  const sim = ST.sim || {};
  const parts = [
    ST.metricsRev || 0,
    ST.interpRev,
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
    // the case view does not read the queue (the case bar is drawn separately): arrivals do not redraw it
    ST.route === 'case' ? '' : fingerprint(ST.items),
    sim.running,
    ST.route === 'case' ? '' : JSON.stringify(sim.pending),
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
    parts.push(JSON.stringify(ST.routed.surgeon.map(routedKey)), [...ST.freshCalls].join());
  if (ST.route === 'worklist') parts.push(ST.loadedOnce, freshArrivals().join());
  return parts.join('|');
}

// Arrivals still inside their "New" window; the list changes when one ages out, so the tag is dropped on the next poll.
function freshArrivals(now = Date.now()) {
  const fresh = [];
  ST.arrived.forEach((t, id) => (now - t < worklist.NEW_MS ? fresh.push(id) : ST.arrived.delete(id)));
  return fresh;
}
// Cases that appear after the first rendered data are arrivals; whatever was there on load (or on a lane switch) is not.
function noteArrivals() {
  if (ST.route !== 'worklist' || !ST.loadedOnce || ST.itemsFor !== JSON.stringify(worklistQuery('worklist'))) return;
  const now = Date.now();
  const fresh = [];
  ST.items.forEach((c) => {
    if (ST.seen.has(c.case_id)) return;
    ST.seen.add(c.case_id);
    if (ST.firstLoad) return;
    ST.arrived.set(c.case_id, now);
    fresh.push(c);
  });
  ST.firstLoad = false;
  announceArrivals(fresh);
  if (fresh.length) speakArrivals(fresh.length);
}
// Screen readers hear one short summary at most every 10 s ("3 new cases in the worklist"), not every row.
let spoken = { n: 0, timer: null };
function speakArrivals(n) {
  spoken.n += n;
  if (spoken.timer) return;
  spoken.timer = setTimeout(() => {
    $('#arrivals-live').textContent = `${spoken.n} new case${spoken.n === 1 ? '' : 's'} in the worklist`;
    spoken = { n: 0, timer: null };
  }, 10000);
}
// A likely-invasive biopsy sorts into tier 2 by P(Invasive), often below the fold, so its arrival gets one short notice
// with a way to open it. Arrivals in the same poll share the notice, and a newer notice replaces the last one. Frozen
// sections are announced when they are sent from the OR controls.
function announceArrivals(fresh) {
  const urgent = fresh.filter((c) => c.lane !== 'surgical' && c.runtime && !c.runtime.done && c.runtime.tier <= 2);
  if (!urgent.length) return;
  const newest = urgent.reduce((a, b) => ((b.runtime.arrived_sim_s || 0) > (a.runtime.arrived_sim_s || 0) ? b : a));
  const pos = queuePosition(ST.items, newest.case_id);
  const where = pos ? `position ${pos} of ${openItems(ST.items).length} open` : 'in the worklist';
  const label = tierInfo(newest.runtime.tier, ST.cfg).label;
  const tiers = [...new Set(urgent.map((c) => c.runtime.tier))].sort().join('–');
  toast(urgent.length === 1 ? `New tier ${tiers} arrival: ${newest.case_id}` : `${urgent.length} new tier ${tiers} arrivals`, {
    key: 'arrival',
    sub: urgent.length === 1 ? `${label} · ${where}` : `Newest ${newest.case_id} · ${label} · ${where}`,
    action: { label: 'Open', href: '#/case/' + encodeURIComponent(newest.case_id) }
  });
}

// ---- in-place patching (worklist) --------------------------------------------------------------------------------------
// The worklist redraws on every arrival. Patching the live tree instead of replacing it keeps each row's identity, so
// hover, focus, native tooltips, an open popover and the arrival highlight survive the poll, and nothing flickers.
const morphKey = (n) => (n.nodeType === 1 ? n.getAttribute('data-case-id') || n.getAttribute('data-group') : null);
// UI state set by script (an open <details>, data-ui-* attributes) is not part of the markup and is kept.
const keepAttr = (el, name) => (name === 'open' && el.tagName === 'DETAILS') || name.startsWith('data-ui-');
function morph(live, next) {
  for (const { name } of [...live.attributes]) if (!next.hasAttribute(name) && !keepAttr(live, name)) live.removeAttribute(name);
  for (const { name, value } of [...next.attributes]) if (live.getAttribute(name) !== value) live.setAttribute(name, value);
  if (live.tagName === 'INPUT') {
    if (live.type === 'checkbox' || live.type === 'radio') live.checked = next.hasAttribute('checked');
    else if (live.value !== (next.getAttribute('value') || '')) live.value = next.getAttribute('value') || '';
  } else if (live.tagName === 'OPTION') live.selected = next.hasAttribute('selected');
  const pool = new Map();
  for (const c of live.childNodes) if (morphKey(c)) pool.set(morphKey(c), c);
  let cur = live.firstChild;
  for (const want of [...next.childNodes]) {
    const key = morphKey(want);
    let match = key ? pool.get(key) : cur && !morphKey(cur) && cur.nodeType === want.nodeType ? cur : null;
    if (match && match.nodeName !== want.nodeName) match = null;
    if (!match) {
      live.insertBefore(want, cur);
      continue;
    }
    if (key) pool.delete(key);
    if (match === cur) cur = cur.nextSibling;
    else live.insertBefore(match, cur);
    if (match.nodeType === 1) morph(match, want);
    else if (match.nodeValue !== want.nodeValue) match.nodeValue = want.nodeValue;
  }
  while (cur) {
    const n = cur.nextSibling;
    live.removeChild(cur);
    cur = n;
  }
}
function patchApp(html) {
  const app = $('#app');
  const live = app.firstElementChild;
  // The loading placeholders are replaced, not patched: content that is new on screen is not a layout shift, while
  // placeholders reshaped into the real summary and toolbar would be. The redraw that follows a replay command is
  // replaced too: it can reorder the whole queue, and a row the user aimed at before it must not stay clickable.
  const replace = ST.simBusy || !live || !live.matches('section.wl:not(.wl-loading)');
  if (ST.route === 'worklist' && app.childElementCount === 1 && !replace) {
    const tpl = document.createElement('template');
    tpl.innerHTML = html;
    const next = tpl.content.firstElementChild;
    if (tpl.content.childElementCount === 1 && next.matches('section.wl')) {
      morph(live, next);
      return true;
    }
  }
  app.innerHTML = html;
  return false;
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
  noteArrivals();
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
  let html;
  try {
    html = view();
  } catch (err) {
    console.error('view failed to render', err);
    html = `<div class="card" data-testid="state-render-error"><h2>This page could not be drawn</h2><p>${esc(err && err.message ? err.message : err)}. The data keeps refreshing in the background; other pages still work.</p></div>`;
  }
  const patched = patchApp(html);
  // tab switches inside a case redraw only the panel: the page-level entrance is for a new route or case
  animateScene(`${ST.route}/${ST.caseId || ''}`, ST.tab);
  if (patched) {
    // the focused element is still the same node unless the poll removed it (e.g. a filtered-out row)
    if (focusSel && !$('#app').contains(document.activeElement))
      (findFocus(focusSel) || $('#app')).focus({ preventScroll: true });
  } else if (searchFocused && $('#case-search')) {
    $('#case-search').focus({ preventScroll: true });
    $('#case-search').setSelectionRange(caret, caret);
  } else if (focusSel) {
    const el = findFocus(focusSel);
    (el || $('#app')).focus({ preventScroll: true });
  }
  if (openDetails.length && !patched)
    $('#app')
      .querySelectorAll('details')
      .forEach((el) => {
        if (openDetails.includes(detailsKey(el))) el.open = true;
      });
  if (!patched) window.scrollTo(0, y);
  if (ST.flashTarget) {
    const el = $(ST.flashTarget);
    if (el) {
      el.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' });
      el.classList.add('flash');
    }
    ST.flashTarget = null;
  }
  tick();
  wantInterp();
}

const PAGE = {
  worklist: 'Worklist',
  case: 'Case review',
  surgeon: 'Operating rooms',
  oncologist: 'Oncology inbox',
  metrics: 'Model evaluation',
  tcga: 'External test',
  spotcheck: 'Evidence review'
};

function pageDescription() {
  // queue counts are unknown until the first worklist response ("0 open" while loading would be wrong)
  if (!ST.loadedOnce && ['worklist', 'surgeon', 'oncologist'].includes(ST.route)) return 'Loading…';
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
  if (ST.route === 'metrics') return 'Held-out test split · 95% CIs';
  if (ST.route === 'tcga') {
    if (ST.caseId) return tcgaName(ST.caseId);
    const n = ST.tcga && ST.tcga.n;
    return n ? `TCGA-BRCA · ${n} fields` : 'TCGA-BRCA';
  }
  return 'Model-highlighted regions';
}

// "TCGA-A7-A4SD_ffpe_tumor_00" -> "TCGA-A7-A4SD · FFPE · tumor · field 0" (the specimen list's wording)
function tcgaName(id) {
  const m = /^(TCGA-[A-Z0-9]+-[A-Z0-9]+)_(ffpe|frozen)_(tumor|normal)_(\d+)$/i.exec(String(id || ''));
  return m ? `${m[1]} · ${m[2].toLowerCase() === 'ffpe' ? 'FFPE' : 'Frozen'} · ${m[3].toLowerCase()} · field ${Number(m[4])}` : id;
}

function updateShell() {
  const crumb = PAGE[ST.route] || PAGE.worklist;
  $('#breadcrumb').textContent = crumb;
  const desc = pageDescription();
  $('#page-description').textContent = desc;
  $('#page-description').title = desc; // the header truncates it on narrower screens
  const titleId = ST.route === 'tcga' && ST.caseId ? tcgaName(ST.caseId) : ST.caseId;
  document.title = `${(ST.route === 'case' || ST.route === 'tcga') && ST.caseId ? titleId + ' · ' : ''}${crumb} · Sepia`;
  $('.workspace').dataset.route = ST.route;
  // TCGA external is a static specimen browser — hide arrival-replay chrome entirely. The root attribute is also set
  // before first paint (index.html), so a #/tcga load does not show the controls and then collapse them.
  document.documentElement.dataset.route = ST.route;
  const replay = $('.replay');
  if (replay) replay.hidden = ST.route === 'tcga';
  $('#replay-toggle').hidden = ST.route === 'tcga';
  // empty until the first worklist response: a "0" badge on load would be a wrong count for a moment
  const queued = ST.loadedOnce ? ST.health?.open_cases ?? openItems(ST.items).length : '';
  $('#queue-count').textContent = queued;
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
  // The destinations are named by their full label (phones show a short one); a count shown on one is part of its name.
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  navName('worklist', 'Worklist', queued === '' ? '' : plural(queued, 'open case', 'open cases'));
  navName('surgeon', 'Operating rooms', liveFrozen ? plural(liveFrozen, 'frozen section awaiting a call', 'frozen sections awaiting a call') : '');
  navName('oncologist', 'Oncology inbox', ST.inboxUnread ? plural(ST.inboxUnread, 'unread result', 'unread results') : '');
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
    ? 'Connecting to Sepia…'
    : `${h.loaded} cases${h.assets?.missing.length ? ` · ${h.assets.missing.length} missing images` : ''}${h.skipped.length ? ` · ${h.skipped.length} rejected` : ''} · ${ST.mock ? 'in-memory demo' : localService ? 'local workspace' : 'shared research workspace'}`;
  $('.data-status').dataset.state =
    offline || !h || !h.loaded || h.synthetic || h.skipped.length || h.assets?.missing.length
      ? 'pending'
      : 'ready';
  $('#connection').textContent = offline ? 'Reconnecting' : h ? 'Connected' : 'Connecting…';
  $('#connection').title = offline ? 'Reconnecting to Sepia' : 'Sepia service';
  $('#connection').dataset.state = offline ? 'offline' : h ? 'online' : 'pending';
  updateReplay();
  syncGlass();
}

function navName(route, label, count) {
  const btn = $(`.glass-nav [data-nav="${route}"]`);
  const name = count ? `${label}, ${count}` : label;
  if (btn.getAttribute('aria-label') !== name) btn.setAttribute('aria-label', name);
}

// Phones and short landscape screens fold the replay and frozen-section controls behind #replay-toggle (theme.css);
// elsewhere the toggle is not displayed and its state does not matter.
function setReplayOpen(open, { focusToggle = false } = {}) {
  const toggle = $('#replay-toggle');
  if (toggle.getAttribute('aria-expanded') === String(open)) return;
  toggle.setAttribute('aria-expanded', String(open));
  if (focusToggle) toggle.focus();
}

function updateReplay() {
  const sim = ST.sim;
  const h = ST.health;
  const box = $('.replay');
  // A replay command can reorder the queue. Keep its old rows from opening while the new order loads.
  $('#app').dataset.replayBusy = String(!!ST.simBusy);
  $('#app').setAttribute('aria-busy', String(!!ST.simBusy));
  const start = $('[data-testid=sim-start]');
  const pp = $('#pp');
  // offline: the commands would fail, and the replay state on screen is stale
  const loaded = !!(h && h.loaded) && ST.fail < 2;
  if (!sim) {
    start.disabled = true;
    return;
  }
  const fresh = !sim.running && !(sim.sim_time_s > 0);
  box.dataset.state = ST.fail >= 2 ? 'paused' : sim.held ? 'held' : sim.running ? 'running' : fresh ? 'ready' : 'paused';
  $('#replay-toggle').dataset.state = box.dataset.state;
  start.hidden = !fresh;
  start.disabled = !loaded || ST.simBusy;
  pp.hidden = fresh;
  pp.setAttribute('aria-pressed', String(!sim.running));
  pp.textContent = sim.running ? 'Pause replay' : 'Resume replay';
  pp.dataset.sim = sim.running ? 'pause' : 'start';
  pp.disabled = !loaded || ST.simBusy;
  $('[data-testid=sim-reset]').disabled = ST.simBusy || !loaded;
  // the buttons ship disabled (index.html): nothing can be sent before the first reply
  const pending = sim.pending ? sim.pending.surgical : null;
  document.querySelectorAll('[data-sim=inject_surgical]').forEach((b) => {
    if (!pending) {
      b.disabled = ST.simBusy || !loaded;
      return;
    }
    const n = pending[b.dataset.variant] || 0;
    b.disabled = n === 0 || ST.simBusy || !loaded;
    b.title = n ? `${n} simulated ${b.dataset.variant === 'normal' ? '' : b.dataset.variant + ' '}frozen sections left` : 'None left: reset the replay to send more';
    const tag = b.querySelector('[data-pending]');
    if (tag) tag.textContent = n;
  });
}

function lookup(id) {
  if (ST.route === 'case' && ST.cases[id]) return ST.cases[id];
  return ST.items.find((c) => c.case_id === id);
}

function tick() {
  const now = tickAt; // a poll's redraw shows the same second as the last tick, so timers never advance early
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
  // compact waiting cells (worklist): biopsies "12:24 / 24 h", frozen sections "08:41 / 20:00"
  document.querySelectorAll('[data-wait]').forEach((el) => {
    const c = lookup(el.dataset.wait);
    if (!c) return;
    const w = waitInfo(timerNow(c, now, ST.cfg, ST.sim));
    if (el.dataset.state !== w.state) {
      // a new state can bring a new shape ("Signed out after 44 min"): redraw the cell
      el.dataset.state = w.state;
      el.className = 'wait ' + w.state;
      el.innerHTML = worklist.waitInner(w);
      return;
    }
    const b = el.querySelector('b');
    if (b && b.textContent !== w.text) b.textContent = w.text;
  });
  tickClock();
}

const simShown = { raw: 0, t: 0 }; // the replay clock last shown (seconds) and the poll value it was based on
function tickClock() {
  const clock = $('#clock');
  const d = new Date(tickAt);
  const text = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  if (clock.textContent !== text) {
    clock.textContent = text;
    clock.dateTime = d.toISOString();
  }
  const sim = ST.sim;
  if (!sim) return;
  const offline = ST.fail >= 2;
  // Held (sim.held: every scheduled case is in and the clock reached its hold, docs/API.md): the clock stands still,
  // although the replay may still be running for frozen sections. Between polls it is never run past the hold.
  const held = !!sim.held;
  const room = sim.hold_at_s == null ? Infinity : Math.max(0, sim.hold_at_s - (Number(sim.sim_time_s) || 0));
  const since =
    sim._fetched_ms && sim.running && !held && !offline
      ? Math.min(room, (Math.max(0, tickAt - sim._fetched_ms) / 1000) * (Number(sim.speed) || 1))
      : 0;
  const fresh = !sim.running && !(sim.sim_time_s > 0);
  const state = offline
    ? 'Replay status unknown'
    : held
      ? 'Replay complete; clock held'
      : sim.running
        ? 'Replay running'
        : fresh
          ? 'Replay ready'
          : 'Replay paused';
  const speed = Number(sim.speed) || 1;
  // Built once, then only the text changes: rebuilding every second restarted the running dot's pulse mid-cycle.
  const box = $('#sim');
  if (!box.querySelector('.rs-t'))
    box.innerHTML =
      // spaces between the parts: a screen reader reads "Replay running +4:02 ×30", not one run-together word
      '<i class="rs-dot" aria-hidden="true"></i><b></b> <span class="rs-t"></span> <span class="rs-x" title="Replay speed"></span>' +
      `${ST.mock ? '<span class="rs-x">mock</span>' : ''}<span class="sr-only"></span>`;
  const set = (sel, text) => {
    const el = box.querySelector(sel);
    if (el.textContent !== text) el.textContent = text;
  };
  // The clock runs on between polls at the replay speed; a poll that lands a little behind the extrapolation (the speed
  // changed, a slow reply) never moves it back. Only a reset, which restarts the replay clock, does.
  const raw = Number(sim.sim_time_s) || 0;
  if (raw < simShown.raw) simShown.t = 0;
  simShown.raw = raw;
  simShown.t = Math.max(simShown.t, raw + since);
  const simClock = '+' + mmss(simShown.t);
  set('b', state);
  set('.rs-t', simClock);
  set('.rs-x', held ? '' : '×' + (speed >= 10 ? Math.round(speed) : Number(speed.toFixed(1)))); // no speed while held
  set('.sr-only', ` · simulator ${offline ? 'status unknown' : held ? 'clock held' : sim.running ? 'running' : fresh ? 'ready' : 'paused'}`);
  // the folded controls' button reads the same state ("Replay running +4:02"; held: "Replay complete clock held")
  const toggle = $('#replay-toggle');
  for (const [sel, text] of [['b', held && !offline ? 'Replay complete' : state], ['.rs-t', held && !offline ? 'clock held' : simClock]]) {
    const el = toggle.querySelector(sel);
    if (el.textContent !== text) el.textContent = text;
  }
}

// ---- case navigation bar (shell): position in the priority queue + the next case ------------------------------------
function renderCasebar() {
  const bar = $('#casebar');
  if (ST.route !== 'case' || !ST.caseId || ST.notFound === ST.caseId) {
    if (!bar.hidden) {
      bar.hidden = true;
      bar.innerHTML = '';
    }
    bar.dataset.off = ''; // no reserved space (theme.css) for a case that is not there
    return;
  }
  delete bar.dataset.off;
  const c = ST.cases[ST.caseId];
  const done = !!(c && c.runtime && c.runtime.done);
  const pos = queuePosition(ST.items, ST.caseId);
  const open = openItems(ST.items).length;
  const next = nextCase(ST.items, ST.caseId);
  const nextTier = next ? tierInfo(next.runtime ? next.runtime.tier : (next.flag || {}).tier ?? 4, ST.cfg) : null;
  let status;
  if (done && c.lane === 'surgical') {
    const call = c.runtime.decision.margin_call;
    status = `<span class="cb-done" data-testid="casebar-done"><b>Call sent to ${esc(c.runtime.or_room || 'the OR')}</b> ${esc(callText(call))}</span>`;
  } else if (done) {
    status = '<span class="cb-done" data-testid="casebar-done"><b>Signed out</b> Result sent to the oncology inbox</span>';
  } else {
    status = pos
      ? `<span class="cb-pos">Case <b>${pos}</b> of ${open} open<span class="cb-more"> in priority order</span></span>`
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
// Replay commands: the follow-up poll is awaited, since the notice that follows reads the new queue.
async function act(fn, okMsg) {
  try {
    const res = await fn();
    if (okMsg) toast(okMsg);
    return res;
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) toast('Not allowed', { tone: 'warn', sub: err.detail });
    else if (err instanceof ApiError && err.status === 422) toast('Request rejected', { tone: 'warn', sub: err.detail });
    else if (err instanceof ApiError && err.status === 0)
      toast('Sepia is not reachable; the replay command was not sent', { tone: 'warn', net: true, sub: err.timedOut ? 'No reply in time.' : undefined });
    else toast('Request failed', { tone: 'warn', sub: String(err.message || err), net: err instanceof ApiError && err.status >= 500 });
    return null;
  } finally {
    await poll(true);
  }
}

// ---- decisions ------------------------------------------------------------------------------------------------------------
// What each action leaves on the case (backend/db.py): how a decision whose reply was lost is recognised.
const AGENT_TARGET = { approve_orders: 'approved', edit_orders: 'edited_and_approved', closer_look: 'deferred_closer_look', dismiss_flag: 'dismissed' };
function decisionApplied(c, action) {
  if (!c) return false;
  const d = (c.runtime && c.runtime.decision) || {};
  if (action === 'sign_out') return !!d.signed_out_at;
  if (action === 'margin_call') return !!d.margin_call;
  return !!(c.agent && c.agent.status === AGENT_TARGET[action]);
}
// A decision sent without a reply: it may still be applied. Each poll looks for it; once a poll shows it, the warning
// gives way to the usual confirmation. Watched for two minutes.
let unanswered = null;
function checkUnanswered() {
  if (!unanswered) return;
  const { id, action, key, since } = unanswered;
  // the open case is refreshed by every poll; any other case is read from the queue
  const c = (ST.route === 'case' && ST.caseId === id && ST.cases[id]) || ST.items.find((x) => x.case_id === id);
  if (c && decisionApplied(c, action)) {
    unanswered = null;
    dismissToasts((el) => el.dataset.key === key);
    announceDecision(action, { case: c });
  } else if (Date.now() - since > 120000) unanswered = null;
}
// A decision request that failed: a conflict or a rejection says why; with no reply, the case is read again first,
// since the request may have reached Sepia. Returns {case} when the decision turns out to be applied.
async function decisionFailed(id, action, err) {
  if (!(err instanceof ApiError)) {
    toast('Request failed', { tone: 'warn', sub: String(err.message || err) });
    return null;
  }
  if (err.status === 409) {
    toast('Already decided in another window', { sub: err.detail });
    return null;
  }
  if (err.status === 422) {
    toast('Request rejected', { tone: 'warn', sub: err.detail });
    return null;
  }
  if (err.status !== 0) {
    toast('Request failed', { tone: 'warn', sub: err.detail || `HTTP ${err.status}`, net: err.status >= 500 });
    return null;
  }
  let fresh = null;
  try {
    fresh = await api.getCase(id);
  } catch (_) {}
  if (fresh && decisionApplied(fresh, action)) return { case: fresh };
  if (!err.timedOut) {
    toast('Sepia is not reachable; your decision was not sent', { tone: 'warn', net: 'unsent', key: `unsent:${id}` });
    return null;
  }
  const key = `unanswered:${id}:${action}`;
  unanswered = { id, action, key, since: Date.now() };
  toast('No reply from Sepia; check the case before retrying', {
    tone: 'warn',
    key,
    sub: 'The decision may still be recorded. This notice is replaced when it shows on the case.'
  });
  return null;
}

function orderList(c) {
  return ((c.agent && c.agent.orders) || []).map((o) => o.items.join(', ')).join(' · ');
}

function announceDecision(action, res) {
  const c = res.case;
  const id = c.case_id;
  // approval in this prototype: nothing is sent to a laboratory system
  if (action === 'approve_orders') toast(`Orders approved for ${id} (not sent to a laboratory system)`, { tone: 'ok', sub: orderList(c) });
  else if (action === 'edit_orders') toast(`Edited orders approved for ${id} (not sent to a laboratory system)`, { tone: 'ok', sub: orderList(c) });
  else if (action === 'closer_look') toast(`${id} marked for a closer look`, { sub: 'Drafted orders stay unplaced until you decide.' });
  else if (action === 'dismiss_flag') toast(`Flag dismissed for ${id}`, { sub: 'No follow-up orders were placed.' });
  else if (action === 'sign_out')
    toast(`${id} signed out`, { tone: 'ok', sub: 'Result sent to the oncology inbox', action: nextCase(ST.items, id) ? { label: 'Open next case', next: true } : { label: 'View inbox', href: '#/oncologist' } });
  else if (action === 'margin_call') {
    const call = c.runtime && c.runtime.decision && c.runtime.decision.margin_call;
    toast(`Margin call sent to ${(c.runtime && c.runtime.or_room) || 'the OR'}`, { tone: 'surg', sub: callText(call), action: { label: 'View OR board', href: '#/surgeon' } });
  }
}

// The reply is applied and confirmed at once; the poll that refreshes the queue follows without holding the verdict.
async function decide(action, payload) {
  if (ST.busy) return; // no double submits (would only produce a confusing 409)
  const id = ST.caseId;
  ST.busy = true;
  draw(true);
  let res = null;
  try {
    res = await api.decide(id, action, payload);
  } catch (err) {
    res = await decisionFailed(id, action, err);
  }
  ST.busy = false;
  if (res && res.case) {
    ST.cases[id] = res.case;
    ST.editing = null;
    if (action === 'margin_call') ST.marginPick = [];
    if (res.case.runtime && res.case.runtime.done) ST.justDone = id;
    announceDecision(action, res);
  } // a rejected edit keeps the editor open
  draw(true);
  poll(true);
}

async function simCommand(cmd, variant) {
  if (ST.simBusy || !ST.cfg) return; // nothing is sent before the app has connected
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
    // a search or filter left on the worklist would hide the new case: say so instead of "top of the worklist"
    const hidden = ST.lane === 'biopsy' || (item && !worklist.filterViews(ST, [toView(item, Date.now(), ST.cfg, ST.sim)]).length);
    toast(`${kind} ${res.injected} received${room ? ' from ' + room : ''}`, {
      tone: 'surg',
      sub: `Simulated · timer started · ${hidden ? 'hidden by the current worklist filters' : 'top of the worklist'}`,
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
  ST.arrived.clear();
  ST.firstLoad = true;
  ST.tierFilter = null;
  ST.freshCalls.clear();
  ST.inboxOpen = null;
  ST.justDone = null;
  unanswered = null;
  // notices that open a case point at cases the reset removed
  dismissToasts((el) => el.dataset.key === 'arrival' || !!el.querySelector('a.toast-act[href^="#/case/"]'));
  if (ST.route === 'case') {
    location.hash = '#/worklist';
    onRoute(); // synchronously: the hashchange event fires after the follow-up poll has started
  }
}

// ---- AI interpretation (js/views/interpret.js) -----------------------------------------------------------------------------
function setInterp(key, patch) {
  ST.interp[key] = { ...(ST.interp[key] || {}), ...patch };
  ST.interpRev++;
  draw(true);
}
async function loadInterp(key) {
  if (ST.interp[key]) return;
  ST.interp[key] = { status: 'loading' };
  try {
    const data = await api.interpretation(key);
    setInterp(key, data && data.status === 'ok' ? { status: 'ok', data } : { status: 'none', data });
  } catch (err) {
    setInterp(key, { status: 'error', error: err.message });
  }
}
async function generateInterp(key, force) {
  if ((ST.interp[key] || {}).status === 'generating') return;
  setInterp(key, { status: 'generating', startedAt: Date.now(), error: null });
  const ticker = setInterval(() => {
    const el = document.querySelector(`[data-ai-elapsed="${ST.interp[key] && ST.interp[key].startedAt}"]`);
    if (el) el.textContent = Math.round((Date.now() - ST.interp[key].startedAt) / 1000);
  }, 1000);
  try {
    const data = await api.interpret(key, force);
    setInterp(key, { status: 'ok', data });
    toast(key === '__results' ? 'Results interpretation ready' : `Interpretation ready for ${key}`, { tone: 'ok' });
  } catch (err) {
    setInterp(key, { status: 'error', error: err.message });
  } finally {
    clearInterval(ticker);
  }
}
// Fetch the cached text when a view that shows the panel is on screen (never generates on its own).
function wantInterp() {
  if (ST.route === 'metrics') loadInterp('__results');
  else if (ST.route === 'case' && ST.caseId && ST.cases[ST.caseId]) {
    loadInterp(ST.caseId);
  }
}

function findingClick(fid) {
  const c = ST.cases[ST.caseId];
  const f = c && (c.findings || []).find((x) => x.id === fid);
  if (!f) return;
  if (ST.tab === 'interp') ST.tab = 'evidence'; // a citation in the interpretation opens the evidence it names
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
  if (e.target.closest('a.skip')) {
    e.preventDefault(); // "#app" is the skip target, not a route
    $('#app').focus();
    return;
  }
  const note = e.target.closest('.toast-item');
  if (note && !e.target.closest('a, button')) {
    dismissToast(note);
    return;
  }
  const t = e.target.closest('button, [data-nav], tr.row, a, [data-inbox-key]');
  if (!t) return;
  const d = t.dataset;
  if (t.id === 'replay-toggle') {
    setReplayOpen(t.getAttribute('aria-expanded') !== 'true');
    return;
  }
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
    dismissToast(t.closest('.toast-item'));
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
    if (ST.cfg && !t.disabled) $('#reset-dialog').showModal();
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
  if (t.matches('tr.row a.cid')) {
    if (ST.simBusy) e.preventDefault(); // like the row: not while a replay command reorders the queue
    return; // otherwise the link navigates
  }
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
  if (d.clearFilter) {
    // one active-filter chip: the rest of the filters stay as they are
    const f = d.clearFilter;
    if (f === 'search') ST.search = '';
    else if (f === 'status') ST.status = 'all';
    else if (f === 'tier') ST.tierFilter = null;
    else if (f === 'flag') ST.flagFilter = 'all';
    else if (f === 'orders') ST.ordersOnly = false;
    ST.page = 0;
    draw(true);
    ($('#app [data-clear-filter]') || $('#case-search'))?.focus();
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
    ST.firstLoad = true; // cases another lane brings into view are not arrivals
    await poll(true);
    return;
  }
  if (t.matches('tr.row')) {
    if (ST.simBusy && ST.route !== 'tcga') return;
    const id = t.dataset.caseId;
    if (!id) return;
    go(ST.route === 'tcga' ? '#/tcga/' + encodeURIComponent(id) : '#/case/' + encodeURIComponent(id));
    return;
  }
  if (d.tab) {
    ST.tab = d.tab;
    draw(true);
    return;
  }
  if (d.interp) {
    generateInterp(d.interpKey, d.interp === 'regenerate');
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
    await (await loadView('spotcheck')).submit(ST, api, toast);
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
// Single-key shortcuts (J/K/N, "/") act only from the page content: nothing focused, the workspace or a worklist row.
// With a button, link, menu or field focused the key is left alone (WCAG 2.1.4).
function shortcutTarget(el) {
  if (!el || el === document.body || el === document.documentElement) return true;
  return $('#app').contains(el) && !isTyping(el) && !el.closest('button, a, summary, [role=button]');
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
    // folded replay controls (phones): Escape from inside them closes them and returns to their button
    const toggle = $('#replay-toggle');
    if (e.target === toggle || (e.target.closest && e.target.closest('#replay-panel'))) {
      const folded = getComputedStyle(toggle).display !== 'none';
      setReplayOpen(false, { focusToggle: folded && e.target !== toggle });
    }
    if (e.target.id === 'case-search' && e.target.value) {
      ST.search = '';
      draw(true);
    }
  }
  const r = e.target.closest && e.target.closest('tr.row');
  if (r && (e.key === 'Enter' || e.key === ' ')) {
    e.preventDefault();
    if (ST.simBusy && ST.route !== 'tcga') return;
    const id = r.dataset.caseId;
    if (!id) return;
    go(ST.route === 'tcga' ? '#/tcga/' + encodeURIComponent(id) : '#/case/' + encodeURIComponent(id));
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey || !shortcutTarget(e.target) || $('#reset-dialog').open) return;
  if (ST.route === 'worklist' || (ST.route === 'tcga' && !ST.caseId)) {
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
  }
  if (ST.route === 'worklist' && e.key === '/') {
    e.preventDefault();
    $('#case-search')?.focus();
    return;
  }
  if ((e.key === 'n' || e.key === 'N') && (ST.route === 'worklist' || ST.route === 'case')) {
    e.preventDefault();
    openNext();
  }
});

// The prioritisation popover opens toward the side with room: it sits at the far left when the toolbar wraps.
document.addEventListener(
  'toggle',
  (e) => {
    const pop = e.target.matches && e.target.matches('details.how') && e.target.open && e.target.querySelector('.how-pop');
    if (!pop) return;
    delete pop.dataset.uiSide;
    if (getComputedStyle(pop).position === 'absolute' && pop.getBoundingClientRect().left < 8) pop.dataset.uiSide = 'left';
  },
  true
);

document.addEventListener('focusin', (e) => {
  if (!e.target.matches || !e.target.matches('tr.row')) return;
  ST.focusId = e.target.dataset.caseId;
  // worklist rows: one tab stop (the row last used); J/K and the arrow keys move between rows
  if (e.target.closest('section.wl')) {
    document.querySelectorAll('section.wl tr.row[tabindex="0"]').forEach((r) => r !== e.target && (r.tabIndex = -1));
    e.target.tabIndex = 0;
  }
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

// Sticky offsets follow the real header height (it wraps onto two rows on narrow screens). On phones the brand row
// scrolls away and only the navigation stays pinned (theme.css: the masthead sticks at top: -var(--mast-hide)).
const phone = matchMedia('(max-width: 640px)');
function syncStick() {
  const top = $('.masthead');
  const root = document.documentElement.style;
  const hide = phone.matches ? Math.max(0, $('.glass-nav').offsetTop - 8) : 0;
  root.setProperty('--mast-hide', hide + 'px');
  const sticky = getComputedStyle(top).position === 'sticky';
  root.setProperty('--stick', (sticky ? top.offsetHeight - hide : 0) + 'px');
}
// Phones: the search field moves to the top of the screen when it is focused, so the matching cases sit between it and
// the on-screen keyboard. Return on a touch keyboard ("Search") closes the keyboard.
const narrow = matchMedia('(max-width: 760px)');
const coarse = matchMedia('(pointer: coarse)');
document.addEventListener('focusin', (e) => {
  if (e.target.id !== 'case-search' || !narrow.matches) return;
  const field = e.target;
  // after the keyboard has opened (the visual viewport shrinks first on iOS)
  setTimeout(() => {
    if (document.activeElement !== field) return;
    const y = field.getBoundingClientRect().top + window.scrollY - (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--stick')) || 0) - 12;
    if (Math.abs(window.scrollY - y) > 4) window.scrollTo({ top: Math.max(0, y), behavior: reducedMotion() ? 'auto' : 'smooth' });
  }, 250);
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.target.id === 'case-search' && coarse.matches) e.target.blur();
});
// The evaluation charts pick cell-label colours from the resolved theme (js/charts.js): redraw them when it changes.
window.addEventListener('pq:theme', () => {
  if (ST.route !== 'metrics') return;
  lastFp = null;
  draw(true);
});
if ('ResizeObserver' in window) new ResizeObserver(syncStick).observe($('.masthead'));
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
      `<div class="card" data-testid="state-offline-boot"><h2>Connecting to Sepia</h2><p>Cases have not loaded yet. ${connectionHelp}</p></div>`;
    setTimeout(boot, 3000);
    return;
  }
  syncStick();
  steadyTick();
  onRoute();
}
boot();
