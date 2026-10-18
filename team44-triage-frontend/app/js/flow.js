// Workflow helpers shared by the shell (app.js), worklist, OR board and oncology inbox. Pure functions + a tiny,
// fail-safe browser-storage wrapper (per-viewer conveniences only: read state, last call seen).
// Research prototype — not for clinical use. Nothing here reads or renders ground truth.
import { esc, fmt, mmss } from './adapter.js';

export const SHORT_CLASS = { Normal: 'Normal', Benign: 'Benign', InSitu: 'In situ', Invasive: 'Invasive' };
export const SUGGESTED = {
  Normal: 'Normal tissue',
  Benign: 'Benign lesion',
  InSitu: 'In situ carcinoma',
  Invasive: 'Invasive carcinoma'
};
export const MARGIN_SHORT = { superior: 'sup', inferior: 'inf', medial: 'med', lateral: 'lat' };
export const MARGIN_ORDER = ['superior', 'inferior', 'medial', 'lateral'];

export const shortClass = (cls) => SHORT_CLASS[cls] || cls;
export const suggestedLabel = (cls) => SUGGESTED[cls] || cls || '—';

/** "superior", "superior and medial", "superior, medial and lateral" */
export function joinNames(names) {
  const list = (names || []).filter(Boolean);
  if (list.length <= 1) return list.join('');
  return list.slice(0, -1).join(', ') + ' and ' + list[list.length - 1];
}

/** Text of a margin call ({call, margins}) as the surgeon reads it. */
export function callText(call) {
  if (!call || !call.call) return '';
  if (call.call === 'tumor_at_margin') {
    const m = call.margins || [];
    return `Tumor at ${joinNames(m)} margin${m.length > 1 ? 's' : ''} — more tissue needed`;
  }
  if (call.call === 'clear') return 'Margins clear';
  if (call.call === 'defer') return 'Cannot determine — defer to permanent sections';
  return String(call.call);
}
export const CALL_TONE = { clear: 'clear', tumor_at_margin: 'tumor', defer: 'defer' };

/** Short suggested-call text for a worklist row. */
export function suggestedCall(margin) {
  if (!margin || !margin.suggested_call) return 'Margin read unavailable';
  if (margin.suggested_call === 'tumor_at_margin') {
    const m = (margin.involved || []).map((x) => MARGIN_SHORT[x] || x);
    return 'Tumor at margin' + (m.length ? ' (' + m.join(', ') + ')' : '');
  }
  return margin.suggested_call === 'clear' ? 'Margins clear' : 'Defer to permanent';
}

/** Compact duration: 45 s, 12 min, 3.4 h, 1.2 d. */
export function dur(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  if (s < 60) return Math.round(s) + ' s';
  return fmt(s / 60);
}

/**
 * Waiting time vs target with SLA state for a timerNow() result.
 * surgical: mm:ss elapsed of a 20-min target (ok / amber / red / stopped).
 * biopsy: elapsed vs its tier's turnaround target on the simulator clock (ok <75%, amber >=75%, red >=100%).
 */
export function waitInfo(t) {
  if (!t || t.kind === 'none') return { text: '—', target: '', state: 'none', frac: null };
  if (t.kind === 'surgical') {
    return {
      text: mmss(t.elapsedS),
      target: t.state === 'stopped' ? 'called' : '/ ' + mmss(t.targetS || 0),
      state: t.state,
      frac: t.usedFrac
    };
  }
  const frac = t.targetS ? t.elapsedS / t.targetS : null;
  const state = t.state === 'stopped' ? 'stopped' : frac === null ? 'none' : frac >= 1 ? 'red' : frac >= 0.75 ? 'amber' : 'ok';
  return {
    text: dur(t.elapsedS),
    target: t.state === 'stopped' ? 'read' : t.targetS ? '/ ' + dur(t.targetS) : '',
    state,
    frac: frac === null ? null : Math.min(1, frac)
  };
}

/** Open (not done) items, in the order given. */
export const openItems = (items) => (items || []).filter((c) => !(c.runtime && c.runtime.done));

/**
 * The next case to read, in the given priority order: the open case after `currentId` while it is still open
 * (browsing down the queue), otherwise the top open case (after a sign-out or margin call the current case is done,
 * so the reader continues at the head of the queue). Never returns the current case.
 */
export function nextCase(items, currentId = null) {
  const open = openItems(items);
  const i = currentId === null ? -1 : open.findIndex((c) => c.case_id === currentId);
  if (i >= 0 && i + 1 < open.length) return open[i + 1];
  return open.find((c) => c.case_id !== currentId) || null;
}

/** Position of a case among the open items (1-based), or null. */
export function queuePosition(items, id) {
  const i = openItems(items).findIndex((c) => c.case_id === id);
  return i < 0 ? null : i + 1;
}

/** Queue composition: open counts by tier and lane, plus orders awaiting review and done count. */
export function composition(views) {
  const open = views.filter((v) => !v.done);
  const byTier = { 1: 0, 2: 0, 3: 0, 4: 0 };
  for (const v of open) byTier[v.tier] = (byTier[v.tier] || 0) + 1;
  return {
    open: open.length,
    done: views.length - open.length,
    byTier,
    surgical: open.filter((v) => v.lane === 'surgical').length,
    biopsy: open.filter((v) => v.lane === 'biopsy').length,
    orders: open.filter((v) => v.agent.drafted).length,
    overdue: open.filter((v) => waitInfo(v.timer).state === 'red').length
  };
}

/** The pathologist's next step for a row (plain words, no system terms). */
export function nextAction(v) {
  if (v.lane === 'surgical') {
    if (v.decision.marginCall) return { text: 'Call sent to ' + (v.room || 'OR'), tone: 'done' };
    if (v.agent.drafted) return { text: 'Make margin call', tone: 'todo', sub: 'recut drafted' };
    return { text: 'Make margin call', tone: 'todo' };
  }
  if (v.decision.signedOut) return { text: 'Signed out', tone: 'done' };
  const s = v.agent.status;
  if (s === 'drafted') return { text: `Review ${v.agent.nOrders} order${v.agent.nOrders === 1 ? '' : 's'}`, tone: 'orders' };
  if (s === 'deferred_closer_look') return { text: 'Closer look, then decide', tone: 'orders' };
  return { text: 'Read and sign out', tone: 'todo' };
}

// ---- per-viewer storage (never required: every read/write is guarded) ---------------------------------------------
export function load(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (_) {
    return fallback;
  }
}
export function save(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (_) {
    /* private mode or blocked storage: state stays in memory */
  }
}
/** Stable identity of a routed result across replay resets (row ids restart after a reset). */
export const routedKey = (r) => `${r.id}|${r.case_id}|${r.at}`;

/** Inline percentage bar (decorative; the number beside it carries the value). */
export function miniBar(frac, cls = '') {
  const w = Math.max(0, Math.min(100, Math.round((Number(frac) || 0) * 100)));
  return `<span class="mbar ${esc(cls)}" aria-hidden="true"><i style="width:${w}%"></i></span>`;
}

/**
 * The conformal method that drives the worklist tiers, read from eval/conformal.json (`drives_tiers`, fallback
 * `lac_marginal`), never hard-coded. Returns a display label ("APS conformal set · 90% target") and the measured
 * held-out coverage of that method when it is present.
 */
export function conformalInfo(metrics, cfg) {
  const target = cfg && Number.isFinite(cfg.alpha) ? Math.round((1 - cfg.alpha) * 100) : 90;
  const rep = metrics && metrics.eval && metrics.eval.conformal;
  const methods = (rep && rep.methods) || {};
  const key = rep && rep.drives_tiers && methods[rep.drives_tiers] ? rep.drives_tiers : methods.lac_marginal ? 'lac_marginal' : null;
  const name = key ? key.split('_')[0].toUpperCase() : null;
  const m = key ? methods[key] : null;
  const est = m && (m.coverage ? m.coverage.estimate : m.estimate);
  return {
    key,
    target,
    label: `${name ? name + ' conformal' : 'Confidence'} set · ${target}% target`,
    coverage: typeof est === 'number' && Number.isFinite(est) ? est : null,
    n: rep && Number.isInteger(rep.n) ? rep.n : null
  };
}
