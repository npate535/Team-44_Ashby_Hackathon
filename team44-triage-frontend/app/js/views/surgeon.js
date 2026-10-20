// Operating-room board (surgeon screen): one panel per simulated OR. A frozen section's timer runs from arrival
// until the pathologist's margin call, then stops and shows the turnaround. New calls are highlighted.
// Live: app.js polls every 2 s and ticks timers every second. Research prototype — simulated frozen sections only.
// Status colours follow the legend: an open frozen section is neutral (ring), a sent call takes its call's colour.
// The model's margin read never reaches this board before the pathologist's call: an open frozen section shows only
// the timer, and a sent call shows the pathologist's call first, with the model's suggestion as secondary text.
import { esc, toView } from '../adapter.js';
import { CALL_TONE, joinNames, routedKey, suggestedCall } from '../flow.js';

// Room header text for the latest call (the legend's wording).
const TONE_STATE = { clear: 'Margins clear', tumor: 'Tumor at margin', defer: 'Defer to permanent' };

// Cards drawn by the previous render of this board. A poll re-render rebuilds the DOM, so only cards that were not
// on screen before get the entry animation (nothing replays, and nothing animates when the board is first opened).
let shownKeys = null;

const CALL_HEAD = {
  clear: ['Margins clear', 'No tumor at the inked margins of this field'],
  defer: ['Cannot determine', 'Defer to permanent sections']
};

function callParts(r) {
  if (r.call === 'tumor_at_margin') {
    const m = r.margins || [];
    return [`Tumor at ${joinNames(m)} margin${m.length > 1 ? 's' : ''}`, 'More tissue needed'];
  }
  return CALL_HEAD[r.call] || [String(r.call), ''];
}

// Durations in words ("8 min 41 s"), so a turnaround never reads as a time of day.
function minSec(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600),
    m = Math.floor((s % 3600) / 60),
    r = s % 60;
  return h ? `${h} h ${m} min` : m ? `${m} min ${r} s` : `${r} s`;
}

const clock = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—');

function variantText(v) {
  if (v.variant === 'degraded') return 'degraded field';
  if (v.variant === 'composite') return 'composite field';
  return 'frozen section';
}

// Open frozen sections in a room, oldest first. Sections injected while the replay clock is paused (or held after the
// replay) share one simulator second, so the real arrival time decides, then the simulator time. Without a wall-clock
// arrival the timer that has run longest came first.
const wallMs = (v) => {
  const t = v.arrivedAt ? Date.parse(v.arrivedAt) : NaN;
  return Number.isFinite(t) ? t : null;
};
export function byArrival(a, b) {
  const ta = wallMs(a);
  const tb = wallMs(b);
  const d =
    ta !== null && tb !== null
      ? ta - tb || (a.arrivedSim || 0) - (b.arrivedSim || 0)
      : ((b.timer && b.timer.elapsedS) || 0) - ((a.timer && a.timer.elapsedS) || 0) || (a.arrivedSim || 0) - (b.arrivedSim || 0);
  return d || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
}

function activeCard(v, enter) {
  const t = v.timer;
  return (
    `<div class="or-case live${enter ? ' enter' : ''}" data-testid="or-card" data-case-id="${esc(v.id)}">` +
    `<div class="or-line"><span class="or-name"><b class="or-id">${esc(v.id)}</b> <span class="or-kind">${esc(variantText(v))}</span></span></div>` +
    // app.js tick() rewrites the timer's text every second, so its "elapsed" label sits beside it, not inside it
    `<div class="or-clock"><div class="or-timer timer ${esc(t.state)}" data-timer="${esc(v.id)}" data-state="${esc(t.state)}" data-testid="or-timer">${esc(t.text)}</div><span class="or-clock-l">elapsed</span></div>` +
    `<div class="or-bar"><i class="${t.state === 'none' ? '' : esc(t.state)}" data-timer-bar="${esc(v.id)}" style="width:${Math.round((t.usedFrac || 0) * 100)}%"></i></div>` +
    // the countdown ("19:52 left", "Overdue 00:23") is kept live by app.js tick() through data-timer-sub
    `<div class="or-sub">Received ${esc(clock(v.arrivedAt))} · Target ${Math.round((t.targetS || 1200) / 60)} min · <span data-timer-sub="${esc(v.id)}">${esc(t.countdown || '')}</span></div>` +
    // no model read here: an unconfirmed margin read must not reach the surgeon before the pathologist's call
    `<div class="or-await" data-testid="or-awaiting">Awaiting the pathologist’s margin call</div>` +
    `<a class="case-link" href="#/case/${encodeURIComponent(v.id)}">Open case</a>` +
    '</div>'
  );
}

function calledCard(v, r, fresh, enter) {
  const [head, detail] = callParts(r);
  const tone = CALL_TONE[r.call] || 'defer';
  const turnaround = Number.isFinite(r.timer_elapsed_s) ? minSec(r.timer_elapsed_s) : null;
  const aiMargins = (r.report && r.report.margin && r.report.margin.involved) || [];
  const suggested = r.suggested_call ? suggestedCall({ suggested_call: r.suggested_call, involved: aiMargins }) : null;
  // "tumor at margin" agrees only when the same margins were named
  const sameMargins = (a, b) => a.length === b.length && a.every((m) => b.includes(m));
  const agreement = !r.suggested_call || r.suggested_call !== r.call
    ? ' · the pathologist’s call differs'
    : r.call === 'tumor_at_margin' && !sameMargins(aiMargins, r.margins || [])
      ? ' · same call, different margins'
      : ' · agrees with the pathologist’s call';
  return (
    `<div class="or-case called${fresh ? ' fresh' : ''}${enter ? ' enter' : ''}" data-testid="or-card" data-case-id="${esc(r.case_id)}">` +
    `<div class="or-line"><span class="or-name"><b class="or-id">${esc(r.case_id)}</b> <span class="or-kind">${esc(v ? variantText(v) : 'frozen section')}</span></span>${fresh ? `<span class="or-new tone-${tone}" data-testid="or-new-call">New call</span>` : ''}</div>` +
    `<div class="or-call tone-${tone}" data-testid="surgeon-result" data-call="${esc(r.call)}"><b>${esc(head)}</b><span>${esc(detail)}</span></div>` +
    `<div class="or-meta"><span>Called ${esc(clock(r.at))} by the reading pathologist</span>` +
    (turnaround ? `<span>Turnaround <b class="timer stopped" data-testid="or-turnaround">${esc(turnaround)}</b> · timer stopped</span>` : '') +
    '</div>' +
    // secondary, and only after the call: the pathologist's call above is the result
    (suggested ? `<div class="or-ai" data-testid="or-model-suggestion">Model suggestion: ${esc(suggested)}${agreement}</div>` : '') +
    `<a class="case-link" href="#/case/${encodeURIComponent(r.case_id)}">Open case</a>` +
    '</div>'
  );
}

export function render(ST, ctx) {
  // first render of a visit (the board is not on screen yet): draw without entry animations
  const onBoard = typeof document !== 'undefined' && !!document.querySelector('[data-testid=surgeon-screen]');
  const prevKeys = onBoard ? shownKeys : null;
  const keys = new Set();
  const enters = (key) => {
    keys.add(key);
    return !!prevKeys && !prevKeys.has(key);
  };
  const views = ST.items.filter((c) => c.lane === 'surgical').map((c) => {
    const v = toView(c, ctx.now, ST.cfg, ST.sim);
    v.arrivedAt = c.runtime && c.runtime.arrived_at;
    return v;
  });
  const byId = new Map(views.map((v) => [v.id, v]));
  const latest = new Map(); // newest call per case (routed list is newest first)
  for (const r of ST.routed.surgeon) if (!latest.has(r.case_id)) latest.set(r.case_id, r);
  const nRooms = (ST.cfg.sim && ST.cfg.sim.or_rooms) || 4;
  const roomNames = Array.from({ length: nRooms }, (_, i) => `OR-${i + 1}`);
  for (const v of views) if (v.room && !roomNames.includes(v.room)) roomNames.push(v.room);
  for (const r of latest.values()) if (r.or_room && !roomNames.includes(r.or_room)) roomNames.push(r.or_room);

  const rooms = roomNames.map((room) => {
    const active = views.filter((v) => v.room === room && !v.done && !latest.has(v.id)).sort(byArrival);
    const calls = [...latest.values()].filter((r) => r.or_room === room); // newest first
    const fresh = calls.some((r) => ST.freshCalls.has(routedKey(r)));
    const state = active.length ? 'busy' : calls.length ? 'called' : 'idle';
    const tone = calls.length ? CALL_TONE[calls[0].call] || 'defer' : null;
    const stateText = active.length
      ? active.length > 1
        ? `${active.length} awaiting a call`
        : 'Awaiting call'
      : calls.length
        ? TONE_STATE[tone]
        : 'Idle';
    let body = active.map((v) => activeCard(v, enters('a:' + v.id))).join('');
    if (calls.length) {
      const [last, ...older] = calls;
      body += calledCard(byId.get(last.case_id), last, ST.freshCalls.has(routedKey(last)), enters('c:' + routedKey(last)));
      if (older.length)
        body += `<details class="or-history"><summary>${older.length} earlier call${older.length === 1 ? '' : 's'}</summary><ul>${older
          .slice(0, 8)
          .map((r) => `<li data-testid="or-history-item"><a href="#/case/${encodeURIComponent(r.case_id)}">${esc(r.case_id)}</a> <span class="tone-${CALL_TONE[r.call] || 'defer'}">${esc(callParts(r)[0])}</span> <span class="sub">${esc(clock(r.at))}</span></li>`)
          .join('')}</ul></details>`;
    }
    if (!body) body = '<div class="or-idle">No frozen section sent from this room.</div>';
    return (
      `<section class="room ${state}${fresh ? ' fresh' : ''}"${tone ? ` data-tone="${tone}"` : ''} data-testid="or-room" data-room="${esc(room)}" aria-label="${esc(room)}: ${esc(stateText)}">` +
      `<header class="room-head"><h2>${esc(room)}</h2><span class="room-state">${esc(stateText)}</span></header>${body}</section>`
    );
  });
  shownKeys = keys;

  const awaiting = views.filter((v) => !v.done).length;
  const intro =
    `<div class="orb-head"><p>Each simulated operating room shows its frozen section. The timer runs until the pathologist sends the margin call, then stops at the turnaround time. Calls appear here the moment they are sent; the model’s suggestion is shown only after the pathologist’s call.</p>` +
    `<div class="orb-key"><span class="k-busy">Awaiting call</span><span class="k-clear">Margins clear</span><span class="k-tumor">Tumor at margin</span><span class="k-defer">Defer to permanent</span></div></div>`;
  const empty = !views.length && !ST.routed.surgeon.length
    ? '<p class="orb-empty" data-testid="surgeon-empty">No frozen section in progress. Send a simulated one with the <b>Frozen section</b> buttons in the header.</p>'
    : '';
  return `<div class="orb" data-testid="surgeon-screen" data-awaiting="${awaiting}">${intro}${empty}<div class="orb-grid">${rooms.join('')}</div><p class="sub orb-foot">Simulation: image borders stand in for inked margins; room assignment is simulated. Not for clinical use.</p></div>`;
}
