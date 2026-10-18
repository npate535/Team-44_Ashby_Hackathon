// Operating-room board (surgeon screen): one panel per simulated OR. A frozen section's timer runs from arrival
// until the pathologist's margin call, then stops and shows the turnaround. New calls are highlighted.
// Live: app.js polls every 2 s and ticks timers every second. Research prototype — simulated frozen sections only.
import { esc, mmss, toView } from '../adapter.js';
import { CALL_TONE, joinNames, routedKey, suggestedCall } from '../flow.js';

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

const clock = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—');

function variantText(v) {
  if (v.variant === 'degraded') return 'degraded field';
  if (v.variant === 'composite') return 'composite field';
  return 'frozen section';
}

function activeCard(v) {
  const t = v.timer;
  return (
    `<div class="or-case live" data-testid="or-card" data-case-id="${esc(v.id)}">` +
    `<div class="or-line"><b class="or-id">${esc(v.id)}</b><span class="or-kind">${esc(variantText(v))}</span></div>` +
    `<div class="or-timer timer ${esc(t.state)}" data-timer="${esc(v.id)}" data-state="${esc(t.state)}" data-testid="or-timer">${esc(t.text)}</div>` +
    `<div class="or-bar"><i class="${t.state === 'none' ? '' : esc(t.state)}" data-timer-bar="${esc(v.id)}" style="width:${Math.round((t.usedFrac || 0) * 100)}%"></i></div>` +
    `<div class="or-sub">Received ${esc(clock(v.arrivedAt))} · target ${esc(mmss(t.targetS || 1200))}</div>` +
    `<div class="or-await" data-testid="or-awaiting">Awaiting the pathologist’s margin call</div>` +
    `<div class="or-ai">AI read: ${esc(suggestedCall(v.margin ? { suggested_call: v.margin.call, involved: v.margin.involved } : null))}${v.reliability.flag ? ' · <b>reliability flag</b>' : ''}</div>` +
    `<a class="case-link" href="#/case/${encodeURIComponent(v.id)}">Open case</a>` +
    '</div>'
  );
}

function calledCard(v, r, fresh) {
  const [head, detail] = callParts(r);
  const tone = CALL_TONE[r.call] || 'defer';
  const turnaround = Number.isFinite(r.timer_elapsed_s) ? mmss(r.timer_elapsed_s) : null;
  const suggested = r.suggested_call ? suggestedCall({ suggested_call: r.suggested_call, involved: (r.report && r.report.margin && r.report.margin.involved) || [] }) : null;
  const agrees = r.suggested_call && r.suggested_call === r.call;
  return (
    `<div class="or-case called${fresh ? ' fresh' : ''}" data-testid="or-card" data-case-id="${esc(r.case_id)}">` +
    `<div class="or-line"><b class="or-id">${esc(r.case_id)}</b><span class="or-kind">${esc(v ? variantText(v) : 'frozen section')}</span>${fresh ? '<span class="or-new" data-testid="or-new-call">New call</span>' : ''}</div>` +
    `<div class="or-call tone-${tone}" data-testid="surgeon-result" data-call="${esc(r.call)}"><b>${esc(head)}</b><span>${esc(detail)}</span></div>` +
    `<div class="or-meta"><span>Called ${esc(clock(r.at))} by the reading pathologist</span>` +
    (turnaround ? `<span>Turnaround <b class="timer stopped" data-testid="or-turnaround">${esc(turnaround)}</b> · timer stopped</span>` : '') +
    '</div>' +
    (suggested ? `<div class="or-ai">AI read: ${esc(suggested)}${agrees ? ' · matches the call' : ' · pathologist decided differently'}</div>` : '') +
    `<a class="case-link" href="#/case/${encodeURIComponent(r.case_id)}">Open case</a>` +
    '</div>'
  );
}

export function render(ST, ctx) {
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
    const active = views.filter((v) => v.room === room && !v.done && !latest.has(v.id)).sort((a, b) => a.arrivedSim - b.arrivedSim);
    const calls = [...latest.values()].filter((r) => r.or_room === room); // newest first
    const fresh = calls.some((r) => ST.freshCalls.has(routedKey(r)));
    const state = active.length ? 'busy' : calls.length ? 'called' : 'idle';
    const stateText = active.length ? (active.length > 1 ? `${active.length} awaiting a call` : 'Awaiting call') : calls.length ? 'Call sent' : 'No specimen';
    let body = active.map(activeCard).join('');
    if (calls.length) {
      const [last, ...older] = calls;
      body += calledCard(byId.get(last.case_id), last, ST.freshCalls.has(routedKey(last)));
      if (older.length)
        body += `<details class="or-history"><summary>${older.length} earlier call${older.length === 1 ? '' : 's'}</summary><ul>${older
          .slice(0, 8)
          .map((r) => `<li data-testid="or-history-item"><a href="#/case/${encodeURIComponent(r.case_id)}">${esc(r.case_id)}</a> <span class="tone-${CALL_TONE[r.call] || 'defer'}">${esc(callParts(r)[0])}</span> <span class="sub">${esc(clock(r.at))}</span></li>`)
          .join('')}</ul></details>`;
    }
    if (!body) body = '<div class="or-idle">No frozen section from this room.</div>';
    return (
      `<section class="room ${state}${fresh ? ' fresh' : ''}" data-testid="or-room" data-room="${esc(room)}" aria-label="${esc(room)}: ${esc(stateText)}">` +
      `<header class="room-head"><h2>${esc(room)}</h2><span class="room-state">${esc(stateText)}</span></header>${body}</section>`
    );
  });

  const awaiting = views.filter((v) => !v.done).length;
  const intro =
    `<div class="orb-head"><p>Each simulated operating room shows its frozen section. The timer runs until the pathologist sends the margin call, then stops at the turnaround time. Calls appear here the moment they are sent.</p>` +
    `<div class="orb-key"><span class="k-busy">Awaiting call</span><span class="k-clear">Margins clear</span><span class="k-tumor">Tumor at margin</span><span class="k-defer">Defer to permanent</span></div></div>`;
  const empty = !views.length && !ST.routed.surgeon.length
    ? '<p class="orb-empty" data-testid="surgeon-empty">No frozen section in progress. Send a simulated one with the <b>Frozen section</b> buttons in the header.</p>'
    : '';
  return `<div class="orb" data-testid="surgeon-screen" data-awaiting="${awaiting}">${intro}${empty}<div class="orb-grid">${rooms.join('')}</div><p class="sub orb-foot">Simulation: image borders stand in for inked margins; room assignment is simulated. Not for clinical use.</p></div>`;
}
