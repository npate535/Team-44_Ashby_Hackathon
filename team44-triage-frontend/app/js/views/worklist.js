// Worklist (adapts the teammate's row/worklist/tasks): tier groups, live timers, reason badges, conformal chips,
// arrival-order vs AI-priority toggle, lane filter, aging badge, tasks card.
import {esc, num, toView} from '../adapter.js';

const NOTE = {
  ai: 'Simulated frozen sections first, then likely invasive, low confidence, unflagged; within a tier by risk, then ' +
      'arrival. Cases waiting ≥75% of their tier’s target move up one tier (never above tier 2).',
  arrival: 'Read roughly first come, first served.',
};

export function setChips(v) {
  if (!v.risk && v.risk !== 0) return '<span class="sub">no score</span>';
  if (!v.setSize) return '<span class="chip empty" data-testid="set-chip">∅ empty</span>';
  return v.set.map(cls => `<span class="chip${cls === 'Invasive' ? ' inv' : ''}" data-testid="set-chip">${esc(cls)}</span>`).join('');
}

export function reasonBadges(v, {skipSurgical = true} = {}) {
  const badges = v.reasons.filter(r => !(skipSurgical && r.code === 'surgical_frozen_section')).map(r =>
    `<span class="reason${r.quality ? ' q' : ''}${r.code === 'aging_promoted' ? ' aging' : ''}" data-testid="reason-${esc(r.code)}">${esc(r.label)}</span>`);
  return badges.length ? `<div class="reasons">${badges.join('')}</div>` : '';
}

export function timerCell(v) {
  const t = v.timer;
  const width = t.usedFrac === null ? 0 : Math.round(t.usedFrac * 100);
  const cls = t.state === 'none' ? '' : t.state;
  return `<span class="timer ${cls}" data-timer="${esc(v.id)}" data-state="${esc(t.state)}">${esc(t.text)}</span>` +
    `<div class="sub" data-timer-sub="${esc(v.id)}">${esc(t.countdown || '')}</div>` +
    (t.usedFrac === null ? '' : `<div class="pbar"><i class="${cls}" data-timer-bar="${esc(v.id)}" style="width:${width}%"></i></div>`);
}

function row(v, ST) {
  const isNew = !ST.seen.has(v.id) && !ST.firstLoad;
  const agent = v.agent.drafted
    ? `<span data-testid="orders-drafted"><span class="dot"></span>Orders drafted (${v.agent.nOrders})</span>`
    : esc(v.agent.text);
  const lane = v.lane === 'surgical'
    ? `<span class="lanetag surg">${esc(v.laneLabel)}${v.variantLabel ? ' · ' + esc(v.variantLabel) : ''}${v.room ? ' · ' + esc(v.room) : ''}</span>`
    : `<span class="lanetag">${esc(v.laneLabel)} · ${esc(v.question)}</span>`;
  return `<tr class="row${isNew ? ' new' : ''}${v.done ? ' done' : ''}" tabindex="0" data-testid="worklist-row" data-case-id="${esc(v.id)}" data-tier="${num(v.tier)}" data-lane="${esc(v.lane)}" data-done="${v.done}">` +
    `<td><b>${esc(v.id)}</b><div>${lane}</div>${reasonBadges(v)}${v.aged && ST.order === 'ai' ? '<div class="aged">Aged: promoted at 75% of target</div>' : ''}</td>` +
    `<td><span class="tier ${v.tierCls}" data-testid="tier-badge">${esc(v.tierLabel)}</span><div class="sub">${v.risk === null ? 'no score' : v.risk + '% P(Invasive), calibrated'}</div></td>` +
    `<td><div class="chips">${setChips(v)}</div>${v.alphaPct ? `<div class="sub">${v.alphaPct}% conformal set</div>` : ''}</td>` +
    `<td>${timerCell(v)}</td>` +
    `<td>${agent}</td></tr>`;
}

export function tasks(views) {
  const t = [];
  for (const v of views) {
    if (v.done) continue;
    if (v.lane === 'surgical') t.push(`Read simulated frozen ${v.id}${v.room ? ' (' + v.room + ')' : ''}: ${v.timer.countdown}`);
    if (v.agent.drafted) t.push(`Review drafted orders, ${v.id}`);
    if (v.reliability.flag && v.lane === 'biopsy') t.push(`Reliability flag: consider recut, ${v.id}`);
  }
  return t.length ? `<ul style="margin:0;padding-left:18px">${t.map(x => `<li>${esc(x)}</li>`).join('')}</ul>`
    : '<div class="note">No open tasks. New tasks appear when the agent drafts orders or a frozen section arrives.</div>';
}

/** Surgical cases are held for injection (settings sim.hold_surgical_for_injection): say so, so none are forgotten. */
export function heldHint(sim) {
  if (!sim || !sim.hold_surgical_for_injection || !sim.pending) return '';
  const p = sim.pending.surgical || {};
  const total = (p.normal || 0) + (p.degraded || 0) + (p.composite || 0);
  if (!total) return '';
  return `<div class="note" data-testid="held-surgical" style="margin:0 0 10px">${total} simulated frozen section${total > 1 ? 's' : ''} waiting ` +
    `(${p.normal || 0} normal, ${p.degraded || 0} degraded, ${p.composite || 0} composite). They arrive only when injected: ` +
    'use <b>+ Frozen</b>, <b>+ Degraded</b> or <b>+ Composite</b> above.</div>';
}

export function render(ST, ctx) {
  const views = ST.items.map(c => toView(c, ctx.now, ST.cfg, ST.sim));
  // data-sort / data-lane-filter (not data-lane: rows carry data-lane for tests and styling)
  const seg = (key, val, label) => `<button data-${key === 'order' ? 'sort' : 'lane-filter'}="${val}" data-testid="${key}-${val}" aria-pressed="${ST[key] === val}">${label}</button>`;
  let h = `<div class="bar"><div class="seg" role="group" aria-label="Sort order">${seg('order', 'arrival', 'Arrival order')}${seg('order', 'ai', 'AI priority')}</div>` +
    `<div class="seg" role="group" aria-label="Lane">${seg('lane', 'all', 'All')}${seg('lane', 'surgical', 'Surgical')}${seg('lane', 'biopsy', 'Biopsy')}</div>` +
    `<div class="note" data-testid="order-note">${NOTE[ST.order]}</div></div>` + heldHint(ST.sim);
  if (!views.length) {
    const pending = ST.sim && ST.sim.pending ? ST.sim.pending.biopsy : null;
    return h + `<div class="card empty-state" data-testid="state-empty"><h2>No cases yet</h2><p>${ST.sim && ST.sim.running
      ? 'Waiting for the next simulated arrival' + (pending !== null ? ` (${pending} biopsies pending).` : '.')
      : 'Start the simulator to replay arrivals, or inject a simulated frozen section.'}</p></div>`;
  }
  h += '<div class="tblw"><table data-testid="worklist"><thead><tr><th>Case</th><th>AI tier</th><th>Conformal set</th><th>Timer</th><th>Agent</th></tr></thead><tbody>';
  let group = null;
  for (const v of views) {
    if (ST.order === 'ai') {
      const g = v.done ? 'done' : v.tier;
      if (g !== group) {
        group = g;
        const n = views.filter(x => (x.done ? 'done' : x.tier) === g).length;
        const label = g === 'done' ? 'Completed (signed out or called)' : `Tier ${num(g)} · ${esc(v.tierLabel)}`;
        h += `<tr class="grp" data-testid="tier-group" data-group="${esc(g)}"><td colspan="5">${label} (${n})</td></tr>`;
      }
    }
    h += row(v, ST);
  }
  h += '</tbody></table></div>';
  h += `<div class="card" style="margin-top:14px" data-testid="tasks"><h2>Tasks</h2>${tasks(views)}</div>`;
  if (views.some(v => v.tier === 4 && !v.done)) h += '<p class="note">Unflagged means read in turn. It does not mean cleared.</p>';
  return h;
}
