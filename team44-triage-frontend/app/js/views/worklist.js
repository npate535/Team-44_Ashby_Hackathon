// Worklist (adapts the teammate's row/worklist/tasks): tier groups, live timers, reason badges, conformal chips,
// arrival-order vs AI-priority toggle, lane filter, aging badge, tasks card.
import { esc, num, toView } from '../adapter.js';

const NOTE = {
  ai:
    'Simulated frozen sections first, then likely invasive, low confidence, unflagged; within a tier by risk, then ' +
    'arrival. Cases waiting ≥75% of their tier’s target move up one tier (never above tier 2).',
  arrival: 'Read roughly first come, first served.'
};

export function setChips(v) {
  if (!v.risk && v.risk !== 0) return '<span class="sub">no score</span>';
  if (!v.setSize) return '<span class="chip empty" data-testid="set-chip">∅ empty</span>';
  return v.set
    .map(
      (cls) =>
        `<span class="chip${cls === 'Invasive' ? ' inv' : ''}" data-testid="set-chip">${esc(cls)}</span>`
    )
    .join('');
}

export function reasonBadges(v, { skipSurgical = true } = {}) {
  const badges = v.reasons
    .filter((r) => !(skipSurgical && r.code === 'surgical_frozen_section'))
    .map(
      (r) =>
        `<span class="reason${r.quality ? ' q' : ''}${r.code === 'aging_promoted' ? ' aging' : ''}" data-testid="reason-${esc(r.code)}">${esc(r.label)}</span>`
    );
  return badges.length ? `<div class="reasons">${badges.join('')}</div>` : '';
}

export function timerCell(v) {
  const t = v.timer;
  const width = t.usedFrac === null ? 0 : Math.round(t.usedFrac * 100);
  const cls = t.state === 'none' ? '' : t.state;
  return (
    `<span class="timer ${cls}" data-timer="${esc(v.id)}" data-state="${esc(t.state)}">${esc(t.text)}</span>` +
    `<div class="sub" data-timer-sub="${esc(v.id)}">${esc(t.countdown || '')}</div>` +
    (t.usedFrac === null
      ? ''
      : `<div class="pbar"><i class="${cls}" data-timer-bar="${esc(v.id)}" style="width:${width}%"></i></div>`)
  );
}

function row(v, ST) {
  const isNew = !ST.seen.has(v.id) && !ST.firstLoad;
  const agent = v.agent.drafted
    ? `<span data-testid="orders-drafted"><span class="dot"></span>Orders drafted (${v.agent.nOrders})</span>`
    : esc(v.agent.text);
  const lane =
    v.lane === 'surgical'
      ? `<span class="lanetag surg">${esc(v.laneLabel)}${v.variantLabel ? ' · ' + esc(v.variantLabel) : ''}${v.room ? ' · ' + esc(v.room) : ''}</span>`
      : `<span class="lanetag">${esc(v.laneLabel)} · ${esc(v.question)}</span>`;
  return (
    `<tr class="row${isNew ? ' new' : ''}${v.done ? ' done' : ''}" tabindex="0" data-testid="worklist-row" data-case-id="${esc(v.id)}" data-tier="${num(v.tier)}" data-lane="${esc(v.lane)}" data-done="${v.done}">` +
    `<td><b>${esc(v.id)}</b><div>${lane}</div>${reasonBadges(v)}${v.aged && ST.order === 'ai' ? '<div class="aged">Aged: promoted at 75% of target</div>' : ''}</td>` +
    `<td><span class="tier ${v.tierCls}" data-testid="tier-badge">${esc(v.tierLabel)}</span><div class="sub">${v.risk === null ? 'no score' : v.risk + '% P(Invasive), calibrated'}</div></td>` +
    `<td><div class="chips">${setChips(v)}</div>${v.alphaPct ? `<div class="sub">${v.alphaPct}% target coverage</div>` : ''}</td>` +
    `<td>${timerCell(v)}</td>` +
    `<td><span class="agent-cell">${agent}</span></td><td class="row-arrow"><svg aria-hidden="true"><use href="icons.svg#arrow"/></svg></td></tr>`
  );
}

export function tasks(views) {
  const t = [];
  for (const v of views) {
    if (v.done) continue;
    if (v.lane === 'surgical')
      t.push({
        id: v.id,
        text: `Read simulated frozen ${v.id}${v.room ? ' (' + v.room + ')' : ''}: ${v.timer.countdown}`
      });
    else if (v.agent.drafted) t.push({ id: v.id, text: `Review drafted orders, ${v.id}` });
    else if (v.reliability.flag && v.lane === 'biopsy')
      t.push({ id: v.id, text: `Reliability flag: consider recut, ${v.id}` });
  }
  return t.length
    ? `<ul style="margin:0;padding-left:18px">${t
        .slice(0, 6)
        .map((x) => `<li><a href="#/case/${encodeURIComponent(x.id)}">${esc(x.text)}</a></li>`)
        .join(
          ''
        )}</ul>${t.length > 6 ? `<div class="sub">${t.length - 6} more cases awaiting action in the worklist.</div>` : ''}`
    : '<div class="note">No open tasks. New tasks appear when the agent drafts orders or a frozen section arrives.</div>';
}

/** Surgical cases are held for injection (settings sim.hold_surgical_for_injection): say so, so none are forgotten. */
export function heldHint(sim) {
  if (!sim || !sim.hold_surgical_for_injection || !sim.pending) return '';
  const p = sim.pending.surgical || {};
  const total = (p.normal || 0) + (p.degraded || 0) + (p.composite || 0);
  if (!total) return '';
  return `<div class="held-hint" data-testid="held-surgical"><span class="hint-mark">i</span><span><b>${total} simulated frozen sections ready.</b> Use the specimen controls to introduce a frozen, degraded, or composite case.</span></div>`;
}

export function render(ST, ctx) {
  const allViews = ST.items.map((c) => toView(c, ctx.now, ST.cfg, ST.sim));
  // data-sort / data-lane-filter (not data-lane: rows carry data-lane for tests and styling)
  const seg = (key, val, label) =>
    `<button data-${key === 'order' ? 'sort' : 'lane-filter'}="${val}" data-testid="${key}-${val}" aria-pressed="${ST[key] === val}">${label}</button>`;
  const open = allViews.filter((v) => !v.done);
  const query = (ST.search || '').trim().toLowerCase();
  const views = allViews.filter(
    (v) =>
      (ST.status !== 'open' || !v.done) &&
      (ST.status !== 'done' || v.done) &&
      (!query ||
        [v.id, v.laneLabel, v.tierLabel, ...v.reasons.map((r) => r.label)]
          .join(' ')
          .toLowerCase()
          .includes(query))
  );
  const stat = (label, value, detail, cls) =>
    `<div class="queue-stat ${cls}"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`;
  let h =
    `<div class="queue-stats" data-testid="queue-stats">` +
    stat('Awaiting review', open.length, 'In the selected lane', 'stat-total') +
    stat(
      'Frozen sections',
      open.filter((v) => v.lane === 'surgical').length,
      'Simulated · time sensitive',
      'stat-frozen'
    ) +
    stat(
      'Likely invasive',
      open.filter((v) => v.tier === 2).length,
      'Prioritized for review',
      'stat-invasive'
    ) +
    stat(
      'Orders to review',
      open.filter((v) => v.agent.drafted).length,
      'Awaiting your decision',
      'stat-orders'
    ) +
    '</div>';
  h +=
    `<section class="queue-panel"><div class="queue-heading"><div><h2>Case worklist</h2><span class="sub">${open.length} open · ${allViews.filter((v) => v.done).length} completed</span></div>` +
    `<label class="search"><svg aria-hidden="true"><use href="icons.svg#search"/></svg><input id="case-search" type="search" placeholder="Search cases or flags…" aria-label="Search cases" value="${esc(ST.search || '')}" autocomplete="off"></label></div>`;
  const status = (value, text) =>
    `<button data-status-filter="${value}" aria-pressed="${(ST.status || 'all') === value}">${text}</button>`;
  h +=
    `<div class="queue-filters"><div class="status-filters" role="group" aria-label="Case status">${status('all', 'All cases')}${status('open', 'Awaiting review')}${status('done', 'Completed')}</div>` +
    `<div class="filter-right"><div class="seg" role="group" aria-label="Lane">${seg('lane', 'all', 'All lanes')}${seg('lane', 'surgical', 'Surgical')}${seg('lane', 'biopsy', 'Biopsy')}</div>` +
    `<div class="seg" role="group" aria-label="Sort order">${seg('order', 'arrival', 'Arrival order')}${seg('order', 'ai', 'AI priority')}</div></div></div>`;
  h += `<details class="priority-note"><summary>How cases are prioritized</summary><p class="note" data-testid="order-note">${NOTE[ST.order]} Invasive in the conformal set always receives tier 2.</p></details>`;
  if (!views.length) {
    const filtered = allViews.length > 0;
    const missing = ST.health && !ST.health.loaded;
    const title = filtered
      ? 'No matching cases'
      : missing
        ? 'Your workspace is ready for case artifacts'
        : ST.sim?.running
          ? 'Listening for the next arrival'
          : 'Ready when you are';
    const description = filtered
      ? 'Try another case ID, flag, or status filter.'
      : missing
        ? 'No valid pipeline cases are loaded. Import the Delta case bundle to connect real images, predictions, and evidence.'
        : ST.sim?.running
          ? 'Cases will appear here as the arrival replay progresses.'
          : 'Start the arrival replay to populate your worklist. Each case opens into its image, supporting evidence, and review actions.';
    return (
      h +
      `<div class="empty-state" data-testid="state-empty"><div class="empty-symbol"><svg aria-hidden="true"><use href="icons.svg#queue"/></svg></div><h2>${title}</h2><p>${description}</p>` +
      (filtered ? '<button class="btn" data-clear-search>Clear search</button>' : '') +
      '</div></section>' +
      heldHint(ST.sim)
    );
  }
  h +=
    '<div class="tblw"><table data-testid="worklist"><thead><tr><th>Case</th><th>Priority & risk</th><th>Confidence set</th><th>Elapsed / target</th><th>Next step</th><th><span class="sr-only">Open case</span></th></tr></thead><tbody>';
  const pageSize = 25;
  const page = Math.min(Math.max(0, ST.page || 0), Math.max(0, Math.ceil(views.length / pageSize) - 1));
  const pageViews = views.slice(page * pageSize, (page + 1) * pageSize);
  let group = null;
  for (const v of pageViews) {
    if (ST.order === 'ai') {
      const g = v.done ? 'done' : v.tier;
      if (g !== group) {
        group = g;
        const n = views.filter((x) => (x.done ? 'done' : x.tier) === g).length;
        const label =
          g === 'done' ? 'Completed (signed out or called)' : `Tier ${num(g)} · ${esc(v.tierLabel)}`;
        h += `<tr class="grp" data-testid="tier-group" data-group="${esc(g)}"><td colspan="6">${label} (${n})</td></tr>`;
      }
    }
    h += row(v, ST);
  }
  h += '</tbody></table></div>';
  if (views.length > pageSize)
    h += `<div class="pagination"><span>Showing ${page * pageSize + 1}–${Math.min((page + 1) * pageSize, views.length)} of ${views.length} cases</span><div><button class="btn" data-page="${page - 1}"${page === 0 ? ' disabled' : ''} aria-label="Previous page">← Previous</button><button class="btn" data-page="${page + 1}"${(page + 1) * pageSize >= views.length ? ' disabled' : ''} aria-label="Next page">Next →</button></div></div>`;
  h += '</section>' + heldHint(ST.sim);
  h += `<div class="card task-card" data-testid="tasks"><div class="task-heading"><span class="task-icon">↗</span><h2>Your next steps</h2></div>${tasks(views)}</div>`;
  if (views.some((v) => v.tier === 4 && !v.done))
    h += '<p class="note">Unflagged means read in turn. It does not mean cleared.</p>';
  return h;
}
