// Worklist: a dense, keyboard-first clinical queue. One line per case, sticky column and tier-group headers,
// queue composition, filters, live waiting times against each tier's target, and an "Open next case" action.
// Tier rules come from the server (core.tiers, plan §1); this view only presents them.
// Research prototype — not for clinical use. Ground truth is never fetched or rendered.
import { esc, num, toView } from '../adapter.js';
import {
  composition,
  conformalInfo,
  miniBar,
  nextAction,
  nextCase,
  shortClass,
  suggestedCall,
  suggestedLabel,
  waitInfo
} from '../flow.js';

const TIER_NAME = { 1: 'Frozen section', 2: 'Likely invasive', 3: 'Low confidence', 4: 'Unflagged' };
const TIER_NOTE = {
  1: 'Simulated frozen sections from the OR · 20-minute target',
  2: 'Invasive suggested, or Invasive in the confidence set',
  3: 'Ambiguous confidence set, low reliability, or too little tissue',
  4: 'Read in arrival order · unflagged does not mean cleared'
};

// Flag tags shown in the Flags column. aggressive_top is carried by the Suggested column itself.
const FLAG_TAG = {
  aggressive_in_conformal_set: { text: 'Invasive in set', tone: 'eos', title: 'Safety rule: Invasive is in the 90% confidence set although it is not the top suggestion' },
  set_size_gt_1: { text: 'Ambiguous', tone: 'amb', title: 'More than one class in the confidence set' },
  empty_set: { text: 'Empty set', tone: 'amb', title: 'No class reached the confidence threshold' },
  low_reliability: { text: 'Low reliability', tone: 'amb', title: 'Reliability flag: the read may not be trustworthy' },
  insufficient_tissue: { text: 'Little tissue', tone: 'amb', title: 'Too few regions passed tissue quality control' },
  aging_promoted: { text: 'Aged', tone: 'eos', title: 'Moved up one tier after waiting ≥75% of its target' }
};
export const FLAG_FILTERS = [
  ['safety', 'Invasive in set (not top)'],
  ['set_size_gt_1', 'Ambiguous set'],
  ['empty_set', 'Empty set'],
  ['low_reliability', 'Low reliability'],
  ['insufficient_tissue', 'Little tissue'],
  ['aging_promoted', 'Aged'],
  ['any', 'Any flag']
];

function codes(v) {
  return v.reasons.map((r) => r.code);
}

function flagTags(v) {
  const c = codes(v);
  return c
    .filter((code) => FLAG_TAG[code] && !(code === 'aggressive_in_conformal_set' && c.includes('aggressive_top')))
    .map((code) => {
      const f = FLAG_TAG[code];
      const text = code === 'set_size_gt_1' ? `${v.setSize} classes` : f.text;
      return `<span class="ftag ${f.tone}" title="${esc(f.title)}" data-testid="reason-${esc(code)}">${esc(text)}</span>`;
    })
    .join('');
}

function matchesFlag(v, flag) {
  if (!flag || flag === 'all') return true;
  const c = codes(v);
  if (flag === 'safety') return c.includes('aggressive_in_conformal_set') && !c.includes('aggressive_top');
  if (flag === 'any') return c.some((code) => FLAG_TAG[code]);
  return c.includes(flag);
}

// ---- exports kept for other views (case header uses reasonBadges) -------------------------------------------------
export function setChips(v) {
  if (!v.risk && v.risk !== 0) return '<span class="sub">no score</span>';
  if (!v.setSize)
    return '<span class="chip empty" data-testid="set-chip" title="No class reached the confidence threshold">None</span>';
  const chip = (cls) =>
    `<span class="chip${cls === 'Invasive' ? ' inv' : ''}" data-testid="set-chip" title="${esc(suggestedLabel(cls))}">${esc(shortClass(cls))}</span>`;
  const order = (v.probs || []).map((p) => p.cls);
  const set = [...v.set].sort((a, b) => order.indexOf(a) - order.indexOf(b));
  if (set.length <= 2) return set.map(chip).join('');
  // 3+ classes: two named chips, then "+N". Invasive is always one of the named chips (safety rule); the other is
  // the most severe remaining class. Named chips keep class order; the hidden ones are listed in the tooltip.
  const keep = new Set(set.includes('Invasive') ? ['Invasive', set.filter((c) => c !== 'Invasive').pop()] : set.slice(-2));
  const shown = set.filter((c) => keep.has(c));
  const rest = set.filter((c) => !keep.has(c));
  return `${shown.map(chip).join('')}<span class="chip more" title="Also in the set: ${esc(rest.map(suggestedLabel).join(', '))}">+${rest.length}</span>`;
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

/** Live waiting cell; app.js tick() refreshes every [data-wait] once a second. */
export function waitCell(v) {
  const w = waitInfo(v.timer);
  return (
    `<span class="wait ${esc(w.state)}" data-wait="${esc(v.id)}" data-state="${esc(w.state)}">` +
    `<b>${esc(w.text)}</b> <span class="wt">${esc(w.target)}</span>` +
    `</span>`
  );
}
export const timerCell = waitCell;

export function tasks() {
  return '';
}

/** Surgical cases are held for injection (settings sim.hold_surgical_for_injection): say so, so none are forgotten. */
export function heldHint(sim) {
  if (!sim || !sim.hold_surgical_for_injection || !sim.pending) return '';
  const p = sim.pending.surgical || {};
  const total = (p.normal || 0) + (p.degraded || 0) + (p.composite || 0);
  if (!total) return '';
  return `<p class="held-hint" data-testid="held-surgical">${total} simulated frozen sections are waiting in the operating rooms. Send one with the <b>Frozen section</b> buttons in the header.</p>`;
}

// ---- pieces --------------------------------------------------------------------------------------------------------
function row(v, ST, rank) {
  const isNew = !ST.seen.has(v.id) && !ST.firstLoad;
  const pInv = v.risk === null ? '—' : v.risk + '%';
  const sugg =
    v.lane === 'surgical'
      ? `<span class="sugg surg">${esc(suggestedCall(v.margin ? { suggested_call: v.margin.call, involved: v.margin.involved } : null))}</span>`
      : `<span class="sugg${v.top === 'Invasive' ? ' inv' : ''}">${esc(suggestedLabel(v.top))}</span>`;
  const lane =
    v.lane === 'surgical'
      ? `<span class="lane surg">Frozen${v.room ? ' · ' + esc(v.room) : ''}</span>${v.variant ? `<span class="variant">${esc(v.variant === 'degraded' ? 'degraded' : 'composite')}</span>` : ''}`
      : '<span class="lane bx">Biopsy</span>';
  const next = nextAction(v);
  const label = `${v.id}, tier ${v.tier} ${TIER_NAME[v.tier] || ''}, ${v.lane === 'surgical' ? 'frozen section' : 'biopsy'}`;
  return (
    `<tr class="row tr-${num(v.tier)}${isNew ? ' new' : ''}${v.done ? ' done' : ''}" tabindex="0" aria-label="${esc(label)}" data-testid="worklist-row" data-case-id="${esc(v.id)}" data-tier="${num(v.tier)}" data-lane="${esc(v.lane)}" data-done="${v.done}" data-reasons="${esc(codes(v).join(' '))}" data-rank="${rank}">` +
    `<td class="c-tier"><span class="tierno t${num(v.tier)}" data-testid="tier-badge" title="Tier ${num(v.tier)} · ${esc(v.tierLabel)}">${num(v.tier)}</span></td>` +
    `<td class="c-case"><div class="cc"><b class="cid" title="${esc(v.id)}">${esc(v.id)}</b>${v.lane === 'surgical' ? `<span class="lane-inline">Frozen${v.room ? ' ' + esc(v.room) : ''}</span>` : ''}${isNew ? '<span class="newtag">New</span>' : ''}</div></td>` +
    `<td class="c-lane">${lane}</td>` +
    `<td class="c-sugg">${sugg}</td>` +
    `<td class="c-p"><span class="pnum${v.risk >= 50 ? ' hi' : ''}">${pInv}</span>${v.risk === null ? '' : miniBar(v.risk / 100, v.risk >= 50 ? 'hi' : '')}</td>` +
    `<td class="c-set"><span class="chips">${setChips(v)}</span></td>` +
    `<td class="c-flags">${flagTags(v) || '<span class="none" aria-label="no flags">–</span>'}</td>` +
    `<td class="c-wait">${waitCell(v)}</td>` +
    `<td class="c-next"><span class="next ${next.tone}"${v.agent.drafted ? ' data-testid="orders-drafted"' : ''}>${esc(next.text)}</span></td>` +
    '</tr>'
  );
}

function compositionBlock(comp, ST, next) {
  const seg = [1, 2, 3, 4]
    .filter((t) => comp.byTier[t])
    .map((t) => `<i class="seg-t${t}" style="flex-grow:${comp.byTier[t]}"></i>`)
    .join('');
  const label = [1, 2, 3, 4].map((t) => `${TIER_NAME[t]} ${comp.byTier[t]}`).join(', ');
  const legend = [1, 2, 3, 4]
    .map(
      (t) =>
        `<button class="leg" data-tier-filter="${t}" aria-pressed="${ST.tierFilter === t}" data-testid="tier-filter-${t}"><i class="sw seg-t${t}" aria-hidden="true"></i>${TIER_NAME[t]}<b>${comp.byTier[t]}</b></button>`
    )
    .join('');
  const nextBtn = next
    ? `<button class="btn p next-btn" data-next-case="${esc(next.case_id)}" data-testid="open-next"><span>Open next case <kbd>N</kbd></span><small>${esc(next.case_id)} · tier ${num(next.runtime && next.runtime.tier)}</small></button>`
    : '<button class="btn p next-btn" disabled data-testid="open-next"><span>Open next case</span><small>queue clear</small></button>';
  return (
    `<div class="wl-summary" data-testid="queue-stats">` +
    `<div class="comp"><div class="comp-head"><b data-testid="open-count">${comp.open}</b> open` +
    `<span class="comp-split"><span class="lane surg">${comp.surgical} frozen</span><span class="lane bx">${comp.biopsy} biopsy</span></span>` +
    `<span class="comp-meta"><span${comp.orders ? ' class="has"' : ''}>${comp.orders} with drafted orders</span>${comp.overdue ? `<span class="over">${comp.overdue} over target</span>` : ''}<span>${comp.done} done</span></span></div>` +
    `<div class="comp-bar" role="img" aria-label="Open cases by tier: ${esc(label)}">${seg || '<i class="seg-empty"></i>'}</div>` +
    `<div class="comp-legend" role="group" aria-label="Filter by tier">${legend}</div></div>` +
    nextBtn +
    '</div>'
  );
}

function howPopover() {
  return (
    '<details class="how" data-testid="how-prioritized"><summary aria-label="How cases are prioritized"><span class="how-q" aria-hidden="true">?</span>How cases are prioritized</summary>' +
    '<div class="how-pop" data-testid="order-note"><ol class="how-tiers">' +
    '<li><span class="tierno t1">1</span><div><b>Frozen section (simulated)</b>Always first. The timer runs from arrival; target 20 minutes, amber at 15.</div></li>' +
    '<li><span class="tierno t2">2</span><div><b>Likely invasive</b>Invasive is the top suggestion. <em>Safety rule:</em> if Invasive is anywhere in the 90% confidence set, the case is tier 2 even when it is not the top suggestion.</div></li>' +
    '<li><span class="tierno t3">3</span><div><b>Low confidence</b>More than one class in the confidence set, reliability flag raised, or too little tissue passed quality control.</div></li>' +
    '<li><span class="tierno t4">4</span><div><b>Unflagged</b>Read in turn. Unflagged does not mean cleared.</div></li></ol>' +
    '<p>Within a tier: higher calibrated P(Invasive) first, then earlier arrival. A tier 3 or 4 case that has waited 75% of its target moves up one tier (never above tier 2). Completed cases sink to the bottom.</p>' +
    '<p class="sub">Targets are illustrative, not validated: tier 2–3 within 24 h, tier 4 within 48 h (simulated clock).</p></div></details>'
  );
}

function toolbar(ST, views) {
  const seg = (attr, testKey, val, label, pressed) =>
    `<button data-${attr}="${val}" data-testid="${testKey}-${val}" aria-pressed="${pressed}">${label}</button>`;
  const status = (value, text) =>
    `<button data-status-filter="${value}" data-testid="status-${value}" aria-pressed="${(ST.status || 'all') === value}">${text}</button>`;
  const count = (f) => views.filter((v) => !v.done && matchesFlag(v, f)).length;
  const flagOpts = FLAG_FILTERS.map(
    ([val, text]) => `<option value="${val}"${ST.flagFilter === val ? ' selected' : ''}>${esc(text)} (${count(val)})</option>`
  ).join('');
  return (
    '<div class="wl-toolbar">' +
    `<label class="search"><svg aria-hidden="true"><use href="icons.svg#search"/></svg><input id="case-search" type="search" placeholder="Search case or flag" aria-label="Search cases" value="${esc(ST.search || '')}" autocomplete="off"><kbd aria-hidden="true">/</kbd></label>` +
    `<div class="seg" role="group" aria-label="Lane">${seg('lane-filter', 'lane', 'all', 'All lanes', ST.lane === 'all')}${seg('lane-filter', 'lane', 'surgical', 'Frozen', ST.lane === 'surgical')}${seg('lane-filter', 'lane', 'biopsy', 'Biopsy', ST.lane === 'biopsy')}</div>` +
    `<div class="seg" role="group" aria-label="Case status">${status('all', 'All')}${status('open', 'Open')}${status('done', 'Done')}</div>` +
    `<label class="sel"><span class="sr-only">Flag</span><select id="flag-filter" data-testid="flag-filter"><option value="all">All flags</option>${flagOpts}</select></label>` +
    `<label class="tog"><input type="checkbox" id="orders-only" data-testid="orders-only"${ST.ordersOnly ? ' checked' : ''}>Drafted orders</label>` +
    `<div class="seg sort" role="group" aria-label="Sort order">${seg('sort', 'order', 'ai', 'AI priority', ST.order === 'ai')}${seg('sort', 'order', 'arrival', 'Arrival', ST.order === 'arrival')}</div>` +
    howPopover() +
    '</div>'
  );
}

function head(ST) {
  const conf = conformalInfo(ST.metrics, ST.cfg);
  return (
    '<thead><tr><th class="c-tier" scope="col"><span class="sr-only">Tier</span></th><th class="c-case" scope="col">Case</th><th class="c-lane" scope="col">Lane</th>' +
    '<th class="c-sugg" scope="col">Suggested</th><th class="c-p" scope="col" title="Calibrated probability of invasive carcinoma">P(Invasive)</th>' +
    `<th class="c-set" scope="col" title="${esc(conf.label)}: the classes the model cannot rule out">Confidence set</th><th class="c-flags" scope="col">Flags</th>` +
    '<th class="c-wait" scope="col" title="Waiting time against the tier target">Waiting</th><th class="c-next" scope="col">Next step</th></tr></thead>'
  );
}

export function filterViews(ST, allViews) {
  const query = (ST.search || '').trim().toLowerCase();
  return allViews.filter(
    (v) =>
      (ST.status !== 'open' || !v.done) &&
      (ST.status !== 'done' || v.done) &&
      (!ST.tierFilter || (v.tier === ST.tierFilter && !v.done)) &&
      matchesFlag(v, ST.flagFilter) &&
      (!ST.ordersOnly || v.agent.drafted) &&
      (!query ||
        [
          v.id,
          v.lane === 'surgical' ? 'frozen surgical' : 'biopsy',
          v.room,
          v.variant,
          v.tierLabel,
          suggestedLabel(v.top),
          ...v.set,
          ...v.reasons.map((r) => r.label)
        ]
          .join(' ')
          .toLowerCase()
          .includes(query))
  );
}

export function render(ST, ctx) {
  const allViews = ST.items.map((c) => toView(c, ctx.now, ST.cfg, ST.sim));
  const comp = composition(allViews);
  const views = filterViews(ST, allViews);
  const filtered = views.length !== allViews.length;
  // "Open next case" follows what is on screen: the first open row of the current filters and order.
  const firstOpen = views.find((v) => !v.done);
  const next = firstOpen ? ST.items.find((c) => c.case_id === firstOpen.id) : nextCase(ST.items, null);
  let h = `<section class="wl" data-testid="worklist-view">` + compositionBlock(comp, ST, next);
  h += heldHint(ST.sim);
  h += toolbar(ST, allViews);
  if (!views.length) {
    const missing = ST.health && !ST.health.loaded;
    const title = allViews.length
      ? 'No matching cases'
      : missing
        ? 'No case artifacts are loaded'
        : ST.sim?.running
          ? 'Waiting for the next arrival'
          : 'The worklist is empty';
    const description = allViews.length
      ? 'Change the search or clear the filters to see the full queue.'
      : missing
        ? 'Import the Delta case bundle so real images, predictions and evidence can load.'
        : ST.sim?.running
          ? 'Cases appear here as the arrival replay brings them in.'
          : 'Start the arrival replay to bring in biopsies. Frozen sections are sent from the OR controls above.';
    return (
      h +
      `<div class="empty-state wl-empty" data-testid="state-empty"><h2>${title}</h2><p>${description}</p>` +
      (allViews.length ? '<button class="btn" data-clear-search>Clear filters</button>' : '') +
      '</div></section>'
    );
  }
  h += `<div class="tblw wl-table-wrap"><table class="wl-table" data-testid="worklist" aria-label="Case worklist${ST.order === 'ai' ? ', AI priority order' : ', arrival order'}">${head(ST)}<tbody>`;
  let group = null;
  const shown = views.slice(0, 400);
  shown.forEach((v, i) => {
    if (ST.order === 'ai') {
      const g = v.done ? 'done' : v.tier;
      if (g !== group) {
        group = g;
        const n = views.filter((x) => (x.done ? 'done' : x.tier) === g).length;
        const name = g === 'done' ? 'Completed' : `Tier ${num(g)} · ${esc(TIER_NAME[g] || v.tierLabel)}`;
        const note = g === 'done' ? 'Signed out, or margin call sent' : TIER_NOTE[g] || '';
        h += `<tr class="grp g-${esc(g)}" data-testid="tier-group" data-group="${esc(g)}"><th colspan="9" scope="colgroup"><span class="g-name">${name}</span><span class="g-n">${n}</span><span class="g-note">${esc(note)}</span></th></tr>`;
      }
    }
    h += row(v, ST, i + 1);
  });
  h += '</tbody></table></div>';
  if (views.length > shown.length) h += `<p class="sub wl-more">Showing the first ${shown.length} of ${views.length} cases. Narrow the filters to see the rest.</p>`;
  if (filtered)
    h += `<p class="wl-foot">${views.length} of ${allViews.length} cases shown. <button class="linkbtn" data-clear-search>Clear filters</button></p>`;
  return h + '</section>';
}
