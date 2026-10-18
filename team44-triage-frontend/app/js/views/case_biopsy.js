// Biopsy case view (plan §8.1 + agent panel). Left: image evidence, grounded summary, Why, reference comparison,
// activity. Right: the decision (orders → sign-out) and prediction & confidence. Cells tab (CELLS_SPEC §7.3).
// Interaction that needs no server round trip (tooltips, viewer, compare, confirmations) lives in case_ui.js.
import { classLabel, esc, mmss, toView } from '../adapter.js';
import { agentPanel } from './agent.js';
import { setRenderer } from './case_ui.js';
import { cellsTab } from './cells.js';
import { heatmap, heatmapControls, heatmapLegend } from './heatmap.js';
import { compared, howSure, summary, why } from './why.js';
import { reasonBadges } from './worklist.js';

const ACTION_TEXT = {
  approve_orders: 'orders placed',
  edit_orders: 'orders edited and placed',
  closer_look: 'closer look first',
  dismiss_flag: 'flag dismissed',
  sign_out: 'signed out',
  margin_call: 'margin call'
};
const CALL_WORDS = { clear: 'margins clear', tumor_at_margin: 'tumor at margin', defer: 'defer to permanent sections' };

export function activity(events) {
  if (!events || !events.length) return '<div class="note">No activity yet.</div>';
  const text = (e) => {
    const p = e.payload || {};
    if (e.type === 'arrived') return `Arrived${p.lane ? ` (${p.lane === 'surgical' ? 'simulated frozen section' : 'biopsy'})` : ''}`;
    if (e.type === 'agent_transition') return `Decision: ${ACTION_TEXT[e.action] || e.action} (${e.from} → ${e.to})`;
    if (e.type === 'routed') return `Routed to the ${p.to || 'next screen'}`;
    if (e.type === 'signed_out') return 'Signed out by the pathologist';
    if (e.type === 'margin_call')
      return `Margin call: ${CALL_WORDS[p.call] || p.call}${p.margins && p.margins.length ? ' · ' + p.margins.join(', ') : ''}`;
    return e.type.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
  };
  return `<ol class="events cv-events">${events
    .map((e) => `<li><time>${esc(new Date(e.at).toLocaleTimeString())}</time><span>${esc(text(e))}</span><span class="sub">sim +${mmss(e.sim_time_s)}</span></li>`)
    .join('')}</ol>`;
}

/** At-a-glance read for the case header (every value from the case JSON). */
function glance(v, cfg) {
  if (v.risk === null && !v.top) return '';
  const chips = v.setSize
    ? v.set.map((c) => `<span class="chip conf">${esc(c)}</span>`).join('')
    : '<span class="chip empty">∅ empty</span>';
  const r = v.reliability;
  return (
    '<dl class="cv-glance" aria-label="Read at a glance">' +
    `<div><dt>Suggested</dt><dd><b>${esc(classLabel(v.top, cfg) || '—')}</b></dd></div>` +
    `<div><dt>P(${esc(classLabel((cfg && cfg.aggressive_class) || 'Invasive', cfg))})</dt><dd><b>${v.risk === null ? '—' : v.risk + '%'}</b> calibrated</dd></div>` +
    `<div><dt>Conformal set</dt><dd class="chips">${chips}</dd></div>` +
    `<div><dt>Reliability</dt><dd>${r.score === null ? '—' : `<b>${r.score.toFixed(2)}</b>`}${r.flag ? ' <span class="pill warn">flag</span>' : ' no flag'}</dd></div>` +
    '</dl>'
  );
}

export function header(v, cfg) {
  return (
    `<button class="back" id="bk" data-testid="back">Back to worklist</button>` +
    `<div class="card cv-header"><div class="cv-head"><div class="cv-id"><h2 data-testid="case-title">${esc(v.id)} <span class="tier ${v.tierCls}" data-testid="tier-badge" title="${esc(v.tierNote)}">${esc(v.tierLabel)}</span></h2>` +
    `<div class="sub">${esc(v.laneLabel)}${v.variantLabel ? ' · ' + esc(v.variantLabel) : ''} · ${esc(v.question)} · arrived sim +${mmss(v.arrivedSim)}${v.synthetic ? ' · <b>synthetic</b>' : ''}</div>${reasonBadges(v)}</div>` +
    (cfg ? glance(v, cfg) : '') +
    '</div>'
  );
}

export function render(ST, ctx) {
  const c = ST.cases[ST.caseId];
  const v = toView(c, ctx.now, ST.cfg, ST.sim);
  setRenderer(() => render(ST, { ...ctx, now: Date.now() }), ctx.assetUrl);
  let h =
    `<section class="cv cv-biopsy" data-testid="case-view" data-lane="biopsy" data-case-id="${esc(v.id)}">` +
    header(v, ST.cfg) +
    `<div class="tabs" role="tablist" aria-label="Case sections"><button role="tab" data-tab="evidence" data-testid="tab-evidence" aria-selected="${ST.tab === 'evidence'}">Evidence</button>` +
    `<button role="tab" data-tab="cells" data-testid="tab-cells" aria-selected="${ST.tab === 'cells'}">Cells${v.cells.status === 'ok' ? ` <span class="tab-n">${(v.cells.tiles || []).length}</span>` : ''}</button></div></div>`;
  if (ST.tab === 'cells') {
    return h + `<div class="cv-tabpanel" role="tabpanel">${cellsTab(v, ST, ctx.assetUrl)}</div></section>`;
  }
  const hm = ST.hm;
  h += '<div class="grid case-layout cv-grid"><div class="cv-main">';
  h +=
    `<div class="card cv-image"><div class="card-h"><h2>Image evidence</h2><span class="sub">${esc(v.image ? `${v.image.grid[0]} × ${v.image.grid[1]} regions of ${v.image.tile} px` : '')}</span></div>` +
    `<figure data-testid="heatmap" class="cv-fig">${heatmapControls(v, hm)}${heatmap(v, hm, ctx.assetUrl)}<figcaption>${v.image ? heatmapLegend(v, hm) : ''}</figcaption></figure></div>`;
  h += `<div class="card" data-testid="summary"><div class="card-h"><h2>Grounded summary</h2></div>${summary(v)}</div>`;
  h += `<div class="card cv-why" data-testid="why"><div class="card-h"><h2>Why: features associated with invasive carcinoma</h2></div>${why(v, hm, ST.cfg, ST.metrics)}</div>`;
  h += `<div class="card" data-testid="compared"><div class="card-h"><h2>Reference comparison</h2></div>${compared(v, ST.cfg, ctx.assetUrl)}</div>`;
  const n = (ST.events[v.id] || []).length;
  h += `<details class="card activity-card" data-testid="case-events"><summary>Activity and decision history <span>${n} event${n === 1 ? '' : 's'}</span></summary>${activity(ST.events[v.id])}</details>`;
  h += '</div><div class="decision-column cv-side">';
  h += agentPanel(v, ST);
  h += `<div class="card" data-testid="how-sure"><div class="card-h"><h2>Prediction and confidence</h2></div>${howSure(v, ST.cfg, ST.metrics)}</div>`;
  if (v.tier === 4) h += '<p class="note">Unflagged means read in turn. It does not mean cleared.</p>';
  return h + '</div></div></section>';
}

