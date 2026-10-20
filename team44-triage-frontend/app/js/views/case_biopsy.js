// Biopsy case view (plan §8.1 + agent panel). Left: image evidence, grounded summary, Why, reference comparison,
// activity. Right: the decision (orders → sign-out) and prediction & confidence. Cells tab (CELLS_SPEC §7.3).
// Interaction that needs no server round trip (tooltips, viewer, compare, confirmations) lives in case_ui.js.
import { classLabel, esc, exactProb, mmss, toView } from '../adapter.js';
import { agentPanel } from './agent.js';
import { keepOpen, setRenderer } from './case_ui.js';
import { cellsTab } from './cells.js';
import { heatmap, heatmapControls, heatmapLegend } from './heatmap.js';
import { interpretPanel } from './interpret.js';
import { aggProb, compared, failedConcepts, fmtPct, howSure, setChip, summary, why } from './why.js';
import { reasonBadges } from './worklist.js';

const ACTION_TEXT = {
  approve_orders: 'orders approved',
  edit_orders: 'orders edited and approved',
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
    if (e.type === 'injected') return 'Simulated frozen section received';
    if (e.type === 'timer_started') return 'Frozen-section timer started';
    if (e.type === 'signed_out') return 'Signed out by the pathologist';
    if (e.type === 'margin_call')
      return `Margin call: ${CALL_WORDS[p.call] || p.call}${p.margins && p.margins.length ? ' · ' + p.margins.join(', ') : ''}`;
    return String(e.type || 'event').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
  };
  return `<ol class="events cv-events">${events
    .map((e) => `<li><time>${esc(e.at ? new Date(e.at).toLocaleTimeString() : '')}</time><span>${esc(text(e))}</span>${Number.isFinite(Number(e.sim_time_s)) && e.sim_time_s !== null ? `<span class="sub">sim +${mmss(e.sim_time_s)}</span>` : ''}</li>`)
    .join('')}</ol>`;
}

/** At-a-glance read for the case header (every value from the case JSON). */
function glance(v, cfg) {
  if (v.risk === null && !v.top) return '';
  const chips = v.setSize
    ? v.set.map((c) => setChip(c, cfg)).join('')
    : '<span class="chip empty">∅ empty</span>';
  const r = v.reliability;
  return (
    '<dl class="cv-glance" aria-label="Read at a glance">' +
    `<div><dt>Suggested</dt><dd><b>${esc(classLabel(v.top, cfg) || '—')}</b></dd></div>` +
    `<div><dt>P(${esc(classLabel((cfg && cfg.aggressive_class) || 'Invasive', cfg))})</dt><dd><b title="${esc(v.risk === null ? '' : exactProb(aggProb(v, cfg), 4))}">${v.risk === null ? '—' : fmtPct(aggProb(v, cfg))}</b> calibrated</dd></div>` +
    `<div><dt>Conformal set</dt><dd class="chips">${chips}</dd></div>` +
    `<div><dt>Reliability</dt><dd>${r.score === null ? 'not computed' : `<b>${r.score.toFixed(2)}</b>${r.flag ? ' <span class="pill warn">flag</span>' : ' no flag'}`}</dd></div>` +
    '</dl>'
  );
}

/** A tab label with a shorter form for phones (the button's aria-label keeps the full name). */
const tabText = (long, short) => `<span class="tab-long">${esc(long)}</span><span class="tab-short" aria-hidden="true">${esc(short)}</span>`;

// A surgical case says "simulated" once, in the tag next to its id (and once in the Margins card); its tier, lane,
// variant and question labels carry no repeat of it. The global prototype notice says so as well.
const unsim = (t) => String(t || '').replace(/\s*\(simulated\)/gi, '').replace(/^simulated\s+(\w)/i, (_, ch) => ch.toUpperCase());
function surgicalLine(v) {
  const variant = v.variant === 'degraded' ? `${v.degradation || 'frozen-section'} artifact` : v.variant === 'composite' ? 'composite test field' : '';
  return [unsim(v.laneLabel), variant, unsim(v.question)].filter(Boolean).map(esc).join(' · ');
}

export function header(v, cfg, { backHref = null, backLabel = 'Back to worklist' } = {}) {
  const back = backHref
    ? `<a class="back" href="${esc(backHref)}" data-testid="back">${esc(backLabel)}</a>`
    : `<button class="back" id="bk" data-testid="back">${esc(backLabel)}</button>`;
  const surg = v.lane === 'surgical';
  // arrival on the replay clock (minutes:seconds after the replay started), not a wall-clock time
  const when = v.external ? '' : ` · arrived +${mmss(v.arrivedSim)}${surg ? ' in the replay' : ' (simulated replay)'}`;
  const simTag = surg
    ? ' <span class="cv-simtag" data-testid="sim-tag" title="Simulated frozen section: a BACH microscopy field stands in for a frozen section, and its borders for the inked margins">Simulated</span>'
    : '';
  return (
    `${back}` +
    `<div class="card cv-header"><div class="cv-head"><div class="cv-id"><h2 data-testid="case-title"><span class="cv-title-id">${esc(v.id)}</span>${simTag} <span class="tier ${v.tierCls}" data-testid="tier-badge" title="${esc(v.tierNote)}">${esc(surg ? unsim(v.tierLabel) : v.tierLabel)}</span></h2>` +
    `<div class="sub">${surg ? surgicalLine(v) : `${esc(v.laneLabel)}${v.variantLabel ? ' · ' + esc(v.variantLabel) : ''} · ${esc(v.question)}`}${when}${v.synthetic ? ' · <b>synthetic</b>' : ''}</div>${reasonBadges(v)}</div>` +
    (cfg ? glance(v, cfg) : '') +
    '</div>'
  );
}

export function render(ST, ctx, opts = {}) {
  const c = ST.cases[ST.caseId];
  const v = toView(c, ctx.now, ST.cfg, ST.sim);
  v.spotFailed = failedConcepts(ST.metrics, v); // concepts whose AI spot check failed: marked wherever they are named
  setRenderer(() => render(ST, { ...ctx, now: Date.now() }, opts), ctx.assetUrl);
  const hideAgent = !!opts.hideAgent || !!v.external;
  // TCGA fields have no plain-language summary (none is generated or served for them): no tab, evidence instead
  const tab = ST.tab === 'cells' ? 'cells' : ST.tab === 'interp' && !v.external ? 'interp' : 'evidence';
  const tabBtn = (id, label, short, extra = '') =>
    `<button role="tab" id="tab-${id}" aria-controls="case-panel" data-tab="${id}" data-testid="tab-${id}" aria-label="${esc(label)}" aria-selected="${tab === id}"${tab === id ? '' : ' tabindex="-1"'}>${tabText(label, short)}${extra}</button>`;
  let h =
    `<section class="cv cv-biopsy" data-testid="case-view" data-lane="biopsy" data-case-id="${esc(v.id)}">` +
    header(v, ST.cfg, opts) +
    '<div class="tabs" role="tablist" aria-label="Case sections">' +
    tabBtn('evidence', 'Image & findings', 'Findings') +
    tabBtn('cells', 'Cell evidence', 'Cells', v.cells.status === 'ok' ? ` <span class="tab-n">${(v.cells.tiles || []).length}</span>` : '') +
    (v.external
      ? ''
      : tabBtn('interp', 'Plain-language summary', 'Plain language', (ST.interp[v.id] || {}).status === 'ok' ? ' <span class="tab-n ai-dot" title="Ready">✓</span>' : '')) +
    '</div></div>';
  if (tab === 'cells') {
    return h + `<div id="case-panel" class="cv-tabpanel" role="tabpanel" aria-labelledby="tab-cells">${cellsTab(v, ST, ctx.assetUrl)}</div></section>`;
  }
  if (tab === 'interp') {
    return h + `<div id="case-panel" class="cv-tabpanel ai-tabpanel" role="tabpanel" aria-labelledby="tab-interp">${interpretPanel(ST, v.id, 'case')}</div></section>`;
  }
  const hm = ST.hm;
  h += '<div id="case-panel" class="cv-evidence-panel" role="tabpanel" aria-labelledby="tab-evidence">';
  // the decision column comes after every evidence control in the tab order: keyboard users can jump straight to it
  if (!hideAgent) h += '<button type="button" class="cv-skip" data-cv-skip data-testid="skip-decision">Skip to the decision</button>';
  h += '<div class="grid case-layout cv-grid"><div class="cv-main">';
  h += `<div class="card" data-testid="summary"><div class="card-h"><h2>Grounded summary</h2><span class="sub">Citations open the supporting evidence</span></div>${summary(v)}</div>`;
  h +=
    `<div class="card cv-image"><div class="card-h"><h2>Image evidence</h2><span class="sub">${esc(v.image ? `${v.image.grid[0]} × ${v.image.grid[1]} regions of ${v.image.tile} px` : '')}</span></div>` +
    `<figure data-testid="heatmap" class="cv-fig">${heatmapControls(v, hm)}${heatmap(v, hm, ctx.assetUrl)}<figcaption>${v.image ? heatmapLegend(v, hm) : ''}</figcaption></figure></div>`;
  h += `<div class="card cv-why" data-testid="why"><div class="card-h"><h2>Why: features associated with invasive carcinoma</h2></div>${why(v, hm, ST.cfg, ST.metrics)}</div>`;
  // an external (TCGA) field has no reference cases and no workflow: no empty cards for them
  if (!v.external || v.similar.length || v.contrast)
    h += `<div class="card" data-testid="compared"><div class="card-h"><h2>Reference comparison</h2></div>${compared(v, ST.cfg, ctx.assetUrl)}</div>`;
  const n = (ST.events[v.id] || []).length;
  if (!v.external || n)
    h += `<details class="card activity-card" data-testid="case-events" data-keep="activity"${keepOpen('activity')}><summary>Activity and decision history <span>${n} event${n === 1 ? '' : 's'}</span></summary>${activity(ST.events[v.id])}</details>`;
  h += '</div><div class="decision-column cv-side">';
  if (!hideAgent) h += agentPanel(v, ST);
  h += `<div class="card" data-testid="how-sure"><div class="card-h"><h2>Prediction and confidence</h2></div>${howSure(v, ST.cfg, ST.metrics, { provOpen: !!keepOpen('prov') })}</div>`;
  if (v.external)
    h += '<p class="note">External TCGA field — read-only. Not part of the arrival worklist.</p>';
  else if (v.tier === 4) h += '<p class="note">Unflagged: read in arrival order. The model has not cleared this case.</p>';
  return h + '</div></div></div></section>';
}

