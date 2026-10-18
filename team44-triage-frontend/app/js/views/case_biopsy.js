// Biopsy case view (plan §8.1 + agent panel): heatmap + summary on the left; Why, How sure, Compared to, Agent,
// Activity on the right; Cells tab. Adapts the teammate's detail() grid/cards.
import { esc, mmss, toView } from '../adapter.js';
import { agentPanel } from './agent.js';
import { cellsTab } from './cells.js';
import { heatmap, heatmapControls } from './heatmap.js';
import { compared, howSure, summary, why } from './why.js';
import { reasonBadges } from './worklist.js';

export function activity(events) {
  if (!events || !events.length) return '<div class="note">No activity yet.</div>';
  const text = (e) => {
    if (e.type === 'agent_transition') return `${e.from} → ${e.to} (${e.action})`;
    if (e.type === 'routed') return `routed to the ${e.payload && e.payload.to}`;
    if (e.type === 'margin_call')
      return `margin call: ${e.payload.call}${e.payload.margins && e.payload.margins.length ? ' · ' + e.payload.margins.join(', ') : ''}`;
    return e.type.replace(/_/g, ' ');
  };
  return `<ul class="events">${events.map((e) => `<li><b>${esc(new Date(e.at).toLocaleTimeString())}</b> · sim +${mmss(e.sim_time_s)} · ${esc(text(e))}</li>`).join('')}</ul>`;
}

export function header(v) {
  return (
    `<button class="back" id="bk" data-testid="back">Back to worklist</button>` +
    `<div class="card"><h2 data-testid="case-title">${esc(v.id)} · <span class="tier ${v.tierCls}" data-testid="tier-badge">${esc(v.tierLabel)}</span></h2>` +
    `<div class="sub">${esc(v.laneLabel)}${v.variantLabel ? ' · ' + esc(v.variantLabel) : ''} · ${esc(v.question)} · arrived sim +${mmss(v.arrivedSim)}${v.synthetic ? ' · <b>synthetic</b>' : ''}</div>${reasonBadges(v)}`
  );
}

export function render(ST, ctx) {
  const c = ST.cases[ST.caseId];
  const v = toView(c, ctx.now, ST.cfg, ST.sim);
  let h =
    `<section data-testid="case-view" data-lane="biopsy" data-case-id="${esc(v.id)}">` +
    header(v) +
    `<div class="tabs" role="tablist"><button role="tab" data-tab="evidence" data-testid="tab-evidence" aria-selected="${ST.tab === 'evidence'}">Evidence</button>` +
    `<button role="tab" data-tab="cells" data-testid="tab-cells" aria-selected="${ST.tab === 'cells'}">Cells</button></div></div>`;
  if (ST.tab === 'cells') {
    return h + `<div style="margin-top:14px">${cellsTab(v, ST, ctx.assetUrl)}</div></section>`;
  }
  const hm = ST.hm;
  h += '<div class="grid case-layout" style="margin-top:14px"><div>';
  h +=
    `<div class="card"><h2>Image evidence</h2><figure data-testid="heatmap" style="margin:0">${heatmapControls(v, hm)}${heatmap(v, hm, ctx.assetUrl)}</figure>` +
    '<div class="sub" style="margin-top:6px">Each square is one 512-px region. Hatched regions failed quality control.</div></div>';
  h += `<div class="card" style="margin-top:14px" data-testid="summary"><h2>Grounded summary</h2>${summary(v)}</div>`;
  h += `<div class="card" style="margin-top:14px" data-testid="why"><h2>Why: features associated with invasive carcinoma</h2>${why(v, hm)}</div>`;
  h += `<div class="card" style="margin-top:14px" data-testid="compared"><h2>Reference comparison</h2>${compared(v, ST.cfg, ctx.assetUrl)}</div>`;
  h += `<details class="card activity-card" style="margin-top:14px" data-testid="case-events"><summary>Activity & decision history <span>${(ST.events[v.id] || []).length} event${(ST.events[v.id] || []).length === 1 ? '' : 's'}</span></summary>${activity(ST.events[v.id])}</details>`;
  h += '</div><div class="decision-column">';
  h += agentPanel(v, ST);
  h += `<div class="card" style="margin-top:14px" data-testid="how-sure"><h2>Prediction & confidence</h2>${howSure(v, ST.cfg, ST.metrics)}</div>`;
  if (v.tier === 4) h += '<p class="note">Unflagged means read in turn. It does not mean cleared.</p>';
  return h + '</div></div></section>';
}
