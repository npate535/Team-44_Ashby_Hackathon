// Surgical case view (simulated frozen section, plan §6.9 / §11 view 3): a big wall-clock timer, the four named margins
// on a specimen diagram and on the image (per-segment strips), per-margin scores against the threshold on a log-odds
// axis, a reliability indicator that explains a "defer" suggestion, and the three calls with confirmation.
// The pathologist makes the call; it goes to the surgeon screen and stops the timer. Everything here is simulated.
import { CALL_TEXT, classLabel, esc, MARGINS, num, ordinalPct, toView } from '../adapter.js';
import { agentPanel } from './agent.js';
import { activity, header } from './case_biopsy.js';
import { keepOpen, pending, setRenderer } from './case_ui.js';
import { cap, fmtLogit, marginBars, marginOverlay, marginStatus, specimenDiagram } from './heatmap.js';
import { aggProb, checksOnly, fmtPct, howSure, summary } from './why.js';

export function callText(call) {
  if (!call) return '';
  if (call.call === 'tumor_at_margin') {
    const m = call.margins || [];
    const list = m.length > 1 ? `${m.slice(0, -1).join(', ')} and ${m[m.length - 1]}` : m.join('');
    return `Tumor at ${list} margin${m.length > 1 ? 's' : ''} — more tissue needed`;
  }
  return CALL_TEXT[call.call] || call.call;
}

const SUGGESTED = {
  clear: 'Margins clear',
  tumor_at_margin: 'Tumor at margin',
  defer: 'Defer to permanent sections'
};

/** Plain-language reasons behind the reliability flag, from the case's own signals. */
function flagReasons(v, cfg) {
  const r = v.reliability;
  const out = [];
  if (r.oodPct !== null && r.oodPct >= 0.9) out.push(`input unlike the reference images (${ordinalPct(r.oodPct)} OOD percentile)`);
  if (v.agreement && v.agreement.conch_musk_same_top === false) out.push('the second model (MUSK) picks a different category');
  if (v.setSize !== 1) out.push(v.setSize ? `${v.setSize} categories in the conformal set` : 'an empty conformal set');
  if (v.degradation) out.push(`simulated ${v.degradation} artifact on this field`);
  return out;
}

function reliabilityBlock(v, cfg, metrics) {
  const r = v.reliability;
  const thr = metrics && metrics.eval && metrics.eval.reliability ? metrics.eval.reliability.threshold : null;
  const gauge =
    r.score === null
      ? ''
      : `<span class="rel-gauge" aria-hidden="true"><i style="width:${Math.round(Math.max(0, Math.min(1, r.score)) * 100)}%"></i>${
          typeof thr === 'number' ? `<b style="left:${Math.round((1 - thr) * 100)}%"></b>` : ''
        }</span>`;
  if (r.flag) {
    const why = flagReasons(v, cfg);
    return (
      `<div class="flag rel-flag" data-testid="reliability-flag" role="status"><b>Reliability flag raised — suggestion: defer to permanent sections.</b>` +
      `<span>Reliability ${r.score === null ? '—' : r.score.toFixed(2)}${why.length ? `; ${esc(why.join('; '))}` : ''}. The margin read is not trustworthy on this field; consider permanent sections.</span>${gauge}` +
      `${typeof thr === 'number' ? `<span class="sub">Flag when the estimated error probability reaches ${thr.toFixed(2)} (reliability ≤ ${(1 - thr).toFixed(2)}), a threshold set on calibration data to catch most errors.</span>` : ''}</div>`
    );
  }
  return (
    `<div class="rel-ok" data-testid="reliability-ok"><span class="ck-i" aria-hidden="true">✓</span><span><b>Reliability ${r.score === null ? '—' : r.score.toFixed(2)} · no flag</b>` +
    `<span class="sub">${r.oodPct === null ? '' : `OOD percentile ${ordinalPct(r.oodPct)} · `}${typeof thr === 'number' ? `flag at ≤ ${(1 - thr).toFixed(2)}` : 'no flag threshold available'}</span></span>${gauge}</div>`
  );
}

function suggestedText(m) {
  if (!m) return 'No margin read';
  if (m.call === 'tumor_at_margin')
    return `Tumor at ${m.involved.length ? m.involved.map(cap).join(', ') : 'a'} margin${m.involved.length === 1 ? '' : 's'}`;
  return SUGGESTED[m.call] || m.call;
}

function callPanel(v, ST) {
  const m = v.margin;
  const done = v.decision.marginCall;
  if (done)
    return `<div class="verdict" data-testid="verdict" role="status" tabindex="-1"><b>${esc(callText(done))}</b> — sent to the surgeon${done.at ? ` at ${esc(new Date(done.at).toLocaleTimeString())}` : ''}. Timer stopped.</div>`;
  const can = v.agent.allowed.includes('margin_call') && !ST.busy;
  const dis = can ? '' : ' disabled';
  const sug = m ? m.call : null;
  const p = pending(v.id);
  const primary = (call) => (sug === call ? ' p' : '');
  let h =
    `<div class="btns call-btns" role="group" aria-label="Margin call">` +
    `<button class="btn${primary('clear')}" data-cv-confirm="margin_clear" data-testid="call-clear"${dis}${p && p.action === 'margin_clear' ? ' aria-expanded="true"' : ''}>Margins clear</button>` +
    `<button class="btn${sug === 'tumor_at_margin' ? ' p dd' : ' d'}" data-call="tumor" data-testid="call-tumor"${dis}${ST.editing === 'margins' ? ' aria-expanded="true"' : ''}>Tumor at margin…</button>` +
    `<button class="btn${primary('defer')}" data-cv-confirm="margin_defer" data-testid="call-defer"${dis}${p && p.action === 'margin_defer' ? ' aria-expanded="true"' : ''}>Defer to permanent</button></div>`;
  if (!can && !ST.busy) h += '<p class="sub">The margin call is not available for this case.</p>';
  if (ST.busy) h += '<p class="sub" role="status">Sending your call…</p>';
  if (ST.editing === 'margins') {
    h +=
      `<div class="pick cv-pick" data-testid="margin-pick" role="group" aria-label="Pick the involved margins"><b class="pick-t">Which margins? Pre-selected from the suggestion; change as needed.</b>` +
      MARGINS.map((name) => {
        const st = m ? marginStatus(m, name) : 'unknown';
        return `<label class="${st}"><input type="checkbox" data-pick="${name}" data-testid="margin-pick-${name}"${ST.marginPick.includes(name) ? ' checked' : ''}>${cap(name)}<span class="sub">${m && m.perMargin[name] !== undefined ? esc(fmtLogit(m.perMargin[name])) : ''}</span></label>`;
      }).join('') +
      `<div class="btns"><button class="btn p dd" data-call="confirm" data-testid="call-confirm"${can && ST.marginPick.length ? '' : ' disabled'}>Send: tumor at margin</button><button class="btn" data-call="cancel" data-testid="call-cancel">Cancel</button></div>` +
      '<p class="sub">Sends “Tumor at the selected margin(s) — more tissue needed” to the surgeon and stops the timer.</p></div>';
  }
  if (p && (p.action === 'margin_clear' || p.action === 'margin_defer')) {
    const clear = p.action === 'margin_clear';
    const against = sug && sug !== (clear ? 'clear' : 'defer') ? ` The suggestion was “${esc(suggestedText(m))}”.` : '';
    h +=
      `<div class="cv-confirm" role="group" aria-labelledby="cv-confirm-t" data-testid="confirm-box" data-action="${p.action}">` +
      `<b id="cv-confirm-t">Send “${clear ? 'Margins clear' : 'Cannot determine — defer to permanent sections'}” to the surgeon?</b>` +
      `<p>This stops the timer and cannot be changed here.${against}</p>` +
      `<div class="btns"><button class="btn p" data-call="${clear ? 'clear' : 'defer'}" data-testid="confirm-action"${dis}>Confirm: ${clear ? 'margins clear' : 'defer'}</button><button class="btn" data-cv-cancel data-testid="confirm-cancel">Cancel</button></div></div>`;
  }
  return h;
}

export function render(ST, ctx) {
  const c = ST.cases[ST.caseId];
  const v = { ...toView(c, ctx.now, ST.cfg, ST.sim), degradation: c.degradation && c.degradation.type ? `${c.degradation.type}` : null };
  setRenderer(() => render(ST, { ...ctx, now: Date.now() }), ctx.assetUrl);
  const t = v.timer;
  const m = v.margin;
  const names = ST.cfg.margin.names;
  const cfg = ST.cfg;
  let h = `<section class="cv cv-surg" data-testid="case-view" data-lane="surgical" data-case-id="${esc(v.id)}">` + header(v) + '</div>';
  h +=
    `<div class="simbanner cv-sim" data-testid="sim-banner" role="note"><span class="sim-tag">Simulated frozen section</span><span>Image borders stand in for inked margins (top = ${esc(names.top)}, bottom = ${esc(names.bottom)}, left = ${esc(names.left)}, right = ${esc(names.right)}).` +
    (v.variant === 'degraded' ? ` This field carries a simulated frozen-section artifact${v.degradation ? ` (${esc(v.degradation)})` : ''} to test the reliability flag.` : '') +
    (v.variant === 'composite' ? ' Composite test field stitched from two source images for the margin-localization test (simulated).' : '') +
    ' Not a validated margin assessment.</span></div>';
  h += '<div class="grid cv-grid"><div class="cv-main">';
  h +=
    `<div class="card cv-margins"><div class="card-h"><h2>Margins</h2><span class="sub">${m && m.stripUm ? `outermost ≈${Math.round(m.stripUm)} µm strip, scored per segment` : ''}</span></div>` +
    (m ? '' : '<div class="note" data-testid="margin-null">Margin read unavailable. You can still make the call.</div>') +
    `<div class="mg-top">${specimenDiagram(v, names)}<div class="mg-bars">${marginBars(v)}</div></div>` +
    marginOverlay(v, ctx.assetUrl, { names }) +
    `<div class="lg-keys ms-keys"><span class="lg-key"><i class="sw-seg hot" aria-hidden="true"></i>Segment above threshold</span><span class="lg-key"><i class="sw-seg warm" aria-hidden="true"></i>Just below threshold</span><span class="lg-key"><i class="sw-seg cool" aria-hidden="true"></i>Below threshold</span></div>` +
    `<p class="sub" data-testid="margin-2d-note">${esc(ST.cfg.margin.note)}</p></div>`;
  h += `<div class="card" data-testid="summary"><div class="card-h"><h2>Summary</h2></div>${summary(v)}</div>`;
  const nEv = (ST.events[v.id] || []).length;
  h += `<details class="card activity-card" data-testid="case-events" data-keep="activity"${keepOpen('activity')}><summary>Activity <span>${nEv} event${nEv === 1 ? '' : 's'}</span></summary>${activity(ST.events[v.id])}</details>`;
  h += '</div><div class="decision-column cv-side">';
  h +=
    `<div class="card cv-timer ${esc(t.state)}"><div class="tm-h"><span>Time since received</span><span class="sub">target ${num(ST.cfg.timer.red_min)} min · amber at ${num(ST.cfg.timer.amber_min)}</span></div>` +
    `<div class="big timer ${t.state}" data-testid="surgical-timer" data-timer="${esc(v.id)}" data-state="${esc(t.state)}" role="timer" aria-live="off">${esc(t.text)}</div>` +
    `<div class="tm-bar" aria-hidden="true"><i data-timer-bar="${esc(v.id)}" class="${t.state === 'none' ? '' : esc(t.state)}" style="width:${t.usedFrac === null ? 0 : Math.round(t.usedFrac * 100)}%"></i></div>` +
    `<div class="tm-sub"><span data-timer-sub="${esc(v.id)}">${esc(t.countdown)}</span>${v.room ? `<span>${esc(v.room)} (simulated)</span>` : ''}</div></div>`;
  h +=
    `<div class="card cv-call"><div class="card-h"><h2>Margin call</h2></div>` +
    (m
      ? `<div class="sugg-call ${esc(m.call)}" data-testid="suggested-call"><span class="sub">Suggested by the read</span><b>${esc(suggestedText(m))}</b>${
          m.call === 'tumor_at_margin' && m.involved.length
            ? `<span class="sub">${m.involved.map((n) => `${cap(n)}${m.lengths && m.lengths[n] ? ` ≈${Math.round(m.lengths[n])} µm` : ''}`).map(esc).join(' · ')}</span>`
            : ''
        }</div>`
      : '') +
    reliabilityBlock(v, cfg, ST.metrics) +
    callPanel(v, ST) +
    '<p class="note">You make the call. It goes to the surgeon screen and stops the timer.</p></div>';
  h += `<div class="card" data-testid="how-sure"><div class="card-h"><h2>Case-level read</h2><span class="sub">${esc(classLabel(v.top, cfg) || '')}${v.risk === null ? '' : ` · ${fmtPct(aggProb(v, cfg))} P(invasive)`}</span></div>${howSure(v, ST.cfg, ST.metrics, { compact: true })}<details class="cv-more" data-keep="checks"${keepOpen('checks')}><summary>Agreement and quality checks</summary>${checksOnly(v, ST.cfg, ST.metrics)}</details></div>`;
  h += agentPanel(v, ST);
  return h + '</div></div></section>';
}
