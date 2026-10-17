// Surgical case view (simulated frozen section): big wall-clock timer, margin strips, reliability, three calls.
import {CALL_TEXT, esc, MARGINS, num, toView} from '../adapter.js';
import {agentPanel} from './agent.js';
import {activity, header} from './case_biopsy.js';
import {marginOverlay} from './heatmap.js';
import {howSure, summary} from './why.js';

export function callText(call) {
  if (!call) return '';
  if (call.call === 'tumor_at_margin') return `Tumor at ${call.margins.join(' and ')} margin — more tissue needed`;
  return CALL_TEXT[call.call] || call.call;
}

export function render(ST, ctx) {
  const c = ST.cases[ST.caseId];
  const v = toView(c, ctx.now, ST.cfg, ST.sim);
  const t = v.timer;
  const m = v.margin;
  const names = ST.cfg.margin.names;
  let h = `<section data-testid="case-view" data-lane="surgical" data-case-id="${esc(v.id)}">` + header(v) + '</div>';
  h += `<div class="simbanner" data-testid="sim-banner" style="margin-top:14px"><b>Simulated frozen section</b> — image borders stand in for inked margins (top = ${esc(names.top)}, bottom = ${esc(names.bottom)}, left = ${esc(names.left)}, right = ${esc(names.right)}).` +
    (v.variant === 'degraded' ? ' This field carries simulated frozen-section artifacts to test the reliability flag.' : '') +
    (v.variant === 'composite' ? ' Composite test field stitched from two source images for the margin-localization test (simulated).' : '') + '</div>';
  h += '<div class="grid"><div>';
  h += `<div class="card"><h2>Margins</h2>${m ? '' : '<div class="note" data-testid="margin-null">Margin read unavailable. You can still make the call.</div>'}${marginOverlay(v, ctx.assetUrl)}` +
    `<p class="sub" data-testid="margin-2d-note">${esc(ST.cfg.margin.note)}</p>` +
    (m && m.stripUm ? `<p class="sub">Each margin = the outermost ${Math.round(m.stripUm)} µm strip, scored per segment${m.threshold !== null ? `; involved when p > ${m.threshold.toFixed(2)}` : ''}.</p>` : '') + '</div>';
  h += `<div class="card" style="margin-top:14px" data-testid="summary"><h2>Summary</h2>${summary(v)}</div>`;
  h += '</div><div>';
  h += `<div class="card"><div class="sub">Time since received (wall clock) · target ${num(ST.cfg.timer.red_min)} min</div>` +
    `<div class="big timer ${t.state}" data-testid="surgical-timer" data-timer="${esc(v.id)}" data-state="${esc(t.state)}">${esc(t.text)}</div>` +
    `<div class="sub" data-timer-sub="${esc(v.id)}">${esc(t.countdown)}</div>` +
    (v.room ? `<div class="sub">${esc(v.room)} (simulated)</div>` : '') + '</div>';
  const r = v.reliability;
  h += `<div class="card" style="margin-top:14px"><h2>Read</h2>` +
    (r.flag ? '<div class="flag" data-testid="reliability-flag">Reliability flag raised — suggestion: defer to permanent sections.</div>'
      : `<div class="kv"><span>Reliability</span><b data-testid="reliability-ok">${r.score === null ? '—' : r.score.toFixed(2)} · no flag</b></div>`) +
    (m ? `<div style="margin:8px 0"><span class="pill" data-testid="suggested-call">Suggested: ${esc(m.call === 'tumor_at_margin' ? 'tumor at ' + m.involved.join(', ') : CALL_TEXT[m.call])}</span></div>` : '');
  const done = v.decision.marginCall;
  if (done) {
    h += `<div class="verdict" data-testid="verdict">${esc(callText(done))} — sent to the surgeon ${esc(new Date(done.at).toLocaleTimeString())}</div>`;
  } else {
    const can = v.agent.allowed.includes('margin_call') && !ST.busy;
    const dis = can ? '' : ' disabled';
    h += `<div class="btns"><button class="btn" data-call="clear" data-testid="call-clear"${dis}>Margins clear</button>` +
      `<button class="btn d" data-call="tumor" data-testid="call-tumor"${dis}>Tumor at margin…</button>` +
      `<button class="btn" data-call="defer" data-testid="call-defer"${dis}>Defer to permanent</button></div>`;
    if (ST.editing === 'margins') {
      h += `<div class="pick" data-testid="margin-pick">${MARGINS.map(name => `<label><input type="checkbox" data-pick="${name}" data-testid="margin-pick-${name}"${ST.marginPick.includes(name) ? ' checked' : ''}>${name[0].toUpperCase() + name.slice(1)}</label>`).join('')}` +
        '<button class="btn p" data-call="confirm" data-testid="call-confirm">Send: tumor at margin</button><button class="btn" data-call="cancel" data-testid="call-cancel">Cancel</button></div>';
    }
  }
  h += '<div class="note" style="margin-top:8px">The pathologist makes the call; it goes to the surgeon screen and stops the timer.</div></div>';
  h += `<div class="card" style="margin-top:14px" data-testid="how-sure"><h2>Case-level read</h2>${howSure(v, ST.cfg)}</div>`;
  h += `<div style="margin-top:14px">${agentPanel(v, ST)}</div>`;
  h += `<div class="card" style="margin-top:14px" data-testid="case-events"><h2>Activity</h2>${activity(ST.events[v.id])}</div>`;
  return h + '</div></div></section>';
}
