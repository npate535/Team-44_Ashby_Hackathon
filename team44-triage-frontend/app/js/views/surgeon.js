// Surgeon screen / OR board (adapts the teammate's orboard): room, countdown, 3-step bar, result in real time.
import { esc, toView } from '../adapter.js';
import { callText } from './case_surgical.js';

const STEPS = ['Received', 'Awaiting call', 'Result called'];

export function render(ST, ctx) {
  const surgical = ST.items
    .filter((c) => c.lane === 'surgical')
    .map((c) => toView(c, ctx.now, ST.cfg, ST.sim));
  const results = new Map();
  for (const r of ST.routed.surgeon) if (!results.has(r.case_id)) results.set(r.case_id, r); // newest first
  if (!surgical.length) {
    return '<div data-testid="surgeon-screen"><div class="card empty-state" data-testid="surgeon-empty"><h2>No specimen in progress</h2><p>Simulated frozen sections appear here when they are injected.</p></div></div>';
  }
  const cards = surgical
    .map((v) => {
      const r = results.get(v.id);
      const step = r ? 2 : 1;
      const steps = STEPS.map(
        (_, i) => `<span class="${i < step || (r && i === 2) ? 'done' : i === step ? 'on' : ''}"></span>`
      ).join('');
      const big = r ? 'Called' : v.timer.countdown.replace(' left', '');
      return (
        `<div class="card" data-testid="or-card" data-case-id="${esc(v.id)}"><div class="sub">${esc(v.room || 'OR')} (simulated) · surgeon view</div>` +
        `<div class="big timer ${v.timer.state}" data-timer="${esc(v.id)}" data-timer-mode="countdown" data-state="${esc(v.timer.state)}">${esc(big)}</div>` +
        `<div class="steps">${steps}</div><b>${STEPS[step]}</b>` +
        (r
          ? `<div class="verdict" data-testid="surgeon-result" data-call="${esc(r.call)}">${esc(callText(r))}<div class="sub">${esc(new Date(r.at).toLocaleTimeString())} · read time ${Math.round(r.timer_elapsed_s / 60)} min</div></div>`
          : '') +
        `<div class="sub">${esc(v.id)} · Simulated frozen section${v.variantLabel ? ' · ' + esc(v.variantLabel) : ''}</div><a class="case-link" href="#/case/${encodeURIComponent(v.id)}">Review case <span aria-hidden="true">→</span></a></div>`
      );
    })
    .join('');
  return `<div data-testid="surgeon-screen"><p class="note">Margin calls arrive here as the pathologist sends them. Research prototype — simulated frozen sections only.</p><div class="or">${cards}</div></div>`;
}
