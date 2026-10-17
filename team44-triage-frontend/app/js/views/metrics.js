// Metrics page: eval JSONs from GET /api/metrics. CP2 placeholder (pending states + raw JSON); charts land in CP3.
import {esc} from '../adapter.js';

const SECTIONS = [
  ['metrics', 'Headline: AUROC, macro-F1, balanced accuracy'], ['calibration', 'Calibration'],
  ['conformal', 'Conformal coverage and set size'], ['concepts', 'Concept sanity check (NCT-CRC, cross-organ)'],
  ['reliability', 'Reliability vs degradation'], ['retrieval', 'Retrieval precision@5'], ['queue_sim', 'Queue simulation'],
];

export function render(ST) {
  const m = ST.metrics;
  if (!m) return '<div data-testid="metrics"><div class="card" data-testid="state-loading"><h2>Loading metrics…</h2></div></div>';
  let h = '<div data-testid="metrics">';
  if (m.synthetic) h += '<div class="mock" data-testid="metrics-synthetic">Synthetic data: these numbers come from generated inputs and say nothing about model performance.</div>';
  for (const [key, title] of SECTIONS) {
    const data = m.eval[key];
    h += `<section class="card" style="margin-bottom:12px" data-testid="metrics-${key.replace(/_/g, '-')}"><h2>${esc(title)}</h2>` +
      (data ? `<details><summary>raw JSON</summary><pre style="white-space:pre-wrap;font-size:11px">${esc(JSON.stringify(data, null, 1))}</pre></details>`
        : `<div class="note" data-testid="metrics-missing-${key}">Pending: eval/${key}.json not produced yet.</div>`) +
      `<div class="sub">source: eval/${key}.json</div></section>`;
  }
  return h + '</div>';
}
