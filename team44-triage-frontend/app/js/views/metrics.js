// Metrics page: every eval JSON from GET /api/metrics. The renderer is shape-tolerant (Task 11 schemas were not
// frozen): every interval-bearing number is listed with its CI and n, known shapes also get an inline-SVG chart,
// and each section keeps a raw-JSON fallback. Missing files show "pending"; synthetic files carry a banner.
import { esc, humanize } from '../adapter.js';
import { barCI, ci, ciText, lineChart, table } from '../svg.js';

const META = new Set([
  'name',
  'created_utc',
  'config_hash',
  'git_sha',
  'git_dirty',
  'versions',
  'platform',
  'synthetic',
  'inputs',
  'outputs',
  'args',
  'note',
  'notes',
  'assumptions',
  'source',
  'description'
]);
const DEGRADE = ['blur', 'jpeg', 'stain', 'resample'];
const POLICY_LABEL = {
  fifo: 'FIFO (arrival order)',
  tiers: 'PathQueue tiers',
  tiers_aging: 'Tiers + aging guard'
};

const SECTIONS = [
  ['metrics', 'Headline: does triage separate invasive cases?', 'headline'],
  ['calibration', 'Is the confidence honest? Calibration', 'calibration'],
  ['conformal', 'Conformal coverage and set size', 'conformal'],
  ['concepts', 'Concept sanity check (NCT-CRC-HE-100K, cross-organ)', 'concepts'],
  ['concepts_bach', 'Concepts on BACH: per-class means', null],
  ['reliability', 'Reliability flag vs simulated frozen-section artifacts', 'reliability'],
  ['retrieval', 'Similar-case retrieval (precision@5)', null],
  ['queue_sim', 'Workflow: queue simulation', 'queue'],
  ['margin', 'Margin read: localizes or echoes?', null],
  ['deletion', 'Explanation faithfulness: deletion test', null],
  ['cells', 'Cell measurements: validation', null],
  ['model_choice', 'Model choice (patient-grouped CV inside train, one-SE rule)', null],
  ['thunder', 'THUNDER benchmark (same split, different protocol)', null],
  ['zeroshot', 'Zero-shot baselines', null],
  ['finetune', 'LoRA fine-tune (single calibration-split comparison)', null],
  ['spotcheck', 'Spot-check (qualitative)', null]
];

/** Every CI-shaped object (or bare number) under obj, with a readable path; does not descend into CI objects. */
export function collect(obj, path = [], out = [], depth = 0) {
  if (depth > 6 || obj === null || obj === undefined) return out;
  if (typeof obj === 'number') {
    out.push({ path, c: ci(obj), bare: true });
    return out;
  }
  if (Array.isArray(obj)) {
    if (obj.length <= 12)
      obj.forEach((v, i) => {
        if (v && typeof v === 'object') collect(v, [...path, labelFor(v, i)], out, depth + 1);
      });
    return out;
  }
  if (typeof obj !== 'object') return out;
  const c = ci(obj);
  if (c && ('lo' in obj || 'ci' in obj || 'n' in obj)) {
    out.push({ path, c, bare: false });
    return out;
  }
  for (const [k, v] of Object.entries(obj)) if (!META.has(k)) collect(v, [...path, k], out, depth + 1);
  return out;
}
const labelFor = (v, i) => v.name || v.model || v.policy || v.type || v.class || v.concept || String(i);
const pathText = (p) => p.map(humanize).join(' · ') || 'value';

function ciTable(data, limit = 24) {
  // bare numbers inside arrays (e.g. reliability-diagram bins) are chart data, not reported quantities
  const rows = collect(data).filter(
    (r) => !r.bare || (r.path.length && !r.path.some((p) => /^\d+$/.test(p)))
  );
  if (!rows.length) return '';
  const fmt = (r) => [
    esc(pathText(r.path)),
    esc(r.bare ? ciText(r.c).replace(' (no CI)', '') + ' (no CI)' : ciText(r.c))
  ];
  const head = rows.slice(0, limit).map(fmt);
  let h = table(['Quantity', 'Value (interval)'], head);
  if (rows.length > limit)
    h += `<details><summary>${rows.length - limit} more</summary>${table(['Quantity', 'Value (interval)'], rows.slice(limit).map(fmt))}</details>`;
  return h;
}

// ---- shape finders --------------------------------------------------------------------------------------------------------
function walk(obj, fn, path = [], depth = 0) {
  if (depth > 6 || !obj || typeof obj !== 'object') return;
  fn(obj, path);
  for (const [k, v] of Object.entries(obj)) if (!META.has(k)) walk(v, fn, [...path, k], depth + 1);
}

/** Reliability-diagram bins: arrays of {confidence-ish, accuracy-ish}. */
function findBins(data) {
  const out = [];
  walk(data, (o, path) => {
    if (!Array.isArray(o) || !o.length || typeof o[0] !== 'object') return;
    const keys = Object.keys(o[0]);
    const xk = keys.find((k) => /conf|mean_p|pred|p_mean|mean_prob/i.test(k));
    const yk = keys.find((k) => /acc|frac|obs|freq|empirical/i.test(k));
    if (!xk || !yk) return;
    out.push({
      name: humanize(path.filter((p) => !/bins|diagram|reliability/i.test(p)).join(' ') || 'bins'),
      points: o
        .filter((b) => typeof b[xk] === 'number' && typeof b[yk] === 'number')
        .map((b) => ({ x: b[xk], y: b[yk], n: b.n ?? b.count }))
    });
  });
  return out.slice(0, 3);
}

/** {blur|jpeg|stain|resample: severities} → line series of accuracy vs severity. */
function findDegradeSeries(data) {
  let found = null;
  walk(data, (o) => {
    if (found || Array.isArray(o)) return;
    const types = Object.keys(o).filter((k) => DEGRADE.includes(k));
    if (types.length < 2) return;
    found = types
      .map((t) => {
        const v = o[t];
        const items = Array.isArray(v) ? v.map((e) => [e.severity ?? e.level, e]) : Object.entries(v || {});
        const points = items
          .map(([sev, e]) => {
            const c = ci(
              e && typeof e === 'object' && !('estimate' in e)
                ? (e.accuracy ?? e.acc ?? e.balanced_accuracy ?? e.value)
                : e
            );
            return c && sev !== undefined ? { x: +sev, y: c.est, lo: c.lo, hi: c.hi, n: c.n } : null;
          })
          .filter(Boolean);
        return { name: humanize(t), points };
      })
      .filter((s) => s.points.length);
  });
  return found || [];
}

/** {fifo, tiers, tiers_aging: {...}} → the first metric all policies share (prefer aggressive time-to-first-look). */
function findPolicyBars(data) {
  let policies = null;
  walk(data, (o) => {
    if (policies || Array.isArray(o)) return;
    if ('fifo' in o && 'tiers' in o) policies = o;
  });
  if (!policies) return null;
  const names = Object.keys(policies).filter((k) => POLICY_LABEL[k] || typeof policies[k] === 'object');
  const per = names.map((n) => collect(policies[n]).filter((r) => !r.bare));
  if (!per.length || !per[0].length) return null;
  const key = (r) => r.path.join('.');
  const common = per[0].map(key).filter((k) => per.every((list) => list.some((r) => key(r) === k)));
  if (!common.length) return null;
  const pick =
    common.find((k) => /aggressive/i.test(k) && /mean/i.test(k)) ||
    common.find((k) => /aggressive/i.test(k)) ||
    common[0];
  const rows = names.map((n, i) => ({
    label: POLICY_LABEL[n] || humanize(n),
    ...per[i].find((r) => key(r) === pick).c,
    series: i
  }));
  return { metric: pick, rows };
}

/** Concept AUROCs keyed by concept id. */
function findConceptBars(data, ids) {
  let rows = null;
  walk(data, (o) => {
    if (rows || Array.isArray(o)) return;
    const keys = Object.keys(o).filter((k) => ids.has(k));
    if (keys.length < 2) return;
    rows = keys
      .map((k) => {
        const v = o[k];
        const c = ci(
          v && typeof v === 'object' && !('estimate' in v) ? (v.auroc ?? v.cav_auroc ?? v.value) : v
        );
        return c ? { label: humanize(k), ...c } : null;
      })
      .filter(Boolean);
  });
  return rows || [];
}

function chartFor(kind, data, ST) {
  if (kind === 'calibration') {
    const series = findBins(data);
    return series.length
      ? lineChart(series, {
          xLabel: 'mean predicted probability',
          yLabel: 'observed frequency',
          yMin: 0,
          yMax: 1,
          title: 'Reliability diagram',
          diagonal: true
        })
      : '';
  }
  if (kind === 'reliability') {
    // one panel per artifact type (small multiples, shared y range): 4 series would exceed the validated palette
    const series = findDegradeSeries(data);
    if (!series.length) return '';
    const ys = series.flatMap((s) => s.points.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]));
    const yMin = Math.max(0, Math.floor(Math.min(...ys) * 10) / 10),
      yMax = Math.min(1, Math.ceil(Math.max(...ys) * 10) / 10);
    return (
      '<h3>Accuracy vs simulated artifact severity (0 = clean)</h3><div class="multiples">' +
      series
        .map(
          (s) =>
            `<div><div class="sub" style="font-weight:600">${esc(s.name)}</div>${lineChart([s], { xLabel: 'severity', yLabel: 'accuracy', yMin, yMax, height: 230, width: 320, title: s.name + ' accuracy vs severity' })}</div>`
        )
        .join('') +
      '</div>'
    );
  }
  if (kind === 'queue') {
    const bars = findPolicyBars(data);
    return bars
      ? `<h3>${esc(pathText(bars.metric.split('.')))} (minutes)</h3>` +
          barCI(bars.rows, { title: 'Queue simulation by policy' })
      : '';
  }
  if (kind === 'concepts') {
    const ids = new Set(Object.values(ST.cfg.labels.concept_groups).flatMap((g) => g.ids));
    const rows = findConceptBars(data, ids);
    return rows.length
      ? '<h3>AUROC per concept</h3>' +
          barCI(rows, { min: 0, max: 1, ref: 0.5, refLabel: 'chance', title: 'Concept AUROCs' })
      : '';
  }
  if (kind === 'conformal') {
    const rows = collect(data)
      .filter((r) => !r.bare && /coverage/i.test(r.path.join(' ')))
      .slice(0, 8)
      .map((r) => ({ label: pathText(r.path), ...r.c }));
    const target = 1 - ST.cfg.alpha;
    return rows.length
      ? `<h3>Empirical coverage (target ${Math.round(target * 100)}%)</h3>` +
          barCI(rows, { min: 0, max: 1, ref: target, refLabel: 'target', title: 'Conformal coverage' })
      : '';
  }
  if (kind === 'headline') {
    const rows = collect(data)
      .filter((r) => !r.bare && /auroc/i.test(r.path.join(' ')))
      .slice(0, 6)
      .map((r) => ({ label: pathText(r.path), ...r.c }));
    return rows.length
      ? '<h3>Invasive-vs-rest AUROC</h3>' + barCI(rows, { min: 0.5, max: 1, title: 'AUROC' })
      : '';
  }
  return '';
}

function notes(data) {
  const list = data.assumptions || data.notes || data.note;
  if (!list) return '';
  const items = Array.isArray(list)
    ? list
    : typeof list === 'object'
      ? Object.entries(list).map(
          ([k, v]) => `${humanize(k)}: ${typeof v === 'object' ? JSON.stringify(v) : v}`
        )
      : [list];
  return `<details><summary>Assumptions and notes</summary><ul>${items.map((i) => `<li>${esc(i)}</li>`).join('')}</ul></details>`;
}

export function render(ST, ctx) {
  const m = ST.metrics;
  if (!m)
    return '<div data-testid="metrics"><div class="card" data-testid="state-loading"><h2>Loading metrics…</h2></div></div>';
  const ev = m.eval || {};
  let h =
    '<div data-testid="metrics"><div class="bar"><div class="note">Every number traces to a file in <code>artifacts/eval/</code>. Intervals are 95% cluster-bootstrap CIs over patient / pseudo-patient groups unless the row says otherwise. Headline case performance uses the clean test split. Each section identifies its own evaluation population; model selection uses grouped training folds. Degraded and composite variants never enter headline numbers.</div>' +
    '<button class="btn" data-metrics="refresh" data-testid="metrics-refresh">Refresh</button></div>';
  if (m.synthetic)
    h +=
      '<div class="mock" data-testid="metrics-synthetic"><b>Synthetic data.</b> These numbers come from generated inputs and say nothing about model performance.</div>';
  if ((m.errors || []).length)
    h += `<div class="flag">Unreadable eval files: ${m.errors.map((e) => esc(e.file)).join(', ')}</div>`;
  const known = new Set(SECTIONS.map((s) => s[0]));
  const sections = [
    ...SECTIONS,
    ...Object.keys(ev)
      .filter((k) => !known.has(k))
      .map((k) => [k, humanize(k), null])
  ];
  for (const [key, title, kind] of sections) {
    const data = ev[key];
    const tid = key.replace(/_/g, '-');
    if (
      !data &&
      !['metrics', 'calibration', 'conformal', 'concepts', 'reliability', 'retrieval', 'queue_sim'].includes(
        key
      )
    )
      continue;
    h += `<section class="card" style="margin-bottom:12px" data-testid="metrics-${tid}"><h2>${esc(title)}</h2>`;
    if (!data) {
      h += `<div class="note" data-testid="metrics-missing-${key}">Pending: <code>eval/${esc(key)}.json</code> has not been produced yet.</div></section>`;
      continue;
    }
    if (data.synthetic) h += '<div class="sub" style="color:var(--amb)">synthetic</div>';
    h +=
      chartFor(kind, data, ST) +
      ciTable(data) +
      notes(data) +
      `<details><summary>raw JSON</summary><pre class="raw">${esc(JSON.stringify(data, null, 1))}</pre></details>` +
      `<div class="sub">source: eval/${esc(key)}.json${data.created_utc ? ' · ' + esc(data.created_utc) : ''}</div></section>`;
  }
  if ((m.figures || []).length) {
    h += `<section class="card" style="margin-bottom:12px" data-testid="metrics-figures"><h2>Figures</h2><div class="cells-grid">${m.figures
      .map(
        (f) =>
          `<figure><img src="${esc(ctx.assetUrl(f.url))}" alt="${esc(f.file)}" loading="lazy" style="aspect-ratio:auto;object-fit:contain;background:#fff"><figcaption class="sub">eval/${esc(f.file)}</figcaption></figure>`
      )
      .join('')}</div></section>`;
  }
  h +=
    '<section class="card" data-testid="limitations"><h2>Limitations</h2><ul>' +
    '<li>The surgical lane is <b>simulated</b>: image borders stand in for inked margins and frozen-section artifacts are synthetic degradations. No frozen-section data was available.</li>' +
    '<li>BACH is small (400 microscopy images; test split 132, with few identified patients), so intervals are wide. Conformal coverage requires exchangeability; compare the observed held-out coverage above with the nominal target. The target is not a per-case guarantee.</li>' +
    '<li>BACH labels are diagnostic categories, not Nottingham grades; concepts are features associated with invasive carcinoma, not causes.</li>' +
    '<li>Follow-up order rules are illustrative and require clinical review. Spot-checks may be AI-reviewed; each result says which.</li>' +
    '<li>Research prototype — not for clinical use.</li></ul></section>';
  return h + '</div>';
}
