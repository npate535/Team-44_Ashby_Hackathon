// TCGA-BRCA external generalization (#/tcga, #/tcga/<id>).
// Story: a BACH-only trained model, applied unchanged to never-seen TCGA tissue — not a TCGA-tuned model.
// Specimens open in the same biopsy case view as BACH. Research prototype — not for clinical use.
import { classLabel, esc, num, pct, toView } from '../adapter.js';
import * as biopsy from './case_biopsy.js';
import { setChips } from './worklist.js';

const PREP = { frozen: 'Frozen section', ffpe: 'FFPE' };
const TIER_NAME = { 1: 'Frozen section', 2: 'Likely invasive', 3: 'Low confidence', 4: 'Unflagged' };

function get(obj, path) {
  let cur = obj;
  for (const k of Array.isArray(path) ? path : String(path).split('.')) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = cur[k];
  }
  return cur;
}

function isNum(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function ci(block) {
  if (isNum(block)) return { est: block, lo: null, hi: null, n: null, ng: null };
  if (!block || typeof block !== 'object') return null;
  const est = block.estimate ?? block.mean ?? block.value;
  if (!isNum(est)) return null;
  return {
    est,
    lo: isNum(block.lo) ? block.lo : null,
    hi: isNum(block.hi) ? block.hi : null,
    n: isNum(block.n) ? block.n : null,
    ng: isNum(block.n_groups) ? block.n_groups : null
  };
}

function pct1(v) {
  return isNum(v) ? `${(100 * v).toFixed(1)}%` : '—';
}

function ciPct(c) {
  if (!c) return 'not available';
  if (isNum(c.lo) && isNum(c.hi)) return `${pct1(c.est)} (95% CI ${pct1(c.lo)}–${pct1(c.hi)})`;
  return pct1(c.est);
}

function ciNum(c, digits = 3) {
  if (!c) return 'not available';
  const f = (v) => v.toFixed(digits);
  if (isNum(c.lo) && isNum(c.hi)) return `${f(c.est)} (95% CI ${f(c.lo)}–${f(c.hi)})`;
  return f(c.est);
}

function nLine(c, unit = 'cases') {
  if (!c || !isNum(c.n)) return '';
  const g = isNum(c.ng) ? `, ${c.ng} patients` : '';
  return `n = ${c.n} ${unit}${g}`;
}

function bachBlock(ST) {
  const M = get(ST, 'metrics.eval.metrics');
  if (!M) return null;
  const model = get(ST, 'metrics.eval.external_tcga.model') || get(M, 'models.production');
  const per = get(M, ['per_model', model]) || (get(M, 'models.production') === model ? M.production : null);
  if (!per) return null;
  return {
    model,
    n: M.n,
    nGroups: M.n_groups,
    split: M.split || 'test',
    accuracy: ci(per.accuracy),
    bal: ci(per.balanced_accuracy),
    invSens: ci(per.invasive_sensitivity),
    invAuroc: ci(per.invasive_vs_rest_auroc)
  };
}

function story(ST, idx) {
  const T = get(ST, 'metrics.eval.external_tcga');
  const bach = bachBlock(ST);
  const model = (T && T.model) || (idx && idx.model) || '—';

  const claim = `<header class="mx-head tcga-claim" data-testid="tcga-claim">
    <h1>Outside BACH</h1>
    <p class="mx-lede">Same probe, trained <b>only on BACH</b> — applied unchanged to never-seen
      TCGA-BRCA tissue (other hospitals, scanners, stains, and real frozen sections).
      No TCGA training, no threshold retuning on this set. Research prototype — not clinically validated.</p>
  </header>`;

  if (!T && !(idx && idx.available)) {
    return `${claim}
      <p class="mx-missing-inline" data-testid="tcga-missing">External results are not loaded yet.
        Run <code>python -m eval.external_tcga_prod --stage score</code>, then <code>python scripts/build_tcga_cases.py</code>.</p>`;
  }

  const p = (T && T.primary) || {};
  const ffpe = ci(p.ffpe_tumor_sensitivity);
  const frzSens = ci(p.frozen_tumor_sensitivity);
  const disc = p.frozen_tumor_vs_normal || {};
  const auroc = ci(disc.auroc);
  const frzSpec = ci(disc.specificity);
  const tier = ci(p.tumor_tier_1_2_rate);
  const gap = p.ffpe_vs_frozen_tumor_sensitivity_gap || {};
  const lim = ((T && T.limitations) || (idx && idx.limitations) || [])
    .map((l) => `<li>${esc(l)}</li>`)
    .join('');

  const bachCard = bach
    ? `<section class="tcga-panel" data-testid="tcga-bach">
        <h3>BACH held-out (same model)</h3>
        <p class="tcga-panel-note">In-domain test split · 4-class image labels · model <code>${esc(bach.model)}</code></p>
        <dl class="tcga-dl">
          <div><dt>Overall accuracy</dt><dd>${esc(ciPct(bach.accuracy))}</dd></div>
          <div><dt>Balanced accuracy</dt><dd>${esc(ciPct(bach.bal))}</dd></div>
          <div><dt>Invasive sensitivity</dt><dd>${esc(ciPct(bach.invSens))}</dd></div>
          <div><dt>Invasive vs rest AUROC</dt><dd>${esc(ciNum(bach.invAuroc))}</dd></div>
        </dl>
        <p class="mx-pop">${bach.n != null ? `n = ${bach.n} images` : ''}${bach.nGroups != null ? `, ${bach.nGroups} patients` : ''} · Source: <code>eval/metrics.json</code></p>
      </section>`
    : `<section class="tcga-panel" data-testid="tcga-bach-missing">
        <h3>BACH held-out</h3>
        <p class="mx-muted">BACH test metrics not loaded yet.</p>
      </section>`;

  const tcgaCard = `<section class="tcga-panel" data-testid="tcga-external">
      <h3>TCGA external (unchanged probe)</h3>
      <p class="tcga-panel-note">Out-of-domain · slide-level calls · not the same metric as BACH accuracy ·
        model <code>${esc(model)}</code>${T && T.quantized === false ? ' · full precision' : ''}</p>
      <dl class="tcga-dl">
        <div><dt>FFPE tumor sensitivity</dt><dd>${esc(ciPct(ffpe))}</dd></div>
        <div><dt>Frozen tumor sensitivity</dt><dd>${esc(ciPct(frzSens))}</dd></div>
        <div><dt>Frozen tumor vs normal AUROC</dt><dd>${esc(ciNum(auroc))}</dd></div>
        <div><dt>Frozen normal specificity</dt><dd>${esc(ciPct(frzSpec))}</dd></div>
      </dl>
      <p class="mx-pop">${nLine(ffpe, 'slides')} ·
        ${(T && T.n_patients) || (idx && idx.n_patients) || '—'} patients ·
        ${(T && T.n_crops) || (idx && idx.n) || '—'} crops ·
        Source: <code>eval/external_tcga.json</code></p>
      ${isNum(gap.estimate)
        ? `<p class="tcga-gap">FFPE vs frozen tumor sensitivity gap: ${pct1(gap.estimate)}
            (FFPE ${pct1(gap.ffpe)} · frozen ${pct1(gap.frozen)}).</p>`
        : ''}
      ${tier ? `<p class="tcga-gap">Tumor slides with any crop in tiers 1–2: ${esc(ciPct(tier))}.</p>` : ''}
    </section>`;

  const compareNote = `<p class="tcga-compare-note" role="note" data-testid="tcga-compare-note">
    <b>How to read this.</b> BACH “accuracy” is four-class image classification on the same dataset family the
    model saw in training. TCGA “sensitivity” is whether a never-seen tumor <em>slide</em> gets at least one
    crop called Invasive. A drop on external data is expected and is the finding — not a reason to retrain
    on TCGA for this demo.
  </p>`;

  const limits = lim
    ? `<section class="tcga-limits" data-testid="tcga-limits">
        <h3>Limitations</h3>
        <ul class="mx-notes">${lim}</ul>
        <p class="mx-pop">Pre-registered in <code>docs/EXTERNAL_TCGA_R2.md</code> before the run.</p>
      </section>`
    : '';

  return `${claim}
    <div class="tcga-compare" data-testid="tcga-compare">${bachCard}${tcgaCard}</div>
    ${compareNote}
    ${limits}`;
}

function itemToCase(item, model) {
  const classes = ['Normal', 'Benign', 'InSitu', 'Invasive'];
  const probsMap = item.probs || {};
  const probs = classes.map((c) => Number(probsMap[c] ?? 0));
  const tier = Number(item.tier) || 4;
  const reasons = item.tier_reasons || [];
  const prep = item.prep === 'frozen' ? 'frozen' : 'ffpe';
  return {
    schema_version: '1.0',
    case_id: item.case_id,
    lane: 'biopsy',
    simulated: false,
    synthetic: false,
    cohort: 'tcga',
    specimen_question: `TCGA-BRCA external field · ${PREP[prep] || prep}`,
    arrived_offset_s: 0,
    image: {
      display: item.image || `tcga/${item.case_id}.jpg`,
      width: 2048,
      height: 1536,
      tile_size: 512,
      grid: [3, 4]
    },
    quality: { qc_tiles_kept: 12, ood_percentile: 0.5, reliability: 0.5, reliability_flag: false },
    prediction: {
      model: model || item.model || null,
      classes,
      probs,
      risk_score: Number(item.p_invasive) || 0,
      conformal_alpha: 0.1,
      conformal_set: item.conformal_set || [],
      attribution: null
    },
    concepts: [],
    tiles: [],
    similar_cases: [],
    contrast_case: null,
    agreement: {},
    margin: null,
    cells: { status: 'disabled' },
    flag: { tier, reasons },
    findings: [],
    summary: null,
    agent: { status: 'none', orders: [], blocked: [], n_orders: 0 },
    runtime: { tier, reasons, allowed_actions: [], arrived_sim_s: 0, done: false },
    tcga_meta: { patient_id: item.patient_id, prep }
  };
}

function row(item, ST, model) {
  const c = itemToCase(item, model);
  const v = toView(c, Date.now(), ST.cfg, null);
  const pInv = pct(v.risk === null ? null : v.risk / 100);
  const prep = PREP[item.prep] || item.prep || '—';
  return (
    `<tr class="row tr-${num(v.tier)}" tabindex="0" data-testid="tcga-row" data-case-id="${esc(v.id)}" data-tier="${num(v.tier)}">` +
    `<td class="c-tier"><span class="tierno t${num(v.tier)}" title="Tier ${num(v.tier)}">${num(v.tier)}</span></td>` +
    `<td class="c-case"><div class="wl-cc"><b class="cid">${esc(v.id)}</b>` +
    `<span class="lane-inline">${esc(item.patient_id || '')}</span></div></td>` +
    `<td class="c-lane"><span class="lane bx">${esc(prep)}</span></td>` +
    `<td class="c-sugg"><span class="sugg${v.top === 'Invasive' ? ' inv' : ''}">${esc(classLabel(v.top, ST.cfg) || '—')}</span></td>` +
    `<td class="c-p"><span class="pnum${v.risk >= 50 ? ' hi' : ''}">${esc(pInv)}</span></td>` +
    `<td class="c-set"><span class="chips">${setChips(v)}</span></td>` +
    `<td class="c-flags"><span class="sub">${esc((v.reasons || []).map((r) => r.label).join(' · ') || '—')}</span></td>` +
    '</tr>'
  );
}

function specimenTable(ST, idx) {
  if (!idx || !idx.available || !(idx.items || []).length) {
    return `<section class="tcga-specimens" data-testid="tcga-specimens-empty">
      <h2>Browse external specimens</h2>
      <p class="mx-muted">Specimen images are not on disk yet. After the eval, run
        <code>python scripts/build_tcga_cases.py</code>.</p>
    </section>`;
  }
  const items = [...idx.items].sort(
    (a, b) => a.tier - b.tier || b.p_invasive - a.p_invasive || a.case_id.localeCompare(b.case_id)
  );
  const byTier = { 1: 0, 2: 0, 3: 0, 4: 0 };
  items.forEach((it) => {
    byTier[it.tier] = (byTier[it.tier] || 0) + 1;
  });
  let group = null;
  let body = '';
  items.forEach((it) => {
    const g = it.tier;
    if (g !== group) {
      group = g;
      body += `<tr class="grp g-${g}" data-testid="tcga-tier-group"><th colspan="7" scope="colgroup">
        <span class="g-name">Tier ${g} · ${esc(TIER_NAME[g] || '')}</span>
        <span class="g-n">${byTier[g] || 0}</span></th></tr>`;
    }
    body += row(it, ST, idx.model);
  });
  return `<section class="wl tcga-wl tcga-specimens" data-testid="tcga-worklist">
    <h2>Browse external specimens</h2>
    <p class="tcga-spec-lede">Open any field the same way as a BACH biopsy — image, suggestion, and confidence.
      These are TCGA crops scored by the BACH-trained probe; ground-truth labels are not shown.</p>
    <div class="tblw wl-table-wrap">
      <table class="wl-table" data-testid="tcga-table" aria-label="TCGA external specimens">
        <thead><tr>
          <th class="c-tier" scope="col"><span class="sr-only">Tier</span></th>
          <th class="c-case" scope="col">Specimen</th>
          <th class="c-lane" scope="col">Prep</th>
          <th class="c-sugg" scope="col">Suggested</th>
          <th class="c-p" scope="col">P(Invasive)</th>
          <th class="c-set" scope="col">Confidence set</th>
          <th class="c-flags" scope="col">Flags</th>
        </tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  </section>`;
}

function worklist(ST) {
  const idx = ST.tcga;
  if (!idx && !ST.metrics) {
    return `<div class="mx tcga-page" data-testid="tcga-loading"><header class="mx-head"><h1>Outside BACH</h1>
      <p class="mx-lede">Loading external generalization results…</p></header></div>`;
  }
  return `<div class="mx tcga-page" data-testid="tcga-page">
    ${story(ST, idx)}
    ${specimenTable(ST, idx)}
  </div>`;
}

function detail(ST, ctx) {
  const id = ST.caseId;
  if (ST.notFound === id) {
    return `<a class="back" href="#/tcga" data-testid="back">Back to TCGA specimens</a>
      <div class="card" data-testid="case-not-found"><h2>Specimen not found</h2>
      <p>${esc(id)} is not in the TCGA external set.</p></div>`;
  }
  if (!ST.cases[id]) {
    return '<div class="card" data-testid="state-loading"><h2>Loading specimen…</h2></div>';
  }
  return biopsy.render(ST, ctx, {
    backHref: '#/tcga',
    backLabel: 'Back to TCGA specimens',
    hideAgent: true
  });
}

export function render(ST, ctx) {
  if (ST.caseId) return detail(ST, ctx);
  return worklist(ST);
}
