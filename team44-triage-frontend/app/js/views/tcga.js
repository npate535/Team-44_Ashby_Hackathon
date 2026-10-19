// TCGA-BRCA external generalization (#/tcga, #/tcga/<id>).
// Story: a BACH-only trained model, applied unchanged to never-seen TCGA tissue — not a TCGA-tuned model.
// Specimens open in the same biopsy case view as BACH. Research prototype — not for clinical use.
import { classLabel, esc, num, pct, toView } from '../adapter.js';
import { atBoundary, boundaryInterval, boundaryText } from '../ci.js';
import * as biopsy from './case_biopsy.js';
import { setChips } from './worklist.js';

const PREP = { frozen: 'Frozen section', ffpe: 'FFPE' };
const TIER_NAME = { 1: 'Frozen section', 2: 'Likely invasive', 3: 'Low confidence', 4: 'Unflagged' };
const GROUP_NAME = { ffpe_tumor: 'FFPE tumor', frozen_tumor: 'frozen tumor', frozen_normal: 'frozen normal' };
// specimen ids are <patient>_<prep>_<sample type>_<field>, e.g. TCGA-A7-A4SD_ffpe_tumor_00
const CROP_ID = /^(.+?)_((?:ffpe|frozen)_[a-z]+_\d+)$/;

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

const p1 = (v) => (100 * v).toFixed(1);

function pct1(v) {
  return isNum(v) ? `${p1(v)}%` : '—';
}

// "22 of 23 slides" when the estimate is an exact share of its n; otherwise ''.
function kOfN(c, unit) {
  if (!c || !unit || !Number.isInteger(c.n) || c.n <= 0) return '';
  const k = c.est * c.n;
  return Math.abs(k - Math.round(k)) < 1e-6 ? `${Math.round(k)} of ${c.n} ${unit}` : '';
}

/**
 * {est, sub} for a share. A bootstrap that collapsed at 0 or 1 is replaced by the exact interval over the true
 * denominator `trueN` ("17 of 17 slides (exact 95% CI 80.5–100%)"); other values keep their bootstrap interval.
 */
function shareStat(block, { unit = '', trueN = null } = {}) {
  const c = ci(block);
  if (!c) return { est: 'not available', sub: '' };
  if (atBoundary(block)) {
    const iv = Number.isInteger(trueN) ? boundaryInterval(block, trueN) : null;
    const text = iv ? boundaryText(iv) : '';
    return {
      est: pct1(c.est),
      sub: text
        ? unit ? text.replace(/^(\d+ of \d+)/, `$1 ${unit}`) : text
        : 'no interval: every resample gave the same value'
    };
  }
  const range = isNum(c.lo) && isNum(c.hi) ? `95% CI ${p1(c.lo)}–${p1(c.hi)}%` : '';
  return { est: pct1(c.est), sub: [kOfN(c, unit), range].filter(Boolean).join(' · ') };
}

function numStat(block, digits = 3) {
  const c = ci(block);
  if (!c) return { est: 'not available', sub: '' };
  const f = (v) => v.toFixed(digits);
  return { est: f(c.est), sub: isNum(c.lo) && isNum(c.hi) ? `95% CI ${f(c.lo)}–${f(c.hi)}` : '' };
}

// Keep a range ("75.6–93.9%") on one line; escaped text in, HTML out.
function keepRanges(html) {
  return html.replace(/(\S+–\S+)/g, '<span class="tcga-nw">$1</span>');
}

// Frozen normal slides (ids <patient>_frozen_normal_<field>) with at least one field in tier 1 or 2.
function normalTierCount(idx) {
  const slides = new Map();
  for (const it of (idx && idx.items) || []) {
    const m = /^(.+_frozen_normal)_\d+$/.exec(String(it.case_id || ''));
    if (m) slides.set(m[1], Math.min(slides.get(m[1]) ?? 9, Number(it.tier) || 4));
  }
  if (!slides.size) return null;
  const k = [...slides.values()].filter((t) => t <= 2).length;
  return { est: `${k} of ${slides.size}`, sub: `${pct1(k / slides.size)} of frozen normal slides; counted from the fields below` };
}

function statRow(label, { est, sub }, testid) {
  return (
    `<div${testid ? ` data-testid="${esc(testid)}"` : ''}><dt>${esc(label)}</dt>` +
    `<dd class="tcga-est">${esc(est)}</dd>` +
    (sub ? `<dd class="tcga-ci">${keepRanges(esc(sub))}</dd>` : '') +
    '</div>'
  );
}

// True denominator of BACH invasive sensitivity: the Invasive row of the production confusion matrix
// (metrics.json `invasive_sensitivity.n` counts every test image). Falls back to external_tcga.json's
// patient-bootstrap reference for the same model. {k, n, nGroups} or null.
function bachInvasiveCount(ST, model) {
  const M = get(ST, 'metrics.eval.metrics');
  const aggressive = get(ST, 'cfg.aggressive_class') || get(M, 'aggressive_class') || 'Invasive';
  const ref = get(ST, 'metrics.eval.external_tcga.bach_reference');
  const refBlock = ref && ref.model === model ? ref.invasive_sensitivity_patient_bootstrap : null;
  const nGroups = refBlock && Number.isInteger(refBlock.n_groups) ? refBlock.n_groups : null;
  const conf = get(M, 'models.production') === model ? get(M, 'production.confusion') : get(M, ['per_model', model, 'confusion']);
  if (conf && Array.isArray(conf.classes) && Array.isArray(conf.matrix)) {
    const i = conf.classes.indexOf(aggressive);
    const rowI = i >= 0 ? conf.matrix[i] : null;
    if (Array.isArray(rowI) && rowI.every(Number.isInteger)) {
      return { k: rowI[i], n: rowI.reduce((a, b) => a + b, 0), nGroups };
    }
  }
  if (refBlock && Number.isInteger(refBlock.n) && isNum(refBlock.estimate)) {
    return { k: Math.round(refBlock.estimate * refBlock.n), n: refBlock.n, nGroups };
  }
  return null;
}

function bachBlock(ST) {
  const M = get(ST, 'metrics.eval.metrics');
  if (!M) return null;
  const model = get(ST, 'metrics.eval.external_tcga.model') || get(M, 'models.production');
  // the production block carries the headline intervals quoted everywhere else; per_model is the comparison table's
  const per = (get(M, 'models.production') === model ? M.production : null) || get(M, ['per_model', model]);
  if (!per) return null;
  return {
    model,
    n: M.n,
    nGroups: M.n_groups,
    split: M.split || 'test',
    accuracy: per.accuracy,
    bal: per.balanced_accuracy,
    invSens: per.invasive_sensitivity,
    invCount: bachInvasiveCount(ST, model),
    invAuroc: per.invasive_vs_rest_auroc
  };
}

// AUROC at 1 (or 0) with a collapsed bootstrap: say what it means instead of printing "1.000–1.000".
function aurocStat(block, nPos, nNeg) {
  const c = ci(block);
  if (!c || !atBoundary(block)) return numStat(block);
  const counted = Number.isInteger(nPos) && Number.isInteger(nNeg);
  const tumor = counted ? `all ${nPos} frozen tumor slides` : 'every frozen tumor slide';
  const normal = counted ? `all ${nNeg} frozen normal slides` : 'every frozen normal slide';
  const [hi, lo] = c.est === 1 ? [tumor, normal] : [normal, tumor];
  return { est: c.est.toFixed(3), sub: `${hi[0].toUpperCase()}${hi.slice(1)} scored above ${lo}.` };
}

function story(ST, idx) {
  const T = get(ST, 'metrics.eval.external_tcga');
  const bach = bachBlock(ST);
  const model = (T && T.model) || (idx && idx.model) || '—';

  const claim = `<header class="mx-head tcga-claim" data-testid="tcga-claim">
    <h2 class="tcga-title">Outside BACH</h2>
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
  const disc = p.frozen_tumor_vs_normal || {};
  const nTumor = get(disc, 'sensitivity.n');
  const nNormal = get(disc, 'specificity.n');
  const gap = p.ffpe_vs_frozen_tumor_sensitivity_gap || {};
  const ffpe = ci(p.ffpe_tumor_sensitivity);
  const frz = ci(p.frozen_tumor_sensitivity);
  const lim = ((T && T.limitations) || (idx && idx.limitations) || [])
    .map((l) => `<li>${esc(l)}</li>`)
    .join('');

  let bachCard = `<section class="tcga-panel" data-testid="tcga-bach-missing">
        <h3>BACH held-out</h3>
        <p class="mx-muted">BACH test metrics not loaded yet.</p>
      </section>`;
  if (bach) {
    const inv = bach.invCount && bach.invCount.k === bach.invCount.n ? bach.invCount : null;
    const invStat = shareStat(bach.invSens, { unit: 'invasive images', trueN: inv ? inv.n : null });
    const exactNote =
      atBoundary(bach.invSens) && inv && Number.isInteger(inv.nGroups) && inv.nGroups < inv.n
        ? `<p class="tcga-foot">The exact interval counts images as independent; the ${inv.n} invasive images come from
            ${inv.nGroups} patients, so it is optimistic.</p>`
        : '';
    const cohort = [
      isNum(bach.n) ? `${bach.n} images` : '',
      isNum(bach.nGroups) ? `${bach.nGroups} patients` : ''
    ].filter(Boolean).join(' from ');
    bachCard = `<section class="tcga-panel" data-testid="tcga-bach">
        <h3>BACH held-out (same model)</h3>
        <p class="tcga-panel-note">In-domain ${esc(bach.split)} split · 4-class image labels · model <code>${esc(bach.model)}</code></p>
        <dl class="tcga-dl">
          ${statRow('Overall accuracy', shareStat(bach.accuracy, { unit: 'images' }))}
          ${statRow('Balanced accuracy', shareStat(bach.bal))}
          ${statRow('Invasive sensitivity', invStat, 'tcga-bach-inv-sens')}
          ${statRow('Invasive vs rest AUROC', numStat(bach.invAuroc))}
        </dl>
        ${exactNote}
        <p class="mx-pop tcga-src">${cohort ? `${esc(cohort)} · ` : ''}Source: <code>eval/metrics.json</code></p>
      </section>`;
  }

  const groups = (T && T.groups) || {};
  const groupText = Object.keys(GROUP_NAME)
    .filter((g) => Number.isInteger(groups[g]))
    .map((g) => `${groups[g]} ${GROUP_NAME[g]}`)
    .join(', ');
  const nPatients = (T && T.n_patients) || (idx && idx.n_patients);
  const nSlides = (T && T.n_slides) || (idx && idx.n_slides);
  const nFields = (T && T.n_crops) || (idx && idx.n);
  const cohort = [
    nPatients ? `${nPatients} patients` : '',
    nSlides ? `${nSlides} slides${groupText ? ` (${groupText})` : ''}` : '',
    nFields ? `${nFields} fields` : ''
  ].filter(Boolean).join(' · ');
  const overlap =
    ffpe && frz && [ffpe.lo, ffpe.hi, frz.lo, frz.hi].every(isNum) && ffpe.lo <= frz.hi && frz.lo <= ffpe.hi;
  const gapNote = isNum(gap.estimate)
    ? `<p class="tcga-foot" data-testid="tcga-gap">FFPE minus frozen tumor sensitivity: ${esc(p1(gap.estimate))} percentage points
        (${esc(pct1(gap.ffpe))} vs ${esc(pct1(gap.frozen))}); a point estimate only${overlap ? ', and the two 95% intervals overlap' : ''}.</p>`
    : '';

  // The cost side of the tier rate: normal slides that would still be escalated (counted from the specimen list).
  const normalTier = normalTierCount(idx);

  const tcgaCard = `<section class="tcga-panel" data-testid="tcga-external">
      <h3>TCGA external (unchanged probe)</h3>
      <p class="tcga-panel-note">Out-of-domain · slide-level calls · not the same metric as BACH accuracy ·
        model <code>${esc(model)}</code>${T && T.quantized === false ? ' · full precision' : ''}</p>
      <dl class="tcga-dl">
        ${statRow('FFPE tumor sensitivity', shareStat(p.ffpe_tumor_sensitivity, { unit: 'slides' }))}
        ${statRow('Frozen tumor sensitivity', shareStat(p.frozen_tumor_sensitivity, { unit: 'slides' }))}
        ${statRow('Frozen tumor vs normal AUROC', aurocStat(disc.auroc, nTumor, nNormal), 'tcga-auroc')}
        ${statRow('Frozen normal specificity', shareStat(disc.specificity, { unit: 'slides', trueN: nNormal }), 'tcga-spec')}
        ${p.tumor_tier_1_2_rate ? statRow('Tumor slides with a field in tiers 1–2', shareStat(p.tumor_tier_1_2_rate, { unit: 'slides' })) : ''}
        ${normalTier ? statRow('Frozen normal slides with a field in tiers 1–2', normalTier, 'tcga-normal-tier') : ''}
      </dl>
      ${gapNote}
      <p class="mx-pop tcga-src" data-testid="tcga-cohort">${cohort ? `${esc(cohort)} · ` : ''}Source: <code>eval/external_tcga.json</code></p>
    </section>`;

  const compareNote = `<p class="tcga-compare-note" role="note" data-testid="tcga-compare-note">
    <b>How to read this.</b> BACH “accuracy” is four-class image classification on the same dataset family the
    model saw in training. TCGA “sensitivity” is whether a never-seen tumor <em>slide</em> gets at least one
    field called Invasive; specificity is whether a normal slide gets none. Lower values on external tissue are
    expected, and with this small cohort the TCGA intervals are wide.
  </p>`;

  const limits = lim
    ? `<section class="tcga-limits" data-testid="tcga-limits">
        <h3>Limitations</h3>
        <ul class="mx-notes">${lim}</ul>
        <p class="mx-pop tcga-src">Pre-registered in <code>docs/EXTERNAL_TCGA_R2.md</code> before the run.</p>
      </section>`
    : '';

  return `${claim}
    <div class="tcga-compare" data-testid="tcga-compare">${bachCard}${tcgaCard}</div>
    ${compareNote}
    ${limits}`;
}

// Worklist-row view of an index item. Not computed for TCGA (CONCH/MUSK reliability, OOD): null, never a placeholder.
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
    specimen_question: `${PREP[prep] || prep} slide`,
    arrived_offset_s: 0,
    image: {
      display: item.image || `tcga/${item.case_id}.jpg`,
      width: 2048,
      height: 1536,
      tile_size: 512,
      grid: [3, 4]
    },
    quality: {
      qc_tiles_kept: Number.isInteger(item.qc_tiles_kept) ? item.qc_tiles_kept : null,
      ood_percentile: null,
      reliability: null,
      reliability_flag: false
    },
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
  // The full id, split so it never truncates: patient barcode, then <prep>_<sample type>_<field>.
  const m = CROP_ID.exec(String(v.id || ''));
  const idHtml = m
    ? `<b class="cid">${esc(m[1])}</b><span class="tcga-fld">_${esc(m[2])}</span>`
    : `<b class="cid">${esc(v.id)}</b>`;
  return (
    `<tr class="row tr-${num(v.tier)}" tabindex="0" data-testid="tcga-row" data-case-id="${esc(v.id)}" data-tier="${num(v.tier)}">` +
    `<td class="c-tier"><span class="tierno t${num(v.tier)}" title="Tier ${num(v.tier)}">${num(v.tier)}</span></td>` +
    `<td class="c-case"><div class="wl-cc tcga-cc" title="${esc(v.id)}">${idHtml}</div></td>` +
    `<td class="c-lane"><span class="lane bx">${esc(prep)}</span></td>` +
    `<td class="c-sugg"><span class="sugg${v.top === 'Invasive' ? ' inv' : ''}">${esc(classLabel(v.top, ST.cfg) || '—')}</span></td>` +
    `<td class="c-p"><span class="pnum${v.risk >= 50 ? ' hi' : ''}">${esc(pInv)}</span></td>` +
    `<td class="c-set"><span class="chips">${setChips(v)}</span></td>` +
    `<td class="c-flags"><span class="sub">${esc((v.reasons || []).map((r) => r.label).join(' · ') || '—')}</span></td>` +
    '</tr>'
  );
}

function specimenTable(ST, idx) {
  if (!idx) {
    // the index request is still in flight: reserve the table's space instead of showing a missing-data message
    return `<section class="tcga-specimens tcga-specimens-loading" data-testid="tcga-specimens-loading" aria-busy="true">
      <h2>Browse external specimens</h2>
      <p class="mx-muted">Loading specimens…</p>
    </section>`;
  }
  if (!idx.available || !(idx.items || []).length) {
    return `<section class="tcga-specimens" data-testid="tcga-specimens-empty">
      <h2>Browse external specimens</h2>
      <p class="mx-muted">No TCGA specimens are available in this workspace. They are built by
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
      Each field is scored by the BACH-trained probe. Its name is the patient barcode, preparation, the TCGA sample
      type of its slide (tumor or normal) and the field number; the model never sees the name, and there are no
      region-level labels.</p>
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
    return `<div class="mx tcga-page" data-testid="tcga-loading"><header class="mx-head"><h2 class="tcga-title">Outside BACH</h2>
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
