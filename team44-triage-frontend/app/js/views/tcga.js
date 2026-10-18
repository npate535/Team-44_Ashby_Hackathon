// TCGA-BRCA external specimens (#/tcga, #/tcga/<id>). Same biopsy case view as BACH — no arrival replay.
// Index: GET /api/tcga. Detail: GET /api/tcga/cases/<id>. Research prototype — not for clinical use.
import { classLabel, esc, num, pct, toView } from '../adapter.js';
import * as biopsy from './case_biopsy.js';
import { setChips } from './worklist.js';

const PREP = { frozen: 'Frozen section', ffpe: 'FFPE' };
const TIER_NAME = { 1: 'Frozen section', 2: 'Likely invasive', 3: 'Low confidence', 4: 'Unflagged' };

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
      model: model || item.model || 'uni2h_ft',
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

function worklist(ST) {
  const idx = ST.tcga;
  if (!idx) {
    return `<div class="mx" data-testid="tcga-loading"><header class="mx-head"><h1>TCGA external</h1>
      <p class="mx-lede">Loading external specimens…</p></header></div>`;
  }
  if (!idx.available || !(idx.items || []).length) {
    return `<div class="mx" data-testid="tcga-missing"><header class="mx-head"><h1>TCGA external</h1>
      <p class="mx-lede">Sepia’s BACH-trained probe on never-seen TCGA-BRCA tissue.</p></header>
      <p class="mx-missing-inline">No TCGA specimens on disk yet. Run <code>python -m eval.external_tcga</code>, then
        <code>python scripts/build_tcga_cases.py</code>.</p></div>`;
  }
  const items = [...idx.items].sort((a, b) => (a.tier - b.tier) || (b.p_invasive - a.p_invasive) || a.case_id.localeCompare(b.case_id));
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
  return `<section class="wl tcga-wl" data-testid="tcga-worklist">
    <header class="mx-head" style="margin-bottom:1rem">
      <h1>TCGA external</h1>
      <p class="mx-lede">${esc(String(idx.n))} fields from ${esc(String(idx.n_patients ?? '—'))} TCGA-BRCA patients
        · model <code>${esc(idx.model || 'uni2h_ft')}</code>
        · full precision${idx.quantized === false ? '' : ''}. Open a specimen the same way as a BACH biopsy.</p>
    </header>
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
