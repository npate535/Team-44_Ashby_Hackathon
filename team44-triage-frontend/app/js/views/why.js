// Evidence panels (plan §8.1): Why (grouped concept associations), How sure (calibrated probabilities, conformal set,
// agreement checklist), Compared to (references + contrast case), Summary with finding references.
// Wording rule (plan §6.4): "features associated with", never causes, never a Nottingham grade; the model suggests.
import { classLabel, coverageStatus, esc, exactProb, FINDING_ID, humanize, num, ordinalPct, pct, signed, thumbUrl } from '../adapter.js';

/** A model probability as a percentage that never reads 100% or 0% (exact value goes in tooltips). */
export const fmtPct = (p) => (!isNum(p) ? '—' : Number(p) >= 0.995 ? '>99%' : Number(p) < 0.005 ? '<1%' : Math.round(Number(p) * 100) + '%');
export const aggProb = (v, cfg) => {
  const agg = (cfg && cfg.aggressive_class) || 'Invasive';
  const row = v.probs.find((x) => x.cls === agg);
  return row ? row.p : v.risk === null ? null : v.risk / 100;
};
const NEGLIGIBLE = 0.05; // |contribution| below this reads as "barely moves" (settings findings.min_abs_contribution)
const isNum = (x) => x !== null && x !== undefined && Number.isFinite(Number(x));
const sz = (x) => (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(2);
const cap = (t) => String(t).replace(/^./, (ch) => ch.toUpperCase());

// ---- Why ---------------------------------------------------------------------------------------------------------------

/**
 * Level of a feature in this case relative to the training cases, in words.
 * - text / concept-vector criteria: ONLY from `value_std` (the case feature standardized against training cases, the
 *   concept model's actual input). Without it no level word is shown, only the push.
 * - measured cell features: reference percentile against benign/normal reference cases.
 * The raw `value` (mean tile z of the 3 strongest regions, a different scale) goes to the tooltip only.
 */
export function levelOf(r, cells) {
  if (r.measured) {
    const f = cells && cells.features ? cells.features[r.id] : null;
    const p = f ? f.ref_percentile : null;
    if (isNum(p)) {
      const q = Math.round(p * 100);
      const word = p >= 0.9 ? 'High' : p >= 0.6 ? 'Above typical' : p > 0.4 ? 'Typical' : p > 0.1 ? 'Below typical' : 'Low';
      return { word, cls: p >= 0.6 ? 'hi' : p > 0.4 ? 'mid' : 'lo', short: `${ordinalPct(p)} percentile`, long: `higher than ${q}% of benign/normal reference cases`, tip: '', pct: Number(p) };
    }
    return { word: '', cls: 'none', short: '', long: '', tip: 'no reference percentile available' };
  }
  const tip = [
    isNum(r.value) ? `Mean score of its 3 strongest regions (tile z): ${sz(Number(r.value))}` : '',
    isNum(r.fracStd) ? `Share of regions above threshold: ${sz(Number(r.fracStd))} SD vs training cases` : '',
  ].filter(Boolean).join(' · ');
  if (isNum(r.valueStd)) {
    const v = Number(r.valueStd);
    const word = v >= 0.5 ? 'Above typical' : v <= -0.5 ? 'Below typical' : 'Typical';
    return { word, cls: v >= 0.5 ? 'hi' : v <= -0.5 ? 'lo' : 'mid', short: `${sz(v)} SD vs training cases`, long: `${sz(v)} SD from the average training case`, tip, std: v };
  }
  return { word: '', cls: 'none', short: '', long: '', tip };
}

/** AI spot check of a concept's top regions (eval/spotcheck.json summary.concept[id] = {n, acceptable}). */
function spotOf(metrics, id) {
  const sc = metrics && metrics.eval && metrics.eval.spotcheck;
  const e = sc && sc.summary && sc.summary.concept ? sc.summary.concept[id] : null;
  if (!e || !isNum(e.n) || !isNum(e.acceptable) || Number(e.n) <= 0) return null;
  const low = Number(e.acceptable) / Number(e.n) <= 0.25;
  return { n: Number(e.n), ok: Number(e.acceptable), low };
}

/** Tiny SD track (−3…+3, typical band ±0.5) so levels scan down the column; percentiles use 0…100. */
function sdTrack(lvl) {
  let x;
  if (isNum(lvl.std)) x = (Math.max(-3, Math.min(3, lvl.std)) + 3) / 6;
  else if (isNum(lvl.pct)) x = Math.max(0, Math.min(1, lvl.pct));
  else return '';
  return `<i class="sd-track${isNum(lvl.pct) ? ' pct' : ''}" aria-hidden="true"><i class="sd-band"></i><i class="sd-dot" style="left:${(x * 100).toFixed(1)}%"></i></i>`;
}

function pushOf(c) {
  if (!isNum(c) || Math.abs(c) < NEGLIGIBLE) return { cls: 'none', word: 'barely moves the invasive score', short: 'negligible' };
  return c > 0
    ? { cls: 'up', word: 'pushes toward invasive carcinoma', short: 'toward invasive' }
    : { cls: 'down', word: 'pushes away from invasive carcinoma', short: 'away from invasive' };
}

function groupRows(v, cfg) {
  const groups = (cfg && cfg.labels && cfg.labels.concept_groups) || {};
  const placed = new Set();
  const out = [];
  for (const [gid, g] of Object.entries(groups)) {
    const rows = v.why.filter((r) => !r.measured && (g.ids || []).includes(r.id));
    rows.forEach((r) => placed.add(r.id));
    if (rows.length) out.push({ id: gid, label: g.label, rows });
  }
  const other = v.why.filter((r) => !r.measured && !placed.has(r.id));
  if (other.length) out.push({ id: 'other', label: 'Other criteria', rows: other });
  const cells = v.why.filter((r) => r.measured);
  if (cells.length) out.push({ id: 'cells', label: 'Measured cell features', rows: cells });
  for (const g of out) g.rows.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return out;
}

function pushBar(c, maxAbs) {
  const x = isNum(c) ? Number(c) / maxAbs : 0;
  const w = Math.max(Math.abs(x), isNum(c) && c !== 0 ? 0.015 : 0);
  return (
    '<svg class="why-bar" viewBox="-1 0 2 1" preserveAspectRatio="none" aria-hidden="true">' +
    '<rect class="wb-track" x="-1" y="0.38" width="2" height="0.24"/>' +
    `<rect class="wb-fill ${x >= 0 ? 'up' : 'down'}" x="${x >= 0 ? 0 : -w}" y="0.12" width="${w}" height="0.76"/>` +
    '<line class="wb-zero" x1="0" x2="0" y1="0" y2="1"/></svg>'
  );
}

function whyRow(r, hm, maxAbs, cells, metrics) {
  const lvl = levelOf(r, cells);
  const push = pushOf(r.contribution);
  const context = r.direction === 'context';
  const cc = r.captionConsistent
    ? '<span class="cc" data-testid="caption-consistent" title="Concept vector agreed with mined captions on held-out data (caption co-mention, not tissue truth)">✓ caption-consistent</span>'
    : r.measured
      ? ''
      : '<span class="cc no" title="No concept vector passed the held-out caption check; text prompts only">text prompts only</span>';
  const spot = r.measured ? null : spotOf(metrics, r.id);
  const spotBadge = spot
    ? `<span class="spot${spot.low ? ' low' : ''}" data-testid="spot-${esc(r.id)}" title="AI-reviewed spot check (Claude), not pathologist-validated: ${spot.ok} of ${spot.n} of this feature's top regions looked like the feature.${spot.low ? ' Top regions often did not show this feature, so discount this row.' : ''}">AI spot check ${spot.ok}/${spot.n}${spot.low ? ' · top regions often did not show this feature' : ''}</span>`
    : '';
  const spread = isNum(r.spread) ? ` Prompt-ensemble spread ${Number(r.spread).toFixed(2)}.` : '';
  const lead = r.measured
    ? lvl.long
      ? `Measured: ${lvl.long}.`
      : 'Measured.'
    : lvl.word
      ? `${lvl.word} for this case (${lvl.short}).`
      : '';
  const how = `In the concept model it ${push.word.replace(' carcinoma', '')}.`;
  const sentence = `${lead}${lead ? ' ' : ''}${how}${context ? ' Context feature, not ranked as a risk criterion.' : ''}`;
  const tiles = (r.tiles || []).filter((t) => Number.isInteger(t));
  // the level and push columns already say both halves of `sentence`; show only what they do not
  const visibleNote = [r.measured && lvl.long ? `Measured: ${lvl.long}.` : '', context ? 'Context feature, not ranked as a risk criterion.' : '']
    .filter(Boolean)
    .join(' ');
  const tipText = [lvl.tip, spread.trim()].filter(Boolean).join(' · ');
  return (
    `<button type="button" class="why-row${hm.why === r.id ? ' on' : ''}" data-testid="why-bar-${esc(r.id)}" data-why="${esc(r.id)}" data-tiles="${esc(tiles.join(','))}"${tipText ? ` title="${esc(tipText)}"` : ''} ` +
    `aria-pressed="${hm.why === r.id}" aria-label="${esc(r.label)}. ${esc(sentence)} Contribution ${esc(isNum(r.contribution) ? signed(r.contribution) : 'not available')}.${tipText ? ' ' + esc(tipText) + '.' : ''}${spot ? ` AI spot check ${spot.ok} of ${spot.n}, not pathologist-validated.` : ''}${tiles.length ? ` Highlights regions ${esc(tiles.join(', '))}.` : ''}">` +
    `<span class="wr-head"><span class="lbl">${esc(r.label)}</span><span class="wr-badges">${cc}${spotBadge}${context ? '<span class="src ctx" title="Plan §6.4: context, not ranked">context</span>' : ''}</span></span>` +
    (lvl.word
      ? `<span class="wr-level ${lvl.cls}"><b>${esc(lvl.word)}</b>${lvl.short ? `<span>${esc(lvl.short)}</span>` : ''}${sdTrack(lvl)}</span>`
      : '<span class="wr-level none"><span>—</span></span>') +
    `<span class="wr-bar">${pushBar(r.contribution, maxAbs)}</span>` +
    `<span class="v ${push.cls}"><b>${isNum(r.contribution) ? signed(r.contribution) : '—'}</b><span>${esc(push.short)}</span></span>` +
    (visibleNote || tiles.length
      ? `<span class="wr-say">${esc(visibleNote)}${tiles.length ? `${visibleNote ? ' ' : ''}<span class="wr-tiles">Strongest regions ${esc(tiles.join(', '))}</span>` : ''}</span>`
      : '') +
    '</button>'
  );
}

export function why(v, hm, cfg, metrics) {
  const rows = v.why;
  if (!rows.length) return '<div class="note">No criterion contributions for this case.</div>';
  const maxAbs = Math.max(1e-9, ...rows.map((r) => Math.abs(num(r.contribution))));
  const conceptPrediction = /^(cb_|concept_bottleneck)/.test(v.model || '');
  const explModel = metrics && metrics.eval && metrics.eval.metrics && metrics.eval.metrics.models ? metrics.eval.metrics.models.b : null;
  const provenance = conceptPrediction
    ? `<p class="explanation-source">These contributions come from the same concept model that made the suggestion (<b>${esc(humanize(v.model))}</b>).</p>`
    : `<p class="explanation-source">Supporting associations from the concept model${explModel ? ` (<b>${esc(explModel)}</b>)` : ''}. The suggested category comes from the production model, <b>${esc(v.model || 'unavailable')}</b>. The concept model's own category is not recorded in this case file, so read these rows as context for the suggestion, not as its cause.</p>`;
  const net = rows.reduce((a, r) => a + (isNum(r.contribution) ? Number(r.contribution) : 0), 0);
  const ups = [...rows].filter((r) => num(r.contribution) >= NEGLIGIBLE).sort((a, b) => b.contribution - a.contribution).slice(0, 2);
  const downs = [...rows].filter((r) => num(r.contribution) <= -NEGLIGIBLE).sort((a, b) => a.contribution - b.contribution).slice(0, 2);
  const list = (xs) => xs.map((r) => `<b>${esc(r.label)}</b> (${signed(r.contribution)})`).join(', ');
  const headline =
    `<div class="why-sum" data-testid="why-summary">` +
    `<div class="ws-net ${net >= 0 ? 'up' : 'down'}"><span>Net push of all rows</span><b>${signed(net)}</b><span>${net >= NEGLIGIBLE ? 'toward invasive' : net <= -NEGLIGIBLE ? 'away from invasive' : 'about neutral'}</span></div>` +
    '<div class="ws-lists">' +
    (ups.length ? `<p><span class="ws-k up">Toward invasive</span><span>${list(ups)}</span></p>` : '') +
    (downs.length ? `<p><span class="ws-k down">Away from invasive</span><span>${list(downs)}</span></p>` : '') +
    '</div></div>';
  const groups = groupRows(v, cfg)
    .map(
      (g) =>
        `<section class="why-group" data-group="${esc(g.id)}"><h3>${esc(g.label)}${
          g.id === 'cells'
            ? '<span class="src m" data-testid="src-measured" title="Measured from segmented nuclei">◆ measured</span>'
            : '<span class="src" data-testid="src-estimated" title="Estimated from text prompts and concept vectors">○ estimated</span>'
        }</h3><div class="why">${g.rows.map((r) => whyRow(r, hm, maxAbs, v.cells, metrics)).join('')}</div></section>`
    )
    .join('');
  const noLevels = !rows.some((r) => !r.measured && isNum(r.valueStd));
  const levelNote = noLevels
    ? '<p class="sub why-nolevel" data-testid="why-no-level">This case file has no training-standardized levels for the text-prompt criteria, so only each feature\'s push is shown; hover a row for its mean tile z.</p>'
    : '';
  return (
    provenance +
    headline +
    levelNote +
    `<div class="why-key" aria-hidden="true"><span>Feature</span><span>Level in this case</span><span class="axis"><span>← away</span><span>push on the invasive score</span><span>toward →</span></span></div>` +
    groups +
    '<p class="sub why-foot">Level: the case feature in standard deviations from the average training case (the concept model\'s input); typical = within ±0.5 SD. Measured cell features use the percentile against benign/normal reference cases. Push: the concept model\'s learned weights × this case\'s standardized features for that criterion (the 3-strongest-region mean shown as the level, plus the share of regions above threshold), so the push need not follow the level\'s sign. Hover a row for the mean tile z of its 3 strongest regions and the standardized share of regions above threshold. Associated features, not causes; not a Nottingham grade. Select a row to outline its regions on the image.</p>'
  );
}

// ---- How sure ----------------------------------------------------------------------------------------------------------

/** Display name of the conformal method that drives tiers (never hard-coded: read from eval/conformal.json). */
export function conformalName(metrics, alphaPct) {
  const m = metrics && metrics.eval && metrics.eval.conformal ? metrics.eval.conformal.drives_tiers : null;
  const kind = !m ? 'Conformal' : /^aps/.test(m) ? 'APS conformal' : /^lac/.test(m) ? 'LAC conformal' : /mondrian/.test(m) ? 'Class-conditional conformal' : 'Conformal';
  return `${kind} set${alphaPct ? ` (${alphaPct}% target)` : ''}`;
}

function setStatement(v, cfg) {
  const names = v.set.map((c) => classLabel(c, cfg));
  const agg = (cfg && cfg.aggressive_class) || 'Invasive';
  let s;
  if (!v.setSize)
    s = 'Empty set: no category reached the conformal threshold. Read this as low confidence; the image may be atypical for the model.';
  else if (v.setSize === 1)
    s = `One category in the set: <b>${esc(names[0])}</b>. This is the most confident outcome the method gives, and still not a guarantee for this case.`;
  else
    s = `${['Two', 'Three', 'Four'][v.setSize - 2] || v.setSize} categories remain possible: ${names
      .map((n) => `<b>${esc(n)}</b>`)
      .join(', ')
      .replace(/, (?=[^,]*$)/, ' or ')}. The model cannot separate them at the ${esc(v.alphaPct)}% target, so treat the read as uncertain between them.`;
  if (v.set.includes('InSitu') && v.set.includes('Invasive'))
    s += ' This is the in situ vs invasive differential; myoepithelial markers (p63, calponin) are the usual way to separate it.';
  if (v.set.includes(agg) && v.top !== agg)
    s += ` ${esc(classLabel(agg, cfg))} is in the set although it is not the top category, so the case is prioritised as likely invasive (safety rule).`;
  return s;
}

function probChart(v, cfg) {
  const agg = (cfg && cfg.aggressive_class) || 'Invasive';
  const rows = v.probs
    .map((p) => {
      const inSet = v.set.includes(p.cls);
      const top = p.cls === v.top;
      const w = Math.max(0, Math.min(100, num(p.p) * 100));
      const tip = `${p.label}: calibrated probability ${exactProb(p.p)}${inSet ? ' · in the conformal set' : ' · outside the conformal set'}${top ? ' · top category' : ''}`;
      return (
        `<div class="pc-row${inSet ? ' in' : ''}${top ? ' top' : ''}${p.cls === agg ? ' agg' : ''}" title="${esc(tip)}" data-testid="prob-${esc(p.cls)}">` +
        `<span class="pc-lbl">${esc(p.label)}${top ? '<span class="pc-tag">suggested</span>' : ''}</span>` +
        `<span class="pc-track" aria-hidden="true"><i style="width:${w.toFixed(1)}%"></i></span>` +
        `<span class="pc-val">${fmtPct(p.p)}</span>` +
        `<span class="pc-set">${inSet ? '<span class="pc-in" aria-label="in the set">in set</span>' : '<span class="pc-out" aria-hidden="true">·</span>'}</span></div>`
      );
    })
    .join('');
  return `<div class="pchart" role="group" aria-label="Calibrated class probabilities; classes in the conformal set are marked">${rows}</div>`;
}

function check(state, label, value, good, extra = '') {
  const icon = { ok: '✓', warn: '!', info: 'i', na: '–' }[state];
  const word = { ok: 'as expected', warn: 'check', info: 'context', na: 'not available' }[state];
  return (
    `<li class="ck ${state}"${extra}><span class="ck-i" aria-hidden="true">${icon}</span>` +
    `<span class="ck-l"><b>${label}</b><span class="ck-good">${good}</span></span>` +
    `<span class="ck-v"><b>${value}</b><span class="sr-only"> (${word})</span></span></li>`
  );
}

export function howSure(v, cfg, metrics, { compact = false } = {}) {
  if (!v.probs.length) return '<div class="note" data-testid="no-prediction">No model prediction in this case file. Read the image directly; nothing here is scored.</div>';
  const ev = (metrics && metrics.eval) || {};
  const ag = v.agreement || {};
  const r = v.reliability;
  const retrieval = v.findings && [...v.findings.values()].find((f) => f.kind === 'retrieval');
  const quality = v.findings && [...v.findings.values()].find((f) => f.kind === 'quality');
  const p5 = ev.retrieval && ev.retrieval.precision_at_k_test ? ev.retrieval.precision_at_k_test.estimate : null;
  const relThr = ev.reliability && isNum(ev.reliability.threshold) ? Number(ev.reliability.threshold) : null;
  const spreads = v.concepts.map((c) => c.spread).filter(isNum);
  const spread = spreads.length ? spreads.reduce((a, b) => a + Number(b), 0) / spreads.length : null;
  const totalRegions = v.image ? num(v.image.grid[0]) * num(v.image.grid[1]) : null;
  const minTiles = quality && quality.data && isNum(quality.data.min_tiles_kept) ? Number(quality.data.min_tiles_kept) : null;

  const nb = retrieval && retrieval.data && isNum(retrieval.data.of)
    ? { k: num(retrieval.data.agree), of: num(retrieval.data.of) }
    : isNum(ag.neighbor_agreement)
      ? { k: null, of: null, frac: Number(ag.neighbor_agreement) }
      : null;
  const nbFrac = nb ? (nb.of ? nb.k / nb.of : nb.frac) : null;
  const checks =
    check(
      ag.conch_musk_same_top === true ? 'ok' : ag.conch_musk_same_top === false ? 'warn' : 'na',
      'Second model (MUSK) agrees',
      ag.conch_musk_same_top === true ? 'same top category' : ag.conch_musk_same_top === false ? 'different top category' : 'not available',
      'Good: the independent MUSK probe picks the same top category.'
    ) +
    check(
      nbFrac === null ? 'na' : nbFrac >= 0.8 ? 'ok' : 'warn',
      'Similar reference cases agree',
      nb === null ? '—' : nb.of ? `${nb.k} of ${nb.of}` : pct(nb.frac),
      `Good: 4–5 of 5 nearest training cases share the suggested category${isNum(p5) ? `; held-out average ${pct(p5)}` : ''}.`,
      ' data-f-target="retrieval"'
    ) +
    check(
      spread === null ? 'na' : 'info',
      'Prompt-ensemble spread',
      spread === null ? '—' : `${spread.toFixed(2)} mean`,
      'Lower means concept scores are stable across prompt phrasings (no pass mark).'
    ) +
    check(
      r.score === null ? 'na' : r.flag ? 'warn' : 'ok',
      'Reliability',
      (r.score === null ? '—' : r.score.toFixed(2)) + (r.flag ? ' <span class="pill warn" data-testid="reliability-flag">flag raised</span>' : ' · no flag'),
      `Score = 1 − estimated error probability.${relThr !== null ? ` Flag when the estimate reaches ${relThr.toFixed(2)} (score ≤ ${(1 - relThr).toFixed(2)}).` : ''}`,
      ' data-f-target="quality"'
    ) +
    check(
      r.oodPct === null ? 'na' : r.oodPct >= 0.9 ? 'warn' : 'ok',
      'Input typicality (OOD percentile)',
      r.oodPct === null ? '—' : ordinalPct(r.oodPct),
      'Distance to reference tissue, as a percentile of calibration cases. Lower is more typical; 90th and above is unusual.',
      ' data-f-target="quality"'
    ) +
    check(
      r.qcTiles === null ? 'na' : minTiles !== null && r.qcTiles < minTiles ? 'warn' : 'ok',
      'Tissue regions read',
      r.qcTiles === null ? '—' : `${num(r.qcTiles)}${totalRegions ? ` of ${totalRegions}` : ''}`,
      minTiles !== null ? `At least ${minTiles} regions must pass tissue QC.` : 'Regions passing tissue QC.',
      ' data-f-target="quality"'
    );

  const observed = coverageStatus(metrics, v.model);
  const method = conformalName(metrics, v.alphaPct);
  const validation = observed
    ? `<div class="${observed.belowTarget ? 'flag' : 'validation-note'} cv-coverage" data-testid="coverage-validation">Held-out coverage of this method: <b>${pct(observed.coverage)}</b> across ${esc(observed.n)} test cases, against a ${pct(observed.target)} target.${observed.belowTarget ? ' The target was not achieved in this experiment, so the set is less reliable than its label suggests.' : ''}</div>`
    : '';
  const chips = v.setSize
    ? v.set.map((cls) => `<span class="chip conf" data-testid="conformal-chip" title="${esc(classLabel(cls, cfg))}">${esc(cls)}</span>`).join('')
    : '<span class="chip empty" data-testid="conformal-chip-empty">∅ empty set</span>';
  const head =
    `<div class="hs-head"><div class="hs-risk"><div class="bigp" data-testid="risk" title="${esc(isNum(aggProb(v, cfg)) ? exactProb(aggProb(v, cfg), 4) : '')}">${v.risk === null ? '—' : fmtPct(aggProb(v, cfg))}</div><div class="sub">calibrated P(${esc(classLabel((cfg && cfg.aggressive_class) || 'Invasive', cfg))})</div></div>` +
    `<div class="hs-sugg"><span class="sub">Suggested category</span><b>${esc(classLabel(v.top, cfg) || '—')}</b><span class="sub"><code>${esc(v.model || 'model unavailable')}</code>, temperature-calibrated</span></div></div>`;
  return (
    head +
    `<h3 class="hs-h">Calibrated probabilities</h3>${probChart(v, cfg)}` +
    `<div class="hs-set"><h3 class="hs-h">${esc(method)}</h3><div class="chips" data-testid="conformal-set">${chips}</div></div>` +
    `<p class="hs-say" data-testid="set-statement">${setStatement(v, cfg)}</p>` +
    '<p class="sub">Target coverage holds on average across exchangeable cases, not as a promise for this individual case.</p>' +
    validation +
    (compact ? '' : `<h3 class="hs-h">Agreement and quality checks</h3><ul class="checks" data-testid="agreement-checks">${checks}</ul>`) +
    `<div class="kv stamp"><span>Stamp</span><span class="sub">${esc(v.model || '—')} · contract ${esc(v.schema || '—')}</span></div>`
  );
}

/** Just the checklist (used by the surgical view next to its own read). */
export function checksOnly(v, cfg, metrics) {
  const html = howSure(v, cfg, metrics);
  const m = html.match(/<ul class="checks"[\s\S]*?<\/ul>/);
  return m ? m[0] : '';
}

// ---- Compared to ---------------------------------------------------------------------------------------------------

export function compared(v, cfg, assetUrl) {
  if (!v.similar.length && !v.contrast) return '<div class="note">No reference cases.</div>';
  const same = v.similar.filter((s) => s.label === v.top).length;
  const thumbs = v.similar
    .map(
      (s, i) =>
        `<button type="button" class="ref-thumb" data-testid="similar-case" data-cv-compare="${i}" aria-label="Compare with reference ${esc(s.case_id)}, reference label ${esc(classLabel(s.label, cfg))}, similarity ${num(s.similarity).toFixed(2)}">` +
        `<span class="cv-thumb"><img src="${esc(thumbUrl(assetUrl(s.image || 'images/' + s.case_id + '.jpg'), 320))}" alt="" data-cv-img width="320" height="240"></span>` +
        `<span class="rt-lbl">${esc(classLabel(s.label, cfg))}</span>${s.label === v.top ? '<span class="rt-match">same as suggested</span>' : '<span class="rt-match diff">differs</span>'}` +
        `<span class="rt-sim">${esc(s.case_id)} · ${num(s.similarity).toFixed(2)}</span></button>`
    )
    .join('');
  let h =
    `<p class="sub ref-intro">Nearest training-split cases by case-embedding similarity (cosine). Labels are the reference cases' own labels.${v.similar.length ? ` <b>${same} of ${v.similar.length}</b> share the suggested category.` : ''}</p>` +
    `<div class="ref-grid">${thumbs}</div>`;
  const c = v.contrast;
  if (c) {
    const label = (id) => (v.concepts.find((k) => k.id === id) || {}).label || humanize(id);
    const lv = (id) => {
      const k = v.why.find((r) => r.id === id);
      const w = k ? levelOf(k, v.cells).word : '';
      return w ? w.toLowerCase() + ' here' : '';
    };
    h +=
      `<div class="contrast cv-contrast" data-testid="contrast-case">` +
      `<button type="button" class="ref-thumb wide" data-cv-compare="contrast" aria-label="Compare side by side with contrast case ${esc(c.case_id)}"><span class="cv-thumb"><img src="${esc(thumbUrl(assetUrl(c.image || 'images/' + c.case_id + '.jpg'), 400))}" alt="" data-cv-img width="400" height="300"></span></button>` +
      `<div><b>Nearest case with a different label:</b> ${esc(classLabel(c.label, cfg))} <span class="sub">(${esc(c.case_id)} · similarity ${num(c.similarity).toFixed(2)})</span>` +
      `<div class="sub">Differs most in</div><div class="chips">${(c.differs_most_in || [])
        .map((id) => `<span class="chip">${esc(label(id))}${lv(id) ? ` <span class="sub">· ${esc(lv(id))}</span>` : ''}</span>`)
        .join('')}</div>` +
      `<button type="button" class="btn sm" data-cv-compare="contrast" data-testid="compare-contrast">Compare side by side</button></div></div>`;
  }
  return h;
}

// ---- Summary -------------------------------------------------------------------------------------------------------

const KIND_TEXT = {
  prediction: 'Prediction',
  concept: 'Concept association',
  retrieval: 'Reference cases',
  quality: 'Quality',
  cells: 'Measured cells',
  margin: 'Margin read'
};

function findingTitle(f) {
  if (!f) return 'Unknown finding';
  const d = f.data || {};
  const kind = KIND_TEXT[f.kind] || humanize(f.kind);
  if (f.kind === 'concept') return `${f.id} · ${kind}: ${d.label || d.concept}${d.tiles && d.tiles.length ? ` (regions ${d.tiles.join(', ')})` : ''}`;
  if (f.kind === 'cells') return `${f.id} · ${kind}: ${d.label || d.feature} — opens the Cells tab`;
  if (f.kind === 'retrieval') return `${f.id} · ${kind}: ${d.agree} of ${d.of} share ${d.label}`;
  return `${f.id} · ${kind}`;
}

export function summary(v) {
  const s = v.summary;
  if (!s || !s.sentences.length) return '<div class="note">No summary.</div>';
  const gen = s.isLLM ? s.generator.slice(6) : '';
  const model = gen.split('+')[0];
  const plusTemplate = /\+template\b/.test(gen);
  const badge = s.isLLM
    ? `<span class="badge llm" data-testid="summary-generator" title="Written by ${esc(model)} from the structured findings only; every sentence cites findings and was checked against them.">LLM-written from findings · ${esc(model)}${plusTemplate ? ' · + required template sentence' : ''}</span>`
    : '<span class="badge" data-testid="summary-generator">Template, from findings</span>';
  const ps = s.sentences
    .map(
      (sent) =>
        `<p class="sum-s">${esc(sent.text)}${(sent.findings || [])
          .filter((f) => FINDING_ID.test(f))
          .map((f) => {
            const fd = v.findings.get(f);
            return `<button type="button" class="fref" data-testid="finding-ref" data-f="${esc(f)}" data-kind="${esc((fd || {}).kind || '')}" title="${esc(findingTitle(fd))}" aria-label="${esc(findingTitle(fd))}">${esc(f)}</button>`;
          })
          .join('')}</p>`
    )
    .join('');
  return (
    `<div class="sum-badges">${badge}</div>${ps}` +
    '<p class="sub">Written only from structured findings; it never saw the image. Hover or focus a finding to see what it refers to; select it to go there.</p>'
  );
}
