// Model evaluation dashboard (#/metrics): a curated, readable view of every eval JSON served by GET /api/metrics.
// Every chart states what it shows in one sentence computed from the numbers at render time (the pipeline is re-run,
// so nothing here is hard-coded), names its evaluation population and source file, and keeps a collapsible table of
// the numbers it draws. Missing files or fields render "Not available in this run"; a broken section never takes the
// page down. Wording: suggested / associated with / simulated; nothing here is clinically validated.
// Ground truth only appears as test-split aggregates (no case-level labels).
import { interpretPanel } from './interpret.js';
import { classLabel, esc, humanize } from '../adapter.js';
import * as C from '../charts.js';

const OPEN = new Set(); // <details> kept open across the app's re-renders
let renderedWidth = 0;

const SECTIONS = [
  ['summary', 'Summary'],
  ['classification', 'Classification'],
  ['confidence', 'Confidence'],
  ['explanations', 'Explanations'],
  ['reliability', 'Frozen-section reliability'],
  ['workflow', 'Workflow'],
  ['cells', 'Cell measurements'],
  ['retrieval', 'Similar cases'],
  ['limitations', 'Limitations & data']
];
const MODEL_LABEL = {
  uni2h_lr: 'UNI2-h linear probe (256-px tiles)',
  uni2h512_lr: 'UNI2-h linear probe (512-px tiles)',
  uni2h_ft: 'UNI2-h fine-tuned (LoRA, ours)',
  conch_probe_lr: 'CONCH linear probe',
  hoptimus0_lr: 'H-optimus-0 linear probe (256-px tiles)',
  hoptimus0_512_lr: 'H-optimus-0 linear probe (512-px tiles)',
  musk_lr: 'MUSK linear probe',
  abmil: 'Attention MIL (ABMIL)',
  fusion_top3: 'Fusion of top-3 encoders',
  cb_text: 'Concept bottleneck (text concepts)',
  cb_text_cells: 'Concept bottleneck (concepts + cells)',
  cb_cells: 'Cell measurements only',
  conch: 'CONCH',
  musk: 'MUSK',
  uni2h: 'UNI2-h',
  pathgenclip: 'PathGen-CLIP'
};
const METHOD_LABEL = {
  lac_marginal: 'LAC (marginal)',
  aps: 'APS',
  aps_marginal: 'APS (marginal)',
  aps_marginal_nonempty: 'APS, never empty',
  mondrian: 'Mondrian (class-wise LAC)'
};
const POLICY_ORDER = ['fifo', 'tiers', 'tiers_aging', 'tiers_aging_day_scale'];
const DEGRADE_LABEL = { blur: 'Blur', jpeg: 'JPEG compression', stain: 'Stain / colour shift', resample: 'Down-up sampling' };

// ---- small helpers -----------------------------------------------------------------------------------------------------
const isNum = C.isNum;
const modelName = (m) => MODEL_LABEL[m] || humanize(m || 'model');
const methodName = (m) => METHOD_LABEL[m] || humanize(m || 'method');
function get(obj, path) {
  let cur = obj;
  for (const k of Array.isArray(path) ? path : String(path).split('.')) {
    if (cur === null || cur === undefined || typeof cur !== 'object') return undefined;
    cur = cur[k];
  }
  return cur;
}
/** {estimate|mean|value, lo, hi, n, n_groups} -> {est, lo, hi, n, ng} or null. */
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
const f2 = (v) => C.fmtNum(v, 2);
const f1 = (v) => C.fmtNum(v, 1);
const pc = (v, d = 0) => C.fmtPct(v, d);
const minus = (s) => (/^-0(\.0+)?%?$/.test(String(s)) ? String(s).slice(1) : String(s).replace(/^-/, '−'));
/** "lo–hi", or "lo to hi" when a bound is negative (an en dash next to a minus sign misreads). */
function range(lo, hi, f) {
  return lo < 0 || hi < 0 ? `${minus(f(lo))} to ${minus(f(hi))}` : `${f(lo)}–${f(hi)}`;
}
function ciText(c, { pct = false, digits = 2, unit = '' } = {}) {
  if (!c) return 'not available';
  const f = pct ? (v) => pc(v, digits === 2 ? 1 : digits) : (v) => C.fmtNum(v, digits);
  return isNum(c.lo) && isNum(c.hi) ? `${minus(f(c.est))}${unit} (95% CI ${range(c.lo, c.hi, f)}${unit})` : `${minus(f(c.est))}${unit}`;
}
function ciShort(c, { pct = false, digits = 2 } = {}) {
  if (!c) return '—';
  const f = pct ? (v) => pc(v, digits === 2 ? 1 : digits) : (v) => C.fmtNum(v, digits);
  return isNum(c.lo) && isNum(c.hi) ? `${minus(f(c.est))} [${range(c.lo, c.hi, f)}]` : minus(f(c.est));
}
function popText(c, unit = 'images') {
  if (!c || !isNum(c.n)) return '';
  return `n = ${c.n} ${unit}${isNum(c.ng) ? `, ${c.ng} patient / pseudo-patient groups` : ''}`;
}
const plural = (n, one, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const lowerFirst = (t) => (/^[A-Z]{2}/.test(t) ? t : t.charAt(0).toLowerCase() + t.slice(1));
/** AUROC-style values near 1 need a third decimal to be told apart. */
const fa = (v) => C.fmtNum(v, isNum(v) && v >= 0.98 ? 3 : 2);
/** A probability for display that is never rounded to 0 or 1: enough digits to differ from the bound, plus log-odds
 *  when it is extreme (0.999823 -> "0.99982 (log-odds +8.6)"). */
function probText(p) {
  if (!isNum(p)) return 'n/a';
  if (p <= 0 || p >= 1) return C.fmtNum(p, 3);
  const tail = Math.min(p, 1 - p);
  const digits = Math.min(8, Math.max(3, Math.ceil(-Math.log10(tail)) + 1));
  let text = p.toFixed(digits);
  if (/^(1\.0+|0\.0+)$/.test(text)) text = p.toPrecision(6);
  const logit = Math.log(p / (1 - p));
  return tail < 0.01 ? `${text} (log-odds ${logit >= 0 ? '+' : '−'}${Math.abs(logit).toFixed(1)})` : text;
}
const nShort = (c) => (c && isNum(c.n) ? `n = ${c.n}${isNum(c.ng) ? `, ${c.ng} groups` : ''}` : '');
const sum = (a) => a.reduce((x, y) => x + (isNum(y) ? y : 0), 0);
const cls = (c, ST) => classLabel(c, ST.cfg);
function shortClass(c) {
  return { Normal: 'Normal', Benign: 'Benign', InSitu: 'In situ', Invasive: 'Invasive' }[c] || humanize(c);
}

// ---- layout (must match metrics.css container breakpoints) --------------------------------------------------------------
function layout() {
  const app = typeof document !== 'undefined' ? document.getElementById('app') : null;
  const W = Math.max(300, Math.round((app && app.clientWidth) || 1100));
  const narrow = W < 600;
  const two = W >= 980;
  const pad = narrow ? 14 : 22;
  const gap = narrow ? 14 : 20;
  const full = W - 2 * pad - 2;
  const half = two ? (W - gap) / 2 - 2 * pad - 2 : full;
  const panel = (inner, cols) => (inner - (cols - 1) * 16) / cols;
  return { W, narrow, two, full, half, panel };
}

// ---- building blocks ------------------------------------------------------------------------------------------------------
function details(id, summary, body) {
  return `<details class="mx-src" data-mx-open="${esc(id)}"${OPEN.has(id) ? ' open' : ''}><summary>${esc(summary)}</summary>${body}</details>`;
}
function table(head, rows, { num = [] } = {}) {
  if (!rows.length) return '<p class="mx-muted">No rows.</p>';
  const numCol = (i) => (num.includes(i) ? ' class="mx-n"' : '');
  // focusable + labelled so a table that scrolls sideways on a phone is reachable by keyboard (axe scrollable-region)
  return `<div class="mx-tablewrap" tabindex="0" role="region" aria-label="Source numbers table (scrolls sideways)"><table class="mx-table"><thead><tr>${head.map((h, i) => `<th scope="col"${numCol(i)}>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c, i) => `<td${numCol(i)}>${esc(c)}</td>`).join('')}</tr>`)
    .join('')}</tbody></table></div>`;
}
/**
 * One figure: descriptive title, computed takeaway, chart, population + source line, source-numbers table.
 * opts: {id, title, takeaway, chart, population, source, table:{head, rows, num}, notes:[...], wide, testid, caveat}
 */
function figure(o) {
  const src = (Array.isArray(o.source) ? o.source : [o.source]).filter(Boolean);
  const head = `<figcaption><h3>${esc(o.title)}</h3>${o.takeaway ? `<p class="mx-take"><span class="mx-take-label">What this shows</span> ${esc(o.takeaway)}</p>` : ''}</figcaption>
    ${o.caveat ? `<p class="mx-caveat" role="note">${esc(o.caveat)}</p>` : ''}`;
  const chart = `<div class="mx-chart">${o.chart || unavailableInline('The chart has no plottable values in this run.')}</div>`;
  const foot = `${(o.notes || []).length ? `<ul class="mx-notes">${o.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : ''}
    <p class="mx-pop">${o.population ? `<span>Population: ${esc(o.population)}</span>` : ''}${src.length ? `<span>Source: ${src.map((s) => `<code>${esc(s)}</code>`).join(', ')}</span>` : ''}</p>
    ${o.table ? details(`${o.id}-src`, 'Source numbers', table(o.table.head, o.table.rows, o.table)) : ''}`;
  const klass = `mx-fig${o.wide ? ' mx-wide' : ''}${o.side ? ' mx-side' : ''}`;
  const attrs = `class="${klass}" id="fig-${esc(o.id)}"${o.testid ? ` data-testid="${esc(o.testid)}"` : ''}`;
  if (o.side) return `<figure ${attrs}><div class="mx-side-text">${head}${foot}</div>${chart}</figure>`;
  return `<figure ${attrs}>${head}${chart}${foot}</figure>`;
}
function unavailableInline(msg) {
  return `<p class="mx-missing-inline">${esc(msg)}</p>`;
}
function missing(key, title, what) {
  return `<figure class="mx-fig mx-missing" data-testid="metrics-missing-${esc(key)}"><figcaption><h3>${esc(title)}</h3></figcaption><p>Not available in this run: <code>eval/${esc(key)}.json</code>${what ? ` (${esc(what)})` : ''} has not been produced or lacks the fields this chart needs.</p></figure>`;
}
function safe(fn, key, title) {
  try {
    return fn();
  } catch (err) {
    console.warn('metrics view:', title, err);
    return missing(key, title, 'could not be read');
  }
}
function section(id, title, lede, body) {
  return `<section class="mx-section" id="mx-${id}" aria-labelledby="mx-${id}-h"><header class="mx-section-head"><h2 id="mx-${id}-h">${esc(title)}</h2>${lede ? `<p>${esc(lede)}</p>` : ''}</header><div class="mx-grid">${body}</div></section>`;
}
function chip(text, tone = 'neutral') {
  return `<span class="mx-chip mx-chip-${tone}">${esc(text)}</span>`;
}

// ---- summary tiles --------------------------------------------------------------------------------------------------------
function tile({ label, value, ci: c, strip, sentence, tone = 'ok', status = null, source }) {
  return `<article class="mx-kpi mx-kpi-${tone}"><h3>${esc(label)}</h3><p class="mx-kpi-value">${esc(value)}</p>${c ? `<p class="mx-kpi-ci">${esc(c)}</p>` : ''}${strip || ''}${status ? `<p class="mx-kpi-status">${status}</p>` : ''}<p class="mx-kpi-text">${esc(sentence)}</p>${source ? `<p class="mx-kpi-src"><code>${esc(source)}</code></p>` : ''}</article>`;
}
function tierMethod(ev) {
  const A = get(ev, 'analytics.conformal');
  const CF = ev.conformal;
  const methods = (CF && CF.methods) || {};
  let key = (A && A.tier_method_key) || (CF && CF.drives_tiers) || 'lac_marginal';
  if (!methods[key] && key === 'aps_marginal' && methods.aps) key = 'aps';
  if (!methods[key] && methods.lac_marginal) key = 'lac_marginal';
  return { key, block: methods[key], preRegistered: (A && A.pre_registered) || 'lac_marginal', methods, postHoc: key !== ((A && A.pre_registered) || 'lac_marginal') };
}
function summaryTiles(ST, ev) {
  const M = ev.metrics,
    CAL = ev.calibration,
    CF = ev.conformal,
    Q = ev.queue_sim;
  const tiles = [];
  const width = 220;
  const prod = get(M, 'production') || {};
  const auc = ci(prod.invasive_vs_rest_auroc);
  if (auc)
    tiles.push(tile({
      label: 'Separates invasive from other cases',
      value: `AUROC ${fa(auc.est)}`,
      ci: isNum(auc.lo) ? `95% CI ${fa(auc.lo)}–${fa(auc.hi)}` : null,
      strip: C.strip({ est: auc.est, lo: auc.lo, hi: auc.hi, min: 0.5, max: 1, width, aria: `AUROC ${fa(auc.est)} on a 0.5 to 1 scale` }),
      sentence: `Chance is 0.50 and 1.00 is perfect ranking: the share of invasive / non-invasive pairs in which the invasive case gets the higher score.`,
      source: 'eval/metrics.json'
    }));
  const cm = get(M, 'production.confusion');
  const k = (M && M.classes ? M.classes : ST.cfg.classes).indexOf(ST.cfg.aggressive_class);
  const sens = ci(prod.invasive_sensitivity),
    spec = ci(prod.invasive_specificity);
  if (sens) {
    const row = cm && cm.matrix && cm.matrix[k];
    tiles.push(tile({
      label: 'Invasive cases called invasive',
      value: pc(sens.est),
      ci: isNum(sens.lo) ? `95% CI ${pc(sens.lo)}–${pc(sens.hi)}` : null,
      strip: C.strip({ est: sens.est, lo: sens.lo, hi: sens.hi, width, aria: `Sensitivity ${pc(sens.est)}` }),
      sentence: row ? `${row[k]} of ${sum(row)} invasive test images had Invasive as the top suggestion (sensitivity).` : 'Sensitivity for the invasive class on the test split.',
      source: 'eval/metrics.json'
    }));
  }
  if (spec) {
    const ii = cm && cm.insitu_invasive;
    const nonInv = cm && cm.matrix ? sum(cm.matrix.filter((_, i) => i !== k).map((r) => sum(r))) : null;
    const falsePos = cm && cm.matrix ? sum(cm.matrix.filter((_, i) => i !== k).map((r) => r[k])) : null;
    tiles.push(tile({
      label: 'Other cases not called invasive',
      value: pc(spec.est),
      ci: isNum(spec.lo) ? `95% CI ${pc(spec.lo)}–${pc(spec.hi)}` : null,
      strip: C.strip({ est: spec.est, lo: spec.lo, hi: spec.hi, width, aria: `Specificity ${pc(spec.est)}` }),
      sentence: isNum(falsePos) ? `${falsePos} of ${nonInv} non-invasive images were called invasive${ii ? `; ${ii.insitu_called_invasive} of them in situ carcinoma, the clinically important confusion` : ''}.` : 'Specificity for the invasive class.',
      source: 'eval/metrics.json'
    }));
  }
  const f1v = ci(prod.macro_f1),
    bacc = ci(prod.balanced_accuracy);
  if (f1v)
    tiles.push(tile({
      label: 'All four classes (macro-F1)',
      value: f2(f1v.est),
      ci: isNum(f1v.lo) ? `95% CI ${f2(f1v.lo)}–${f2(f1v.hi)}` : null,
      strip: C.strip({ est: f1v.est, lo: f1v.lo, hi: f1v.hi, width, aria: `Macro-F1 ${f2(f1v.est)}` }),
      sentence: `Normal, benign, in situ and invasive weighted equally${bacc ? `; balanced accuracy ${f2(bacc.est)}` : ''}.${(() => {
        const mtx = cm && cm.matrix;
        if (!mtx) return '';
        const names = (M && M.classes) || ST.cfg.classes;
        let best = null,
          errs = 0;
        for (let i = 0; i < mtx.length; i++)
          for (let j = i + 1; j < mtx.length; j++) {
            const v = mtx[i][j] + mtx[j][i];
            errs += v;
            if (!best || v > best.v) best = { v, i, j };
          }
        return best && best.v ? ` Most errors: ${shortClass(names[best.i])} ↔ ${shortClass(names[best.j])} (${best.v} of ${errs}).` : '';
      })()}`,
      source: 'eval/metrics.json'
    }));
  const eceA = ci(get(CAL, 'after.ece')),
    eceB = ci(get(CAL, 'before.ece'));
  if (eceA)
    tiles.push(tile({
      label: 'Calibration error after temperature',
      value: `ECE ${f2(eceA.est)}`,
      ci: isNum(eceA.lo) ? `95% CI ${f2(eceA.lo)}–${f2(eceA.hi)}` : null,
      strip: C.strip({ est: eceA.est, lo: eceA.lo, hi: eceA.hi, min: 0, max: Math.max(0.3, eceB ? eceB.est * 1.4 : 0), target: eceB ? eceB.est : null, width, aria: `ECE ${f2(eceA.est)}; tick marks the value before temperature` }),
      sentence: eceB ? `Was ${f2(eceB.est)} before temperature scaling (tick). Lower is better; ${CAL.ece_bins || 10}-bin ECE is biased upward at n = ${CAL.n}.` : 'Expected calibration error of the top-class confidence (lower is better).',
      source: 'eval/calibration.json'
    }));
  const tm = tierMethod(ev);
  const cov = ci(get(tm.block, 'coverage'));
  if (cov && CF) {
    const target = CF.target_coverage ?? 1 - ST.cfg.alpha;
    const below = cov.est < target;
    const short = (target - cov.est) * 100;
    const pre = tm.postHoc ? ci(get(tm.methods, [tm.preRegistered, 'coverage'])) : null;
    tiles.push(tile({
      label: `Conformal coverage (${methodName(tm.key)})`,
      value: pc(cov.est, 1),
      ci: isNum(cov.lo) ? `95% CI ${pc(cov.lo, 1)}–${pc(cov.hi, 1)}` : null,
      strip: C.strip({ est: cov.est, lo: cov.lo, hi: cov.hi, target, width, tone: below ? 'warn' : 'ok', aria: `Coverage ${pc(cov.est, 1)} against a ${pc(target)} target` }),
      tone: below ? 'warn' : 'ok',
      status: below ? `${chip('Below target', 'warn')} ${esc(f1(short))} points short of ${esc(pc(target))}` : tm.postHoc ? `${chip('At target, chosen after test results', 'warn')}` : `${chip('Meets target', 'ok')} target ${esc(pc(target))}`,
      sentence: tm.postHoc
        ? `This set drives the tiers but was chosen after viewing test results, so its coverage is optimistic. Pre-registered ${methodName(tm.preRegistered)}: ${pre ? pc(pre.est, 1) : 'n/a'}.`
        : 'Share of test images whose true class is inside the suggested set. The 90% guarantee needs exchangeable data and holds on average over calibration sets, not for this cohort.',
      source: 'eval/conformal.json'
    }));
  }
  const tiersQ = get(Q, 'results.tiers'),
    fifoQ = get(Q, 'results.fifo');
  const firstLook = ci(get(tiersQ, 'vs_fifo.aggressive_mean'));
  if (firstLook && tiersQ && fifoQ) {
    tiles.push(tile({
      label: 'Invasive cases seen sooner (simulated)',
      value: `${f1(Math.abs(firstLook.est))} min ${firstLook.est <= 0 ? 'sooner' : 'later'}`,
      ci: isNum(firstLook.lo) ? (firstLook.hi <= 0 ? `95% of simulated days: ${f1(-firstLook.hi)} to ${f1(-firstLook.lo)} min sooner` : `95% of simulated days: ${range(firstLook.lo, firstLook.hi, f1)} min change`) : null,
      sentence: `Mean time to first look ${f1(get(fifoQ, 'aggressive_mean.mean'))} min in arrival order vs ${f1(get(tiersQ, 'aggressive_mean.mean'))} min with Sepia tiers, in a simulated reading day.`,
      source: 'eval/queue_sim.json'
    }));
    const cost = ci(get(tiersQ, 'vs_fifo.non_flagged_mean'));
    if (cost)
      tiles.push(tile({
        label: 'Delay added to unflagged cases (simulated)',
        value: `${cost.est >= 0 ? '+' : '−'}${f1(Math.abs(cost.est))} min`,
        ci: isNum(cost.lo) ? `95% of simulated days: ${range(cost.lo, cost.hi, f1)} min` : null,
        tone: cost.est > 0 ? 'warn' : 'ok',
        status: cost.est > 0 ? chip('Cost of prioritizing', 'warn') : null,
        sentence: `Unflagged (tier 4) cases wait ${f1(get(fifoQ, 'non_flagged_mean.mean'))} → ${f1(get(tiersQ, 'non_flagged_mean.mean'))} min on average. Prioritizing flagged cases is paid for by the rest of the queue.`,
        source: 'eval/queue_sim.json'
      }));
  }
  if (!tiles.length) return missing('metrics', 'Headline results', 'headline metrics');
  return inBrief(ST, ev) + `<div class="mx-kpis" data-testid="metrics-headline">${tiles.join('')}</div>`;
}
/** Three to five computed sentences: what holds up, and what to be careful about. */
function inBrief(ST, ev) {
  const M = ev.metrics,
    CAL = ev.calibration,
    CF = ev.conformal,
    Q = ev.queue_sim,
    REL = ev.reliability;
  const good = [],
    care = [];
  const auc = ci(get(M, 'production.invasive_vs_rest_auroc')),
    f1v = ci(get(M, 'production.macro_f1'));
  const ii = get(M, 'production.confusion.insitu_invasive');
  if (auc) (auc.est >= 0.8 ? good : care).push(`Invasive carcinoma is separated from everything else ${auc.est >= 0.98 ? 'almost perfectly' : auc.est >= 0.9 ? 'well' : auc.est >= 0.8 ? 'moderately' : 'poorly'} on held-out images (AUROC ${fa(auc.est)}${isNum(auc.lo) ? `, CI ${fa(auc.lo)}–${fa(auc.hi)}` : ''}).`);
  if (f1v) care.push(`The four-class call is less certain (macro-F1 ${f2(f1v.est)}, CI ${isNum(f1v.lo) ? `${f2(f1v.lo)}–${f2(f1v.hi)}` : 'n/a'})${ii ? `; ${ii.insitu_called_invasive} of ${ii.n_insitu} in situ images were called invasive` : ''}.`);
  const eA = ci(get(CAL, 'after.ece')),
    eB = ci(get(CAL, 'before.ece'));
  if (eA && eB) (eA.est <= eB.est ? good : care).push(`Temperature scaling ${eA.est <= eB.est ? 'reduced' : 'did not reduce'} calibration error (ECE ${f2(eB.est)} → ${f2(eA.est)}).`);
  const tm = tierMethod(ev);
  const cov = ci(get(tm.block, 'coverage'));
  if (cov && CF) {
    const target = CF.target_coverage ?? 1 - ST.cfg.alpha;
    const pre = ci(get(tm.methods, [tm.preRegistered, 'coverage']));
    if (tm.postHoc) care.push(`Suggested-set coverage is ${pc(cov.est, 1)} against a ${pc(target)} target only for a set chosen after viewing test results; the pre-registered ${methodName(tm.preRegistered)} covered ${pre ? pc(pre.est, 1) : 'n/a'}.`);
    else (cov.est >= target ? good : care).push(`Suggested-set coverage is ${pc(cov.est, 1)} against a ${pc(target)} target.`);
  }
  const low = [...lowPlausibility(ev)].map((id) => get(ev, ['concepts_bach', 'concepts', id, 'label']) || humanize(id));
  if (low.length) care.push(`An AI spot check (not a pathologist) found that the top regions for ${low.join(', ').toLowerCase()} rarely show those features; explanations citing them should be discounted.`);
  const rec = ci(get(REL, 'error_recall_test'));
  if (rec) care.push(`The reliability flag caught ${pc(rec.est)} of wrong reads on test data (it was tuned to catch ${pc(REL.target_error_recall)} on calibration data).`);
  const fl = ci(get(Q, 'results.tiers.vs_fifo.aggressive_mean')),
    cost = ci(get(Q, 'results.tiers.vs_fifo.non_flagged_mean'));
  if (fl && cost) care.push(`In simulation, invasive cases are seen ${f1(Math.abs(fl.est))} min sooner on average, paid for by ${f1(cost.est)} min longer waits for unflagged cases.`);
  if (!good.length && !care.length) return '';
  const li = (t, kind) => `<li class="mx-brief-${kind}"><span class="mx-g mx-g-${kind === 'good' ? 'ok' : 'warn'}" aria-hidden="true"></span><span class="sr-only">${kind === 'good' ? 'Holds up: ' : 'Be careful: '}</span>${esc(t)}</li>`;
  return `<div class="mx-brief"><h3>In brief</h3><ul>${good.map((t) => li(t, 'good')).join('')}${care.map((t) => li(t, 'care')).join('')}</ul></div>`;
}

// ---- classification ------------------------------------------------------------------------------------------------------
function rocPoints(curve, name) {
  if (!curve || !Array.isArray(curve.fpr)) return [];
  return curve.fpr.map((x, i) => ({
    x,
    y: curve.tpr[i],
    tip: [`Sensitivity ${pc(curve.tpr[i])}, false-positive rate ${pc(x)}`, `${name}${isNum(curve.threshold && curve.threshold[i]) ? `, flags cases scoring ≥ ${probText(curve.threshold[i])}` : ', nothing flagged'}`]
  }));
}
function figRoc(ST, ev, L) {
  const A = ev.analytics,
    M = ev.metrics;
  const curves = get(A, 'roc.invasive_vs_rest');
  if (!curves || !curves.production) return missing('analytics', 'ROC curve: invasive vs rest', 'ROC curves');
  const b = get(M, 'models.b') || get(A, 'models.b');
  const prodName = `Production read (${modelName(get(M, 'models.production') || get(A, 'models.production'))}, calibrated)`;
  const series = [{ name: prodName, color: C.COLOR[1], step: true, markers: 'hit', points: rocPoints(curves.production, prodName) }];
  if (b && curves[b]) series.push({ name: `Model B: ${modelName(b)}`, color: C.COLOR[2], step: true, markers: 'hit', dash: true, points: rocPoints(curves[b], modelName(b)) });
  const w = Math.min(L.half, 460);
  const chart = C.lineChart({
    width: w,
    height: w,
    square: true,
    x: { min: 0, max: 1, label: 'False-positive rate (non-invasive cases flagged)', format: (v) => pc(v) },
    y: { min: 0, max: 1, label: 'Sensitivity (invasive cases flagged)', format: (v) => pc(v) },
    series,
    refs: [{ kind: 'diag', label: 'chance' }],
    aria: `ROC curves, invasive versus rest. Production AUROC ${f2(curves.production.auroc)}${b && curves[b] ? `, model B ${f2(curves[b].auroc)}` : ''}.`
  });
  const diff = ci(get(M, 'a_vs_b.difference_a_minus_b'));
  const aucA = ci(get(M, 'production.invasive_vs_rest_auroc'));
  const noDiff = diff && isNum(diff.lo) && diff.lo <= 0 && diff.hi >= 0;
  const take = `The production read reaches AUROC ${aucA ? ciText(aucA, { digits: aucA.est >= 0.98 ? 3 : 2 }) : fa(curves.production.auroc)}${b && curves[b] ? `; the explainable concept-bottleneck model B reaches ${fa(curves[b].auroc)}` : ''}.${diff ? ` The paired difference A − B is ${ciText(diff, { digits: 3 })}${noDiff ? ', so no difference is detectable on this test set' : ''}.` : ''} A curve hugging the top-left corner flags invasive cases without flagging others.`;
  const rows = Object.entries(curves).map(([name, c]) => [name === 'production' ? prodName : modelName(name), f2(c.auroc), String(c.n_pos), String(c.n_neg), String((c.fpr || []).length)]);
  return figure({
    id: 'roc',
    title: 'ROC curve: invasive vs rest',
    takeaway: take,
    chart,
    population: `held-out test split, ${curves.production.n_pos} invasive vs ${curves.production.n_neg} other images`,
    source: ['eval/analytics.json', 'eval/metrics.json'],
    table: { head: ['Model', 'AUROC', 'Invasive', 'Other', 'Curve points'], rows, num: [1, 2, 3, 4] }
  });
}
function figConfusion(ST, ev, L) {
  const conf = get(ev, 'analytics.confusion') || null;
  const M = ev.metrics;
  const counts = (conf && conf.counts) || get(M, 'production.confusion.matrix');
  const classes = (conf && conf.classes) || get(M, 'production.confusion.classes') || ST.cfg.classes;
  if (!Array.isArray(counts)) return missing('metrics', 'Confusion matrix', 'confusion matrix');
  const totals = counts.map((r) => sum(r));
  const norm = counts.map((r, i) => r.map((v) => (totals[i] ? v / totals[i] : null)));
  const chart = C.heatmap({
    width: Math.min(L.half, 560),
    rows: classes.map((c) => shortClass(c)),
    cols: classes.map((c) => shortClass(c)),
    values: norm,
    scale: { kind: 'sequential', max: 1 },
    text: (i, j) => `${counts[i][j]}`,
    tip: (i, j) => [`${counts[i][j]} of ${totals[i]} (${pc(norm[i][j])})`, `True ${cls(classes[i], ST)}, suggested ${cls(classes[j], ST)}`],
    colLabel: 'Suggested class (top of the calibrated read)',
    cellH: 46,
    aria: `Confusion matrix, rows true class, columns suggested class. ${classes.map((c, i) => `${c}: ${counts[i][i]} of ${totals[i]} correct`).join('; ')}.`
  });
  let worst = null;
  counts.forEach((r, i) => r.forEach((v, j) => {
    if (i !== j && (!worst || v > worst.v)) worst = { v, i, j };
  }));
  const ii = get(M, 'production.confusion.insitu_invasive');
  const take = `${classes.map((c, i) => `${shortClass(c)} ${counts[i][i]}/${totals[i]}`).join(', ')} correct. ${ii ? `${ii.insitu_called_invasive} of ${ii.n_insitu} in situ images were called invasive and ${ii.invasive_called_insitu} of ${ii.n_invasive} invasive images were called in situ. ` : ''}${worst && worst.v ? `The most frequent error is ${shortClass(classes[worst.i])} → ${shortClass(classes[worst.j])} (${worst.v}).` : ''}`;
  return figure({
    id: 'confusion',
    title: 'Confusion matrix',
    takeaway: take,
    chart: chart + C.scaleLegend({ caption: 'Shade = share of the row', lo: '0%', hi: '100%' }),
    population: `held-out test split, ${sum(totals)} images (rows = true class)`,
    source: conf ? 'eval/analytics.json' : 'eval/metrics.json',
    table: { head: ['True class', ...classes.map(shortClass), 'Total'], rows: counts.map((r, i) => [cls(classes[i], ST), ...r.map(String), String(totals[i])]), num: classes.map((_, i) => i + 1).concat([classes.length + 1]) }
  });
}
function figPr(ST, ev, L) {
  const PR = get(ev, 'analytics.pr.models');
  const prod = PR && PR.production;
  if (!prod || !Array.isArray(prod.recall) || !prod.recall.length) return '';
  const b = get(ev, 'metrics.models.b');
  const k = ST.cfg.aggressive_class;
  // interpolated precision (best precision at this recall or higher): the usual monotone PR frontier
  const pts = (c, name) => {
    const best = new Map();
    c.recall.forEach((r, i) => {
      const q = c.precision[i];
      if (isNum(r) && isNum(q) && (!best.has(r) || q > best.get(r).q)) best.set(r, { q, t: c.threshold[i] });
    });
    const rs = [...best.keys()].sort((x, y) => x - y);
    let run = 0;
    const out = [];
    for (let i = rs.length - 1; i >= 0; i--) {
      run = Math.max(run, best.get(rs[i]).q);
      const t = best.get(rs[i]).t;
      out.unshift({ x: rs[i], y: run, tip: [`Precision ${pc(run)} at recall ${pc(rs[i])}`, `${name}${isNum(t) ? `, flags cases scoring ≥ ${probText(t)}` : ''}`] });
    }
    return out;
  };
  const series = [{ name: 'Production read (calibrated)', color: C.COLOR[1], step: 'vh', markers: 'hit', keepOrder: true, points: pts(prod, 'Production read') }];
  if (b && PR[b] && PR[b].recall.length) series.push({ name: `Model B: ${modelName(b)}`, color: C.COLOR[2], step: 'vh', dash: true, markers: 'hit', keepOrder: true, points: pts(PR[b], modelName(b)) });
  const w = Math.min(L.half, 460);
  const chart = C.lineChart({ width: w, height: Math.round(w * 0.72), x: { min: 0, max: 1, label: `Recall (share of ${shortClass(k).toLowerCase()} cases flagged)`, format: (v) => pc(v) }, y: { min: 0, max: 1, label: 'Precision (interpolated)', format: (v) => pc(v) }, series, refs: isNum(prod.prevalence) ? [{ kind: 'h', value: prod.prevalence, label: `prevalence ${pc(prod.prevalence)}` }] : [], aria: `Precision-recall curve for ${k}, average precision ${fa(prod.average_precision)}` });
  const atFull = prod.recall.map((r, i) => [r, prod.precision[i]]).filter(([r]) => r >= 1).map(([, q]) => q);
  const bestAtFull = atFull.length ? Math.max(...atFull) : null;
  return figure({
    id: 'pr',
    title: 'Precision–recall: invasive vs rest',
    takeaway: `Average precision ${fa(prod.average_precision)}${b && PR[b] ? ` (model B ${fa(PR[b].average_precision)})` : ''}. ${isNum(bestAtFull) ? `Flagging all ${prod.n_pos} invasive images, ${pc(bestAtFull)} of the flagged images are invasive at best.` : ''} The dashed line is the ${pc(prod.prevalence)} share of invasive images: a useless score would sit on it.`,
    chart,
    population: `held-out test split, ${prod.n_pos} invasive of ${prod.n} images`,
    source: 'eval/analytics.json',
    notes: ['The curve shows the best precision reachable at each recall; average precision is computed from the raw curve (scikit-learn).'],
    table: { head: ['Model', 'Average precision', 'Positives', 'Images'], rows: Object.entries(PR).map(([m, c]) => [m === 'production' ? 'Production read (calibrated)' : modelName(m), fa(c.average_precision), String(c.n_pos), String(c.n)]), num: [1, 2, 3] }
  });
}
function figOvr(ST, ev, L) {
  const ovr = get(ev, 'analytics.roc.production_one_vs_rest');
  if (!ovr) return '';
  const entries = Object.entries(ovr).filter(([, c]) => c && Array.isArray(c.fpr) && c.fpr.length);
  if (!entries.length) return '';
  const cols = L.full < 520 ? 2 : 4;
  const pw = L.panel(L.full, cols);
  const panels = entries.map(([c, curve]) => ({
    title: `${shortClass(c)}: AUROC ${f2(curve.auroc)}`,
    body: C.lineChart({
      width: pw,
      height: pw,
      square: true,
      x: { min: 0, max: 1, label: 'False-positive rate', format: (v) => pc(v), ticks: [0, 0.5, 1] },
      y: { min: 0, max: 1, label: 'Sensitivity', format: (v) => pc(v), ticks: [0, 0.5, 1] },
      series: [{ name: shortClass(c), color: C.COLOR[1], step: true, markers: 'hit', points: rocPoints(curve, `${shortClass(c)} vs rest`) }],
      refs: [{ kind: 'diag' }],
      aria: `${c} versus rest ROC, AUROC ${f2(curve.auroc)}`
    })
  }));
  const worst = entries.reduce((a, b) => (b[1].auroc < a[1].auroc ? b : a));
  const best = entries.reduce((a, b) => (b[1].auroc > a[1].auroc ? b : a));
  const macro = ci(get(ev, 'metrics.production.macro_auroc'));
  return figure({
    id: 'ovr',
    wide: true,
    title: 'One-vs-rest ROC by class',
    takeaway: `One-vs-rest AUROC is highest for ${shortClass(best[0])} (${f2(best[1].auroc)}) and lowest for ${shortClass(worst[0])} (${f2(worst[1].auroc)})${macro ? `; their average, the macro AUROC, is ${ciText(macro)}` : ''}.`,
    chart: C.multiples(panels, { cols }),
    population: 'held-out test split, production calibrated probabilities',
    source: ['eval/analytics.json', 'eval/metrics.json'],
    table: { head: ['Class vs rest', 'AUROC', 'Positives', 'Negatives'], rows: entries.map(([c, v]) => [cls(c, ST), f2(v.auroc), String(v.n_pos), String(v.n_neg)]), num: [1, 2, 3] }
  });
}
function figModels(ST, ev, L) {
  const M = ev.metrics;
  const per = get(M, 'per_model');
  if (!per) return missing('metrics', 'Model comparison', 'per-model metrics');
  const prod = get(M, 'models.production'),
    b = get(M, 'models.b');
  const names = Object.keys(per);
  const order = names.sort((x, y) => (ci(per[y].macro_f1)?.est ?? -1) - (ci(per[x].macro_f1)?.est ?? -1));
  const color = (m) => (m === prod ? C.COLOR[1] : m === b ? C.COLOR[2] : C.COLOR.muted);
  const role = (m) => (m === prod ? 'production (model A)' : m === b ? 'model B, explainable' : '');
  const rowsFor = (metric) => order.map((m) => {
    const c = ci(per[m][metric]);
    return c ? { label: modelName(m), sub: role(m), est: c.est, lo: c.lo, hi: c.hi, color: color(m), emphasis: m === prod } : null;
  }).filter(Boolean);
  const w = L.two ? L.half : L.full;
  const f1Chart = C.forest({ width: w, rows: rowsFor('macro_f1'), x: { min: 0, max: 1, label: 'Macro-F1, four classes (higher is better)' }, aria: 'Macro-F1 by model with 95% confidence intervals' });
  const aucChart = C.forest({ width: w, rows: rowsFor('invasive_vs_rest_auroc'), x: { min: 0.5, max: 1, label: 'AUROC, invasive vs rest', valueFormat: fa }, refs: [{ value: 0.5 }], aria: 'Invasive-vs-rest AUROC by model with 95% confidence intervals' });
  const pf = ci(per[prod] && per[prod].macro_f1);
  const overlap = pf ? order.filter((m) => m !== prod && isNum(ci(per[m].macro_f1)?.hi) && ci(per[m].macro_f1).hi >= pf.lo).length : 0;
  const reason = get(ev, 'model_choice.production_reason');
  const take = pf
    ? `${modelName(prod)} has macro-F1 ${ciText(pf)}. ${overlap} of ${order.length - 1} other models have intervals that overlap it, so the test split alone cannot rank them; the choice was made by cross-validation inside train${reason ? ` (${reason})` : ''}.`
    : 'Macro-F1 and AUROC per evaluated model.';
  const rows = order.map((m) => [modelName(m), role(m) || '—', ciShort(ci(per[m].macro_f1)), ciShort(ci(per[m].invasive_vs_rest_auroc)), ciShort(ci(per[m].balanced_accuracy)), per[m].probabilities || '']);
  return figure({
    id: 'models',
    wide: true,
    title: 'Candidate models on the test split',
    takeaway: take,
    chart: `<div class="mx-pair">${C.multiples([
      { title: 'Four-class macro-F1', body: f1Chart },
      { title: 'Invasive vs rest AUROC', body: aucChart }
    ], { cols: L.two ? 2 : 1 })}</div>` + C.legend([
      { name: 'Production model (A)', color: C.COLOR[1], kind: 'dot' },
      { name: 'Explainable model B', color: C.COLOR[2], kind: 'dot' },
      { name: 'Other evaluated models', color: C.COLOR.muted, kind: 'dot' }
    ]),
    population: popText(ci(per[prod] && per[prod].macro_f1)) || 'held-out test split',
    source: 'eval/metrics.json',
    table: { head: ['Model', 'Role', 'Macro-F1', 'Invasive AUROC', 'Balanced accuracy', 'Probabilities'], rows }
  });
}
function figSelection(ST, ev, L) {
  const MC = ev.model_choice;
  const cands = get(MC, 'candidates');
  if (!Array.isArray(cands)) return '';
  const chosen = cands.filter((c) => c.chosen_within_candidate && isNum(c.cv_log_loss));
  if (!chosen.length) return '';
  const fam = get(MC, 'family_choice.black_box') || {};
  const prod = get(MC, 'production');
  const sorted = [...chosen].sort((a, b) => a.cv_log_loss - b.cv_log_loss);
  const rows = sorted.map((c) => ({
    label: modelName(c.model),
    sub: c.family === 'concept' ? 'concept bottleneck' : '',
    est: c.cv_log_loss,
    lo: isNum(c.cv_log_loss_se) ? c.cv_log_loss - c.cv_log_loss_se : null,
    hi: isNum(c.cv_log_loss_se) ? c.cv_log_loss + c.cv_log_loss_se : null,
    color: c.model === prod ? C.COLOR[1] : C.COLOR.muted,
    emphasis: c.model === prod,
    value: `${C.fmtNum(c.cv_log_loss, 3)} ± ${C.fmtNum(c.cv_log_loss_se, 3)}`
  }));
  const refs = isNum(fam.limit) ? [{ value: fam.limit, label: 'one-SE limit' }] : [];
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0, label: 'Cross-validated log-loss inside train (lower is better)' }, refs, aria: 'Cross-validated log-loss per candidate, with one standard error' });
  const p = get(MC, 'protocol') || {};
  return figure({
    id: 'selection',
    title: 'Model selection inside the training split',
    takeaway: `The lowest cross-validated log-loss was ${modelName(String(fam.best || '').split('@')[0])} (${C.fmtNum(fam.best_loss, 3)}); the one-SE rule picks the simplest candidate within one standard error (limit ${C.fmtNum(fam.limit, 3)}), which is ${modelName(prod)}. The test split was not used${p.test_split_used === false ? ' (recorded)' : ''}.`,
    chart,
    population: `${p.n_train ?? '?'} training images, ${p.n_train_groups ?? '?'} groups, ${p.folds ?? '?'}-fold grouped CV × ${p.repeats ?? '?'} repeats`,
    source: 'eval/model_choice.json',
    table: { head: ['Candidate', 'CV log-loss', '± SE', 'Features'], rows: sorted.map((c) => [modelName(c.model), C.fmtNum(c.cv_log_loss, 3), C.fmtNum(c.cv_log_loss_se, 3), String(c.n_features ?? '')]), num: [1, 2, 3] }
  });
}
function figThunder(ST, ev, L) {
  const TH = ev.thunder;
  if (!TH || !TH.published) return missing('thunder', 'Comparison with published THUNDER results', 'THUNDER comparison');
  const rows = [{ header: true, label: 'Published THUNDER results (no interval reported)' }];
  for (const [kind, vals] of Object.entries(TH.published))
    for (const [m, v] of Object.entries(vals || {}))
      if (isNum(v)) rows.push({ label: `${modelName(m)}, ${kind === 'zero_shot' ? 'zero-shot' : 'linear probe'}`, est: v, color: C.COLOR.muted, hollow: true, value: `${f1(v)}` });
  rows.push({ header: true, label: 'This project, same 132 test images' });
  for (const o of TH.ours || []) {
    const c = ci(o.macro_f1);
    if (c && (o.role || '').startsWith('production')) rows.push({ label: `${modelName(o.model)} (production, calibrated)`, est: c.est, lo: c.lo, hi: c.hi, color: C.COLOR[1], emphasis: true, value: ciShort(c, { digits: 1 }) });
  }
  for (const o of TH.ours_zero_shot || []) {
    const c = ci(o.macro_f1);
    if (c) rows.push({ label: `${modelName(o.model)}, zero-shot`, est: c.est, lo: c.lo, hi: c.hi, color: C.COLOR[3], value: ciShort(c, { digits: 1 }) });
  }
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0, max: 100, label: 'Macro-F1 (%)', format: (v) => C.fmtNum(v, 0) }, aria: 'Macro-F1 percent, published THUNDER results versus this project' });
  const prod = (TH.ours || []).find((o) => (o.role || '').startsWith('production'));
  const pc_ = prod && ci(prod.macro_f1);
  const pubBest = Object.entries(get(TH, 'published.linear_probe') || {}).sort((a, b) => b[1] - a[1])[0];
  const inside = pc_ && pubBest && isNum(pc_.lo) && pubBest[1] >= pc_.lo && pubBest[1] <= pc_.hi;
  // zero-shot vs linear probe, computed (published pairs of the same encoder; our zero-shot reads vs our probe)
  const pubZ = get(TH, 'published.zero_shot') || {},
    pubL = get(TH, 'published.linear_probe') || {};
  const gaps = Object.keys(pubZ).filter((m) => isNum(pubZ[m]) && isNum(pubL[m])).map((m) => pubL[m] - pubZ[m]);
  const oursZ = (TH.ours_zero_shot || []).map((o) => ci(o.macro_f1)).filter(Boolean);
  const zsText = gaps.length && gaps.every((g) => g > 0)
    ? ` Published zero-shot text matching scores ${gaps.length > 1 ? `${f1(Math.min(...gaps))}–${f1(Math.max(...gaps))}` : f1(gaps[0])} points below the linear probe of the same encoder${oursZ.length && pc_ && oursZ.every((z) => z.est < pc_.est) ? '; every zero-shot read of ours is also below the production probe' : ''}.`
    : '';
  return figure({
    id: 'thunder',
    title: 'Comparison with published THUNDER results',
    caveat: `Same split, different protocol: ${String(TH.label || '').replace(/^same split, different protocol\s*/i, '').replace(/^\(|\)$/g, '') || 'published probes resize the whole image; ours tiles the image'}. Not compared with the ICIAR 2018 challenge, which used a hidden test set.`,
    takeaway: pc_ && pubBest ? `Our production macro-F1 is ${ciText(pc_, { digits: 1, unit: '%' })}; the best published linear probe (${modelName(pubBest[0])}) is ${f1(pubBest[1])}%, which ${inside ? 'lies inside our interval' : 'lies outside our interval'}.${zsText}` : 'Published THUNDER results next to this project.',
    chart,
    population: `${TH.n ?? 132} test images, ${TH.n_groups ?? '?'} groups (ours); published values carry no CI`,
    source: 'eval/thunder.json',
    notes: TH.source ? [`Published values: ${TH.source}.`] : [],
    table: { head: ['Result', 'Macro-F1 (%)', 'Balanced accuracy (%)'], rows: [...Object.entries(TH.published).flatMap(([k, v]) => Object.entries(v || {}).map(([m, x]) => [`Published ${modelName(m)} (${humanize(k)})`, f1(x), '—'])), ...(TH.ours || []).map((o) => [`${modelName(o.model)} (${o.role})`, ciShort(ci(o.macro_f1), { digits: 1 }), ciShort(ci(o.balanced_accuracy), { digits: 1 })])] }
  });
}

// ---- confidence ----------------------------------------------------------------------------------------------------------
function relSeries(bins, name, color, dash, xKey = 'confidence', yKey = 'accuracy') {
  return {
    name,
    color,
    dash,
    points: (bins || []).filter((b) => isNum(b[xKey]) && isNum(b[yKey])).map((b) => ({ x: b[xKey], y: b[yKey], tip: [`observed ${pc(b[yKey])} vs predicted ${pc(b[xKey])}`, `${name}: ${b.count} images, scores ${C.fmtNum(b.lo, 2)}–${C.fmtNum(b.hi, 2)}`] }))
  };
}
function figReliability(ST, ev, L) {
  const CAL = ev.calibration;
  const A = get(ev, 'analytics.reliability');
  const before = get(CAL, 'before.reliability_diagram.bins'),
    after = get(CAL, 'after.reliability_diagram.bins');
  if (!after) return missing('calibration', 'Calibration of top-class confidence', 'reliability diagram');
  const cols = L.two && A ? 2 : 1;
  const inner = L.full;
  const pw = cols === 2 ? L.panel(inner, 2) : Math.min(inner, 520);
  const T = CAL.temperature;
  const series = (b, a, xKey, yKey) => [b && relSeries(b, 'Before temperature', C.COLOR[3], true, xKey, yKey), relSeries(a, `After temperature${isNum(T) ? ` (T = ${f2(T)})` : ''}`, C.COLOR[1], false, xKey, yKey)].filter(Boolean);
  const mk = (sr, min, xl, yl, aria) => C.lineChart({
    width: pw,
    height: pw,
    square: true,
    x: { min, max: 1, label: xl, format: (v) => pc(v) },
    y: { min, max: 1, label: yl, format: (v) => pc(v) },
    series: sr,
    refs: [{ kind: 'diag' }],
    legendItems: [],
    aria
  });
  // the top-class confidence of K classes cannot fall below 1/K, so that axis starts there
  const kMin = Math.floor((100 / Math.max(2, ST.cfg.classes.length))) / 100;
  const top = series(before, after, 'confidence', 'accuracy');
  const panels = [{ title: 'Top-class confidence', body: mk(top, kMin, 'Mean confidence in the bin', 'Observed accuracy', 'Reliability diagram of top-class confidence before and after temperature scaling') }];
  if (A && get(A, 'aggressive.after.bins'))
    panels.push({ title: `Risk score: P(${ST.cfg.aggressive_class})`, body: mk(series(get(A, 'aggressive.before.bins'), get(A, 'aggressive.after.bins'), 'mean_predicted', 'observed'), 0, `Mean P(${ST.cfg.aggressive_class}) in the bin`, `Observed ${shortClass(ST.cfg.aggressive_class).toLowerCase()} share`, 'Reliability diagram of the invasive risk score') });
  const legendTop = C.legend([...top.map((sr) => ({ name: sr.name, color: sr.color, kind: sr.dash ? 'dash' : 'line' })), { name: 'Diagonal: perfect calibration', color: 'var(--mut)', kind: 'dash' }]);
  const e = (side, k) => ci(get(CAL, [side, k]));
  const nllUp = e('after', 'nll') && e('before', 'nll') && e('after', 'nll').est > e('before', 'nll').est;
  const take = `Temperature scaling${isNum(T) ? ` (T = ${f2(T)}, fit on ${CAL.temperature_fit_on === 'train_oof' || !CAL.temperature_fit_on ? 'out-of-fold training' : CAL.temperature_fit_on} logits)` : ''} moved ECE from ${f2(e('before', 'ece')?.est)} to ${f2(e('after', 'ece')?.est)} and Brier score from ${f2(e('before', 'brier')?.est)} to ${f2(e('after', 'brier')?.est)}${nllUp ? `, while log-loss rose slightly (${f2(e('before', 'nll').est)} → ${f2(e('after', 'nll').est)})` : ''}. Points above the diagonal are under-confident; each point is an equal-count bin of about ${Math.round((CAL.n || 0) / (CAL.ece_bins || 10))} images, so single bins are noisy.`;
  const rows = [];
  for (const side of ['before', 'after'])
    for (const b of get(CAL, [side, 'reliability_diagram', 'bins']) || []) rows.push([humanize(side), `${C.fmtNum(b.lo, 2)}–${C.fmtNum(b.hi, 2)}`, String(b.count), pc(b.confidence, 1), pc(b.accuracy, 1)]);
  return figure({
    id: 'reliability',
    wide: true,
    testid: 'metrics-calibration',
    title: 'Calibration of top-class confidence',
    takeaway: take,
    chart: legendTop + C.multiples(panels, { cols }),
    population: `held-out test split, n = ${CAL.n} images, ${CAL.n_groups} groups; ${CAL.ece_bins}-bin equal-mass binning`,
    source: ['eval/calibration.json', A ? 'eval/analytics.json' : null],
    table: { head: ['Temperature', 'Confidence range', 'Images', 'Mean confidence', 'Observed accuracy'], rows, num: [2, 3, 4] },
    notes: [`ECE ${ciShort(e('before', 'ece'))} before, ${ciShort(e('after', 'ece'))} after; Brier ${ciShort(e('before', 'brier'))} → ${ciShort(e('after', 'brier'))}; NLL ${ciShort(e('before', 'nll'))} → ${ciShort(e('after', 'nll'))} (95% CIs).`]
  });
}
function figRiskHist(ST, ev, L) {
  const H = get(ev, 'analytics.risk_histogram');
  if (!H || !H.by_class) return '';
  const edges = H.bin_edges || [];
  const cats = edges.slice(0, -1).map((e, i) => `${C.fmtNum(e, 1)}–${C.fmtNum(edges[i + 1], 1)}`);
  const classes = Object.keys(H.by_class).filter((c) => Array.isArray(H.by_class[c]));
  const nOf = (c) => (H.n_by_class && isNum(H.n_by_class[c]) ? H.n_by_class[c] : sum(H.by_class[c]));
  const ymax = Math.max(...classes.flatMap((c) => H.by_class[c]), 1);
  const cols = L.full < 640 ? 1 : 2;
  const pw = L.panel(L.full, cols);
  const panels = classes.map((c) => ({
    title: `${cls(c, ST)} (n = ${nOf(c)})`,
    body: C.columns({
      width: pw,
      height: 150,
      categories: cats.map((cat, i) => (L.narrow || pw < 420 ? (i % 2 ? '' : C.fmtNum(edges[i], 1)) : cat)),
      series: [{ name: cls(c, ST), color: c === ST.cfg.aggressive_class ? C.COLOR[2] : C.COLOR[1], values: H.by_class[c] }],
      y: { max: ymax * 1.15, label: 'Images' },
      x: { label: `P(${ST.cfg.aggressive_class}), calibrated` },
      tip: (j) => [`${H.by_class[c][j]} images`, `${cls(c, ST)}, P(${ST.cfg.aggressive_class}) ${cats[j]}`],
      aria: `${c}: histogram of the calibrated invasive risk score`
    })
  }));
  const k = ST.cfg.aggressive_class;
  const above = (c, t) => sum(H.by_class[c].filter((_, i) => edges[i] >= t));
  const benignish = classes.filter((c) => c !== k && c !== 'InSitu');
  const takeParts = [];
  for (const c of benignish) takeParts.push(`${above(c, 0.1) === 0 ? 'all' : `${nOf(c) - above(c, 0.1)} of ${nOf(c)}`} ${shortClass(c).toLowerCase()} images scored below 0.1`);
  if (H.by_class.InSitu) takeParts.push(`${above('InSitu', 0.5)} of ${nOf('InSitu')} in situ images scored 0.5 or more`);
  if (H.by_class[k]) takeParts.push(`${above(k, 0.9)} of ${nOf(k)} invasive images scored 0.9 or more`);
  return figure({
    id: 'riskhist',
    wide: true,
    title: `Distribution of P(${k}) by true class`,
    takeaway: takeParts.length ? takeParts.join('; ').replace(/^./, (m) => m.toUpperCase()) + '. In situ carcinoma is where scores spread out, which is why the worklist flags any case with Invasive in its set.' : '',
    chart: C.multiples(panels, { cols }),
    population: 'held-out test split, calibrated probabilities, grouped by true class',
    source: 'eval/analytics.json',
    table: { head: ['Score bin', ...classes.map(shortClass)], rows: cats.map((cat, i) => [cat, ...classes.map((c) => String(H.by_class[c][i]))]), num: classes.map((_, i) => i + 1) }
  });
}
function methodOrder(tm) {
  const keys = Object.keys(tm.methods || {});
  const rest = keys.filter((k) => k !== tm.key && k !== tm.preRegistered);
  return [tm.key, ...(tm.preRegistered !== tm.key && keys.includes(tm.preRegistered) ? [tm.preRegistered] : []), ...rest].filter((k, i, a) => keys.includes(k) && a.indexOf(k) === i);
}
function methodTag(k, tm) {
  if (k === tm.key && tm.postHoc) return 'drives the tiers, chosen after viewing test results';
  if (k === tm.key) return 'drives the tiers, pre-registered';
  if (k === tm.preRegistered) return 'pre-registered';
  return 'reported alongside, never drives tiers';
}
function figCoverage(ST, ev, L) {
  const CF = ev.conformal;
  if (!CF || !CF.methods) return missing('conformal', 'Conformal coverage by method', 'coverage');
  const tm = tierMethod(ev);
  const target = CF.target_coverage ?? 1 - ST.cfg.alpha;
  const classes = ST.cfg.classes;
  const rows = [];
  const color = (k) => (k === tm.key ? C.COLOR[1] : k === tm.preRegistered ? C.COLOR[2] : C.COLOR.muted);
  for (const k of methodOrder(tm)) {
    const m = CF.methods[k];
    rows.push({ header: true, label: `${methodName(k)}: ${methodTag(k, tm)}` });
    const all = ci(m.coverage);
    if (all) rows.push({ label: 'All test images', est: all.est, lo: all.lo, hi: all.hi, color: color(k), emphasis: true, value: ciShort(all, { pct: true, digits: 0 }), tip: [ciShort(all, { pct: true }), `${methodName(k)}, all test images (${popText(all)})`] });
    if (k !== tm.key && k !== tm.preRegistered) continue; // other methods: overall only (all rows stay in the table)
    for (const c of classes) {
      const pcov = ci(get(m, ['per_class', c]));
      if (pcov) rows.push({ label: cls(c, ST), est: pcov.est, lo: pcov.lo, hi: pcov.hi, color: color(k), value: ciShort(pcov, { pct: true, digits: 0 }), tip: [ciShort(pcov, { pct: true }), `${methodName(k)}, ${cls(c, ST)} (${popText(pcov)})`] });
    }
  }
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0, max: 1, label: 'Coverage: true class inside the set', format: (v) => pc(v) }, refs: [{ value: target, label: `${pc(target)} target` }], aria: `Conformal coverage by class and method against a ${pc(target)} target` });
  const tb = tm.block;
  const cov = ci(tb && tb.coverage);
  const per = classes.map((c) => [c, ci(get(tb, ['per_class', c]))]).filter(([, v]) => v);
  const low = per.length ? per.reduce((a, b) => (b[1].est < a[1].est ? b : a)) : null;
  const fs = CF.finite_sample;
  const take = cov
    ? `${methodName(tm.key)} covered ${pc(cov.est, 1)} of test images against a ${pc(target)} target${cov.est < target ? ` (${f1((target - cov.est) * 100)} points short${isNum(cov.hi) && cov.hi < target ? '; even the upper CI bound is below target' : ''})` : ''}.${low ? ` Coverage was lowest for ${shortClass(low[0]).toLowerCase()} (${pc(low[1].est)}).` : ''}${tm.postHoc && ci(get(CF, ['methods', tm.preRegistered, 'coverage'])) ? ` The pre-registered ${methodName(tm.preRegistered)} covered ${pc(ci(get(CF, ['methods', tm.preRegistered, 'coverage'])).est, 1)}.` : ''}`
    : 'Conformal coverage by method.';
  const notes = [];
  if (fs && fs.note) notes.push(`With ${fs.n_calibration} calibration images, coverage for a random calibration draw varies (Beta(${(fs.beta || []).join(', ')}), SD ${pc(fs.sd, 1)}); the guarantee is an average over calibration draws, not a promise for this test set.`);
  if (tm.postHoc) notes.push(`${methodName(tm.key)} replaced the pre-registered ${methodName(tm.preRegistered)} after the test results were seen, so its test coverage is optimistic.`);
  return figure({
    id: 'coverage',
    title: `Conformal coverage against the ${pc(target)} target`,
    takeaway: take,
    chart,
    notes,
    population: `held-out test split, n = ${CF.n} images, ${CF.n_groups} groups; alpha = ${CF.alpha}`,
    source: 'eval/conformal.json',
    table: { head: ['Method', 'Class', 'Coverage [95% CI]', 'n', 'Groups'], rows: methodOrder(tm).flatMap((k) => [['all', ...classes]].flat().map((c) => {
      const b = c === 'all' ? ci(CF.methods[k].coverage) : ci(get(CF.methods[k], ['per_class', c]));
      return b ? [methodName(k), c === 'all' ? 'All' : cls(c, ST), ciShort(b, { pct: true }), String(b.n ?? ''), String(b.ng ?? '')] : null;
    }).filter(Boolean)) }
  });
}
function figSetSize(ST, ev, L) {
  const CF = ev.conformal;
  const tm = tierMethod(ev);
  if (!CF || !CF.methods) return '';
  const order = methodOrder(tm);
  const nCls = ST.cfg.classes.length;
  const RAMP = [32, 58, 82, 100]; // 1 class must stay distinct from the card surface in dark mode
  const sizeColor = (s) => (s === 0 ? 'var(--cx-empty)' : `color-mix(in srgb, var(--cx1) ${RAMP[Math.min(s, RAMP.length) - 1]}%, var(--cx-seq0))`);
  const sizes = [...Array(nCls + 1).keys()];
  // one histogram source for chart, sentence and table; a method without one is left out (never shown as zeros)
  const hist = (k) => {
    const raw = get(CF, ['methods', k, 'set_size_histogram']) || get(ev, ['analytics', 'conformal', 'methods', k, 'set_size']);
    return raw && typeof raw === 'object' && Object.values(raw).some(isNum) ? raw : null;
  };
  const withHist = order.filter((k) => hist(k));
  if (!withHist.length) return '';
  const h = (k, s) => (isNum(hist(k)[String(s)]) ? hist(k)[String(s)] : 0);
  const rows = withHist.map((k) => ({ label: methodName(k), parts: sizes.map((s) => ({ name: s === 0 ? 'Empty set' : plural(s, 'class', 'classes'), value: h(k, s), color: sizeColor(s), onStrong: s >= 3 })) }));
  const chart = C.stackedRows({ width: L.two ? L.half : L.full, rows, legendItems: sizes.map((s) => ({ name: s === 0 ? 'Empty set' : plural(s, 'class', 'classes'), color: sizeColor(s), kind: 'rect' })), aria: 'Share of test images by conformal set size, per method' });
  const total = (k) => sum(sizes.map((s) => h(k, s)));
  const n = CF.n || total(withHist[0]);
  const multi = (k) => sum(sizes.filter((s) => s >= 2).map((s) => h(k, s)));
  const lac = hist('lac_marginal') ? 'lac_marginal' : null;
  const parts = [];
  if (lac) parts.push(`${methodName(lac)} returned an empty set for ${h(lac, 0)} of ${total(lac)} images (${pc(total(lac) ? h(lac, 0) / total(lac) : null)}) and a multi-class set for ${multi(lac)}`);
  if (tm.key !== lac && hist(tm.key)) parts.push(`${methodName(tm.key)} returned ${multi(tm.key)} multi-class sets and ${h(tm.key, 0)} empty ones`);
  else if (tm.key === lac && hist('aps')) parts.push(`APS returned ${multi('aps')} multi-class sets`);
  const skipped = order.filter((k) => !hist(k));
  return figure({
    id: 'setsize',
    title: 'Conformal set size by method',
    takeaway: (parts.length ? parts.join('; ') + '. ' : '') + 'An empty set means no class reached the threshold; the worklist treats it as low confidence (tier 3).',
    chart,
    notes: skipped.length ? [`No set-size counts in this run for ${skipped.map(methodName).join(', ')}.`] : [],
    population: `held-out test split, n = ${n} images`,
    source: 'eval/conformal.json',
    table: { head: ['Method', ...sizes.map((s) => (s === 0 ? 'Empty' : String(s)))], rows: withHist.map((k) => [methodName(k), ...sizes.map((s) => String(h(k, s)))]), num: sizes.map((s) => s + 1) }
  });
}
function figComposition(ST, ev, L) {
  const AC = get(ev, 'analytics.conformal');
  if (!AC || !AC.methods) return '';
  const tm = tierMethod(ev);
  const key = AC.methods[tm.key] ? tm.key : Object.keys(AC.methods)[0];
  const comp = get(AC, ['methods', key, 'composition']) || [];
  if (!comp.length) return '';
  const k = ST.cfg.aggressive_class;
  const isDiff = (m) => m.includes(k) && m.includes('InSitu');
  const rows = comp.map((c) => ({
    label: c.members.length ? `{${c.members.map(shortClass).join(', ')}}` : 'Empty set {}',
    value: c.count,
    color: isDiff(c.members) ? C.COLOR[2] : c.members.length ? C.COLOR[1] : 'var(--cx-empty)',
    emphasis: isDiff(c.members),
    valueText: `${c.count} (${pc(c.count / (AC.methods[key].n || sum(comp.map((x) => x.count)) || 1))})`
  }));
  const chart = C.hbars({ width: L.two ? L.half : L.full, rows, aria: `Conformal set composition for ${methodName(key)}` });
  const diff = comp.filter((c) => isDiff(c.members)).reduce((a, c) => a + c.count, 0);
  const others = Object.entries(AC.methods).filter(([m]) => m !== key).map(([m, b]) => [m, (b.composition || []).filter((c) => isDiff(c.members)).reduce((a, c) => a + c.count, 0)]);
  return figure({
    id: 'composition',
    title: 'Differentials in the tier-driving set',
    takeaway: `${methodName(key)} put both in situ and invasive carcinoma in the set for ${plural(diff, 'image')}: the ambiguity that drafts p63 / calponin.${others.length ? ` For comparison: ${others.map(([m, n]) => `${methodName(m)} ${n}`).join(', ')}.` : ''}`,
    chart,
    population: `held-out test split, n = ${AC.methods[key].n ?? sum(comp.map((x) => x.count))} images${AC.methods[key].column ? ` (predictions.csv ${AC.methods[key].column})` : ''}`,
    source: 'eval/analytics.json',
    table: { head: ['Set', 'Images'], rows: comp.map((c) => [c.members.length ? c.members.map((m) => cls(m, ST)).join(' + ') : 'Empty', String(c.count)]), num: [1] }
  });
}

// ---- explanations ------------------------------------------------------------------------------------------------------
function directionVerdict(c, k) {
  const auc = ci(c[`auroc_${k}_vs_rest`]);
  if (c.direction === 'context' || !auc) return { kind: 'neutral', text: 'context only' };
  const up = c.direction === 'up';
  const ok = c.direction_consistent ?? (up ? auc.est > 0.5 : auc.est < 0.5);
  const clear = isNum(auc.lo) && isNum(auc.hi) && (auc.lo > 0.5 || auc.hi < 0.5);
  if (!ok) return { kind: 'bad', text: 'contradicts expected direction' };
  return { kind: 'ok', text: clear ? 'as expected' : 'as expected, interval crosses 0.5' };
}
function figConceptHeat(ST, ev, L) {
  const CB = ev.concepts_bach;
  if (!CB || !CB.concepts) return missing('concepts_bach', 'Concept scores by class', 'concept class means');
  const classes = ST.cfg.classes;
  const k = ST.cfg.aggressive_class;
  const ids = Object.keys(CB.concepts);
  const values = ids.map((id) => classes.map((c) => ci(get(CB.concepts[id], ['class_means', c]))?.est ?? null));
  // shade by the deviation from each concept's own mean across classes, so the pattern across classes is visible
  const centred = values.map((row) => {
    const v = row.filter(isNum);
    const mean = v.length ? sum(v) / v.length : 0;
    return row.map((x) => (isNum(x) ? x - mean : null));
  });
  const max = Math.max(0.5, ...centred.flat().filter(isNum).map(Math.abs));
  const verdicts = ids.map((id) => directionVerdict(CB.concepts[id], k));
  const lowSet = lowPlausibility(ev);
  const zText = (v) => (Math.abs(v) < 0.05 ? '0.0' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1));
  const chart = C.heatmap({
    width: L.two ? L.half : L.full,
    rows: ids.map((id) => (CB.concepts[id].label || humanize(id)) + (lowSet.has(id) ? ' †' : '')),
    cols: classes.map(shortClass),
    values: centred,
    flags: verdicts.map((v) => v.kind),
    scale: { kind: 'diverging', max },
    text: (i, j) => (isNum(values[i][j]) ? zText(values[i][j]) : 'n/a'),
    tip: (i, j) => {
      const c = ci(get(CB.concepts[ids[i]], ['class_means', classes[j]]));
      return [`mean z ${ciShort(c)}`, `${CB.concepts[ids[i]].label || ids[i]} in ${cls(classes[j], ST)} (${popText(c)})`];
    },
    cellH: 34,
    aria: 'Mean concept score (z) per true class, test split'
  });
  const contradictions = (CB.contradictions || []).length ? CB.contradictions : ids.filter((_, i) => verdicts[i].kind === 'bad');
  const checked = verdicts.filter((v) => v.kind !== 'neutral').length;
  const okN = verdicts.filter((v) => v.kind === 'ok').length;
  const highest = (id) => CB.concepts[id].highest_class;
  const exp = ids.filter((id) => Array.isArray(CB.concepts[id].expected_highest));
  const expOk = exp.filter((id) => CB.concepts[id].highest_as_expected).length;
  return figure({
    id: 'concepts',
    title: 'Named features by class',
    takeaway: `${okN} of ${checked} directional concepts move the expected way between invasive and other images${contradictions.length ? `; contradictions: ${contradictions.map((c) => (CB.concepts[c] && CB.concepts[c].label) || c).join(', ')}` : '; none contradicts its expected direction'}. ${exp.length ? `${expOk} of ${exp.length} concepts with an expected "highest class" peak there (e.g. ${CB.concepts[exp[0]].label} highest in ${shortClass(highest(exp[0]))}).` : ''} These are associations, not causes, and not Nottingham grades.`,
    chart: chart + C.scaleLegend({ kind: 'diverging', caption: 'Numbers: mean z-score. Shade: below or above that concept\'s average across classes', lo: 'lower', hi: 'higher' }) + `<p class="mx-glyph-key"><span class="mx-g mx-g-ok" aria-hidden="true"></span>as expected <span class="mx-g mx-g-bad" aria-hidden="true"></span>contradicts <span class="mx-g mx-g-neutral" aria-hidden="true"></span>context, no direction</p>`,
    population: `held-out test split, ${CB.n_cases} images, ${CB.n_groups} groups; feature: ${/__top3$/.test(CB.feature || '') ? 'mean of the 3 highest-scoring tiles' : CB.feature || 'case-level concept score'}`,
    source: lowSet.size ? ['eval/concepts_bach.json', 'eval/spotcheck.json'] : 'eval/concepts_bach.json',
    notes: lowSet.size ? [`† Discount these rows: an AI spot check judged ${LOW_PLAUSIBILITY * 100}% or fewer of their top-scoring regions to show the feature (${[...lowSet].map((id) => (CB.concepts[id] && CB.concepts[id].label) || id).join(', ')}). See "Spot check of top-scoring regions" below.`] : [],
    table: { head: ['Concept', ...classes.map(shortClass), 'Expected', 'Check'], rows: ids.map((id, i) => [CB.concepts[id].label || id, ...values[i].map((v) => (isNum(v) ? v.toFixed(2) : 'n/a')), CB.concepts[id].direction, verdicts[i].text]), num: classes.map((_, i) => i + 1) }
  });
}
function figConceptDirection(ST, ev, L) {
  const CB = ev.concepts_bach;
  if (!CB || !CB.concepts) return '';
  const k = ST.cfg.aggressive_class;
  const ids = Object.keys(CB.concepts);
  const lowSet = lowPlausibility(ev);
  const rows = ids.map((id) => {
    const c = CB.concepts[id];
    const auc = ci(c[`auroc_${k}_vs_rest`]);
    if (!auc) return null;
    const v = directionVerdict(c, k);
    const expected = c.direction === 'up' ? `expected higher in ${shortClass(k).toLowerCase()}` : c.direction === 'down' ? `expected lower in ${shortClass(k).toLowerCase()}` : 'context, no expected direction';
    return { label: (c.label || humanize(id)) + (lowSet.has(id) ? ' †' : ''), sub: `${expected}${c.weak ? ' (weak at tile scale)' : ''}${lowSet.has(id) ? '; low spot-check plausibility' : ''}`, est: auc.est, lo: auc.lo, hi: auc.hi, color: v.kind === 'bad' ? C.COLOR[2] : v.kind === 'ok' ? C.COLOR[1] : C.COLOR.muted, tip: [ciShort(auc), `${c.label}: ${v.text}`] };
  }).filter(Boolean);
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0, max: 1, label: `AUROC of the concept score, ${shortClass(k).toLowerCase()} vs rest` }, refs: [{ value: 0.5, label: 'no association' }], aria: 'Concept score AUROC for invasive versus rest, with expected direction' });
  const strongest = rows.reduce((a, b) => (Math.abs(b.est - 0.5) > Math.abs(a.est - 0.5) ? b : a), rows[0]);
  const byDir = (d) => ids.map((id) => [CB.concepts[id], ci(CB.concepts[id][`auroc_${k}_vs_rest`])]).filter(([c, a]) => c.direction === d && a);
  const describe = (d, want) => {
    const list = byDir(d);
    if (!list.length) return '';
    const ok = list.filter(([, a]) => (want === 'above' ? a.est > 0.5 : a.est < 0.5));
    const clear = ok.filter(([, a]) => isNum(a.lo) && (want === 'above' ? a.lo > 0.5 : a.hi < 0.5));
    return `${ok.length} of ${list.length} concepts expected ${want === 'above' ? 'higher' : 'lower'} in invasive carcinoma sit ${want} 0.5 (${clear.length} with the whole interval ${want} 0.5)`;
  };
  const parts = [describe('up', 'above'), describe('down', 'below')].filter(Boolean);
  return figure({
    id: 'direction',
    title: 'Concept direction against pathology expectation',
    takeaway: rows.length ? `Above 0.5 means the concept scores higher in invasive images, below 0.5 lower. ${parts.join('; ')}. The strongest association is ${strongest.label} (${ciShort(strongest)}).` : '',
    chart: chart + C.legend([{ name: 'As expected', color: C.COLOR[1], kind: 'dot' }, { name: 'Contradicts expected direction', color: C.COLOR[2], kind: 'dot' }, { name: 'Context only', color: C.COLOR.muted, kind: 'dot' }]),
    population: `held-out test split, ${CB.n_cases} images, ${CB.n_groups} groups`,
    source: 'eval/concepts_bach.json',
    table: { head: ['Concept', 'Expected', `AUROC ${k} vs rest [95% CI]`, 'Check'], rows: ids.map((id) => [CB.concepts[id].label || id, CB.concepts[id].direction, ciShort(ci(CB.concepts[id][`auroc_${k}_vs_rest`])), directionVerdict(CB.concepts[id], k).text]) }
  });
}
/** Recorded spot-check verdicts per concept (and for cell overlays), counted from eval/spotcheck.json reviews. */
export function spotVerdicts(sc) {
  const out = { concepts: {}, cells: null, reviewers: [], kinds: [] };
  if (!sc || typeof sc !== 'object') return out;
  const blank = () => ({ n: 0, acceptable: 0, unsure: 0, not_acceptable: 0 });
  const reviewers = new Map();
  for (const r of Array.isArray(sc.reviews) ? sc.reviews : []) {
    if (!r || typeof r.item_id !== 'string') continue;
    const parts = r.item_id.split(':');
    const bucket = parts[0] === 'cells' ? (out.cells ||= blank()) : parts[0] === 'concept' && parts[1] ? (out.concepts[parts[1]] ||= blank()) : null;
    if (!bucket) continue;
    bucket.n += 1;
    if (r.verdict in bucket) bucket[r.verdict] += 1;
    reviewers.set(r.reviewer || 'unnamed', r.reviewer_kind || 'unknown');
  }
  if (!Object.keys(out.concepts).length && sc.summary && sc.summary.concept)
    for (const [id, b] of Object.entries(sc.summary.concept)) out.concepts[id] = { n: b.n, acceptable: b.acceptable, unsure: null, not_acceptable: null };
  out.reviewers = [...reviewers.entries()].map(([name, kind]) => ({ name, kind }));
  out.kinds = [...new Set(out.reviewers.map((r) => r.kind))];
  return out;
}
const LOW_PLAUSIBILITY = 0.25; // share of top regions judged acceptable at or below which a concept is flagged
function lowPlausibility(ev) {
  const v = spotVerdicts(ev.spotcheck);
  return new Set(Object.entries(v.concepts).filter(([, b]) => b.n >= 5 && b.acceptable / b.n <= LOW_PLAUSIBILITY).map(([id]) => id));
}
function figSpotcheck(ST, ev, L) {
  const v = spotVerdicts(ev.spotcheck);
  const ids = Object.keys(v.concepts);
  if (!ids.length) return '';
  const label = (id) => get(ev, ['concepts_bach', 'concepts', id, 'label']) || get(ev, ['concepts', 'concepts', id, 'label']) || humanize(id);
  const VC = { acceptable: 'var(--grn)', unsure: 'var(--amb)', not_acceptable: 'var(--eos)' };
  const VN = { acceptable: 'Shows the feature', unsure: 'Unsure', not_acceptable: 'Does not show it' };
  const order = ids.sort((a, b) => v.concepts[b].acceptable / v.concepts[b].n - v.concepts[a].acceptable / v.concepts[a].n);
  const rows = order.map((id) => ({ label: label(id), parts: ['acceptable', 'unsure', 'not_acceptable'].map((k) => ({ name: VN[k], value: v.concepts[id][k] || 0, color: VC[k], onStrong: true })) }));
  if (v.cells) rows.push({ label: 'Cell-segmentation overlays', parts: ['acceptable', 'unsure', 'not_acceptable'].map((k) => ({ name: VN[k], value: v.cells[k] || 0, color: VC[k], onStrong: true })) });
  const chart = C.stackedRows({ width: L.two ? L.half - 10 : L.full, rows, legendItems: Object.keys(VN).map((k) => ({ name: VN[k], color: VC[k], kind: 'rect' })), aria: 'Spot check of the top-scoring regions per concept' });
  const low = order.filter((id) => v.concepts[id].acceptable / v.concepts[id].n <= LOW_PLAUSIBILITY);
  const high = order.filter((id) => v.concepts[id].acceptable / v.concepts[id].n >= 0.85);
  const ai = v.kinds.length === 1 && v.kinds[0] === 'ai';
  const n = Math.max(...order.map((id) => v.concepts[id].n));
  return figure({
    id: 'spotcheck',
    wide: true,
    side: true,
    title: 'Spot check of top-scoring regions',
    caveat: `${ai ? 'AI-reviewed (not a pathologist)' : `Reviewed by: ${v.kinds.join(', ')}`}: a plausibility check of each concept's ${n} highest-scoring regions, not pathologist validation. Reviewers: ${v.reviewers.map((r) => (/\bAI\b/.test(r.name) ? r.name : `${r.name} (${r.kind})`)).join('; ')}.`,
    takeaway: `${high.length ? `${high.map(label).join(', ')}: at least 85% of top regions looked right. ` : ''}${low.length ? `${low.map((id) => `${label(id)} (${v.concepts[id].acceptable}/${v.concepts[id].n})`).join(', ')} rarely did, so explanations that cite these concepts should be discounted.` : 'No concept fell to 25% or below.'}`,
    chart,
    population: `${n} top-scoring regions per concept from the test split${v.cells ? `; ${v.cells.n} cell overlays` : ''}`,
    source: 'eval/spotcheck.json',
    notes: low.length ? [`Marked † in the concept charts above: ${LOW_PLAUSIBILITY * 100}% or fewer of the top regions judged to show the feature.`] : [],
    table: { head: ['Concept', 'Shows it', 'Unsure', 'Does not', 'Reviewed'], rows: [...order.map((id) => [label(id), String(v.concepts[id].acceptable), String(v.concepts[id].unsure ?? '—'), String(v.concepts[id].not_acceptable ?? '—'), String(v.concepts[id].n)]), ...(v.cells ? [['Cell overlays', String(v.cells.acceptable), String(v.cells.unsure), String(v.cells.not_acceptable), String(v.cells.n)]] : [])], num: [1, 2, 3, 4] }
  });
}
function figNct(ST, ev, L) {
  const CO = ev.concepts;
  const nct = get(CO, 'nct.concepts');
  if (!nct) return missing('concepts', 'Cross-organ check (NCT-CRC-HE-100K)', 'NCT-CRC sanity check');
  const labels = get(CO, 'concepts') || {};
  const rows = [];
  const reversed = [];
  for (const [id, b] of Object.entries(nct)) {
    rows.push({ header: true, label: `${(labels[id] && labels[id].label) || humanize(id)} → colorectal ${(b.positive_classes || []).join(' + ')}` });
    for (const [kind, name, color] of [['text', 'Text prompt score', C.COLOR[1]], ['cav', 'Caption-trained concept vector', C.COLOR[3]]]) {
      const c = ci(b[kind]);
      if (!c) continue;
      if (c.est < 0.5) reversed.push(`${(labels[id] && labels[id].label) || id} (${name.toLowerCase()}, ${f2(c.est)})`);
      rows.push({ label: name, est: c.est, lo: c.lo, hi: c.hi, color, tip: [ciShort(c), `${name}, n = ${c.n} tiles`] });
    }
  }
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0, max: 1, label: 'AUROC for the matching colorectal tissue class' }, refs: [{ value: 0.5, label: 'chance' }], aria: 'Cross-organ concept sanity AUROCs on NCT-CRC-HE-100K' });
  const text = Object.entries(nct).map(([id, b]) => [id, ci(b.text)]).filter(([, c]) => c);
  const good = text.filter(([, c]) => c.est >= 0.8).length;
  return figure({
    id: 'nct',
    testid: 'metrics-concepts',
    title: 'Cross-organ check (NCT-CRC-HE-100K)',
    caveat: 'Cross-organ sanity check on colorectal tiles (NCT-CRC-HE-100K): evidence that tissue-level concepts transfer, not breast validation.',
    takeaway: `Text-prompt scores reach AUROC ≥ 0.80 for ${good} of ${text.length} concepts${reversed.length ? `; reversed (below chance): ${reversed.join(', ')}` : ''}.${(ci(get(nct, 'pleomorphism.text'))?.est ?? 1) < 0.7 ? ' Pleomorphism is weak here, plausibly because colorectal tumour is not breast carcinoma.' : ''}`,
    chart: chart + C.legend([{ name: 'Text prompt score', color: C.COLOR[1], kind: 'dot' }, { name: 'Caption-trained concept vector', color: C.COLOR[3], kind: 'dot' }]),
    population: `${get(CO, 'nct.n_tiles') ?? '?'} colorectal tiles (${Object.keys(get(CO, 'nct.per_class_counts') || {}).length} tissue classes), encoder ${get(CO, 'nct.encoder') || ''}`,
    source: 'eval/concepts.json',
    table: { head: ['Concept', 'Positive classes', 'Text AUROC', 'Vector AUROC'], rows: Object.entries(nct).map(([id, b]) => [(labels[id] && labels[id].label) || id, (b.positive_classes || []).join(', '), ciShort(ci(b.text)), ciShort(ci(b.cav))]) }
  });
}
function figCav(ST, ev, L) {
  const cav = get(ev, 'concepts.cav');
  if (!cav || !cav.concepts) return '';
  const min = cav.min_auroc;
  const rows = Object.entries(cav.concepts).map(([id, b]) => {
    const c = ci(b.heldout_auroc);
    if (!c) return null;
    const label = get(ev, ['concepts', 'concepts', id, 'label']) || humanize(id);
    return { label, sub: b.caption_consistent ? 'caption-consistent' : 'below the bar: text prompt used', est: c.est, lo: c.lo, hi: c.hi, color: b.caption_consistent ? C.COLOR[3] : C.COLOR.muted, tip: [ciShort(c), `${label}: ${b.n_pos} positive / ${b.n_neg} negative captions, ${b.n_groups} groups`] };
  }).filter(Boolean);
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0.5, max: 1, label: 'Held-out AUROC against caption co-mentions' }, refs: isNum(min) ? [{ value: min, label: 'consistency bar' }] : [], aria: 'Caption-consistency of concept vectors' });
  const pass = rows.filter((r) => r.sub === 'caption-consistent').length;
  const fail = rows.filter((r) => r.sub !== 'caption-consistent').map((r) => r.label);
  return figure({
    id: 'cav',
    title: 'Caption consistency of concept vectors',
    caveat: String(cav.label || 'Agreement with caption co-mentions, not tissue truth, not validation.'),
    takeaway: `${pass} of ${rows.length} concept vectors clear the ${isNum(min) ? f2(min) : ''} held-out AUROC bar${fail.length ? `; ${fail.join(', ')} did not, so its text prompt is used instead` : ''}. Captions mention a feature; they do not prove it is in the image.`,
    chart,
    population: `held-out caption folds (${(cav.sets_used || []).join(', ')}), grouped by video / slide; encoder ${cav.encoder || ''}`,
    source: 'eval/concepts.json',
    table: { head: ['Concept', 'Held-out AUROC [95% CI]', 'Positives', 'Negatives', 'Groups', 'Consistent'], rows: Object.entries(cav.concepts).map(([id, b]) => [id, ciShort(ci(b.heldout_auroc)), String(b.n_pos), String(b.n_neg), String(b.n_groups), b.caption_consistent ? 'yes' : 'no']) }
  });
}
function figDeletion(ST, ev, L) {
  const D = ev.deletion;
  const models = get(D, 'models');
  if (!models) return missing('deletion', 'Deletion test of highlighted regions', 'deletion test');
  const prodName = get(ev, 'metrics.models.production');
  const name = prodName && models[prodName] ? prodName : Object.keys(models)[0];
  const m = models[name];
  const primary = get(D, 'settings.primary_subset') || 'predicted_aggressive';
  const n = get(D, 'settings.n_tiles') ?? 3;
  const subsets = [[primary, primary === 'predicted_aggressive' ? 'Images the model calls invasive' : humanize(primary)], ['all_eligible', 'All eligible test images']].filter(([k], i, a) => m[k] && a.findIndex((x) => x[0] === k) === i);
  const label = { mean_drop_risk: `Top-${n} risk tiles removed`, mean_drop_concept: `Top-${n} tiles of the top concept removed`, mean_drop_random: `${n} random tiles removed`, risk_minus_random: 'Risk tiles minus random', concept_minus_random: 'Concept tiles minus random' };
  const f3 = (v) => minus(C.fmtNum(v, 3));
  const show = (c) => `${f3(c.est)} [${range(c.lo, c.hi, (v) => C.fmtNum(v, 3))}]`;
  const excludes0 = (c) => c && isNum(c.lo) && (c.lo > 0 || c.hi < 0);
  const all = subsets.flatMap(([k]) => Object.keys(label).map((q) => ci(m[k][q])).filter(Boolean)).flatMap((c) => [c.lo, c.hi, c.est]).filter(isNum);
  const pad = 0.004;
  const xr = { min: Math.min(0, ...all) - pad, max: Math.max(...all) + pad, label: `Drop in P(${ST.cfg.aggressive_class}) after removing tiles`, format: (v) => minus(C.fmtNum(v, 2)), valueFormat: (v) => C.fmtNum(v, 3) };
  const cols = L.two ? 2 : 1;
  const pw = cols === 2 ? L.panel(L.full, 2) : L.full;
  const panels = subsets.map(([key, title]) => {
    const rows = [];
    for (const q of ['mean_drop_risk', 'mean_drop_concept', 'mean_drop_random', 'risk_minus_random', 'concept_minus_random']) {
      const c = ci(m[key][q]);
      if (!c) continue;
      const diff = q.includes('minus');
      rows.push({ label: label[q], sub: diff ? 'paired difference' : 'mean drop', est: c.est, lo: c.lo, hi: c.hi, color: diff ? (excludes0(c) ? C.COLOR[1] : C.COLOR.muted) : C.COLOR[3], hollow: !diff, value: show(c), tip: [show(c), `${label[q]}: ${excludes0(c) ? 'interval excludes zero' : 'interval includes zero'} (n = ${c.n}, ${c.ng} groups)`] });
    }
    return { title: `${title} (n = ${m[key].n}, ${m[key].n_groups} groups)`, body: C.forest({ width: pw, rows, x: xr, refs: [{ value: 0, label: 'no effect' }], aria: `Deletion test, ${title}` }) };
  });
  const p = m[primary] || {};
  const dr = ci(p.mean_drop_risk),
    dc = ci(p.mean_drop_concept),
    dx = ci(p.mean_drop_random),
    rr = ci(p.risk_minus_random),
    cr = ci(p.concept_minus_random);
  const verdict = (c) => (c && isNum(c.lo) ? (excludes0(c) ? 'interval excludes zero' : 'interval includes zero, inconclusive') : 'no interval');
  const take = dr && dx && rr
    ? `In images the model calls invasive, removing the ${n} highest-risk tiles lowers P(${ST.cfg.aggressive_class}) by ${f3(dr.est)} on average versus ${f3(dx.est)} for ${n} random tiles: a paired difference of ${show(rr)} (${verdict(rr)}).${dc && cr ? ` Removing the top concept's ${n} tiles changes it by ${f3(dc.est)}; versus random ${show(cr)} (${verdict(cr)}).` : ''} Effects are small because a mean-pooled score is spread over many tiles: weak evidence of faithfulness, not proof.`
    : 'Deletion test results.';
  return figure({
    id: 'deletion',
    wide: true,
    title: 'Deletion test of highlighted regions',
    takeaway: take,
    chart: C.multiples(panels, { cols }) + C.legend([{ name: 'Mean drop', color: C.COLOR[3], kind: 'ring' }, { name: 'Difference vs random, interval excludes 0', color: C.COLOR[1], kind: 'dot' }, { name: 'Difference vs random, inconclusive', color: C.COLOR.muted, kind: 'dot' }]),
    population: `held-out test split, ${modelName(name)}; ${get(D, 'settings.n_random') ?? '?'} random ${n}-tile draws per image`,
    source: 'eval/deletion.json',
    table: { head: ['Subset', 'Quantity', 'Estimate [95% CI]', 'n'], rows: subsets.flatMap(([key, t]) => Object.keys(label).map((q) => {
      const c = ci(m[key][q]);
      return c ? [t, label[q], `${minus(C.fmtNum(c.est, 4))} [${range(c.lo, c.hi, (v) => C.fmtNum(v, 4))}]`, String(c.n ?? '')] : null;
    }).filter(Boolean)) }
  });
}

// ---- reliability & frozen sections ---------------------------------------------------------------------------------------
function figDegradation(ST, ev, L) {
  const REL = ev.reliability;
  const curve = get(REL, 'accuracy_vs_degradation.curve');
  // an artifact type with no readable severity point is left out (it would otherwise read "—" and win "hurts most")
  const types = curve && typeof curve === 'object' ? Object.keys(curve).filter((t) => curve[t] && Object.values(curve[t]).some((b) => ci(b))) : [];
  if (!types.length) return missing('reliability', 'Accuracy under simulated frozen-section artifacts', 'accuracy vs degradation');
  const all = types.flatMap((t) => Object.values(curve[t]).map(ci).filter(Boolean)).flatMap((c) => [c.lo, c.hi, c.est]).filter(isNum);
  const ymin = Math.max(0, Math.floor(Math.min(...all) * 10) / 10),
    ymax = Math.min(1, Math.ceil(Math.max(...all) * 10) / 10);
  const cols = L.full < 560 ? 1 : L.full < 900 ? 2 : 4;
  const pw = L.panel(L.full, cols);
  const panels = types.map((t) => {
    const pts = Object.entries(curve[t]).map(([sev, b]) => {
      const c = ci(b);
      return c ? { x: +sev, y: c.est, lo: c.lo, hi: c.hi, tip: [`accuracy ${ciShort(c, { pct: true })}`, `${DEGRADE_LABEL[t] || humanize(t)}, severity ${sev}${+sev === 0 ? ' (clean)' : ''}, n = ${c.n}`] } : null;
    }).filter(Boolean);
    const last = pts[pts.length - 1];
    return {
      title: `${DEGRADE_LABEL[t] || humanize(t)}: ${pc(pts[0]?.y)} → ${pc(last?.y)}`,
      body: C.lineChart({ width: pw, height: Math.max(170, Math.min(230, pw * 0.75)), x: { min: 0, max: Math.max(...pts.map((p) => p.x)), label: 'Severity (0 = clean)', ticks: pts.map((p) => p.x), format: (v) => String(v) }, y: { min: ymin, max: ymax, label: 'Accuracy', format: (v) => pc(v) }, series: [{ name: t, color: C.COLOR[1], band: true, points: pts }], aria: `Accuracy versus ${t} severity` })
    };
  });
  const finals = types.map((t) => {
    const ks = Object.keys(curve[t]).filter((k) => ci(curve[t][k])).sort((a, b) => Number(a) - Number(b));
    return [t, ci(curve[t][ks[ks.length - 1]]).est, ci(curve[t][ks[0]]).est];
  });
  const worst = finals.reduce((a, b) => (b[1] < a[1] ? b : a));
  const skipped = Object.keys(curve).filter((t) => !types.includes(t));
  return figure({
    id: 'degradation',
    wide: true,
    testid: 'metrics-reliability',
    title: 'Accuracy under simulated frozen-section artifacts',
    caveat: 'Simulated degradations of microscopy images (no frozen-section data was available). Shaded bands are 95% cluster-bootstrap intervals.',
    takeaway: `Four-class accuracy falls from ${pc(finals[0][2])} on clean single-view reads to ${finals.map(([t, v]) => `${pc(v)} (${lowerFirst(DEGRADE_LABEL[t] || t)})`).join(', ')} at the highest severity; ${lowerFirst(DEGRADE_LABEL[worst[0]] || worst[0])} hurts most. This drop is why degraded reads carry a reliability flag and can be deferred to permanent sections.`,
    chart: C.multiples(panels, { cols }),
    notes: skipped.length ? [`No accuracy values in this run for ${skipped.map((t) => lowerFirst(DEGRADE_LABEL[t] || humanize(t))).join(', ')}.`] : [],
    population: `held-out test split, ${get(REL, 'accuracy_vs_degradation.model') ? modelName(get(REL, 'accuracy_vs_degradation.model')) : 'production model'}; clean = ${get(REL, 'accuracy_vs_degradation.clean_read') || 'single view'} reads`,
    source: 'eval/reliability.json',
    table: { head: ['Artifact', 'Severity', 'Accuracy [95% CI]', 'n'], rows: types.flatMap((t) => Object.entries(curve[t]).map(([s, b]) => [DEGRADE_LABEL[t] || t, s, ciShort(ci(b), { pct: true }), String(ci(b)?.n ?? '')])) }
  });
}
function figReliabilityScore(ST, ev, L) {
  const REL = ev.reliability;
  if (!REL || !REL.error_auroc_test) return '';
  const rows = [];
  const comb = ci(REL.error_auroc_test);
  rows.push({ label: 'Combined reliability score', est: comb.est, lo: comb.lo, hi: comb.hi, color: C.COLOR[1], emphasis: true });
  const fl = { ood_percentile: 'Distance from training images (OOD)', set_size: 'Conformal set size', conch_musk_disagree: 'CONCH vs MUSK disagreement' };
  for (const [k, b] of Object.entries(REL.feature_auroc_test || {})) {
    const c = ci(b);
    if (c) rows.push({ label: fl[k] || humanize(k), sub: 'single signal', est: c.est, lo: c.lo, hi: c.hi, color: C.COLOR.muted });
  }
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0, max: 1, label: 'AUROC for ranking wrong reads above right ones' }, refs: [{ value: 0.5, label: 'chance' }], aria: 'Reliability score error-detection AUROC' });
  const rec = ci(REL.error_recall_test);
  const fr = REL.flag_rate_test || {};
  const flags = isNum(fr.clean) && isNum(fr.degraded) ? C.columns({ width: Math.min(L.two ? L.half : L.full, 320), height: 150, categories: ['Clean reads', 'Simulated degraded reads'], series: [{ name: 'Flagged', color: C.COLOR[1], values: [fr.clean, fr.degraded] }], y: { max: 1, label: 'Share flagged', format: (v) => pc(v), labelFormat: (v) => pc(v), tipFormat: (v) => pc(v, 1) }, aria: `Reliability flag rate: clean ${pc(fr.clean)}, degraded ${pc(fr.degraded)}` }) : '';
  return figure({
    id: 'relscore',
    title: 'Reliability flag: error detection',
    takeaway: `The combined score ranks wrong reads above correct ones with AUROC ${ciText(comb)}. At the threshold chosen on calibration (target: catch ${pc(REL.target_error_recall)} of errors; ${pc(REL.error_recall_calibration)} there) it caught ${rec ? ciText(rec, { pct: true, digits: 0 }) : 'n/a'} of test-read errors — weaker on held-out data. It flags ${pc(fr.clean)} of clean and ${pc(fr.degraded)} of simulated degraded reads.`,
    chart: chart + (flags ? `<div class="cx-panel-title">Share of reads flagged</div>${flags}` : ''),
    population: `${REL.n_test_reads} test reads (clean + simulated degraded), ${REL.n_test_groups} groups`,
    source: 'eval/reliability.json',
    table: { head: ['Signal', 'Error AUROC [95% CI]'], rows: rows.map((r) => [r.label, ciShort(r)]) }
  });
}
function figDeferral(ST, ev, L) {
  const DF = ev.deferral;
  const title = 'Accuracy on kept reads as the least reliable reads are deferred';
  const H = DF && DF.headline;
  const pops = (DF && DF.populations) || {};
  const head = H && pops[H.population];
  const kept = ci(H && H.accuracy_kept),
    all = ci(H && H.accuracy_all);
  if (!head || !kept || !all) return missing('deferral', title, 'deferral curve');
  const POP = { all_test_reads: 'All test reads (clean + simulated degraded)', clean_test: 'Clean test images' };
  const RANK = { reliability: 'Reliability score', top_prob: 'Top-class probability', set_size: 'Conformal set size', random: 'Random order' };
  const popColor = { all_test_reads: C.COLOR[1], clean_test: C.COLOR[2] };
  const order = ['all_test_reads', 'clean_test'].filter((k) => pops[k] && pops[k].rankings).concat(Object.keys(pops).filter((k) => !['all_test_reads', 'clean_test'].includes(k) && pops[k] && pops[k].rankings));
  const rowsOf = (k, r) => (get(pops, [k, 'rankings', r]) || []).filter((x) => x && isNum(x.fraction) && isNum(x.accuracy));
  const at = (k, r, f) => rowsOf(k, r).find((x) => Math.abs(x.fraction - f) < 1e-9);
  const pts = (k, r) => rowsOf(k, r).filter((x) => x.fraction <= 0.5 + 1e-9).map((x) => ({ x: x.fraction, y: x.accuracy, tip: [`accuracy on kept reads ${pc(x.accuracy, 1)}`, `${POP[k] || humanize(k)}, ${RANK[r] || humanize(r)}: ${pc(x.fraction)} deferred, ${x.kept} kept`] }));
  const series = [];
  for (const k of order) {
    const color = popColor[k] || C.COLOR[3];
    series.push({ name: `${POP[k] || humanize(k)}: reliability ranking`, color, points: pts(k, 'reliability') });
    series.push({ name: `${POP[k] || humanize(k)}: random deferral`, color, dash: true, markers: 'hit', points: pts(k, 'random') });
  }
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  const w = L.two ? L.half : L.full;
  const chart = ys.length
    ? C.lineChart({ width: w, height: 280, x: { min: 0, max: 0.5, label: 'Share of reads deferred (least reliable first)', ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], format: (v) => pc(v) }, y: { min: Math.max(0, Math.floor(Math.min(...ys) * 20) / 20), max: Math.min(1, Math.ceil(Math.max(...ys) * 20) / 20), label: 'Accuracy on kept reads', format: (v) => pc(v) }, series, refs: isNum(H.fraction) ? [{ kind: 'v', value: H.fraction, label: 'pre-registered headline' }] : [], aria: 'Accuracy on kept reads versus share of reads deferred, by population; random deferral dashed' })
    : '';
  const f = H.fraction;
  const verb = (a, b) => (a > b + 1e-9 ? 'raised' : a < b - 1e-9 ? 'lowered' : 'left');
  const nAll = head.n_reads;
  const nDef = isNum(H.deferred) ? H.deferred : null;
  const sentences = [
    `Deferring the ${pc(f)} of ${POP[H.population] ? lowerFirst(POP[H.population]) : 'reads'} with the lowest reliability score${nDef !== null && isNum(nAll) ? ` (${C.fmtNum(nDef, 0)} of ${C.fmtNum(nAll, 0)})` : ''} ${verb(kept.est, all.est)} accuracy on the kept reads from ${ciText(all, { pct: true })} to ${ciText(kept, { pct: true })}${verb(kept.est, all.est) === 'left' ? ' unchanged' : ''}; deferring ${pc(f)} at random keeps it at ${pc(H.accuracy_random_20, 1)} on average.`
  ];
  const alt = ['top_prob', 'set_size'].map((r) => [r, at(H.population, r, f)]).filter(([, x]) => x);
  if (alt.length) sentences.push(`At the same ${pc(f)}, accuracy on kept reads was ${alt.map(([r, x]) => `${pc(x.accuracy, 1)} when ranking by ${lowerFirst(RANK[r])}`).join(' and ')}.`);
  const cl = at('clean_test', 'reliability', f),
    c0 = at('clean_test', 'reliability', 0);
  if (H.population !== 'clean_test' && cl && c0) sentences.push(`On the ${pops.clean_test.n_reads} clean test images alone, reliability deferral ${verb(cl.accuracy, c0.accuracy)} accuracy ${verb(cl.accuracy, c0.accuracy) === 'left' ? `at ${pc(c0.accuracy, 1)}` : `from ${pc(c0.accuracy, 1)} to ${pc(cl.accuracy, 1)}`}.`);
  const gain = ci(H.gain_vs_all);
  const notes = [];
  if (gain && isNum(gain.lo) && isNum(gain.hi)) notes.push(`Paired change in accuracy, kept minus all reads (reported in addition to the pre-registered headline): ${minus(C.fmtNum(gain.est * 100, 1))} points (95% CI ${range(gain.lo * 100, gain.hi * 100, (v) => C.fmtNum(v, 1))}).`);
  const i0 = (k) => at(k, 'reliability', 0);
  for (const k of order) {
    const a = i0(k),
      b = at(k, 'reliability', f);
    if (a && b && isNum(a.invasive_missed_kept) && isNum(b.invasive_missed_kept)) notes.push(`${POP[k] || humanize(k)}: invasive reads called non-invasive among kept reads ${a.invasive_missed_kept} → ${b.invasive_missed_kept}; in situ reads called invasive ${a.insitu_overcalled_kept} → ${b.insitu_overcalled_kept}; share of errors deferred ${pc(b.errors_deferred_share)} at ${pc(f)}.`);
  }
  const fmtCount = (v) => (isNum(v) ? (Number.isInteger(v) ? String(v) : C.fmtNum(v, 1)) : '—');
  const rows = [];
  for (const k of order) for (const r of Object.keys(pops[k].rankings)) for (const x of rowsOf(k, r)) rows.push([POP[k] || humanize(k), RANK[r] || humanize(r), pc(x.fraction), String(x.kept), pc(x.accuracy, 1), isNum(x.errors_deferred_share) ? pc(x.errors_deferred_share) : '—', fmtCount(x.invasive_missed_kept), fmtCount(x.insitu_overcalled_kept)]);
  const popText2 = order.map((k) => `${POP[k] || humanize(k)}: ${C.fmtNum(pops[k].n_reads, 0)} reads, ${pops[k].n_groups} patient / pseudo-patient groups`).join('; ');
  return figure({
    id: 'deferral',
    testid: 'metrics-deferral',
    title,
    caveat: 'Degraded reads are simulated frozen-section artifacts. The ranking, fractions and headline were pre-registered before the analysis was run. Random deferral (dashed) is the mean of 1,000 random orders.',
    takeaway: sentences.join(' '),
    chart,
    notes,
    population: `${popText2}; each read scored by the model that produced it`,
    source: 'eval/deferral.json',
    table: { head: ['Population', 'Ranking', 'Deferred', 'Kept', 'Accuracy', 'Errors deferred', 'Invasive missed (kept)', 'In situ called invasive (kept)'], rows, num: [2, 3, 4, 5, 6, 7] }
  });
}
function figMargin(ST, ev, L) {
  const MG = ev.margin;
  if (!MG) return missing('margin', 'Simulated margin read', 'margin evaluation');
  const rows = [];
  const add = (label, b, color, sub) => {
    const c = ci(b);
    if (c) rows.push({ label, sub: sub || nShort(c), est: c.est, lo: c.lo, hi: c.hi, n: c.n, color, value: ciShort(c, { pct: true, digits: 0 }), tip: [ciShort(c, { pct: true }), `${label} (${nShort(c)})`] });
  };
  rows.push({ header: true, label: 'Benign test images: false alarms (lower is better)' });
  add('Margin wrongly called involved', get(MG, 'benign_fpr.pooled_per_margin'), C.COLOR[2]);
  add('Any of 4 margins wrongly called', get(MG, 'benign_fpr.any_margin'), C.COLOR[2]);
  rows.push({ header: true, label: 'Composite fields: half tumour, half benign (simulated)' });
  add('Tumour-free margin called clear', get(MG, 'composite.exact_negative_margin_clear_rate'), C.COLOR[1]);
  add('Benign-half segments below threshold', get(MG, 'composite.benign_half_segment_clear_rate'), C.COLOR[1]);
  add('Tumour-side margin called involved', get(MG, 'composite.tumor_side_margin_sensitivity'), C.COLOR[1]);
  add('Carcinoma-half segments above threshold', get(MG, 'composite.carcinoma_half_segment_hit_rate'), C.COLOR[1]);
  rows.push({ header: true, label: 'Annotated whole-slide windows' });
  add('Sensitivity (involved margins found)', get(MG, 'wsi_windows.sensitivity'), C.COLOR[3]);
  add('Specificity (clear margins kept clear)', get(MG, 'wsi_windows.specificity'), C.COLOR[3]);
  rows.push({ header: true, label: 'Carcinoma test images (diagnosis-echo check)' });
  add('At least one margin called clear', get(MG, 'echo.p_at_least_one_clear_margin'), C.COLOR.muted);
  const chart = C.forest({ width: L.two ? L.half : L.full, rows, x: { min: 0, max: 1, label: 'Rate', format: (v) => pc(v) }, aria: 'Simulated margin read evaluation' });
  const g = (p) => ci(get(MG, p))?.est;
  return figure({
    id: 'margin',
    title: 'Simulated margin read',
    caveat: 'Simulation: image borders stand in for inked margins (top superior, bottom inferior, left medial, right lateral). Composite fields are synthetic constructions from test images.',
    takeaway: `On composite fields the read kept ${pc(g('composite.exact_negative_margin_clear_rate'))} of tumour-free margins clear and called the tumour-side margin involved ${pc(g('composite.tumor_side_margin_sensitivity'))} of the time; on annotated slide windows sensitivity was ${pc(g('wsi_windows.sensitivity'))} at ${pc(g('wsi_windows.specificity'))} specificity. The threshold is conservative (few false alarms on benign images, ${pc(g('benign_fpr.pooled_per_margin'))}) and misses many involved margins, so the surgeon-facing call stays with the pathologist.`,
    chart,
    population: `test images (${get(MG, 'benign_fpr.n_benign_test') ?? '?'} benign, ${get(MG, 'echo.n_carcinoma_test') ?? '?'} carcinoma), ${get(MG, 'composite.n_composites') ?? '?'} composites, ${get(MG, 'wsi_windows.n_windows') ?? '?'} WSI windows from ${get(MG, 'wsi_windows.n_patients') ?? '?'} patients; threshold ${probText(MG.threshold)} from benign calibration images`,
    source: 'eval/margin.json',
    table: { head: ['Check', 'Rate [95% CI]', 'Population'], rows: rows.filter((r) => !r.header).map((r) => [r.label, ciShort(r, { pct: true }), r.sub || '']) }
  });
}

// ---- workflow --------------------------------------------------------------------------------------------------------
function figTiers(ST, ev, L) {
  const T = get(ev, 'analytics.triage');
  if (!T || !T.tiers_by_class) return '';
  const classes = Object.keys(T.tiers_by_class);
  const tiers = ['1', '2', '3', '4'];
  const counts = classes.map((c) => tiers.map((t) => T.tiers_by_class[c][t] || 0));
  const totals = counts.map((r) => sum(r));
  const tierLabel = (t) => (ST.cfg.labels && ST.cfg.labels.tiers && ST.cfg.labels.tiers[t]) || `Tier ${t}`;
  const chart = C.heatmap({
    width: L.two ? L.half : L.full,
    rows: classes.map((c) => `${shortClass(c)} (n = ${totals[classes.indexOf(c)]})`),
    // phones: four columns cannot hold the tier names without splitting words; number them and name them in a note
    cols: tiers.map((t) => (L.full < 520 ? `Tier ${t}` : `${t}. ${tierLabel(t)}`)),
    values: counts.map((r, i) => r.map((v) => (totals[i] ? v / totals[i] : null))),
    scale: { kind: 'sequential', max: 1 },
    text: (i, j) => String(counts[i][j]),
    tip: (i, j) => [`${counts[i][j]} of ${totals[i]} images`, `True ${cls(classes[i], ST)} → tier ${tiers[j]} (${tierLabel(tiers[j])})`],
    colLabel: 'Worklist tier assigned by the read',
    cellH: 40,
    aria: 'Worklist tier by true class on the test split'
  });
  const k = ST.cfg.aggressive_class;
  const ki = classes.indexOf(k);
  const nonInvT2 = classes.reduce((a, c, i) => a + (i === ki ? 0 : counts[i][1]), 0);
  return figure({
    id: 'tiers',
    title: 'Worklist tier by true class',
    takeaway: ki >= 0 ? `${counts[ki][0] + counts[ki][1]} of ${totals[ki]} invasive test images land in tier 1 or 2 (${counts[ki][0]} through the simulated surgical lane); ${T.aggressive_not_flagged ?? counts[ki][2] + counts[ki][3]} would wait in tiers 3–4. ${nonInvT2} non-invasive images were sent to tier 2, the price of the safety rule.` : '',
    chart: chart + C.scaleLegend({ caption: 'Shade = share of the row', lo: '0%', hi: '100%' }),
    notes: L.full < 520 ? [`Tiers: ${tiers.map((t) => `${t} ${lowerFirst(tierLabel(t))}`).join(', ')}.`] : [],
    population: `held-out test split, ${T.n ?? sum(totals)} images; static tiers (no aging)`,
    source: 'eval/analytics.json',
    table: { head: ['True class', ...tiers.map((t) => `Tier ${t}`)], rows: classes.map((c, i) => [cls(c, ST), ...counts[i].map(String)]), num: [1, 2, 3, 4] }
  });
}
function figReasons(ST, ev, L) {
  const T = get(ev, 'analytics.triage');
  const by = T && T.reasons_by_class;
  if (!by) return '';
  const classes = Object.keys(by);
  const n = get(ev, 'analytics.n_by_class') || {};
  const order = ['aggressive_top', 'aggressive_in_conformal_set', 'set_size_gt_1', 'empty_set', 'low_reliability', 'insufficient_tissue', 'surgical_frozen_section'];
  const present = [...new Set([...order, ...classes.flatMap((c) => Object.keys(by[c]))])].filter((r) => classes.some((c) => by[c][r]));
  if (!present.length) return '';
  const label = (r) => (ST.cfg.labels && ST.cfg.labels.reasons && ST.cfg.labels.reasons[r]) || humanize(r);
  const counts = present.map((r) => classes.map((c) => by[c][r] || 0));
  const chart = C.heatmap({
    width: L.two ? L.half : L.full,
    rows: present.map(label),
    cols: classes.map(shortClass),
    values: counts.map((row) => row.map((v, j) => (n[classes[j]] ? v / n[classes[j]] : null))),
    scale: { kind: 'sequential', max: 1 },
    text: (i, j) => String(counts[i][j]),
    tip: (i, j) => [`${counts[i][j]} of ${n[classes[j]] ?? '?'} images`, `${label(present[i])}: true ${cls(classes[j], ST)}`],
    colLabel: 'True class',
    cellH: 34,
    aria: 'Flag reasons by true class, test split'
  });
  const k = ST.cfg.aggressive_class;
  const insituInSet = (by.InSitu || {}).aggressive_in_conformal_set || 0;
  const benignish = classes.filter((c) => c !== k && c !== 'InSitu');
  const topReason = (c) => Object.entries(by[c]).filter(([r]) => r !== 'surgical_frozen_section').sort((a, b) => b[1] - a[1])[0];
  return figure({
    id: 'reasons',
    title: 'Flag reasons by true class',
    takeaway: `${insituInSet} in situ images had Invasive in the suggested set, which the safety rule sends to tier 2 (tier 1 if they arrive in the simulated surgical lane). ${benignish.map((c) => {
      const t = topReason(c);
      return t ? `${shortClass(c)}: most often "${label(t[0]).toLowerCase()}" (${t[1]} of ${n[c] ?? '?'})` : '';
    }).filter(Boolean).join('; ')}. One case can carry several reasons.`,
    chart: chart + C.scaleLegend({ caption: 'Shade = share of the class', lo: '0%', hi: '100%' }),
    population: `held-out test split, ${T.n} images; reasons from core.tiers`,
    source: 'eval/analytics.json',
    table: { head: ['Reason', ...classes.map(shortClass)], rows: present.map((r, i) => [label(r), ...counts[i].map(String)]), num: classes.map((_, i) => i + 1) }
  });
}
function figQueue(ST, ev, L) {
  const Q = ev.queue_sim;
  const AQ = get(ev, 'analytics.queue');
  if (!Q || !Q.results) return missing('queue_sim', 'Queue simulation: waiting time by policy', 'queue simulation');
  const policies = POLICY_ORDER.filter((p) => Q.results[p]).concat(Object.keys(Q.results).filter((p) => !POLICY_ORDER.includes(p)));
  const pname = (p) => (Q.policies && Q.policies[p]) || humanize(p);
  const color = (p) => (p === 'fifo' ? C.COLOR.muted : p === 'tiers' ? C.COLOR[1] : p === 'tiers_aging' ? C.COLOR[3] : C.COLOR[2]);
  const w = L.two ? L.half : L.full;
  const box = (metric, title, aria) => {
    const rows = policies.map((p) => {
      const d = get(AQ, ['distributions', p, metric]);
      if (d && isNum(d.p50)) return { label: pname(p), q: d, color: color(p) };
      const s = get(Q, ['results', p, metric]);
      return s && isNum(s.mean) ? { label: pname(p), q: { p2_5: s.lo, p50: s.mean, p97_5: s.hi, mean: s.mean }, color: color(p), sub: 'mean and 95% of days' } : null;
    }).filter(Boolean);
    return rows.length ? `<div class="cx-panel-title">${esc(title)}</div>` + C.boxPlot({ width: w, rows, x: { min: 0, label: 'Minutes', unit: ' min' }, aria }) : '';
  };
  const tiers = Q.results.tiers || {},
    fifo = Q.results.fifo || {};
  const dA = get(AQ, 'vs_fifo.tiers.aggressive_mean'),
    dN = get(AQ, 'vs_fifo.tiers.non_flagged_mean');
  const promo = Q.aging_promotions || {};
  // the comparison sentence needs both policies; with one missing, say what is there instead of printing "—"
  const both = (m) => isNum(get(fifo, [m, 'mean'])) && isNum(get(tiers, [m, 'mean']));
  const firstLook = both('aggressive_mean')
    ? `Sepia tiers ${get(tiers, 'aggressive_mean.mean') <= get(fifo, 'aggressive_mean.mean') ? 'cut' : 'raised'} the mean time to first look for invasive cases from ${f1(get(fifo, 'aggressive_mean.mean'))} to ${f1(get(tiers, 'aggressive_mean.mean'))} min${dA && isNum(dA.share_days_lower) ? ` (sooner on ${pc(dA.share_days_lower)} of days)` : ''}`
    : '';
  const unflagged = both('non_flagged_mean')
    ? `unflagged cases waited ${f1(get(fifo, 'non_flagged_mean.mean'))} → ${f1(get(tiers, 'non_flagged_mean.mean'))} min${dN && isNum(dN.share_days_higher) ? ` (longer on ${pc(dN.share_days_higher)} of days)` : ''}`
    : '';
  const compare = firstLook && unflagged ? `${firstLook}, while ${unflagged}.` : firstLook || unflagged ? `${firstLook || unflagged}.` : 'the arrival-order and Sepia-tier results are not both present in this run, so no comparison is drawn.';
  const take = `In ${isNum(Q.reps) ? Q.reps + ' ' : ''}simulated reading days, ${compare} ${isNum(promo.tiers_aging) && promo.tiers_aging === 0 ? 'The configured aging guard never promoted a case within an 8-hour day, so it has no measured effect here.' : ''}`;
  const a = Q.assumptions || {};
  const assumptions = [a.arrivals, a.reader, isNum(a.utilization) ? `reader utilization ${pc(a.utilization)}` : null, a.tiers, a.interval, a.truth_use].filter(Boolean);
  const rows = [];
  for (const p of policies) for (const m of ['aggressive_mean', 'aggressive_p90', 'non_flagged_mean', 'max_wait', 'overtime']) {
    const s = get(Q, ['results', p, m]);
    if (s && isNum(s.mean)) rows.push([pname(p), humanize(m), f1(s.mean), `${f1(s.lo)}–${f1(s.hi)}`]);
  }
  return figure({
    id: 'queue',
    wide: true,
    testid: 'metrics-queue-sim',
    title: 'Queue simulation: waiting time by policy',
    caveat: 'A simulation with stated assumptions, not a measured workflow outcome. Boxes span the middle 50% of simulated days; whiskers 95% of days; the line is the median and the diamond the mean.',
    takeaway: take,
    chart: `<div class="mx-pair">${C.multiples([
      { title: '', body: box('aggressive_mean', 'Invasive cases: mean time to first look per day', 'Time to first look for invasive cases by policy') },
      { title: '', body: box('non_flagged_mean', 'Unflagged (tier 4) cases: mean wait per day', 'Wait for unflagged cases by policy') }
    ], { cols: L.two ? 2 : 1 })}</div>`,
    notes: assumptions.map((s) => `Assumption: ${String(s).replace(/\b(aps_marginal_nonempty|lac_marginal|aps_marginal)\b/g, (m) => methodName(m))}.`),
    population: `${Q.n_cases} clean test cases drawn with replacement; ${Q.reps} simulated days${AQ && get(AQ, 'consistency.matches_queue_sim') ? ' (distributions replayed and checked against queue_sim.json)' : ''}`,
    source: ['eval/queue_sim.json', AQ ? 'eval/analytics.json' : null],
    table: { head: ['Policy', 'Measure (minutes)', 'Mean', '95% of days'], rows, num: [2] }
  });
}

// ---- cells ------------------------------------------------------------------------------------------------------------
function figCells(ST, ev, L) {
  const CE = ev.cells;
  const feats = get(CE, 'direction_checks.features');
  if (!feats) return missing('cells', 'Measured cell features by class', 'cell direction checks');
  const classes = ST.cfg.classes;
  const cols = L.full < 560 ? 1 : L.full < 900 ? 2 : 3;
  const pw = L.panel(L.full, cols);
  const tone = { as_expected: 'ok', wrong_way: 'warn', inconclusive: 'neutral', context: 'neutral' };
  const vtext = { as_expected: 'as expected', wrong_way: 'wrong way', inconclusive: 'inconclusive', context: 'context' };
  const panels = Object.entries(feats).map(([id, f]) => {
    const rows = classes.map((c) => {
      const m = ci(get(f, ['class_means', c]));
      return m ? { label: shortClass(c), est: m.est, lo: m.lo, hi: m.hi, color: c === ST.cfg.aggressive_class ? C.COLOR[2] : C.COLOR[1], tip: [`${ciShort(m)}${f.unit ? ' ' + f.unit : ''}`, `${f.display_name || id} in ${cls(c, ST)} (${popText(m)})`] } : null;
    }).filter(Boolean);
    const vals = rows.flatMap((r) => [r.lo, r.hi, r.est]).filter(isNum);
    const span = Math.max(...vals) - Math.min(...vals) || 1;
    return {
      title: `${f.display_name || humanize(id)}${f.unit ? ` (${f.unit})` : ''}`,
      badge: vtext[f.verdict] || f.verdict,
      badgeTone: tone[f.verdict] || 'neutral',
      body: C.forest({ width: pw, rows, rowH: 24, valueWidth: 0, x: { min: Math.max(0, Math.min(...vals) - span * 0.08), max: Math.max(...vals) + span * 0.08 }, aria: `${f.display_name || id}: class means with 95% CI` })
    };
  });
  const counts = {};
  for (const f of Object.values(feats)) counts[f.verdict] = (counts[f.verdict] || 0) + 1;
  const name = (v) => Object.entries(feats).filter(([, f]) => f.verdict === v).map(([id, f]) => (f.display_name || id).toLowerCase());
  const directional = Object.values(feats).filter((f) => f.expected !== 'context').length;
  return figure({
    id: 'cells',
    wide: true,
    title: 'Measured cell features by class',
    caveat: 'Automated nuclei segmentation and typing (HoVer-Net, PanNuke classes). Not pathologist-validated; cell types are model outputs.',
    takeaway: `Of ${directional} features with an expected direction (contrast: ${get(CE, 'direction_checks.contrast') || 'invasive vs benign'}), ${plural(counts.as_expected || 0, 'differs', 'differ')} as expected${name('as_expected').length ? ` (${name('as_expected').join(', ')})` : ''}, ${plural(counts.wrong_way || 0, 'goes', 'go')} the wrong way${name('wrong_way').length ? ` (${name('wrong_way').join(', ')})` : ''} and ${plural(counts.inconclusive || 0, 'is', 'are')} inconclusive. Measurements support, but do not drive, the production read.`,
    chart: C.multiples(panels, { cols }) + C.legend([{ name: shortClass(ST.cfg.aggressive_class), color: C.COLOR[2], kind: 'dot' }, { name: 'Other classes', color: C.COLOR[1], kind: 'dot' }]),
    population: `held-out test split, ${CE.n_test_cases} images with cell outputs, ${CE.n_test_groups} groups; counts per class shown in the tooltips`,
    source: 'eval/cells.json',
    table: { head: ['Feature', 'Expected', 'Verdict', 'Difference [95% CI]', ...classes.map(shortClass)], rows: Object.entries(feats).map(([id, f]) => [f.display_name || id, f.expected, vtext[f.verdict] || f.verdict, ciShort(ci(f.difference)), ...classes.map((c) => C.fmtNum(ci(get(f, ['class_means', c]))?.est, 2))]) }
  });
}

// ---- retrieval -------------------------------------------------------------------------------------------------------
function figRetrieval(ST, ev, L) {
  const R = ev.retrieval;
  const p = ci(get(R, 'precision_at_k_test'));
  if (!p) return missing('retrieval', 'Retrieval precision of reference cases', 'precision@k');
  const rows = [{ label: 'All test images', est: p.est, lo: p.lo, hi: p.hi, color: C.COLOR[1], emphasis: true, value: ciShort(p, { pct: true, digits: 0 }) }];
  for (const [c, v] of Object.entries(R.per_class_precision_at_k_test || {})) if (isNum(v)) rows.push({ label: cls(c, ST), sub: 'no interval', est: v, color: C.COLOR[1], hollow: true, value: pc(v) });
  const chart = C.forest({ width: L.two ? L.half - 10 : L.full, rows, x: { min: 0, max: 1, label: `Share of the ${R.k} nearest training images with the same label`, format: (v) => pc(v) }, aria: 'Retrieval precision at k' });
  const per = Object.entries(R.per_class_precision_at_k_test || {}).filter(([, v]) => isNum(v)).sort((a, b) => b[1] - a[1]);
  return figure({
    id: 'retrieval',
    wide: true,
    side: true,
    title: 'Retrieval precision of reference cases',
    takeaway: `On average ${pc(p.est)} of each test image's ${R.k} nearest training images share its label (95% CI ${range(p.lo, p.hi, (v) => pc(v))}).${per.length >= 2 ? ` Best for ${per.slice(0, 2).map(([c, v]) => `${shortClass(c).toLowerCase()} (${pc(v)})`).join(' and ')}; weakest for ${per.slice(-2).map(([c, v]) => `${shortClass(c).toLowerCase()} (${pc(v)})`).join(' and ')}, so reference cases are context for the reader, not a vote.` : ''}`,
    chart,
    population: `held-out test split queries against a library of ${R.library || 'training cases'}; encoder ${R.encoder || ''}`,
    source: 'eval/retrieval.json',
    table: { head: ['Group', `Precision@${R.k}`], rows: rows.map((r) => [r.label, r.value]) }
  });
}

// ---- data quality, limitations, provenance ------------------------------------------------------------------------------
function figDataQuality(ST, ev, L) {
  const DQ = ev.qc;
  if (!DQ || !DQ.insufficient_tissue_cases) return '';
  const it = DQ.insufficient_tissue_cases || {};
  const classes = Object.keys(it.old_by_class || {});
  const chart = classes.length ? C.columns({
    width: L.two ? L.half : L.full,
    height: 200,
    categories: classes.map(shortClass),
    series: [{ name: 'Before the fix', color: C.COLOR.muted, values: classes.map((c) => it.old_by_class[c]) }, { name: 'After the fix', color: C.COLOR[1], values: classes.map((c) => (it.new_by_class || {})[c] ?? 0) }],
    y: { label: 'Images' },
    zeroLabels: true,
    aria: 'Images with insufficient tissue before and after the QC fix'
  }) : '';
  const qp = DQ.qc_pass_rate || {};
  const grids = Object.keys(qp.old || {});
  return figure({
    id: 'dataquality',
    title: 'Tissue QC before and after the fix',
    takeaway: `A tissue-mask fix ${grids.length ? `raised the tile QC pass rate ${grids.map((g) => `${pc(qp.old[g])} → ${pc((qp.new || {})[g])} (${g})`).join(', ')} and ` : ''}${isNum(it.old) && isNum(it.new) ? `reduced images flagged "insufficient tissue" from ${it.old} to ${it.new}` : 'changed which images are flagged "insufficient tissue"'}${isNum(it.old) && it.old_by_class && isNum(it.old_by_class.Invasive) && it.old_by_class.Invasive * 2 > it.old ? `; ${it.old_by_class.Invasive} of the ${it.old} were invasive, so the bug mostly hid real tumour tissue` : ''}.`,
    chart,
    population: `all ${DQ.n_cases ?? ''} BACH images (every split), ${DQ.n_tiles ?? ''} tiles; tile-level QC, not a model result`,
    source: 'eval/qc.json',
    notes: DQ.validation_old_rule_display_vs_fullres ? [`Check: the old rule recomputed on display images matches the full-resolution tissue fractions (Pearson r ${C.fmtNum(DQ.validation_old_rule_display_vs_fullres.pearson_r, 3)}).`] : [],
    table: { head: ['Class', 'Insufficient before', 'Insufficient after'], rows: classes.map((c) => [cls(c, ST), String(it.old_by_class[c]), String((it.new_by_class || {})[c] ?? '')]), num: [1, 2] }
  });
}
const LIMITATIONS = [
  'Research prototype on a small public dataset (BACH, CC BY-NC-ND 4.0); not a clinical device and not for clinical use.',
  'BACH labels are diagnostic categories, not Nottingham grades; "likely aggressive" means "likely invasive carcinoma".',
  'Small test set (THUNDER split, 132 images) with partial patient IDs: pseudo-patient groups and wide intervals.',
  'The surgical lane is simulated: image borders stand in for inked margins and frozen-section artifacts are simulated degradations; composite fields are synthetic constructions from test images.',
  'Concept scores are features associated with the classes, not causes and not validated grade components; caption consistency measures caption co-mention, not tissue truth.',
  'Conformal coverage assumes exchangeable data; it is not a per-case guarantee and does not carry over to degraded or composite inputs.',
  'Follow-up order rules are illustrative and require clinical review; the agent only drafts orders. Spot checks say whether a human or an AI reviewed them.',
  'The queue simulation is a model with stated assumptions, not a measured workflow outcome.'
];
const CITATIONS = [
  ['BACH dataset', 'Aresta G, et al. BACH: Grand challenge on breast cancer histology images. Medical Image Analysis, 2019.'],
  ['CONCH', 'Lu MY, et al. A visual-language foundation model for computational pathology. Nature Medicine, 2024.'],
  ['MUSK', 'Xiang J, et al. A vision–language foundation model for precision oncology. Nature, 2025.'],
  ['UNI / UNI2-h', 'Chen RJ, et al. Towards a general-purpose foundation model for computational pathology. Nature Medicine, 2024 (UNI2-h model release, Mahmood Lab).'],
  ['H-optimus-0', 'Bioptimus, H-optimus-0 model release, 2024.'],
  ['PathGen-CLIP', 'Sun Y, et al. PathGen-1.6M: pathology image–text pairs via multi-agent collaboration, 2024.'],
  ['THUNDER benchmark', 'Marza P, et al. THUNDER tile-level histopathology benchmark; leaderboard at mics-lab.github.io/thunder.'],
  ['NCT-CRC-HE-100K', 'Kather JN, et al. Predicting survival from colorectal cancer histology slides using deep learning. PLoS Medicine, 2019.'],
  ['Quilt-1M', 'Ikezogwo WO, et al. Quilt-1M: one million image–text pairs for histopathology. NeurIPS Datasets and Benchmarks, 2023.'],
  ['HoVer-Net, TIAToolbox', 'Graham S, et al. Medical Image Analysis, 2019; Pocock J, et al. Communications Medicine, 2022; PanNuke classes (Gamper J, et al.).'],
  ['Calibration and conformal', 'Guo C, et al. (temperature scaling), ICML 2017; Sadinle M, et al. (LAC), JASA 2019; Romano Y, et al. (APS), NeurIPS 2020.'],
  ['ABMIL', 'Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance learning. ICML 2018.'],
  ['Triage cost', 'Annarumma M, et al. Automated triaging of adult chest radiographs. Radiology, 2019.'],
  ['Compute and tooling', 'NCSA Delta; ncsa/ashby-hackathon-2026 tooling (MIT); Lumen for grounded summaries.']
];
function limitationsSection(ST, ev, m, L, ctx) {
  const files = Object.entries(ev)
    .filter(([, d]) => d && typeof d === 'object')
    .map(([k, d]) => [`eval/${k}.json`, d.created_utc || '—', d.git_sha ? String(d.git_sha).slice(0, 7) + (d.git_dirty ? ' (uncommitted changes)' : '') : '—', d.synthetic ? 'synthetic' : 'pipeline output']);
  const expected = Object.keys(m.expected || {}).filter((k) => !ev[k]);
  return `<figure class="mx-fig mx-wide" data-testid="limitations"><figcaption><h3>Limitations</h3></figcaption><ul class="mx-limits">${LIMITATIONS.map((l) => `<li>${esc(l)}</li>`).join('')}</ul></figure>` +
    figDataQuality(ST, ev, L) +
    `<figure class="mx-fig"><figcaption><h3>Datasets, models and methods</h3></figcaption><dl class="mx-cite">${CITATIONS.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl></figure>` +
    `<figure class="mx-fig mx-wide"><figcaption><h3>Data provenance</h3><p class="mx-take">Every number on this page is read from these files at page load; re-running the pipeline updates the page.${expected.length ? ` Not produced in this run: ${esc(expected.map((e) => `${e}.json`).join(', '))}.` : ''}</p></figcaption>${table(['File', 'Created (UTC)', 'Git commit', 'Kind'], files)}${(m.figures || []).length ? details('figures-png', `Static PNG figures (${m.figures.length})`, `<div class="mx-pngs">${m.figures.map((f) => `<a href="${esc(ctx && ctx.assetUrl ? ctx.assetUrl(f.url) : f.url)}" target="_blank" rel="noopener">${esc(f.file)}</a>`).join('')}</div>`) : ''}</figure>`;
}

// ---- page -------------------------------------------------------------------------------------------------------------
function scope(ST, ev) {
  const M = ev.metrics || {},
    A = ev.analytics || {};
  const n = M.n ?? A.n,
    ng = M.n_groups ?? A.n_groups,
    vx = M.variants_excluded ?? A.variants_excluded;
  const prod = get(M, 'models.production') || get(A, 'models.production');
  const parts = [];
  if (isNum(n)) parts.push(`<b>${n}</b> held-out test images from <b>${esc(ng ?? '?')}</b> patient or pseudo-patient groups`);
  if (isNum(vx)) parts.push(`${vx} simulated degraded / composite variants kept out of headline numbers`);
  parts.push(`intervals are 95% cluster-bootstrap CIs over ${isNum(n) ? 'those groups' : 'patient or pseudo-patient groups'}`);
  if (prod) parts.push(`production read: ${esc(modelName(prod))}`);
  const text = parts.join('; ');
  return `<p class="mx-scope">${text.charAt(0).toUpperCase() + text.slice(1)}.</p>`;
}
function nav() {
  return `<nav class="mx-nav" aria-label="Evaluation sections"><div class="mx-nav-inner">${SECTIONS.map(([id, label]) => `<button type="button" data-mx-jump="mx-${id}">${esc(label)}</button>`).join('')}</div></nav>`;
}

export function render(ST, ctx) {
  const m = ST.metrics;
  if (!m) return '<div data-testid="metrics" class="mx"><div class="card" data-testid="state-loading"><h2>Loading evaluation results…</h2></div></div>';
  const L = layout();
  renderedWidth = L.W;
  const ev = m.eval || {};
  const S = (fn, key, title) => safe(() => fn(ST, ev, L), key, title);
  let h = '<div data-testid="metrics" class="mx">';
  h += `<div class="mx-top">${scope(ST, ev)}<button class="btn" data-metrics="refresh" data-testid="metrics-refresh">Reload results</button></div>`;
  if (m.synthetic)
    h += '<div class="mock" data-testid="metrics-synthetic"><b>Synthetic data.</b> These numbers come from generated inputs and say nothing about model performance.</div>';
  if ((m.errors || []).length) h += `<div class="flag" role="alert">Unreadable eval files: ${m.errors.map((e) => esc(e.file)).join(', ')}</div>`;
  h += nav();
  h += `<div class="mx-interp">${interpretPanel(ST, '__results', 'results')}</div>`;
  h += section('summary', 'Summary', 'The headline numbers, each with its uncertainty. The two workflow tiles come from a simulation, not measured clinical use.', safe(() => summaryTiles(ST, ev), 'metrics', 'Summary'));
  h += section('classification', 'Classification', 'Separation of invasive carcinoma from the other classes, error patterns, and model selection.',
    S(figRoc, 'analytics', 'ROC') + `<div class="mx-stack">${S(figConfusion, 'metrics', 'Confusion matrix') + S(figPr, 'analytics', 'Precision-recall')}</div>` + S(figOvr, 'analytics', 'One-vs-rest ROC') + S(figModels, 'metrics', 'Model comparison') + S(figSelection, 'model_choice', 'Model selection') + S(figThunder, 'thunder', 'Benchmark'));
  h += section('confidence', 'Confidence', 'Calibration of the probabilities and coverage of the conformal sets.',
    S(figReliability, 'calibration', 'Calibration') + S(figRiskHist, 'analytics', 'Risk histogram') + S(figCoverage, 'conformal', 'Coverage') + `<div class="mx-stack">${S(figSetSize, 'conformal', 'Set size') + S(figComposition, 'analytics', 'Set composition')}</div>`);
  h += section('explanations', 'Explanations', 'Named histologic features are associated with the classes. These checks test whether they behave like the criteria they are named after.',
    S(figConceptHeat, 'concepts_bach', 'Concept heatmap') + S(figConceptDirection, 'concepts_bach', 'Concept direction') + S(figSpotcheck, 'spotcheck', 'Spot check') + S(figNct, 'concepts', 'Cross-organ check') + S(figCav, 'concepts', 'Caption consistency') + S(figDeletion, 'deletion', 'Deletion test'));
  h += section('reliability', 'Frozen-section reliability (simulated)', 'Frozen sections were simulated by degrading microscopy images and by building half-tumour composite fields.',
    S(figDegradation, 'reliability', 'Accuracy vs degradation') + S(figReliabilityScore, 'reliability', 'Reliability score') + S(figDeferral, 'deferral', 'Deferral') + S(figMargin, 'margin', 'Margin read'));
  h += section('workflow', 'Workflow', 'Where cases land in the worklist, and what prioritizing them costs the rest of the queue in a simulated reading day.',
    S(figTiers, 'analytics', 'Tiers') + S(figReasons, 'analytics', 'Flag reasons') + S(figQueue, 'queue_sim', 'Queue simulation'));
  h += section('cells', 'Cell measurements', 'Automated nuclei measurements by class, checked against the direction pathology expects.', S(figCells, 'cells', 'Cells'));
  h += section('retrieval', 'Similar cases', 'The reference cases shown beside each case come from nearest neighbours in the training library.', S(figRetrieval, 'retrieval', 'Retrieval'));
  h += section('limitations', 'Limitations and data', 'What these results do not show, where they come from, and whom to credit.', safe(() => limitationsSection(ST, ev, m, L, ctx), 'metrics', 'Limitations'));
  return h + '</div>';
}

// ---- page behaviour (installed once; the app re-renders views as strings) ------------------------------------------------
function install() {
  if (typeof document === 'undefined' || document.__mxInstalled) return;
  document.__mxInstalled = true;
  document.addEventListener('toggle', (e) => {
    const d = e.target;
    if (!d || !d.matches || !d.matches('details[data-mx-open]')) return;
    if (d.open) OPEN.add(d.dataset.mxOpen);
    else OPEN.delete(d.dataset.mxOpen);
  }, true);
  document.addEventListener('click', (e) => {
    const b = e.target && e.target.closest && e.target.closest('[data-mx-jump]');
    if (!b) return;
    const el = document.getElementById(b.dataset.mxJump);
    if (!el) return;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    const h = el.querySelector('h2');
    if (h) {
      h.setAttribute('tabindex', '-1');
      h.focus({ preventScroll: true });
    }
  });
  let raf = 0;
  const markActive = () => {
    raf = 0;
    const navEl = document.querySelector('.mx-nav');
    if (!navEl) return;
    const offset = navEl.getBoundingClientRect().bottom + 24; // the nav sits below the sticky header
    let current = null;
    for (const [id] of SECTIONS) {
      const s = document.getElementById(`mx-${id}`);
      if (s && s.getBoundingClientRect().top - offset <= 0) current = `mx-${id}`;
    }
    navEl.querySelectorAll('[data-mx-jump]').forEach((btn) => {
      if (btn.dataset.mxJump === current) {
        if (btn.getAttribute('aria-current') !== 'true') {
          btn.setAttribute('aria-current', 'true');
          // keep the active section visible in the horizontally scrolling nav (phones) without moving the page
          const left = btn.offsetLeft - navEl.clientWidth / 2 + btn.offsetWidth / 2;
          navEl.scrollTo({ left: Math.max(0, left), behavior: 'auto' });
        }
      } else btn.removeAttribute('aria-current');
    });
  };
  window.addEventListener('scroll', () => {
    if (!raf) raf = requestAnimationFrame(markActive);
  }, { passive: true });
  // Charts are drawn for the measured width; when it changes materially, ask the app to re-render this route.
  let timer = 0;
  const onResize = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (!document.querySelector('[data-testid=metrics].mx')) return;
      const app = document.getElementById('app');
      const w = app ? app.clientWidth : 0;
      if (w && renderedWidth && Math.abs(w - renderedWidth) > 24) window.dispatchEvent(new HashChangeEvent('hashchange'));
    }, 220);
  };
  if (typeof ResizeObserver !== 'undefined') {
    const start = () => {
      const app = document.getElementById('app');
      if (app) new ResizeObserver(onResize).observe(app);
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
    else start();
  } else window.addEventListener('resize', onResize);
}
install();
