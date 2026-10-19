// Image evidence (plan §8.1 "Where"): the display JPEG with an absolutely positioned SVG in source-pixel coordinates
// (the downscaled JPEG scales for free) and an HTML region grid on top for hover / focus / click.
// Biopsy: per-tile heatmap (exact region contribution / risk / a concept's tile z / ABMIL attention) with highlighted
// tiles. The contribution layer, when the case has one, is the default: each region's additive term of the production
// model's centered calibrated logit (prediction.attribution), pink toward invasive and blue away.
// Surgical: the four margins as per-segment strips (BUILD_PLAN §A12), a specimen diagram and per-margin score bars.
// Every number shown comes from the case JSON; nothing here is a diagnosis.
import { esc, MARGINS, num } from '../adapter.js';

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const Z_FULL = 3; // concept layers: z <= 0 is not shaded, z >= Z_FULL is fully shaded (display mapping only)
export const cap = (s) => String(s || '').replace(/^./, (c) => c.toUpperCase());
const f2 = (x) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? '—' : Number(x).toFixed(2));
/** Tile probability as a percentage, like every other probability on the page: never 100% / 0%. */
const fp2 = (x) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? '—' : Number(x) >= 0.995 ? '>99%' : Number(x) < 0.005 ? '<1%' : Math.round(Number(x) * 100) + '%');
const sz = (x) => (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(2);
const isNum = (x) => x !== null && x !== undefined && Number.isFinite(Number(x));
/** Concepts whose AI spot check failed (set by the case view from the evaluation; empty elsewhere). */
const failedSet = (v) => (v && v.spotFailed instanceof Set ? v.spotFailed : new Set());
const FAILED_NOTE = 'AI spot check failed';

/** True when the case carries an exact region decomposition (prediction.attribution + tile contributions). */
export function hasContributions(v) {
  return !!v.attribution && v.tiles.some((t) => isNum(t.contribution));
}

/** Largest |contribution| over the case's regions (the contribution layer's full-shade value). */
export function maxAbsContribution(v) {
  return Math.max(0, ...v.tiles.filter((t) => isNum(t.contribution)).map((t) => Math.abs(Number(t.contribution))));
}

/** "Region 7 · +0.42 toward invasive" (the contribution layer's tooltip line). */
export function contributionLine(idx, c) {
  if (!isNum(c)) return `Region ${idx} · contribution not available`;
  const x = Number(c);
  return `Region ${idx} · ${sz(x)} ${x > 0 ? 'toward' : x < 0 ? 'away from' : 'neutral for'} invasive`;
}

/** True when at least one region carries a tile P(Invasive) (the CONCH tile probe; not computed for TCGA specimens). */
export function hasRisk(v) {
  return v.tiles.some((t) => isNum(t.risk));
}

/** Heatmap layers available for a case: exact contribution (first, the default) when present, risk when any region has
 * a value, every concept with tile values, attention when present. */
export function layers(v) {
  const out = hasContributions(v) ? [{ id: 'contribution', label: 'Region contribution (exact)', kind: 'contribution' }] : [];
  if (hasRisk(v) || !out.length) out.push({ id: 'risk', label: 'Overall risk (tile P(Invasive))', kind: 'risk' });
  const tileConcepts = new Set(v.tiles.flatMap((t) => Object.keys(t.concepts || {})));
  const failed = failedSet(v);
  for (const c of v.concepts)
    if (tileConcepts.has(c.id)) out.push({ id: c.id, label: c.label + (failed.has(c.id) ? ' (spot check failed)' : ''), kind: 'concept', failed: failed.has(c.id) });
  if (v.tiles.some((t) => t.attention !== null && t.attention !== undefined))
    out.push({ id: 'attention', label: 'ABMIL attention', kind: 'attention' });
  return out;
}

export function layerOf(v, id) {
  return layers(v).find((l) => l.id === id) || layers(v)[0];
}

function tileValue(t, layer, maxAttention, maxContribution) {
  if (layer === 'contribution') return isNum(t.contribution) && maxContribution > 0 ? clamp01(Math.abs(Number(t.contribution)) / maxContribution) : 0;
  if (layer === 'risk') return clamp01(num(t.risk));
  if (layer === 'attention') return maxAttention > 0 ? clamp01((t.attention || 0) / maxAttention) : 0;
  const z = t.concepts ? t.concepts[layer] : undefined;
  return z === undefined || z === null ? 0 : clamp01(z / Z_FULL);
}

function conceptLabel(v, id) {
  // measured cell features are Why rows too (no concept entry): use their configured label, not the raw id
  return (v.concepts.find((c) => c.id === id) || (v.why || []).find((r) => r.id === id) || {}).label || id;
}

/** Plain-text + HTML description of one region, used by the tooltip and the accessible name. */
export function regionInfo(v, hm, idx) {
  const cols = num(v.image.grid[1]);
  const t = v.tiles.find((x) => x.idx === idx);
  const where = `row ${Math.floor(idx / cols) + 1}, column ${(idx % cols) + 1}`;
  const linked = (hm.hl || []).includes(idx) && hm.why ? conceptLabel(v, hm.why) : null;
  const cellTiles = v.cells && v.cells.status === 'ok' ? (v.cells.tiles || []).map((c) => c.idx) : [];
  if (!t) {
    const text = `Region ${idx} (${where}): failed tissue quality control, not used by the model.`;
    return {
      text,
      html:
        `<b>Region ${idx}</b><span class="tip-sub">${esc(where)}</span>` +
        '<span class="tip-qc bad">Failed tissue QC</span><span>Mostly background or too little tissue, so the model did not read it.</span>'
    };
  }
  const zs = Object.entries(t.concepts || {})
    .filter(([, z]) => Number.isFinite(Number(z)))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  const layer = layerOf(v, hm.sel);
  const failed = failedSet(v);
  const zRow = ([id, z]) =>
    failed.has(id)
      ? `<span class="tip-row muted"><span>${esc(conceptLabel(v, id))} <i>${FAILED_NOTE}</i></span><b>${esc(sz(z))}</b></span>`
      : `<span class="tip-row"><span>${esc(conceptLabel(v, id))}</span><b>${esc(sz(z))}</b></span>`;
  if (layer.kind === 'contribution') {
    const line = contributionLine(idx, t.contribution);
    const text =
      `${line} (${where}). ` +
      (strongestText(v, zs) ? `Strongest concept scores: ${strongestText(v, zs)}. ` : '') +
      (isNum(t.risk) ? `Tile P(Invasive) ${fp2(t.risk)}. ` : '') +
      'Passed tissue QC.' +
      (linked ? ` Linked to ${linked}.` : '');
    const html =
      `<b>${esc(line)}</b><span class="tip-sub">${esc(where)}</span>` +
      '<span class="tip-sub">Additive term of the model\'s invasive score (centered calibrated logit)</span>' +
      (zs.length ? `<span class="tip-h">Strongest concept scores (z vs. reference regions)</span>${zs.map(zRow).join('')}` : '') +
      (isNum(t.risk) ? `<span class="tip-row"><span>Tile P(Invasive)</span><b>${fp2(t.risk)}</b></span>` : '') +
      `<span class="tip-qc ok">Passed tissue QC</span>` +
      (linked ? `<span class="tip-link">Linked to: ${esc(linked)}</span>` : '');
    return { text, html };
  }
  const cur =
    layer.kind === 'concept' && t.concepts && t.concepts[layer.id] !== undefined
      ? { k: `${conceptLabel(v, layer.id)} (z)`, v: sz(t.concepts[layer.id]) }
      : layer.kind === 'attention' && t.attention !== null && t.attention !== undefined
        ? { k: 'ABMIL attention', v: Number(t.attention).toFixed(3) }
        : null;
  const strongest = zs.map(([id, z]) => `${conceptLabel(v, id)} ${sz(z)}${failed.has(id) ? ` (${FAILED_NOTE})` : ''}`);
  const text =
    `Region ${idx} (${where}). ` +
    (isNum(t.risk) ? `Tile P(Invasive) ${fp2(t.risk)}. ` : '') +
    (cur ? `${cur.k} ${cur.v}. ` : '') +
    (strongest.length ? `Strongest concept scores: ${strongest.join('; ')}. ` : '') +
    'Passed tissue QC.' +
    (linked ? ` Linked to ${linked}.` : '');
  const html =
    `<b>Region ${idx}</b><span class="tip-sub">${esc(where)}</span>` +
    (isNum(t.risk)
      ? `<span class="tip-row"><span>Tile P(Invasive)</span><b>${fp2(t.risk)}</b></span>`
      : '') +
    (cur ? `<span class="tip-row"><span>${esc(cur.k)}</span><b>${esc(cur.v)}</b></span>` : '') +
    (strongest.length ? `<span class="tip-h">Strongest concept scores (z vs. reference regions)</span>${zs.map(zRow).join('')}` : '') +
    `<span class="tip-qc ok">Passed tissue QC</span>` +
    (linked ? `<span class="tip-link">Linked to: ${esc(linked)}</span>` : '') +
    (cellTiles.includes(idx) ? '<span class="tip-sub">Cell measurements available in the Cells tab</span>' : '');
  return { text, html };
}

function strongestText(v, zs) {
  const failed = failedSet(v);
  return zs.map(([id, z]) => `${conceptLabel(v, id)} ${sz(z)}${failed.has(id) ? ` (${FAILED_NOTE})` : ''}`).join('; ');
}

/** The SVG overlay (fills, QC hatch, highlight outlines). hlClass lets the full-screen viewer avoid `rect.hl`. */
export function overlaySvg(v, hm, { hlClass = 'hl', idPrefix = 'hm', opacity = null } = {}) {
  const w = num(v.image.w),
    h = num(v.image.h),
    tile = num(v.image.tile);
  const [rows, cols] = v.image.grid.map(num);
  const maxA = Math.max(0, ...v.tiles.map((t) => t.attention || 0));
  const maxC = maxAbsContribution(v);
  const byIdx = new Map(v.tiles.map((t) => [t.idx, t]));
  const hl = new Set(hm.hl || []);
  const op = opacity === null ? hm.opacity : opacity;
  const layer = layerOf(v, hm.sel);
  let rects = '';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c,
        t = byIdx.get(idx);
      if (!t) {
        rects += `<rect class="hm-drop" style="fill:url(#${idPrefix}-hatch)" data-tile="${idx}" x="${c * tile}" y="${r * tile}" width="${tile}" height="${tile}"/>`;
        continue;
      }
      const val = tileValue(t, layer.id, maxA, maxC);
      const sign = layer.kind === 'contribution' ? (num(t.contribution) < 0 ? ' neg' : ' pos') : '';
      rects += `<rect class="hm-tile${sign}" data-testid="hm-tile" data-tile="${idx}" data-v="${val.toFixed(4)}" x="${num(t.x)}" y="${num(t.y)}" width="${tile}" height="${tile}" fill-opacity="${(val * op).toFixed(3)}"/>`;
    }
  }
  for (const idx of hl) {
    const t = byIdx.get(idx);
    if (!t) continue;
    rects += `<rect class="${hlClass}-halo" x="${num(t.x)}" y="${num(t.y)}" width="${tile}" height="${tile}"/>`;
    rects += `<rect class="${hlClass}" data-hl="${idx}" x="${num(t.x)}" y="${num(t.y)}" width="${tile}" height="${tile}"/>`;
  }
  const hatch =
    `<defs><pattern id="${idPrefix}-hatch" width="48" height="48" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">` +
    '<rect width="48" height="48" class="hatch-bg"/><line x1="0" y1="0" x2="0" y2="48" class="hatch-ln"/></pattern></defs>';
  return (
    `<svg class="hm-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true" data-kind="${layer.kind}">` +
    hatch +
    rects +
    '</svg>'
  );
}

/** Focusable region grid (HTML buttons positioned in % of the image). Hover/focus shows a tooltip (case_ui.js). */
export function regionGrid(v, hm, { scope = 'case' } = {}) {
  const w = num(v.image.w),
    h = num(v.image.h),
    tile = num(v.image.tile);
  const [rows, cols] = v.image.grid.map(num);
  const byIdx = new Map(v.tiles.map((t) => [t.idx, t]));
  const hl = new Set(hm.hl || []);
  const layer = layerOf(v, hm.sel);
  const maxA = Math.max(0, ...v.tiles.map((t) => t.attention || 0));
  const valueText = (t) => {
    if (!t) return 'QC';
    if (layer.kind === 'contribution') return isNum(t.contribution) ? sz(Number(t.contribution)) : '—';
    if (layer.kind === 'risk') return fp2(t.risk);
    if (layer.kind === 'attention') return maxA > 0 ? f2((t.attention || 0) / maxA) : '—';
    const z = t.concepts ? t.concepts[layer.id] : undefined;
    return z === undefined || z === null ? '—' : sz(Number(z)).replace('.00', '.0');
  };
  let out = '';
  const first = hl.size ? Math.min(...hl) : 0; // roving tabindex: one tab stop, arrow keys move between regions
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const info = regionInfo(v, hm, idx);
      const x = c * tile,
        y = r * tile;
      const style = `left:${((x / w) * 100).toFixed(3)}%;top:${((y / h) * 100).toFixed(3)}%;width:${((Math.min(tile, w - x) / w) * 100).toFixed(3)}%;height:${((Math.min(tile, h - y) / h) * 100).toFixed(3)}%`;
      out +=
        `<button type="button" class="rg${byIdx.has(idx) ? '' : ' drop'}${hl.has(idx) ? ' on' : ''}" data-rg="${idx}" data-rg-scope="${scope}" tabindex="${idx === first ? 0 : -1}" style="${style}" ` +
        `aria-label="${esc(info.text)}" data-tip="${esc(info.html)}"><span class="rg-n" aria-hidden="true">${idx}</span><span class="rg-v" aria-hidden="true">${esc(valueText(byIdx.get(idx)))}</span></button>`;
    }
  }
  return `<div class="rg-grid" role="group" data-cols="${cols}" aria-label="Image regions, ${rows} rows by ${cols} columns. Arrow keys move between regions; Enter zooms in.">${out}</div>`;
}

export function heatmap(v, hm, assetUrl) {
  if (!v.image) return '<div class="note">No display image.</div>';
  const w = num(v.image.w),
    h = num(v.image.h);
  return (
    `<div class="hm cv-hm" style="aspect-ratio:${w}/${h}"><img src="${esc(assetUrl(v.image.src))}" alt="Display image for ${esc(v.id)} with a tile heatmap overlay" data-testid="heatmap-img" data-cv-img decoding="async">` +
    overlaySvg(v, hm) +
    regionGrid(v, hm) +
    `<button type="button" class="hm-expand" data-cv-open="viewer" data-testid="viewer-open" aria-label="Open the full-screen image viewer"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 8V3h5M17 8V3h-5M3 12v5h5M17 12v5h-5"/></svg><span>Full screen</span></button>` +
    '</div>'
  );
}

/** Legend with units for the active layer, plus the QC-hatch and highlight swatches. */
export function heatmapLegend(v, hm) {
  const layer = layerOf(v, hm.sel);
  const maxC = maxAbsContribution(v);
  const scale =
    layer.kind === 'contribution'
      ? { title: 'Region contribution (centered logit)', lo: `${sz(-maxC)} away`, mid: '0', hi: `${sz(maxC)} toward`, note: "Each region's push toward (pink) or away from (blue) invasive carcinoma. The pushes plus a constant add up exactly to the model's invasive score." }
      : layer.kind === 'risk'
      ? { title: 'Tile P(Invasive)', lo: '0%', mid: '50%', hi: '100%', note: 'Per-region probability from the tile-level probe. Shading is proportional to the probability; it localises the read, it does not grade a region.' }
      : layer.kind === 'attention'
        ? { title: 'ABMIL attention', lo: '0', mid: '', hi: 'case max', note: 'Relative attention weight, scaled to the most attended region in this case.' }
        : {
            title: `${conceptLabel(v, layer.id)}: concept z`,
            lo: '≤ 0',
            mid: '+1.5',
            hi: `≥ +${Z_FULL}`,
            note:
              'z = concept score of the region relative to reference (training) regions. Regions at or below the reference average are not shaded.' +
              (layer.failed ? " This feature failed its AI spot check (its top regions did not reliably show it), so read the shading as an association only, not as the feature's location." : '')
          };
  const linked = hm.hl && hm.hl.length;
  return (
    `<div class="cv-legend" data-testid="heatmap-legend" data-kind="${layer.kind}">` +
    `<div class="lg-scale"><span class="lg-title">${esc(scale.title)}</span><span class="lg-bar" aria-hidden="true"></span>` +
    `<span class="lg-ticks"><span>${esc(scale.lo)}</span><span>${esc(scale.mid)}</span><span>${esc(scale.hi)}</span></span></div>` +
    '<div class="lg-keys">' +
    '<span class="lg-key"><i class="sw-hatch" aria-hidden="true"></i><span>Hatched: failed tissue QC, not read by the model</span></span>' +
    `<span class="lg-key${linked ? '' : ' dim'}"><i class="sw-hl" aria-hidden="true"></i><span>${linked ? `Outlined: regions linked to <b>${esc(conceptLabel(v, hm.why) || 'the selection')}</b> (${hm.hl.map(esc).join(', ')})` : 'Outlined: regions linked to a selected feature'}</span></span>` +
    '</div>' +
    `<p class="lg-note">${esc(scale.note)} Hover or focus a region for its values; click it to zoom in.</p></div>`
  );
}

export function heatmapControls(v, hm) {
  const ls = layers(v);
  const cur = layerOf(v, hm.sel).id;
  const group = (kind, label) => {
    const opts = ls
      .filter((l) => l.kind === kind)
      .map((l) => `<option value="${esc(l.id)}"${l.id === cur ? ' selected' : ''}>${esc(l.label)}</option>`)
      .join('');
    return opts ? (label ? `<optgroup label="${esc(label)}">${opts}</optgroup>` : opts) : '';
  };
  return (
    `<div class="hmctl cv-hmctl"><label>Layer <select data-testid="heatmap-select" data-hm="sel" aria-label="Heatmap layer">${group('contribution')}${group('risk')}${group('concept', 'Concept scores (z)')}${group('attention', 'Model attention')}</select></label>` +
    `<label>Overlay opacity <input type="range" min="0" max="1" step="0.05" value="${hm.opacity}" data-testid="heatmap-opacity" data-hm="opacity" aria-label="Overlay opacity"></label></div>`
  );
}

// ---- surgical margins -------------------------------------------------------------------------------------------------
// The margin probe saturates (scores like 0.99976 vs a 0.99982 threshold), so scores are shown on a natural log-odds
// axis clipped to ±LOGIT_CLIP; the raw probability is only shown in tooltips, formatted ">0.9999" rather than "1.0".
export const LOGIT_CLIP = 12;
export function logit(p) {
  const q = Math.min(1 - 1e-12, Math.max(1e-12, num(p)));
  return Math.max(-LOGIT_CLIP, Math.min(LOGIT_CLIP, Math.log(q / (1 - q))));
}
export function fmtProb(p) {
  const x = num(p);
  if (x >= 0.9999) return '>0.9999';
  if (x <= 0.0001) return '<0.0001';
  return x >= 0.99 || x < 0.01 ? x.toFixed(4) : x.toFixed(3);
}
export const fmtLogit = (p) => {
  const l = logit(p);
  return (l >= LOGIT_CLIP ? '≥ +' : l <= -LOGIT_CLIP ? '≤ −' : l > 0 ? '+' : l < 0 ? '−' : '±') + Math.abs(l).toFixed(1);
};

/** Per-margin status against the configured threshold; `near` = within 1 log-odds unit below it. */
export function marginStatus(m, name) {
  const p = m.perMargin[name];
  if (p === undefined || p === null) return 'unknown';
  if (m.involved.includes(name)) return 'involved';
  if (m.threshold !== null && num(p) > m.threshold) return 'involved';
  if (m.threshold !== null && logit(p) > logit(m.threshold) - 1) return 'near';
  return 'clear';
}
export const STATUS_TEXT = { involved: 'above threshold', near: 'just below threshold', clear: 'below threshold', unknown: 'not scored' };
const SHORT_STATUS = { involved: 'above', near: 'just below', clear: 'below', unknown: 'not scored' };

function segClass(m, s) {
  const thr = m.threshold;
  const hot = thr !== null ? s.p > thr : m.involved.includes(s.margin) && s.p >= m.perMargin[s.margin];
  const warm = !hot && thr !== null && logit(s.p) > logit(thr) - 1;
  return { hot, cls: hot ? 'hot' : warm ? 'warm' : 'cool' };
}

/** The image with every margin segment drawn as an inked-border strip, coloured by threshold status. */
export function marginOverlay(v, assetUrl, { names = null } = {}) {
  if (!v.image) return '<div class="note">No display image.</div>';
  const w = num(v.image.w),
    h = num(v.image.h);
  const m = v.margin;
  let rects = '';
  if (m) {
    const thr = m.threshold;
    for (const s of m.segments) {
      const { hot, cls } = segClass(m, s);
      rects +=
        `<rect class="mseg ${cls}" data-testid="margin-seg" data-margin="${esc(s.margin)}" data-p="${num(s.p)}" data-involved="${hot && m.involved.includes(s.margin)}" x="${num(s.x)}" y="${num(s.y)}" width="${num(s.size)}" height="${num(s.size)}">` +
        `<title>${esc(cap(s.margin))} margin segment · tumour signal ${esc(fmtLogit(s.p))} log-odds (p ${esc(fmtProb(s.p))})${thr !== null ? ` · threshold ${esc(fmtLogit(thr))}` : ''} · simulated</title></rect>`;
    }
  }
  const n = names || {};
  const side = (pos, name) =>
    names && name ? `<span class="ms-side ${pos} ${m ? marginStatus(m, name) : ''}">${esc(cap(name))}</span>` : '';
  return (
    `<div class="ms-frame${names ? ' labelled' : ''}">${side('top', n.top)}${side('bottom', n.bottom)}${side('left', n.left)}${side('right', n.right)}` +
    `<div class="hm cv-hm surg" style="aspect-ratio:${w}/${h}"><img src="${esc(assetUrl(v.image.src))}" alt="Display image for ${esc(v.id)} with margin strips (simulated frozen section)" data-testid="surgical-img" data-cv-img decoding="async">` +
    `<svg class="hm-svg msvg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">${rects}</svg>` +
    `<button type="button" class="hm-expand" data-cv-open="viewer" data-testid="viewer-open" aria-label="Open the full-screen image viewer"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 8V3h5M17 8V3h-5M3 12v5h5M17 12v5h-5"/></svg><span>Full screen</span></button></div></div>` +
    (m
      ? `<div class="mlab" data-testid="margin-labels">${MARGINS.map((name) => {
          const st = marginStatus(m, name);
          // "involved" is a call; only a tumor_at_margin suggestion uses it. Otherwise (e.g. defer) report the raw read.
          const call = m.call === 'tumor_at_margin';
          const word = st === 'involved' ? (call ? 'involved' : 'above threshold') : STATUS_TEXT[st];
          const len = m.lengths && m.lengths[name] && st === 'involved' ? `≈${Math.round(m.lengths[name])} µm ${word}` : word;
          return `<div class="${call && m.involved.includes(name) ? 'inv' : ''} ${st}" data-testid="margin-label-${name}" title="${esc(cap(name))}: p ${esc(fmtProb(m.perMargin[name] ?? 0))}${call ? '' : ' (raw read, not a call)'}"><b>${cap(name)}</b><span>${esc(len)}</span></div>`;
        }).join('')}</div>`
      : '')
  );
}

/** Schematic specimen: the four named margins as thick sides coloured by status against the threshold. */
export function specimenDiagram(v, names) {
  const m = v.margin;
  if (!m) return '';
  const st = (name) => marginStatus(m, name);
  const n = names || { top: 'superior', bottom: 'inferior', left: 'medial', right: 'lateral' };
  const W = 280,
    H = 210,
    x0 = 52,
    y0 = 34,
    x1 = W - 52,
    y1 = H - 34;
  const side = (name, a, b, c, d) =>
    `<line class="sp-side ${st(name)}" x1="${a}" y1="${b}" x2="${c}" y2="${d}"><title>${esc(cap(name))}: ${esc(STATUS_TEXT[st(name)])}</title></line>`;
  const label = (name, x, y, rot = 0) =>
    `<text class="sp-lbl ${st(name)}" x="${x}" y="${y}" text-anchor="middle"${rot ? ` transform="rotate(${rot} ${x} ${y})"` : ''}>${esc(cap(name))}${st(name) === 'involved' ? ' ●' : ''}</text>`;
  const desc = MARGINS.map((name) => `${cap(name)} ${STATUS_TEXT[st(name)]}`).join('; ');
  return (
    `<svg class="specimen" viewBox="0 0 ${W} ${H}" role="img" aria-label="Specimen diagram (simulated): ${esc(desc)}" data-testid="specimen-diagram">` +
    `<rect class="sp-body" x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" rx="8"/>` +
    `<text class="sp-mid" x="${W / 2}" y="${H / 2 - 3}" text-anchor="middle">Specimen field</text>` +
    `<text class="sp-mid sub" x="${W / 2}" y="${H / 2 + 13}" text-anchor="middle">borders stand in for inked margins</text>` +
    side(n.top, x0, y0, x1, y0) +
    side(n.bottom, x0, y1, x1, y1) +
    side(n.left, x0, y0, x0, y1) +
    side(n.right, x1, y0, x1, y1) +
    label(n.top, W / 2, y0 - 12) +
    label(n.bottom, W / 2, y1 + 24) +
    label(n.left, x0 - 16, H / 2, -90) +
    label(n.right, x1 + 16, H / 2, 90) +
    '</svg>'
  );
}

/** Per-margin tumour signal on a log-odds axis with the case's threshold marked. */
export function marginBars(v) {
  const m = v.margin;
  if (!m) return '';
  const lo = -LOGIT_CLIP,
    hi = LOGIT_CLIP;
  const pos = (l) => ((Math.max(lo, Math.min(hi, l)) - lo) / (hi - lo)) * 100;
  const thr = m.threshold;
  const ticks = [-12, -6, 0, 6, 12].map((t) => `<span style="left:${pos(t).toFixed(2)}%">${t > 0 ? '+' + t : t === 0 ? '0' : '−' + Math.abs(t)}</span>`).join('');
  const rows = MARGINS.map((name) => {
    const p = m.perMargin[name];
    const s = marginStatus(m, name);
    const has = p !== undefined && p !== null;
    const l = has ? logit(p) : 0;
    const len = m.lengths && m.lengths[name] ? `≈${Math.round(m.lengths[name])} µm of ${name} margin` : '';
    const tip = has ? `${cap(name)}: tumour signal ${fmtLogit(p)} log-odds (p ${fmtProb(p)})${thr !== null ? `; threshold ${fmtLogit(thr)} (p ${fmtProb(thr)})` : ''}` : `${cap(name)}: not scored`;
    return (
      `<div class="mb-row ${s}" data-testid="margin-bar-${name}" data-status="${s}" title="${esc(tip)}">` +
      `<span class="mb-name">${esc(cap(name))}</span>` +
      `<span class="mb-track" aria-hidden="true"><i style="left:${pos(Math.min(0, l)).toFixed(2)}%;width:${Math.abs(pos(l) - pos(0)).toFixed(2)}%"></i><b class="mb-zero" style="left:${pos(0)}%"></b>${thr !== null ? `<b class="mb-thr" style="left:${pos(logit(thr)).toFixed(2)}%"></b>` : ''}</span>` +
      `<span class="mb-val"><b>${has ? esc(fmtLogit(p)) : '—'}</b><span class="mb-st">${esc(SHORT_STATUS[s])}</span></span>` +
      `<span class="sr-only">${esc(tip)}. ${esc(STATUS_TEXT[s])}. ${esc(len)}</span></div>`
    );
  }).join('');
  return (
    `<div class="mbars" data-testid="margin-bars"><div class="mb-head"><span>Tumour signal at margin (log-odds) · involved when above the threshold</span>${thr !== null ? `<span class="mb-key"><b class="mb-thr-key"></b>threshold ${esc(fmtLogit(thr))}</span>` : ''}</div>${rows}` +
    `<div class="mb-axis" aria-hidden="true"><span class="mb-name"></span><span class="mb-ticks">${ticks}</span><span class="mb-val"></span></div></div>` +
    `<p class="lg-note">Each margin's signal is its highest-scoring border segment from the tile tumour probe, on a log-odds axis (clipped at ±${LOGIT_CLIP}) so near-1 probabilities stay apart; hover a row for the raw probability. ${
      thr !== null ? 'The threshold was tuned on benign calibration images to keep false tumour-at-margin calls rare.' : 'No threshold was supplied with this case.'
    } Simulated: not a validated margin assessment.</p>`
  );
}
