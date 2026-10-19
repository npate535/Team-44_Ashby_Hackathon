// PathQueue SVG chart primitives for the Model evaluation dashboard. Dependency-free and offline: every function
// returns an HTML/SVG string (the app renders views as strings), sized for the width the caller passes so text is
// drawn at ~1:1 scale (viewBox = width; CSS width 100%).
//
// Conventions (dataviz skill): series colours are the .mx-scoped tokens --cx1 (hematoxylin indigo), --cx2 (eosin
// coral), --cx3 (teal), validated all-pairs for CVD in light and dark (metrics.css); text always uses ink tokens;
// 2px lines, r>=4 markers with a surface ring, hairline solid grid, dashed lines only for reference/target lines;
// legends for >= 2 series; long labels are measured and wrapped, never clipped. Every value mark carries data-tip
// (plain text, first line = value) for the shared tooltip layer below; every chart is role="img" with an aria-label
// and is keyboard focusable (arrow keys step through its marks). Data tables are the caller's job (<details>).
import { esc } from './adapter.js';

export const COLOR = {
  1: 'var(--cx1)',
  2: 'var(--cx2)',
  3: 'var(--cx3)',
  muted: 'var(--cx-mute)',
  ink: 'var(--ink)'
};
const FONT = 11; // px, axis + label text
const CHAR = 0.58; // average glyph advance in em for the UI sans (conservative, so labels are never clipped)
let seq = 0;
const uid = (p) => `${p}${++seq}`;

// ---- numbers and text ----------------------------------------------------------------------------------------------
export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export function fmtNum(v, digits = 2) {
  if (!isNum(v)) return '—';
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString('en-US');
  return v.toFixed(digits);
}
/** Precision by magnitude: 1,235 / 54.7 / 0.17. */
export function fmtSmart(v) {
  if (!isNum(v)) return '—';
  const a = Math.abs(v);
  return fmtNum(v, a >= 100 ? 0 : a >= 10 ? 1 : 2);
}
export const fmtPct = (v, digits = 0) => (isNum(v) ? (v * 100).toFixed(digits) + '%' : '—');
// The estimate undercounts wide glyphs ("%", capitals, digits in bold), which left the deferral y-axis title about 1.5 px
// from its tick labels. In a browser the text is also measured (semibold, the chart font, once the fonts are ready) and
// the larger width wins, so nothing that fitted before gets narrower.
let measureCtx = null;
let measureFont = null;
const measured = new Map();
function measuredWidth(s, size) {
  if (typeof document === 'undefined' || !document.fonts || document.fonts.status !== 'loaded') return 0;
  const key = `${size}|${s}`;
  if (!measured.has(key)) {
    try {
      measureCtx = measureCtx || document.createElement('canvas').getContext('2d');
      measureFont = measureFont || getComputedStyle(document.body).fontFamily;
      measureCtx.font = `600 ${size}px ${measureFont}`;
      measured.set(key, measureCtx.measureText(s).width);
    } catch (_) {
      measured.set(key, 0);
    }
  }
  return measured.get(key);
}
export function textWidth(s, size = FONT) {
  const str = String(s);
  return Math.max(str.length * size * CHAR, measuredWidth(str, size));
}
/** Greedy word wrap into lines no wider than maxPx (a single long word is hard-split). */
export function wrap(s, maxPx, size = FONT) {
  const words = String(s).split(/\s+/).filter(Boolean);
  const maxChars = Math.max(4, Math.floor(maxPx / (size * CHAR)));
  const lines = [];
  let cur = '';
  for (let w of words) {
    while (w.length > maxChars) {
      if (cur) {
        lines.push(cur);
        cur = '';
      }
      lines.push(w.slice(0, maxChars - 1) + '-');
      w = w.slice(maxChars - 1);
    }
    if (!cur) cur = w;
    else if ((cur + ' ' + w).length <= maxChars) cur += ' ' + w;
    else {
      lines.push(cur);
      cur = w;
    }
  }
  if (cur) lines.push(cur);
  return lines.length ? lines : [''];
}
// Wrapped lines are separate tspans; a zero-size space between them keeps copied text readable ("512-px regions",
// not "512-pxregions") without moving any line (it has no width, so start, middle and end anchoring are unchanged).
function textLines(lines, x, y, { anchor = 'start', size = FONT, cls = 'cx-t', lh = 1.25, weight = null } = {}) {
  const w = weight ? ` font-weight="${weight}"` : '';
  return `<text class="${cls}" x="${r1(x)}" y="${r1(y)}" font-size="${size}" text-anchor="${anchor}"${w}>${lines
    .map((l, i) => `<tspan x="${r1(x)}" dy="${i ? size * lh : 0}">${esc(l)}</tspan>`)
    .join('<tspan font-size="0"> </tspan>')}</text>`;
}
const r1 = (v) => Math.round(v * 10) / 10;

/** Nice tick values covering [min, max]. */
export function ticks(min, max, count = 5) {
  if (!isNum(min) || !isNum(max) || max <= min) return [min];
  const out = niceTicks(min, max, count);
  return out.length >= 3 ? out : niceTicks(min, max, count * 2);
}
function niceTicks(min, max, count) {
  const raw = (max - min) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let t = Math.ceil(min / step - 1e-9) * step; t <= max + step * 1e-6; t += step) out.push(+t.toFixed(10));
  return out;
}
/** Decimals that show a tick step exactly (0.25 -> 2, 0.2 -> 1, 5 -> 0). */
function decimalsFor(step) {
  if (!isNum(step) || step <= 0) return 2;
  for (let d = 0; d <= 4; d++) if (Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) < 1e-6) return d;
  return 4;
}

function tipAttr(lines) {
  const text = (Array.isArray(lines) ? lines : [lines]).filter((l) => l !== null && l !== undefined && l !== '').join('\n');
  return text ? ` data-tip="${esc(text)}"` : '';
}
function svgOpen(w, h, aria, cls = '') {
  return `<svg class="chart cx ${cls}" viewBox="0 0 ${r1(w)} ${r1(h)}" width="${r1(w)}" height="${r1(h)}" style="max-width:${r1(w)}px" role="img" aria-label="${esc(aria)}" tabindex="0" focusable="true" preserveAspectRatio="xMinYMin meet">`;
}
/** Legend chips (HTML). kind: 'line' | 'rect' | 'dot' | 'dash'. */
export function legend(items) {
  if (!items || items.length < 1) return '';
  return `<ul class="cx-legend">${items
    .map(
      (it) =>
        `<li><span class="cx-key cx-key-${it.kind || 'line'}" style="--k:${it.color || COLOR[1]}"></span>${esc(it.name)}</li>`
    )
    .join('')}</ul>`;
}

// ---- axes ------------------------------------------------------------------------------------------------------------
function yAxis({ L, R, W, T, B, H, ys, sy, fmt, label }) {
  let g = '';
  for (const t of ys) {
    const y = sy(t);
    g += `<line class="cx-grid" x1="${L}" x2="${W - R}" y1="${r1(y)}" y2="${r1(y)}"/>`;
    g += `<text class="cx-tick" x="${L - 6}" y="${r1(y + 3.5)}" text-anchor="end" font-size="${FONT}">${esc(fmt(t))}</text>`;
  }
  if (label) {
    const cy = (T + H - B) / 2;
    g += `<text class="cx-axis-label" x="12" y="${r1(cy)}" text-anchor="middle" font-size="${FONT}" transform="rotate(-90 12 ${r1(cy)})">${esc(label)}</text>`;
  }
  return g;
}
function xAxis({ L, R, W, H, B, xs, sx, fmt, label, base = true }) {
  let g = base ? `<line class="cx-axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>` : '';
  for (const t of xs) {
    const text = fmt(t),
      tw = textWidth(text),
      X = sx(t);
    // keep end ticks inside the drawing (never clipped by the svg edge)
    const anchor = X + tw / 2 > W - 1 ? 'end' : X - tw / 2 < 1 ? 'start' : 'middle';
    g += `<text class="cx-tick" x="${r1(anchor === 'end' ? Math.min(X + 4, W - 1) : anchor === 'start' ? Math.max(X - 4, 1) : X)}" y="${H - B + 15}" text-anchor="${anchor}" font-size="${FONT}">${esc(text)}</text>`;
  }
  const lines = Array.isArray(label) ? label : label ? wrap(label, W - L - R + 40) : [];
  if (lines.length) g += textLines(lines, (L + W - R) / 2, H - B + 32, { anchor: 'middle', cls: 'cx-axis-label', lh: 1.2 });
  return g;
}

// ---- line / step chart -----------------------------------------------------------------------------------------------
/**
 * series: [{name, color, points:[{x, y, lo?, hi?, tip?}], step?, band?, markers?: 'all'|'hit'|'none', dash?}]
 * x/y: {min, max, label, ticks?, format?(v)}; refs: [{kind:'diag'|'h'|'v', value?, label?}]
 */
export function lineChart({ width = 560, height = 300, x = {}, y = {}, series = [], refs = [], aria = '', legendItems = null, square = false }) {
  series = series.filter((s) => s && s.points && s.points.some((p) => isNum(p.x) && isNum(p.y)));
  if (!series.length) return '';
  const all = series.flatMap((s) => s.points);
  const xmin = x.min ?? Math.min(...all.map((p) => p.x)),
    xmax = x.max ?? Math.max(...all.map((p) => p.x));
  const yvals = all.flatMap((p) => [p.y, p.lo, p.hi]).filter(isNum);
  let ymin = y.min ?? Math.min(...yvals),
    ymax = y.max ?? Math.max(...yvals);
  if (ymax <= ymin) ymax = ymin + 1;
  const xt = x.ticks || ticks(xmin, xmax, width < 360 ? 4 : 5);
  const yt = y.ticks || ticks(ymin, ymax, height < 220 ? 3 : 5);
  const xf = x.format || ((v) => fmtNum(v, decimalsFor((xt[1] ?? 1) - (xt[0] ?? 0))));
  const yf = y.format || ((v) => fmtNum(v, decimalsFor((yt[1] ?? 1) - (yt[0] ?? 0))));
  const tickW = Math.max(...yt.map((t) => textWidth(yf(t))));
  const L = Math.ceil(tickW + (y.label ? 26 : 10)),
    R = 14,
    T = 12;
  const xl = x.label ? wrap(x.label, width - L - R + 30) : [];
  const B = 26 + xl.length * 14;
  let W = width,
    H = height;
  if (square) H = Math.min(height, W - L - R + T + B);
  const pw = W - L - R,
    ph = H - T - B;
  const sx = (v) => L + ((v - xmin) / (xmax - xmin || 1)) * pw;
  const sy = (v) => T + ph - ((Math.max(ymin, Math.min(ymax, v)) - ymin) / (ymax - ymin)) * ph;
  let g = yAxis({ L, R, W, T, B, H, ys: yt, sy, fmt: yf, label: y.label });
  g += xAxis({ L, R, W, H, B, xs: xt, sx, fmt: xf, label: xl });
  for (const ref of refs) {
    if (ref.kind === 'diag') {
      const a = Math.max(xmin, ymin),
        b = Math.min(xmax, ymax);
      g += `<line class="cx-ref" x1="${r1(sx(a))}" y1="${r1(sy(a))}" x2="${r1(sx(b))}" y2="${r1(sy(b))}"/>`;
      if (ref.label) g += textLines([ref.label], sx(b) - 4, sy(b) + (sy(a) - sy(b)) * 0.12 + 16, { anchor: 'end', cls: 'cx-ref-t' });
    } else if (ref.kind === 'h' && isNum(ref.value)) {
      g += `<line class="cx-ref" x1="${L}" x2="${W - R}" y1="${r1(sy(ref.value))}" y2="${r1(sy(ref.value))}"/>`;
      if (ref.label) g += textLines([ref.label], W - R - 2, sy(ref.value) - 5, { anchor: 'end', cls: 'cx-ref-t' });
    } else if (ref.kind === 'v' && isNum(ref.value)) {
      g += `<line class="cx-ref" x1="${r1(sx(ref.value))}" x2="${r1(sx(ref.value))}" y1="${T}" y2="${H - B}"/>`;
      if (ref.label) g += textLines([ref.label], sx(ref.value) + 4, T + 10, { cls: 'cx-ref-t' });
    }
  }
  let marks = '';
  series.forEach((s) => {
    const color = s.color || COLOR[1];
    const pts = s.points.filter((p) => isNum(p.x) && isNum(p.y));
    const sorted = s.keepOrder ? pts : [...pts].sort((a, b) => a.x - b.x || a.y - b.y);
    if (s.band && sorted.every((p) => isNum(p.lo) && isNum(p.hi)) && sorted.length > 1) {
      const up = sorted.map((p) => `${r1(sx(p.x))},${r1(sy(p.hi))}`).join(' ');
      const dn = [...sorted].reverse().map((p) => `${r1(sx(p.x))},${r1(sy(p.lo))}`).join(' ');
      g += `<polygon class="cx-band" points="${up} ${dn}" style="fill:${color}"/>`;
    }
    let d = '';
    sorted.forEach((p, i) => {
      const X = r1(sx(p.x)),
        Y = r1(sy(p.y));
      if (!i) d += `M${X},${Y}`;
      else if (s.step === 'vh') d += `V${Y}H${X}`;
      else if (s.step) d += `H${X}V${Y}`;
      else d += `L${X},${Y}`;
    });
    g += `<path class="cx-line${s.dash ? ' cx-dash' : ''}" d="${d}" style="stroke:${color}"/>`;
    const mode = s.markers || 'all';
    for (const p of sorted) {
      const X = r1(sx(p.x)),
        Y = r1(sy(p.y));
      if (mode === 'all') marks += `<circle class="cx-dot" cx="${X}" cy="${Y}" r="4.5" style="fill:${color}"/>`;
      if (mode !== 'none' && p.tip) marks += `<circle class="cx-hit" cx="${X}" cy="${Y}" r="11"${tipAttr(p.tip)}/>`;
    }
    if (s.endLabel) {
      const last = sorted[sorted.length - 1];
      marks += textLines([s.endLabel], Math.min(sx(last.x) + 8, W - R - textWidth(s.endLabel)), sy(last.y) + 4, { cls: 'cx-t' });
    }
  });
  const items = legendItems ?? (series.length > 1 ? series.map((s) => ({ name: s.name, color: s.color, kind: s.dash ? 'dash' : 'line' })) : []);
  return legend(items) + svgOpen(W, H, aria) + g + marks + '</svg>';
}

// ---- forest / dot-and-whisker ------------------------------------------------------------------------------------------
/**
 * rows: [{label, sub?, est, lo?, hi?, color?, hollow?, header?, value?, tip?}]
 * x: {min, max, label, format?}; refs: [{value, label}]
 */
export function forest({ width = 560, rows = [], x = {}, refs = [], aria = '', rowH = null, valueWidth = null }) {
  rows = rows.filter((r) => r && (r.header || isNum(r.est)));
  if (!rows.length) return '';
  const narrow = width < 470;
  const vals = rows.flatMap((r) => [r.est, r.lo, r.hi]).filter(isNum);
  const xmin = x.min ?? Math.min(...vals),
    xmax = x.max ?? Math.max(...vals);
  const xt = x.ticks || ticks(xmin, xmax, narrow ? 4 : 5);
  const xf = x.format || ((v) => fmtNum(v, decimalsFor((xt[1] ?? 1) - (xt[0] ?? 0))));
  const vf = x.valueFormat || x.format || fmtSmart;
  const vText = (r) => r.value ?? (isNum(r.lo) && isNum(r.hi) ? `${vf(r.est)} [${vf(r.lo)}–${vf(r.hi)}]` : vf(r.est));
  const VW = narrow ? 0 : (valueWidth ?? Math.min(150, Math.max(...rows.map((r) => (r.header ? 0 : textWidth(vText(r))))) + 14));
  const labelMax = Math.max(90, Math.min(width * (narrow ? 0.48 : 0.36), 260));
  const prepared = rows.map((r) => {
    const lines = r.header ? wrap(r.label, width - 8, FONT) : wrap(r.label, labelMax, FONT + 0.5);
    const sub = r.sub && !r.header ? wrap(r.sub, labelMax, FONT - 1) : [];
    const valueLine = narrow && !r.header ? [vText(r)] : [];
    return { ...r, lines, subLines: [...sub, ...valueLine] };
  });
  const body = prepared.filter((p) => !p.header);
  const L = Math.ceil(Math.min(labelMax, Math.max(0, ...body.map((p) => Math.max(...p.lines.map((l) => textWidth(l, FONT + 0.5)), ...p.subLines.map((l) => textWidth(l, FONT - 1)))))) + 14);
  const refLabels = refs.some((r) => r && r.label && isNum(r.value));
  const R = Math.max(VW + 10, narrow ? 12 : 0),
    T = refLabels ? 20 : 6;
  const xl = x.label ? wrap(x.label, width - L - R + 40) : [];
  const B = 26 + xl.length * 14;
  const lineH = (FONT + 0.5) * 1.25;
  const heights = prepared.map((p) => (p.header ? p.lines.length * FONT * 1.25 + 14 : Math.max(rowH ?? 26, (p.lines.length + p.subLines.length) * lineH + 10)));
  const H = T + heights.reduce((a, b) => a + b, 0) + B;
  const W = width,
    pw = W - L - R;
  const sx = (v) => L + ((Math.max(xmin, Math.min(xmax, v)) - xmin) / (xmax - xmin || 1)) * pw;
  let g = '';
  for (const t of xt) g += `<line class="cx-grid" x1="${r1(sx(t))}" x2="${r1(sx(t))}" y1="${T}" y2="${H - B}"/>`;
  g += xAxis({ L, R, W, H, B, xs: xt, sx, fmt: xf, label: xl });
  for (const ref of refs) {
    if (!isNum(ref.value)) continue;
    g += `<line class="cx-ref" x1="${r1(sx(ref.value))}" x2="${r1(sx(ref.value))}" y1="${T - (ref.label ? 4 : 0)}" y2="${H - B}"/>`;
    if (ref.label) g += refLabel(ref.label, sx(ref.value), 11, L, W - R);
  }
  let y = T,
    marks = '';
  prepared.forEach((p, i) => {
    const h = heights[i];
    const cy = y + h / 2;
    if (p.header) {
      // a full-width band: covers the gridlines so the heading never collides with the plot
      g += `<rect x="0" y="${r1(y + 1)}" width="${W}" height="${r1(h - 2)}" class="cx-band-head"/>`;
      g += textLines(p.lines, 2, y + 6 + FONT, { cls: 'cx-t cx-strong', weight: 600 });
      y += h;
      return;
    }
    const labelTop = cy - ((p.lines.length + p.subLines.length - 1) * lineH) / 2 + 4;
    g += textLines(p.lines, L - 12, labelTop, { anchor: 'end', size: FONT + 0.5, cls: p.emphasis ? 'cx-t cx-strong' : 'cx-t', weight: p.emphasis ? 600 : null });
    if (p.subLines.length)
      g += textLines(p.subLines, L - 12, labelTop + p.lines.length * lineH, { anchor: 'end', size: FONT - 1, cls: 'cx-sub' });
    const color = p.color || COLOR[1];
    if (isNum(p.lo) && isNum(p.hi))
      marks += `<line class="cx-whisker" x1="${r1(sx(p.lo))}" x2="${r1(sx(p.hi))}" y1="${r1(cy)}" y2="${r1(cy)}" style="stroke:${color}"/>` +
        `<line class="cx-whisker" x1="${r1(sx(p.lo))}" x2="${r1(sx(p.lo))}" y1="${r1(cy - 5)}" y2="${r1(cy + 5)}" style="stroke:${color}"/>` +
        `<line class="cx-whisker" x1="${r1(sx(p.hi))}" x2="${r1(sx(p.hi))}" y1="${r1(cy - 5)}" y2="${r1(cy + 5)}" style="stroke:${color}"/>`;
    marks += p.hollow
      ? `<circle class="cx-dot cx-hollow" cx="${r1(sx(p.est))}" cy="${r1(cy)}" r="5" style="stroke:${color}"/>`
      : `<circle class="cx-dot" cx="${r1(sx(p.est))}" cy="${r1(cy)}" r="5" style="fill:${color}"/>`;
    if (!narrow) g += textLines([vText(p)], W - VW + 4, cy + 4, { cls: p.emphasis ? 'cx-t cx-strong cx-num' : 'cx-t cx-num', weight: p.emphasis ? 600 : null });
    marks += `<rect class="cx-hitrow" x="0" y="${r1(y)}" width="${W}" height="${r1(h)}"${tipAttr(p.tip || [vText(p), p.label + (p.sub ? ` · ${p.sub}` : '')])}/>`;
    y += h;
  });
  return svgOpen(W, H, aria) + g + marks + '</svg>';
}
/** Reference-line label above the plot, kept inside [lo, hi]. */
function refLabel(text, x, y, lo, hi) {
  const w = textWidth(text, FONT - 0.5);
  let anchor = 'middle';
  if (x - w / 2 < lo - 10) anchor = 'start';
  else if (x + w / 2 > hi + 10) anchor = 'end';
  return textLines([text], x, y, { anchor, size: FONT - 0.5, cls: 'cx-ref-t' });
}

// ---- heatmap --------------------------------------------------------------------------------------------------------
/**
 * values: rows x cols numbers (null = n/a); text(r, c) -> cell text; tip(r, c) -> tooltip lines.
 * scale: {kind:'sequential', max} | {kind:'diverging', max}. Cell shading uses colour-mix of the .mx tokens.
 */
// ---- cell label colour -----------------------------------------------------------------------------------------------
// White or ink (#1b1a2e), whichever contrasts more with the cell fill as the browser resolves it: palette tokens, theme
// and color-mix included. Without a DOM (node tests) the cell keeps its class colour (the measured crossover).
const INK_RGB = [0x1b / 255, 0x1a / 255, 0x2e / 255];
const labelFills = new Map();
function luminance(rgb) {
  const f = (c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2]);
}
function resolveRgb(color) {
  const probe = document.createElement('i');
  probe.className = 'mx'; // the chart tokens are scoped to the dashboard
  probe.style.cssText = `position:absolute;visibility:hidden;pointer-events:none;color:${color}`;
  document.body.append(probe);
  const c = getComputedStyle(probe).color;
  probe.remove();
  let m = c.match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/);
  if (m) return [m[1], m[2], m[3]].map((v) => Number(v) / 255);
  m = c.match(/^color\(srgb\s+([\d.e-]+)\s+([\d.e-]+)\s+([\d.e-]+)/);
  return m ? [m[1], m[2], m[3]].map(Number) : null;
}
function cellLabelFill(fill) {
  if (typeof document === 'undefined' || !document.body || typeof getComputedStyle !== 'function') return null;
  const dark = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const key = `${dark}|${fill}`;
  if (!labelFills.has(key)) {
    const rgb = resolveRgb(fill);
    const L = rgb && luminance(rgb);
    labelFills.set(key, rgb ? (1.05 / (L + 0.05) >= (L + 0.05) / (luminance(INK_RGB) + 0.05) ? '#ffffff' : '#1b1a2e') : null);
  }
  return labelFills.get(key);
}

export function heatmap({ width = 560, rows = [], cols = [], values = [], text = null, tip = null, scale = { kind: 'sequential' }, aria = '', colLabel = '', rowLabel = '', cellH = 40, flags = null }) {
  if (!rows.length || !cols.length) return '';
  const narrow = width < 470;
  const labelMax = Math.min(width * (narrow ? 0.34 : 0.3), 230);
  const rowLines = rows.map((r) => wrap(r, labelMax, FONT));
  const L = Math.ceil(Math.min(labelMax, Math.max(...rowLines.flat().map((l) => textWidth(l)))) + 12) + (flags ? 18 : 0);
  const cw = (width - L - 4) / cols.length;
  // narrow columns: shrink the header font (down to 9px) before any word would be hard-split ("Invas-ive")
  const longest = Math.max(...cols.flatMap((c) => String(c).split(/\s+/).map((w) => w.length)));
  let colFont = FONT - 0.5;
  while (colFont > 9 && longest * colFont * CHAR > cw - 4) colFont -= 0.5;
  const colLines = cols.map((c) => wrap(c, cw - 4, colFont));
  const headH = Math.max(...colLines.map((l) => l.length)) * (colFont * 1.2) + 10 + (colLabel ? 16 : 0);
  const rh = Math.max(cellH, ...rowLines.map((l) => l.length * FONT * 1.25 + 10));
  const H = headH + rows.length * rh + (rowLabel ? 4 : 2);
  const flat = values.flat().filter(isNum);
  const max = scale.max ?? Math.max(...flat.map(Math.abs), 1e-9);
  let g = '';
  if (colLabel) g += textLines([colLabel], L + (width - L) / 2, 11, { anchor: 'middle', cls: 'cx-axis-label' });
  cols.forEach((c, j) => {
    g += textLines(colLines[j], L + cw * j + cw / 2, headH - 8 - (colLines[j].length - 1) * colFont * 1.2, { anchor: 'middle', size: colFont, cls: 'cx-t', lh: 1.2 });
  });
  let marks = '';
  rows.forEach((r, i) => {
    const y = headH + i * rh;
    g += textLines(rowLines[i], L - 8 - (flags ? 18 : 0), y + rh / 2 - ((rowLines[i].length - 1) * FONT * 1.25) / 2 + 4, { anchor: 'end', cls: 'cx-t' });
    if (flags && flags[i]) g += flagGlyph(L - 14, y + rh / 2, flags[i]);
    cols.forEach((c, j) => {
      const v = values[i] && values[i][j];
      const x = L + cw * j;
      let fill = 'var(--cx-na)',
        on = 'cx-on-weak';
      if (isNum(v)) {
        if (scale.kind === 'diverging') {
          const t = Math.min(1, Math.abs(v) / max);
          const p = Math.round(8 + 88 * t);
          fill = `color-mix(in srgb, ${v < 0 ? 'var(--cx-neg)' : 'var(--cx-pos)'} ${p}%, var(--cx-mid))`;
          if (p > 82) on = v < 0 ? 'cx-on-indigo' : 'cx-on-coral'; // measured crossover: ink reads better below ~82%
        } else {
          const t = Math.min(1, Math.max(0, v / max));
          const p = Math.round(6 + 90 * t);
          fill = `color-mix(in srgb, var(--cx-seq) ${p}%, var(--cx-seq0))`;
          if (p > 82) on = 'cx-on-indigo';
        }
      }
      const label = text ? text(i, j) : isNum(v) ? fmtNum(v) : 'n/a';
      const ink = cellLabelFill(fill);
      marks += `<g class="cx-cell"${tipAttr(tip ? tip(i, j) : [label, `${r} · ${c}`])}><rect x="${r1(x + 1)}" y="${r1(y + 1)}" width="${r1(cw - 2)}" height="${r1(rh - 2)}" rx="3" style="fill:${fill}"/>` +
        `<text x="${r1(x + cw / 2)}" y="${r1(y + rh / 2 + 4)}" text-anchor="middle" font-size="${cw < 46 ? FONT - 1 : FONT + 0.5}" class="${on} cx-num"${ink ? ` style="fill:${ink}"` : ''}>${esc(label)}</text></g>`;
    });
  });
  return svgOpen(width, H, aria) + g + marks + '</svg>';
}
function flagGlyph(x, y, kind) {
  // a status glyph that does not rely on colour alone: check = as expected, cross = contradiction, dash = context
  if (kind === 'ok') return `<path class="cx-glyph-ok" d="M${x - 5},${y} l3.5,3.5 l6,-7" />`;
  if (kind === 'bad') return `<path class="cx-glyph-bad" d="M${x - 4},${y - 4} l8,8 M${x + 4},${y - 4} l-8,8" />`;
  return `<path class="cx-glyph-neutral" d="M${x - 4},${y} h8" />`;
}
/** Colour-scale legend for heatmaps (HTML). */
export function scaleLegend({ kind = 'sequential', lo = '', mid = '', hi = '', caption = '' }) {
  const grad = kind === 'diverging' ? 'cx-grad-div' : 'cx-grad-seq';
  return `<div class="cx-scale"><span class="cx-scale-cap">${esc(caption)}</span><span class="cx-scale-lo">${esc(lo)}</span><span class="cx-grad ${grad}"></span><span class="cx-scale-hi">${esc(hi)}</span>${mid ? `<span class="cx-scale-mid">${esc(mid)}</span>` : ''}</div>`;
}

// ---- bars (vertical grouped / stacked, horizontal) ---------------------------------------------------------------------
/**
 * categories: [label]; series: [{name, color, values:[number]}]; mode 'grouped' | 'stacked'.
 * y: {label, max?, format?}; valueLabels: true puts a value above each bar when it fits.
 */
export function columns({ width = 560, height = 240, categories = [], series = [], mode = 'grouped', y = {}, x = {}, aria = '', valueLabels = true, zeroLabels = false, tip = null, legendItems = null }) {
  if (!categories.length || !series.length) return '';
  const totals = categories.map((_, j) => (mode === 'stacked' ? series.reduce((a, s) => a + (isNum(s.values[j]) ? s.values[j] : 0), 0) : Math.max(...series.map((s) => (isNum(s.values[j]) ? s.values[j] : 0)))));
  const ymax = y.max ?? (Math.max(...totals, 1e-9) * 1.12);
  const yt = ticks(0, ymax, height < 200 ? 3 : 4).filter((t) => t <= ymax + 1e-9);
  const yf = y.format || ((v) => fmtNum(v, decimalsFor((yt[1] ?? 1) - (yt[0] ?? 0))));
  const L = Math.ceil(Math.max(...yt.map((t) => textWidth(yf(t)))) + (y.label ? 26 : 10));
  const R = 6;
  const W = width;
  const band = (W - L - R) / categories.length;
  const catLines = categories.map((c) => wrap(c, band - 4, FONT - 0.5));
  const B = Math.max(...catLines.map((l) => l.length)) * FONT * 1.2 + 10 + (x.label ? 18 : 0);
  const T = 14,
    H = height;
  const ph = H - T - B;
  const sy = (v) => T + ph - (Math.min(v, ymax) / ymax) * ph;
  let g = yAxis({ L, R, W, T, B, H, ys: yt, sy, fmt: yf, label: y.label });
  g += `<line class="cx-axis" x1="${L}" x2="${W - R}" y1="${r1(H - B)}" y2="${r1(H - B)}"/>`;
  catLines.forEach((lines, j) => (g += textLines(lines, L + band * j + band / 2, H - B + 14, { anchor: 'middle', size: FONT - 0.5, cls: 'cx-t' })));
  if (x.label) g += `<text class="cx-axis-label" x="${r1((L + W - R) / 2)}" y="${H - 5}" text-anchor="middle" font-size="${FONT}">${esc(x.label)}</text>`;
  let marks = '';
  const gap = 2;
  categories.forEach((cat, j) => {
    const x0 = L + band * j;
    if (mode === 'stacked') {
      const bw = Math.min(28, band * 0.6);
      let acc = 0;
      series.forEach((s, k) => {
        const v = isNum(s.values[j]) ? s.values[j] : 0;
        if (v <= 0) return;
        const yTop = sy(acc + v),
          yBot = sy(acc);
        const hgt = Math.max(0, yBot - yTop - (acc > 0 ? gap : 0));
        marks += `<rect class="cx-bar" x="${r1(x0 + (band - bw) / 2)}" y="${r1(yTop)}" width="${r1(bw)}" height="${r1(hgt)}" rx="2" style="fill:${s.color || COLOR[k + 1]}"${tipAttr(tip ? tip(j, k) : [fmtNum(v, 0), `${s.name} · ${cat}`])}/>`;
        acc += v;
      });
      if (valueLabels) marks += textLines([fmtNum(totals[j], 0)], x0 + band / 2, sy(totals[j]) - 5, { anchor: 'middle', cls: 'cx-t cx-num', size: FONT - 0.5 });
    } else {
      const inner = Math.min(band * 0.8, series.length * 24 + (series.length - 1) * gap);
      const bw = (inner - (series.length - 1) * gap) / series.length;
      series.forEach((s, k) => {
        const v = s.values[j];
        const bx = x0 + (band - inner) / 2 + k * (bw + gap);
        if (!isNum(v)) return;
        const yTop = sy(v),
          hgt = Math.max(v > 0 ? 1.5 : 0, H - B - yTop);
        marks += `<path class="cx-bar" d="${barPath(bx, H - B - hgt, bw, hgt)}" style="fill:${s.color || COLOR[k + 1]}"/>`;
        marks += `<rect class="cx-hit" x="${r1(bx - 1)}" y="${T}" width="${r1(bw + 2)}" height="${r1(ph)}"${tipAttr(tip ? tip(j, k) : [y.tipFormat ? y.tipFormat(v) : yf(v), `${s.name} · ${cat}`])}/>`;
        if (valueLabels && bw >= textWidth(fmtNum(v, 0), FONT - 1.5) - 2 && (v > 0 || (zeroLabels && v === 0)))
          marks += textLines([y.labelFormat ? y.labelFormat(v) : fmtNum(v, 0)], bx + bw / 2, yTop - 4, { anchor: 'middle', cls: 'cx-t cx-num', size: FONT - 1.5 });
      });
    }
  });
  const items = legendItems ?? (series.length > 1 ? series.map((s, k) => ({ name: s.name, color: s.color || COLOR[k + 1], kind: 'rect' })) : []);
  return legend(items) + svgOpen(W, H, aria) + g + marks + '</svg>';
}
/** Column with a 4px rounded data end and a square baseline. */
function barPath(x, y, w, h) {
  if (h <= 0) return '';
  const r = Math.min(4, w / 2, h);
  return `M${r1(x)},${r1(y + h)}V${r1(y + r)}Q${r1(x)},${r1(y)} ${r1(x + r)},${r1(y)}H${r1(x + w - r)}Q${r1(x + w)},${r1(y)} ${r1(x + w)},${r1(y + r)}V${r1(y + h)}Z`;
}

/** Horizontal 100% stacked bars: rows [{label, parts:[{name, value, color}]}] (counts; shares computed). */
export function stackedRows({ width = 560, rows = [], aria = '', legendItems = null, unit = '' }) {
  rows = rows.filter((r) => r && r.parts && r.parts.some((p) => isNum(p.value) && p.value > 0));
  if (!rows.length) return '';
  const labelMax = Math.min(Math.max(width * 0.32, 120), 200);
  const lines = rows.map((r) => wrap(r.label, labelMax));
  const L = Math.ceil(Math.min(labelMax, Math.max(...lines.flat().map((l) => textWidth(l)))) + 12);
  const R = 8,
    T = 4;
  // a row grows with its wrapped label (phones wrap long concept names onto 2-3 lines; never overlap the next row)
  const heights = lines.map((l) => Math.max(34, l.length * FONT * 1.25 + 12));
  const tops = heights.map((_, i) => T + heights.slice(0, i).reduce((a, b) => a + b, 0));
  const H = T + heights.reduce((a, b) => a + b, 0) + 22;
  const pw = width - L - R;
  let g = '',
    marks = '';
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const x = L + t * pw;
    g += `<line class="cx-grid" x1="${r1(x)}" x2="${r1(x)}" y1="${T}" y2="${H - 22}"/><text class="cx-tick" x="${r1(x)}" y="${H - 7}" text-anchor="${t === 1 ? 'end' : t === 0 ? 'start' : 'middle'}" font-size="${FONT}">${t * 100}%</text>`;
  }
  rows.forEach((r, i) => {
    const y = tops[i],
      rh = heights[i];
    const total = r.parts.reduce((a, p) => a + (isNum(p.value) ? p.value : 0), 0);
    g += textLines(lines[i], L - 8, y + rh / 2 - ((lines[i].length - 1) * FONT * 1.25) / 2 + 4, { anchor: 'end', cls: 'cx-t' });
    let acc = 0;
    r.parts.forEach((p, k) => {
      const v = isNum(p.value) ? p.value : 0;
      if (v <= 0) return;
      const x = L + (acc / total) * pw,
        w = (v / total) * pw;
      const share = v / total;
      marks += `<g class="cx-seg"${tipAttr([`${fmtNum(v, 0)}${unit} (${fmtPct(share)})`, `${p.name} · ${r.label}`])}><rect x="${r1(x + (acc ? 1 : 0))}" y="${r1(y + rh / 2 - 11)}" width="${r1(Math.max(0.5, w - (acc ? 2 : 0)))}" height="22" rx="2" style="fill:${p.color || COLOR[k + 1]}"/>`;
      const lab = `${fmtNum(v, 0)}`;
      if (w > textWidth(lab) + 10) marks += `<text x="${r1(x + w / 2)}" y="${r1(y + rh / 2 + 4)}" text-anchor="middle" font-size="${FONT}" class="${p.onStrong ? 'cx-on-indigo' : 'cx-on-weak'} cx-num">${esc(lab)}</text>`;
      marks += '</g>';
      acc += v;
    });
  });
  return (legendItems ? legend(legendItems) : '') + svgOpen(width, H, aria) + g + marks + '</svg>';
}

/** Horizontal bars from zero: rows [{label, value, color?, tip?, valueText?}]; value labels at the bar tip. */
export function hbars({ width = 560, rows = [], x = {}, aria = '' }) {
  rows = rows.filter((r) => r && isNum(r.value));
  if (!rows.length) return '';
  const max = x.max ?? Math.max(...rows.map((r) => r.value), 1e-9);
  const labelMax = Math.min(width * 0.42, 240);
  const lines = rows.map((r) => wrap(r.label, labelMax));
  const L = Math.ceil(Math.min(labelMax, Math.max(...lines.flat().map((l) => textWidth(l)))) + 12);
  const vt = (r) => r.valueText ?? fmtNum(r.value, 0);
  const R = Math.max(...rows.map((r) => textWidth(vt(r)))) + 12;
  const rh = Math.max(28, ...lines.map((l) => l.length * FONT * 1.25 + 12));
  const H = rows.length * rh + 4;
  const pw = width - L - R;
  let g = `<line class="cx-axis" x1="${L}" x2="${L}" y1="0" y2="${H}"/>`,
    marks = '';
  rows.forEach((r, i) => {
    const y = i * rh;
    const w = Math.max(r.value > 0 ? 2 : 0, (r.value / max) * pw);
    g += textLines(lines[i], L - 8, y + rh / 2 - ((lines[i].length - 1) * FONT * 1.25) / 2 + 4, { anchor: 'end', cls: r.emphasis ? 'cx-t cx-strong' : 'cx-t', weight: r.emphasis ? 600 : null });
    if (w > 0) marks += `<path class="cx-bar" d="${hbarPath(L, y + rh / 2 - 8, w, 16)}" style="fill:${r.color || COLOR[1]}"/>`;
    marks += textLines([vt(r)], L + w + 6, y + rh / 2 + 4, { cls: 'cx-t cx-num' });
    marks += `<rect class="cx-hitrow" x="0" y="${r1(y)}" width="${width}" height="${r1(rh)}"${tipAttr(r.tip || [vt(r), r.label])}/>`;
  });
  return svgOpen(width, H, aria) + g + marks + '</svg>';
}
function hbarPath(x, y, w, h) {
  const r = Math.min(4, w, h / 2);
  return `M${r1(x)},${r1(y)}H${r1(x + w - r)}Q${r1(x + w)},${r1(y)} ${r1(x + w)},${r1(y + r)}V${r1(y + h - r)}Q${r1(x + w)},${r1(y + h)} ${r1(x + w - r)},${r1(y + h)}H${r1(x)}Z`;
}

/** Grid of titled panels (HTML); minWidth drives the responsive column count in metrics.css. */
export function multiples(panels, { cols = 2, cls = '' } = {}) {
  const items = panels.filter((p) => p && p.body);
  if (!items.length) return '';
  return `<div class="cx-multiples ${cls}" style="--cols:${cols}">${items
    .map((p) => `<div class="cx-panel"><div class="cx-panel-title">${esc(p.title)}${p.badge ? ` <span class="mx-chip mx-chip-${p.badgeTone || 'neutral'}">${esc(p.badge)}</span>` : ''}</div>${p.body}</div>`)
    .join('')}</div>`;
}

// ---- box / interval plot -------------------------------------------------------------------------------------------
/** rows: [{label, sub?, q:{p2_5, p25, p50, p75, p97_5, mean}, color}] horizontal boxes; x: {min?, max?, label}. */
export function boxPlot({ width = 560, rows = [], x = {}, aria = '', refs = [] }) {
  rows = rows.filter((r) => r && r.q && isNum(r.q.p50));
  if (!rows.length) return '';
  const vals = rows.flatMap((r) => [r.q.p2_5, r.q.p97_5, r.q.mean]).filter(isNum);
  const xmin = x.min ?? Math.min(0, ...vals),
    xmax = x.max ?? Math.max(...vals) * 1.04;
  const narrow = width < 470;
  const xt = ticks(xmin, xmax, narrow ? 4 : 6);
  const xf = x.format || ((v) => fmtNum(v, decimalsFor((xt[1] ?? 1) - (xt[0] ?? 0))));
  const labelMax = Math.min(width * (narrow ? 0.4 : 0.3), 220);
  const prepared = rows.map((r) => ({ ...r, lines: wrap(r.label, labelMax, FONT + 0.5), subLines: r.sub ? wrap(r.sub, labelMax, FONT - 1) : [] }));
  const L = Math.ceil(Math.min(labelMax, Math.max(...prepared.flatMap((p) => [...p.lines.map((l) => textWidth(l, FONT + 0.5)), ...p.subLines.map((l) => textWidth(l, FONT - 1))]))) + 14);
  const R = 14,
    T = 6;
  const xl = x.label ? wrap(x.label, width - L - R + 30) : [];
  const B = 26 + xl.length * 14;
  const lineH = (FONT + 0.5) * 1.25;
  const heights = prepared.map((p) => Math.max(44, (p.lines.length + p.subLines.length) * lineH + 14));
  const H = T + heights.reduce((a, b) => a + b, 0) + B;
  const pw = width - L - R;
  const sx = (v) => L + ((Math.max(xmin, Math.min(xmax, v)) - xmin) / (xmax - xmin || 1)) * pw;
  let g = '';
  for (const t of xt) g += `<line class="cx-grid" x1="${r1(sx(t))}" x2="${r1(sx(t))}" y1="${T}" y2="${H - B}"/>`;
  g += xAxis({ L, R, W: width, H, B, xs: xt, sx, fmt: xf, label: xl });
  for (const ref of refs) if (isNum(ref.value)) g += `<line class="cx-ref" x1="${r1(sx(ref.value))}" x2="${r1(sx(ref.value))}" y1="${T}" y2="${H - B}"/>`;
  let y = T,
    marks = '';
  prepared.forEach((p, i) => {
    const h = heights[i],
      cy = y + h / 2,
      q = p.q,
      color = p.color || COLOR[1];
    const top = cy - ((p.lines.length + p.subLines.length - 1) * lineH) / 2 + 4;
    g += textLines(p.lines, L - 12, top, { anchor: 'end', size: FONT + 0.5, cls: 'cx-t' });
    if (p.subLines.length) g += textLines(p.subLines, L - 12, top + p.lines.length * lineH, { anchor: 'end', size: FONT - 1, cls: 'cx-sub' });
    if (isNum(q.p2_5) && isNum(q.p97_5)) marks += `<line class="cx-whisker" x1="${r1(sx(q.p2_5))}" x2="${r1(sx(q.p97_5))}" y1="${r1(cy)}" y2="${r1(cy)}" style="stroke:${color}"/>`;
    for (const e of [q.p2_5, q.p97_5]) if (isNum(e)) marks += `<line class="cx-whisker" x1="${r1(sx(e))}" x2="${r1(sx(e))}" y1="${r1(cy - 5)}" y2="${r1(cy + 5)}" style="stroke:${color}"/>`;
    if (isNum(q.p25) && isNum(q.p75)) marks += `<rect class="cx-box" x="${r1(sx(q.p25))}" y="${r1(cy - 9)}" width="${r1(Math.max(2, sx(q.p75) - sx(q.p25)))}" height="18" rx="3" style="fill:${color};stroke:${color}"/>`;
    marks += `<line class="cx-median" x1="${r1(sx(q.p50))}" x2="${r1(sx(q.p50))}" y1="${r1(cy - 11)}" y2="${r1(cy + 11)}" style="stroke:${color}"/>`;
    if (isNum(q.mean)) marks += `<path class="cx-mean" d="M${r1(sx(q.mean))},${r1(cy - 5)} l5,5 l-5,5 l-5,-5 Z"/>`;
    marks += `<rect class="cx-hitrow" x="0" y="${r1(y)}" width="${width}" height="${r1(h)}"${tipAttr(p.tip || [`median ${xf(q.p50)}${x.unit || ''}`, `${p.label}: middle 50% ${xf(q.p25)}–${xf(q.p75)}, 95% of days ${xf(q.p2_5)}–${xf(q.p97_5)}, mean ${xf(q.mean)}`])}/>`;
    y += h;
  });
  return svgOpen(width, H, aria) + g + marks + '</svg>';
}

// ---- KPI interval strip ------------------------------------------------------------------------------------------------
/** A compact 0..1 (or [min, max]) track: CI band, estimate dot, optional target tick. */
export function strip({ est, lo, hi, min = 0, max = 1, target = null, width = 200, aria = '', tone = 'ok' }) {
  if (!isNum(est)) return '';
  const H = 22,
    P = 6;
  const sx = (v) => P + ((Math.max(min, Math.min(max, v)) - min) / (max - min || 1)) * (width - 2 * P);
  let g = `<line class="cx-track" x1="${P}" x2="${width - P}" y1="11" y2="11"/>`;
  if (isNum(lo) && isNum(hi)) g += `<rect class="cx-ci cx-ci-${tone}" x="${r1(sx(lo))}" y="7" width="${r1(Math.max(3, sx(hi) - sx(lo)))}" height="8" rx="4"/>`;
  if (isNum(target)) g += `<line class="cx-target" x1="${r1(sx(target))}" x2="${r1(sx(target))}" y1="2" y2="20"/>`;
  g += `<circle class="cx-est cx-est-${tone}" cx="${r1(sx(est))}" cy="11" r="5"/>`;
  return `<svg class="cx-strip" viewBox="0 0 ${width} ${H}" width="${width}" height="${H}" role="img" aria-label="${esc(aria)}">${g}</svg>`;
}

// ---- tooltip + keyboard layer (installed once) -------------------------------------------------------------------------
let tipEl = null;
function tooltip() {
  if (tipEl && document.body.contains(tipEl)) return tipEl;
  tipEl = document.createElement('div');
  tipEl.className = 'cx-tooltip';
  tipEl.setAttribute('role', 'tooltip');
  tipEl.hidden = true;
  document.body.appendChild(tipEl);
  return tipEl;
}
function showTip(target, clientX, clientY) {
  const text = target.getAttribute('data-tip');
  if (!text) return;
  const el = tooltip();
  el.textContent = '';
  text.split('\n').forEach((line, i) => {
    const row = document.createElement(i ? 'span' : 'strong');
    row.textContent = line; // labels come from data files: textContent, never innerHTML
    el.appendChild(row);
  });
  el.hidden = false;
  const box = el.getBoundingClientRect();
  let x = clientX + 14,
    y = clientY + 14;
  if (x + box.width > window.innerWidth - 8) x = Math.max(8, clientX - box.width - 14);
  if (y + box.height > window.innerHeight - 8) y = Math.max(8, clientY - box.height - 14);
  el.style.left = x + 'px';
  el.style.top = y + 'px';
  document.querySelectorAll('.cx-active').forEach((n) => n !== target && n.classList.remove('cx-active'));
  target.classList.add('cx-active');
}
function hideTip() {
  if (tipEl) tipEl.hidden = true;
  document.querySelectorAll('.cx-active').forEach((n) => n.classList.remove('cx-active'));
}
function centre(node) {
  const b = node.getBoundingClientRect();
  return [b.left + b.width / 2, b.top + Math.min(b.height / 2, 20)];
}
export function installInteractions() {
  if (typeof document === 'undefined' || document.__cxInstalled) return;
  document.__cxInstalled = true;
  document.addEventListener('pointermove', (e) => {
    const t = e.target && e.target.closest && e.target.closest('svg.cx [data-tip]');
    if (t) showTip(t, e.clientX, e.clientY);
    else if (tipEl && !tipEl.hidden && !(document.activeElement && document.activeElement.matches && document.activeElement.matches('svg.cx'))) hideTip();
  });
  document.addEventListener('pointerleave', hideTip);
  window.addEventListener('scroll', () => {
    if (!(document.activeElement && document.activeElement.matches && document.activeElement.matches('svg.cx'))) hideTip();
  }, { passive: true });
  document.addEventListener('focusin', (e) => {
    const svg = e.target && e.target.matches && e.target.matches('svg.cx') ? e.target : null;
    if (!svg) return;
    const marks = [...svg.querySelectorAll('[data-tip]')];
    svg.__cxIndex = 0;
    if (marks[0]) showTip(marks[0], ...centre(marks[0]));
  });
  document.addEventListener('focusout', (e) => {
    if (e.target && e.target.matches && e.target.matches('svg.cx')) hideTip();
  });
  document.addEventListener('keydown', (e) => {
    const svg = document.activeElement;
    if (!svg || !svg.matches || !svg.matches('svg.cx')) return;
    if (e.key === 'Escape') {
      hideTip();
      return;
    }
    const marks = [...svg.querySelectorAll('[data-tip]')];
    if (!marks.length) return;
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -1e9, End: 1e9 }[e.key];
    if (step === undefined) return;
    e.preventDefault();
    const i = Math.max(0, Math.min(marks.length - 1, (svg.__cxIndex || 0) + step));
    svg.__cxIndex = i;
    showTip(marks[i], ...centre(marks[i]));
  });
}
installInteractions();
