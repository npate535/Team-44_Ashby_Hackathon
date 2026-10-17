// Inline-SVG chart helpers for the Metrics page (no chart CDN; the demo runs offline). Series colors are CSS tokens
// --s1/--s2/--s3 (validated with the dataviz palette checks in light and dark); text uses ink tokens. Every mark has
// a <title> hover tooltip; values that matter are direct-labeled; a legend is shown for >= 2 series.
import {esc} from './adapter.js';

const W = 600;
export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)'];
const f2 = v => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (+v).toFixed(Math.abs(v) < 10 ? 2 : 1));

/** {estimate|value, lo, hi | ci:[lo,hi], n, n_groups} → normalized {est, lo, hi, n, ng} or null. */
export function ci(obj) {
  if (obj === null || obj === undefined) return null;
  if (typeof obj === 'number') return {est: obj, lo: null, hi: null, n: null, ng: null};
  if (typeof obj !== 'object' || Array.isArray(obj)) return null;
  const est = obj.estimate ?? obj.value ?? obj.mean ?? obj.est;
  if (typeof est !== 'number') return null;
  const lo = obj.lo ?? (Array.isArray(obj.ci) ? obj.ci[0] : null) ?? obj.lower ?? null;
  const hi = obj.hi ?? (Array.isArray(obj.ci) ? obj.ci[1] : null) ?? obj.upper ?? null;
  return {est, lo: typeof lo === 'number' ? lo : null, hi: typeof hi === 'number' ? hi : null,
          n: obj.n ?? null, ng: obj.n_groups ?? null, level: typeof obj.ci === 'number' ? obj.ci : 0.95};
}

export function ciText(c) {
  if (!c) return '—';
  let s = f2(c.est);
  if (c.lo !== null && c.hi !== null) s += ` (${Math.round((c.level || 0.95) * 100)}% CI ${f2(c.lo)}–${f2(c.hi)}`;
  else s += ' (no CI';
  if (c.n !== null) s += `, n=${c.n}`;
  if (c.ng !== null) s += `, ${c.ng} groups`;
  return s + ')';
}

function legend(names) {
  if (names.length < 2) return '';
  return `<div class="legend" style="flex-wrap:wrap;margin-bottom:4px">${names.map((n, i) => `<span><i class="sw" style="background:${SERIES[i % SERIES.length]}"></i>${esc(n)}</span>`).join('')}</div>`;
}

function axisTicks(min, max, n = 5) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push(min + (max - min) * i / n);
  return out;
}

/** Horizontal bars with CI whiskers. rows [{label, est, lo, hi, series?}] ; opts {min, max, ref, refLabel}. */
export function barCI(rows, {min = 0, max = null, ref = null, refLabel = '', title = ''} = {}) {
  if (!rows.length) return '';
  const hiVals = rows.map(r => r.hi ?? r.est);
  max = max ?? (Math.max(...hiVals) * 1.1 || 1);
  const lw = 170, pad = 8, bh = 18, gap = 10, top = 6, plotW = W - lw - 70;
  const H = top + rows.length * (bh + gap) + 24;
  const x = v => lw + Math.max(0, Math.min(1, (v - min) / (max - min))) * plotW;
  let g = '';
  for (const t of axisTicks(min, max, 4)) {
    g += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - 22}" stroke="var(--ln)" stroke-width="1"/><text x="${x(t)}" y="${H - 8}" font-size="11" text-anchor="middle" fill="var(--mut)">${f2(t)}</text>`;
  }
  if (ref !== null) g += `<line x1="${x(ref)}" x2="${x(ref)}" y1="${top}" y2="${H - 22}" stroke="var(--ink)" stroke-dasharray="4 3"/><text x="${x(ref) + 4}" y="${top + 10}" font-size="11" fill="var(--mut)">${esc(refLabel)}</text>`;
  rows.forEach((r, i) => {
    const y = top + i * (bh + gap);
    const color = SERIES[(r.series ?? 0) % SERIES.length];
    const tip = `${r.label}: ${ciText(r)}`;
    g += `<text x="${lw - pad}" y="${y + bh * 0.7}" font-size="12" text-anchor="end" fill="var(--ink)">${esc(r.label)}</text>`;
    g += `<g><title>${esc(tip)}</title><rect x="${x(min)}" y="${y}" width="${Math.max(2, x(r.est) - x(min))}" height="${bh}" rx="4" fill="${color}"/>`;
    if (r.lo !== null && r.hi !== null) g += `<line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${y + bh / 2}" y2="${y + bh / 2}" stroke="var(--ink)" stroke-width="2"/><line x1="${x(r.lo)}" x2="${x(r.lo)}" y1="${y + 3}" y2="${y + bh - 3}" stroke="var(--ink)" stroke-width="2"/><line x1="${x(r.hi)}" x2="${x(r.hi)}" y1="${y + 3}" y2="${y + bh - 3}" stroke="var(--ink)" stroke-width="2"/>`;
    g += `<rect x="0" y="${y - 3}" width="${W}" height="${bh + 6}" fill="transparent"/></g>`;
    g += `<text x="${Math.max(x(r.hi ?? r.est), x(r.est)) + 6}" y="${y + bh * 0.7}" font-size="11" fill="var(--mut)">${f2(r.est)}</text>`;
  });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${g}</svg>`;
}

/** Line chart. series [{name, points:[{x, y, lo?, hi?}]}]; opts {xLabel, yLabel, yMin, yMax, xTicks}. */
export function lineChart(series, {xLabel = '', yLabel = '', yMin = null, yMax = null, title = '', diagonal = false,
                                   height = 300, width = W} = {}) {
  const pts = series.flatMap(s => s.points);
  if (!pts.length) return '';
  if (series.length > SERIES.length) throw new Error('more series than validated colors: use small multiples');
  const xs = pts.map(p => p.x), ys = pts.flatMap(p => [p.y, p.lo ?? p.y, p.hi ?? p.y]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  yMin = yMin ?? Math.min(...ys); yMax = yMax ?? Math.max(...ys);
  if (yMax === yMin) yMax = yMin + 1;
  const L = 48, R = series.length > 1 ? 110 : 24, T = 10, B = 40, H = height;
  const px = v => L + (v - x0) / (x1 - x0) * (width - L - R), py = v => H - B - (v - yMin) / (yMax - yMin) * (H - T - B);
  let g = '';
  for (const t of axisTicks(yMin, yMax, 4)) g += `<line x1="${L}" x2="${width - R}" y1="${py(t)}" y2="${py(t)}" stroke="var(--ln)"/><text x="${L - 6}" y="${py(t) + 4}" font-size="11" text-anchor="end" fill="var(--mut)">${f2(t)}</text>`;
  const uniqX = [...new Set(xs)].sort((a, b) => a - b);
  const xt = uniqX.length <= 8 ? uniqX : axisTicks(x0, x1, 4);
  for (const t of xt) g += `<text x="${px(t)}" y="${H - B + 16}" font-size="11" text-anchor="middle" fill="var(--mut)">${f2(t)}</text>`;
  g += `<line x1="${L}" x2="${width - R}" y1="${H - B}" y2="${H - B}" stroke="var(--mut)"/>`;
  if (diagonal) g += `<line x1="${px(Math.max(x0, yMin))}" y1="${py(Math.max(x0, yMin))}" x2="${px(Math.min(x1, yMax))}" y2="${py(Math.min(x1, yMax))}" stroke="var(--mut)" stroke-dasharray="4 4"/><text x="${px(Math.min(x1, yMax)) - 4}" y="${py(Math.min(x1, yMax)) + 14}" font-size="11" text-anchor="end" fill="var(--mut)">perfect calibration</text>`;
  series.forEach((s, i) => {
    const color = SERIES[i % SERIES.length];
    const sorted = [...s.points].sort((a, b) => a.x - b.x);
    const band = sorted.filter(p => p.lo !== null && p.lo !== undefined && p.hi !== null && p.hi !== undefined);
    if (band.length === sorted.length && band.length > 1) {
      const up = band.map(p => `${px(p.x)},${py(p.hi)}`).join(' '), dn = [...band].reverse().map(p => `${px(p.x)},${py(p.lo)}`).join(' ');
      g += `<polygon points="${up} ${dn}" fill="${color}" fill-opacity="0.14"/>`;
    }
    g += `<polyline points="${sorted.map(p => `${px(p.x)},${py(p.y)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
    for (const p of sorted) g += `<g><title>${esc(`${s.name} · ${xLabel} ${f2(p.x)}: ${f2(p.y)}${p.lo !== undefined && p.lo !== null ? ` (CI ${f2(p.lo)}–${f2(p.hi)})` : ''}${p.n ? `, n=${p.n}` : ''}`)}</title><circle cx="${px(p.x)}" cy="${py(p.y)}" r="4.5" fill="${color}" stroke="var(--sf)" stroke-width="2"/><circle cx="${px(p.x)}" cy="${py(p.y)}" r="12" fill="transparent"/></g>`;
    const last = sorted[sorted.length - 1];
    if (series.length > 1) g += `<text x="${px(last.x) + 8}" y="${py(last.y) + 4 + i * 2}" font-size="11" fill="var(--ink)">${esc(s.name)}</text>`;
  });
  g += `<text x="${(L + width - R) / 2}" y="${H - 6}" font-size="11" text-anchor="middle" fill="var(--mut)">${esc(xLabel)}</text>`;
  g += `<text x="12" y="${(T + H - B) / 2}" font-size="11" text-anchor="middle" fill="var(--mut)" transform="rotate(-90 12 ${(T + H - B) / 2})">${esc(yLabel)}</text>`;
  return legend(series.map(s => s.name)) + `<svg class="chart" viewBox="0 0 ${width} ${H}" role="img" aria-label="${esc(title)}">${g}</svg>`;
}

/** Simple table from rows of cells (strings already escaped by the caller). */
export function table(head, rows) {
  return `<div class="tblw"><table class="mt"><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
