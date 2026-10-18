// Display image + absolutely positioned SVG in source-pixel coordinates (the downscaled JPEG scales for free).
// Biopsy: per-tile heatmap (risk / a concept's tile z / ABMIL attention) with highlighted tiles.
// Surgical: the four margins as per-segment colored strips (BUILD_PLAN §A12).
import {esc, MARGINS, num} from '../adapter.js';

const clamp01 = x => Math.max(0, Math.min(1, x));

/** Heatmap layers available for a case: risk, every concept with tile values, attention when present. */
export function layers(v) {
  const out = [{id: 'risk', label: 'Overall risk (tile P(Invasive))'}];
  const tileConcepts = new Set(v.tiles.flatMap(t => Object.keys(t.concepts || {})));
  for (const c of v.concepts) if (tileConcepts.has(c.id)) out.push({id: c.id, label: c.label});
  if (v.tiles.some(t => t.attention !== null && t.attention !== undefined)) out.push({id: 'attention', label: 'ABMIL attention'});
  return out;
}

function tileValue(t, layer, maxAttention) {
  if (layer === 'risk') return t.risk;
  if (layer === 'attention') return maxAttention > 0 ? (t.attention || 0) / maxAttention : 0;
  const z = t.concepts ? t.concepts[layer] : undefined;
  return z === undefined ? 0 : clamp01(z / 3);       // z <= 0 transparent; z >= 3 full
}

const hatch = '<defs><pattern id="hatch" width="40" height="40" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
  '<rect width="40" height="40" fill="var(--bl)"/><line x1="0" y1="0" x2="0" y2="40" stroke="var(--mut)" stroke-width="10"/></pattern></defs>';

export function heatmap(v, hm, assetUrl) {
  if (!v.image) return '<div class="note">No display image.</div>';
  const w = num(v.image.w), h = num(v.image.h), tile = num(v.image.tile);
  const [rows, cols] = v.image.grid.map(num);
  const maxA = Math.max(0, ...v.tiles.map(t => t.attention || 0));
  const byIdx = new Map(v.tiles.map(t => [t.idx, t]));
  const hl = new Set(hm.hl || []);
  let rects = '';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c, t = byIdx.get(idx);
      if (!t) {
        rects += `<rect class="hm-drop" data-tile="${idx}" x="${c * tile}" y="${r * tile}" width="${tile}" height="${tile}"><title>Region ${idx}: failed quality control</title></rect>`;
        continue;
      }
      const val = tileValue(t, hm.sel, maxA);
      rects += `<rect class="hm-tile" data-testid="hm-tile" data-tile="${idx}" data-v="${val.toFixed(4)}" x="${num(t.x)}" y="${num(t.y)}" width="${tile}" height="${tile}" fill-opacity="${(val * hm.opacity).toFixed(3)}"><title>Region ${idx}: ${esc(hm.sel)} ${val.toFixed(2)}</title></rect>`;
    }
  }
  for (const idx of hl) {
    const t = byIdx.get(idx);
    if (t) rects += `<rect class="hl" data-hl="${idx}" x="${num(t.x)}" y="${num(t.y)}" width="${tile}" height="${tile}"/>`;
  }
  return `<div class="hm"><img src="${esc(assetUrl(v.image.src))}" alt="Display image for ${esc(v.id)} (tile heatmap overlay)" data-testid="heatmap-img">` +
    `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">${hatch}${rects}</svg></div>`;
}

export function heatmapControls(v, hm) {
  const opts = layers(v).map(l => `<option value="${esc(l.id)}"${l.id === hm.sel ? ' selected' : ''}>${esc(l.label)}</option>`).join('');
  return `<div class="hmctl"><label>Show <select data-testid="heatmap-select" data-hm="sel">${opts}</select></label>` +
    `<label>Opacity <input type="range" min="0" max="1" step="0.05" value="${hm.opacity}" data-testid="heatmap-opacity" data-hm="opacity"></label>` +
    `<span class="legend">low<b></b>high</span>${hm.hl && hm.hl.length ? `<span class="sub">highlighted: regions ${hm.hl.map(esc).join(', ')}</span>` : ''}</div>`;
}

/** Margin strips: eosin when p > threshold, amber when p > threshold/2, else a thin green outline only. */
export function marginOverlay(v, assetUrl) {
  if (!v.image) return '<div class="note">No display image.</div>';
  const w = num(v.image.w), h = num(v.image.h);
  const m = v.margin;
  let rects = '';
  if (m) {
    const thr = m.threshold;               // null: color only the strongest segment of each involved margin
    for (const s of m.segments) {
      const hot = thr !== null ? s.p > thr : m.involved.includes(s.margin) && s.p >= m.perMargin[s.margin];
      const warm = thr !== null && s.p > thr / 2;
      const fill = hot ? 'var(--eos)' : warm ? 'var(--amb)' : 'none';
      const op = hot ? 0.75 : warm ? 0.45 : 0;
      const stroke = hot || warm ? 'var(--sf)' : 'var(--grn)';
      rects += `<rect class="mseg" data-testid="margin-seg" data-margin="${esc(s.margin)}" data-p="${num(s.p)}" data-involved="${hot && m.involved.includes(s.margin)}" x="${num(s.x)}" y="${num(s.y)}" width="${num(s.size)}" height="${num(s.size)}" fill="${fill}" fill-opacity="${op}" stroke="${stroke}"><title>${esc(s.margin)} segment p=${num(s.p).toFixed(2)}</title></rect>`;
    }
  }
  return `<div class="hm"><img src="${esc(assetUrl(v.image.src))}" alt="Display image for ${esc(v.id)} with margin strips" data-testid="surgical-img">` +
    `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">${rects}</svg></div>` +
    (m ? `<div class="mlab" data-testid="margin-labels">${MARGINS.map(name => {
      const inv = m.involved.includes(name);
      const len = m.lengths && m.lengths[name] ? ` · ≈${Math.round(m.lengths[name])} µm` : '';
      return `<div class="${inv ? 'inv' : ''}" data-testid="margin-label-${name}">${name[0].toUpperCase() + name.slice(1)} ${(m.perMargin[name] ?? 0).toFixed(2)}${len}</div>`;
    }).join('')}</div>` : '');
}
