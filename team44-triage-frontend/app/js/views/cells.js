// Cells tab (CELLS_SPEC §7.3): top tiles with the overlay toggle, type legend, per-tile counts, measured features
// with reference-percentile bars. Wording: "measured features associated with invasive carcinoma".
import {cssColor, cssUrl, esc, humanize, num, signed} from '../adapter.js';

function crop(v, t, assetUrl) {
  const {w, h, tile} = v.image;
  const size = `${(w / tile) * 100}% ${(h / tile) * 100}%`;
  const pos = `${w > tile ? (t.x / (w - tile)) * 100 : 0}% ${h > tile ? (t.y / (h - tile)) * 100 : 0}%`;
  return `<div class="crop" role="img" aria-label="Region ${t.idx} of the display image" style="background-image:url('${esc(cssUrl(assetUrl(v.image.src)))}');background-size:${size};background-position:${pos}"></div>`;
}

function value(f) {
  if (f.value === null || f.value === undefined) return 'not measured';
  const n = Math.abs(f.value) >= 100 ? Math.round(f.value).toLocaleString() : (+f.value).toFixed(2);
  return n + (f.unit ? ' ' + f.unit : '');
}

export function cellsTab(v, ST, assetUrl) {
  const cells = v.cells;
  const cfg = ST.cfg;
  if (!cells || cells.status !== 'ok') {
    return `<div class="card" data-testid="cells-tab"><h2>Cells</h2><div class="note" data-testid="cells-disabled">Cell measurements unavailable for this case (${cells && cells.status === 'failed' ? 'segmentation failed' : 'module disabled'}).</div></div>`;
  }
  const types = cfg.cells.types;
  const colors = cfg.cells.overlay_colors;
  const typeName = t => (cfg.labels.cell_types && cfg.labels.cell_types[t]) || humanize(t);
  const legend = `<div class="legend" data-testid="cells-legend" style="flex-wrap:wrap">${types.map(t => `<span data-testid="cells-legend-item"><i class="sw" style="background:${cssColor(colors[t])}"></i>${esc(typeName(t))}</span>`).join('')}</div>`;
  const tileById = new Map(v.tiles.map(t => [t.idx, t]));
  const figs = cells.tiles.map(ct => {
    const t = tileById.get(ct.idx);
    const img = ST.cellsOverlay ? `<img src="${esc(assetUrl(ct.overlay))}" alt="Cell overlay, region ${num(ct.idx)}" data-testid="cells-overlay-img">`
      : (t ? crop(v, t, assetUrl) : '<div class="note">Region not in the display grid.</div>');
    const counts = types.map(type => `<div class="kv"><span><i class="sw" style="background:${cssColor(colors[type])};display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px"></i>${esc(typeName(type))}</span><b>${num(ct.counts[type])}</b></div>`).join('');
    return `<figure data-testid="cells-tile" data-tile="${num(ct.idx)}"${ST.cellsTile === ct.idx ? ' class="flash"' : ''}>${img}<figcaption class="sub">Region ${num(ct.idx)}</figcaption>${counts}</figure>`;
  }).join('');
  const label = id => (cfg.labels.cell_features && cfg.labels.cell_features[id])
    || (v.concepts.find(c => c.id === id) || {}).label || humanize(id);
  const feats = Object.entries(cells.features || {}).map(([id, f]) => {
    const p = f.ref_percentile;
    const bar = p === null || p === undefined ? '<div class="sub">no reference percentile</div>'
      : `<div class="pctbar" aria-hidden="true"><b style="left:${(p * 100).toFixed(1)}%"></b></div><div class="sub">higher than ${Math.round(p * 100)}% of benign reference cases</div>`;
    return `<div class="act" data-testid="cells-feature-${esc(id)}"><b>${esc(label(id))}</b> <span class="src m">◆ measured</span>` +
      `<div class="kv"><span>Value</span><b>${esc(value(f))}</b></div>${bar}` +
      `<div class="sub">contribution ${f.contribution === null || f.contribution === undefined ? 'not used by the model' : signed(f.contribution)}</div></div>`;
  }).join('');
  const flag = cells.flags && cells.flags.insufficient_tumor_nuclei
    ? '<div class="flag">Too few tumor nuclei for nuclear morphology features; those are left blank.</div>' : '';
  return `<div class="card" data-testid="cells-tab"><h2>Measured features associated with invasive carcinoma</h2>` +
    `<div class="sub">${esc(cells.model || '')} at ${esc(cells.mpp_model)} µm/px · measurements describe the selected regions; they are associations, not causes.</div>` +
    `<div class="hmctl"><label><input type="checkbox" data-testid="cells-overlay-toggle" data-cells="overlay"${ST.cellsOverlay ? ' checked' : ''}> Show cell overlay</label>${legend}</div>` +
    `${flag}<div class="cells-grid">${figs}</div><h3>Features</h3>${feats || '<div class="note">No features.</div>'}</div>`;
}
