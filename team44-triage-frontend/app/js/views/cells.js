// Cells tab (CELLS_SPEC §7.3): measured-cell summary, per-region composition and overlays, and every measured feature
// against benign/normal reference cases. Automated segmentation, not pathologist-validated; wording: "measured
// features associated with invasive carcinoma", never a causal claim about cells.
import { cssColor, cssUrl, esc, humanize, num, ordinalPct, signed } from '../adapter.js';
import { fmtPct, refShare } from './why.js';

const isNum = (x) => x !== null && x !== undefined && Number.isFinite(Number(x));

// Display grouping only (labels, units and directions come from config/cell_features.yaml via /api/config).
const GROUPS = [
  { label: 'Tumor cells', ids: ['neo_fraction', 'neo_density'] },
  { label: 'Nuclear size and shape', ids: ['nuc_area_mean', 'nuc_area_cv', 'nuc_shape_cv'] },
  { label: 'Immune context', ids: ['infl_fraction', 'til_proximity'] },
  { label: 'Other cell populations', ids: ['dead_fraction', 'epi_fraction'] }
];

function crop(v, t, assetUrl) {
  const [w, h, tile] = [num(v.image.w), num(v.image.h), num(v.image.tile)];
  t = { idx: num(t.idx), x: num(t.x), y: num(t.y) };
  const size = `${(w / tile) * 100}% ${(h / tile) * 100}%`;
  const pos = `${w > tile ? (t.x / (w - tile)) * 100 : 0}% ${h > tile ? (t.y / (h - tile)) * 100 : 0}%`;
  return `<div class="crop cv-crop" role="img" aria-label="Region ${t.idx} of the display image, no overlay" style="background-image:url('${esc(cssUrl(assetUrl(v.image.src)))}');background-size:${size};background-position:${pos}"></div>`;
}

function fmtValue(f, unit) {
  if (!f || !isNum(f.value)) return 'not measured';
  const x = Number(f.value);
  const u = unit || f.unit || '';
  if (!u && x >= 0 && x <= 1) return `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
  const n = Math.abs(x) >= 100 ? Math.round(x).toLocaleString() : x.toFixed(2);
  return n + (u ? ' ' + u : '');
}

function pctTrack(p) {
  if (!isNum(p)) return '<div class="sub">No reference percentile.</div>';
  const q = Math.max(0, Math.min(1, Number(p)));
  const cls = q >= 0.9 ? 'hi' : q <= 0.1 ? 'lo' : 'mid';
  return `<div class="pct-track ${cls}" aria-hidden="true"><span class="pct-band" title="Middle 80% of benign/normal reference cases"></span><b style="left:${(q * 100).toFixed(1)}%"></b></div>`;
}

/** One percentile axis for the whole feature list, in the reference column (the rows draw only their tracks). */
const pctAxis = (useCol) =>
  `<div class="cf-row cf-axis" aria-hidden="true"><span></span><span></span><div class="pct-ticks">${[0, 10, 50, 90, 100]
    .map((t) => `<span style="left:${t}%">${t}</span>`)
    .join('')}</div>${useCol ? '<span></span>' : ''}</div>`;

/** AI spot check of the cell overlays (eval/spotcheck.json summary.cells), if recorded. */
function spotNote(metrics) {
  const sc = metrics && metrics.eval && metrics.eval.spotcheck;
  const e = sc && sc.summary ? sc.summary.cells : null;
  if (!e || !isNum(e.n) || !isNum(e.acceptable) || Number(e.n) <= 0) return '';
  return ` AI spot check of overlays (AI-reviewed, not a pathologist): ${Number(e.acceptable)} of ${Number(e.n)} judged acceptable.`;
}

export function cellsTab(v, ST, assetUrl) {
  const cells = v.cells;
  const cfg = ST.cfg;
  if (!cells || cells.status !== 'ok') {
    const unavailable = !(cfg.cells && cfg.cells.enabled)
      ? 'the cell module is disabled'
      : cells && cells.status === 'failed'
        ? 'automated nuclei segmentation did not complete for this case'
        : v.reliability.qcTiles === 0
        ? 'no regions passed quality control'
        : 'no measured output is available for this case';
    return `<div class="card cv-cells" data-testid="cells-tab"><div class="card-h"><h2>Measured cells</h2></div><div class="note" data-testid="cells-disabled">${v.external ? 'Cell measurements are not computed for TCGA fields.' : `Cell measurements are unavailable: ${unavailable}.`} The read and its explanation do not depend on them.</div></div>`;
  }
  const types = cfg.cells.types;
  const colors = cfg.cells.overlay_colors;
  const typeName = (t) => (cfg.labels.cell_types && cfg.labels.cell_types[t]) || humanize(t);
  const meta = cfg.cell_features || {};
  const label = (id) =>
    (cfg.labels.cell_features && cfg.labels.cell_features[id]) || (meta[id] && meta[id].label) || humanize(id);
  const tiles = cells.tiles || [];
  const total = (ct) => types.reduce((a, t) => a + num(ct.counts && ct.counts[t]), 0);
  const allN = tiles.reduce((a, ct) => a + total(ct), 0);
  const feats = cells.features || {};

  const legend = `<div class="legend cells-legend" data-testid="cells-legend">${types
    .map((t) => `<span data-testid="cells-legend-item"><i class="sw" style="background:${cssColor(colors[t])}"></i>${esc(typeName(t))}</span>`)
    .join('')}</div>`;

  const stat = (id, title) => {
    const f = feats[id];
    if (!f || !isNum(f.value)) return '';
    const p = f.ref_percentile;
    return (
      `<div class="cs-stat"><span class="cs-k">${esc(title || label(id))}</span><b>${esc(fmtValue(f, meta[id] && meta[id].unit))}</b>` +
      `<span class="sub">${isNum(p) ? `${ordinalPct(p)} percentile of benign/normal references` : 'no reference percentile'}</span></div>`
    );
  };
  const stats =
    `<div class="cs-stats"><div class="cs-stat"><span class="cs-k">Nuclei segmented</span><b>${allN.toLocaleString()}</b><span class="sub">in ${tiles.length} selected region${tiles.length === 1 ? '' : 's'}</span></div>` +
    stat('neo_fraction') +
    stat('nuc_area_cv') +
    (feats.til_proximity && isNum(feats.til_proximity.value) ? stat('til_proximity') : stat('infl_fraction')) +
    '</div>';

  const cnt = (ct, t) => num(ct.counts && ct.counts[t]);
  const bar = (ct) => {
    const n = total(ct);
    if (!n) return '<div class="sub">No nuclei counted.</div>';
    return `<div class="comp-bar" role="img" aria-label="${esc(types.map((t) => `${typeName(t)} ${cnt(ct, t)}`).join(', '))}">${types
      .filter((t) => cnt(ct, t) > 0)
      .map((t) => `<i style="flex:${cnt(ct, t)};background:${cssColor(colors[t])}" title="${esc(typeName(t))}: ${cnt(ct, t)}"></i>`)
      .join('')}</div>`;
  };
  const tileById = new Map(v.tiles.map((t) => [t.idx, t]));
  const figs = tiles
    .map((ct) => {
      const t = tileById.get(ct.idx);
      const img = ST.cellsOverlay
        ? `<img src="${esc(assetUrl(ct.overlay))}" alt="Cell overlay for region ${num(ct.idx)}: nuclei outlined and coloured by type" data-testid="cells-overlay-img" data-cv-img>`
        : t
          ? crop(v, t, assetUrl)
          : '<div class="note">Region not in the display grid.</div>';
      const counts = types
        .map(
          (type) =>
            `<li><span><i class="sw" style="background:${cssColor(colors[type])}"></i>${esc(typeName(type))}</span><b>${cnt(ct, type).toLocaleString()}</b></li>`
        )
        .join('');
      return (
        `<figure class="cv-cell-fig${ST.cellsTile === ct.idx ? ' is-target' : ''}" data-testid="cells-tile" data-tile="${num(ct.idx)}">${img}` +
        `<figcaption><b>Region ${num(ct.idx)}</b><span class="sub">${total(ct).toLocaleString()} nuclei${t && isNum(t.risk) ? ` · tile P(Invasive) ${fmtPct(t.risk)}` : ''}</span></figcaption>` +
        `${bar(ct)}<ul class="cell-counts">${counts}</ul></figure>`
      );
    })
    .join('');

  const listed = new Set(GROUPS.flatMap((g) => g.ids));
  const extra = Object.keys(feats).filter((id) => !listed.has(id));
  const shownGroups = [...GROUPS, ...(extra.length ? [{ label: 'Other measured features', ids: extra }] : [])]
    .map((g) => ({ ...g, ids: g.ids.filter((id) => id in feats) }))
    .filter((g) => g.ids.length);
  const shownIds = shownGroups.flatMap((g) => g.ids);
  const unused = (f) => !f || f.contribution === null || f.contribution === undefined;
  // said once above the list instead of on every row: every listed feature is unused by the model / measured
  const noneUsed = shownIds.length > 0 && shownIds.every((id) => unused(feats[id]));
  const anyMeasured = shownIds.some((id) => feats[id] && isNum(feats[id].value));
  const row = (id) => {
    const f = feats[id];
    const measured = f && isNum(f.value);
    const p = f ? f.ref_percentile : null;
    const dir = meta[id] && meta[id].direction;
    const ref = isNum(p) ? refShare(p) : 'no reference percentile';
    const contrib = unused(f) ? 'not used by the prediction model' : `concept-model push ${signed(f.contribution)}`;
    return (
      `<div class="cf-row" data-testid="cells-feature-${esc(id)}">` +
      `<div class="cf-name"><b>${esc(label(id))}</b>${measured ? '' : '<span class="src">unavailable</span>'}${dir === 'context' ? '<span class="src ctx">context</span>' : ''}</div>` +
      `<div class="cf-val"><b>${esc(fmtValue(f, meta[id] && meta[id].unit))}</b></div>` +
      `<div class="cf-ref">${pctTrack(p)}<span class="sub">${esc(ref)}</span></div>` +
      (noneUsed ? '' : `<div class="cf-use sub">${esc(contrib)}</div>`) +
      '</div>'
    );
  };
  const groups = shownGroups.map((g) => `<section class="cf-group"><h3>${esc(g.label)}</h3>${g.ids.map(row).join('')}</section>`).join('');
  const cfNote = anyMeasured
    ? `<span class="src m">◆ measured</span> Values are measured from the segmented nuclei${noneUsed ? ' and are not used by the prediction model' : ''}.`
    : noneUsed
      ? 'These features are not used by the prediction model.'
      : '';

  const flag =
    cells.flags && cells.flags.insufficient_tumor_nuclei
      ? '<div class="flag">Too few tumor nuclei for nuclear morphology features; those are left blank.</div>'
      : '';
  return (
    `<div class="card cv-cells" data-testid="cells-tab"><div class="card-h"><h2>Measured features associated with invasive carcinoma</h2>` +
    `<span class="sub">${esc(cells.model || '')}${cells.mpp_model ? ` at ${esc(cells.mpp_model)} µm/px` : ''}</span></div>` +
    `<p class="cells-warn" role="note"><b>Not pathologist-validated.</b> Nuclei are segmented and typed automatically (PanNuke classes) on the selected regions only. Counts and measurements are estimates; they are associations with the read, not causes.${spotNote(ST.metrics)}</p>` +
    stats +
    `<div class="cells-ctl"><label class="cells-toggle"><input type="checkbox" data-testid="cells-overlay-toggle" data-cells="overlay"${ST.cellsOverlay ? ' checked' : ''}> Show cell overlay</label>${legend}</div>` +
    `${flag}<div class="cells-grid cv-cells-grid">${figs}</div>` +
    `<h3 class="cf-h">Measured features against benign/normal reference cases</h3><p class="sub cf-intro">The marker shows this case's percentile among benign/normal reference cases; the shaded band is their middle 80%.</p>` +
    (cfNote ? `<p class="sub cf-note" data-testid="cells-features-note">${cfNote}</p>` : '') +
    `<div class="cell-features cv-cf${noneUsed ? ' cf-nouse' : ''}">${groups ? pctAxis(!noneUsed) + groups : '<div class="note">No features.</div>'}</div></div>`
  );
}
