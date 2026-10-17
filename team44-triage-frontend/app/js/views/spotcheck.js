// Spot-check (plan §5.4, CELLS_SPEC §8.1): a qualitative check of top concept tiles and cell overlays.
// Items come from GET /api/spotcheck/items (artifacts spotcheck/items.json); they never carry class labels.
import {esc} from '../adapter.js';

const VERDICTS = [['acceptable', 'ok', 'Looks right'], ['not_acceptable', 'not_ok', 'Looks wrong'], ['unsure', 'unsure', 'Unsure']];

export function render(ST, ctx) {
  const spot = ST.spot;
  if (!spot) return '<div class="card" data-testid="spotcheck"><h2>Spot-check</h2><div class="note" data-testid="state-loading">Loading items…</div></div>';
  if (!spot.available || !spot.items.length) {
    return '<div class="card" data-testid="spotcheck"><h2>Spot-check</h2><div class="note" data-testid="spotcheck-empty">No spot-check items yet (step 08 writes spotcheck/items.json).</div></div>';
  }
  const groups = {};
  for (const item of spot.items) (groups[item.kind === 'cells' ? 'Cell overlays' : item.label] ||= []).push(item);
  const done = Object.keys(ST.spotDraft).length;
  let h = '<div class="card" data-testid="spotcheck"><h2>Spot-check (qualitative)</h2>' +
    '<p class="note">For each region, does the image show the named feature? This is a qualitative check, not a validation. ' +
    'Record whether a human or an AI reviewed: AI-reviewed checks are reported as such.</p>' +
    (spot.synthetic ? '<div class="mock">Synthetic items: generated placeholders, not tissue.</div>' : '') +
    `<div class="hmctl"><label>Reviewer <input type="text" maxlength="80" data-testid="spot-reviewer" id="spot-reviewer" placeholder="name" value="${esc(ST.spotReviewer || '')}"></label>` +
    `<label>Kind <select data-testid="spot-kind" id="spot-kind"><option value="human"${ST.spotKind === 'ai' ? '' : ' selected'}>human</option><option value="ai"${ST.spotKind === 'ai' ? ' selected' : ''}>ai</option></select></label>` +
    `<button class="btn p" data-spot="submit" data-testid="spot-submit">Submit ${done} review${done === 1 ? '' : 's'}</button>` +
    (ST.spotResult ? `<span class="verdict" data-testid="spot-result" style="margin:0">${esc(ST.spotResult)}</span>` : '') + '</div></div>';
  const names = Object.keys(groups);
  const current = names.includes(ST.spotGroup) ? ST.spotGroup : names[0];
  const opts = names.map(n => {
    const done = groups[n].filter(i => ST.spotDraft[i.item_id]).length;
    return `<option value="${esc(n)}"${n === current ? ' selected' : ''}>${esc(n)} (${done}/${groups[n].length})</option>`;
  }).join('');
  h += `<div class="hmctl" style="margin-top:14px"><label>Criterion <select data-testid="spot-group" data-spot-group="1">${opts}</select></label>` +
    '<span class="sub">one criterion at a time; marks are kept when you switch</span></div>';
  for (const [title, items] of Object.entries(groups)) {
    if (title !== current) continue;
    h += `<div class="card" style="margin-top:14px"><h2>${esc(title)}</h2><div class="cells-grid">`;
    for (const item of items) {
      const radios = VERDICTS.map(([value, tid, label]) => `<label><input type="radio" name="v-${esc(item.item_id)}" data-spot-item="${esc(item.item_id)}" data-spot-verdict="${value}" data-testid="spot-verdict-${tid}"${ST.spotDraft[item.item_id] === value ? ' checked' : ''}> ${label}</label>`).join(' ');
      h += `<div data-testid="spot-item" data-kind="${esc(item.kind)}" data-case-id="${esc(item.case_id)}" data-ref="${esc(item.item_id)}">` +
        `<img src="${esc(ctx.assetUrl(item.image))}" alt="${esc(item.label)} candidate region" loading="lazy">` +
        `<div class="sub">${esc(item.case_id)} · region ${esc(item.tile_idx)}${item.score !== null && item.score !== undefined ? ' · score ' + (+item.score).toFixed(2) : ''}</div>` +
        `<div class="edit">${radios}</div></div>`;
    }
    h += '</div></div>';
  }
  return h;
}

export async function submit(ST, api, toast) {
  const reviewer = (document.getElementById('spot-reviewer').value || '').trim();
  const kind = document.getElementById('spot-kind').value;
  ST.spotReviewer = reviewer; ST.spotKind = kind;
  const reviews = Object.entries(ST.spotDraft).map(([item_id, verdict]) => ({item_id, verdict}));
  if (!reviewer) { toast('Enter a reviewer name'); return; }
  if (!reviews.length) { toast('Mark at least one item'); return; }
  try {
    const res = await api.spotcheck({reviewer, reviewer_kind: kind, reviews});
    ST.spotResult = `Recorded ${res.recorded} review${res.recorded === 1 ? '' : 's'} (${res.reviewer_kind})`;
    ST.spotDraft = {};
  } catch (err) {
    toast(`Spot-check not saved: ${err.detail || err.message}`);
  }
}
