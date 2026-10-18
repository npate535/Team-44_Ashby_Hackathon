// Evidence review (spot-check, plan §5.4, CELLS_SPEC §8.1): a qualitative look at the regions the model scores highest
// for each named feature, and at cell-segmentation overlays. Items come from GET /api/spotcheck/items (artifacts
// spotcheck/items.json); they never carry class labels. Reviews are recorded with the reviewer's name and whether a
// human or an AI reviewed; this is a qualitative check, not a validation.
import { esc } from '../adapter.js';
import { spotVerdicts } from './metrics.js';

// Recorded reviews (eval/spotcheck.json via GET /api/metrics): fetched once per visit, refreshed after a submit.
let recorded = null;
let fetching = false;
let showRecorded = false; // off by default so a new review stays independent of earlier ones
function loadRecorded(ST) {
  if (recorded !== null || fetching || ST.mock || typeof fetch === 'undefined') return;
  fetching = true;
  const base = new URLSearchParams(location.search).get('api') || '';
  fetch(base + '/api/metrics', { headers: { Accept: 'application/json' } })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      recorded = (d && d.eval && d.eval.spotcheck) || { reviews: [] };
      window.dispatchEvent(new HashChangeEvent('hashchange')); // the app re-renders the current route
    })
    .catch(() => {
      recorded = { reviews: [], unavailable: true };
    })
    .finally(() => {
      fetching = false;
    });
}
const VERDICT_TEXT = { acceptable: 'shows it', not_acceptable: 'does not show it', unsure: 'unsure' };
function recordedPanel(sc, groups) {
  const v = spotVerdicts(sc);
  const ids = Object.keys(v.concepts);
  if (!ids.length && !v.cells) return '';
  const labelOf = {};
  for (const list of Object.values(groups)) for (const it of list) if (it.kind === 'concept') labelOf[it.target] = it.label;
  const bar = (b) => {
    const seg = (k, cls) => (b[k] ? `<span class="sc-seg sc-seg-${cls}" style="flex:${b[k]}" title="${b[k]} ${VERDICT_TEXT[k]}"></span>` : '');
    return `<span class="sc-stack" aria-hidden="true">${seg('acceptable', 'ok')}${seg('unsure', 'unsure')}${seg('not_acceptable', 'not_ok')}</span>`;
  };
  const row = (name, b) => `<li><span class="sc-rec-name">${esc(name)}</span>${bar(b)}<span class="sc-rec-n">${b.acceptable}/${b.n} show it${b.unsure ? `, ${b.unsure} unsure` : ''}</span></li>`;
  const ai = v.kinds.length === 1 && v.kinds[0] === 'ai';
  return `<section class="sc-recorded" aria-labelledby="sc-rec-h"><div class="sc-rec-head"><h3 id="sc-rec-h">Reviews already recorded</h3>
    <p>${ai ? '<b>AI-reviewed, not a pathologist.</b> ' : ''}${esc(v.reviewers.map((r) => (/\bAI\b/.test(r.name) ? r.name : `${r.name} (${r.kind === 'ai' ? 'AI' : r.kind})`)).join('; '))}. A plausibility check, not validation.</p>
    <label class="sc-toggle"><input type="checkbox" id="sc-show-recorded"${showRecorded ? ' checked' : ''}> Show recorded verdicts on each region</label></div>
    <ul class="sc-rec-list">${ids.map((id) => row(labelOf[id] || id, v.concepts[id])).join('')}${v.cells ? row('Cell overlays', v.cells) : ''}</ul>
    <p class="sc-fine"><span class="sc-key sc-seg-ok"></span>shows it <span class="sc-key sc-seg-unsure"></span>unsure <span class="sc-key sc-seg-not_ok"></span>does not show it</p></section>`;
}

const VERDICTS = [
  ['acceptable', 'ok', 'Yes'],
  ['not_acceptable', 'not_ok', 'No'],
  ['unsure', 'unsure', 'Unsure']
];
const groupOf = (item) => (item.kind === 'cells' ? 'Cell overlays' : item.label);

function progressBar(done, total) {
  const pct = total ? Math.round((100 * done) / total) : 0;
  return `<span class="sc-bar" role="img" aria-label="${done} of ${total} marked"><span style="width:${pct}%"></span></span>`;
}

export function render(ST, ctx) {
  const spot = ST.spot;
  if (!spot)
    return '<div class="sc card" data-testid="spotcheck"><h2>Evidence review</h2><p class="note" data-testid="state-loading">Loading review items…</p></div>';
  if (!spot.available || !spot.items.length) {
    return '<div class="sc card" data-testid="spotcheck"><h2>Evidence review</h2><p class="note" data-testid="spotcheck-empty">No review items yet. The case-building step (08) writes spotcheck/items.json with the highest-scoring regions per feature.</p></div>';
  }
  const groups = {};
  for (const item of spot.items) (groups[groupOf(item)] ||= []).push(item);
  loadRecorded(ST);
  const sc = recorded || (ST.metrics && ST.metrics.eval && ST.metrics.eval.spotcheck) || null;
  const priorByItem = {};
  for (const r of (sc && Array.isArray(sc.reviews) ? sc.reviews : [])) if (r && r.item_id) (priorByItem[r.item_id] ||= []).push(r);
  const names = Object.keys(groups);
  const current = names.includes(ST.spotGroup) ? ST.spotGroup : names[0];
  const marked = (list) => list.filter((i) => ST.spotDraft[i.item_id]).length;
  const done = Object.keys(ST.spotDraft).length;
  const total = spot.items.length;

  let h = '<div class="sc" data-testid="spotcheck">';
  h += `<section class="sc-intro"><div class="sc-intro-text"><h2>Concept region review</h2>
    <p>Each image is a region the model scored highest for one named feature (or a cell-segmentation overlay). Mark whether you see that feature. This is a qualitative check of the explanations, not a validation, and no diagnosis is shown here.</p>
    <p class="sc-fine">Your marks stay in this browser tab until you record them. Each record states who reviewed and whether the reviewer was a human or an AI; AI-reviewed checks are reported as such.</p></div>
    <div class="sc-total"><span class="sc-total-n">${done}<small> of ${total}</small></span><span class="sc-total-l">regions marked</span>${progressBar(done, total)}</div></section>`;
  if (spot.synthetic) h += '<div class="mock">Synthetic items: generated placeholders, not tissue.</div>';
  if (sc) h += recordedPanel(sc, groups);

  h += `<section class="sc-record" aria-label="Record your review">
    <label class="sc-field"><span>Reviewer</span><input type="text" maxlength="80" data-testid="spot-reviewer" id="spot-reviewer" placeholder="Your name" autocomplete="name" value="${esc(ST.spotReviewer || '')}"></label>
    <label class="sc-field"><span>Reviewed by</span><select data-testid="spot-kind" id="spot-kind"><option value="human"${ST.spotKind === 'ai' ? '' : ' selected'}>a human</option><option value="ai"${ST.spotKind === 'ai' ? ' selected' : ''}>an AI</option></select></label>
    <button class="btn p" data-spot="submit" data-testid="spot-submit"${done ? '' : ' aria-disabled="true"'}>Record ${done} review${done === 1 ? '' : 's'}</button>
    ${ST.spotResult ? `<span class="sc-result" role="status" data-testid="spot-result">${esc(ST.spotResult)}</span>` : ''}
  </section>`;

  const opts = names.map((n) => `<option value="${esc(n)}"${n === current ? ' selected' : ''}>${esc(n)} (${marked(groups[n])}/${groups[n].length})</option>`).join('');
  h += `<section class="sc-criteria" aria-label="Choose a feature to review">
    <label class="sc-field sc-select"><span>Feature</span><select data-testid="spot-group" data-spot-group="1">${opts}</select></label>
    <div class="sc-chips" role="list">${names
      .map((n) => {
        const m = marked(groups[n]);
        const all = groups[n].length;
        return `<button type="button" role="listitem" class="sc-chip${n === current ? ' sc-chip-on' : ''}${m === all ? ' sc-chip-done' : ''}" data-sc-group="${esc(n)}"${n === current ? ' aria-current="true"' : ''}><span>${esc(n)}</span><small>${m}/${all}</small></button>`;
      })
      .join('')}</div>
    <p class="sc-fine">One feature at a time; marks are kept when you switch.</p>
  </section>`;

  const items = groups[current];
  const isCells = items[0] && items[0].kind === 'cells';
  h += `<section class="sc-group" aria-labelledby="sc-q"><header><h3 id="sc-q">${isCells ? 'Do the drawn nuclei outlines and types look plausible?' : `Does this region show: ${esc(current)}?`}</h3>
    <p>${items.length} regions${isCells ? ' with model-drawn nuclei' : ', highest concept score first'}; ${marked(items)} marked</p></header><div class="sc-grid">`;
  for (const item of items) {
    const verdict = ST.spotDraft[item.item_id] || '';
    const radios = VERDICTS.map(
      ([value, tid, label]) =>
        `<label class="sc-v sc-v-${tid}"><input type="radio" name="v-${esc(item.item_id)}" data-spot-item="${esc(item.item_id)}" data-spot-verdict="${value}" data-testid="spot-verdict-${tid}"${verdict === value ? ' checked' : ''}> ${label}</label>`
    ).join('');
    h += `<article class="sc-item" data-testid="spot-item" data-kind="${esc(item.kind)}" data-case-id="${esc(item.case_id)}" data-ref="${esc(item.item_id)}"${verdict ? ` data-verdict="${esc(verdict)}"` : ''}>
      <img src="${esc(ctx.assetUrl(item.image))}" alt="${esc(item.label)}: candidate region ${esc(item.tile_idx)} from ${esc(item.case_id)}" loading="lazy">
      <p class="sc-meta"><span>${esc(item.case_id)}</span><span>region ${esc(item.tile_idx)}</span>${item.score !== null && item.score !== undefined ? `<span>score ${(+item.score).toFixed(2)}</span>` : ''}</p>
      ${showRecorded && priorByItem[item.item_id] ? `<ul class="sc-prior">${priorByItem[item.item_id].map((r) => `<li><b>${esc(r.reviewer_kind === 'ai' ? 'AI' : r.reviewer_kind || 'Reviewer')}: ${esc(VERDICT_TEXT[r.verdict] || r.verdict)}</b>${r.note ? ` (${esc(r.note)})` : ''}<span> ${esc(r.reviewer || '')}</span></li>`).join('')}</ul>` : ''}
      <fieldset class="sc-verdicts"><legend class="sr-only">Does region ${esc(item.tile_idx)} of ${esc(item.case_id)} show ${esc(item.label)}?</legend>${radios}</fieldset>
    </article>`;
  }
  h += '</div></section></div>';
  return h;
}

export async function submit(ST, api, toast) {
  const reviewer = (document.getElementById('spot-reviewer').value || '').trim();
  const kind = document.getElementById('spot-kind').value;
  ST.spotReviewer = reviewer;
  ST.spotKind = kind;
  const reviews = Object.entries(ST.spotDraft).map(([item_id, verdict]) => ({ item_id, verdict }));
  if (!reviewer) {
    toast('Enter a reviewer name before recording');
    document.getElementById('spot-reviewer').focus();
    return;
  }
  if (!reviews.length) {
    toast('Mark at least one region first');
    return;
  }
  try {
    const res = await api.spotcheck({ reviewer, reviewer_kind: kind, reviews });
    ST.spotResult = `Recorded ${res.recorded} review${res.recorded === 1 ? '' : 's'} (${res.reviewer_kind})`;
    ST.spotDraft = {};
    recorded = null; // refetch the merged eval/spotcheck.json
  } catch (err) {
    toast(`Review not recorded: ${err.detail || err.message}`);
  }
}

// Feature chips drive the same <select> the app already listens to (change on [data-spot-group]).
if (typeof document !== 'undefined' && !document.__scInstalled) {
  document.__scInstalled = true;
  document.addEventListener('change', (e) => {
    if (!e.target || e.target.id !== 'sc-show-recorded') return;
    showRecorded = e.target.checked;
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  });
  document.addEventListener('click', (e) => {
    const chip = e.target && e.target.closest && e.target.closest('[data-sc-group]');
    if (!chip) return;
    const select = document.querySelector('select[data-spot-group]');
    if (!select) return;
    select.value = chip.dataset.scGroup;
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
}
