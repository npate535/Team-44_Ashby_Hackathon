// Oncology inbox: one readable report card per signed-out biopsy — suggested category with calibrated probability and
// confidence set, key associated features, the orders the pathologist placed, the sign-out time, grounded summary
// sentences and the disclaimer. Unread/read state is a per-viewer convenience (browser storage, never required).
// Research prototype — not for clinical use. Ground truth is never fetched or rendered.
import { esc, isLLM } from '../adapter.js';
import { conformalInfo, pctText, routedKey, shortClass, suggestedLabel } from '../flow.js';

const ORDER_TYPE = { stain: 'Stains', recut: 'Recut', deeper_levels: 'Deeper levels', new_sample: 'New sample' };
const AGENT_NOTE = {
  none: 'No follow-up orders were drafted for this case.',
  approved: 'Drafted orders approved by the pathologist.',
  edited_and_approved: 'Drafted orders edited, then approved by the pathologist.',
  dismissed: 'The pathologist dismissed the flag; no orders were placed.'
};

const when = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return `${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · ${d.toLocaleDateString([], { month: 'short', day: 'numeric' })}`;
};

function probOf(s, cls) {
  if (!s || !s.classes || !s.probs) return null;
  const i = s.classes.indexOf(cls);
  return i < 0 ? null : s.probs[i];
}

function topOf(s) {
  if (!s || !s.probs || !s.probs.length) return null;
  return s.classes[s.probs.indexOf(Math.max(...s.probs))];
}

function features(rep) {
  const list = (rep && rep.features) || [];
  if (!list.length) return '<p class="rep-none">No associated features were recorded for this read.</p>';
  return (
    '<ul class="rep-feat">' +
    list
      .map((f) => {
        const toward = Number(f.contribution) > 0;
        const dir = f.direction === 'context' ? 'context feature' : toward ? 'toward invasive' : 'away from invasive';
        const tone = f.direction === 'context' ? 'ctx' : toward ? 'up' : 'down';
        return `<li><span>${esc(f.label)}</span><span class="dir ${tone}">${esc(dir)}</span></li>`;
      })
      .join('') +
    '</ul>'
  );
}

function orders(r) {
  const placed = r.orders || [];
  if (!placed.length) return `<p class="rep-none">${esc(AGENT_NOTE[r.agent_status] || 'No orders placed.')}</p>`;
  return (
    '<ul class="rep-orders">' +
    placed
      .map(
        (o) =>
          `<li data-testid="inbox-order"><span class="otype">${esc(ORDER_TYPE[o.type] || o.type)}</span><b>${esc((o.items || []).join(', '))}</b><span class="ostatus">Ordered</span></li>`
      )
      .join('') +
    `</ul><p class="rep-note">${esc(AGENT_NOTE[r.agent_status] || '')}</p>`
  );
}

function card(r, ST, conf) {
  const key = routedKey(r);
  const unread = !ST.inboxRead.has(key);
  const s = r.suggestion || {};
  const rep = r.report || {};
  const top = rep.top_class || topOf(s);
  const pInv = probOf(s, (ST.cfg && ST.cfg.aggressive_class) || 'Invasive');
  const set = s.conformal_set || [];
  const chips = set.length
    ? set.map((c) => `<span class="chip${c === 'Invasive' ? ' inv' : ''}" data-testid="inbox-set-chip">${esc(shortClass(c))}</span>`).join('')
    : '<span class="chip empty">None</span>';
  const sentences = ((r.summary && r.summary.sentences) || []).map((x) => `<li>${esc(x.text)}</li>`).join('');
  const gen = r.summary && r.summary.generator;
  // generator: "template" | "lumen:<model>" | "lumen:<model>+template" (some sentences from the template fallback)
  const model = isLLM(gen) ? String(gen).slice(6).split('+')[0] : null;
  const genText = model
    ? `Written by ${esc(model)} from structured findings only${String(gen).includes('+template') ? ', with template sentences' : ''}`
    : 'Template summary from structured findings';
  const q = rep.quality || {};
  const quality = q.reliability_flag
    ? '<span class="qflag">Reliability flag raised on this read</span>'
    : q.qc_tiles_kept !== undefined && q.qc_tiles_kept < 3
      ? '<span class="qflag">Little tissue passed quality control</span>'
      : '';
  return (
    `<article class="rep${unread ? ' unread' : ''}" data-testid="oncologist-item" data-case-id="${esc(r.case_id)}" data-inbox-key="${esc(key)}" data-unread="${unread}">` +
    `<header class="rep-head"><span class="udot" aria-hidden="true"></span><h2><a href="#/case/${encodeURIComponent(r.case_id)}">${esc(r.case_id)}</a></h2>` +
    `<span class="rep-kind">Breast biopsy · BACH field${r.synthetic ? ' · synthetic' : ''}</span>` +
    `<span class="rep-time" data-testid="inbox-signed-out">Signed out ${esc(when(r.at))}</span>` +
    (unread ? `<span class="sr-only">Unread.</span><button class="linkbtn rep-mark" data-inbox-read="${esc(key)}">Mark as read</button>` : '') +
    '</header>' +
    '<div class="rep-grid">' +
    `<section class="rep-cell"><h3>Suggested category</h3><p class="rep-cat${top === 'Invasive' ? ' inv' : ''}" data-testid="inbox-category">${esc(suggestedLabel(top))}</p>` +
    `<p class="rep-sub">P(Invasive) <b>${pctText(pInv)}</b>, calibrated</p></section>` +
    `<section class="rep-cell"><h3>${esc(conf.label)}</h3><p class="chips">${chips}</p>` +
    `<p class="rep-sub">${set.length > 1 ? 'More than one category cannot be ruled out.' : set.length === 1 ? 'One category in the set.' : 'No category reached the threshold.'}${conf.coverage !== null ? ` Held-out coverage ${Math.round(conf.coverage * 100)}%${conf.n ? ` (n = ${conf.n})` : ''}.` : ''}</p></section>` +
    `<section class="rep-cell"><h3>Key associated features</h3>${features(rep)}</section>` +
    `<section class="rep-cell"><h3>Follow-up orders</h3>${orders(r)}</section>` +
    '</div>' +
    `<section class="rep-sum"><h3>Summary <span>${genText}</span></h3>${sentences ? `<ul>${sentences}</ul>` : '<p class="rep-none">No summary sentences.</p>'}</section>` +
    `<footer class="rep-foot">${quality}<span>Research prototype. Suggestions only: the pathologist’s sign-out is the result of record. “Associated with” does not mean caused by.</span><a class="case-link" href="#/case/${encodeURIComponent(r.case_id)}">Open case</a></footer>` +
    '</article>'
  );
}

export function render(ST) {
  const items = ST.routed.oncologist;
  if (!items.length) {
    return '<div class="inbox" data-testid="oncologist-inbox"><div class="card empty-state" data-testid="oncologist-empty"><h2>No results yet</h2><p>Biopsies appear here as soon as the pathologist signs them out, with the orders that were placed.</p></div></div>';
  }
  const conf = conformalInfo(ST.metrics, ST.cfg);
  const unread = items.filter((r) => !ST.inboxRead.has(routedKey(r))).length;
  const head =
    `<div class="inbox-head"><p><b>${items.length}</b> signed-out result${items.length === 1 ? '' : 's'}${unread ? ` · <b class="u">${unread} unread</b>` : ''}. Newest first.</p>` +
    (unread ? '<button class="btn" data-inbox-read-all data-testid="inbox-read-all">Mark all as read</button>' : '') +
    '</div>';
  return `<div class="inbox" data-testid="oncologist-inbox">${head}${items.map((r) => card(r, ST, conf)).join('')}</div>`;
}
