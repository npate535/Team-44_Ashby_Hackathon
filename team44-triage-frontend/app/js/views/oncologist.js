// Oncologist inbox: signed-out biopsy summaries with the orders the pathologist placed.
import {esc} from '../adapter.js';

const STATUS = {none: 'no orders drafted', approved: 'drafted orders approved', edited_and_approved: 'orders edited and approved',
                dismissed: 'flag dismissed, no orders'};

export function render(ST) {
  const items = ST.routed.oncologist;
  if (!items.length) {
    return '<div data-testid="oncologist-inbox"><div class="card empty-state" data-testid="oncologist-empty"><h2>Inbox empty</h2><p>Signed-out biopsy results appear here.</p></div></div>';
  }
  const cards = items.map(r => {
    const orders = r.orders.length ? r.orders.map(o => `<span class="chip">${esc(o.items.join(', '))}</span>`).join('') : '<span class="sub">none</span>';
    const sentences = ((r.summary && r.summary.sentences) || []).map(s => `<p style="margin:5px 0">${esc(s.text)}</p>`).join('');
    return `<article class="card" data-testid="oncologist-item" data-case-id="${esc(r.case_id)}" style="margin-bottom:12px">` +
      `<h2>${esc(r.case_id)}</h2><div class="sub">Signed out ${esc(new Date(r.at).toLocaleString())} · ${esc(STATUS[r.agent_status] || r.agent_status)}${r.synthetic ? ' · synthetic' : ''}</div>` +
      `${sentences}<h3>Orders placed</h3><div class="chips">${orders}</div>` +
      '<p class="sub">Research prototype; suggestions only. The pathologist’s sign-out is the result of record.</p></article>';
  }).join('');
  return `<div data-testid="oncologist-inbox">${cards}</div>`;
}
