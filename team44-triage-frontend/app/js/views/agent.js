// Agent panel (biopsy): drafted orders with rationale + rule ID; Order tests / Edit / Take a closer look first /
// Dismiss flag; the teammate's "Pathologist approval" card around Sign out. Nothing is ordered until approved.
// Surgical: only the visibly blocked actions (stains are never drafted in the simulated frozen-section lane).
import {esc} from '../adapter.js';

const STATUS = {none: 'No orders drafted', drafted: 'Drafted — awaiting your decision', approved: 'Orders approved',
                edited_and_approved: 'Edited and approved', deferred_closer_look: 'Closer look first',
                dismissed: 'Flag dismissed'};
const ORDER_TYPE = {stain: 'Stain', recut: 'Recut', deeper_levels: 'Deeper levels', new_sample: 'New sample'};
const ORDER_STATUS = {drafted: 'drafted', approved: 'ordered', edited: 'ordered (edited)', removed: 'removed',
                      dismissed: 'dismissed'};

function blockedRows(v) {
  const rows = v.agent.blocked.map(b => `<div class="act blk" data-testid="agent-blocked"><b>Blocked</b> ${esc(b.action)}<div class="sub">${esc(b.reason)}${b.rule_id ? ' · ' + esc(b.rule_id) : ''}</div></div>`);
  if (v.lane === 'surgical') {
    rows.push('<div class="act blk" data-testid="agent-guard"><b>Guard</b> no stain orders in the simulated frozen-section lane — margin call only<div class="sub">Agent actions are blocked here by design.</div></div>');
  }
  return rows.join('');
}

export function agentPanel(v, ST) {
  const head = '<h2>Drafted follow-up orders</h2><div class="note">Illustrative rules; require clinical review. Nothing is ordered until you approve.</div>';
  if (v.lane === 'surgical') {
    return `<div class="card" data-testid="agent-panel">${head}<div style="margin-top:8px">${blockedRows(v)}</div></div>`;
  }
  const status = v.agent.status;
  const pill = `<span class="pill${status === 'drafted' ? ' warn' : status === 'none' ? '' : ' ok'}" data-testid="agent-status" data-status="${esc(status)}">${esc(STATUS[status] || status)}</span>`;
  const detail = v.agent.orderStatus || v.agent.orders.map(o => ({...o, status: 'drafted', drafted_items: o.items}));
  const rationale = Object.fromEntries(v.agent.orders.map(o => [o.rule_id, o.rationale]));
  const editing = ST.editing === 'orders';
  let orders = '';
  for (const o of detail) {
    const off = o.status === 'removed' || o.status === 'dismissed';
    const items = editing
      ? `<div class="edit">${o.drafted_items.map(item => `<label><input type="checkbox" data-edit-rule="${esc(o.rule_id)}" data-edit-item="${esc(item)}"${(ST.editDraft[o.rule_id] || []).includes(item) ? ' checked' : ''} data-testid="edit-item">${esc(item)}</label>`).join('')}</div>`
      : `<b>${esc(o.items.join(', ') || o.drafted_items.join(', '))}</b>`;
    orders += `<div class="act${off ? ' off' : ''}" data-testid="agent-order" data-rule="${esc(o.rule_id)}" data-status="${esc(o.status)}">` +
      `<span class="pill">${esc(ORDER_TYPE[o.type] || o.type)}</span> ${items}` +
      `<div class="sub">${esc(rationale[o.rule_id] || '')} · rule <code>${esc(o.rule_id)}</code> · ${esc(ORDER_STATUS[o.status] || o.status)}</div></div>`;
  }
  if (!detail.length) orders = '<div class="note">No orders drafted. The agent drafts orders only for flagged cases.</div>';
  const can = a => v.agent.allowed.includes(a) && !ST.busy;
  const btn = (act, label, cls = '') => `<button class="btn${cls}" data-act="${act}" data-testid="action-${act === 'approve_orders' ? 'approve' : act === 'edit_orders' ? 'edit' : act === 'closer_look' ? 'closer' : act === 'dismiss_flag' ? 'dismiss' : 'signout'}"${can(act) ? '' : ' disabled'}>${label}</button>`;
  const actions = editing
    ? '<div class="btns"><button class="btn p" data-act="edit_save" data-testid="action-edit-save">Order selected</button><button class="btn" data-act="edit_cancel" data-testid="action-edit-cancel">Cancel</button><span class="sub">Edits can only remove drafted items.</span></div>'
    : `<div class="btns">${btn('approve_orders', 'Order tests', ' p')}${btn('edit_orders', 'Edit')}${btn('closer_look', 'Take a closer look first')}${btn('dismiss_flag', 'Dismiss flag', ' d')}</div>`;
  const signed = v.decision.signedOut;
  const approval = `<div class="card" style="margin-top:14px" data-testid="approval"><h2>Pathologist approval</h2><div class="note">The only place a diagnosis is made.</div>` +
    (signed ? `<div class="verdict" data-testid="verdict">Signed out — routed to the oncologist inbox ${esc(new Date(signed).toLocaleTimeString())}</div>`
      : `<div class="btns">${btn('sign_out', 'Sign out', ' p')}${!can('sign_out') && status !== 'none' ? '<span class="sub">Act on the drafted orders first.</span>' : ''}</div>`) +
    '</div>';
  return `<div class="card" data-testid="agent-panel">${head}<div style="margin:8px 0">${pill}</div>${orders}${blockedRows(v)}${signed ? '' : actions}</div>${approval}`;
}
