// Decision panel (plan §9): drafted orders with rationale + rule ID, a state stepper, the four agent actions, and the
// sign-out card. Irreversible actions (ordering, dismissing, signing out) ask for an inline confirmation first: the
// first button only opens the confirmation (data-cv-confirm, handled in case_ui.js); the confirm button carries the
// real data-act that app.js sends to the backend. Approval is recorded in Sepia only: nothing is sent to a laboratory or
// ordering system (docs/SCOPE_MATRIX.md).
// Surgical: only the visibly blocked actions (stains are never drafted in the simulated frozen-section lane).
import { esc } from '../adapter.js';
import { pending } from './case_ui.js';

const STATUS = {
  none: 'No orders drafted',
  drafted: 'Drafted, awaiting your decision',
  approved: 'Orders approved',
  edited_and_approved: 'Edited and approved',
  deferred_closer_look: 'Closer look first',
  dismissed: 'Flag dismissed'
};
const ORDER_TYPE = { stain: 'Stain', recut: 'Recut', deeper_levels: 'Deeper levels', new_sample: 'New sample' };
const ORDER_STATUS = { drafted: 'drafted', approved: 'approved', edited: 'approved (edited)', removed: 'removed', dismissed: 'dismissed' };
// what an approval means here, in one place: the confirmation, the outcome and the panel note all say it
const NOT_SENT = 'nothing is sent to a laboratory or ordering system';
const RULES_NOTE = `Drafted by illustrative rules that need clinical review. You approve or edit them in Sepia; ${NOT_SENT}.`;
const TESTID = { approve_orders: 'approve', edit_orders: 'edit', closer_look: 'closer', dismiss_flag: 'dismiss', sign_out: 'signout' };

function blockedRows(v) {
  const rows = v.agent.blocked.map(
    (b) =>
      `<div class="act blk" data-testid="agent-blocked"><b>Blocked</b> ${esc(b.action)}<div class="sub">${esc(b.reason)}${b.rule_id ? ' · ' + esc(b.rule_id) : ''}</div></div>`
  );
  if (v.lane === 'surgical') {
    rows.push(
      '<div class="act blk" data-testid="agent-guard"><b>Guard</b> no stain orders in the simulated frozen-section lane<div class="sub">The agent may only draft non-stain follow-up (recut, deeper levels) here; the result is your margin call.</div></div>'
    );
  }
  return rows.join('');
}

function lastTransition(events) {
  const t = (events || []).filter((e) => e.type === 'agent_transition');
  return t.length ? t[t.length - 1] : null;
}
const clock = (iso) => (iso ? new Date(iso).toLocaleTimeString() : '');

/** Orders as shown: runtime order status when the backend has it, else the drafted orders. */
function orderList(v) {
  return v.agent.orderStatus || v.agent.orders.map((o, seq) => ({ ...o, seq, status: 'drafted', drafted_items: o.items }));
}
const itemsOf = (o) => (o.items && o.items.length ? o.items : o.drafted_items || o.items || []);
const liveItems = (detail) => detail.filter((o) => o.status !== 'removed' && o.status !== 'dismissed').flatMap(itemsOf);

function stepper(v, signed) {
  const s = v.agent.status;
  const hasOrders = s !== 'none';
  const decided = ['approved', 'edited_and_approved', 'dismissed'].includes(s) || !hasOrders;
  const step = (state, n, title, sub) =>
    `<li class="${state}"${state === 'current' ? ' aria-current="step"' : ''}><span class="st-dot" aria-hidden="true">${state === 'done' ? '✓' : n}</span><span class="st-t"><b>${title}</b><span>${sub}</span></span></li>`;
  const nOrders = orderList(v).length;
  const decisionSub = !hasOrders
    ? 'nothing to decide'
    : s === 'drafted'
      ? 'approve, edit, look closer or dismiss'
      : s === 'deferred_closer_look'
        ? 'closer look in progress'
        : (STATUS[s] || s).toLowerCase();
  return (
    '<ol class="stepper" aria-label="Decision steps">' +
    step(hasOrders ? 'done' : 'st-skip', 1, 'Agent draft', hasOrders ? `${nOrders} order${nOrders === 1 ? '' : 's'} from rules` : 'no rule fired') +
    step(decided ? (hasOrders ? 'done' : 'st-skip') : 'current', 2, 'Your decision', decisionSub) +
    step(signed ? 'done' : decided ? 'current' : 'todo', 3, 'Sign out', signed ? 'routed to oncologist' : 'routes to the oncologist') +
    '</ol>'
  );
}

function confirmBox(v, ST, detail) {
  const p = pending(v.id);
  if (!p) return '';
  const items = liveItems(detail);
  const text = {
    approve_orders: {
      title: `Approve ${items.length} item${items.length === 1 ? '' : 's'}?`,
      body: `${items.map(esc).join(', ')}. The approval is recorded in Sepia and cannot be withdrawn here; ${NOT_SENT}.`,
      ok: 'Confirm approval'
    },
    dismiss_flag: {
      title: 'Dismiss the flag and drop all drafted orders?',
      body: 'No follow-up is approved for this case. The case stays open for sign-out.',
      ok: 'Confirm dismiss'
    },
    sign_out: {
      title: `Sign out ${esc(v.id)}?`,
      body: `The result goes to the oncologist inbox with ${
        v.agent.status === 'none' || v.agent.status === 'dismissed' ? 'no follow-up orders' : `${items.length} approved follow-up item${items.length === 1 ? '' : 's'}`
      }. Sign-out is final in this workspace.`,
      ok: 'Confirm sign-out'
    }
  }[p.action];
  if (!text) return '';
  const can = v.agent.allowed.includes(p.action) && !ST.busy;
  return (
    `<div class="cv-confirm" role="group" aria-labelledby="cv-confirm-t" aria-describedby="cv-confirm-b" data-testid="confirm-box" data-action="${esc(p.action)}">` +
    `<b id="cv-confirm-t">${text.title}</b><p id="cv-confirm-b">${text.body}</p>` +
    `<div class="btns"><button class="btn p${p.action === 'dismiss_flag' ? ' dd' : ''}" data-act="${esc(p.action)}" data-testid="confirm-action"${can ? '' : ' disabled'}>${text.ok}</button>` +
    '<button class="btn" data-cv-cancel data-testid="confirm-cancel">Cancel</button></div></div>'
  );
}

function feedback(v, ST, signed) {
  const s = v.agent.status;
  const ev = lastTransition(ST.events && ST.events[v.id]);
  const when = ev ? ` at ${esc(clock(ev.at))}` : '';
  if (signed) return '';
  if (s === 'approved' || s === 'edited_and_approved') {
    const items = liveItems(orderList(v));
    return `<div class="cv-done" role="status" data-testid="decision-feedback" tabindex="-1"><span aria-hidden="true">✓</span><span><b>${items.length} item${items.length === 1 ? '' : 's'} approved in Sepia${when}:</b> ${items.map(esc).join(', ')}. Not sent to a laboratory system. Next: sign out to route the result.</span></div>`;
  }
  if (s === 'dismissed') return `<div class="cv-done" role="status" data-testid="decision-feedback" tabindex="-1"><span aria-hidden="true">✓</span><span><b>Flag dismissed${when}.</b> No follow-up approved. Next: sign out.</span></div>`;
  if (s === 'deferred_closer_look')
    return `<div class="cv-done info" role="status" data-testid="decision-feedback" tabindex="-1"><span aria-hidden="true">i</span><span><b>Closer look recorded${when}.</b> Orders are still drafted; approve, edit or dismiss them when ready.</span></div>`;
  return '';
}

export function agentPanel(v, ST) {
  const head =
    `<h2>Follow-up orders</h2><p class="note">${RULES_NOTE}</p>`;
  if (v.lane === 'surgical' && !v.agent.orders.length && !(v.agent.orderStatus || []).length) {
    return `<div class="card cv-decide" data-testid="agent-panel">${head}<div style="margin-top:8px">${blockedRows(v)}</div></div>`;
  }
  const status = v.agent.status;
  const signed = v.decision.signedOut;
  const pill = `<span class="pill${status === 'drafted' || status === 'deferred_closer_look' ? ' warn' : status === 'none' ? '' : ' ok'}" data-testid="agent-status" data-status="${esc(status)}">${esc(STATUS[status] || status)}</span>`;
  const detail = orderList(v);
  const rationale = Object.fromEntries(v.agent.orders.map((o) => [o.rule_id + '|' + o.type, o.rationale]));
  const editing = ST.editing === 'orders';
  let orders = '';
  for (const o of detail) {
    const off = o.status === 'removed' || o.status === 'dismissed';
    const shown = itemsOf(o);
    const items = editing
      ? `<div class="edit">${(o.drafted_items || o.items || [])
          .map(
            (item) =>
              `<label><input type="checkbox" data-edit-seq="${esc(o.seq)}" data-edit-item="${esc(item)}"${(ST.editDraft[o.seq] || []).includes(item) ? ' checked' : ''} data-testid="edit-item">${esc(item)}</label>`
          )
          .join('')}</div>`
      : // an item that only repeats its order type ("Deeper levels" · "deeper levels") is not shown twice
        `<span class="ord-items">${shown
          .filter((i) => shown.length > 1 || String(i).toLowerCase() !== String(ORDER_TYPE[o.type] || o.type).toLowerCase())
          .map((i) => `<b>${esc(i)}</b>`)
          .join('')}</span>`;
    orders +=
      `<div class="act ord ${esc(o.status)}${off ? ' off' : ''}" data-testid="agent-order" data-seq="${esc(o.seq)}" data-rule="${esc(o.rule_id)}" data-status="${esc(o.status)}">` +
      `<div class="ord-h"><span class="pill">${esc(ORDER_TYPE[o.type] || o.type)}</span>${items}<span class="ord-st">${esc(ORDER_STATUS[o.status] || o.status)}</span></div>` +
      `<div class="sub">${esc(rationale[o.rule_id + '|' + o.type] || o.rationale || '')} <span class="ord-rule">rule <code>${esc(o.rule_id)}</code></span></div></div>`;
  }
  if (!detail.length) orders = '<div class="note">No follow-up rule applies to this case, so nothing was drafted.</div>';
  if (v.lane === 'surgical')
    orders = '<div class="note">Non-stain follow-up only (the reliability flag is raised); the margin call is separate.</div>' + orders;
  const can = (a) => v.agent.allowed.includes(a) && !ST.busy;
  const p = pending(v.id);
  const btn = (act, label, cls = '', confirm = false) =>
    `<button class="btn${cls}" ${confirm ? 'data-cv-confirm' : 'data-act'}="${act}" data-testid="action-${TESTID[act]}"${can(act) ? '' : ' disabled'}${p && p.action === act ? ' aria-expanded="true"' : confirm ? ' aria-expanded="false"' : ''}>${label}</button>`;
  const nItems = liveItems(detail).length;
  const open = ['approve_orders', 'edit_orders', 'closer_look', 'dismiss_flag'].some((a) => v.agent.allowed.includes(a));
  // the same label/state case_ui.js sets live on each tick; a poll re-render must not re-enable an empty edit
  const nSel = Object.values(ST.editDraft || {}).reduce((a, xs) => a + (Array.isArray(xs) ? xs.length : 0), 0);
  const actions = editing
    ? `<div class="btns"><button class="btn p" data-act="edit_save" data-testid="action-edit-save"${nSel && can('edit_orders') ? '' : ' disabled'}>${nSel ? `Approve ${nSel} selected` : 'Select at least one item'}</button><button class="btn" data-act="edit_cancel" data-testid="action-edit-cancel">Cancel</button><span class="sub">Untick items to drop them; edits can only remove drafted items, and the selected ones are approved at once.</span></div>`
    : open
      ? `<div class="btns cv-actions">${btn('approve_orders', `Approve orders${nItems ? ` (${nItems})` : ''}`, ' p', true)}${btn('edit_orders', 'Edit')}${btn('closer_look', 'Take a closer look first')}${btn('dismiss_flag', 'Dismiss flag', ' d', true)}</div>` +
        (ST.busy ? '<p class="sub" role="status">Saving your decision…</p>' : '')
      : '';
  const box = p && p.action !== 'sign_out' ? confirmBox(v, ST, detail) : '';
  if (v.lane === 'surgical') {
    // no sign-out in the frozen lane: the margin call is the result
    return `<div class="card cv-decide" data-testid="agent-panel">${head}<div style="margin:8px 0">${pill}</div>${orders}${blockedRows(v)}${open || editing ? actions + box : ''}</div>`;
  }
  const signReason =
    status === 'drafted' || status === 'deferred_closer_look'
      ? 'Decide on the drafted orders first: approve, edit or dismiss them.'
      : ST.busy
        ? 'Saving your decision…'
        : '';
  const approval =
    `<div class="card cv-signout" data-testid="approval"><h2>Pathologist sign-out</h2><p class="note">The diagnosis is recorded at sign-out. The suggestion above is decision support.</p>` +
    (signed
      ? `<div class="verdict" data-testid="verdict" role="status" tabindex="-1">Signed out, routed to the oncologist inbox at ${esc(new Date(signed).toLocaleTimeString())}</div>`
      : `<div class="btns">${btn('sign_out', 'Sign out…', ' p', true)}${!can('sign_out') && signReason ? `<span class="sub" data-testid="signout-reason">${esc(signReason)}</span>` : ''}</div>` +
        (p && p.action === 'sign_out' ? confirmBox(v, ST, detail) : '')) +
    '</div>';
  return (
    `<div class="card cv-decide" data-testid="agent-panel"><div class="cv-decide-h"><h2>Follow-up orders</h2>${pill}</div>` +
    stepper(v, signed) +
    (detail.length ? `<p class="note">${RULES_NOTE}</p>` : '') +
    `<div class="ord-list">${orders}</div>${blockedRows(v)}${feedback(v, ST, signed)}${signed ? '' : actions + box}</div>${approval}`
  );
}
