// ?mock=1: an in-memory stand-in for the backend with the same method surface as api.js, built from the committed
// fixtures (mock/manifest.json) and a config snapshot (mock/config.json, checked against the backend by
// tests/test_frontend_static.py). It never recomputes tiers (core.tiers is the only tier implementation): it shows
// each fixture's static flag. Everything here is synthetic and labeled so.

import { ApiError } from './api.js';

const clone = (v) => JSON.parse(JSON.stringify(v));
function deepMerge(base, patch) {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) return patch;
  const out = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    out[k] =
      v &&
      typeof v === 'object' &&
      !Array.isArray(v) &&
      base &&
      typeof base[k] === 'object' &&
      base[k] !== null
        ? deepMerge(base[k], v)
        : v;
  }
  return out;
}
const basename = (p) =>
  String(p || '')
    .split('/')
    .pop();
const variantOf = (c) => c.simulated_variant || 'normal';

const ALLOWED = {
  drafted: ['approve_orders', 'edit_orders', 'closer_look', 'dismiss_flag'],
  deferred_closer_look: ['approve_orders', 'edit_orders', 'dismiss_flag'],
  none: ['sign_out'],
  approved: ['sign_out'],
  edited_and_approved: ['sign_out'],
  dismissed: ['sign_out']
};
const TARGET = {
  approve_orders: 'approved',
  edit_orders: 'edited_and_approved',
  closer_look: 'deferred_closer_look',
  dismiss_flag: 'dismissed'
};

export async function createMockApi() {
  const manifestUrl = new URL('../mock/manifest.json', import.meta.url);
  const manifest = await (await fetch(manifestUrl)).json();
  const config = await (await fetch(new URL('../mock/config.json', import.meta.url))).json();
  config.synthetic = true;
  const docBase = new URL(manifest.asset_base, document.baseURI);
  const sources = {};
  for (const entry of manifest.cases) {
    if (!sources[entry.src]) {
      const res = await fetch(new URL(entry.src, document.baseURI));
      if (!res.ok) throw new ApiError(res.status, 'mock fixture ' + entry.src + ' did not load');
      sources[entry.src] = await res.json();
    }
  }
  const speed = config.sim.speed;
  const hold = config.sim.hold_surgical_for_injection;
  let state;
  let routedSeq = 0;
  // Mirrors backend Backend.routed_report(): read-only context for a routed result (never ground truth).
  function report(x, to) {
    const c = x.c;
    const probs = c.prediction.probs;
    const out = {
      lane: c.lane,
      top_class: c.prediction.classes[probs.indexOf(Math.max(...probs))],
      tier: c.flag.tier,
      reasons: [...c.flag.reasons],
      quality: { reliability: c.quality.reliability, reliability_flag: c.quality.reliability_flag, qc_tiles_kept: c.quality.qc_tiles_kept },
      simulated: !!c.simulated,
      arrived_at: x.arrivedWall ? new Date(x.arrivedWall).toISOString() : null
    };
    if (to === 'oncologist')
      out.features = [...(c.concepts || [])]
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 3)
        .map((k) => ({ id: k.id, label: k.label, contribution: k.contribution, direction: k.direction, source: k.source || 'text' }));
    else {
      const m = c.margin || {};
      out.margin = { suggested_call: m.suggested_call, involved: m.involved || [], per_margin: m.per_margin || {}, threshold: m.threshold ?? null };
      out.simulated_variant = c.simulated_variant || null;
      out.degradation = c.degradation || null;
    }
    return out;
  }

  function seed() {
    const cases = manifest.cases.map((entry) => {
      const c = deepMerge(clone(sources[entry.src]), entry.patch || {});
      delete c.eval_only; // ground truth never enters the mock either
      c.case_id = entry.id || c.case_id;
      c.synthetic = true;
      return {
        c,
        inject: entry.inject || (c.lane === 'surgical' && hold ? variantOf(c) : null),
        arrived: false,
        arrivedSim: null,
        arrivedWall: null,
        status: c.agent.status,
        orders: clone(c.agent.orders || []),
        orderStatus: (c.agent.orders || []).map(() => 'drafted'),
        drafted: (c.agent.orders || []).map((o) => [...o.items]),
        signedOut: null,
        call: null,
        events: []
      };
    });
    state = { cases, running: false, base: 0, started: null, routed: { surgeon: [], oncologist: [] } };
  }
  seed();

  const simNow = () =>
    state.base + (state.running ? ((performance.now() - state.started) / 1000) * speed : 0);
  const simStatus = () => ({
    running: state.running,
    speed,
    sim_time_s: simNow(),
    hold_surgical_for_injection: hold,
    cases: state.cases.length,
    arrived: state.cases.filter((x) => x.arrived).length,
    pending: pending()
  });
  function pending() {
    const out = { biopsy: 0, surgical: { normal: 0, degraded: 0, composite: 0 } };
    for (const x of state.cases)
      if (!x.arrived) {
        if (x.inject) out.surgical[x.inject] += 1;
        else out.biopsy += 1;
      }
    return out;
  }
  function log(x, type, extra = {}) {
    x.events.push({
      id: x.events.length + 1,
      case_id: x.c.case_id,
      type,
      at: new Date().toISOString(),
      sim_time_s: simNow(),
      ...extra
    });
  }
  function tick() {
    const sim = simNow();
    for (const x of state.cases) {
      if (!x.arrived && !x.inject && x.c.arrived_offset_s <= sim) {
        Object.assign(x, { arrived: true, arrivedSim: x.c.arrived_offset_s, arrivedWall: Date.now() });
        log(x, 'arrived');
      }
    }
  }
  const room = (id) => 'OR-' + (([...id].reduce((a, ch) => (a * 31 + ch.charCodeAt(0)) >>> 0, 7) % 4) + 1);
  function runtime(x) {
    const done = !!(x.signedOut || x.call);
    const allowed = done ? [] : x.c.lane === 'surgical' ? ['margin_call'] : ALLOWED[x.status];
    let timer;
    if (x.c.lane === 'surgical') {
      const end = x.call ? x.call.wall : Date.now();
      const elapsed = (end - x.arrivedWall) / 1000;
      const amber = config.timer.amber_min * 60,
        red = config.timer.red_min * 60;
      timer = {
        kind: 'surgical',
        clock: 'wall',
        elapsed_s: elapsed,
        amber_s: amber,
        red_s: red,
        state: x.call ? 'stopped' : elapsed >= red ? 'red' : elapsed >= amber ? 'amber' : 'ok'
      };
    } else {
      const end = x.signedOut ? x.signedOut.sim : simNow();
      timer = {
        kind: 'biopsy',
        clock: 'sim',
        elapsed_s: Math.max(0, end - x.arrivedSim),
        target_s: config.aging.turnaround_target_min[String(x.c.flag.tier)] * 60,
        state: x.signedOut ? 'stopped' : 'none'
      };
    }
    return {
      tier: x.c.flag.tier,
      reasons: x.c.flag.reasons,
      aging_promoted: false,
      arrived_sim_s: x.arrivedSim,
      injected: !!x.inject,
      timer,
      done,
      allowed_actions: allowed,
      decision: {
        signed_out_at: x.signedOut ? x.signedOut.at : null,
        margin_call: x.call ? { call: x.call.call, margins: x.call.margins, at: x.call.at } : null
      },
      or_room: x.c.lane === 'surgical' ? room(x.c.case_id) : null,
      orders: x.orders.map((o, i) => ({
        seq: i,
        rule_id: o.rule_id,
        type: o.type,
        status: x.orderStatus[i],
        drafted_items: x.drafted[i],
        items: o.items
      }))
    };
  }
  function full(x) {
    const c = clone(x.c);
    c.agent = {
      ...c.agent,
      status: x.status,
      orders: x.orders.filter((_, i) => x.orderStatus[i] !== 'removed')
    };
    return { ...c, runtime: runtime(x), _fetched_ms: Date.now() };
  }
  function item(x) {
    const f = full(x);
    f.agent = {
      status: x.status,
      n_orders: f.agent.orders.length,
      orders_drafted: x.status === 'drafted',
      blocked: (x.c.agent.blocked || []).length
    };
    return f;
  }
  function find(id) {
    tick();
    const x = state.cases.find((y) => y.c.case_id === id && y.arrived);
    if (!x) throw new ApiError(404, 'case ' + id + ' not found (unknown, or not arrived yet)');
    return x;
  }
  function sortKey(x, order) {
    const done = x.signedOut || x.call ? 1 : 0;
    return order === 'arrival'
      ? [done, x.arrivedSim, x.c.case_id]
      : [done, x.c.flag.tier, -x.c.prediction.risk_score, x.arrivedSim, x.c.case_id];
  }
  const cmp = (a, b) => {
    for (let i = 0; i < a.length; i++) {
      if (a[i] < b[i]) return -1;
      if (a[i] > b[i]) return 1;
    }
    return 0;
  };

  function decide(id, action, payload) {
    if (
      !['approve_orders', 'edit_orders', 'closer_look', 'dismiss_flag', 'sign_out', 'margin_call'].includes(
        action
      )
    )
      throw new ApiError(422, 'unknown action ' + action); // same order of checks as backend.db
    const x = find(id);
    const allowed = runtime(x).allowed_actions;
    if (!allowed.includes(action)) throw new ApiError(409, action + ' is not allowed now (mock)');
    const from = x.status;
    const now = new Date().toISOString();
    if (!['margin_call', 'edit_orders'].includes(action) && Object.keys(payload).length)
      throw new ApiError(422, action + ' takes no payload');
    if (action === 'margin_call') {
      const ORDER = ['superior', 'inferior', 'medial', 'lateral'];
      if (!['clear', 'tumor_at_margin', 'defer'].includes(payload.call))
        throw new ApiError(422, 'unknown margin call');
      const raw = payload.margins || [];
      if (raw.some((m) => !ORDER.includes(m)) || new Set(raw).size !== raw.length)
        throw new ApiError(422, 'unknown or repeated margin');
      if (payload.call === 'tumor_at_margin' && !raw.length)
        throw new ApiError(422, 'tumor_at_margin needs at least one margin');
      if (payload.call !== 'tumor_at_margin' && raw.length)
        throw new ApiError(422, 'call ' + payload.call + ' takes no margins');
      const margins = ORDER.filter((m) => raw.includes(m));
      x.call = { call: payload.call, margins, at: now, wall: Date.now() };
      const elapsed = (x.call.wall - x.arrivedWall) / 1000;
      state.routed.surgeon.unshift({
        id: ++routedSeq,
        report: report(x, 'surgeon'),
        case_id: id,
        to: 'surgeon',
        kind: 'margin_call',
        call: payload.call,
        margins,
        at: now,
        timer_elapsed_s: elapsed,
        simulated: true,
        or_room: room(id),
        synthetic: true,
        suggested_call: x.c.margin && x.c.margin.suggested_call
      });
      log(x, 'routed', { action, payload: { to: 'surgeon' } });
      log(x, 'margin_call', { action, payload: { call: payload.call, margins } });
    } else if (action === 'sign_out') {
      x.signedOut = { at: now, sim: simNow() };
      const placed = x.orders
        .filter((_, i) => ['approved', 'edited'].includes(x.orderStatus[i]))
        .map((o) => ({ type: o.type, rule_id: o.rule_id, items: o.items }));
      state.routed.oncologist.unshift({
        id: ++routedSeq,
        report: report(x, 'oncologist'),
        case_id: id,
        to: 'oncologist',
        kind: 'sign_out',
        at: now,
        agent_status: x.status,
        orders: placed,
        summary: x.c.summary,
        synthetic: true,
        suggestion: {
          classes: x.c.prediction.classes,
          probs: x.c.prediction.probs,
          risk_score: x.c.prediction.risk_score,
          conformal_set: x.c.prediction.conformal_set,
          conformal_alpha: x.c.prediction.conformal_alpha
        }
      });
      log(x, 'routed', { action, payload: { to: 'oncologist' } });
      log(x, 'signed_out', { action });
    } else {
      if (action === 'edit_orders') {
        const kept = {};
        for (const e of payload.orders || []) {
          const drafted = x.drafted[e.seq];
          if (!drafted || e.items.some((i) => !drafted.includes(i)))
            throw new ApiError(422, 'edits may only remove drafted items');
          if (e.items.length) kept[e.seq] = drafted.filter((i) => e.items.includes(i));
        }
        if (!Object.keys(kept).length) throw new ApiError(422, 'keep at least one item, or dismiss the flag');
        x.orders.forEach((o, i) => {
          if (kept[i]) {
            o.items = kept[i];
            x.orderStatus[i] = 'edited';
          } else x.orderStatus[i] = 'removed';
        });
      } else if (action === 'approve_orders') x.orderStatus = x.orderStatus.map(() => 'approved');
      else if (action === 'dismiss_flag') x.orderStatus = x.orderStatus.map(() => 'dismissed');
      x.status = TARGET[action];
      log(x, 'agent_transition', { from, to: x.status, action, payload });
    }
    return { result: { case_id: id, action, from, to: x.status }, case: full(x) };
  }

  return {
    mock: true,
    base: '',
    health: async () => ({
      status: 'ok',
      mock: true,
      synthetic: true,
      loaded: state.cases.length,
      skipped: [],
      open_cases: state.cases.filter((x) => x.arrived && !x.signedOut && !x.call).length,
      open_surgical: state.cases.filter((x) => x.arrived && x.c.lane === 'surgical' && !x.call).length,
      sim: simStatus()
    }),
    config: async () => clone(config),
    async worklist({ lane = 'all', order = 'ai' } = {}) {
      tick();
      const now = Date.now();
      const rows = state.cases
        .filter((x) => x.arrived && (lane === 'all' || x.c.lane === lane))
        .sort((a, b) => cmp(sortKey(a, order), sortKey(b, order)));
      return {
        server_time: now / 1000,
        sim: { ...simStatus(), _fetched_ms: now },
        order,
        lane,
        synthetic: true,
        items: rows.map(item)
      };
    },
    getCase: async (id) => full(find(id)),
    events: async (id) => clone(find(id).events),
    decide: async (id, action, payload = {}) => decide(id, action, payload),
    routed: async (to) => clone(state.routed[to] || []),
    async sim(cmd, body) {
      let injected = null;
      if (cmd === 'start' && !state.running) {
        state.running = true;
        state.started = performance.now();
      } else if (cmd === 'pause' && state.running) {
        tick();
        state.base = simNow();
        state.running = false;
      } else if (cmd === 'reset') seed();
      else if (cmd === 'inject_surgical') {
        const variant = (body && body.variant) || 'normal';
        const x = state.cases.find((y) => !y.arrived && y.inject === variant);
        if (!x)
          throw new ApiError(409, 'no pending simulated frozen section of variant ' + variant + ' (mock)');
        Object.assign(x, { arrived: true, arrivedSim: simNow(), arrivedWall: Date.now() });
        log(x, 'injected');
        injected = x.c.case_id;
      }
      tick();
      return { command: cmd, injected, sim: simStatus() };
    },
    async metrics() {
      try {
        const res = await fetch(new URL('../mock/metrics.json', import.meta.url));
        if (res.ok) return await res.json();
      } catch (_) {
        /* optional file */
      }
      return { eval: {}, figures: [], errors: [], expected: {}, synthetic: true };
    },
    async spotcheckItems() {
      const items = [];
      for (const x of state.cases) {
        for (const t of (x.c.cells && x.c.cells.tiles) || []) {
          items.push({
            item_id: `cells:overlay:${x.c.case_id}:${t.idx}`,
            kind: 'cells',
            target: 'overlay',
            label: 'Cell overlay',
            case_id: x.c.case_id,
            tile_idx: t.idx,
            image: t.overlay,
            score: null
          });
        }
      }
      return { available: items.length > 0, synthetic: true, created_utc: null, items };
    },
    // no model calls in fixture mode: the panel shows its empty state and an explanatory error on generate
    interpretation: async () => ({ status: 'none', model: null }),
    interpret: async () => { throw new ApiError(503, 'Interpretation is not available in fixture mode (?mock=1).'); },
    spotcheck: async (body) => ({
      recorded: body.reviews.length,
      total: body.reviews.length,
      reviewer_kind: body.reviewer_kind,
      summary: {},
      mock: true
    }),
    assetUrl: (p) => new URL((manifest.asset_map && manifest.asset_map[p]) || basename(p), docBase).href
  };
}
