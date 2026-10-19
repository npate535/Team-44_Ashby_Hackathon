// View-model adapter: every contract/server field the UI reads goes through here (docs/research/frontend_connect.md §2).
// Pure functions, no DOM. Tolerates a worklist item or a full CasePublic, with or without `runtime`.
// Research prototype — not for clinical use. Ground truth (`eval_only`) is stripped and never rendered.

/** @typedef {import('./types.js').CasePublic} CasePublic */

export const TIER = {
  1: {
    label: 'Surgical frozen section (simulated)',
    cls: 't-1',
    note: 'Simulated frozen section, timer running'
  },
  2: { label: 'Likely invasive', cls: 't-2', note: 'Invasive is the top class or is in the conformal set' },
  3: { label: 'Low confidence', cls: 't-3', note: 'More than one class possible, or reliability flag' },
  4: { label: 'Unflagged', cls: 't-4', note: 'Read in arrival order; not cleared by the model' }
};
export const REASON = {
  aggressive_top: 'Invasive carcinoma suggested',
  aggressive_in_conformal_set: 'Invasive in the confidence set',
  set_size_gt_1: 'More than one class in the confidence set',
  empty_set: 'Empty confidence set',
  low_reliability: 'Low reliability',
  insufficient_tissue: 'Insufficient tissue',
  aging_promoted: 'Promoted: waiting time',
  surgical_frozen_section: 'Simulated frozen section'
};
const QUALITY_REASONS = new Set(['low_reliability', 'insufficient_tissue']);
export const MARGINS = ['superior', 'inferior', 'medial', 'lateral'];
export const CALL_TEXT = {
  clear: 'Margins clear',
  tumor_at_margin: 'Tumor at margin — more tissue needed',
  defer: 'Cannot determine — defer to permanent sections'
};

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export function esc(value) {
  return value === null || value === undefined ? '' : String(value).replace(/[&<>"']/g, (ch) => ESC[ch]);
}

// Teammate helpers (kept): minutes -> "12 min" / "3.5 h" / "1.2 d"; remaining time against a target.
export function fmt(m) {
  m = Math.max(0, Math.round(m));
  if (m < 120) return m + ' min';
  const h = m / 60;
  return h < 48 ? h.toFixed(1) + ' h' : (h / 24).toFixed(1) + ' d';
}
export function rem(elapsedMin, targetMin) {
  const r = targetMin - elapsedMin;
  return r < 0 ? 'Overdue ' + fmt(-r) : fmt(r) + ' left';
}
export function mmss(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600),
    m = Math.floor((s % 3600) / 60),
    r = s % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return (h ? h + ':' + pad(m) : pad(m)) + ':' + pad(r);
}
/** A URL safe inside CSS url('...') within a style attribute. */
export function cssUrl(u) {
  return encodeURI(String(u)).replace(/['()]/g, (c) => '%' + c.charCodeAt(0).toString(16));
}
/** Widths built ahead of time at /thumbs/<w>/<kind>/<name> (backend route; static files on the deployed site). */
export const THUMB_WIDTHS = [240, 320, 400];
/** A downscaled copy of a media image for thumbnails: the pre-built path for those widths, else the backend's `?w=`
 *  (mock/static servers ignore the query and serve the full image). */
export function thumbUrl(url, width) {
  const u = String(url || '');
  if (!u || u.includes('?')) return u;
  const w = Math.round(width);
  const m = THUMB_WIDTHS.includes(w) && u.match(/^(.*\/)?(images|tcga)\/([A-Za-z0-9._-]+\.jpe?g)$/i);
  return m ? `${m[1] || ''}thumbs/${w}/${m[2]}/${m[3]}` : `${u}?w=${w}`;
}
/** Only #rgb / #rrggbb colors from config are inlined into styles. */
export function cssColor(c) {
  return /^#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$/.test(String(c)) ? c : 'var(--mut)';
}
export const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
export const pct = (p) => (p === null || p === undefined ? '—' : Math.round(p * 100) + '%');
export const signed = (v) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(2);
export const humanize = (id) =>
  String(id)
    .replace(/_/g, ' ')
    .replace(/^./, (c) => c.toUpperCase());
/** 1 -> "1st", 2 -> "2nd", 11 -> "11th", 22 -> "22nd". */
export function ordinal(n) {
  const k = Math.round(Number(n));
  const t = k % 100;
  return k + (t >= 11 && t <= 13 ? 'th' : { 1: 'st', 2: 'nd', 3: 'rd' }[k % 10] || 'th');
}
/** A 0–1 percentile as an ordinal that never reads "100th" / "0th". */
export function ordinalPct(p) {
  if (p === null || p === undefined || !Number.isFinite(Number(p))) return '—';
  const x = Number(p);
  return x >= 0.995 ? '>99th' : x < 0.005 ? '<1st' : ordinal(x * 100);
}
/** A probability with `digits` decimals for tooltips: never "1.000" / "0.000" (">0.999" / "<0.001" instead). */
export function exactProb(p, digits = 3) {
  if (p === null || p === undefined || !Number.isFinite(Number(p))) return '—';
  const x = Number(p),
    step = 10 ** -digits;
  if (x >= 1 - step / 2) return '>' + (1 - step).toFixed(digits);
  if (x < step / 2) return '<' + step.toFixed(digits);
  return x.toFixed(digits);
}
export const isLLM = (generator) => typeof generator === 'string' && generator.startsWith('lumen:');
export const FINDING_ID = /^F[1-9][0-9]*$/;

export function tierInfo(tier, cfg) {
  const base = TIER[tier] || { label: 'Tier ' + tier, cls: 't-4', note: '' };
  const label = cfg && cfg.labels && cfg.labels.tiers && cfg.labels.tiers[String(tier)];
  return { ...base, label: label || base.label };
}

export function reasonLabel(code, cfg) {
  const label = cfg && cfg.labels && cfg.labels.reasons && cfg.labels.reasons[code];
  return label || REASON[code] || code; // unknown codes render raw: nothing is hidden
}

export function classLabel(cls, cfg) {
  return (cfg && cfg.labels && cfg.labels.classes && cfg.labels.classes[cls]) || cls;
}

/** Strip ground truth defensively (the API never sends it; raw fixtures in ?mock=1 do). */
export function stripGroundTruth(c) {
  if (c && Object.prototype.hasOwnProperty.call(c, 'eval_only')) {
    console.warn('eval_only removed from case', c.case_id);
    const copy = { ...c };
    delete copy.eval_only;
    return copy;
  }
  return c;
}

/**
 * Live timer for a case at `nowMs`: interpolates from the server's elapsed_s at fetch time.
 * Surgical: wall clock, amber/red from cfg.timer. Biopsy: simulator clock (× sim.speed while running).
 */
export function timerNow(c, nowMs, cfg, sim) {
  const t = c && c.runtime && c.runtime.timer;
  if (!t) return { kind: 'none', elapsedS: 0, targetS: null, usedFrac: null, state: 'none', text: '—' };
  const since = Math.max(0, (nowMs - (c._fetched_ms || nowMs)) / 1000);
  if (t.kind === 'surgical') {
    const stopped = t.state === 'stopped';
    const elapsedS = t.elapsed_s + (stopped ? 0 : since);
    const amber = t.amber_s ?? cfg.timer.amber_min * 60,
      red = t.red_s ?? cfg.timer.red_min * 60; // settings margin.timer
    const state = stopped ? 'stopped' : elapsedS >= red ? 'red' : elapsedS >= amber ? 'amber' : 'ok';
    const left = red - elapsedS;
    return {
      kind: 'surgical',
      elapsedS,
      targetS: red,
      usedFrac: Math.min(1, elapsedS / red),
      state,
      text: mmss(elapsedS),
      countdown: stopped
        ? 'Stopped at ' + mmss(elapsedS)
        : left >= 0
          ? mmss(left) + ' left'
          : 'Overdue ' + mmss(-left)
    };
  }
  const running = sim && sim.running && t.state !== 'stopped';
  const elapsedS = t.elapsed_s + (running ? since * (sim.speed || 1) : 0);
  const targetS = t.target_s ?? null;
  return {
    kind: 'biopsy',
    elapsedS,
    targetS,
    usedFrac: targetS ? Math.min(1, elapsedS / targetS) : null,
    state: t.state === 'stopped' ? 'stopped' : 'none',
    text: t.state === 'stopped' ? 'Read in ' + fmt(elapsedS / 60) : 'Waiting ' + fmt(elapsedS / 60),
    countdown: targetS ? rem(elapsedS / 60, targetS / 60) : ''
  };
}

/** Agent status → user-facing marker text (worklist Agent column). */
export function agentText(c) {
  const rt = c.runtime || {};
  if (rt.decision && rt.decision.margin_call) return 'Margin call sent';
  if (rt.decision && rt.decision.signed_out_at) return 'Signed out';
  const status = c.agent && c.agent.status;
  return (
    {
      drafted: 'Orders drafted',
      approved: 'Orders approved',
      edited_and_approved: 'Edited + approved',
      deferred_closer_look: 'Closer look',
      dismissed: 'Flag dismissed'
    }[status] || 'None'
  );
}

// Client enablement table (frontend_connect §4.5) used only when the server sends no allowed_actions.
export function allowedActions(c) {
  const rt = c.runtime || {};
  if (Array.isArray(rt.allowed_actions)) return rt.allowed_actions;
  if (rt.done) return [];
  if (c.lane === 'surgical') return ['margin_call'];
  return (
    {
      drafted: ['approve_orders', 'edit_orders', 'closer_look', 'dismiss_flag'],
      deferred_closer_look: ['approve_orders', 'edit_orders', 'dismiss_flag']
    }[c.agent && c.agent.status] || ['sign_out']
  );
}

/** Why-panel rows: concepts + cell features not already listed as concepts, by |contribution|. */
export function whyRows(c, cfg) {
  const rows = (c.concepts || []).map((k) => ({
    id: k.id,
    label: k.label,
    value: k.value,
    valueStd: k.value_std ?? null, // optional: case feature standardized against training cases
    fracStd: k.frac_std ?? null, // optional: share of regions above threshold, standardized against training cases
    contribution: k.contribution,
    spread: k.spread,
    tiles: k.top_tiles || [],
    source: k.source || 'text',
    measured: k.source === 'cell',
    captionConsistent: k.caption_consistent === true,
    direction: k.direction,
    unit: k.source === 'cell' ? ((c.cells && c.cells.features && c.cells.features[k.id]) || {}).unit : null
  }));
  const ids = new Set(rows.map((r) => r.id));
  const cells = c.cells && c.cells.status === 'ok' ? c.cells : null;
  if (cells) {
    const cellTiles = (cells.tiles || []).map((t) => t.idx);
    for (const [id, f] of Object.entries(cells.features || {})) {
      if (ids.has(id) || f.contribution === null || f.contribution === undefined) continue;
      const label =
        (cfg && cfg.labels && cfg.labels.cell_features && cfg.labels.cell_features[id]) || humanize(id);
      rows.push({
        id,
        label,
        value: f.value,
        contribution: f.contribution,
        spread: null,
        tiles: cellTiles,
        source: 'cell',
        measured: true,
        captionConsistent: false,
        direction: null,
        unit: f.unit
      });
    }
  }
  return rows.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

/** Normalized margin block or null (a surgical case should always carry one; the view shows margin-null). */
export function marginView(m) {
  if (!m) return null;
  return {
    call: m.suggested_call,
    involved: m.involved || [],
    perMargin: m.per_margin || {},
    threshold: m.threshold ?? null,
    segments: m.segments || [],
    stripUm: m.strip_um ?? null,
    lengths: m.involved_length_um || {}
  };
}

/**
 * The view model for one case (worklist item or full case).
 * @param {object} raw  API item / CasePublic (+ runtime, + _fetched_ms stamped by the API layer)
 * @param {number} nowMs
 * @param {object} cfg  /api/config
 * @param {object} sim  latest simulator status
 */
export function toView(raw, nowMs, cfg, sim) {
  const c = stripGroundTruth(raw);
  const rt = c.runtime || {};
  const flag = c.flag || {};
  const tier = rt.tier ?? flag.tier ?? 4;
  const reasons = (rt.reasons ?? flag.reasons ?? []).map((code) => ({
    code,
    label: reasonLabel(code, cfg),
    quality: QUALITY_REASONS.has(code)
  }));
  const pred = c.prediction || null;
  const classes = (pred && pred.classes) || [];
  const probs = (pred && pred.probs) || [];
  const topIdx = probs.length ? probs.indexOf(Math.max(...probs)) : -1;
  const set = (pred && pred.conformal_set) || [];
  const q = c.quality || {};
  const variant = c.simulated_variant || null;
  const info = tierInfo(tier, cfg);
  return {
    id: c.case_id,
    lane: c.lane,
    simulated: !!c.simulated,
    variant,
    synthetic: !!c.synthetic,
    external: c.cohort === 'tcga',
    laneLabel:
      c.cohort === 'tcga'
        ? 'TCGA-BRCA external'
        : c.lane === 'surgical'
          ? 'Simulated frozen section'
          : 'Biopsy',
    variantLabel:
      variant === 'degraded'
        ? 'simulated artifact: ' + ((c.degradation && c.degradation.type) || 'degraded')
        : variant === 'composite'
          ? 'composite field (simulated)'
          : null,
    question:
      c.specimen_question ||
      (c.lane === 'surgical' ? 'Margin status (simulated)' : 'Breast microscopy field (BACH)'),
    room: rt.or_room || null,
    tier,
    tierLabel: info.label,
    tierCls: info.cls,
    tierNote: info.note,
    reasons,
    aged: reasons.some((r) => r.code === 'aging_promoted'),
    risk: pred ? Math.round(pred.risk_score * 100) : null,
    probs: classes.map((cls, i) => ({ cls, label: classLabel(cls, cfg), p: probs[i] })),
    top: topIdx >= 0 ? classes[topIdx] : null,
    set,
    setSize: set.length,
    alphaPct: pred ? Math.round((1 - pred.conformal_alpha) * 100) : null,
    reliability: {
      score: q.reliability ?? null,
      flag: !!q.reliability_flag,
      oodPct: q.ood_percentile ?? null,
      qcTiles: q.qc_tiles_kept ?? null
    },
    agreement: c.agreement || {},
    timer: timerNow(c, nowMs, cfg, sim),
    agent: {
      status: (c.agent && c.agent.status) || 'none',
      orders: (c.agent && c.agent.orders) || [],
      blocked: (c.agent && c.agent.blocked) || [],
      drafted: (c.agent && c.agent.status) === 'drafted',
      nOrders:
        c.agent && c.agent.n_orders !== undefined
          ? c.agent.n_orders
          : ((c.agent && c.agent.orders) || []).length,
      text: agentText(c),
      allowed: allowedActions(c),
      orderStatus: rt.orders || null
    },
    decision: {
      signedOut: rt.decision ? rt.decision.signed_out_at : null,
      marginCall: rt.decision ? rt.decision.margin_call : null
    },
    done: !!rt.done,
    arrivedSim: rt.arrived_sim_s ?? c.arrived_offset_s,
    image: c.image
      ? {
          src: c.image.display,
          w: c.image.width,
          h: c.image.height,
          tile: c.image.tile_size,
          grid: c.image.grid
        }
      : null,
    tiles: c.tiles || [],
    concepts: c.concepts || [],
    why: c.concepts ? whyRows(c, cfg) : [],
    similar: c.similar_cases || [],
    contrast: c.contrast_case || null,
    findings: new Map((c.findings || []).map((f) => [f.id, f])),
    summary: c.summary
      ? {
          sentences: c.summary.sentences || [],
          generator: c.summary.generator,
          isLLM: isLLM(c.summary.generator)
        }
      : null,
    cells: c.cells && c.cells.status ? c.cells : { status: 'disabled' },
    margin: marginView(c.margin),
    model: pred ? pred.model : null,
    attribution: (pred && pred.attribution) || null, // exact region decomposition of the read (tiles[].contribution)
    schema: c.schema_version || null
  };
}

/** Structural fingerprint: re-render only when these change (frontend_connect §3.4). */
export function fingerprint(items) {
  return JSON.stringify(
    (items || []).map((c) => [
      c.case_id,
      c.runtime && c.runtime.tier,
      c.runtime && c.runtime.reasons,
      c.agent && c.agent.status,
      c.runtime && c.runtime.decision,
      c.runtime && c.runtime.done
    ])
  );
}

/** Held-out aggregate coverage for the case's actual prediction model (never a per-case truth label). */
export function coverageStatus(payload, model) {
  const ev = payload && payload.eval;
  if (!ev || !ev.metrics || !ev.conformal || ev.metrics.models?.production !== model) return null;
  const report = ev.conformal;
  const coverage = report.methods?.[report.drives_tiers]?.coverage?.estimate;
  const target = report.target_coverage;
  if (![coverage, target].every((v) => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1))
    return null;
  if (!Number.isInteger(report.n) || report.n <= 0) return null;
  return { coverage, target, n: report.n, belowTarget: coverage < target };
}
