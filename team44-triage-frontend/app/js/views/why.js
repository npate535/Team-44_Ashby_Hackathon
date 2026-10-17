// Evidence panels (plan §8.1): Why (signed contributions), How sure, Compared to, Summary with finding refs.
import {classLabel, esc, FINDING_ID, num, pct, signed} from '../adapter.js';

export function why(v, hm) {
  const rows = v.why;
  if (!rows.length) return '<div class="note">No criterion contributions for this case.</div>';
  const maxAbs = Math.max(1e-9, ...rows.map(r => Math.abs(r.contribution)));
  const body = rows.map(r => {
    const x = r.contribution / maxAbs;
    const bar = `<svg viewBox="-1 0 2 1" preserveAspectRatio="none" aria-hidden="true"><rect x="-1" y="0.42" width="2" height="0.16" fill="var(--bl)"/>` +
      `<rect x="${Math.min(0, x)}" y="0.1" width="${Math.abs(x)}" height="0.8" rx="0.02" fill="${x >= 0 ? 'var(--eos)' : 'var(--hema)'}"/>` +
      '<line x1="0" x2="0" y1="0" y2="1" stroke="var(--ink)" stroke-width="0.02"/></svg>';
    const src = r.measured ? '<span class="src m" data-testid="src-measured" title="Measured from segmented nuclei">◆ measured</span>'
      : '<span class="src" data-testid="src-estimated" title="Estimated from text prompts / concept vectors">○ estimated</span>';
    const cc = r.captionConsistent ? '<span class="cc" data-testid="caption-consistent" title="Concept vector agreed with mined captions on held-out data (caption co-mention, not tissue truth)">caption-consistent</span>' : '';
    const title = r.spread !== null && r.spread !== undefined ? ` title="prompt-ensemble spread ${r.spread.toFixed(2)}"` : '';
    return `<button class="why-row${hm.why === r.id ? ' on' : ''}" data-testid="why-bar-${esc(r.id)}" data-why="${esc(r.id)}"${title}>` +
      `<span class="lbl">${esc(r.label)}${src}${cc}</span>${bar}<span class="v">${signed(r.contribution)}</span></button>`;
  }).join('');
  return `<div class="axis"><span>← away from invasive</span><span>toward invasive →</span></div><div class="why">${body}</div>` +
    '<p class="sub">Contribution toward / away from calibrated P(Invasive). Associated features, not causes; not a Nottingham grade. Click a bar to highlight its regions.</p>';
}

export function howSure(v, cfg) {
  const bars = v.probs.map(p => `<div class="cbar"><span>${esc(p.label)}</span><span class="t"><i style="width:${Math.round(p.p * 100)}%"></i></span><span>${pct(p.p)}</span></div>`).join('');
  const chips = v.setSize ? v.set.map(cls => `<span class="chip conf" data-testid="conformal-chip">${esc(cls)}</span>`).join('')
    : '<span class="chip empty" data-testid="conformal-chip-empty">∅ empty set</span>';
  const setNote = v.setSize === 1 ? 'single class = confident' : v.setSize ? `${v.setSize} classes possible` : 'no class reached the threshold';
  const ag = v.agreement;
  const neighbor = v.findings && [...v.findings.values()].find(f => f.kind === 'retrieval');
  const neighborText = neighbor ? `${neighbor.data.agree} of ${neighbor.data.of} references share the suggested class`
    : ag.neighbor_agreement !== null && ag.neighbor_agreement !== undefined ? pct(ag.neighbor_agreement) + ' of references agree' : '—';
  const spreads = v.concepts.map(c => c.spread).filter(s => s !== null && s !== undefined);
  const spread = spreads.length ? (spreads.reduce((a, b) => a + b, 0) / spreads.length).toFixed(2) : '—';
  const r = v.reliability;
  const kv = (k, val, extra = '') => `<div class="kv"${extra}><span>${k}</span><b>${val}</b></div>`;
  return `<div class="bigp" data-testid="risk">${v.risk === null ? '—' : v.risk + '%'}</div><div class="sub">calibrated P(Invasive) · suggested: ${esc(classLabel(v.top, cfg))}</div>` +
    `<div style="margin:10px 0">${bars}</div>` +
    `<h3>${v.alphaPct}% conformal set</h3><div class="chips" data-testid="conformal-set">${chips}</div><div class="sub">${setNote}</div>` +
    '<div style="margin-top:10px">' +
    kv('CONCH–MUSK agreement', ag.conch_musk_same_top === true ? 'same top class' : ag.conch_musk_same_top === false ? 'disagree' : '—') +
    kv('Neighbor agreement', esc(neighborText)) +
    kv('Prompt-ensemble spread', spread) +
    kv('Reliability', (r.score === null ? '—' : r.score.toFixed(2)) + (r.flag ? ' <span class="pill warn" data-testid="reliability-flag">flag raised</span>' : ''), ' data-f-target="quality"') +
    kv('Input check (OOD percentile)', r.oodPct === null ? '—' : Math.round(r.oodPct * 100)) +
    kv('Regions kept by QC', r.qcTiles === null ? '—' : num(r.qcTiles)) +
    `<div class="kv"><span>Stamp</span><span class="sub">${esc(v.model || '—')} · contract ${esc(v.schema || '—')}</span></div></div>`;
}

export function compared(v, cfg, assetUrl) {
  if (!v.similar.length && !v.contrast) return '<div class="note">No reference cases.</div>';
  const look = v.similar.map(s => `<div data-testid="similar-case"><img src="${esc(assetUrl(s.image))}" alt="Reference ${esc(s.case_id)}" loading="lazy">${esc(classLabel(s.label, cfg))}<br>${s.similarity.toFixed(2)}</div>`).join('');
  let h = `<div class="sub" style="margin-bottom:6px">Train-split reference cases (labels are reference labels)</div><div class="look">${look}</div>`;
  const c = v.contrast;
  if (c) {
    const label = id => (v.concepts.find(k => k.id === id) || {}).label || id;
    h += `<div class="contrast" data-testid="contrast-case"><img src="${esc(assetUrl(c.image || 'images/' + c.case_id + '.jpg'))}" alt="Contrast reference ${esc(c.case_id)}">` +
      `<div><b>Nearest different label:</b> ${esc(classLabel(c.label, cfg))} (${c.similarity.toFixed(2)})<div class="sub">differs most in</div>` +
      `<div class="chips">${c.differs_most_in.map(id => `<span class="chip">${esc(label(id))}</span>`).join('')}</div></div></div>`;
  }
  return h;
}

export function summary(v) {
  const s = v.summary;
  if (!s || !s.sentences.length) return '<div class="note">No summary.</div>';
  const badge = s.isLLM ? `<span class="badge llm" data-testid="summary-generator">LLM-written from findings · ${esc(s.generator.slice(6))}</span>`
    : '<span class="badge" data-testid="summary-generator">Template</span>';
  const ps = s.sentences.map(sent => `<p style="margin:6px 0">${esc(sent.text)}${sent.findings.filter(f => FINDING_ID.test(f))
    .map(f => `<button class="fref" data-testid="finding-ref" data-f="${esc(f)}" title="${esc((v.findings.get(f) || {}).kind || '')}">${esc(f)}</button>`).join('')}</p>`).join('');
  return `<div>${badge}</div>${ps}<p class="sub">Written only from structured findings; it never saw the image.</p>`;
}
