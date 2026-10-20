// Evidence in plain language (backend/interpret.py): an LLM, reached through the Cursor agent CLI, restates the case's
// measured evidence (never ground truth) in Markdown: bottom line, confidence and uncertainty, the tissue evidence, how
// far each explanation type has been validated, disagreements, and what to check. It adds no new evidence about why the
// classifier decided; the panel says so, links the validation, and shows the server's number check of the text.
// The text is cached server-side; the state lives in ST.interp.
import { esc } from '../adapter.js';
import { SHORT_CLASS } from '../flow.js';

// ---- display-only wording pass (the cached text and its server checks stay exactly as generated) ----------------------
// Class keys read as the case view's labels ("InSitu" → "In situ"), and a class probability never reads 100% or 0%
// (the app shows >99% / <1% everywhere; the exact value is in the Prediction card). Only a percentage directly after a
// class name ("Invasive (100%)", "Invasive 100%, others 0%") is a class probability; other numbers are left alone.
const CLASS_KEYS = Object.keys(SHORT_CLASS).filter((k) => SHORT_CLASS[k] !== k);
const CLASS_WORDS = [...new Set(Object.values(SHORT_CLASS))].map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
const PROB_AFTER_CLASS = new RegExp(`\\b(${CLASS_WORDS}|others?)(\\s*\\(\\s*|:?\\s+)(100|0)(?:\\.0+)?%`, 'g');
export function plainText(src) {
  let s = String(src || '');
  for (const k of CLASS_KEYS) s = s.replace(new RegExp(`\\b${k}\\b`, 'g'), SHORT_CLASS[k]);
  return s.replace(PROB_AFTER_CLASS, (_, cls, gap, n) => `${cls}${gap}${n === '100' ? '>99' : '<1'}%`);
}

// Trust tags as shown: "Validated" in the generated text means an AI-reviewed spot check passed (≥ 15/20 acceptable),
// not a clinical or pathologist validation, so it reads "Spot-check passed".
const TAGS = {
  Validated: ['ok', 'Spot-check passed', 'Passed the AI-reviewed spot check (at least 15 of 20 regions acceptable). An AI check, not a pathologist or clinical validation.'],
  Weak: ['weak', 'Weak', 'Mixed or missing AI-reviewed spot check (about 8 to 14 of 20 regions acceptable)'],
  Unreliable: ['bad', 'Unreliable', 'Failed the AI-reviewed spot check (under 8 of 20 regions acceptable) or flagged as weak'],
  Measured: ['meas', 'Measured', 'Measured from segmented cells']
};

// ---- Markdown → HTML (escaped first; headings, lists, tables, quotes, code, emphasis, [F1] citations) ---------------------
function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, '$1<em>$2</em>')
    .replace(/(^|[^_\w])_([^_\s][^_]*?)_(?!\w)/g, '$1<em>$2</em>')
    // a trust tag ends its line or cell, optionally followed by its spot-check count and citations
    .replace(/\b(Validated|Weak|Unreliable|Measured)\b(?=(?:\s*\(\d+\/\d+\))?(?:\s*\[F\d+(?:\s*,\s*F\d+)*\])?\s*[.;)]?\s*$)/g, (w) =>
      `<span class="ai-tag ${TAGS[w][0]}" title="${esc(TAGS[w][2])}">${TAGS[w][1]}</span>`)
    .replace(/\(Validated\)/g, '(spot-check passed)')
    .replace(/\bValidated\b/g, 'Spot-checked') // the same label used in prose ("Validated stroma … lean invasive")
    .replace(/([↑↓]) invasive/g, (_, a) => `<span class="ai-dir ${a === '↑' ? 'up' : 'down'}">${a} invasive</span>`)
    .replace(/\[(F\d+(?:\s*,\s*F\d+)*)\]/g, (_, ids) =>
      ids.split(/\s*,\s*/).map((f) => `<button class="ai-cite" data-f="${f}" title="Show finding ${f} in the evidence">${f}</button>`).join(''))
    // citation chips stay on the line of the word before them: word (or a short tagged phrase) + chips do not wrap
    .replace(
      /((?:<(strong|em|code)>[^<]{1,32}<\/\2>)|(?:<span class="ai-(?:dir|tag) [a-z]+"[^>]*>[^<]*<\/span>)|[^\s<>]+)( ?)((?:<button class="ai-cite"[^>]*>[^<]*<\/button>)+)/g,
      '<span class="ai-nb">$1$3$4</span>'
    );
}

const cells = (row) => row.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());

export function markdown(src) {
  const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    let m;
    if ((m = line.match(/^```/))) {
      const buf = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) buf.push(lines[i]);
      i++;
      out.push(`<pre><code>${esc(buf.join('\n'))}</code></pre>`);
    } else if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
      const level = Math.min(4, m[1].length + 1); // ## → h3 inside the card
      out.push({ h: m[2].replace(/#+\s*$/, '').trim(), html: `<h${level}>${inline(m[2].replace(/#+\s*$/, ''))}</h${level}>` });
      i++;
    } else if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(line)) {
      out.push('<hr>');
      i++;
    } else if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const head = cells(line);
      const align = cells(lines[i + 1]).map((c) => (/^:-+:$/.test(c) ? 'center' : /-:$/.test(c) ? 'right' : ''));
      const rows = [];
      for (i += 2; i < lines.length && /^\s*\|/.test(lines[i]); i++) rows.push(cells(lines[i]));
      const td = (tag, c, j) => `<${tag}${align[j] ? ` style="text-align:${align[j]}"` : ''}>${inline(c)}</${tag}>`;
      out.push(`<div class="ai-table"><table><thead><tr>${head.map((c, j) => td('th', c, j)).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c, j) => td('td', c, j)).join('')}</tr>`).join('')}</tbody></table></div>`);
    } else if (/^\s*>/.test(line)) {
      const buf = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) buf.push(lines[i].replace(/^\s*>\s?/, ''));
      out.push(`<blockquote>${markdown(buf.join('\n'))}</blockquote>`);
    } else if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d/.test(line);
      const items = [];
      for (; i < lines.length; i++) {
        const l = lines[i];
        const item = l.match(/^\s*(?:[-*+]|\d+[.)])\s+(.*)$/);
        if (item && (/^\s{0,1}\S/.test(l) || !items.length)) items.push([item[1]]);
        else if (l.trim() && /^\s+/.test(l) && items.length) items[items.length - 1].push(l.trim());
        else break;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map((it) => {
        const [first, ...rest] = it;
        const nested = rest.length && /^([-*+]|\d+[.)])\s/.test(rest[0]) ? markdown(rest.join('\n')) : rest.length ? ' ' + inline(rest.join(' ')) : '';
        return `<li>${inline(first)}${nested}</li>`;
      }).join('')}</${tag}>`);
    } else {
      const buf = [];
      for (; i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*>|\s*([-*+]|\d+[.)])\s+|\s*\|)/.test(lines[i]); i++) buf.push(lines[i].trim());
      if (!buf.length) { buf.push(lines[i].trim()); i++; }
      out.push(`<p>${inline(buf.join(' '))}</p>`);
    }
  }
  // Group each heading with the content after it, so sections can be styled (the first one as the "bottom line").
  let html = '';
  let open = false;
  let n = 0;
  for (const block of out) {
    if (typeof block === 'object') {
      if (open) html += '</section>';
      const slug = block.h.toLowerCase().replace(/[^a-z]+/g, '-').replace(/^-|-$/g, '');
      html += `<section class="ai-sec ai-sec-${esc(slug)}${n++ === 0 ? ' ai-lead' : ''}">${block.html}`;
      open = true;
    } else html += block;
  }
  return html + (open ? '</section>' : '');
}

// Server check: every number in the text was matched against the evidence and validation JSON it was given.
function verifyBadge(v) {
  if (!v || !Number.isFinite(v.checked)) return '';
  const bad = v.unsupported || [];
  const tip = bad.length ? `Not found in the evidence: ${bad.join(', ')}` : 'Every number in the text matches the evidence it was given';
  return `<span class="ai-verify ${bad.length ? 'warn' : 'ok'}" title="${esc(tip)}" data-testid="interp-verify">${bad.length ? `${v.checked - bad.length}/${v.checked} numbers traced` : `${v.checked}/${v.checked} numbers traced`}</span>`;
}

// Claim check (backend/interpret.py check_claims): verdict line, arrows, trust tags and OOD reading vs rules computed
// from the JSON; failures were sent back to the model once to fix. Remaining issues are listed, never hidden.
function claimsBadge(c) {
  if (!c) return '<span class="ai-verify" title="Verdict, arrows, trust tags and OOD reading not checked yet">claims not checked</span>';
  const left = c.remaining || [];
  const fixed = (c.found || []).length - left.length;
  if (left.length) return `<span class="ai-verify warn" title="${esc(plainText(left.join(' ')))}">${left.length} claim issue${left.length === 1 ? '' : 's'}</span>`;
  return `<span class="ai-verify ok" title="Verdict, arrows, trust tags and OOD reading match rules computed from the data${fixed ? `; ${fixed} corrected by a repair pass` : ''}">claims checked${fixed ? ` · ${fixed} fixed` : ''}</span>`;
}

// ---- panel -----------------------------------------------------------------------------------------------------------------
const when = (iso) => {
  const d = iso ? new Date(iso) : null;
  return d && !isNaN(d) ? d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
};

function skeleton(s) {
  const secs = s.startedAt ? Math.max(0, Math.round((Date.now() - s.startedAt) / 1000)) : 0;
  return (
    '<div class="ai-loading" role="status" aria-live="polite">' +
    '<div class="ai-pulse" aria-hidden="true"><i></i><i></i><i></i></div>' +
    `<div><b>Generating summary…</b><span class="sub">A language model is summarizing the probabilities, conformal set, reliability and tissue features. This usually takes 1–2 minutes${secs ? ` · <span data-ai-elapsed="${s.startedAt}">${secs}</span> s` : ''}.</span></div>` +
    '</div>' +
    '<div class="ai-skel" aria-hidden="true"><i style="width:38%"></i><i></i><i style="width:92%"></i><i style="width:30%"></i><i></i><i style="width:84%"></i><i style="width:66%"></i></div>'
  );
}

/**
 * key: case id, or '__results' for the evaluation page. scope: 'case' | 'results'.
 * Buttons carry data-interp (generate | regenerate | retry) and data-interp-key; app.js does the requests.
 */
export function interpretPanel(ST, key, scope = 'case') {
  const s = (ST.interp && ST.interp[key]) || { status: 'idle' };
  const d = s.data || {};
  const what = scope === 'results' ? 'these evaluation results' : 'this case';
  const btn = (act, label, primary) =>
    `<button class="btn${primary ? ' p' : ''}" data-interp="${act}" data-interp-key="${esc(key)}" data-testid="interp-${act}"${s.status === 'generating' ? ' disabled' : ''}>${label}</button>`;
  let body;
  let meta = '';
  if (s.status === 'idle' || s.status === 'loading') {
    body = '<div class="ai-skel short" aria-hidden="true"><i style="width:60%"></i><i></i></div>';
  } else if (s.status === 'generating') {
    body = skeleton(s);
  } else if (s.status === 'error') {
    body = `<div class="ai-empty err" role="alert"><b>Interpretation unavailable</b><p>${esc(s.error || 'Unknown error')}</p>${btn('retry', 'Try again', true)}</div>`;
  } else if (s.status === 'ok' && d.markdown) {
    const bad = (d.verification && d.verification.unsupported) || [];
    body =
      `<article class="ai-md" data-testid="interp-text">${markdown(plainText(d.markdown))}</article>` +
      ((d.claims && (d.claims.remaining || []).length) ? `<p class="ai-unverified" role="note"><b>Check these statements:</b> ${d.claims.remaining.map((t) => esc(plainText(t))).join(' ')}</p>` : '') +
      (bad.length ? `<p class="ai-unverified" role="note"><b>Check these numbers:</b> ${bad.map(esc).join(', ')} could not be traced to the evidence (often a sum or difference the model worked out; verify before quoting).</p>` : '');
    meta =
      verifyBadge(d.verification) +
      (scope === 'case' ? claimsBadge(d.claims) : '') +
      `<span class="ai-chip" title="Language model used through the ${d.provider === 'claude' ? 'Claude Code' : 'Cursor agent'} CLI">${esc(d.model || 'model')}</span>` +
      `<span class="sub">${d.cached ? 'cached · ' : ''}${esc(when(d.generated_at))}${d.elapsed_s ? ` · ${Number(d.elapsed_s).toFixed(0)} s` : ''}</span>`;
  } else if (d.on_demand === false) {
    body =
      '<div class="ai-empty">' +
      `<p>No plain-language summary has been generated for ${what} in this workspace. The measured evidence it would restate is shown in full on this page.</p></div>`;
  } else {
    body =
      '<div class="ai-empty">' +
      `<p>A language model restates the measured evidence for ${what} in plain language: how confident the call is, which tissue features the app measured, which of those explanation types have passed a validation test, where the evidence disagrees, and what to check on the slide.</p>` +
      '<ul class="ai-points"><li><b>Confidence</b> calibrated probability, conformal set, reliability</li><li><b>Evidence</b> concepts and measured cells, by region</li><li><b>Validation</b> deletion test, spot check, calibration</li></ul>' +
      `${btn('generate', 'Generate interpretation', true)}<p class="sub">Uses only ${scope === 'results' ? 'docs/RESULTS.md' : "this case's evidence JSON (no ground truth)"}; the text is cached for the demo.</p></div>`;
  }
  const regen = s.status === 'ok' && d.markdown && d.on_demand !== false ? btn('regenerate', 'Regenerate', false) : '';
  return (
    `<div class="card ai-card" data-testid="interpretation" data-interp-state="${esc(s.status)}">` +
    `<div class="card-h ai-h"><h2><span class="ai-mark" aria-hidden="true"></span>Evidence in plain language</h2><div class="ai-meta">${meta}${regen}</div></div>` +
    body +
    `<p class="ai-foot"><b>What this is:</b> a plain-language summary of evidence the app already measured, written by a language model. It adds no new evidence about why the classifier decided; a fluent explanation is not proof of what the model used. Automatic checks confirm that its numbers appear in the evidence and that its verdict, arrows, trust tags and OOD reading follow the data; they do not prove every sentence is right. The tested evidence is the deletion test, calibration and conformal coverage ${scope === 'results' ? 'in the sections below' : 'on the <a href="#/metrics">evaluation page</a>'}. Research prototype, not clinically validated.</p>` +
    '</div>'
  );
}
