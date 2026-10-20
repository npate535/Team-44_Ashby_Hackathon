// Case-view interaction layer (biopsy + surgical): region tooltips, the full-screen pan/zoom viewer, the side-by-side
// reference comparison, inline confirmations for irreversible actions, and finding / feature cross-highlighting.
// app.js owns state, polling and the decision calls; this module only adds behaviour that needs no server round trip.
// It listens on `document` (registered before app.js's own listeners, so it runs first) and never calls the API.
import { classLabel, esc, humanize, num, signed, thumbUrl, toView } from '../adapter.js';
import { contrastSpot } from './why.js';

const ST = () => window.__PQ_ST__ || null;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduced = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---- re-rendering ---------------------------------------------------------------------------------------------------------
let rerenderFn = null;
let assetUrlFn = (p) => '/' + String(p || '').replace(/^\/+/, '');
/** case_biopsy / case_surgical register how to re-render the current case (same HTML app.js would draw). */
export function setRenderer(fn, assetUrl) {
  rerenderFn = fn;
  if (assetUrl) assetUrlFn = assetUrl;
}
function rerender(focusSel) {
  const app = $('#app');
  if (!app || !rerenderFn) return;
  const y = window.scrollY;
  app.innerHTML = rerenderFn();
  window.scrollTo(0, y);
  if (focusSel) {
    const el = $(focusSel, app);
    if (el) el.focus({ preventScroll: false });
  }
}

// ---- state that must survive app.js's poll re-render (#app.innerHTML is replaced every time the data changes) ----------
// An element is re-found in the new DOM by its identifying data attributes (+ position among equal matches).
const KEY_ATTRS = ['data-testid', 'data-rg', 'data-rg-scope', 'data-why', 'data-f', 'data-tab', 'data-act', 'data-cv-confirm', 'data-call', 'data-pick', 'data-edit-seq', 'data-edit-item', 'data-cv-compare', 'data-cv-open', 'data-hm', 'data-cells', 'data-keep'];
function keyOf(el) {
  const app = $('#app');
  // case views only: the worklist and the other screens restore their own focus (app.js / their owners)
  if (!el || !el.closest || !app || el === app || !app.contains(el) || !el.closest('[data-testid=case-view]')) return null;
  const sel = (node) => {
    const parts = KEY_ATTRS.filter((a) => node.hasAttribute(a)).map((a) => `[${a}="${CSS.escape(node.getAttribute(a))}"]`);
    return parts.length ? node.tagName.toLowerCase() + parts.join('') : null;
  };
  let s = sel(el);
  if (!s && el.tagName === 'SUMMARY' && el.parentElement) s = sel(el.parentElement) && `${sel(el.parentElement)} > summary`;
  if (!s) return null;
  return { sel: s, idx: Math.max(0, $$(s, app).indexOf(el)), el };
}
function resolveKey(k) {
  if (!k) return null;
  if (k.el && k.el.isConnected) return k.el;
  const all = $$(k.sel, $('#app') || document);
  return all[k.idx] || all[0] || null;
}
let lastFocus = null;
let kbInput = false; // last interaction was the keyboard (vs a pointer)
const stageKb = (on) => $$('.vw-stage').forEach((st) => st.classList.toggle('kb', on)); // focus ring for keyboard use only
document.addEventListener('keydown', () => {
  kbInput = true;
  stageKb(true);
}, true);
document.addEventListener('pointerdown', () => {
  kbInput = false;
  stageKb(false);
}, true);
// a decision control (order, sign out, margin call…) disappears or is disabled once the decision is sent: move
// keyboard focus to the outcome (feedback / verdict) when it appears instead of dropping it on <body>
const DECISION = /data-act|data-call|data-cv-confirm|confirm-action/;
let focusOutcome = false;
document.addEventListener('focusin', (e) => {
  const app = $('#app');
  focusOutcome = false;
  if (app && app.contains(e.target)) lastFocus = keyOf(e.target);
  else if (!(e.target.closest && e.target.closest('dialog'))) lastFocus = null;
});
document.addEventListener('focusout', (e) => {
  const t = e.target;
  if (e.relatedTarget) return;
  // a real blur (click on the background) leaves the element in place; a re-render removes it
  setTimeout(() => {
    if (lastFocus && lastFocus.el === t && t.isConnected && document.activeElement === document.body) lastFocus = null;
  });
});
// <details> the reader opened stay open (per case) across re-renders
const keptOpen = new Set();
export function keepOpen(key) {
  const s = ST();
  return s && keptOpen.has(`${s.caseId}|${key}`) ? ' open' : '';
}
document.addEventListener(
  'toggle',
  (e) => {
    const d = e.target;
    const s = ST();
    if (!(d instanceof HTMLDetailsElement) || !d.dataset.keep || !d.isConnected || !s || !s.caseId) return;
    const k = `${s.caseId}|${d.dataset.keep}`;
    if (d.open) keptOpen.add(k);
    else keptOpen.delete(k);
  },
  true
);
function afterRender() {
  // keyboard focus: put it back on the same control when the re-render destroyed it (focus fell to <body>)
  if (lastFocus && lastFocus.el && !lastFocus.el.isConnected) {
    const ae = document.activeElement;
    if (!ae || ae === document.body) {
      const el = resolveKey(lastFocus);
      // form controls get focus back only for keyboard users: a focused select/checkbox/slider makes app.js treat
      // the page as "typing" (N = next case ignored, polls paused), which a mouse user never asked for
      const formCtl = el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName);
      if (el && !el.disabled && (!formCtl || kbInput)) el.focus({ preventScroll: true });
      else {
        focusOutcome = DECISION.test(lastFocus.sel);
        lastFocus = null;
      }
    }
  }
  if (focusOutcome && (!document.activeElement || document.activeElement === document.body)) {
    const out = $('#app [data-testid=decision-feedback], #app [data-testid=verdict]');
    if (out) out.focus({ preventScroll: true }); // focusin clears focusOutcome
  }
  markEntrances();
  // region tooltip: follow the replacement region if it is still hovered or focused, else hide it
  if (tipFor && !tipFor.isConnected) {
    const host = $('#app');
    const next = host && $(`[data-rg="${CSS.escape(tipFor.dataset.rg || '')}"][data-rg-scope="case"]`, host);
    if (next && tipWanted(next)) showTip(next);
    else hideTip();
  }
}
// Entrance motion for what just appeared on this case (a confirmation, the decision outcome, a new heatmap layer). Every
// re-render rebuilds #app, so the class is added here, once, never by the HTML: a poll re-render does not replay it.
const ENTER = {
  '[data-testid=confirm-box]': (el) => 'confirm:' + el.dataset.action,
  '[data-testid=decision-feedback]': (el) => 'feedback:' + el.textContent.slice(0, 40),
  '[data-testid=verdict]': () => 'verdict',
  '.cv-hm:not(.surg) > .hm-svg': (el) => 'layer:' + (($('#app [data-testid=heatmap-select]') || {}).value || el.dataset.kind)
};
let shown = { caseId: null, tab: null, keys: new Set() };
function markEntrances() {
  const s = ST();
  const view = $('#app [data-testid=case-view]');
  if (!view || !s) {
    shown = { caseId: null, tab: null, keys: new Set() };
    return;
  }
  const keys = new Set();
  const same = shown.caseId === view.dataset.caseId && shown.tab === s.tab; // first draw of a case or tab: no motion
  for (const [sel, key] of Object.entries(ENTER)) {
    for (const el of $$(sel, view)) {
      const k = key(el);
      keys.add(k);
      if (same && !shown.keys.has(k) && !reduced()) el.classList.add('cv-enter');
    }
  }
  shown = { caseId: view.dataset.caseId, tab: s.tab, keys };
}
function watchApp() {
  const app = $('#app');
  if (!app) return false;
  new MutationObserver(afterRender).observe(app, { childList: true });
  return true;
}
if (!watchApp()) document.addEventListener('DOMContentLoaded', watchApp, { once: true });

// ---- confirmations --------------------------------------------------------------------------------------------------------
let confirmState = null; // {caseId, action}
export function pending(caseId) {
  return confirmState && confirmState.caseId === caseId ? confirmState : null;
}
function setPending(action) {
  const s = ST();
  confirmState = s && s.caseId ? { caseId: s.caseId, action } : null;
  if (s && s.editing === 'margins') s.editing = null; // one open decision at a time
  rerender('[data-testid=confirm-action]');
}
function clearPending(focusSel) {
  if (!confirmState) return;
  confirmState = null;
  if (focusSel) rerender(focusSel);
}
window.addEventListener('hashchange', () => {
  confirmState = null;
  lastFocus = null;
  focusOutcome = false;
  hideTip();
  // a modal left open would keep showing the previous case over the new one
  for (const m of [viewer, cmp]) {
    if (m && m.d.open) {
      m.returnFocus = null;
      m.d.close();
    }
  }
  // the header's Reset confirmation (index.html) belongs to the screen it was opened on: Back closes it too
  const reset = $('#reset-dialog');
  if (reset && reset.open) reset.close();
});

// Going offline (app.js shows #offline after 2 failed polls) or back online changes which decision buttons are enabled
// (agent.js and case_surgical.js read ST.fail), but nothing in the poll fingerprint changes: redraw the open case when
// the banner appears or goes. Only on a real change: app.js re-sets `hidden` after every poll.
function watchOffline() {
  const el = $('#offline');
  if (!el) return false;
  let wasOffline = !el.hidden;
  new MutationObserver(() => {
    const offline = !el.hidden;
    if (offline === wasOffline) return;
    wasOffline = offline;
    const s = ST();
    if (s && s.route === 'case' && $('#app [data-testid=case-view]')) rerender();
  }).observe(el, { attributes: true, attributeFilter: ['hidden'] });
  return true;
}
if (!watchOffline()) document.addEventListener('DOMContentLoaded', watchOffline, { once: true });

// ---- tooltip ----------------------------------------------------------------------------------------------------------------
function tipEl(host) {
  const dlg = host && host.closest('dialog');
  let el = dlg ? $('.cv-tip', dlg) : $('#cv-tip');
  if (!el) {
    el = document.createElement('div');
    el.className = 'cv-tip';
    el.setAttribute('role', 'tooltip');
    el.hidden = true;
    if (dlg) dlg.appendChild(el);
    else {
      el.id = 'cv-tip';
      document.body.appendChild(el);
    }
  }
  return el;
}
let tipFor = null;
function showTip(target) {
  const html = target.dataset.tip;
  if (!html) return;
  const el = tipEl(target);
  el.innerHTML = html;
  el.hidden = false;
  tipFor = target;
  const r = target.getBoundingClientRect();
  const tw = el.offsetWidth,
    th = el.offsetHeight;
  const vw = window.innerWidth,
    vh = window.innerHeight;
  let x = r.right + 8,
    y = r.top;
  if (x + tw > vw - 8) x = r.left - tw - 8;
  if (x < 8) x = Math.max(8, Math.min(vw - tw - 8, r.left + r.width / 2 - tw / 2));
  if (x === Math.max(8, Math.min(vw - tw - 8, r.left + r.width / 2 - tw / 2))) y = r.bottom + 8 > vh - th - 8 ? r.top - th - 8 : r.bottom + 8;
  y = Math.max(8, Math.min(vh - th - 8, y));
  el.style.left = x + 'px';
  el.style.top = y + 'px';
}
function hideTip() {
  $$('.cv-tip').forEach((el) => (el.hidden = true));
  tipFor = null;
}
/** A region's tooltip belongs on screen only while the pointer is over it or the keyboard focused it. Focus that a
 *  script moved there (closing the viewer returns focus to the region that opened it) does not count: that tooltip
 *  would stay up after the pointer has gone, covering the panel beside the image. */
const tipWanted = (el) => !!el && el.isConnected && (el.matches(':hover') || (kbInput && document.activeElement === el));
window.addEventListener(
  'scroll',
  () => {
    if (tipWanted(tipFor)) showTip(tipFor);
    else hideTip();
  },
  { passive: true }
);
// the pointer left the window, or the window lost focus: no tooltip can still be wanted
document.addEventListener('pointerout', (e) => {
  if (!e.relatedTarget && !(kbInput && tipFor && document.activeElement === tipFor)) hideTip();
});
window.addEventListener('blur', hideTip);

// ---- cross-highlighting (finding chips, why rows) ---------------------------------------------------------------------------
function peekTiles(ids, on) {
  $$('#app .rg').forEach((b) => b.classList.toggle('peek', on && ids.includes(Number(b.dataset.rg))));
}
// a Why row peeks its regions on the case image; a look-alike table row also outlines the look-alike's region
const PEEK_ROW = '.why-row[data-tiles], .ct-row[data-tiles]';
function peekRow(row, on) {
  peekTiles(on ? row.dataset.tiles.split(',').filter(Boolean).map(Number) : [], on);
  const box = row.classList.contains('ct-row') && row.closest('.cv-contrast');
  if (box) $$(`.ct-mark[data-ct="${CSS.escape(row.dataset.ct || '')}"]`, box).forEach((m) => m.classList.toggle('on', on));
}
function linkTargets(chip) {
  const s = ST();
  const c = s && s.cases[s.caseId];
  const f = c && (c.findings || []).find((x) => x.id === chip.dataset.f);
  if (!f) return { els: [], tiles: [] };
  const d = f.data || {};
  const sel = {
    prediction: ['[data-testid=how-sure]'],
    concept: [`[data-why="${CSS.escape(String(d.concept || ''))}"]`],
    retrieval: ['[data-testid=compared]', '[data-f-target=retrieval]'],
    quality: ['[data-f-target=quality]'],
    cells: ['[data-testid=tab-cells]', `[data-testid=cells-tile][data-tile="${(d.tiles || [])[0]}"]`],
    margin: ['[data-testid=margin-labels]', '[data-testid=specimen-diagram]', '[data-testid=margin-bars]']
  }[f.kind] || [];
  return { els: sel.flatMap((q) => $$(q, $('#app'))), tiles: f.kind === 'concept' || f.kind === 'cells' ? d.tiles || [] : [] };
}
function linkOn(chip, on) {
  const { els, tiles } = linkTargets(chip);
  els.forEach((el) => el.classList.toggle('cv-linked', on));
  peekTiles(tiles, on);
}

// ---- pan / zoom -------------------------------------------------------------------------------------------------------------
class PanZoom {
  constructor(stage, canvas, aspect, onChange) {
    Object.assign(this, { stage, canvas, aspect, onChange, s: 1, tx: 0, ty: 0, fw: 0, fh: 0, max: 8 });
    this.pointers = new Map();
    this.moved = 0;
    const up = (e) => {
      this.pointers.delete(e.pointerId);
      if (!this.pointers.size) stage.classList.remove('grabbing');
    };
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = stage.getBoundingClientRect();
      this.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0018));
    }, { passive: false });
    // Safari: a trackpad pinch arrives as gesture events (not ctrl+wheel); without this it zooms the whole page
    let pinch = 1;
    stage.addEventListener('gesturestart', (e) => {
      e.preventDefault();
      pinch = 1;
    });
    stage.addEventListener('gesturechange', (e) => {
      e.preventDefault();
      const r = stage.getBoundingClientRect();
      const x = Number.isFinite(e.clientX) ? e.clientX - r.left : r.width / 2;
      const y = Number.isFinite(e.clientY) ? e.clientY - r.top : r.height / 2;
      if (e.scale > 0) this.zoomAt(x, y, e.scale / pinch);
      pinch = e.scale || 1;
    });
    stage.addEventListener('gestureend', (e) => e.preventDefault());
    stage.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.moved = 0;
      stage.classList.add('grabbing');
    });
    stage.addEventListener('pointermove', (e) => {
      const prev = this.pointers.get(e.pointerId);
      if (!prev) return;
      if (e.pointerType === 'mouse' && e.buttons === 0) {
        up(e); // released outside the stage before a drag started
        return;
      }
      // capture only once this is a drag: capturing on pointerdown retargets the click to the stage, so a plain
      // click on a region (zoom to it) never reached the region
      if (this.moved > 4 && !stage.hasPointerCapture(e.pointerId)) {
        try {
          stage.setPointerCapture(e.pointerId);
        } catch {
          /* pointer already gone */
        }
      }
      const pts = [...this.pointers.values()];
      if (this.pointers.size === 2) {
        const other = pts.find((p) => p !== prev);
        const d0 = Math.hypot(prev.x - other.x, prev.y - other.y);
        const d1 = Math.hypot(e.clientX - other.x, e.clientY - other.y);
        const r = stage.getBoundingClientRect();
        if (d0 > 0) this.zoomAt((e.clientX + other.x) / 2 - r.left, (e.clientY + other.y) / 2 - r.top, d1 / d0);
      } else {
        this.tx += e.clientX - prev.x;
        this.ty += e.clientY - prev.y;
        this.apply();
      }
      this.moved += Math.abs(e.clientX - prev.x) + Math.abs(e.clientY - prev.y);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    });
    stage.addEventListener('pointerup', up);
    stage.addEventListener('pointercancel', up);
    // a drag must not count as a click on a region underneath
    stage.addEventListener('click', (e) => {
      if (this.moved > 5) {
        e.stopPropagation();
        e.preventDefault();
      }
    }, true);
    stage.addEventListener('keydown', (e) => {
      const r = stage.getBoundingClientRect();
      const k = e.key || '';
      if (k.startsWith('Arrow') && e.target.closest('[data-rg]')) return; // arrow keys move between regions there
      if (k === '+' || k === '=') this.zoomAt(r.width / 2, r.height / 2, 1.4);
      else if (k === '-' || k === '_') this.zoomAt(r.width / 2, r.height / 2, 1 / 1.4);
      else if (k === '0') this.fit();
      else if (k.startsWith('Arrow')) {
        const step = 60;
        this.tx += k === 'ArrowLeft' ? step : k === 'ArrowRight' ? -step : 0;
        this.ty += k === 'ArrowUp' ? step : k === 'ArrowDown' ? -step : 0;
        this.apply();
      } else return;
      e.preventDefault();
    });
  }
  layout() {
    const sw = this.stage.clientWidth,
      sh = this.stage.clientHeight;
    if (sw / sh > this.aspect) {
      this.fh = sh;
      this.fw = sh * this.aspect;
    } else {
      this.fw = sw;
      this.fh = sw / this.aspect;
    }
    this.canvas.style.width = this.fw + 'px';
    this.canvas.style.height = this.fh + 'px';
    this.apply();
  }
  fit() {
    this.s = 1;
    this.tx = (this.stage.clientWidth - this.fw) / 2;
    this.ty = (this.stage.clientHeight - this.fh) / 2;
    this.apply();
  }
  zoomAt(cx, cy, f) {
    const s2 = Math.max(1, Math.min(this.max, this.s * f));
    this.tx = cx - (cx - this.tx) * (s2 / this.s);
    this.ty = cy - (cy - this.ty) * (s2 / this.s);
    this.s = s2;
    this.apply();
  }
  focusFrac(fx, fy, fw, fh) {
    const sw = this.stage.clientWidth,
      sh = this.stage.clientHeight;
    this.s = Math.max(1, Math.min(this.max, 0.8 * Math.min(sw / (fw * this.fw), sh / (fh * this.fh))));
    this.tx = sw / 2 - (fx + fw / 2) * this.fw * this.s;
    this.ty = sh / 2 - (fy + fh / 2) * this.fh * this.s;
    this.apply();
  }
  set(st) {
    Object.assign(this, st);
    this.apply(true);
  }
  apply(silent = false) {
    const sw = this.stage.clientWidth,
      sh = this.stage.clientHeight;
    const W = this.fw * this.s,
      H = this.fh * this.s;
    this.tx = W <= sw ? (sw - W) / 2 : Math.min(0, Math.max(sw - W, this.tx));
    this.ty = H <= sh ? (sh - H) / 2 : Math.min(0, Math.max(sh - H, this.ty));
    this.canvas.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.s})`;
    this.canvas.style.setProperty('--inv-s', String(1 / this.s));
    if (!silent && this.onChange) this.onChange(this);
  }
}

// ---- full-screen viewer -----------------------------------------------------------------------------------------------------
let viewer = null;
/** The trigger may have been replaced by a poll re-render while the modal was open: focus its replacement. */
function returnFocusTo(k) {
  const el = resolveKey(k);
  if (el && !el.disabled) el.focus({ preventScroll: true });
}
/** Keys inside a modal belong to it: app.js's page shortcuts (N = next case, Escape collapsing every <details>)
 *  must not act on the page behind. Region arrow keys are handled here because document never sees them. */
function isolateKeys(d) {
  d.addEventListener('keydown', (e) => {
    regionArrow(e);
    if (e.key === 'Escape') hideTip();
    e.stopPropagation();
  });
}
function cleanClone(html) {
  return html
    .replace(/\sdata-testid="[^"]*"/g, '')
    .replace(/class="hl-halo"/g, 'class="vhl-halo"')
    .replace(/class="hl"/g, 'class="vhl"')
    .replace(/id="hm-hatch"/g, 'id="vw-hatch"')
    .replace(/url\(#hm-hatch\)/g, 'url(#vw-hatch)')
    .replace(/data-rg-scope="case"/g, 'data-rg-scope="viewer"');
}
function ensureViewer() {
  if (viewer) return viewer;
  const d = document.createElement('dialog');
  d.className = 'cv-viewer';
  d.setAttribute('aria-labelledby', 'cv-vw-title');
  d.innerHTML =
    '<div class="vw-bar"><div class="vw-title"><h2 id="cv-vw-title"></h2><span class="vw-sub"></span></div>' +
    '<div class="vw-tools"><label class="vw-ov"><input type="checkbox" data-vw="overlay" checked> Overlay</label>' +
    '<label class="vw-op">Opacity <input type="range" min="0" max="1" step="0.05" data-vw="opacity"></label>' +
    '<span class="vw-zoom"><button type="button" class="btn" data-vw="out" aria-label="Zoom out">−</button><output aria-live="polite">100%</output><button type="button" class="btn" data-vw="in" aria-label="Zoom in">+</button><button type="button" class="btn" data-vw="fit">Fit</button></span>' +
    '<button type="button" class="btn p" data-vw="close">Close</button></div></div>' +
    '<div class="vw-stage" tabindex="0" aria-label="Image. Drag or use the arrow keys to pan; scroll, pinch, or press + and − to zoom; 0 fits the image."><div class="vw-canvas"></div></div>' +
    '<div class="vw-foot"></div>';
  document.body.appendChild(d);
  const stage = $('.vw-stage', d),
    canvas = $('.vw-canvas', d);
  const pz = new PanZoom(stage, canvas, 4 / 3, (p) => {
    $('output', d).textContent = Math.round(p.s * 100) + '%';
  });
  d.addEventListener('click', (e) => {
    const b = e.target.closest('[data-vw]');
    if (b) {
      const r = stage.getBoundingClientRect();
      if (b.dataset.vw === 'in') pz.zoomAt(r.width / 2, r.height / 2, 1.5);
      if (b.dataset.vw === 'out') pz.zoomAt(r.width / 2, r.height / 2, 1 / 1.5);
      if (b.dataset.vw === 'fit') pz.fit();
      if (b.dataset.vw === 'close') d.close();
      return;
    }
    const rg = e.target.closest('[data-rg]');
    if (rg) {
      e.stopPropagation();
      focusRegion(pz, rg);
    }
  });
  d.addEventListener('change', (e) => {
    if (e.target.dataset.vw === 'overlay') canvas.classList.toggle('no-overlay', !e.target.checked);
  });
  d.addEventListener('input', (e) => {
    if (e.target.dataset.vw === 'opacity') {
      const op = parseFloat(e.target.value);
      $$('.hm-tile', canvas).forEach((r) => r.setAttribute('fill-opacity', (parseFloat(r.dataset.v) * op).toFixed(3)));
    }
  });
  d.addEventListener('close', () => {
    canvas.innerHTML = '';
    hideTip();
    returnFocusTo(viewer.returnFocus);
  });
  isolateKeys(d);
  window.addEventListener('resize', () => d.open && pz.layout());
  viewer = { d, pz, stage, canvas, returnFocus: null };
  return viewer;
}
function focusRegion(pz, rg) {
  const st = rg.style;
  pz.focusFrac(parseFloat(st.left) / 100, parseFloat(st.top) / 100, parseFloat(st.width) / 100, parseFloat(st.height) / 100);
}
function openViewer(trigger, regionIdx = null) {
  const host = (trigger && trigger.closest('.cv-hm')) || $('#app .cv-hm');
  if (!host) return;
  const s = ST();
  const v = s && s.cases[s.caseId] ? toView(s.cases[s.caseId], Date.now(), s.cfg, s.sim) : null;
  const vw = ensureViewer();
  const img = $('img', host);
  const svg = $('svg.hm-svg', host);
  const grid = $('.rg-grid', host);
  vw.canvas.innerHTML =
    `<img src="${esc(img.getAttribute('src'))}" alt="${esc(img.getAttribute('alt') || '')}" draggable="false">` +
    (svg ? cleanClone(svg.outerHTML) : '') +
    (grid ? cleanClone(grid.outerHTML) : '');
  vw.canvas.classList.remove('no-overlay');
  $('[data-vw=overlay]', vw.d).checked = true;
  const surgical = host.classList.contains('surg');
  const opacity = s && s.hm ? s.hm.opacity : 0.6;
  $('[data-vw=opacity]', vw.d).value = String(opacity);
  $('.vw-op', vw.d).hidden = surgical;
  const layerSel = $('#app [data-testid=heatmap-select]');
  const layerName = layerSel && layerSel.selectedOptions[0] ? layerSel.selectedOptions[0].textContent : '';
  $('#cv-vw-title', vw.d).textContent = `${v ? v.id : ''} · ${surgical ? 'Margin strips (simulated frozen section)' : 'Image evidence'}`;
  $('.vw-sub', vw.d).textContent = surgical ? 'Margin segments by tumor signal: pink above the threshold · amber just below · grey-green below' : layerName ? `Layer: ${layerName}` : '';
  const legend = $('#app [data-testid=heatmap-legend]');
  $('.vw-foot', vw.d).innerHTML =
    (legend && !surgical ? cleanClone(legend.outerHTML) : '') +
    '<p class="lg-note">Display image (downscaled from the source field); detail at high zoom is limited by its resolution.</p>';
  const w = v && v.image ? num(v.image.w) : img.naturalWidth || 4;
  const h = v && v.image ? num(v.image.h) : img.naturalHeight || 3;
  vw.pz.aspect = w / h;
  vw.returnFocus = keyOf(trigger);
  vw.d.showModal();
  stageKb(kbInput);
  vw.pz.layout();
  vw.pz.fit();
  if (regionIdx !== null) {
    const rg = $(`[data-rg="${regionIdx}"]`, vw.canvas);
    if (rg) focusRegion(vw.pz, rg);
  }
  vw.stage.focus({ preventScroll: true });
}

// ---- side-by-side reference comparison -------------------------------------------------------------------------------------
let cmp = null;
function ensureCompare() {
  if (cmp) return cmp;
  const d = document.createElement('dialog');
  d.className = 'cv-compare';
  d.setAttribute('aria-labelledby', 'cv-cmp-title');
  d.innerHTML =
    '<div class="vw-bar"><div class="vw-title"><h2 id="cv-cmp-title">Compare with a reference case</h2><span class="vw-sub">Pan and zoom are linked between the two images.</span></div>' +
    '<div class="vw-tools"><span class="vw-zoom"><button type="button" class="btn" data-cmp="prev" aria-label="Previous reference">‹ Prev</button><button type="button" class="btn" data-cmp="next" aria-label="Next reference">Next ›</button></span>' +
    '<span class="vw-zoom"><button type="button" class="btn" data-cmp="out" aria-label="Zoom out">−</button><output aria-live="polite">100%</output><button type="button" class="btn" data-cmp="in" aria-label="Zoom in">+</button><button type="button" class="btn" data-cmp="fit">Fit</button></span>' +
    '<button type="button" class="btn p" data-cmp="close">Close</button></div></div>' +
    '<div class="cmp-panes"><figure class="cmp-pane q"><div class="vw-stage" tabindex="0" aria-label="This case. Drag to pan, scroll to zoom."><div class="vw-canvas"></div></div><figcaption></figcaption></figure>' +
    '<figure class="cmp-pane r"><div class="vw-stage" tabindex="0" aria-label="Reference case. Drag to pan, scroll to zoom."><div class="vw-canvas"></div></div><figcaption></figcaption></figure></div>' +
    '<div class="cmp-strip" role="group" aria-label="Choose a reference"></div>';
  document.body.appendChild(d);
  const panes = $$('.cmp-pane', d).map((p) => ({ stage: $('.vw-stage', p), canvas: $('.vw-canvas', p), cap: $('figcaption', p) }));
  let syncing = false;
  const pzs = panes.map((p, i) =>
    new PanZoom(p.stage, p.canvas, 4 / 3, (src) => {
      $('output', d).textContent = Math.round(src.s * 100) + '%';
      if (syncing) return;
      syncing = true;
      const o = pzs[1 - i];
      // same relative view in the other pane (fractions of its own fitted size)
      const fx = src.tx / src.fw,
        fy = src.ty / src.fh;
      o.set({ s: src.s, tx: fx * o.fw, ty: fy * o.fh });
      syncing = false;
    })
  );
  d.addEventListener('click', (e) => {
    const b = e.target.closest('[data-cmp]');
    if (!b) return;
    const k = b.dataset.cmp;
    const r = panes[0].stage.getBoundingClientRect();
    if (k === 'close') d.close();
    else if (k === 'in') pzs[0].zoomAt(r.width / 2, r.height / 2, 1.5);
    else if (k === 'out') pzs[0].zoomAt(r.width / 2, r.height / 2, 1 / 1.5);
    else if (k === 'fit') {
      pzs.forEach((p) => p.fit());
    } else if (k === 'prev' || k === 'next') showRef(cmp.i + (k === 'next' ? 1 : -1));
    else if (k.startsWith('pick:')) showRef(Number(k.slice(5)));
  });
  d.addEventListener('keydown', (e) => {
    if (e.target.closest('.vw-stage')) return;
    if (e.key === 'ArrowRight') showRef(cmp.i + 1);
    if (e.key === 'ArrowLeft') showRef(cmp.i - 1);
  });
  d.addEventListener('close', () => {
    panes.forEach((p) => (p.canvas.innerHTML = ''));
    returnFocusTo(cmp.returnFocus);
  });
  isolateKeys(d);
  window.addEventListener('resize', () => d.open && pzs.forEach((p) => p.layout()));
  cmp = { d, panes, pzs, i: 0, list: [], v: null, returnFocus: null };
  return cmp;
}
function showRef(i) {
  const c = cmp;
  if (!c.list.length) return;
  c.i = (i + c.list.length) % c.list.length;
  const s = ST();
  const v = c.v;
  const cfg = s && s.cfg;
  const ref = c.list[c.i];
  const assetUrl = assetUrlFn;
  const detail = ref.kind === 'contrast' && Array.isArray(ref.detail) && ref.detail.length ? ref.detail : null;
  c.panes[0].canvas.innerHTML = `<img src="${esc(assetUrl(v.image.src))}" alt="This case, ${esc(v.id)}" draggable="false">` + (detail ? cmpMarks(v, detail, 'query_region') : '');
  c.panes[1].canvas.innerHTML = `<img src="${esc(assetUrl(ref.image || 'images/' + ref.case_id + '.jpg'))}" alt="Reference ${esc(ref.case_id)}" draggable="false">` + (detail ? cmpMarks(v, detail, 'contrast_region') : '');
  c.panes[0].cap.innerHTML = `<b>This case</b> ${esc(v.id)} · suggested ${esc(classLabel(v.top, cfg) || '—')}`;
  const label = (id) => (v.concepts.find((k) => k.id === id) || {}).label || humanize(id);
  const diffChip = (d, i) => {
    const spot = contrastSpot(d);
    return `<span class="chip${d.reliable ? '' : ' muted'}" title="${esc(spot.tip)}"><b class="cmp-n">${i + 1}</b> ${esc(d.label || label(d.concept))} ${signed(num(d.delta))} SD${spot.why ? ` · ${esc(spot.why)}` : ''}</span>`;
  };
  c.panes[1].cap.innerHTML =
    `<b>${ref.kind === 'contrast' ? 'Nearest case with a different label (look-alike)' : `Reference ${c.i + 1} of ${c.list.filter((x) => x.kind !== 'contrast').length}`}</b> ${esc(ref.case_id)} · reference label <b>${esc(classLabel(ref.label, cfg))}</b> (training split) · similarity ${num(ref.similarity).toFixed(2)}` +
    (detail
      ? `<span class="cmp-diff">Differs most in (this case minus look-alike, SD of training cases): ${detail.map(diffChip).join('')}</span><span class="cmp-key">Numbered outlines mark each feature's strongest region in each image.</span>`
      : ref.kind === 'contrast' && ref.differs_most_in && ref.differs_most_in.length
        ? `<span class="cmp-diff">Differs most in: ${ref.differs_most_in.map((id) => `<span class="chip">${esc(label(id))}</span>`).join('')}</span>`
        : '');
  $('.cmp-strip', c.d).innerHTML = c.list
    .map(
      (r, j) =>
        `<button type="button" class="ref-thumb mini${j === c.i ? ' on' : ''}" data-cmp="pick:${j}" aria-pressed="${j === c.i}" aria-label="${esc(r.case_id)}, ${esc(classLabel(r.label, cfg))}"><span class="cv-thumb"><img src="${esc(thumbUrl(assetUrl(r.image || 'images/' + r.case_id + '.jpg'), 240))}" alt="" data-cv-img></span><span class="rt-lbl">${r.kind === 'contrast' ? 'Contrast · ' : ''}${esc(classLabel(r.label, cfg))}</span></button>`
    )
    .join('');
  c.pzs.forEach((p) => {
    p.aspect = num(v.image.w) / num(v.image.h) || 4 / 3;
    p.layout();
    p.fit();
  });
}
/** Numbered outlines of each look-alike feature's strongest region (contract grid, source pixels; both images share
 *  the case's size and grid), matching the numbered caption chips. Unreliable features are dashed. */
function cmpMarks(v, detail, key) {
  const [gr, gc] = v.image.grid || [3, 4];
  const t = num(v.image.tile) || 512;
  const marks = detail
    .map((d, i) => {
      const r = d[key];
      if (!Number.isInteger(r) || r < 0 || r >= gr * gc) return '';
      const x = (r % gc) * t,
        y = Math.floor(r / gc) * t;
      return `<g class="cmp-mark${d.reliable ? '' : ' unreliable'}"><rect x="${x}" y="${y}" width="${t}" height="${t}"/><text x="${x + 28 + i * 96}" y="${y + 112}">${i + 1}</text></g>`;
    })
    .join('');
  return marks ? `<svg class="cmp-marks" viewBox="0 0 ${num(v.image.w)} ${num(v.image.h)}" preserveAspectRatio="none" aria-hidden="true">${marks}</svg>` : '';
}
function openCompare(trigger, which) {
  const s = ST();
  if (!s || !s.cases[s.caseId]) return;
  const v = toView(s.cases[s.caseId], Date.now(), s.cfg, s.sim);
  if (!v.image) return;
  const c = ensureCompare();
  c.v = v;
  c.list = [...v.similar.map((x) => ({ ...x, kind: 'similar' })), ...(v.contrast ? [{ ...v.contrast, kind: 'contrast' }] : [])];
  c.returnFocus = keyOf(trigger);
  c.d.showModal();
  stageKb(kbInput);
  showRef(which === 'contrast' ? c.list.length - 1 : Number(which) || 0);
  // On a phone the dialog scrolls (case.css) and WebKit scrolls a focused element into view despite preventScroll,
  // which opened it with the bar out of sight: focus the top pane there (pan and zoom are linked either way).
  const scrolls = c.d.scrollHeight > c.d.clientHeight;
  c.panes[scrolls ? 0 : 1].stage.focus({ preventScroll: true });
}

// ---- event wiring -------------------------------------------------------------------------------------------------------------
document.addEventListener('click', (e) => {
  const t = e.target.closest('button, [data-rg]');
  if (!t) return;
  const d = t.dataset;
  if (d.cvConfirm) {
    e.preventDefault();
    if (!t.disabled) setPending(d.cvConfirm);
    return;
  }
  if (d.cvCancel !== undefined) {
    const action = confirmState && confirmState.action;
    clearPending(action ? `[data-cv-confirm="${action}"]` : null);
    return;
  }
  if ((d.act || d.call) && confirmState) confirmState = null; // the real action is about to be sent by app.js
  if (d.cvSkip !== undefined) {
    const to = $('#app [data-testid=agent-panel] button:not([disabled]), #app [data-testid=approval] button:not([disabled]), #app [data-testid=verdict]');
    if (to) to.focus();
    return;
  }
  if (d.cvOpen === 'viewer') {
    openViewer(t);
    return;
  }
  if (d.cvCompare !== undefined) {
    openCompare(t, d.cvCompare);
    return;
  }
  if (d.rg !== undefined && d.rgScope === 'case') {
    hideTip();
    openViewer(t, Number(d.rg));
  }
});

/** Arrow keys move between image regions (roving tabindex). True when the key was a region arrow key. */
function regionArrow(e) {
  const rg = e.target.closest && e.target.closest('[data-rg]');
  if (!rg || typeof e.key !== 'string' || !e.key.startsWith('Arrow')) return false;
  const grid = rg.closest('.rg-grid');
  if (!grid) return false; // region links outside the image grid (Why card) keep normal key behaviour
  const cols = Number(grid.dataset.cols) || 1;
  const all = $$('[data-rg]', grid);
  const i = Number(rg.dataset.rg);
  const col = i % cols;
  // left/right stop at the row's edge instead of wrapping onto the neighbouring row
  if ((e.key === 'ArrowLeft' && col === 0) || (e.key === 'ArrowRight' && col === cols - 1)) {
    e.preventDefault();
    return true;
  }
  const j = i + ({ ArrowLeft: -1, ArrowRight: 1, ArrowUp: -cols, ArrowDown: cols }[e.key] || 0);
  const next = all.find((b) => Number(b.dataset.rg) === j);
  e.preventDefault();
  if (next) {
    all.forEach((b) => b.setAttribute('tabindex', b === next ? '0' : '-1'));
    next.focus();
  }
  return true;
}

document.addEventListener('keydown', (e) => {
  if (regionArrow(e)) return;
  // Evidence / Cells tabs: left/right (and Home/End) switch tabs, as the ARIA tabs pattern expects
  const tab = e.target.closest && e.target.closest('.cv [role=tab]');
  if (tab && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
    const tabs = $$('[role=tab]', tab.closest('[role=tablist]') || tab.parentElement);
    const i = tabs.indexOf(tab);
    const j = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    e.preventDefault();
    if (tabs[j] && tabs[j] !== tab) {
      tabs[j].focus();
      tabs[j].click(); // app.js switches ST.tab; the re-render keeps focus on the new tab
    }
    return;
  }
  if (e.key === 'Escape') {
    hideTip();
    if (confirmState && e.target.closest && e.target.closest('.cv-confirm')) {
      const action = confirmState.action;
      clearPending(`[data-cv-confirm="${action}"]`);
    }
  }
});

document.addEventListener('pointerover', (e) => {
  const rg = e.target.closest && e.target.closest('[data-rg]');
  if (rg && rg !== tipFor) showTip(rg);
  const chip = e.target.closest && e.target.closest('.fref[data-f]');
  if (chip) linkOn(chip, true);
  const row = e.target.closest && e.target.closest(PEEK_ROW);
  if (row) peekRow(row, true);
});
document.addEventListener('pointerout', (e) => {
  const rg = e.target.closest && e.target.closest('[data-rg]');
  if (rg && !(e.relatedTarget && rg.contains(e.relatedTarget))) hideTip();
  const chip = e.target.closest && e.target.closest('.fref[data-f]');
  if (chip && !(e.relatedTarget && chip.contains(e.relatedTarget))) linkOn(chip, false);
  const row = e.target.closest && e.target.closest(PEEK_ROW);
  if (row && !(e.relatedTarget && row.contains(e.relatedTarget))) peekRow(row, false);
});
document.addEventListener('focusin', (e) => {
  const t = e.target;
  if (!t.closest) return;
  if (t.matches('[data-rg]') && kbInput) showTip(t);
  if (t.matches('.fref[data-f]')) linkOn(t, true);
  const row = t.closest(PEEK_ROW);
  if (row) peekRow(row, true);
});
document.addEventListener('focusout', (e) => {
  const t = e.target;
  if (!t.matches) return;
  if (t.matches('[data-rg]')) hideTip();
  if (t.matches('.fref[data-f]')) linkOn(t, false);
  const row = t.closest && t.closest(PEEK_ROW);
  if (row && !(e.relatedTarget && row.contains(e.relatedTarget))) peekRow(row, false);
});

// image loading state: a missing file is explained instead of leaving an empty box
document.addEventListener(
  'error',
  (e) => {
    const img = e.target;
    if (!(img instanceof HTMLImageElement) || !img.hasAttribute('data-cv-img')) return;
    if (img.hasAttribute('srcset')) {
      // a downscaled copy failed (heatmap.js inlineImg): fall back to the full display image in `src`
      img.removeAttribute('srcset');
      img.removeAttribute('sizes');
      return;
    }
    const box = img.closest('.cv-thumb, .cv-hm, .cv-crop');
    if (box) {
      box.classList.add('is-missing');
      box.setAttribute('data-missing', `Image unavailable: ${img.getAttribute('src')}`);
    }
  },
  true
);
document.addEventListener(
  'load',
  (e) => {
    const img = e.target;
    if (img instanceof HTMLImageElement && img.hasAttribute('data-cv-img')) {
      const box = img.closest('.cv-thumb');
      if (box) box.classList.add('is-loaded');
    }
  },
  true
);

// live count on the order editor's save button
document.addEventListener('change', (e) => {
  if (e.target.dataset && e.target.dataset.editItem !== undefined) {
    const n = $$('#app [data-edit-item]').filter((x) => x.checked).length;
    const b = $('#app [data-testid=action-edit-save]');
    if (b) {
      b.textContent = n ? `Approve ${n} selected` : 'Select at least one item';
      b.disabled = !n; // the backend rejects an empty edit; dismiss the flag to drop every order
    }
  }
  if (e.target.dataset && e.target.dataset.pick !== undefined) {
    const n = $$('#app [data-pick]').filter((x) => x.checked).length;
    const b = $('#app [data-testid=call-confirm]');
    if (b) b.disabled = !n; // a tumor-at-margin call needs at least one margin
  }
});

export { reduced };
