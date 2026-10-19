// Shared glass navigation, motion and persistent appearance. No data or workflow changes.
// Motion is short and ease-out (about 200 ms), only follows a user action, and is skipped under reduced motion.
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const EASE = 'cubic-bezier(.2, .8, .2, 1)';
let scene = null;
let tabGeometry = null;
let navGeometry = null;
let navCurrent = null;

function positionLens(container, selected, lens, previous = null) {
  if (!selected || !lens) return null;
  const box = container.getBoundingClientRect();
  const rect = selected.getBoundingClientRect();
  const x = rect.left - box.left + container.scrollLeft - container.clientLeft;
  lens.style.width = `${rect.width}px`;
  lens.style.transform = `translateX(${x}px)`;
  if (previous && !reduced() && (previous.x !== x || previous.width !== rect.width)) {
    lens.animate([
      { transform: `translateX(${previous.x}px)`, width: `${previous.width}px` },
      { transform: `translateX(${x}px)`, width: `${rect.width}px` }
    ], { duration: 220, easing: EASE });
  }
  return { x, width: rect.width };
}

// Phones: the pill scrolls sideways. The current destination is brought to the middle and the edges fade where more
// destinations are out of view.
function updateFade(nav) {
  const max = nav.scrollWidth - nav.clientWidth;
  const start = nav.scrollLeft > 2;
  const end = max > 2 && nav.scrollLeft < max - 2;
  const fade = start && end ? 'both' : start ? 'start' : end ? 'end' : '';
  if ((nav.dataset.fade || '') !== fade) nav.dataset.fade = fade;
}
function revealCurrent(nav, current) {
  if (!current || nav.scrollWidth <= nav.clientWidth) return;
  const box = nav.getBoundingClientRect();
  const rect = current.getBoundingClientRect();
  const left = nav.scrollLeft + rect.left + rect.width / 2 - (box.left + box.width / 2);
  nav.scrollTo({ left, behavior: navGeometry && !reduced() ? 'smooth' : 'auto' });
}

export function syncGlass() {
  const nav = document.querySelector('.glass-nav');
  if (nav) {
    const current = nav.querySelector('[aria-current="page"]');
    const key = current ? current.dataset.nav : null;
    const changed = key !== navCurrent;
    // the lens glides only when the destination changes; counts arriving or a resize move it without animation
    if (changed) revealCurrent(nav, current);
    navGeometry = positionLens(nav, current, nav.querySelector('.nav-lens'), changed ? navGeometry : null);
    navCurrent = key;
    if (navGeometry) nav.dataset.lens = '';
    updateFade(nav);
  }
  const tabs = document.querySelector('.cv .tabs');
  if (tabs) {
    let lens = tabs.querySelector('.tab-lens');
    if (!lens) {
      lens = document.createElement('span');
      lens.className = 'tab-lens';
      lens.setAttribute('aria-hidden', 'true');
      tabs.prepend(lens);
      tabGeometry = positionLens(tabs, tabs.querySelector('[aria-selected="true"]'), lens, tabGeometry);
    } else tabGeometry = positionLens(tabs, tabs.querySelector('[aria-selected="true"]'), lens);
  } else tabGeometry = null;
}

// A new page or case eases in; polling redraws of the same page do not.
export function animateScene(key) {
  syncGlass();
  if (scene !== null && scene !== key && !reduced()) {
    const panel = document.querySelector('#app > section');
    if (panel) panel.animate([
      { opacity: 0, transform: 'translateY(6px)' },
      { opacity: 1, transform: 'translateY(0)' }
    ], { duration: 220, easing: EASE });
  }
  scene = key;
}

const scheme = matchMedia('(prefers-color-scheme: dark)');
const isDark = () => document.documentElement.dataset.theme === 'dark' ||
  (!document.documentElement.dataset.theme && scheme.matches);
function updateAppearance() {
  const dark = isDark();
  const toggle = document.querySelector('#theme-toggle');
  toggle.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
  toggle.title = `Switch to ${dark ? 'light' : 'dark'} theme`;
  document.querySelector('meta[name="theme-color"]').content = dark ? '#101e1d' : '#f2f5f1';
}
document.querySelector('#theme-toggle').addEventListener('click', () => {
  const theme = isDark() ? 'light' : 'dark';
  // one cross-fade for the whole page instead of some controls easing their colours and the rest snapping
  const root = document.documentElement;
  const apply = () => {
    root.classList.add('theme-switching');
    root.dataset.theme = theme;
    updateAppearance();
    requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove('theme-switching')));
  };
  if (document.startViewTransition && !reduced()) document.startViewTransition(apply);
  else apply();
  try { localStorage.setItem('pq.theme', theme); } catch (_) {}
});
scheme.addEventListener('change', updateAppearance);
updateAppearance();

let frame;
function scheduleGlass() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(syncGlass);
}
new MutationObserver(scheduleGlass).observe(document.querySelector('#app'), { childList: true, subtree: true });
const navEl = document.querySelector('.glass-nav');
if ('ResizeObserver' in window) new ResizeObserver(scheduleGlass).observe(navEl);
navEl.addEventListener('scroll', () => updateFade(navEl), { passive: true });
window.addEventListener('resize', scheduleGlass);
document.fonts.ready.then(scheduleGlass);
