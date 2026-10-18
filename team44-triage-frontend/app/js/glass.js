// Shared glass navigation, motion and persistent appearance. No data or workflow changes.
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
let scene = null;
let tabGeometry = null;

function positionLens(container, selected, lens, previous = null) {
  if (!selected || !lens) return;
  const box = container.getBoundingClientRect();
  const rect = selected.getBoundingClientRect();
  const x = rect.left - box.left + container.scrollLeft - container.clientLeft;
  lens.style.width = `${rect.width}px`;
  lens.style.transform = `translateX(${x}px)`;
  if (previous && !reduced() && (previous.x !== x || previous.width !== rect.width)) {
    lens.animate([
      { transform: `translateX(${previous.x}px)`, width: `${previous.width}px` },
      { transform: `translateX(${x}px)`, width: `${rect.width}px` }
    ], { duration: 520, easing: 'cubic-bezier(.22, 1.15, .36, 1)' });
  }
  return { x, width: rect.width };
}

export function syncGlass() {
  const nav = document.querySelector('.glass-nav');
  if (nav) positionLens(nav, nav.querySelector('[aria-current="page"]'), nav.querySelector('.nav-lens'));
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

export function animateScene(key) {
  syncGlass();
  if (scene !== null && scene !== key && !reduced()) {
    const panel = document.querySelector('#app > section');
    if (panel) panel.animate([
      { opacity: 0, transform: 'translateY(8px)', filter: 'blur(3px)' },
      { opacity: 1, transform: 'translateY(0)', filter: 'blur(0)' }
    ], { duration: 380, easing: 'cubic-bezier(.22, 1, .36, 1)' });
  }
  scene = key;
}

const scheme = matchMedia('(prefers-color-scheme: dark)');
function updateAppearance() {
  const dark = document.documentElement.dataset.theme === 'dark' ||
    (!document.documentElement.dataset.theme && scheme.matches);
  const toggle = document.querySelector('#theme-toggle');
  toggle.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
  toggle.title = `Switch to ${dark ? 'light' : 'dark'} theme`;
  document.querySelector('meta[name="theme-color"]').content = dark ? '#101e1d' : '#f2f5f1';
}
document.querySelector('#theme-toggle').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark' ||
    (!document.documentElement.dataset.theme && scheme.matches);
  const theme = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('pq.theme', theme); } catch (_) {}
  updateAppearance();
});
scheme.addEventListener('change', updateAppearance);
updateAppearance();

let frame;
function scheduleGlass() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(syncGlass);
}
new MutationObserver(scheduleGlass).observe(document.querySelector('#app'), { childList: true, subtree: true });
if ('ResizeObserver' in window) new ResizeObserver(scheduleGlass).observe(document.querySelector('.glass-nav'));
window.addEventListener('resize', scheduleGlass);
document.fonts.ready.then(scheduleGlass);
