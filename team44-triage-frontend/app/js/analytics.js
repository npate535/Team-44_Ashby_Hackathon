// Vercel's HTML integration, with explicit page views for this hash router.
// This runs independently of app.js; a blocked analytics script cannot hold up the workspace.
(() => {
  const hosts = ['sepia-ashby.xyz', 'www.sepia-ashby.xyz', 'sepia-ashby.vercel.app', 'sepia-ashby-sepia2.vercel.app'];
  if (!hosts.includes(location.hostname) || new URLSearchParams(location.search).get('mock') === '1') return;

  window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
  window.va('beforeSend', (event) => {
    const url = new URL(event.url);
    url.hash = '';
    url.search = '';
    return { ...event, url: url.href };
  });
  const pages = ['worklist', 'surgeon', 'oncologist', 'metrics', 'tcga', 'spotcheck'];
  let previous;

  function pageview() {
    const [name, id] = location.hash.replace(/^#\/?/, '').split('/');
    if (name === 'app') return; // the skip-to-content link is not navigation
    const page = name === 'case' && id ? 'case' : pages.includes(name) ? name : 'worklist';
    // Count navigation between cases while grouping their analytics under one page.
    // Case identifiers, query strings and workspace contents are not sent.
    const path = '/' + page + (page === 'tcga' && id ? '/specimen' : '');
    const key = path + ((page === 'case' || page === 'tcga') && id ? '|' + id : '');
    if (key === previous) return;
    previous = key;
    window.va('pageview', { route: path, path });
  }

  const script = document.createElement('script');
  script.src = '/_vercel/insights/script.js';
  script.defer = true;
  script.dataset.disableAutoTrack = '1'; // avoid a duplicate initial view
  document.head.appendChild(script);
  window.addEventListener('hashchange', pageview);
  pageview();
})();
