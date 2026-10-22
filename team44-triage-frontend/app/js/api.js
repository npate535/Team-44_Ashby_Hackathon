// Live API client for the PathQueue backend (docs/API.md). Same method surface as mock.js.
// base '' = same origin (the backend serves this app at /); ?api=http://host:port for a separate server.

// A detail longer than this (a stack trace, a page of markup) is cut short: it is shown in a notification.
const DETAIL_MAX = 140;
const short = (s) => (s.length > DETAIL_MAX ? s.slice(0, DETAIL_MAX - 1).trimEnd() + '…' : s);

/** status 0: no reply (timedOut: the request went out and no answer came back in time; else it could not be sent). */
export class ApiError extends Error {
  constructor(status, detail, { timedOut = false } = {}) {
    detail = detail ? short(String(detail)) : detail;
    super(detail || ('HTTP ' + status));
    this.status = status;
    this.detail = detail;
    this.timedOut = timedOut;
  }
}

const TIMEOUT_MS = 10000; // a cold start of the hosted backend can take several seconds

export function createApi({base = ''} = {}) {
  async function req(method, path, body, timeoutMs = TIMEOUT_MS) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetch(base + path, {
        method, signal: ctl.signal,
        headers: body === undefined ? {Accept: 'application/json'}
          : {Accept: 'application/json', 'Content-Type': 'application/json'},
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      const text = await res.text();
      let data = null;
      let json = true;
      try { data = text ? JSON.parse(text) : null; } catch (_) { json = false; }
      // Not JSON (a proxy's error page, a host that answered for a missing service): never shown as it is.
      if (!json) throw new ApiError(res.ok ? 502 : res.status, `Sepia did not respond (HTTP ${res.status})`);
      if (!res.ok) {
        const detail = data && data.detail;
        throw new ApiError(res.status, typeof detail === 'string' ? detail : detail === undefined ? '' : JSON.stringify(detail));
      }
      return data;
    } catch (err) {
      if (err instanceof ApiError) throw err;
      const timedOut = err.name === 'AbortError';
      throw new ApiError(0, timedOut ? 'timed out' : String(err.message || err), { timedOut });
    } finally {
      clearTimeout(timer);
    }
  }
  const stamp = (obj, ms) => { if (obj && typeof obj === 'object') obj._fetched_ms = ms; return obj; };
  const q = encodeURIComponent;
  return {
    mock: false,
    base,
    health: () => req('GET', '/api/health'),
    config: () => req('GET', '/api/config'),
    async worklist({lane = 'all', order = 'ai', includeContext = false} = {}) {
      const data = await req('GET', `/api/worklist?lane=${q(lane)}&order=${q(order)}` +
        (includeContext ? '&include_context=true' : ''));
      const now = Date.now();
      (data.items || []).forEach(item => stamp(item, now));
      stamp(data.sim, now);
      return data;
    },
    async getCase(id) { return stamp(await req('GET', `/api/cases/${q(id)}`), Date.now()); },
    events: id => req('GET', `/api/cases/${q(id)}/events`),
    async decide(id, action, payload = {}) {
      const res = await req('POST', `/api/cases/${q(id)}/decision`, {action, payload});
      stamp(res.case, Date.now());
      return res;
    },
    routed: to => req('GET', `/api/routed?to=${q(to)}`),
    sim: (cmd, body) => req('POST', `/api/sim/${q(cmd)}`, body),
    metrics: () => req('GET', '/api/metrics'),
    tcga: () => req('GET', '/api/tcga'),
    async getTcgaCase(id) { return stamp(await req('GET', `/api/tcga/cases/${q(id)}`), Date.now()); },
    spotcheckItems: () => req('GET', '/api/spotcheck/items'),
    spotcheck: body => req('POST', '/api/spotcheck', body),
    // LLM interpretation: GET = cached or {status: 'none'}; POST generates (a model call can take a minute or two)
    interpretation: key => req('GET', key === '__results' ? '/api/interpretation/results' : `/api/cases/${q(key)}/interpretation`),
    interpret: (key, force = false) => req('POST', (key === '__results' ? '/api/interpretation/results' : `/api/cases/${q(key)}/interpretation`) + (force ? '?force=true' : ''), undefined, 420000),
    assetUrl: p => base + '/' + String(p || '').replace(/^\/+/, ''),
  };
}
