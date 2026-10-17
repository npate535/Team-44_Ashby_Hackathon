// Live API client for the PathQueue backend (docs/API.md). Same method surface as mock.js.
// base '' = same origin (the backend serves this app at /); ?api=http://host:port for a separate server.

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || ('HTTP ' + status));
    this.status = status;
    this.detail = detail;
  }
}

const TIMEOUT_MS = 4000;

export function createApi({base = ''} = {}) {
  async function req(method, path, body) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(base + path, {
        method, signal: ctl.signal,
        headers: body === undefined ? {Accept: 'application/json'}
          : {Accept: 'application/json', 'Content-Type': 'application/json'},
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch (_) { data = {detail: text}; }
      if (!res.ok) {
        const detail = data && data.detail;
        throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail));
      }
      return data;
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(0, err.name === 'AbortError' ? 'timed out' : String(err.message || err));
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
    async worklist({lane = 'all', order = 'ai'} = {}) {
      const data = await req('GET', `/api/worklist?lane=${q(lane)}&order=${q(order)}`);
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
    spotcheckItems: () => req('GET', '/api/spotcheck/items'),
    spotcheck: body => req('POST', '/api/spotcheck', body),
    assetUrl: p => base + '/' + String(p || '').replace(/^\/+/, ''),
  };
}
