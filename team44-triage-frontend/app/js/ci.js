// Intervals for results at the boundary (every case right, or no false positives).
// A percentile bootstrap collapses to a point there ("100%–100%"), which reads as zero uncertainty.
// Report the count with an exact Clopper–Pearson interval instead. At k = n or k = 0 it has a closed form.

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

// True when a bootstrap block {estimate, lo, hi} sits on 0 or 1 with a zero-width interval.
export function atBoundary(block) {
  if (!block || typeof block !== 'object') return false;
  const e = block.estimate;
  return isNum(e) && (e === 0 || e === 1) && block.lo === e && block.hi === e;
}

// Exact two-sided interval for k = n (lo = (α/2)^(1/n), hi = 1) or k = 0 (lo = 0, hi = 1 − (α/2)^(1/n)).
export function exactBoundary(k, n, level = 0.95) {
  if (!Number.isInteger(n) || n <= 0 || (k !== 0 && k !== n)) return null;
  const edge = Math.pow((1 - level) / 2, 1 / n);
  return k === n ? { k, n, lo: edge, hi: 1, level } : { k, n, lo: 0, hi: 1 - edge, level };
}

// For a boundary block and its true denominator: {k, n, lo, hi, level}, else null.
export function boundaryInterval(block, n, level = 0.95) {
  if (!atBoundary(block)) return null;
  return exactBoundary(block.estimate === 1 ? n : 0, n, level);
}

const pc = (v) => {
  const s = (100 * v).toFixed(1);
  return s.endsWith('.0') ? s.slice(0, -2) : s;
};

// "31 of 31 (exact 95% CI 89–100%)"; "0 of 71 (exact 95% CI 0–5.1%)".
export function boundaryText(iv) {
  if (!iv) return '';
  return `${iv.k} of ${iv.n} (exact ${Math.round(iv.level * 100)}% CI ${pc(iv.lo)}–${pc(iv.hi)}%)`;
}
