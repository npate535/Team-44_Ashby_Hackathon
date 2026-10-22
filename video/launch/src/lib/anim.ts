// Timing and easing helpers. Every scene works in absolute seconds (t), not frames.
import {Easing, interpolate} from 'remotion';

export const E = {
  out: Easing.bezier(0.16, 1, 0.3, 1),
  inOut: Easing.bezier(0.65, 0, 0.35, 1),
  in: Easing.bezier(0.7, 0, 0.84, 0),
  soft: Easing.bezier(0.45, 0, 0.2, 1),
  back: Easing.bezier(0.34, 1.56, 0.64, 1),
  linear: (x: number) => x,
};

const CLAMP = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

export const lerp = (a: number, b: number, p: number) => a + (b - a) * p;

/** 0 before a, 1 after b, eased in between. */
export const ramp = (t: number, a: number, b: number, e: (x: number) => number = E.inOut) =>
  b <= a ? (t >= a ? 1 : 0) : interpolate(t, [a, b], [0, 1], {...CLAMP, easing: e});

/** Visibility envelope: fades in at a, out at b. */
export const env = (t: number, a: number, b: number, fi = 0.35, fo = 0.35) =>
  Math.min(ramp(t, a, a + fi, E.out), 1 - ramp(t, b - fo, b, E.inOut));

export type Key<T> = [number, T, ((x: number) => number)?];

/** Keyframed objects of numbers; `z` (zoom) interpolates geometrically so zooms feel even. */
export function kf<T extends Record<string, number>>(t: number, keys: Key<T>[]): T {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [t1, v1, e] = keys[i];
    const [t0, v0] = keys[i - 1];
    if (t <= t1) {
      const p = (e || E.inOut)(Math.min(1, Math.max(0, (t - t0) / (t1 - t0 || 1))));
      const o: Record<string, number> = {...v0};
      for (const k of Object.keys(v1)) {
        const a = v0[k] ?? v1[k];
        const b = v1[k];
        o[k] = k === 'z' ? Math.exp(lerp(Math.log(a), Math.log(b), p)) : lerp(a, b, p);
      }
      return o as T;
    }
  }
  return {...keys[keys.length - 1][1]};
}

/** Keyframed single number. */
export const kn = (t: number, keys: [number, number, ((x: number) => number)?][]) =>
  kf(t, keys.map(([a, v, e]) => [a, {v}, e] as Key<{v: number}>)).v;

/** Deterministic pseudo-random in [0, 1). */
export const rnd = (i: number, salt = 1) => {
  const x = Math.sin(i * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/** Decaying shake offset in px, starting at a. */
export const shake = (t: number, a: number, amp = 8, dur = 0.45) => {
  if (t < a || t > a + dur) return {x: 0, y: 0};
  const k = 1 - (t - a) / dur;
  return {x: Math.sin((t - a) * 90) * amp * k * k, y: Math.cos((t - a) * 77) * amp * 0.7 * k * k};
};
