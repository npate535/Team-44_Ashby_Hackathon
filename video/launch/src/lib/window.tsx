// The browser window that shows real Sepia screenshots, a camera over it, and screen-space overlays that are
// projected from page coordinates (CSS px of the 1600-wide capture), so rings and the cursor land on real elements.
import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import meta from '../data/meta.json';
import {E, lerp, ramp} from './anim';
import {C, F} from './theme';

export const WIN_W = 1600;
export const BAR = 46;
export const VIEW_H = 1000;
export const WIN_H = VIEW_H + BAR;
export const BASE = 0.9;

export type Rect = {x: number; y: number; w: number; h: number; text?: string};
type Shot = {rects: Record<string, Rect | Rect[] | null>; scrollY: number; pageH: number; full: boolean};
export const SHOTS = (meta as unknown as {shots: Record<string, Shot>}).shots;
export const LAPSE = (meta as unknown as {lapse: {frames: {f: string; t: number; event?: string}[]; hero_at: number}}).lapse;

export const R = (shot: string, key: string, i = 0): Rect => {
  const v = SHOTS[shot]?.rects[key];
  const r = Array.isArray(v) ? v[i] : v;
  if (!r) throw new Error(`rect ${shot}.${key}[${i}] missing`);
  return r;
};
export const union = (...rs: Rect[]): Rect => {
  const x = Math.min(...rs.map((r) => r.x));
  const y = Math.min(...rs.map((r) => r.y));
  return {x, y, w: Math.max(...rs.map((r) => r.x + r.w)) - x, h: Math.max(...rs.map((r) => r.y + r.h)) - y};
};
export const pad = (r: Rect, p: number, py = p): Rect => ({x: r.x - p, y: r.y - py, w: r.w + 2 * p, h: r.h + 2 * py});
export const center = (r: Rect) => ({x: r.x + r.w / 2, y: r.y + r.h / 2});

export type Cam = {px: number; py: number; z: number; scroll: number; dx?: number; dy?: number; fill?: number; rx?: number; ry?: number};

/** Camera that frames the whole window at zoom z with the page scrolled to `scroll`. */
export const fit = (scroll = 0, z = 1, extra: Partial<Cam> = {}): Cam => ({px: WIN_W / 2, py: scroll + WIN_H / 2 - BAR, z, scroll, ...extra});
/** Camera centred on a page rect. */
export const on = (r: Rect, z: number, scroll: number, extra: Partial<Cam> = {}): Cam => ({px: center(r).x, py: center(r).y, z, scroll, ...extra});

export const camXf = (c: Cam) => {
  const S = BASE * c.z;
  let tx = 960 - c.px * S;
  let ty = 540 - (BAR + c.py - c.scroll) * S;
  const fill = c.fill ?? 1;
  const w = WIN_W * S;
  const h = WIN_H * S;
  if (w > 1920) tx = lerp(tx, Math.min(0, Math.max(1920 - w, tx)), fill);
  if (h > 1080) ty = lerp(ty, Math.min(0, Math.max(1080 - h, ty)), fill);
  return {S, tx: tx + (c.dx ?? 0), ty: ty + (c.dy ?? 0)};
};
export const proj = (c: Cam, x: number, y: number) => {
  const {S, tx, ty} = camXf(c);
  return {x: tx + x * S, y: ty + (BAR + y - c.scroll) * S};
};
export const projRect = (c: Cam, r: Rect): Rect => {
  const a = proj(c, r.x, r.y);
  const S = camXf(c).S;
  return {x: a.x, y: a.y, w: r.w * S, h: r.h * S};
};

export type Layer = {src: string; y: number; h: number; x?: number; w?: number; opacity: number};
/** A captured shot as a window layer in page coordinates. */
export const L = (name: string, opacity = 1): Layer => {
  const s = SHOTS[name];
  if (!s) throw new Error(`shot ${name} missing`);
  return s.full
    ? {src: `captures/${name}.png`, y: 0, h: s.pageH, opacity}
    : {src: `captures/${name}.png`, y: s.scrollY, h: VIEW_H, opacity};
};
/** An element screenshot placed over its page rect (e.g. the heatmap in its hover state). */
export const LE = (file: string, r: Rect, opacity = 1): Layer => ({src: `captures/${file}.png`, x: r.x, y: r.y, w: r.w, h: r.h, opacity});

const TitleBar: React.FC<{url: string}> = ({url}) => (
  <div style={{position: 'absolute', left: 0, top: 0, width: WIN_W, height: BAR, background: '#0a1514',
    borderBottom: '1px solid #ffffff12', display: 'flex', alignItems: 'center'}}>
    <div style={{display: 'flex', gap: 9, marginLeft: 20}}>
      {['#ff5f57', '#febc2e', '#28c840'].map((c) => (
        <div key={c} style={{width: 13, height: 13, borderRadius: 7, background: c, opacity: 0.85}} />
      ))}
    </div>
    <div style={{position: 'absolute', left: '50%', transform: 'translateX(-50%)', top: 9, height: 28, minWidth: 380,
      borderRadius: 9, background: '#ffffff0d', border: '1px solid #ffffff10', display: 'flex', alignItems: 'center',
      justifyContent: 'center', gap: 8, color: C.mut, fontFamily: F.ui, fontSize: 15, letterSpacing: 0.2}}>
      <svg width="12" height="12" viewBox="0 0 12 12"><path d="M3 5V4a3 3 0 0 1 6 0v1M2.5 5h7v5.5h-7z" fill="none" stroke={C.mut} strokeWidth="1.2" /></svg>
      {url}
    </div>
  </div>
);

export const AppWindow: React.FC<{
  cam: Cam;
  layers: Layer[];
  url?: string;
  sticky?: {src: string; h: number};
  opacity?: number;
  blur?: number;
  children?: React.ReactNode; // page-coordinate content (scrolls and zooms with the page)
}> = ({cam, layers, url = 'sepia-ashby.xyz', sticky, opacity = 1, blur = 0, children}) => {
  const {S, tx, ty} = camXf(cam);
  const cx = WIN_W / 2;
  const cy = WIN_H / 2;
  const rx = cam.rx ?? 0;
  const ry = cam.ry ?? 0;
  if (opacity <= 0.001) return null;
  return (
    <AbsoluteFill style={{perspective: 2600, perspectiveOrigin: '50% 50%'}}>
      <div style={{position: 'absolute', left: 0, top: 0, width: WIN_W, height: WIN_H, transformOrigin: `${cx}px ${cy}px`,
        transform: `translate(${tx + (S - 1) * cx}px, ${ty + (S - 1) * cy}px) scale(${S}) rotateX(${rx}deg) rotateY(${ry}deg)`,
        borderRadius: 16, overflow: 'hidden', background: C.bg2, opacity,
        boxShadow: '0 40px 120px #000000aa, 0 0 0 1px #ffffff14, 0 0 80px #a1d7be14',
        filter: blur > 0.05 ? `blur(${blur}px)` : undefined}}>
        <TitleBar url={url} />
        <div style={{position: 'absolute', left: 0, top: BAR, width: WIN_W, height: VIEW_H, overflow: 'hidden'}}>
          <div style={{position: 'absolute', left: 0, top: 0, width: WIN_W, transform: `translateY(${-cam.scroll}px)`}}>
            {layers.filter((l) => l.opacity > 0.001).map((l, i) => (
              <Img key={l.src + i} src={staticFile(l.src)} style={{position: 'absolute', left: l.x ?? 0, top: l.y,
                width: l.w ?? WIN_W, height: l.h, opacity: l.opacity}} />
            ))}
            {children}
          </div>
          {sticky && cam.scroll > 1 ? (
            <div style={{position: 'absolute', left: 0, top: 0, width: WIN_W, height: sticky.h, overflow: 'hidden',
              opacity: Math.min(1, cam.scroll / 40), boxShadow: '0 6px 24px #00000055'}}>
              <Img src={staticFile(sticky.src)} style={{width: WIN_W}} />
            </div>
          ) : null}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------------------------------------------
// Screen-space overlays

/** Outline that draws itself around a projected rect. */
export const Ring: React.FC<{r: Rect; p: number; color?: string; radius?: number; width?: number; opacity?: number; pad?: number}> = ({
  r, p, color = C.eos, radius = 12, width = 3, opacity = 1, pad: pd = 6}) => {
  if (p <= 0 || opacity <= 0) return null;
  const x = r.x - pd, y = r.y - pd, w = r.w + 2 * pd, h = r.h + 2 * pd;
  const per = 2 * (w + h);
  return (
    <svg style={{position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none'}} width={1920} height={1080}>
      <rect x={x} y={y} width={w} height={h} rx={radius} fill="none" stroke={color} strokeWidth={width}
        strokeDasharray={`${per * p} ${per}`} opacity={opacity}
        style={{filter: `drop-shadow(0 0 10px ${color}) drop-shadow(0 0 22px ${color}88)`}} />
    </svg>
  );
};

/** Dims everything except a projected rect. */
export const Spot: React.FC<{r: Rect; o: number; radius?: number; pad?: number}> = ({r, o, radius = 14, pad: pd = 10}) => {
  if (o <= 0.001) return null;
  return (
    <svg style={{position: 'absolute', left: 0, top: 0}} width={1920} height={1080}>
      <defs>
        <mask id={`m${Math.round(r.x)}${Math.round(r.y)}`}>
          <rect width={1920} height={1080} fill="white" />
          <rect x={r.x - pd} y={r.y - pd} width={r.w + 2 * pd} height={r.h + 2 * pd} rx={radius} fill="black" />
        </mask>
      </defs>
      <rect width={1920} height={1080} fill="#020807" opacity={0.62 * o} mask={`url(#m${Math.round(r.x)}${Math.round(r.y)})`} />
    </svg>
  );
};

/** Glass label anchored to a screen point. */
export const Callout: React.FC<{
  x: number; y: number; p: number; title?: string; body: string; color?: string;
  side?: 'right' | 'left' | 'top' | 'bottom'; maxW?: number; size?: number;
}> = ({x, y, p, title, body, color = C.eos, side = 'right', maxW = 470, size = 27}) => {
  if (p <= 0.001) return null;
  const off = 26;
  const k = E.back(Math.min(1, p));
  const tr = {
    right: `translate(${off + (1 - k) * 24}px, -50%)`,
    left: `translate(calc(-100% - ${off + (1 - k) * 24}px), -50%)`,
    top: `translate(-50%, calc(-100% - ${off + (1 - k) * 18}px))`,
    bottom: `translate(-50%, ${off + (1 - k) * 18}px)`,
  }[side];
  return (
    <div style={{position: 'absolute', left: x, top: y, opacity: Math.min(1, p * 1.6), transform: tr, maxWidth: maxW,
      width: 'max-content', padding: '16px 22px 17px', borderRadius: 18, background: 'rgba(12, 24, 23, 0.82)',
      border: `1px solid ${color}55`, boxShadow: `0 18px 50px #000000aa, inset 0 1px 0 #ffffff14, 0 0 30px ${color}22`,
      backdropFilter: 'blur(16px)', fontFamily: F.ui, color: C.ink}}>
      {title ? (
        <div style={{fontSize: size * 0.62, fontWeight: 600, letterSpacing: 0.4, color, marginBottom: 5}}>{title}</div>
      ) : null}
      <div style={{fontSize: size, lineHeight: 1.22, fontWeight: 500, letterSpacing: -0.2}}>{body}</div>
    </div>
  );
};

/** macOS-style pointer with a click ripple; `click` is the click time in seconds. */
export const Cursor: React.FC<{x: number; y: number; t: number; o: number; clicks?: number[]}> = ({x, y, t, o, clicks = []}) => {
  if (o <= 0.001) return null;
  const last = clicks.filter((c) => t >= c - 0.12).pop();
  const press = last === undefined ? 0 : Math.max(0, 1 - Math.abs(t - last) / 0.12);
  const rip = last === undefined ? 0 : ramp(t, last, last + 0.55, E.out);
  return (
    <div style={{position: 'absolute', left: x, top: y, opacity: o}}>
      {rip > 0 && rip < 1 ? (
        <div style={{position: 'absolute', left: -32 * rip - 4, top: -32 * rip - 4, width: 64 * rip + 8, height: 64 * rip + 8,
          borderRadius: '50%', border: `3px solid ${C.hema}`, opacity: 1 - rip, boxShadow: `0 0 20px ${C.hema}`}} />
      ) : null}
      <svg width="34" height="44" viewBox="0 0 17 22" style={{transform: `scale(${1 - press * 0.16})`, transformOrigin: '2px 2px',
        filter: 'drop-shadow(0 3px 6px #000000aa)'}}>
        <path d="M1.5 1.5v16.2l4.1-3.9 2.7 6.3 2.9-1.2-2.7-6.2h5.7z" fill="#fff" stroke="#111" strokeWidth="1.1" strokeLinejoin="round" />
      </svg>
    </div>
  );
};

/** Pointer path: keyframes of page points projected through the current camera. */
export const cursorAt = (cam: Cam, t: number, keys: [number, number, number][]) => {
  let pt = {x: keys[0][1], y: keys[0][2]};
  for (let i = 1; i < keys.length; i++) {
    const [t1, x1, y1] = keys[i];
    const [t0, x0, y0] = keys[i - 1];
    if (t <= t0) break;
    const p = E.inOut(Math.min(1, (t - t0) / (t1 - t0)));
    pt = {x: lerp(x0, x1, p), y: lerp(y0, y1, p)};
  }
  return proj(cam, pt.x, pt.y);
};
