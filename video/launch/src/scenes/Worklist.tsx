// Real worklist: the replay time-lapse (cases arriving and sorting), how a read is made, then the tiers.
import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import {E, env, kf, lerp, ramp, Key} from '../lib/anim';
import {display, ui} from '../lib/stage';
import {C} from '../lib/theme';
import {AppWindow, Callout, Cam, Cursor, cursorAt, fit, L, LAPSE, Layer, on, pad, proj, projRect, R, Ring, SHOTS, Spot, union} from '../lib/window';
import {W} from '../timeline';

const rows = SHOTS.wl_full.rects.rows as {x: number; y: number; w: number; h: number; text: string}[];
const row = (id: string) => rows.find((r) => r.text.includes(id))!;
export const HERO = row('BACH_0167');
const TIER1 = union(R('wl_full', 'tiers', 0), R('wl_full', 'rows', 0));
const TIER2 = union(R('wl_full', 'tiers', 1), rows[1], rows[2], rows[3]);
const INSET = union(row('BACH_0087'), row('BACH_0398'));
const t3i = rows.findIndex((r) => r.y > R('wl_full', 'tiers', 2).y);
const TIER3 = union(R('wl_full', 'tiers', 2), rows[t3i], rows[t3i + 1]);
const sc = (r: {y: number; h: number}) => Math.max(0, r.y + r.h / 2 - 500);

const frames = LAPSE.frames;
const ci = Math.max(0, frames.findIndex((f) => f.event === 'composite'));
const tc = frames[ci].t;
const lapseT = (vt: number) =>
  vt < 13 ? 0.3 : vt < 20.45 ? lerp(0.3, tc + 1.0, E.soft((vt - 13) / 7.45)) : tc + 1.0 + (vt - 20.45);
const lapseFrame = (vt: number) => {
  const lt = lapseT(vt);
  let k = 0;
  while (k + 1 < frames.length && frames[k + 1].t <= lt) k++;
  return frames[k].f;
};

export const T_CLICK = 27.42;

export const worklistCam = (t: number): Cam =>
  kf<Cam>(t, [
    [12.75, fit(0, 0.6, {rx: 26, ry: -18, dy: 190, dx: 0, fill: 1})],
    [14.3, fit(0, 1, {rx: 0, ry: 0, dy: 0}), E.out],
    [14.75, fit(0, 1, {dx: 0})],
    [15.35, fit(0, 0.72, {dx: -380}), E.inOut],
    [19.7, fit(0, 0.74, {dx: -380})],
    [20.25, on(TIER1, 1.5, 0, {dx: 0}), E.inOut],
    [22.0, on(TIER1, 1.52, 0)],
    [22.55, on(TIER2, 1.5, 0), E.inOut],
    [23.25, on(TIER2, 1.52, 0)],
    [23.9, on(INSET, 1.55, sc(INSET)), E.inOut],
    [24.85, on(INSET, 1.57, sc(INSET))],
    [25.45, on(TIER3, 1.5, sc(TIER3)), E.inOut],
    [26.35, on(TIER3, 1.52, sc(TIER3))],
    [27.0, on(HERO, 1.5, sc(HERO)), E.inOut],
    [T_CLICK + 0.05, on(HERO, 1.55, sc(HERO))],
    [T_CLICK + 0.5, on({...HERO, x: 60, w: 200}, 4.2, sc(HERO), {fill: 0}), E.in],
  ] as Key<Cam>[]);

const Node: React.FC<{y: number; p: number; children: React.ReactNode; h?: number}> = ({y, p, children, h = 158}) => (
  <div style={{position: 'absolute', left: 1268, top: y, width: 590, height: h, borderRadius: 20, padding: '20px 24px',
    background: 'rgba(16, 32, 30, 0.86)', border: '1px solid #a1d7be2e', boxShadow: '0 20px 60px #00000088, inset 0 1px 0 #ffffff12',
    opacity: Math.min(1, p * 1.5), transform: `translateX(${(1 - E.back(p)) * 60}px) scale(${0.96 + 0.04 * p})`,
    display: 'flex', alignItems: 'center', gap: 22}}>
    {children}
  </div>
);

const Pipe: React.FC<{y0: number; y1: number; p: number; t: number}> = ({y0, y1, p, t}) => (
  <div style={{position: 'absolute', left: 1268 + 295, top: y0, width: 2, height: (y1 - y0) * p, background: `linear-gradient(${C.hema}, ${C.hema}55)`}}>
    {p > 0.9 ? <div style={{position: 'absolute', left: -4, top: ((t * 90) % (y1 - y0)), width: 10, height: 10, borderRadius: 5, background: C.hema,
      boxShadow: `0 0 14px ${C.hema}`}} /> : null}
  </div>
);

const Pipeline: React.FC<{t: number}> = ({t}) => {
  const o = env(t, 14.55, 19.95, 0.3, 0.45);
  if (o <= 0) return null;
  const pA = ramp(t, 14.6, 15.2, E.out);
  const pB = ramp(t, W('worklist', 'pathology') - 0.25, W('worklist', 'pathology') + 0.35, E.out);
  const pC = ramp(t, W('worklist', 'reads') - 0.05, W('worklist', 'reads') + 0.55, E.out);
  const pD = ramp(t, W('worklist', 'calibrated') - 0.1, W('worklist', 'calibrated') + 0.5, E.out);
  const grid = ramp(t, 15.0, 16.2, E.inOut);
  const probs: [string, number][] = [['Normal', 1.2], ['Benign', 0.1], ['In situ', 26.1], ['Invasive', 72.6]];
  return (
    <AbsoluteFill style={{opacity: o}}>
      <Pipe y0={128 + 158} y1={330} p={pB} t={t} />
      <Pipe y0={330 + 158} y1={532} p={pC} t={t + 0.3} />
      <Pipe y0={532 + 158} y1={734} p={pD} t={t + 0.6} />
      <Node y={128} p={pA}>
        <div style={{position: 'relative', width: 176, height: 132, borderRadius: 10, overflow: 'hidden', flexShrink: 0}}>
          <Img src={staticFile('thumbs/BACH_0167.jpg')} style={{width: 176, height: 132, objectFit: 'cover'}} />
          {[1, 2, 3].map((k) => <div key={'v' + k} style={{position: 'absolute', left: (176 * k) / 4, top: 0, width: 1.5, height: 132 * grid, background: '#ffffffcc'}} />)}
          {[1, 2].map((k) => <div key={'h' + k} style={{position: 'absolute', top: (132 * k) / 3, left: 0, height: 1.5, width: 176 * grid, background: '#ffffffcc'}} />)}
        </div>
        <div>
          <div style={ui(26, 600)}>One microscopy field</div>
          <div style={{...ui(21, 450, C.mut), marginTop: 6}}>2048 × 1536 px, read as 12 regions of 512 px</div>
        </div>
      </Node>
      <Node y={330} p={pB}>
        <div style={{...display(50, 520), color: C.hema, width: 176, textAlign: 'center', whiteSpace: 'nowrap'}}>UNI2-h</div>
        <div>
          <div style={ui(26, 600)}>Pathology foundation model</div>
          <div style={{...ui(21, 450, C.mut), marginTop: 6}}>Embeds each region; weights frozen</div>
        </div>
      </Node>
      <Node y={532} p={pC}>
        <div style={{width: 176, display: 'flex', justifyContent: 'center'}}>
          <svg width="120" height="90" viewBox="0 0 120 90">
            {[0, 1, 2, 3].map((i) => <circle key={i} cx="14" cy={12 + i * 22} r="6" fill={C.mut} />)}
            {[0, 1, 2, 3].map((i) => <line key={'l' + i} x1="20" y1={12 + i * 22} x2="100" y2="45" stroke={C.hema} strokeWidth="1.6" opacity={0.75} />)}
            <circle cx="106" cy="45" r="9" fill={C.hema} />
          </svg>
        </div>
        <div>
          <div style={ui(26, 600)}>Linear probe</div>
          <div style={{...ui(21, 450, C.mut), marginTop: 6}}>Temperature-scaled; test calibration error 0.055</div>
        </div>
      </Node>
      <Node y={734} p={pD} h={206}>
        <div style={{width: '100%'}}>
          <div style={{...ui(24, 600), marginBottom: 12}}>Calibrated probabilities and a 90% confidence set</div>
          {probs.map(([k, v], i) => {
            const inSet = k === 'In situ' || k === 'Invasive';
            const bp = ramp(t, W('worklist', 'calibrated') + 0.1 + i * 0.08, W('worklist', 'calibrated') + 0.8 + i * 0.08, E.out);
            return (
              <div key={k} style={{display: 'flex', alignItems: 'center', gap: 14, height: 30}}>
                <div style={{...ui(19, 500, C.mut), width: 92}}>{k}</div>
                <div style={{flex: 1, height: 9, borderRadius: 5, background: '#ffffff12'}}>
                  <div style={{width: `${Math.max(1, v) * bp}%`, height: 9, borderRadius: 5, background: k === 'Invasive' ? C.eos : k === 'In situ' ? C.hema : C.mut}} />
                </div>
                <div style={{...ui(19, 600), width: 54, textAlign: 'right'}}>{v < 1 ? '<1%' : `${Math.round(v)}%`}</div>
                <div style={{...ui(15, 600, inSet ? C.ink : 'transparent'), width: 54, padding: '2px 0', textAlign: 'center', borderRadius: 6,
                  border: inSet ? `1px solid ${C.ink}66` : 'none', opacity: ramp(t, W('worklist', 'probe'), W('worklist', 'probe') + 0.4)}}>in set</div>
              </div>
            );
          })}
        </div>
      </Node>
    </AbsoluteFill>
  );
};

export const Worklist: React.FC<{t: number}> = ({t}) => {
  if (t < 12.7 || t > 28.1) return null;
  const cam = worklistCam(t);
  const winIn = ramp(t, 12.75, 13.6, E.out);
  const toFull = ramp(t, 21.3, 21.7);
  const layers: Layer[] = [
    {src: `captures/lapse/${lapseFrame(t)}`, y: 0, h: 1000, opacity: 1 - toFull},
    L('wl_full', toFull),
  ];
  const fadeOut = 1 - ramp(t, T_CLICK + 0.25, T_CLICK + 0.55);
  const r1 = projRect(cam, TIER1), r2 = projRect(cam, TIER2), ri = projRect(cam, INSET), r3 = projRect(cam, TIER3);
  const tF = W('worklist', 'frozen'), tA = W('worklist', 'anything'), tR = W('worklist', 'ruled'), tL = W('worklist', 'low-confidence');
  const ring1 = ramp(t, tF - 0.05, tF + 0.5, E.out) * (1 - ramp(t, 22.05, 22.35));
  const ring2 = ramp(t, tA - 0.05, tA + 0.5, E.out) * (1 - ramp(t, 23.3, 23.6));
  const ringI = ramp(t, tR - 0.25, tR + 0.3, E.out) * (1 - ramp(t, 24.9, 25.2));
  const ring3 = ramp(t, tL - 0.1, tL + 0.45, E.out) * (1 - ramp(t, 26.35, 26.65));
  const cur = cursorAt(cam, t, [[26.55, 760, HERO.y + 140], [27.25, 128, HERO.y + HERO.h / 2]]);
  return (
    <AbsoluteFill style={{opacity: fadeOut}}>
      <AppWindow cam={cam} layers={layers} opacity={winIn} blur={(1 - winIn) * 10}
        sticky={{src: 'captures/wl_final.png', h: 93}} />
      <Spot r={r1} o={ring1 * 0.8} />
      <Spot r={r2} o={ring2 * 0.8} />
      <Spot r={ri} o={ringI * 0.8} />
      <Spot r={r3} o={ring3 * 0.8} />
      <Ring r={r1} p={ring1} color={C.eos} />
      <Callout x={r1.x + r1.w * 0.36} y={r1.y + r1.h} side="bottom" p={ring1} title="Tier 1" body="Frozen sections first, 20-minute target" color={C.eos} />
      <Ring r={r2} p={ring2} color={C.eos} />
      <Callout x={r2.x + r2.w * 0.36} y={r2.y + r2.h} side="bottom" p={ring2} title="Tier 2" body="Invasive anywhere in the 90% confidence set" color={C.eos} />
      <Ring r={ri} p={ringI} color={C.eos} />
      <Callout x={ri.x + ri.w * 0.5} y={ri.y} side="top" p={ringI} title="Still tier 2" body="Invasive is not the top call, but it is in the set" color={C.eos} />
      <Ring r={r3} p={ring3} color={C.amb} />
      <Callout x={r3.x + r3.w * 0.36} y={r3.y + r3.h} side="bottom" p={ring3} title="Tier 3" body="Ambiguous set, low reliability or too little tissue" color={C.amb} />
      <Cursor x={cur.x} y={cur.y} t={t} o={env(t, 26.55, T_CLICK + 0.3, 0.2, 0.2)} clicks={[T_CLICK]} />
      <Pipeline t={t} />
    </AbsoluteFill>
  );
};
