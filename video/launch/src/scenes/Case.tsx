// Real case view of BACH_0167: exact region terms, the confidence set that kept the true class, the spot check, and
// approving the drafted orders.
import React from 'react';
import {AbsoluteFill} from 'remotion';
import hero from '../data/hero.json';
import {E, env, kf, Key, lerp, ramp, shake} from '../lib/anim';
import {display, ui} from '../lib/stage';
import {C} from '../lib/theme';
import {AppWindow, Callout, Cam, Cursor, cursorAt, fit, L, LE, Layer, on, proj, projRect, R, Rect, Ring, Spot, union} from '../lib/window';
import {T_CLICK} from './Worklist';
import {W} from '../timeline';

const HM = R('case_full', 'heatmap_img');
const HMCARD = R('case_full', 'heatmap');
const TILES = Array.from({length: 11}, (_, i) => R('case_full', 'tiles', i));
const T5 = TILES[5];
const TIP: Rect = {x: 485, y: 1172, w: 300, h: 266};          // hover tooltip of region 5 (from heatmap_hover5.png)
const PRED: Rect = {x: 942, y: 1071, w: 626, h: 200};
const RISK = R('case_full', 'risk');
const PROB_ROWS: Rect = {x: 962, y: 1268, w: 585, h: 60};       // In situ and Invasive rows, both "in set"
const SET = R('case_full', 'set');
const SETZONE = union(PROB_ROWS, SET);
const BADGE: Rect = {x: 207, y: 338, w: 90, h: 20};             // "Likely invasive"
const MULTI: Rect = {x: 401, y: 394, w: 228, h: 19};            // "More than one class in the confidence set"
const HEADTAGS = union(R('case_full', 'title'), MULTI);
const ORD0 = R('case_full', 'orders', 0);
const AGENT = R('case_full', 'agent');
const PASS = union(R('case_full', 'why_rows', 1), R('case_full', 'why_rows', 2));
const MIT = R('case_full', 'why_rows', 0);
const MIT_TAG = R('case_full', 'spot_tag');
const MIT_NAME: Rect = {x: 61, y: 2215, w: 150, h: 20};
const SUMP2: Rect = {x: 53, y: 637, w: 668, h: 62};             // "...did not pass an AI spot check, so they are not given as reasons."
const APPROVE = R('flow_0', 'approve');
const CONFIRM = R('flow_1', 'confirm');
const sc = (r: Rect) => Math.max(0, r.y + r.h / 2 - 500);

export const C1 = 58.58, C2 = 59.3;     // approve, confirm clicks
export const CASE_END = 60.95;

const tHm = W('heatmap', 'read');
const tSum = W('heatmap', 'sum');
const tLinear = W('heatmap', 'linear');
const tExact = W('heatmap', 'exact');
const tSo = W('twist', 'so');
const tShows = W('heatmap', 'heatmap');
const t73 = W('twist', 'seventy-three');
const tSitu = W('twist', 'actually');
const tNever = W('twist', 'never');
const tNinety = W('twist', 'ninety-percent');
const tFlag = W('twist', 'flagged');
const tP63 = W('twist', 'p');
const tPassed = W('spot', 'passed');
const tFailed = W('spot', 'failed');
const tZero = W('spot', 'zero');
const tNeverS = W('spot', 'never');

export const caseCam = (t: number): Cam =>
  kf<Cam>(t, [
    [T_CLICK + 0.33, fit(0, 1.16)],
    [T_CLICK + 1.0, fit(0, 1.0), E.out],
    [tHm - 0.1, fit(0, 1.0)],
    [tHm + 0.7, on(HM, 1.2, sc(HM), {dx: -360}), E.inOut],
    [tShows - 0.45, on(HM, 1.22, sc(HM), {dx: -360})],
    [tShows + 0.15, on(union(T5, TIP), 1.75, sc(HM), {dx: 0}), E.inOut],
    [t73 - 0.3, on(union(T5, TIP), 1.86, sc(HM))],
    [t73 + 0.45, on(PRED, 2.0, sc(PRED)), E.inOut],
    [tSitu, on(PRED, 2.08, sc(PRED))],
    [tNever - 0.35, on(PRED, 1.9, sc(PRED))],
    [tNever + 0.25, on(SETZONE, 2.15, sc(SETZONE)), E.inOut],
    [tSo - 0.1, on(SETZONE, 2.2, sc(SETZONE))],
    [tSo + 0.55, on(HEADTAGS, 1.8, 0), E.inOut],
    [tP63 - 0.6, on(HEADTAGS, 1.84, 0)],
    [tP63 - 0.05, on(ORD0, 2.05, 0), E.inOut],
    [49.9, on(ORD0, 2.15, 0)],
    [50.6, on(PASS, 1.85, sc(PASS)), E.inOut],
    [tFailed - 0.85, on(PASS, 1.88, sc(PASS))],
    [tFailed - 0.25, on(MIT, 2.25, sc(MIT)), E.inOut],
    [tNeverS - 0.45, on(MIT, 2.32, sc(MIT))],
    [tNeverS + 0.2, on(SUMP2, 2.05, 200), E.inOut],
    [57.5, on(SUMP2, 2.1, 200)],
    [58.1, on(AGENT, 1.62, 0), E.inOut],
    [58.8, on(AGENT, 1.62, 0)],
    [59.2, on({...AGENT, y: AGENT.y + 90}, 1.62, 0)],
    [60.5, on({...AGENT, y: AGENT.y + 40}, 1.66, 0)],
    [CASE_END, on({...AGENT, y: AGENT.y + 40}, 1.66, 0, {dx: -2300}), E.in],
  ] as Key<Cam>[]);

const VALS = hero.tiles.map((x) => x.c);
const TOTAL = VALS.reduce((a, b) => a + b, 0) + hero.constant;
const fmt = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;

const SumPanel: React.FC<{t: number; cam: Cam}> = ({t, cam}) => {
  const o = env(t, tLinear + 0.35, tShows - 0.1, 0.4, 0.35);
  if (o <= 0) return null;
  const PX = 1236, PY = 214, CW = 140, CH = 92;
  const slot = (row: number, col: number) => ({x: PX + 32 + col * (CW + 8), y: PY + 132 + row * (CH + 8)});
  const land = (i: number) => tExact + 0.3 + i * 0.05;
  const count = ramp(t, land(0) + 0.4, land(10) + 0.75, E.out);
  return (
    <AbsoluteFill style={{opacity: o}}>
      <div style={{position: 'absolute', left: PX, top: PY, width: 4 * (CW + 8) + 56, height: 600, borderRadius: 24,
        background: 'rgba(14, 28, 26, 0.9)', border: '1px solid #f0a4be33', boxShadow: '0 30px 80px #000000aa, inset 0 1px 0 #ffffff12',
        transform: `translateX(${(1 - E.out(Math.min(1, o))) * 80}px)`}}>
        <div style={{position: 'absolute', left: 32, top: 26, ...display(40, 500)}}>Invasive score</div>
        <div style={{position: 'absolute', left: 32, top: 78, ...ui(20, 450, C.mut)}}>The model is linear, so each region adds an exact term</div>
        {hero.grid && [0, 1, 2].flatMap((r) => [0, 1, 2, 3].map((c) => {
          const s = slot(r, c);
          const has = hero.tiles.find((x) => x.row === r && x.col === c);
          return (
            <div key={`${r}${c}`} style={{position: 'absolute', left: s.x - PX, top: s.y - PY, width: CW, height: CH, borderRadius: 12,
              border: '1px dashed #ffffff22', background: has ? 'transparent' : 'repeating-linear-gradient(45deg, #ffffff08 0 6px, transparent 6px 12px)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', ...ui(15, 500, C.mut)}}>
              {has ? '' : 'QC failed, not read'}
            </div>
          );
        }))}
        <div style={{position: 'absolute', left: 32, top: 444, ...ui(24, 500, C.mut), opacity: ramp(t, land(10) - 0.2, land(10) + 0.2)}}>
          constant <span style={{color: C.ink, fontWeight: 600}}>{fmt(hero.constant)}</span>
        </div>
        <div style={{position: 'absolute', left: 32, right: 32, top: 488, height: 1, background: '#ffffff1f'}} />
        <div style={{position: 'absolute', left: 32, top: 502, display: 'flex', alignItems: 'baseline', gap: 16, opacity: ramp(t, land(0) + 0.25, land(0) + 0.55)}}>
          <span style={{...display(66, 520), color: C.eos}}>= {(TOTAL * count).toFixed(2)}</span>
          <span style={{...ui(19, 450, C.mut)}}>invasive logit (centered, calibrated)</span>
        </div>
      </div>
      {hero.tiles.map((tile, i) => {
        const r = TILES[i];
        const a = proj(cam, r.x + r.w / 2, r.y + r.h / 2);
        const s = slot(tile.row, tile.col);
        const pop = ramp(t, tLinear + 0.15 + i * 0.06, tLinear + 0.5 + i * 0.06, E.back);
        const fly = ramp(t, land(i) - 0.55, land(i), E.inOut);
        const x = lerp(a.x - 62, s.x + 8, fly), y = lerp(a.y - 30, s.y + 16, fly);
        const strength = Math.min(1, tile.c / 0.4);
        return (
          <div key={i} style={{position: 'absolute', left: x, top: y, width: 124, height: 60, borderRadius: 14,
            transform: `scale(${pop})`, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: `rgba(240, 164, 190, ${0.25 + strength * 0.65})`, border: '1px solid #ffffff40',
            boxShadow: `0 8px 26px #00000088, 0 0 ${24 * strength}px #f0a4be66`, ...ui(28, 650, '#1d0a12')}}>
            {fmt(tile.c)}
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

const Stamp: React.FC<{t: number}> = ({t}) => {
  const a = tSitu + 0.05;
  if (t < a - 0.05 || t > tSo + 0.2) return null;
  const hit = ramp(t, a, a + 0.32, E.back);
  const dock = ramp(t, tNever - 0.55, tNever - 0.05, E.inOut);
  const out = ramp(t, tSo - 0.3, tSo + 0.1);
  const x = lerp(1180, 1585, dock), y = lerp(560, 128, dock), s = lerp(1, 0.52, dock);
  return (
    <div style={{position: 'absolute', left: x, top: y, transform: `translate(-50%, -50%) scale(${(1.7 - 0.7 * hit) * s}) rotate(${lerp(-9, -3, hit)}deg)`,
      opacity: Math.min(1, hit * 2) * (1 - out), padding: '26px 40px 30px', borderRadius: 22, background: 'rgba(30, 14, 22, 0.92)',
      border: `3px solid ${C.eos}`, boxShadow: `0 30px 90px #000000cc, 0 0 60px #f0a4be44`, textAlign: 'center'}}>
      <div style={{...ui(19, 600, C.eos), letterSpacing: 2, textTransform: 'uppercase'}}>BACH reference label</div>
      <div style={{...display(70, 540), marginTop: 8}}>In situ carcinoma</div>
      <div style={{...ui(23, 500, C.amb), marginTop: 10}}>The model's top call is wrong</div>
    </div>
  );
};

export const Case: React.FC<{t: number}> = ({t}) => {
  if (t < T_CLICK + 0.25 || t > CASE_END + 0.1) return null;
  const cam = caseCam(t);
  const sh = shake(t, tSitu + 0.08, 9);
  const camS = {...cam, dx: (cam.dx ?? 0) + sh.x, dy: (cam.dy ?? 0) + sh.y};
  const hover = ramp(t, tShows - 0.05, tShows + 0.15) * (1 - ramp(t, t73 + 0.1, t73 + 0.35));
  const layers: Layer[] = [
    L('case_full', 1),
    LE('heatmap_hover5', HMCARD, hover),
    L('flow_0', ramp(t, 57.6, 57.9)),
    L('flow_1', ramp(t, C1 + 0.03, C1 + 0.12)),
    L('flow_2', ramp(t, C2 + 0.03, C2 + 0.14)),
  ];
  const appear = ramp(t, T_CLICK + 0.3, T_CLICK + 0.6);
  const ringRisk = ramp(t, t73 + 0.35, t73 + 0.8, E.out) * (1 - ramp(t, tNever - 0.4, tNever - 0.1));
  const dimAll = ramp(t, tSitu, tSitu + 0.2) * (1 - ramp(t, tNever - 0.55, tNever - 0.1));
  const ringRows = ramp(t, tNever, tNever + 0.45, E.out) * (1 - ramp(t, tSo - 0.2, tSo + 0.05));
  const ringSet = ramp(t, tNinety, tNinety + 0.45, E.out) * (1 - ramp(t, tSo - 0.2, tSo + 0.05));
  const ringFlag = ramp(t, tSo + 0.5, tSo + 0.9, E.out) * (1 - ramp(t, tP63 - 0.55, tP63 - 0.3));
  const ringOrd = ramp(t, tP63 + 0.1, tP63 + 0.5, E.out) * (1 - ramp(t, 49.85, 50.15));
  const ringPass = ramp(t, tPassed - 0.2, tPassed + 0.3, E.out) * (1 - ramp(t, tFailed - 0.7, tFailed - 0.45));
  const ringMit = ramp(t, tFailed, tFailed + 0.4, E.out) * (1 - ramp(t, tNeverS - 0.4, tNeverS - 0.1));
  const strike = ramp(t, tZero, tZero + 0.4, E.inOut) * (1 - ramp(t, tNeverS - 0.4, tNeverS - 0.1));
  const ringSum = ramp(t, tNeverS + 0.25, tNeverS + 0.7, E.out) * (1 - ramp(t, 57.45, 57.7));
  const ringOk = ramp(t, C2 + 0.25, C2 + 0.7, E.out) * (1 - ramp(t, 60.55, 60.8));
  const pr = (r: Rect) => projRect(camS, r);
  const cur = cursorAt(camS, t, [[57.95, 1300, 960], [58.5, APPROVE.x + 60, APPROVE.y + 20], [59.2, CONFIRM.x + 60, CONFIRM.y + 20], [60.4, CONFIRM.x + 140, CONFIRM.y + 70]]);
  const curH = cursorAt(camS, t, [[tShows - 0.6, T5.x + 300, T5.y + 260], [tShows, T5.x + 110, T5.y + 110]]);
  return (
    <AbsoluteFill>
      <AppWindow cam={camS} layers={layers} opacity={appear} sticky={{src: 'captures/case_top.png', h: 93}}
        blur={ramp(t, CASE_END - 0.3, CASE_END) * 14} />
      <SumPanel t={t} cam={camS} />
      <Cursor x={curH.x} y={curH.y} t={t} o={env(t, tShows - 0.6, t73 + 0.3, 0.2, 0.2)} />
      <Ring r={pr(RISK)} p={ringRisk} color={C.eos} pad={10} />
      <AbsoluteFill style={{background: '#020807', opacity: dimAll * 0.6}} />
      <Spot r={pr(SETZONE)} o={Math.max(ringRows, ringSet) * 0.75} />
      <Ring r={pr(PROB_ROWS)} p={ringRows} color={C.hema} />
      <Ring r={pr(SET)} p={ringSet} color={C.eos} pad={8} />
      <Callout x={pr(SET).x + pr(SET).w / 2} y={pr(SET).y + pr(SET).h + 8} side="bottom" p={ringSet}
        title="APS conformal set, 90% target" body="In situ stays in. The case is not treated as settled." color={C.eos} maxW={520} />
      <Spot r={pr(HEADTAGS)} o={ringFlag * 0.7} />
      <Ring r={pr(BADGE)} p={ringFlag} color={C.eos} pad={7} />
      <Ring r={pr(MULTI)} p={ringFlag} color={C.eos} pad={7} />
      <Spot r={pr(ORD0)} o={ringOrd * 0.75} />
      <Ring r={pr(ORD0)} p={ringOrd} color={C.hema} />
      <Callout x={pr(ORD0).x} y={pr(ORD0).y + pr(ORD0).h / 2} side="left" p={ringOrd}
        title="Drafted, not ordered" body="Myoepithelial stains for in situ vs invasive" color={C.hema} maxW={420} />
      <Spot r={pr(PASS)} o={ringPass * 0.7} />
      <Ring r={pr(PASS)} p={ringPass} color={C.grn} />
      <Callout x={pr(PASS).x + pr(PASS).w} y={pr(PASS).y + pr(PASS).h / 2} side="left" p={ringPass}
        title="AI spot check" body="18 of 20 regions passed" color={C.grn} maxW={360} />
      <Spot r={pr(MIT)} o={ringMit * 0.7} />
      <Ring r={pr(MIT_TAG)} p={ringMit} color={C.amb} pad={7} />
      {strike > 0 ? (
        <div style={{position: 'absolute', left: pr(MIT_NAME).x - 6, top: pr(MIT_NAME).y + pr(MIT_NAME).h * 0.55,
          width: (pr(MIT_NAME).w + 12) * strike, height: 4, borderRadius: 2, background: C.amb, boxShadow: `0 0 12px ${C.amb}`}} />
      ) : null}
      <Callout x={pr(MIT).x + pr(MIT).w * 0.62} y={pr(MIT).y + pr(MIT).h / 2} side="right" p={ringMit}
        title="Spot check failed, 0 of 20" body="Association only. Never given as a reason." color={C.amb} maxW={380} />
      <Ring r={pr(SUMP2)} p={ringSum} color={C.amb} />
      <Ring r={pr(union(R('flow_2', 'agent'), R('flow_2', 'orders', 1)))} p={ringOk} color={C.grn} />
      <Callout x={pr(R('flow_2', 'agent')).x} y={pr(R('flow_2', 'agent')).y + 120} side="left" p={ringOk}
        title="Approved by the pathologist" body="Recorded in Sepia. Nothing goes to a lab system." color={C.grn} maxW={400} />
      <Cursor x={cur.x} y={cur.y} t={t} o={env(t, 57.95, 60.6, 0.2, 0.2)} clicks={[C1, C2]} />
      <Stamp t={t} />
    </AbsoluteFill>
  );
};
