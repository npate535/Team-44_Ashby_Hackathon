// Held-out results drawn from the actual 132 test images (labels from data/cases.csv, calls from data/predictions.csv),
// then the external TCGA-BRCA test on the real page.
import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import testset from '../data/testset.json';
import {E, env, kf, Key, lerp, ramp, rnd} from '../lib/anim';
import {display, ui} from '../lib/stage';
import {C} from '../lib/theme';
import {AppWindow, Cam, L, on, projRect, R, Ring, Spot} from '../lib/window';
import {W} from '../timeline';
import {FROZEN_END} from './Frozen';

type TC = {id: string; label: string; top: string};
const CASES = (testset as {cases: TC[]}).cases;
const SUM = (testset as {summary: Record<string, number>}).summary;
const COLS = 12, TW = 76, TH = 57, G = 6, GX = 118, GY = 214;

const tAll = W('results', 'all');
const tSpec = W('results', 'ninety-seven');
const tExt = W('results', 'external');
const t96 = W('results', 'ninety-six');
const t84 = W('results', 'eighty-four');
export const RESULTS_END = 85.75;

const Count: React.FC<{v: number; style: React.CSSProperties}> = ({v, style}) => <span style={style}>{Math.round(v)}</span>;

const Panel: React.FC<{t: number; a: number; b: number; children: React.ReactNode}> = ({t, a, b, children}) => {
  const o = env(t, a, b, 0.4, 0.3);
  if (o <= 0) return null;
  return (
    <div style={{position: 'absolute', left: 1210, top: 270, width: 620, opacity: o, transform: `translateY(${(1 - E.out(Math.min(1, ramp(t, a, a + 0.5)))) * 30}px)`}}>
      {children}
    </div>
  );
};

const Mosaic: React.FC<{t: number}> = ({t}) => {
  if (t < FROZEN_END - 0.3 || t > tExt + 1.2) return null;
  const exitAll = ramp(t, tExt - 0.35, tExt + 0.6, E.in);
  const invP = ramp(t, tAll, tAll + 1.6, E.inOut);
  const specP = ramp(t, tSpec - 0.15, tSpec + 0.35);
  const ccx = GX + (COLS * (TW + G)) / 2, ccy = GY + (11 * (TH + G)) / 2;
  let invRank = 0;
  return (
    <AbsoluteFill>
      {CASES.map((c, i) => {
        const col = i % COLS, row = Math.floor(i / COLS);
        const gx = GX + col * (TW + G), gy = GY + row * (TH + G);
        const d = Math.hypot(gx - ccx, gy - ccy) / 700;
        const a0 = FROZEN_END - 0.15 + d * 0.75 + rnd(i) * 0.25;
        const p = ramp(t, a0, a0 + 0.8, E.out);
        const sx = 960 + (rnd(i, 2) - 0.5) * 2400, sy = 540 + (rnd(i, 3) - 0.5) * 1500;
        const isInv = c.label === 'Invasive';
        const fp = !isInv && c.top === 'Invasive';
        const k = isInv ? invRank++ : -1;
        const lit = isInv ? ramp(t, tAll + k * 0.045, tAll + 0.3 + k * 0.045, E.out) : 0;
        const dimInv = invP * (isInv ? 0 : 0.7) * (1 - specP);
        const dimSpec = specP * (isInv ? 0.75 : 0);
        const ex = ramp(t, tExt - 0.35 + rnd(i, 5) * 0.35, tExt + 0.35 + rnd(i, 5) * 0.35, E.in);
        const x = lerp(sx, gx, p), y = lerp(sy, gy, p) + ex * (400 + rnd(i, 6) * 500);
        const border = fp && specP > 0 ? C.amb : isInv && lit > 0 && specP < 1 ? C.eos : '#ffffff18';
        return (
          <div key={c.id} style={{position: 'absolute', left: x, top: y, width: TW, height: TH, borderRadius: 7, overflow: 'hidden',
            opacity: p * (1 - dimInv - dimSpec) * (1 - ex), transform: `rotate(${(1 - p) * (rnd(i, 4) - 0.5) * 50}deg) scale(${(0.6 + 0.4 * p) * (1 + lit * (1 - specP) * 0.06)})`,
            border: `2px solid ${border}`, boxShadow: lit > 0 && specP < 1 ? `0 0 ${16 * lit}px ${C.eos}aa` : fp && specP > 0 ? `0 0 14px ${C.amb}` : 'none'}}>
            <Img src={staticFile(`thumbs/${c.id}.jpg`)} style={{width: TW, height: TH, objectFit: 'cover'}} />
          </div>
        );
      })}
      <div style={{opacity: 1 - exitAll}}>
        <Panel t={t} a={FROZEN_END - 0.1} b={tAll - 0.05}>
          <div style={ui(26, 600, C.hema)}>Held-out test split</div>
          <Count v={SUM.n * ramp(t, FROZEN_END, FROZEN_END + 1.6, E.out)} style={{...display(190, 500), display: 'block', marginTop: 4}} />
          <div style={{...ui(27, 450, C.mut), lineHeight: 1.35, marginTop: 8}}>BACH images from 24 patient groups, kept out of training and model selection</div>
        </Panel>
        <Panel t={t} a={tAll} b={tSpec - 0.15}>
          <div style={ui(26, 600, C.eos)}>Invasive images called invasive</div>
          <div style={{...display(190, 500), color: C.eos, marginTop: 4}}>
            <Count v={SUM.invasive_called * ramp(t, tAll, tAll + 1.6, E.inOut)} style={{}} />
            <span style={{color: C.mut, fontSize: 110}}> / {SUM.invasive}</span>
          </div>
          <div style={{...ui(27, 450, C.mut), marginTop: 8}}>Exact 95% CI 88.8–100%</div>
        </Panel>
        <Panel t={t} a={tSpec - 0.15} b={tExt + 0.2}>
          <div style={ui(26, 600, C.hema)}>Non-invasive images not called invasive</div>
          <div style={{...display(190, 500), color: C.hema, marginTop: 4}}>97%</div>
          <div style={{...ui(27, 450, C.mut), marginTop: 8, lineHeight: 1.35}}>
            {SUM.non_invasive_not_called} of {SUM.non_invasive}. <span style={{color: C.amb}}>3 in situ images were called invasive.</span>
          </div>
        </Panel>
      </div>
    </AbsoluteFill>
  );
};

const EXT = R('tcga', 'external');
const COH = R('tcga', 'cohort');

const Stat: React.FC<{t: number; a: number; y: number; big: string; label: string; sub: string; color: string}> = ({t, a, y, big, label, sub, color}) => {
  const p = ramp(t, a - 0.1, a + 0.45, E.out);
  if (p <= 0) return null;
  return (
    <div style={{position: 'absolute', left: 1236, top: y, width: 600, padding: '24px 30px 26px', borderRadius: 22,
      background: 'rgba(14, 28, 26, 0.92)', border: `1px solid ${color}44`, boxShadow: `0 26px 70px #000000aa, 0 0 40px ${color}1c`,
      opacity: p, transform: `translateX(${(1 - E.back(p)) * 70}px)`}}>
      <div style={{display: 'flex', alignItems: 'baseline', gap: 20}}>
        <div style={{...display(108, 520), color}}>{big}</div>
        <div style={{...ui(30, 600, C.ink)}}>{sub}</div>
      </div>
      <div style={{...ui(25, 500, C.mut), marginTop: 6}}>{label}</div>
    </div>
  );
};

const Tcga: React.FC<{t: number}> = ({t}) => {
  if (t < tExt - 0.4 || t > RESULTS_END + 0.1) return null;
  const inn = ramp(t, tExt - 0.35, tExt + 0.3, E.out);
  const out = ramp(t, RESULTS_END - 0.45, RESULTS_END, E.in);
  const cam = kf<Cam>(t, [[tExt - 0.35, on(EXT, 1.05, 0, {dx: 1800})], [tExt + 0.35, on(EXT, 1.3, 0, {dx: -340}), E.out],
    [RESULTS_END - 0.45, on(EXT, 1.42, 0, {dx: -340})], [RESULTS_END, on(EXT, 1.1, 0, {dx: -340, dy: -60}), E.in]] as Key<Cam>[]);
  const ring = ramp(t, tExt + 0.45, tExt + 1.0, E.out) * (1 - out);
  const pr = projRect(cam, EXT);
  return (
    <AbsoluteFill style={{opacity: 1 - out}}>
      <AppWindow cam={cam} layers={[L('tcga')]} opacity={inn} blur={(1 - inn) * 12 + out * 10} />
      <Spot r={pr} o={ring * 0.6} />
      <Ring r={pr} p={ring} color={C.hema} />
      <Stat t={t} a={t96} y={210} big="95.7%" sub="22 of 23" label="FFPE tumor slides called invasive" color={C.eos} />
      <Stat t={t} a={t84} y={452} big="84.0%" sub="21 of 25" label="Frozen tumor slides called invasive" color={C.eos} />
      <div style={{position: 'absolute', left: 1240, top: 700, width: 600, ...ui(23, 450, C.mut), lineHeight: 1.4,
        opacity: ramp(t, t96 + 0.6, t96 + 1.2)}}>
        TCGA-BRCA, 25 patients, 65 slides. Same model, unchanged; protocol written before the run.
      </div>
    </AbsoluteFill>
  );
};

export const Results: React.FC<{t: number}> = ({t}) => (
  <>
    <Mosaic t={t} />
    <Tcga t={t} />
  </>
);
