// Closing montage of three real views, then the end card with credits.
import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import {E, env, lerp, ramp} from '../lib/anim';
import {display, ui} from '../lib/stage';
import {C} from '../lib/theme';
import {AppWindow, fit, L} from '../lib/window';
import {W} from '../timeline';
import {RESULTS_END} from './Results';

const tOrders = W('close', 'orders');
const tShows = W('close', 'shows');
const tPath = W('close', 'pathologist');
export const END_IN = 89.85;

const VIEWS = [
  {shot: 'wl_final', label: 'Worklist', at: tOrders, dx: -560, ry: 30},
  {shot: 'case_top', label: 'Case evidence', at: tShows, dx: 0, ry: 0},
  {shot: 'inbox', label: 'Oncology inbox', at: tPath, dx: 560, ry: -30},
];

export const Close: React.FC<{t: number}> = ({t}) => {
  if (t < RESULTS_END - 0.3 || t > END_IN + 0.6) return null;
  const out = ramp(t, END_IN - 0.55, END_IN + 0.35, E.in);
  return (
    <AbsoluteFill>
      {VIEWS.map((v, i) => {
        const inn = ramp(t, RESULTS_END - 0.2 + i * 0.14, RESULTS_END + 0.6 + i * 0.14, E.out);
        const next = i < 2 ? VIEWS[i + 1].at : 99;
        const active = ramp(t, v.at - 0.2, v.at + 0.3, E.out) * (1 - ramp(t, next - 0.2, next + 0.3));
        const anyActive = ramp(t, VIEWS[0].at - 0.2, VIEWS[0].at + 0.3);
        const z = lerp(0.47, 0.56, active) * lerp(1, 0.62, out);
        const drift = (t - RESULTS_END) * 5 * (i - 1);
        const cam = {...fit(0, z), dx: v.dx * lerp(1, 0.45, out) + drift, dy: (1 - inn) * 460 - 30 - out * 50,
          rx: (1 - inn) * 32 + 5 * (1 - active), ry: v.ry * (1 - active)};
        const bright = lerp(lerp(0.85, 0.5, anyActive), 1.05, active);
        return (
          <AbsoluteFill key={v.shot} style={{zIndex: active > 0.5 ? 3 : i === 1 ? 2 : 1, filter: `brightness(${bright})`}}>
            <AppWindow cam={cam} layers={[L(v.shot)]} opacity={inn * (1 - out)} />
            <div style={{position: 'absolute', left: 960 + v.dx * lerp(1, 0.45, out) - 220, width: 440, top: 830 - out * 40, textAlign: 'center',
              ...ui(30, 600, C.ink), opacity: inn * (1 - out) * lerp(0.55, 1, active)}}>{v.label}</div>
          </AbsoluteFill>
        );
      })}
    </AbsoluteFill>
  );
};

const NAMES = ['Rishabh Bansal', 'Aarushi Das', 'Cheryl Tsai', 'Nadia Figueroa', 'Neev Patel'];

export const End: React.FC<{t: number}> = ({t}) => {
  if (t < END_IN - 0.2) return null;
  const mark = ramp(t, END_IN, END_IN + 0.9, E.out);
  const word = ramp(t, END_IN + 0.25, END_IN + 1.0, E.out);
  const url = ramp(t, END_IN + 0.7, END_IN + 1.3, E.out);
  const by = ramp(t, END_IN + 1.25, END_IN + 1.8, E.out);
  const fine = ramp(t, END_IN + 2.4, END_IN + 3.2, E.out);
  const fade = ramp(t, 96.2, 96.95, E.inOut);
  const ring = ramp(t, END_IN, END_IN + 1.6, E.out);
  return (
    <AbsoluteFill style={{opacity: 1 - fade}}>
      <div style={{position: 'absolute', left: 960 - 330, top: 300 - 330, width: 660, height: 660, borderRadius: '50%',
        border: `1px solid ${C.sepia}`, opacity: (1 - ring) * 0.5, transform: `scale(${0.3 + ring * 1.2})`}} />
      <div style={{position: 'absolute', left: 0, right: 0, top: 168, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 30}}>
        <Img src={staticFile('brand/sepia-mark-256.png')} style={{width: 132, height: 132, opacity: mark,
          transform: `scale(${0.7 + 0.3 * mark}) rotate(${(1 - mark) * -12}deg)`, filter: `blur(${(1 - mark) * 10}px) drop-shadow(0 8px 30px #d4a37344)`}} />
        <div style={{...display(132, 470), opacity: word, transform: `translateX(${(1 - word) * -30}px)`}}>Sepia</div>
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 354, textAlign: 'center', ...ui(34, 500, C.hema), letterSpacing: 0.5,
        opacity: url, transform: `translateY(${(1 - url) * 14}px)`}}>sepia-ashby.xyz</div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 520, textAlign: 'center', ...ui(22, 600, C.mut), letterSpacing: 4,
        textTransform: 'uppercase', opacity: by}}>Created by</div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 566, display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '14px 0'}}>
        {NAMES.map((n, i) => {
          const p = ramp(t, END_IN + 1.45 + i * 0.13, END_IN + 2.05 + i * 0.13, E.out);
          return (
            <div key={n} style={{display: 'flex', alignItems: 'center', opacity: p, transform: `translateY(${(1 - p) * 18}px)`}}>
              {i > 0 ? <span style={{width: 6, height: 6, borderRadius: 3, background: C.sepia, margin: '0 26px', opacity: 0.8}} /> : null}
              <span style={{...ui(38, 550, C.ink)}}>{n}</span>
            </div>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: 160, right: 160, top: 880, textAlign: 'center', ...ui(19, 450, C.mut), lineHeight: 1.6, opacity: fine * 0.9}}>
        Research prototype, not for clinical use. Data: BACH (Aresta et al., 2019) and TCGA-BRCA. Models: UNI2-h, CONCH and MUSK.
        <br />Compute: NCSA Delta. Built for the 2026 Ashby Prize in Computational Science Hackathon.
      </div>
    </AbsoluteFill>
  );
};
