// Custom illustration: a first-in, first-out queue of real BACH test images with their reference labels.
import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import testset from '../data/testset.json';
import {E, kn, ramp} from '../lib/anim';
import {display, ui} from '../lib/stage';
import {C} from '../lib/theme';
import {W} from '../timeline';

const LABEL: Record<string, string> = {Normal: 'Normal tissue', Benign: 'Benign lesion', InSitu: 'In situ carcinoma', Invasive: 'Invasive carcinoma'};
const cases = (testset as {cases: {id: string; label: string}[]}).cases;
const calm = cases.filter((c) => c.label === 'Benign' || c.label === 'Normal').slice(3, 11);
const inv = cases.find((c) => c.id === 'BACH_0128') ?? cases.find((c) => c.label === 'Invasive')!;
export const QUEUE = [...calm.slice(0, 7), inv, calm[7]];
const INV = 7;

const ROW_H = 78, GAP = 10, TOP = 168, LEFT = 960 - 420;

export const Problem: React.FC<{t: number}> = ({t}) => {
  if (t < 3.9 || t > 13.9) return null;
  const tInv = W('problem', 'invasive');
  const tBehind = W('problem', 'behind');
  const tSort = 12.15;
  const hl = ramp(t, tInv - 0.1, tInv + 0.45, E.out);
  const sortP = ramp(t, tSort, tSort + 0.75, E.back);
  const exit = ramp(t, 12.95, 13.7, E.inOut);
  const head = ramp(t, 4.05, 4.7, E.out);
  const brace = ramp(t, tBehind, tBehind + 0.6, E.out) * (1 - ramp(t, tSort - 0.2, tSort + 0.1));
  return (
    <AbsoluteFill style={{opacity: 1 - exit, transform: `scale(${1 - exit * 0.08}) translateY(${-exit * 30}px)`,
      filter: exit > 0.01 ? `blur(${exit * 8}px)` : undefined}}>
      <div style={{position: 'absolute', left: LEFT, top: 92, width: 840, display: 'flex', justifyContent: 'space-between',
        alignItems: 'baseline', opacity: head, transform: `translateY(${(1 - head) * 14}px)`}}>
        <div style={{...display(44, 460)}}>Breast biopsy queue</div>
        <div style={{...ui(22, 500, C.mut)}}>read in arrival order</div>
      </div>
      {QUEUE.map((c, i) => {
        const enter = ramp(t, 4.15 + i * 0.09, 4.85 + i * 0.09, E.out);
        const slot = i === INV ? kn(sortP, [[0, INV], [1, 0]], ) : i < INV ? i + sortP : i;
        const y = TOP + slot * (ROW_H + GAP);
        const isInv = i === INV;
        const dim = isInv ? 0 : hl * 0.42 * (1 - sortP * 0.6);
        const lift = isInv ? hl * 1 : 0;
        return (
          <div key={c.id} style={{position: 'absolute', left: LEFT, top: y, width: 840, height: ROW_H, borderRadius: 16,
            background: isInv ? `linear-gradient(90deg, #f0a4be${Math.round(16 + hl * 20).toString(16)}, #182b28ee)` : '#182b28e6',
            border: `1px solid ${isInv ? `rgba(240,164,190,${0.2 + hl * 0.6})` : '#ffffff12'}`,
            boxShadow: isInv ? `0 0 ${40 * hl}px #f0a4be44, 0 14px 40px #00000066` : '0 10px 30px #00000055',
            opacity: enter * (1 - dim), transform: `translateX(${(1 - enter) * 90}px) scale(${1 + lift * 0.03})`,
            display: 'flex', alignItems: 'center', gap: 22, padding: '0 22px', zIndex: isInv ? 2 : 1}}>
            <div style={{...ui(22, 600, isInv ? C.eos : C.mut), width: 34, textAlign: 'center'}}>
              {Math.round(i === INV ? kn(sortP, [[0, 8], [1, 1]]) : i < INV ? i + 1 + Math.round(sortP) : i + 1)}
            </div>
            <Img src={staticFile(`thumbs/${c.id}.jpg`)} style={{width: 80, height: 60, borderRadius: 8, objectFit: 'cover'}} />
            <div style={{...ui(24, 560), flex: 1}}>{c.id}</div>
            <div style={{...ui(21, 600, isInv ? '#2a0f19' : C.ink), padding: '7px 14px', borderRadius: 10,
              background: isInv ? C.eos : '#ffffff10', border: isInv ? 'none' : '1px solid #ffffff14'}}>{LABEL[c.label]}</div>
          </div>
        );
      })}
      {/* brace: the cases read before the invasive one */}
      <div style={{position: 'absolute', left: LEFT - 46, top: TOP + 6, width: 18, height: INV * (ROW_H + GAP) - GAP - 12,
        borderLeft: `3px solid ${C.hema}`, borderTop: `3px solid ${C.hema}`, borderBottom: `3px solid ${C.hema}`, borderRadius: '10px 0 0 10px',
        opacity: brace, clipPath: `inset(0 0 ${(1 - brace) * 100}% 0)`}} />
      <div style={{position: 'absolute', left: LEFT - 70, top: TOP + (INV * (ROW_H + GAP)) / 2 - 20, transform: 'translateX(-100%)',
        ...ui(26, 560, C.hema), opacity: brace, whiteSpace: 'nowrap', textAlign: 'right'}}>
        7 read first
      </div>
      <div style={{position: 'absolute', left: LEFT + 860, top: TOP + INV * (ROW_H + GAP) + ROW_H / 2 - 18,
        ...ui(26, 560, C.eos), opacity: hl * (1 - sortP), whiteSpace: 'nowrap', transform: `translateX(${(1 - hl) * 20}px)`}}>
        8th in line
      </div>
    </AbsoluteFill>
  );
};
