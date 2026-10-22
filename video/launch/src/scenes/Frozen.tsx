// Real OR board and frozen-section cases: the surgeon sees only a timer until the pathologist's margin call; a degraded
// input raises the reliability flag.
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {E, env, kf, Key, ramp} from '../lib/anim';
import {C} from '../lib/theme';
import {AppWindow, Callout, Cam, Cursor, cursorAt, fit, L, on, projRect, R, Rect, Ring, Spot, union} from '../lib/window';
import {CASE_END} from './Case';
import {W} from '../timeline';

const CARD0 = R('or_before', 'cards', 0);
const TIMER0 = R('or_before', 'timer', 0);
const AWAIT0 = R('or_before', 'awaiting', 0);
const CARD1 = R('or_after', 'cards', 0);
const RESULT1 = R('or_after', 'result', 0);
const TURN1 = R('or_after', 'turnaround', 0);
const MC = union(R('frozen_call0', 'suggested'), R('frozen_call0', 'defer'));
const TUMOR = R('frozen_call0', 'tumor');
const SEND = R('frozen_call1', 'confirm');
const PICKS = R('frozen_call1', 'picks', 0);
const FLAG = R('degraded_full', 'flag');
const DEFER = R('degraded_full', 'defer');
const DTITLE = R('degraded_full', 'title');

const tTimer = W('workflow', 'timer');
const tNever = W('workflow', 'never');
const tCalls = W('workflow', 'calls');
const tUnfam = W('workflow', 'unfamiliar');
const tRel = W('workflow', 'reliability');
const tDefer = W('workflow', 'defer');

const K1 = CASE_END - 0.2;          // OR board in
const K2 = tCalls - 1.05;           // frozen case in
const CL1 = K2 + 0.45, CL2 = K2 + 1.12;
const K3 = tUnfam - 0.55;           // OR board after the call
const K4 = tUnfam + 0.5;            // degraded case
export const FROZEN_END = 71.45;

const whip = (t: number, a: number, b: number) => ({inn: ramp(t, a, a + 0.38, E.out), out: ramp(t, b - 0.3, b, E.in)});

export const Frozen: React.FC<{t: number}> = ({t}) => {
  if (t < K1 || t > FROZEN_END) return null;
  // OR board, before the call
  const w1 = whip(t, K1, K2 + 0.15);
  const camOR = kf<Cam>(t, [[K1, on(CARD0, 2.0, 0, {dx: 2300})], [K1 + 0.42, on(CARD0, 2.05, 0), E.out], [K2, on(CARD0, 2.25, 0)], [K2 + 0.2, on(CARD0, 2.25, 0, {dx: -2300}), E.in]] as Key<Cam>[]);
  // frozen case, margin call
  const w2 = whip(t, K2, K3 + 0.15);
  const camFZ = kf<Cam>(t, [[K2, on(MC, 1.7, 450, {dx: 2300})], [K2 + 0.35, on(MC, 1.72, 450), E.out], [CL1 + 0.1, on(MC, 1.72, 450)],
    [CL1 + 0.45, on(union(MC, PICKS), 1.62, 450), E.inOut], [K3, on(union(MC, PICKS), 1.62, 450)], [K3 + 0.2, on(union(MC, PICKS), 1.62, 450, {dx: -2300}), E.in]] as Key<Cam>[]);
  // OR board, after the call
  const w3 = whip(t, K3, K4 + 0.15);
  const camOR2 = kf<Cam>(t, [[K3, on(CARD1, 2.0, 0, {dx: 2300})], [K3 + 0.38, on(CARD1, 2.05, 0), E.out], [K4, on(CARD1, 2.15, 0)], [K4 + 0.2, on(CARD1, 2.15, 0, {dx: -2300}), E.in]] as Key<Cam>[]);
  // degraded case
  const w4 = {inn: ramp(t, K4, K4 + 0.38, E.out), out: ramp(t, FROZEN_END - 0.5, FROZEN_END - 0.05, E.in)};
  const camDG = kf<Cam>(t, [[K4, fit(0, 1.0, {dx: 2300})], [K4 + 0.4, fit(0, 1.02), E.out], [tRel - 0.5, on(union(DTITLE, FLAG), 1.12, 0)],
    [tRel + 0.15, on(FLAG, 1.85, 420), E.inOut], [tDefer - 0.4, on(union(FLAG, DEFER), 1.9, 420)], [FROZEN_END - 0.5, on(union(FLAG, DEFER), 1.95, 420)],
    [FROZEN_END, on(union(FLAG, DEFER), 1.4, 420, {dy: 40}), E.in]] as Key<Cam>[]);

  const ringTimer = ramp(t, tTimer - 0.1, tTimer + 0.35, E.out) * (1 - ramp(t, K2 - 0.1, K2 + 0.1));
  const ringAwait = ramp(t, tNever - 0.05, tNever + 0.4, E.out) * (1 - ramp(t, K2 - 0.1, K2 + 0.1));
  const ringRes = ramp(t, K3 + 0.3, K3 + 0.7, E.out) * (1 - ramp(t, K4 - 0.05, K4 + 0.1));
  const ringFlag = ramp(t, tRel - 0.1, tRel + 0.4, E.out);
  const ringDefer = ramp(t, tDefer - 0.15, tDefer + 0.25, E.out);
  const pulse = 1 + 0.06 * Math.sin(Math.max(0, t - tDefer) * 9) * Math.exp(-Math.max(0, t - tDefer) * 2);
  const pOR = (r: Rect) => projRect(camOR, r);
  const pOR2 = (r: Rect) => projRect(camOR2, r);
  const pDG = (r: Rect) => projRect(camDG, r);
  const cur = cursorAt(camFZ, t, [[K2 + 0.1, 1330, 1010], [CL1 - 0.05, TUMOR.x + 95, TUMOR.y + 24], [CL2 - 0.05, SEND.x + 75, SEND.y + 20], [K3, SEND.x + 120, SEND.y + 60]]);
  return (
    <AbsoluteFill>
      {t < K2 + 0.25 ? (
        <>
          <AppWindow cam={camOR} layers={[L('or_before')]} opacity={w1.inn * (1 - w1.out)} blur={(1 - w1.inn) * 10 + w1.out * 14} />
          <Ring r={pOR(TIMER0)} p={ringTimer} color={C.hema} pad={8} />
          <Callout x={pOR(CARD0).x + pOR(CARD0).w + 34} y={pOR(TIMER0).y + pOR(TIMER0).h / 2} side="right" p={ringTimer}
            title="Surgeon's board" body="A running timer for each frozen section" color={C.hema} maxW={420} />
          <Ring r={pOR(AWAIT0)} p={ringAwait} color={C.eos} pad={8} />
          <Callout x={pOR(CARD0).x + pOR(CARD0).w + 34} y={pOR(AWAIT0).y + pOR(AWAIT0).h / 2 + 40} side="right" p={ringAwait}
            title="No model read here" body="It appears only after the pathologist's call" color={C.eos} maxW={420} />
        </>
      ) : null}
      {t > K2 - 0.05 && t < K3 + 0.25 ? (
        <>
          <AppWindow cam={camFZ} layers={[L('frozen_call0'), L('frozen_call1', ramp(t, CL1 + 0.03, CL1 + 0.12)), L('frozen_call2', ramp(t, CL2 + 0.03, CL2 + 0.14))]}
            opacity={w2.inn * (1 - w2.out)} blur={(1 - w2.inn) * 10 + w2.out * 14} sticky={{src: 'captures/frozen_top.png', h: 93}} />
          <Cursor x={cur.x} y={cur.y} t={t} o={env(t, K2 + 0.15, K3 + 0.1, 0.2, 0.15)} clicks={[CL1, CL2]} />
        </>
      ) : null}
      {t > K3 - 0.05 && t < K4 + 0.25 ? (
        <>
          <AppWindow cam={camOR2} layers={[L('or_after')]} opacity={w3.inn * (1 - w3.out)} blur={(1 - w3.inn) * 10 + w3.out * 14} />
          <Ring r={pOR2(union(RESULT1, TURN1))} p={ringRes} color={C.eos} pad={10} />
          <Callout x={pOR2(CARD1).x + pOR2(CARD1).w + 34} y={pOR2(RESULT1).y + pOR2(RESULT1).h / 2 + 30} side="right" p={ringRes}
            title="Pathologist's call, then the model's" body="Timer stopped at 2 min 11 s" color={C.eos} maxW={420} />
        </>
      ) : null}
      {t > K4 - 0.05 ? (
        <>
          <AppWindow cam={camDG} layers={[L('degraded_full')]} opacity={w4.inn * (1 - w4.out)} blur={(1 - w4.inn) * 10 + w4.out * 10}
            sticky={{src: 'captures/degraded_top.png', h: 93}} />
          <Spot r={pDG(union(FLAG, DEFER))} o={ringFlag * 0.65 * (1 - w4.out)} />
          <Ring r={pDG(FLAG)} p={ringFlag * (1 - w4.out)} color={C.amb} />
          <Callout x={pDG(FLAG).x} y={pDG(FLAG).y + pDG(FLAG).h / 2} side="left" p={ringFlag * (1 - w4.out)}
            title="Reliability flag" body="Unfamiliar input. Suggested call: defer to permanent sections." color={C.amb} maxW={430} />
          <div style={{position: 'absolute', left: 0, top: 0, transformOrigin: `${pDG(DEFER).x + pDG(DEFER).w / 2}px ${pDG(DEFER).y + pDG(DEFER).h / 2}px`, transform: `scale(${pulse})`}}>
            <Ring r={pDG(DEFER)} p={ringDefer * (1 - w4.out)} color={C.amb} pad={8} width={4} />
          </div>
        </>
      ) : null}
    </AbsoluteFill>
  );
};
