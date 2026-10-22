import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import {E, ramp} from '../lib/anim';
import {display, ui} from '../lib/stage';
import {C} from '../lib/theme';

export const Title: React.FC<{t: number}> = ({t}) => {
  if (t > 4.6) return null;
  const exit = ramp(t, 3.82, 4.42, E.in);
  const mark = ramp(t, 0.3, 1.5, E.out);
  const sub = ramp(t, 1.75, 2.5, E.out);
  const sweep = ramp(t, 2.1, 3.3, E.inOut);
  const ringP = ramp(t, 0.55, 2.2, E.out);
  const glow = 0.35 + 0.25 * Math.sin(t * 2.2);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', opacity: 1 - exit,
      transform: `scale(${1 + exit * 0.45})`, filter: exit > 0.01 ? `blur(${exit * 14}px)` : undefined}}>
      <div style={{position: 'absolute', width: 520, height: 520, borderRadius: '50%', top: 540 - 260 - 130, left: 960 - 260,
        border: `1px solid ${C.sepia}`, opacity: (1 - ringP) * 0.6, transform: `scale(${0.4 + ringP * 1.3})`}} />
      <div style={{position: 'absolute', width: 420, height: 420, borderRadius: '50%', top: 540 - 210 - 130, left: 960 - 210,
        background: `radial-gradient(circle, ${C.sepia}33, transparent 65%)`, opacity: mark * glow}} />
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', transform: 'translateY(-20px)'}}>
        <Img src={staticFile('brand/sepia-mark-256.png')} style={{width: 176, height: 176, opacity: mark,
          transform: `scale(${0.72 + 0.28 * mark}) rotate(${(1 - mark) * -14}deg)`,
          filter: `blur(${(1 - mark) * 16}px) drop-shadow(0 10px 40px #d4a37355)`}} />
        <div style={{display: 'flex', marginTop: 26, height: 168, overflow: 'hidden', paddingTop: 6}}>
          {'Sepia'.split('').map((ch, i) => {
            const p = ramp(t, 0.85 + i * 0.075, 1.65 + i * 0.075, E.out);
            return (
              <span key={i} style={{...display(158, 470), display: 'inline-block', transform: `translateY(${(1 - p) * 120}px)`,
                opacity: p, backgroundImage: `linear-gradient(100deg, ${C.ink} 0%, ${C.ink} ${sweep * 140 - 30}%, #ffffff ${sweep * 140 - 15}%, ${C.ink} ${sweep * 140}%, ${C.ink} 100%)`,
                backgroundSize: '500% 100%', backgroundPosition: `${-i * 25}% 0`, WebkitBackgroundClip: 'text', color: 'transparent'}}>
                {ch}
              </span>
            );
          })}
        </div>
        <div style={{...ui(33, 450, C.mut), marginTop: 18, opacity: sub, transform: `translateY(${(1 - sub) * 16}px)`, letterSpacing: 0.3}}>
          Confidence-aware triage for breast pathology
        </div>
      </div>
    </AbsoluteFill>
  );
};
