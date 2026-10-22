// Backdrop, film grain, fonts and shared type styles.
import React, {useEffect, useState} from 'react';
import {AbsoluteFill, continueRender, delayRender, staticFile} from 'remotion';
import {C, DISPLAY_AXES, F} from './theme';

export const useFonts = () => {
  const [handle] = useState(() => delayRender('fonts'));
  useEffect(() => {
    const faces = [
      new FontFace('Fraunces', `url(${staticFile('fonts/fraunces-var.woff2')}) format('woff2')`, {weight: '100 900'}),
      new FontFace('Bricolage Grotesque', `url(${staticFile('fonts/bricolage-grotesque-var.woff2')}) format('woff2')`, {weight: '200 800', stretch: '75% 100%'}),
    ];
    Promise.all(faces.map((f) => f.load()))
      .then((loaded) => {
        loaded.forEach((f) => document.fonts.add(f));
        continueRender(handle);
      })
      .catch((e) => {
        console.error(e);
        continueRender(handle);
      });
  }, [handle]);
};

/** Deep hematoxylin-green field with slow eosin and hematoxylin glows. */
export const Backdrop: React.FC<{t: number; tint?: number}> = ({t, tint = 1}) => {
  const a = t * 0.07;
  const x1 = 560 + Math.sin(a) * 260, y1 = 380 + Math.cos(a * 1.3) * 160;
  const x2 = 1420 + Math.cos(a * 0.8) * 240, y2 = 720 + Math.sin(a * 1.1) * 150;
  return (
    <AbsoluteFill style={{background: `radial-gradient(1400px 900px at 50% 40%, ${C.bg1}, ${C.bg0} 70%)`}}>
      <AbsoluteFill style={{opacity: tint, background:
        `radial-gradient(900px 700px at ${x1}px ${y1}px, #a1d7be1c, transparent 70%),` +
        `radial-gradient(820px 620px at ${x2}px ${y2}px, #f0a4be16, transparent 70%),` +
        `radial-gradient(600px 400px at 50% 110%, #d4a37310, transparent 70%)`}} />
      <AbsoluteFill style={{backgroundImage: 'linear-gradient(#ffffff06 1px, transparent 1px), linear-gradient(90deg, #ffffff06 1px, transparent 1px)',
        backgroundSize: '64px 64px', backgroundPosition: `${(t * 6) % 64}px ${(t * 3) % 64}px`,
        maskImage: 'radial-gradient(ellipse 70% 60% at 50% 50%, black, transparent)', opacity: 0.55}} />
    </AbsoluteFill>
  );
};

export const Grain: React.FC<{frame: number; o?: number}> = ({frame, o = 0.05}) => {
  const ox = (frame * 137) % 512, oy = (frame * 251) % 512;
  return (
    <AbsoluteFill style={{pointerEvents: 'none', opacity: o, mixBlendMode: 'overlay',
      backgroundImage: `url(${staticFile('brand/grain.png')})`, backgroundPosition: `${ox}px ${oy}px`}} />
  );
};

export const Vignette: React.FC<{o?: number}> = ({o = 1}) => (
  <AbsoluteFill style={{pointerEvents: 'none', opacity: o,
    background: 'radial-gradient(ellipse 85% 80% at 50% 50%, transparent 55%, #000000a0 100%)'}} />
);

export const display = (size: number, weight = 480): React.CSSProperties => ({
  fontFamily: F.display, fontSize: size, fontWeight: weight, fontVariationSettings: DISPLAY_AXES, letterSpacing: -0.02 * size,
  color: C.ink, lineHeight: 1.02,
});
export const ui = (size: number, weight = 500, color = C.ink): React.CSSProperties => ({
  fontFamily: F.ui, fontSize: size, fontWeight: weight, color, letterSpacing: -0.005 * size,
});
