// The launch video: title, problem, real worklist, real case evidence, frozen-section workflow, results, close, credits.
import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Backdrop, Grain, useFonts, Vignette} from './lib/stage';
import {Case} from './scenes/Case';
import {Close, End} from './scenes/Close';
import {Frozen} from './scenes/Frozen';
import {Problem} from './scenes/Problem';
import {Results} from './scenes/Results';
import {Title} from './scenes/Title';
import {Worklist} from './scenes/Worklist';
import {Soundtrack} from './Soundtrack';
import {FPS} from './timeline';

export const Launch: React.FC = () => {
  useFonts();
  const frame = useCurrentFrame();
  const t = frame / FPS;
  return (
    <AbsoluteFill style={{background: '#060f0e', overflow: 'hidden'}}>
      <Backdrop t={t} />
      <Title t={t} />
      <Problem t={t} />
      <Worklist t={t} />
      <Case t={t} />
      <Frozen t={t} />
      <Results t={t} />
      <Close t={t} />
      <End t={t} />
      <Vignette o={0.85} />
      <Grain frame={frame} o={0.045} />
      <Soundtrack />
    </AbsoluteFill>
  );
};
