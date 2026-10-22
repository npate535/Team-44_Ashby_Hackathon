import React from 'react';
import {Composition} from 'remotion';
import {Launch} from './Launch';
import {DURATION_S, FPS} from './timeline';

export const Root: React.FC = () => (
  <Composition id="Launch" component={Launch} durationInFrames={Math.round(DURATION_S * FPS)} fps={FPS} width={1920} height={1080} />
);
