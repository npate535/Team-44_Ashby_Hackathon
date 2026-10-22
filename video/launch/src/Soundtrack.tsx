// Narration, the score (ducked under speech) and sound effects, all keyed to the same word times as the picture.
import React from 'react';
import {Audio, Sequence, interpolate, staticFile} from 'remotion';
import {C1, C2, CASE_END} from './scenes/Case';
import {END_IN} from './scenes/Close';
import {FROZEN_END} from './scenes/Frozen';
import {RESULTS_END} from './scenes/Results';
import {T_CLICK} from './scenes/Worklist';
import {FPS, PLACE, SECS, W, clipLen, fr, speechSpan} from './timeline';

type Cue = [number, string, number];

const tw = (s: Parameters<typeof W>[0], w: string, n = 0) => W(s, w, n);

export const CUES: Cue[] = [
  // title
  [0.2, 'riser', 0.62], [0.5, 'shimmer', 0.7], [0.95, 'glass_swipe', 0.35], [3.84, 'whoosh_deep', 0.6], [3.93, 'impact', 0.75],
  // problem
  ...Array.from({length: 9}, (_, i) => [4.15 + i * 0.09, 'tick', 0.16] as Cue),
  [tw('problem', 'invasive') - 0.05, 'sub_hit', 0.45], [tw('problem', 'invasive'), 'pop', 0.35],
  [tw('problem', 'behind'), 'swoosh_up', 0.28], [12.1, 'whoosh_soft', 0.55], [12.55, 'pop', 0.32],
  // worklist
  [12.75, 'whoosh_deep', 0.5], [13.1, 'glass_swipe', 0.25],
  [14.6, 'pop', 0.3], [tw('worklist', 'pathology') - 0.2, 'pop', 0.3], [tw('worklist', 'reads'), 'pop', 0.3], [tw('worklist', 'calibrated'), 'count', 0.22],
  [19.75, 'swoosh_up', 0.28], [tw('worklist', 'frozen') - 0.05, 'tick', 0.4], [tw('worklist', 'anything') - 0.05, 'tick', 0.4],
  [23.35, 'whoosh_soft', 0.22], [tw('worklist', 'low-confidence') - 0.6, 'whoosh_soft', 0.25], [tw('worklist', 'low-confidence') - 0.1, 'tick', 0.4],
  [26.4, 'whoosh_soft', 0.22], [T_CLICK, 'click', 0.75], [T_CLICK + 0.12, 'whoosh_deep', 0.45],
  // case: exact region terms
  [tw('heatmap', 'read') - 0.1, 'whoosh_soft', 0.3],
  ...Array.from({length: 11}, (_, i) => [tw('heatmap', 'sum') - 0.55 + i * 0.07, 'tick', 0.14] as Cue),
  [tw('heatmap', 'sum') + 0.25, 'swoosh_up', 0.35], [tw('heatmap', 'sum') + 1.05, 'count', 0.3], [tw('heatmap', 'sum') + 1.75, 'sub_hit', 0.4],
  [tw('heatmap', 'heatmap') - 0.3, 'whoosh_soft', 0.3], [tw('heatmap', 'heatmap') + 0.05, 'pop', 0.25],
  // twist
  [tw('twist', 'seventy-three') - 0.2, 'whoosh_soft', 0.3], [tw('twist', 'seventy-three') + 0.4, 'tick', 0.35],
  [tw('twist', 'actually'), 'stamp', 0.8], [tw('twist', 'actually') + 0.02, 'sub_hit', 0.5],
  [tw('twist', 'never') - 0.5, 'swoosh_up', 0.3], [tw('twist', 'never'), 'tick', 0.35], [tw('twist', 'ninety-percent'), 'tick', 0.38],
  [tw('twist', 'flagged') - 0.1, 'whoosh_soft', 0.28], [tw('twist', 'flagged') + 0.1, 'pop', 0.28],
  [tw('twist', 'p') - 0.6, 'whoosh_soft', 0.28], [tw('twist', 'p'), 'pop', 0.3],
  // spot check
  [50.1, 'whoosh_deep', 0.3], [tw('spot', 'passed') - 0.1, 'chime', 0.22], [tw('spot', 'failed') - 0.5, 'whoosh_soft', 0.25],
  [tw('spot', 'failed'), 'alert', 0.32], [tw('spot', 'zero'), 'swoosh_up', 0.25], [tw('spot', 'never') - 0.05, 'whoosh_soft', 0.38],
  // approve
  [57.95, 'pop', 0.18], [C1, 'click', 0.75], [C2, 'click', 0.75], [C2 + 0.15, 'chime', 0.4],
  // frozen sections
  [CASE_END - 0.35, 'whoosh_deep', 0.5], [CASE_END + 0.1, 'clock', 0.2], [tw('workflow', 'timer') - 0.05, 'tick', 0.35], [tw('workflow', 'never'), 'pop', 0.3],
  [tw('workflow', 'calls') - 1.1, 'whoosh_soft', 0.45], [tw('workflow', 'calls') - 0.6, 'click', 0.7], [tw('workflow', 'calls') + 0.07, 'click', 0.7],
  [tw('workflow', 'unfamiliar') - 0.6, 'whoosh_soft', 0.42], [tw('workflow', 'unfamiliar') - 0.2, 'chime', 0.3],
  [tw('workflow', 'unfamiliar') + 0.45, 'whoosh_soft', 0.42], [tw('workflow', 'reliability') - 0.05, 'alert', 0.32], [tw('workflow', 'defer') - 0.1, 'sub_hit', 0.38],
  // results
  [FROZEN_END - 0.45, 'whoosh_deep', 0.5], [FROZEN_END - 0.1, 'glass_swipe', 0.4], [FROZEN_END + 0.1, 'count', 0.26],
  [tw('results', 'all'), 'count', 0.3], [tw('results', 'all') + 1.6, 'sub_hit', 0.42],
  [tw('results', 'ninety-seven') - 0.1, 'sub_hit', 0.36], [tw('results', 'external') - 0.4, 'whoosh_deep', 0.5],
  [tw('results', 'ninety-six') - 0.05, 'sub_hit', 0.38], [tw('results', 'eighty-four') - 0.05, 'sub_hit', 0.38],
  // close and end card
  [RESULTS_END - 0.35, 'glass_swipe', 0.4], [tw('close', 'orders'), 'tick', 0.22], [tw('close', 'shows'), 'tick', 0.22], [tw('close', 'pathologist'), 'tick', 0.22],
  [END_IN - 1.7, 'reverse_swell', 0.45], [END_IN + 0.02, 'impact', 0.62], [END_IN + 0.1, 'shimmer', 0.5], [END_IN - 0.3, 'pad_tail', 0.55],
];

// Score level: lower under narration, fuller over the title, gaps and end card.
const SPANS = SECS.map((s) => speechSpan(s));
const duck = (t: number) => {
  let d = 0;
  for (const {a, b} of SPANS) d = Math.max(d, interpolate(t, [a - 0.35, a, b, b + 0.45], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  return d;
};
const musicVol = (t: number) => {
  const base = interpolate(t, [0, 0.4, 3.9, 4.6, 92, 96.9], [0, 0.55, 0.55, 0.34, 0.34, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return base * (1 - 0.65 * duck(t));
};

export const Soundtrack: React.FC = () => (
  <>
    <Audio src={staticFile('audio/music/score_plan.mp3')} volume={(f) => musicVol(f / FPS)} />
    {SECS.map((s) => (
      <Sequence key={s} from={fr(PLACE[s])} durationInFrames={Math.ceil(clipLen(s) * FPS) + 2}>
        <Audio src={staticFile(`audio/vo/${s}.wav`)} volume={1} />
      </Sequence>
    ))}
    {CUES.map(([t, name, v], i) => (
      <Sequence key={i} from={Math.max(0, fr(t))} durationInFrames={fr(name === 'pad_tail' ? 9.5 : 4)}>
        <Audio src={staticFile(`audio/sfx_n/${name}.wav`)}
          volume={name === 'pad_tail' ? (f) => v * interpolate(f / FPS, [0, 0.8, 5.2, 7.3], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}) : v} />
      </Sequence>
    ))}
  </>
);
