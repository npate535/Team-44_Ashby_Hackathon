// Where each narration clip sits in the cut, and word-level times derived from the ElevenLabs alignment.
import sections from './data/sections.json';

export const FPS = 60;
export const DURATION_S = 97;

export const PLACE = {
  problem: 4.0,
  worklist: 12.5,
  heatmap: 26.9,
  twist: 36.3,
  spot: 50.1,
  workflow: 57.9,
  results: 71.2,
  close: 85.7,
} as const;
export type Sec = keyof typeof PLACE;

type Word = {w: string; t0: number; t1: number};
type Section = {id: Sec; start: number; end: number; clip_start: number; clip_end: number; words: Word[]};
const SECTIONS = (sections as unknown as {sections: Section[]}).sections;
const byId = Object.fromEntries(SECTIONS.map((s) => [s.id, s])) as Record<Sec, Section>;

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9'-]/g, '');

/** Video time (s) at which a word starts; nth counts repeats of the same word within the section. */
export const W = (sec: Sec, word: string, nth = 0): number => {
  const s = byId[sec];
  let k = -1;
  for (const w of s.words) {
    if (norm(w.w) === norm(word) && ++k === nth) return PLACE[sec] + (w.t0 - s.clip_start);
  }
  throw new Error(`word "${word}" #${nth} not found in ${sec}`);
};

/** Video time (s) at which a word ends. */
export const WE = (sec: Sec, word: string, nth = 0): number => {
  const s = byId[sec];
  let k = -1;
  for (const w of s.words) {
    if (norm(w.w) === norm(word) && ++k === nth) return PLACE[sec] + (w.t1 - s.clip_start);
  }
  throw new Error(`word "${word}" #${nth} not found in ${sec}`);
};

export const clipLen = (sec: Sec) => byId[sec].clip_end - byId[sec].clip_start;
export const speechSpan = (sec: Sec) => ({
  a: PLACE[sec] + (byId[sec].start - byId[sec].clip_start),
  b: PLACE[sec] + (byId[sec].end - byId[sec].clip_start),
});
export const SECS = Object.keys(PLACE) as Sec[];
export const fr = (s: number) => Math.round(s * FPS);
