# Sepia launch video

A 97-second launch video built in [Remotion](https://www.remotion.dev/) from real screenshots of the running app, with narration, score and sound effects from ElevenLabs. The render goes to `outputs/launch-video/` and is not tracked. Captures and generated audio live in `public/`, which Git ignores because the captures contain BACH-derived imagery.

## What is real and what is drawn

- **Real UI.** Every app screen is a 2× screenshot of Sepia running locally on the 328 real case files. `capture/capture.py` drives a throwaway server: it runs the arrival replay (the time-lapse), approves orders, sends a margin call and opens a degraded frozen section. Highlights, callouts and the cursor are placed with element rectangles recorded during capture (`src/data/meta.json`). `prep.py` writes everything in `src/data/`, which the repository's `data/` ignore rule keeps out of Git.
- **Drawn only to make a point.**
  - The first-in, first-out queue uses real test images and their reference labels.
  - The pipeline card shows the hero case's real probabilities.
  - The region-sum panel uses the hero case's exact region terms from `artifacts/cases/BACH_0167.json`: 11 regions, total 2.88.
  - The 132-image mosaic uses the actual held-out test split (labels from `data/cases.csv`, calls from `data/predictions.csv`). `prep.py` checks it against the reported 31 of 31 and 98 of 101.
  - The TCGA figures are the production-model numbers in `docs/FACT_SHEET.md`.

## Rebuild

ElevenLabs key: `~/.config/ashby-2026/elevenlabs.env` (`ELEVENLABS_API_KEY=...`, mode 0600, never committed).

```sh
cd video/launch && npm install
python3 audio/gen_narration.py cjVigY5qzO86Huf0OWal /tmp/vo     # Eric on eleven_v4, one take, word timestamps
python3 audio/gen_music.py public/audio/music                    # score from a section plan timed to the cut
python3 audio/gen_sfx.py public/audio/sfx && python3 audio/normalize_sfx.py public/audio/sfx public/audio/sfx_n
# capture: start a throwaway server on a fresh workspace DB (see .claude/launch.json "sepia-video"), then
../../.venv/bin/python capture/capture.py http://127.0.0.1:8123 public/captures
../../.venv/bin/python prep.py /tmp/vo
node scripts/stills.mjs out/stills 20 40 60                       # spot-check frames
npm run render                                                     # or see the render/master commands below
```

Master: `scripts/master.sh` renders at CRF 16, then normalizes the audio to −14 LUFS and −1 dBTP with two-pass `loudnorm`.

**Change the voice:** run `gen_narration.py` with another voice ID, then `prep.py`. Visual beats follow the new word timestamps. If a section runs longer than its slot, adjust `PLACE` in `src/timeline.ts`. The script lives in `narration.json` and uses v4 audio tags such as `[confident]` and `[dramatic]`.
