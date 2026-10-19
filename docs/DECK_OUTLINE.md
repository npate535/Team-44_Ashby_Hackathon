# Deck outline: Sepia (20 main slides, divider, appendix)

The deck is generated from data files. `outputs/deck/extract_data.py` reads every number into `outputs/deck/data.json` from:

- `artifacts/eval/*.json`, including `fact_sheet.json`, `human_baseline.json`, `attention_comparison.json` and, once
  it holds the production run, `external_tcga.json`;
- the case files (`artifacts/cases/`) and the model column of `data/predictions.csv`;
- `config/literature.yaml` (verified citations and the corrected `deck_references`), `config/concepts.yaml` (feature
  groups), `core/contract.py` (the spot-check cut-off) and `docs/SCOPE_MATRIX.md` (feature statuses);
- `outputs/deck/assets_sepia/manifest.json` (which simulated cases the screenshots show).

`outputs/deck/build.js` formats `data.json` into `outputs/deck/Sepia_Team44.pptx`; no number is typed into it. The
deck builder lives in the git-ignored `outputs/deck/`. The version before this rebuild is kept in
`outputs/deck_backup_pre_judge/`.

Copy rules: plain, professional; titles are descriptive or number-backed assertions; no slogans, taglines, question
titles or "X, not Y" constructions; no team number in body copy. Footer "Sepia · Research prototype, not for clinical
use" plus the page number. BACH credit "Images: BACH (Aresta et al., 2019), CC BY-NC-ND 4.0" on every slide that
shows BACH imagery. Simulated inputs (arrivals, frozen sections, margins, operating rooms, the queue) are labelled
simulated; the feature review is labelled "AI spot check (Claude, not a pathologist)". Speaker notes on the 20 main
slides form an eight-minute talk track (about 1,100 words). Fine-tuned UNI2-h variants are not shown (team decision,
2026-10-18).

## Main slides

| # | Section | Title (numbers filled from data.json) | Content | Sources |
|---|---|---|---|---|
| 1 | Need and case | Sepia: confidence-aware triage for breast pathology | Hackathon name, one-paragraph description, research-prototype pill, worklist screenshot | screenshot |
| 2 | Need and case | Sepia orders a breast pathology worklist and shows the evidence; the pathologist makes every call | Three problems (arrival order; B-Path concordance by category; CAP frozen-section benchmark); who uses Sepia, when, what they decide, and what Sepia does not do | literature.yaml (Elmore 2015, Novis 1997) |
| 3 | Need and case | BACH_0167 arrived last of {n} biopsies and moved to tier {t} | Arrival → read and set → tier → orders drafted in Sepia only (never sent) → sign-out waits; failed-spot-check features named but not given as reasons; reference label in situ | case file BACH_0167, spotcheck |
| 4 | Mechanism | Which data feeds each component: {400} BACH microscopy images train, calibrate and test the read | Table of every dataset with unit, count and role (fits read / calibrates / tests / explains / checks / external test / not used); who built each part (pretrained frozen, trained by the team, rule-based, language model); compute | fact_sheet (`splits`, `bach`, `corpora`, `components`) |
| 5 | Mechanism | {400} BACH microscopy images split by patient group into {218} / {50} / {132}; no group appears in two splits | One field = 2048 × 1536 px read as 3 × 4 regions; split bar and per-class table with groups and known patients; leakage controls incl. the Quilt near-duplicate check | fact_sheet (`bach`, `splits`, `disjointness`, `quilt_near_duplicates`) |
| 6 | Mechanism | The production read was chosen by grouped cross-validation inside the {218} training images | CV log-loss forest with the one-SE limit; encoders; selection rule; pre-registered 512-px round; attention MIL line | model_choice, metrics, attention_comparison |
| 7 | Mechanism | Temperature-scaled probabilities feed a conformal set; the set decides the tier | Logits → temperature → APS set → tier rule; BACH_0167 probabilities; safety rule; disclosed LAC → APS change | calibration, conformal, case file |
| 8 | Evidence | All {31} invasive test images flagged: invasive-vs-rest AUROC {0.999}, macro one-vs-rest AUROC {0.965} | Both AUROC conventions defined, sensitivity, specificity, macro-F1, accuracy; ROC; confusion matrix; THUNDER footnote with our protocol (one view per region) | fact_sheet `test.auroc_conventions`, metrics, analytics, thunder |
| 9 | Evidence | {3} of {30} in situ images were read as invasive; {0} of {31} invasive as in situ | P(Invasive) histogram; safety rule; drafted p63/calponin | analytics, conformal |
| 10 | Evidence | ECE {0.184} → {0.055} after temperature scaling; tier-set coverage {91.7%} | Reliability diagram; coverage of four conformal methods; 1,000 re-splits | calibration, conformal, conformal_resplit |
| 11 | Evidence | Four pathologists reading BACH field images scored {73–96%} accuracy; Sepia scored {86%} on a different set of BACH test images | BACH reader study table (per-class sensitivity, accuracy) next to Sepia recall; B-Path concordance by category next to Sepia counterparts; "not directly comparable" band | human_baseline.json, literature.yaml |
| 12 | Evidence | External test on TCGA-BRCA with the unchanged production model | Pre-registered protocol (docs/EXTERNAL_TCGA_R2.md) and results | external_tcga.json (production run) — **pending** |
| 13 | Evidence | Deferring the least reliable {20%} of test reads lifts kept accuracy from {85.8%} to {89.8%} | Deferral curve (y-axis from 0%); error AUROC; worst-artifact accuracy; reliability flag crop; which model scored the reads | deferral, reliability, data/predictions.csv |
| 14 | Evidence | Removing the {3} highest-contribution regions lowers P(Invasive) by {0.032}; random regions change it by {−0.001} | Exact contribution heatmap; method; deletion forest (random, tile-probe, feature, contribution); evidence vs description sentence | deletion |
| 15 | Evidence | {3} of {9} named features mirror Nottingham components; AI spot check: invasion {19/20}, mitoses {0/20} | Feature z-score table grouped into Nottingham-related and additional features, with spot-check column (case-view cut-off); Nottingham definition and "no grade"; look-alike contrast screenshot (BACH_0087) and counts of reliable contrasts across case files | concepts_bach, spotcheck, concepts.yaml, case files, screenshot |
| 16 | Workflow | Simulated queue: invasive cases first read after {6.0} min instead of {20.2} min; unflagged cases wait {21.4} min longer | Simulated queue chart with assumptions; cited benchmarks (Novis 1997 frozen sections; Zarbo 1996 routine biopsies) with their intervals; "different measures" note | queue_sim, analytics, literature.yaml |
| 17 | Workflow | {8} parts evaluated on BACH, {1} on external slides, {4} only in simulation, {5} app-only, {4} proposed or not claimed | Five-column scope matrix | docs/SCOPE_MATRIX.md |
| 18 | Credibility | Limitations: curated public images, a simulated workflow and no clinical validation | Six limitations, curated BACH first | fact_sheet, metrics, conformal, spotcheck |
| 19 | Next steps | Next steps: whole slides, external sites, a reader study, a silent frozen-section trial and other organs | Five proposed steps | docs/WSI_AND_EXTENSION.md |
| 20 | References | References: datasets, methods, models, clinical literature and compute | The 22 verified `deck_references` (allocation number omitted from the Delta acknowledgement) plus Elmore 2015, Novis 1997, Zarbo 1996 | literature.yaml |

Then a full-bleed "End of presentation" divider ("Appendix follows").

## Appendix

| # | Title | Sources |
|---|---|---|
| A1 | All {31} invasive test images reached tiers 1–2 (tiers by true class) | analytics |
| A2 | Simulated margins: clear margins kept clear in {100%} of composites; {45%} of involved WSI windows found | margin, screenshots |
| A3 | Six problems found during the build, and how each was handled | qc, conformal, case files |
| A4 | Every evaluated model on the held-out test split (fine-tuned variants omitted) | thunder |
| A5 | Attention MIL and the linear probe are not separable on test: macro-F1 difference {−0.023} [{−0.073}, {+0.044}] | attention_comparison |
| A6 | All {328} shipped case summaries are written by a template from structured findings; plain-language tab; spot-check rule | fact_sheet, case files, screenshot |
| A7 | Accuracy under simulated artifacts | reliability, data/predictions.csv |
| A8 | Deletion test as shown on the evaluation page | deletion, screenshot |
| A9 | Oncology inbox after sign-out of a clearly invasive read (BACH_0084) | case file, spotcheck, screenshot |

## Pending inputs

Only these may be missing, and only in a draft built with `--allow-pending` (both scripts refuse a final build
otherwise):

- `artifacts/eval/external_tcga.json` holding the production-model run (slide 12). While it still holds the earlier
  fine-tune run, extraction marks slide 12 pending. Once it lands, slide 12's final layout must be written for its
  schema (build.js stops with a message until then).
- `config/literature.yaml` `deck_references` (slide 20).
- `outputs/deck/assets_sepia/contrast_BACH_0087.png` (slide 15), captured by `capture_screens.py`.
- The round-3 analyses (deletion contribution arm, conformal re-splits, deferral curve); all present now.

## Rebuild

From the main checkout, with the worktree (or merged repo) that holds the current artifacts:

```sh
R=/path/to/repo-or-worktree                      # holds artifacts/, config/, docs/, core/
python outputs/deck/extract_data.py --artifacts $R/artifacts --data-root <main checkout>/data   # final: fails on any missing input
node   outputs/deck/build.js                                                                     # refuses a data.json with pending inputs
outputs/deck/render.sh                                                                           # PDF + render_sepia/slide-NN.jpg
# drafting only: add --allow-pending to extract_data.py and build.js
```

Screenshots: start a throwaway server from `$R`'s code over a copy of `$R/artifacts` (never the live database), then
run `outputs/deck/capture_screens.py http://127.0.0.1:<port> outputs/deck/assets_sepia` (details in its docstring).
Delete `outputs/deck/assets_sepia/crops/` after replacing screenshots, because crops are cached by box.
