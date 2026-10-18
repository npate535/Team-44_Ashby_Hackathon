# Deck outline: Sepia (20 main slides + backup)

The deck is built, not typed. `outputs/deck/extract_data.py` reads every number from `artifacts/eval/*.json` (plus the
BACH_0167 case file and the model column of `data/predictions.csv`) into `outputs/deck/data.json`; `outputs/deck/build.js`
formats them into `outputs/deck/Sepia_Team44.pptx`. Rebuild whenever an eval file changes (commands at the end).
Screenshots come from `outputs/deck/capture_screens.py` against a throwaway server on a copy of `artifacts/`.

Copy rules: plain, professional; assertion titles backed by a number; no slogans or question titles. Footer
"Sepia · Research prototype, not for clinical use" plus the page number. BACH credit "Images: BACH (Aresta et al.,
2019), CC BY-NC-ND 4.0" on every slide that shows BACH imagery. Frozen sections, margins, arrivals and operating rooms
are called simulated. Speaker notes on every main slide are the 8-minute talk track.

| # | Section | Title (numbers filled from data.json) | Content | Source |
|---|---|---|---|---|
| 1 | Aims | Sepia: confidence-aware triage for breast pathology | Team 44 · 2026 Ashby Prize in Computational Science Hackathon; research prototype, not for clinical use; worklist screenshot | screenshot |
| 2 | Aims | Aim: read likely invasive carcinoma first, with its evidence and a calibrated confidence | Arrival-order queues; frozen sections need an answer in about {target} minutes; AI flags rarely explain. Three aims | queue_sim (frozen target) |
| 3 | Aims | BACH_0167 arrived last of {n} biopsies and moved to tier {t} | Arrival → read and set {In situ, Invasive} → tier 2 → p63/calponin ordered; ER/PR/HER2/Ki-67 drafted for use if invasion is confirmed → sign-out waits for the stain result (not simulated). The summary cites mitoses and necrosis; the caption gives their AI spot checks so the pathologist discounts them. Reference label in situ | case file BACH_0167, spotcheck, metrics |
| 4 | Methods | Foundation-model reads computed on NCSA Delta drive a four-tier worklist | Delta → laptop analytics → app; two lanes; four tiers; safety rule | design |
| 5 | Methods | BACH: {400} microscopy images split by patient group into {218} / {50} / {132} | THUNDER split, test classes and patient groups, tissue QC | model_choice, conformal, metrics, analytics, qc |
| 6 | Methods | The production read was chosen by grouped cross-validation inside the training split | CV log-loss forest with the one-SE limit; encoders on Delta; pre-registered 512-px round (round 1 → 2 macro-F1); LoRA fine-tuning of UNI2-h on Delta (3 recipes) beat identically processed frozen UNI2-h on CV but not on test (0.852 vs 0.858 macro-F1), so production is unchanged (`finetune`) | model_choice, metrics |
| 7 | Methods | Temperature-scaled probabilities feed a conformal set; the set decides the tier | Logits → temperature → APS set → tier rule; BACH_0167 probabilities; safety rule; disclosed LAC→APS change | calibration, conformal, case file |
| 8 | Results | All {31} invasive test images flagged; invasive-vs-rest AUROC {0.999} | AUROC, sensitivity, specificity (percent intervals), macro-F1 on the 0–100 scale of the published THUNDER UNI2-h probe; ROC; confusion matrix | metrics, analytics, thunder |
| 9 | Results | {3} of {30} in situ images were read as invasive; {0} of {31} invasive as in situ | P(Invasive) histogram for in situ vs invasive; safety rule, p63/calponin | analytics, conformal |
| 10 | Results | ECE {before} → {after} after temperature scaling; tier-set coverage {91.7%} | Reliability diagram; coverage of four methods (tier set chosen post hoc, disclosed); coverage across 1,000 re-splits | calibration, analytics, conformal, conformal_resplit (B) |
| 11 | Results | All {31} invasive test images reached tiers 1–2 | Tier by true class table; all invasive images were read as invasive, so those routed to the simulated frozen lane would have reached tier 2 anyway (computed); in situ images lifted by the safety rule | analytics |
| 12 | Results | Deferring the least reliable {20%} of mixed-model reads lifts kept accuracy from {a} to {b} | Deferral curve (y-axis from 0%); legible crop of the reliability flag; reliability error AUROC; degradation accuracy; **model mix stated next to every degraded-read number** (degraded reads are scored by fallback models, not the production model) | deferral (C), reliability, data/predictions.csv |
| 13 | Results | Removing the {k} highest-contribution regions lowers P(Invasive) by {x}, vs {y} for random regions | Exact region contributions (method), contribution heatmap, deletion comparison of random / tile-probe / concept / contribution arms | deletion (A arm), screenshot |
| 14 | Results | Spot check: invasion shown in {19/20} top regions, mitoses in {0/20} | Concept z-score by class; AI spot check (not a pathologist) with failures stated | concepts_bach, spotcheck |
| 15 | Results | Simulated margins: clear margins kept clear in {100%} of composites; {45%} of involved WSI windows found | Margin read stats; OR board; defer path | margin, screenshots |
| 16 | Results | In simulation, invasive cases are seen {14.2} min sooner and unflagged cases wait {25.2} min longer | Queue simulation with assumptions; aging guard had no measured effect | queue_sim, analytics |
| 17 | Challenges | Six problems found during the build, and how each was handled | QC bug, LAC empty sets, PathMMU/BACH mismatch, one GPU per team, partial patient IDs, LLM summary validation | qc, conformal, case files |
| 18 | Challenges | Next steps move from public images toward a silent prospective trial | External validation, pathologist review, whole slides, silent prospective trial | — |
| 19 | Challenges | Limitations: a research prototype on public data, without clinical validation | Six limitations | metrics, conformal |
| 20 | References | References: datasets, models, methods and compute | 22 citations in one style (first author + et al. for more than two authors), incl. concept bottleneck (Koh), Mondrian conformal (Vovk), Gemma; what still needs checking is in `outputs/deck/REFERENCES_TO_VERIFY.md` | README, metrics view, docs/research |
| B1 | Backup | Backup: every evaluated model on the held-out test split | Test macro-F1 of every candidate; published THUNDER probes; zero-shot | thunder |
| B2 | Backup | Backup: summaries are written only from structured findings and checked | Summary screenshot, validator steps, counts; states that the validator does not yet check whether a cited feature passed its spot check (this summary cites mitoses 0/20, necrosis 3/20) | case files, spotcheck |
| B3 | Backup | Backup: accuracy under simulated artifacts, with the models that scored each read | Degradation curves, per-signal error AUROC, model mix per severity | reliability, data/predictions.csv |
| B4 | Backup | Backup: deletion test as shown on the evaluation page | Evaluation-page screenshot | deletion |
| B5 | Backup | Backup: oncology inbox after sign-out of a clearly invasive read (BACH_0084) | Inbox item; cited features with spot checks (19/20, 18/20) | case file BACH_0084, spotcheck |

## Rebuild

```sh
python outputs/deck/extract_data.py            # final: fails if any required file or key is missing
node   outputs/deck/build.js                   # refuses a data.json with pending inputs
outputs/deck/render.sh                         # PDF + render_sepia/slide-NN.jpg
# drafting only: add --allow-pending to both commands
```

Screenshots (after workstream A's contribution layer lands, re-capture so slide 13 shows it):
see the docstring of `outputs/deck/capture_screens.py` (throwaway server on port 8011 over a copy of `artifacts/`).
