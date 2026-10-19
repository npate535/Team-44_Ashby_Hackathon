# Sepia results

> Research prototype — not for clinical use. Every number below is quoted from the eval file named beside it, with its 95% cluster-bootstrap CI (patient / pseudo-patient groups) unless stated.

Generated 2026-10-19T02:09:36Z from 21 eval files (git 19b15557e966c050e4a0cc0df67c0546d00464a6, 700a3b61b0f999a12e7a1a7b54058d6495b93689, 7045b84ab1fb8b386c44e0e9a60a98e9f974c330, bc76a1db582cac260a8734e7437050686420d0c2, e465ed396ce0c448c7c11f1a32260a779d3fd525).

## Tissue quality control (fixed in this run)

Source: `eval/qc.json` (pipeline/01q_requalify_qc). The original tissue mask required HSV value < 0.92; HSV value is the brightest RGB channel, so pale H&E tissue was rejected. With saturation > 0.07 alone, contract-tile QC pass rate went 75.1% -> 98.7% and insufficient-tissue images 51 -> 0 (before: Normal 8, Benign 3, InSitu 8, Invasive 32). Tile fractions were recomputed on display JPEGs; the old rule recomputed the same way reproduces the full-resolution Delta values (r = 0.998, mean |diff| = 0.012). Every tile had been embedded on Delta, so no new GPU work was needed; steps 04-08 and the evaluation were re-run.

## Classification (test split)

Source: `eval/metrics.json` · test n = 132, 24 patient / pseudo-patient groups · 2248 simulated variants excluded · production model `uni2h512_lr`.

| Metric | Estimate [95% CI] |
|---|---|
| Invasive vs rest AUROC | 0.999 [0.996, 1.000] (n 132, 24 groups) |
| Macro AUROC | 0.965 [0.883, 0.995] (n 132, 24 groups) |
| Macro-F1 | 0.86 [0.68, 0.93] (n 132, 24 groups) |
| Balanced accuracy | 0.87 [0.68, 0.94] (n 132, 24 groups) |
| Accuracy | 0.86 [0.76, 0.94] (n 132, 24 groups) |
| Invasive sensitivity | 1.000 [1.000, 1.000] (n 132, 24 groups) |
| Invasive specificity | 0.970 [0.936, 1.000] (n 132, 24 groups) |

Model A `uni2h512_lr` vs Model B `cb_text` (Invasive-vs-rest AUROC (uncalibrated scores)): A 0.999 [0.996, 1.000] (n 132, 24 groups); B 1.000 [1.000, 1.000] (n 132, 24 groups); paired difference A − B 0.00 [0.00, 0.00] (n 132, 24 groups).

In situ ↔ invasive confusion: 3 of 30 in situ cases called invasive; 0 of 31 invasive cases called in situ.

## Calibration

Source: `eval/calibration.json` · temperature 0.342 fit on train_oof · 10 equal-mass ECE bins (biased at small n).

| | ECE | Brier | NLL |
|---|---|---|---|
| before temperature | 0.18 [0.12, 0.25] (n 132, 24 groups) | 0.28 [0.16, 0.40] (n 132, 24 groups) | 0.57 [0.35, 0.82] (n 132, 24 groups) |
| after temperature | 0.05 [0.03, 0.18] (n 132, 24 groups) | 0.23 [0.09, 0.41] (n 132, 24 groups) | 0.57 [0.18, 1.32] (n 132, 24 groups) |

## Conformal prediction sets

Source: `eval/conformal.json` · alpha 0.1 (target coverage 90%) · tiers use aps_marginal_nonempty.

| Method | Coverage | Mean set size | Empty sets |
|---|---|---|---|
| lac_marginal | 72.0% [60.8%, 85.2%] (n 132, 24 groups) | 0.79 [0.74, 0.87] (n 132, 24 groups) | 21.2% |
| aps | 88.6% [77.7%, 95.2%] (n 132, 24 groups) | 1.27 [1.11, 1.34] (n 132, 24 groups) | 3.0% |
| mondrian | 53.8% [40.4%, 80.7%] (n 132, 24 groups) | 0.64 [0.51, 0.89] (n 132, 24 groups) | 36.4% |
| aps_marginal_nonempty | 91.7% [80.8%, 98.1%] (n 132, 24 groups) | 1.30 [1.16, 1.37] (n 132, 24 groups) | 0.0% |

Per-class coverage of the tier set (aps_marginal_nonempty): Normal 95.7% [87.5%, 100.0%] (n 23, 6 groups); Benign 85.4% [50.0%, 100.0%] (n 48, 6 groups); InSitu 90.0% [66.7%, 94.7%] (n 30, 6 groups); Invasive 100.0% [100.0%, 100.0%] (n 31, 17 groups).

Per-class LAC coverage (pre-registered): Normal 69.6% [34.6%, 92.9%] (n 23, 6 groups); Benign 64.6% [40.0%, 79.6%] (n 48, 6 groups); InSitu 60.0% [33.3%, 68.4%] (n 30, 6 groups); Invasive 96.8% [90.0%, 100.0%] (n 31, 17 groups).

**Post-hoc change, disclosed:** marginal LAC was the pre-registered tier set. On the held-out test split it returned empty sets for 21.2% of cases and no multi-class sets, so it gave the reader no differential; the 50-image calibration split is far easier than the test split, which likely violates the exchangeability the guarantee needs. Tiers now use the set named above (APS, Romano et al. 2020, with the argmax class added to empty sets). The switch was made after viewing test results, so its observed coverage is optimistic and is not an independent confirmation.

Under exchangeability and continuous scores, coverage conditional on a random calibration set follows ~Beta(46, 5) (mean 0.902, SD 0.041); the 1 - alpha = 0.90 guarantee is marginal over calibration sets, not a guarantee for this test cohort. Compare the observed held-out coverage with the target.

## Conformal coverage across 1,000 random re-splits

Source: `eval/conformal_resplit.json` · pre-registered in `docs/VALIDATION_ROUND3.md` §1 · model uni2h512_lr · 182 held-out images (calibration + test), 34 patient groups kept whole · calibration part at least 50 images (50–85, median 53) · alpha 0.1 · seed 20261018.

Coverage percentiles describe variation across calibration/test partitions of the same images, not a confidence interval for a new cohort. The original split is the THUNDER partition the app uses.

| Method | Coverage mean | Median | 2.5th–97.5th pct | Re-splits ≥ target | Mean set size | Empty sets | Invasive in set (invasive) | Invasive in set (other) | Original split coverage |
|---|---|---|---|---|---|---|---|---|---|
| lac_marginal | 89.3% | 89.1% | 67.7%–99.2% | 45.2% | 1.16 | 3.3% | 99.4% | 7.3% | 72.0% |
| aps | 90.0% | 89.2% | 76.9%–100.0% | 46.0% | 1.27 | 4.8% | 92.5% | 9.3% | 88.6% |
| mondrian | 92.4% | 93.7% | 73.5%–100.0% | 72.9% | 1.90 | 0.5% | 93.9% | 27.2% | 53.8% |
| aps_marginal_nonempty | 94.5% | 93.6% | 89.3%–100.0% | 96.1% | 1.31 | 0.0% | 100.0% | 9.3% | 91.7% |

Mondrian had at least one class with too few calibration images for a finite threshold in 80.0% of re-splits; that class was then in every set.

Original-split coverage equals `eval/conformal.json` for every method.

aps_marginal_nonempty drives the worklist tiers and was chosen after viewing test coverage on the original split; this check was pre-registered afterwards and applies the same rules to every method.

## Benchmark anchor (THUNDER split)

Source: `eval/thunder.json` · **same split, different protocol (published: whole-image resize; ours: tiles + TTA)** · THUNDER official leaderboards, BACH F1 scores; verified 2026-10-17 at https://mics-lab.github.io/thunder/leaderboards/.

| Model | Macro-F1 (%) | Balanced accuracy (%) |
|---|---|---|
| published linear probe · conch | 84.4 (no CI) | — |
| published linear probe · uni2h | 84.7 (no CI) | — |
| published linear probe · musk | 69.9 (no CI) | — |
| published zero-shot · conch | 56.1 (no CI) | — |
| published zero-shot · musk | 62.5 (no CI) | — |
| ours · uni2h512_lr (production (calibrated)) | 86.0 [67.8, 92.8] | 87.1 [67.6, 93.8] |
| ours · abmil (evaluated model) | 83.7 [70.5, 91.3] | 84.8 [70.8, 93.1] |
| ours · cb_cells (evaluated model) | 24.4 [15.7, 33.4] | 34.1 [24.0, 40.7] |
| ours · cb_text (evaluated model) | 70.8 [53.0, 82.3] | 74.2 [58.2, 86.5] |
| ours · cb_text_cells (evaluated model) | 66.7 [56.5, 76.1] | 74.0 [65.7, 79.5] |
| ours · conch_probe_lr (evaluated model) | 78.4 [63.2, 88.0] | 81.9 [67.1, 90.7] |
| ours · fusion_top3 (evaluated model) | 84.5 [68.3, 92.6] | 85.8 [68.0, 93.9] |
| ours · hoptimus0_512_lr (evaluated model) | 81.1 [64.0, 90.1] | 81.9 [63.1, 91.7] |
| ours · hoptimus0_lr (evaluated model) | 78.2 [60.1, 87.4] | 78.6 [60.6, 88.9] |
| ours · musk_lr (evaluated model) | 69.4 [55.8, 81.8] | 70.5 [57.5, 86.3] |
| ours · uni2h512_lr (evaluated model) | 86.0 [69.2, 92.7] | 87.1 [69.1, 93.6] |
| ours · uni2h_lr (evaluated model) | 83.0 [60.7, 92.3] | 83.5 [62.7, 93.8] |

- Same split, different protocol: published THUNDER probes resize the whole image; we embed tiles with test-time augmentation and select models by grouped CV inside train.
- Not compared with the 87% ICIAR 2018 challenge result, which used a different, hidden test set.
- Published numbers carry no CI here; ours are cluster-bootstrap 95% CIs in percent.

## Queue simulation (workflow benefit and cost)

Source: `eval/queue_sim.json` · 1000 simulated days · minutes, mean [2.5–97.5 percentile over days].

| Policy | Invasive: mean first look | Invasive: P90 | Unflagged: mean wait | Overtime |
|---|---|---|---|---|
| FIFO (arrival order) | 20.2 [3.5, 60.1] | 37.7 [8.1, 100.8] | 20.0 [4.2, 59.0] | 29.3 [0.0, 104.3] |
| Sepia tiers | 6.0 [1.9, 12.9] | 12.9 [4.8, 31.5] | 41.4 [5.5, 143.0] | 29.3 [0.0, 104.3] |
| Sepia tiers + aging guard (configured targets) | 6.0 [1.9, 12.9] | 12.9 [4.8, 31.5] | 41.4 [5.5, 143.0] | 29.3 [0.0, 104.3] |
| Sensitivity: tiers + aging, day-scale targets | 6.0 [1.9, 12.9] | 12.9 [4.8, 31.5] | 41.3 [5.5, 141.1] | 29.3 [0.0, 104.3] |

Sepia tiers vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 21.4 [0.7, 86.5] min**; false negatives: not available (no simulated day had a truly invasive case outside tiers 1-2).

Sepia tiers + aging guard (configured targets) vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 21.4 [0.7, 86.5] min**; false negatives: not available (no simulated day had a truly invasive case outside tiers 1-2).

Sensitivity: tiers + aging, day-scale targets vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 21.3 [0.7, 84.2] min**; false negatives: not available (no simulated day had a truly invasive case outside tiers 1-2).

Aging guard: share of reads promoted per day — Sepia tiers + aging guard (configured targets) 0.0%; Sensitivity: tiers + aging, day-scale targets 2.6%.

Assumptions: arrivals: Poisson, 6 per hour for 8 h; cases drawn uniformly with replacement from the clean test split; reader: one reader, fixed 9 min per case, non-preemptive, works overtime until empty; utilization: 0.9; tiers: core.tiers (surgical 1; Invasive top or in the aps_marginal_nonempty conformal set 2; set size != 1, reliability flag or insufficient tissue 3; else 4); common_random_numbers: True; seed: 44; interval: 2.5-97.5 percentiles over simulated days (simulation uncertainty only); truth_use: true labels only score the waits; they never drive the queue.

- Prioritizing flagged cases delays non-flagged ones: the vs_fifo rows report that cost (and the delay of false negatives) next to the benefit (cf. Annarumma 2019).
- A simulation with stated assumptions, not a measured workflow outcome.
- At the configured aging targets (tier 1: 20 min, tier 2: 1440 min, tier 3: 1440 min, tier 4: 2880 min) no case waits long enough to be promoted within a 8 h day, so tiers and tiers_aging coincide: the aging guard has NO measured effect here and must not be claimed to protect unflagged cases.
- tiers_aging_day_scale is a sensitivity run with illustrative day-scale aging targets (settings queue_sim.aging_scenarios), not the configured system: 2.6% of reads promoted; unflagged mean wait 41.4 -> 41.3 min vs plain tiers.

## Deferral (risk–coverage) curve

Source: `eval/deferral.json` · pre-registered in docs/VALIDATION_ROUND3.md section 2 · production model `uni2h512_lr`. Each read is scored by the model that produced it in predictions.csv (counts per population below). Reads are deferred to closer review in ranking order; accuracy is four-class accuracy on the kept reads.

- **Headline.** All test reads (clean + simulated degraded), 20% deferred by reliability score (343 of 1716 reads): accuracy on kept reads 89.8% [78.4%, 98.1%] (n 1716, 24 groups), against 85.8% [75.1%, 93.7%] (n 1716, 24 groups) with no deferral and 85.8% for random deferral of 20%.
- Paired change, kept minus all (in addition to the pre-registered headline): 4.0 points [2.0, 7.0].

Each cell: accuracy on kept reads · share of all errors deferred · invasive reads called non-invasive among kept reads · in situ reads called invasive among kept reads (random: means over orders).

### Clean test images (132 reads, 24 groups)

Scoring models: Clean reads: 132 by uni2h512_lr.

| Deferred | Kept | Reliability score | Top-class probability | Conformal set size | Random (mean of orders) |
|---|---|---|---|---|---|
| 0% | 132 | 86.4% · 0.0% · 0 · 3 | 86.4% · 0.0% · 0 · 3 | 86.4% · 0.0% · 0 · 3 | 86.4% · 0.0% · 0.0 · 3.0 |
| 5% | 126 | 86.5% · 5.6% · 0 · 3 | 88.1% · 16.7% · 0 · 3 | 88.1% · 16.7% · 0 · 3 | 86.4% · 4.6% · 0.0 · 2.9 |
| 10% | 119 | 89.1% · 27.8% · 0 · 3 | 88.2% · 22.2% · 0 · 3 | 89.1% · 27.8% · 0 · 3 | 86.3% · 9.8% · 0.0 · 2.7 |
| 15% | 113 | 89.4% · 33.3% · 0 · 3 | 89.4% · 33.3% · 0 · 3 | 89.4% · 33.3% · 0 · 3 | 86.3% · 14.2% · 0.0 · 2.6 |
| 20% | 106 | 90.6% · 44.4% · 0 · 2 | 91.5% · 50.0% · 0 · 1 | 90.6% · 44.4% · 0 · 2 | 86.3% · 19.3% · 0.0 · 2.5 |
| 25% | 99 | 90.9% · 50.0% · 0 · 1 | 90.9% · 50.0% · 0 · 1 | 89.9% · 44.4% · 0 · 2 | 86.3% · 24.7% · 0.0 · 2.3 |
| 30% | 93 | 91.4% · 55.6% · 0 · 1 | 92.5% · 61.1% · 0 · 0 | 90.3% · 50.0% · 0 · 1 | 86.4% · 29.5% · 0.0 · 2.1 |
| 35% | 86 | 91.9% · 61.1% · 0 · 1 | 94.2% · 72.2% · 0 · 0 | 93.0% · 66.7% · 0 · 0 | 86.3% · 34.6% · 0.0 · 2.0 |
| 40% | 80 | 91.2% · 61.1% · 0 · 1 | 93.8% · 72.2% · 0 · 0 | 93.8% · 72.2% · 0 · 0 | 86.3% · 39.1% · 0.0 · 1.9 |
| 45% | 73 | 90.4% · 61.1% · 0 · 1 | 93.2% · 72.2% · 0 · 0 | 93.2% · 72.2% · 0 · 0 | 86.3% · 44.5% · 0.0 · 1.7 |
| 50% | 66 | 89.4% · 61.1% · 0 · 1 | 92.4% · 72.2% · 0 · 0 | 92.4% · 72.2% · 0 · 0 | 86.4% · 50.1% · 0.0 · 1.5 |

### All test reads (clean + simulated degraded) (1716 reads, 24 groups)

Scoring models: Clean reads: 132 by uni2h512_lr. Simulated degraded reads: 1,584 by uni2h512_lr.

| Deferred | Kept | Reliability score | Top-class probability | Conformal set size | Random (mean of orders) |
|---|---|---|---|---|---|
| 0% | 1716 | 85.8% · 0.0% · 3 · 39 | 85.8% · 0.0% · 3 · 39 | 85.8% · 0.0% · 3 · 39 | 85.8% · 0.0% · 3.0 · 39.0 |
| 5% | 1631 | 87.9% · 19.3% · 3 · 33 | 87.5% · 16.4% · 2 · 34 | 88.2% · 20.9% · 3 · 36 | 85.8% · 4.9% · 2.9 · 37.1 |
| 10% | 1545 | 89.5% · 33.6% · 0 · 31 | 88.8% · 29.1% · 2 · 31 | 89.4% · 33.2% · 2 · 31 | 85.8% · 9.9% · 2.7 · 35.1 |
| 15% | 1459 | 89.5% · 37.3% · 0 · 29 | 89.9% · 39.8% · 2 · 29 | 90.0% · 40.2% · 2 · 26 | 85.8% · 14.9% · 2.6 · 33.2 |
| 20% | 1373 | 89.8% · 42.6% · 0 · 26 | 91.3% · 51.2% · 0 · 14 | 91.0% · 49.6% · 0 · 15 | 85.8% · 19.9% · 2.4 · 31.3 |
| 25% | 1287 | 90.4% · 49.2% · 0 · 25 | 91.8% · 57.0% · 0 · 10 | 91.3% · 54.1% · 0 · 12 | 85.8% · 24.9% · 2.2 · 29.3 |
| 30% | 1202 | 91.3% · 57.0% · 0 · 17 | 92.4% · 62.7% · 0 · 5 | 91.9% · 60.2% · 0 · 9 | 85.8% · 29.9% · 2.1 · 27.3 |
| 35% | 1116 | 92.5% · 65.6% · 0 · 8 | 93.3% · 69.3% · 0 · 2 | 93.2% · 68.9% · 0 · 2 | 85.8% · 35.0% · 1.9 · 25.4 |
| 40% | 1030 | 92.2% · 67.2% · 0 · 8 | 93.8% · 73.8% · 0 · 0 | 93.8% · 73.8% · 0 · 0 | 85.8% · 40.0% · 1.8 · 23.3 |
| 45% | 944 | 91.7% · 68.0% · 0 · 6 | 94.1% · 77.0% · 0 · 0 | 94.1% · 77.0% · 0 · 0 | 85.8% · 45.0% · 1.7 · 21.4 |
| 50% | 858 | 91.4% · 69.7% · 0 · 4 | 93.6% · 77.5% · 0 · 0 | 93.6% · 77.5% · 0 · 0 | 85.8% · 50.1% · 1.5 · 19.4 |

- Degraded variants are simulated frozen-section artifacts applied to microscopy images; no real frozen sections were available.
- Deferred reads are sent to closer review; this measures accuracy on the reads kept, not a clinical outcome.

## Simulated margin read

Source: `eval/margin.json` · threshold P = 0.99982 [0.84738, 0.99989] (n 30, 6 groups), from 30 benign calibration images.

- Benign test images, false tumor-at-margin per margin (pooled over 4 margins): 0.0% [0.0%, 0.0%] (n 284, 7 groups) (the threshold targets ≤ 5% per margin; 71 of 71 benign test images scored). Image-level, any of 4 margins: 0.0% [0.0%, 0.0%] (n 71, 7 groups) (no 5% target at image level).
- Carcinoma test images with ≥ 1 margin called clear: 47.5% [25.0%, 64.7%] (n 61, 19 groups); Spearman(max margin score, case risk) 0.56 [0.09, 0.80] (n 61, 19 groups).
- Composite fields (64 of 64 scored): exact-negative margin called clear 100.0% [100.0%, 100.0%] (n 64, 41 groups); benign-half segments below threshold 100.0% [100.0%, 100.0%] (n 768, 41 groups); carcinoma-half segments above threshold 40.0% [30.8%, 50.9%] (n 768, 41 groups); tumor-side margin involved 70.3% [57.7%, 82.4%] (n 64, 41 groups). An echo of the diagnosis would score ~0% clear.
- Annotated WSI windows (84, 10 patients): sensitivity 45.4% [25.4%, 65.2%] (n 141, 10 groups), specificity 98.5% [96.7%, 100.0%] (n 195, 10 groups).

- Simulated margins: image borders stand in for inked margins (top superior, bottom inferior, left medial, right lateral); anterior/posterior are not representable in 2D.
- Raw-score evaluation: the reliability defer rule is a separate safety layer.
- Composite fields are built from test images only and never enter headline metrics.

## TCGA-BRCA external test (production model)

Source: `eval/external_tcga.json` · pre-registered in `docs/EXTERNAL_TCGA_R2.md` before the run · model `uni2h512_lr` (frozen UNI2-h, 1 view per 512-px region, as in its BACH embeddings), BACH temperature 0.342, APS q-hat 0.931 (alpha 0.1) with randomized sets as in the app, unchanged tier rule · no retraining and no threshold changes.

TCGA-BRCA has no benign or in situ labels. Only invasive carcinoma (Primary Tumor slides) and normal tissue (Solid Tissue Normal slides, frozen) are tested, and labels are per slide, not per crop. A slide's score is the maximum calibrated P(Invasive) over its crops (up to 6 random tissue-rich fields); a slide is called invasive when any crop's top class is Invasive. Intervals: 95% bootstrap over patients. The reliability flag and the OOD percentile are not computed (CONCH and MUSK are not on this laptop), so `reliability_flag` is false for every crop.

| Analysis | 25 patients (tile-API crops) | first 8 patients (openslide crops) |
|---|---|---|
| FFPE tumor slides called invasive (sensitivity) | 95.7% [87.0%, 100.0%] (n 23, 23 groups) | 87.5% [62.5%, 100.0%] (n 8, 8 groups) |
| Frozen tumor slides called invasive (sensitivity) | 84.0% [68.0%, 96.0%] (n 25, 25 groups) | 100.0% [100.0%, 100.0%] (n 8, 8 groups) |
| Frozen normal slides not called invasive (specificity) | 100.0% [100.0%, 100.0%] (n 17, 17 groups) | 100.0% [100.0%, 100.0%] (n 7, 7 groups) |
| Frozen tumor vs frozen normal, AUROC of the slide score | 1.000 [1.000, 1.000] (n 42, 25 groups) | 1.000 [1.000, 1.000] (n 15, 8 groups) |
| Tumor slides with a crop in tier 1–2 | 97.9% [93.6%, 100.0%] (n 48, 25 groups) | 93.8% [81.2%, 100.0%] (n 16, 8 groups) |
| FFPE minus frozen tumor sensitivity | 11.7 points | -12.5 points |
| Patients / slides / crops | 25 / 65 / 353 | 8 / 23 / 132 |

25 patients (tile-API crops): no tissue-rich field: TCGA-BH-A0B8 frozen_normal, TCGA-AC-A3EH frozen_normal, TCGA-AC-A2FK frozen_normal, TCGA-E2-A15K frozen_normal, TCGA-E9-A1N9 frozen_normal, TCGA-E2-A1IO frozen_normal, TCGA-E9-A1NH frozen_normal, TCGA-BH-A204 frozen_normal; fewer than 6 crops: TCGA-A7-A4SD frozen_normal (3), TCGA-A7-A4SD frozen_tumor (1), TCGA-AC-A2FM frozen_normal (1), TCGA-BH-A0H3 frozen_normal (2), TCGA-BH-A0HK frozen_normal (2), TCGA-BH-A18N frozen_normal (5), TCGA-BH-A18T frozen_normal (1), TCGA-BH-A1EO frozen_normal (1), TCGA-BH-A1FR frozen_normal (1).

First 8 patients (openslide crops): no tissue-rich field: TCGA-BH-A0B8 frozen_normal; fewer than 6 crops: TCGA-BH-A0H3 frozen_normal (1), TCGA-BH-A18N frozen_normal (5).

### BACH test vs TCGA (25 patients)

BACH: 132 held-out test images (unit: single BACH image (2048 x 1536 field), not a slide); TCGA: slides, with frozen normal slides as the normal comparator. The units differ, so the comparison is indicative.

| Statistic | BACH test | TCGA |
|---|---|---|
| Invasive / tumor called invasive | 100.0% [100.0%, 100.0%] (n 31, 17 groups) | FFPE 95.7% [87.0%, 100.0%] (n 23, 23 groups); frozen 84.0% [68.0%, 96.0%] (n 25, 25 groups) |
| Normal not called invasive | 100.0% [100.0%, 100.0%] (n 23, 6 groups) | 100.0% [100.0%, 100.0%] (n 17, 17 groups) |
| Invasive vs normal AUROC | 1.000 [1.000, 1.000] (n 54, 22 groups) | 1.000 [1.000, 1.000] (n 42, 25 groups) |
| Invasive / tumor in tier 1–2 | 100.0% [100.0%, 100.0%] (n 31, 17 groups) | 97.9% [93.6%, 100.0%] (n 48, 25 groups) |
| Reliability flag rate | 16.7% (clean test) | not computed |

### Pre-registered sensitivity analysis S1: slides with exactly 6 crops

| Analysis | 25 patients (tile-API crops) | first 8 patients (openslide crops) |
|---|---|---|
| FFPE tumor slides called invasive (sensitivity) | 95.7% [87.0%, 100.0%] (n 23, 23 groups) | 87.5% [62.5%, 100.0%] (n 8, 8 groups) |
| Frozen tumor slides called invasive (sensitivity) | 83.3% [66.7%, 95.8%] (n 24, 24 groups) | 100.0% [100.0%, 100.0%] (n 8, 8 groups) |
| Frozen normal slides not called invasive (specificity) | 100.0% [100.0%, 100.0%] (n 9, 9 groups) | 100.0% [100.0%, 100.0%] (n 5, 5 groups) |
| Frozen tumor vs frozen normal, AUROC of the slide score | 1.000 [1.000, 1.000] (n 33, 24 groups) | 1.000 [1.000, 1.000] (n 13, 8 groups) |
| Tumor slides with a crop in tier 1–2 | 97.9% [93.5%, 100.0%] (n 47, 25 groups) | 93.8% [81.2%, 100.0%] (n 16, 8 groups) |
| Slides analysed | 56 | 21 |

### Checks behind the run

- Parity gate (original BACH TIFF (Zenodo record 3632035)): 5 BACH test images embedded on the laptop match the stored Delta embeddings with region cosine ≥ 0.99968 (mean 0.99996); calibrated probabilities differ from `predictions.csv` by at most 0.0009; pass = True.
- Tile source: re-extracting 23 openslide crops from their manifest coordinates through the GDC tile API gave a mean absolute pixel difference of at most 0.0 (0–255 units). The tile path picked the same fields of view on 13 of 24 slides; decision: extract all 25 patients through tiles (pre-registered branch: FOV picks were not identical on every earlier openslide slide).
- Why picks can differ (one slide downloaded and checked): openslide's get_thumbnail reads the slide's own pyramid level 2 (downsample 16.004). GDC renders DeepZoom level 'downsample 16' from slide level 1 (openslide get_best_level_for_downsample(16) = 1 because 16 < 16.004) and LANCZOS-downsamples it, so the two thumbnails differ slightly; tissue fractions near the 0.75 cut can cross it, which changes the candidate list and therefore the seeded random picks. Full-resolution pixels are identical (equivalence MAD 0.0 on 23 crops).
- Latency per 2048 × 1536 crop (Apple M3 Max; macOS 27.0.1; torch 2.14.1; device mps; bf16 autocast): median 0.73 s (IQR 0.69–0.82 s, n 351, first 2 excluded); embedding alone median 0.63 s. Single fields, not whole slides.

Limitations of this test:

- Labels are slide-level sample types (Primary Tumor, Solid Tissue Normal); a tumor-slide crop may show only stroma or fat, and there are no tumor outlines.
- TCGA-BRCA has no benign or in situ labels: four-class accuracy, in situ vs invasive and the p63/calponin rule are not tested; specificity is measured only against frozen normal tissue adjacent to tumor.
- The reliability flag and the OOD percentile are not computed (CONCH and MUSK are not on this laptop); reliability_flag is forced false, which can move a crop from tier 3 to tier 4 but never changes tier 1-2 placement.
- The sample is small (at most 25 patients); bootstrap intervals over patients are wide.
- protocol25 crops come from the GDC DeepZoom tile API rather than local openslide decoding; re-extracting 23 openslide crops through the API reproduced them pixel for pixel, but the API's thumbnails can shift which tissue-rich fields are sampled (tile_equivalence).
- BACH reference values are per image; TCGA values are per slide (maximum over up to 6 fields), so the BACH-vs-TCGA comparison is indicative, not a matched test.
- Latency is measured on one laptop (Apple MPS) for single 2048 x 1536 fields, not whole slides.
- Research prototype; not clinically validated.
- Credit: TCGA Research Network (https://www.cancer.gov/tcga), via the NCI Genomic Data Commons.

## Other evaluations (concepts, reliability, retrieval, deletion, cells)

### `eval/model_choice.json`

- cv_metrics · conch_probe_lr · log_loss: 0.31 [0.23, 0.40] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · auroc_invasive: 0.993 [0.981, 1.000] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · balanced_accuracy: 0.90 [0.86, 0.94] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · macro_f1: 0.90 [0.85, 0.94] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · macro_auroc: 0.986 [0.972, 0.995] (n 218, 49 groups)
- cv_metrics · uni2h_lr · log_loss: 0.26 [0.20, 0.33] (n 218, 49 groups)
- cv_metrics · uni2h_lr · auroc_invasive: 0.999 [0.998, 1.000] (n 218, 49 groups)
- cv_metrics · uni2h_lr · balanced_accuracy: 0.949 [0.916, 0.981] (n 218, 49 groups)
- cv_metrics · uni2h_lr · macro_f1: 0.951 [0.915, 0.980] (n 218, 49 groups)
- cv_metrics · uni2h_lr · macro_auroc: 0.993 [0.984, 0.999] (n 218, 49 groups)
- cv_metrics · uni2h512_lr · log_loss: 0.36 [0.30, 0.43] (n 218, 49 groups)
- cv_metrics · uni2h512_lr · auroc_invasive: 1.000 [1.000, 1.000] (n 218, 49 groups)
- cv_metrics · uni2h512_lr · balanced_accuracy: 0.953 [0.922, 0.982] (n 218, 49 groups)
- cv_metrics · uni2h512_lr · macro_f1: 0.955 [0.920, 0.982] (n 218, 49 groups)
- cv_metrics · uni2h512_lr · macro_auroc: 0.993 [0.983, 0.999] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · log_loss: 0.32 [0.25, 0.40] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · auroc_invasive: 0.997 [0.991, 1.000] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · balanced_accuracy: 0.910 [0.867, 0.952] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · macro_f1: 0.913 [0.865, 0.952] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · macro_auroc: 0.990 [0.978, 0.998] (n 218, 49 groups)
- cv_metrics · hoptimus0_512_lr · log_loss: 0.43 [0.37, 0.49] (n 218, 49 groups)
- cv_metrics · hoptimus0_512_lr · auroc_invasive: 0.998 [0.995, 1.000] (n 218, 49 groups)
- cv_metrics · hoptimus0_512_lr · balanced_accuracy: 0.928 [0.887, 0.966] (n 218, 49 groups)
- cv_metrics · hoptimus0_512_lr · macro_f1: 0.929 [0.884, 0.965] (n 218, 49 groups)
- cv_metrics · hoptimus0_512_lr · macro_auroc: 0.991 [0.979, 0.998] (n 218, 49 groups)
- cv_metrics · musk_lr · log_loss: 0.33 [0.24, 0.44] (n 218, 49 groups)
- cv_metrics · musk_lr · auroc_invasive: 0.983 [0.953, 0.998] (n 218, 49 groups)
- cv_metrics · musk_lr · balanced_accuracy: 0.89 [0.84, 0.94] (n 218, 49 groups)
- cv_metrics · musk_lr · macro_f1: 0.89 [0.84, 0.93] (n 218, 49 groups)
- cv_metrics · musk_lr · macro_auroc: 0.981 [0.965, 0.993] (n 218, 49 groups)
- cv_metrics · cb_text · log_loss: 0.48 [0.38, 0.59] (n 218, 49 groups)
- cv_metrics · cb_text · auroc_invasive: 0.990 [0.975, 0.999] (n 218, 49 groups)
- cv_metrics · cb_text · balanced_accuracy: 0.85 [0.80, 0.90] (n 218, 49 groups)
- cv_metrics · cb_text · macro_f1: 0.85 [0.79, 0.90] (n 218, 49 groups)
- cv_metrics · cb_text · macro_auroc: 0.956 [0.933, 0.975] (n 218, 49 groups)
- cv_metrics · cb_text_cells · log_loss: 0.73 [0.64, 0.82] (n 218, 49 groups)
- cv_metrics · cb_text_cells · auroc_invasive: 0.988 [0.973, 0.998] (n 218, 49 groups)
- cv_metrics · cb_text_cells · balanced_accuracy: 0.76 [0.71, 0.82] (n 218, 49 groups)
- cv_metrics · cb_text_cells · macro_f1: 0.76 [0.69, 0.82] (n 218, 49 groups)
- cv_metrics · cb_text_cells · macro_auroc: 0.933 [0.905, 0.957] (n 218, 49 groups)
- cv_metrics · cb_cells · log_loss: 1.32 [1.27, 1.38] (n 218, 49 groups)
- cv_metrics · cb_cells · auroc_invasive: 0.68 [0.58, 0.78] (n 218, 49 groups)
- cv_metrics · cb_cells · balanced_accuracy: 0.37 [0.31, 0.44] (n 218, 49 groups)
- cv_metrics · cb_cells · macro_f1: 0.33 [0.26, 0.39] (n 218, 49 groups)
- cv_metrics · cb_cells · macro_auroc: 0.62 [0.57, 0.67] (n 218, 49 groups)
- cv_metrics · abmil · log_loss: 0.18 [0.07, 0.32] (n 218, 49 groups)
- cv_metrics · abmil · auroc_invasive: 1.000 [1.000, 1.000] (n 218, 49 groups)
- cv_metrics · abmil · balanced_accuracy: 0.946 [0.906, 0.981] (n 218, 49 groups)
- cv_metrics · abmil · macro_f1: 0.946 [0.901, 0.980] (n 218, 49 groups)
- cv_metrics · abmil · macro_auroc: 0.992 [0.983, 0.999] (n 218, 49 groups)
- cv_metrics · fusion_top3 · log_loss: 0.19 [0.11, 0.31] (n 218, 49 groups)
- cv_metrics · fusion_top3 · auroc_invasive: 0.999 [0.997, 1.000] (n 218, 49 groups)
- cv_metrics · fusion_top3 · balanced_accuracy: 0.947 [0.910, 0.980] (n 218, 49 groups)
- cv_metrics · fusion_top3 · macro_f1: 0.948 [0.907, 0.979] (n 218, 49 groups)
- cv_metrics · fusion_top3 · macro_auroc: 0.991 [0.979, 0.999] (n 218, 49 groups)
- ablation_concepts_cells · cb_text · log_loss: 0.48 [0.38, 0.59] (n 218, 49 groups)
- ablation_concepts_cells · cb_text · auroc_invasive: 0.990 [0.975, 0.999] (n 218, 49 groups)
- ablation_concepts_cells · cb_text · balanced_accuracy: 0.85 [0.80, 0.90] (n 218, 49 groups)
- ablation_concepts_cells · cb_text · macro_f1: 0.85 [0.79, 0.90] (n 218, 49 groups)
- ablation_concepts_cells · cb_text · macro_auroc: 0.956 [0.933, 0.975] (n 218, 49 groups)
- ablation_concepts_cells · cb_text_cells · log_loss: 0.73 [0.64, 0.82] (n 218, 49 groups)
- ablation_concepts_cells · cb_text_cells · auroc_invasive: 0.988 [0.973, 0.998] (n 218, 49 groups)
- ablation_concepts_cells · cb_text_cells · balanced_accuracy: 0.76 [0.71, 0.82] (n 218, 49 groups)
- ablation_concepts_cells · cb_text_cells · macro_f1: 0.76 [0.69, 0.82] (n 218, 49 groups)
- ablation_concepts_cells · cb_text_cells · macro_auroc: 0.933 [0.905, 0.957] (n 218, 49 groups)
- ablation_concepts_cells · cb_cells · log_loss: 1.32 [1.27, 1.38] (n 218, 49 groups)
- ablation_concepts_cells · cb_cells · auroc_invasive: 0.68 [0.58, 0.78] (n 218, 49 groups)
- ablation_concepts_cells · cb_cells · balanced_accuracy: 0.37 [0.31, 0.44] (n 218, 49 groups)
- ablation_concepts_cells · cb_cells · macro_f1: 0.33 [0.26, 0.39] (n 218, 49 groups)
- ablation_concepts_cells · cb_cells · macro_auroc: 0.62 [0.57, 0.67] (n 218, 49 groups)

### `eval/concepts.json`

- cav · concepts · tubules · heldout_auroc: 0.73 [0.63, 0.82] (n 380, 163 groups)
- cav · concepts · pleomorphism · heldout_auroc: 0.75 [0.69, 0.81] (n 606, 273 groups)
- cav · concepts · mitoses · heldout_auroc: 0.78 [0.72, 0.83] (n 699, 313 groups)
- cav · concepts · invasion · heldout_auroc: 0.78 [0.74, 0.82] (n 810, 420 groups)
- cav · concepts · invasion · wsi_check · text: 0.89 [0.85, 0.94] (n 1950, 10 groups)
- cav · concepts · invasion · wsi_check · cav: 0.87 [0.83, 0.92] (n 1950, 10 groups)
- cav · concepts · duct_confined · heldout_auroc: 0.79 [0.75, 0.83] (n 806, 404 groups)
- cav · concepts · duct_confined · wsi_check · text: 0.84 [0.75, 0.92] (n 1950, 10 groups)
- cav · concepts · duct_confined · wsi_check · cav: 0.90 [0.84, 0.94] (n 1950, 10 groups)
- cav · concepts · normal_lobules · heldout_auroc: 0.77 [0.67, 0.85] (n 527, 245 groups)
- cav · concepts · benign_stroma_fat · heldout_auroc: 0.84 [0.79, 0.89] (n 580, 304 groups)
- cav · concepts · lymphocytes · heldout_auroc: 0.79 [0.73, 0.85] (n 457, 221 groups)
- cav · concepts · necrosis · heldout_auroc: 0.87 [0.81, 0.91] (n 640, 252 groups)
- nct · concepts · lymphocytes · text: 0.998 [0.996, 0.999] (n 2700, 2700 groups)
- nct · concepts · lymphocytes · cav: 0.32 [0.29, 0.35] (n 2700, 2700 groups)
- nct · concepts · necrosis · text: 0.939 [0.927, 0.950] (n 2700, 2700 groups)
- nct · concepts · necrosis · cav: 0.952 [0.940, 0.963] (n 2700, 2700 groups)
- nct · concepts · pleomorphism · text: 0.59 [0.56, 0.62] (n 2700, 2700 groups)
- nct · concepts · pleomorphism · cav: 0.61 [0.58, 0.65] (n 2700, 2700 groups)
- nct · concepts · benign_stroma_fat · text: 0.87 [0.85, 0.88] (n 2700, 2700 groups)
- nct · concepts · benign_stroma_fat · cav: 0.80 [0.78, 0.82] (n 2700, 2700 groups)

### `eval/concepts_bach.json`

- concepts · tubules · class_means · Normal: 1.07 [0.85, 1.32] (n 23, 6 groups)
- concepts · tubules · class_means · Benign: 1.141 [0.978, 1.358] (n 48, 6 groups)
- concepts · tubules · class_means · InSitu: 0.44 [0.19, 0.85] (n 30, 6 groups)
- concepts · tubules · class_means · Invasive: -0.52 [-0.90, -0.12] (n 31, 17 groups)
- concepts · tubules · auroc_Invasive_vs_rest: 0.08 [0.01, 0.20] (n 132, 24 groups)
- concepts · pleomorphism · class_means · Normal: 0.33 [0.22, 0.46] (n 23, 6 groups)
- concepts · pleomorphism · class_means · Benign: -0.02 [-0.22, 0.29] (n 48, 6 groups)
- concepts · pleomorphism · class_means · InSitu: 0.29 [-0.18, 1.16] (n 30, 6 groups)
- concepts · pleomorphism · class_means · Invasive: 1.46 [1.20, 1.72] (n 31, 17 groups)
- concepts · pleomorphism · auroc_Invasive_vs_rest: 0.949 [0.903, 0.983] (n 132, 24 groups)
- concepts · mitoses · class_means · Normal: 0.21 [-0.04, 0.41] (n 23, 6 groups)
- concepts · mitoses · class_means · Benign: 0.11 [-0.25, 0.40] (n 48, 6 groups)
- concepts · mitoses · class_means · InSitu: 0.89 [0.51, 1.32] (n 30, 6 groups)
- concepts · mitoses · class_means · Invasive: 1.32 [1.05, 1.56] (n 31, 17 groups)
- concepts · mitoses · auroc_Invasive_vs_rest: 0.851 [0.752, 0.960] (n 132, 24 groups)
- concepts · invasion · class_means · Normal: 0.43 [0.19, 0.66] (n 23, 6 groups)
- concepts · invasion · class_means · Benign: 0.20 [-0.14, 0.59] (n 48, 6 groups)
- concepts · invasion · class_means · InSitu: 0.41 [0.29, 0.59] (n 30, 6 groups)
- concepts · invasion · class_means · Invasive: 1.31 [1.14, 1.48] (n 31, 17 groups)
- concepts · invasion · auroc_Invasive_vs_rest: 0.964 [0.903, 0.993] (n 132, 24 groups)
- concepts · duct_confined · class_means · Normal: -0.17 [-0.34, 0.02] (n 23, 6 groups)
- concepts · duct_confined · class_means · Benign: 0.33 [-0.33, 0.86] (n 48, 6 groups)
- concepts · duct_confined · class_means · InSitu: 1.08 [0.90, 1.27] (n 30, 6 groups)
- concepts · duct_confined · class_means · Invasive: 0.10 [-0.08, 0.32] (n 31, 17 groups)
- concepts · duct_confined · auroc_Invasive_vs_rest: 0.37 [0.14, 0.68] (n 132, 24 groups)
- concepts · normal_lobules · class_means · Normal: 1.47 [1.30, 1.59] (n 23, 6 groups)
- concepts · normal_lobules · class_means · Benign: 0.87 [0.52, 1.33] (n 48, 6 groups)
- concepts · normal_lobules · class_means · InSitu: 0.61 [0.52, 0.77] (n 30, 6 groups)
- concepts · normal_lobules · class_means · Invasive: 0.09 [-0.07, 0.22] (n 31, 17 groups)
- concepts · normal_lobules · auroc_Invasive_vs_rest: 0.14 [0.02, 0.23] (n 132, 24 groups)
- concepts · benign_stroma_fat · class_means · Normal: 1.24 [1.07, 1.34] (n 23, 6 groups)
- concepts · benign_stroma_fat · class_means · Benign: 0.84 [0.61, 1.20] (n 48, 6 groups)
- concepts · benign_stroma_fat · class_means · InSitu: 0.59 [0.49, 0.78] (n 30, 6 groups)
- concepts · benign_stroma_fat · class_means · Invasive: -0.08 [-0.37, 0.19] (n 31, 17 groups)
- concepts · benign_stroma_fat · auroc_Invasive_vs_rest: 0.15 [0.03, 0.26] (n 132, 24 groups)
- concepts · lymphocytes · class_means · Normal: 0.674 [0.263, 0.998] (n 23, 6 groups)
- concepts · lymphocytes · class_means · Benign: 0.62 [0.48, 0.93] (n 48, 6 groups)
- concepts · lymphocytes · class_means · InSitu: 0.70 [0.44, 1.37] (n 30, 6 groups)
- concepts · lymphocytes · class_means · Invasive: 0.54 [0.29, 0.82] (n 31, 17 groups)
- concepts · lymphocytes · auroc_Invasive_vs_rest: 0.44 [0.27, 0.62] (n 132, 24 groups)
- concepts · necrosis · class_means · Normal: -0.33 [-0.53, -0.10] (n 23, 6 groups)
- concepts · necrosis · class_means · Benign: 0.37 [0.05, 0.62] (n 48, 6 groups)
- concepts · necrosis · class_means · InSitu: 1.40 [0.86, 2.07] (n 30, 6 groups)
- concepts · necrosis · class_means · Invasive: 1.25 [1.12, 1.40] (n 31, 17 groups)
- concepts · necrosis · auroc_Invasive_vs_rest: 0.787 [0.622, 0.960] (n 132, 24 groups)

### `eval/zeroshot.json`

- encoders · conch · macro_f1: 0.46 [0.35, 0.55] (n 132, 24 groups)
- encoders · conch · balanced_accuracy: 0.53 [0.46, 0.59] (n 132, 24 groups)
- encoders · conch · accuracy: 0.62 [0.44, 0.76] (n 132, 24 groups)
- encoders · musk · macro_f1: 0.48 [0.38, 0.61] (n 132, 24 groups)
- encoders · musk · balanced_accuracy: 0.54 [0.50, 0.67] (n 132, 24 groups)
- encoders · musk · accuracy: 0.63 [0.46, 0.77] (n 132, 24 groups)
- encoders · pathgenclip · macro_f1: 0.54 [0.44, 0.67] (n 132, 24 groups)
- encoders · pathgenclip · balanced_accuracy: 0.56 [0.51, 0.70] (n 132, 24 groups)
- encoders · pathgenclip · accuracy: 0.64 [0.49, 0.76] (n 132, 24 groups)

### `eval/reliability.json`

- accuracy_vs_degradation · curve · blur · 0: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 1: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 2: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 3: 0.87 [0.78, 0.95] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 0: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 1: 0.86 [0.74, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 2: 0.86 [0.74, 0.95] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 3: 0.841 [0.714, 0.953] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 0: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 1: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 2: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 3: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 0: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 1: 0.86 [0.75, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 2: 0.87 [0.78, 0.95] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 3: 0.82 [0.73, 0.92] (n 132, 24 groups)
- error_auroc_test: 0.69 [0.58, 0.87] (n 1716, 24 groups)
- feature_auroc_test · ood_percentile: 0.46 [0.38, 0.59] (n 1716, 24 groups)
- feature_auroc_test · set_size: 0.69 [0.60, 0.85] (n 1716, 24 groups)
- feature_auroc_test · conch_musk_disagree: 0.49 [0.41, 0.68] (n 1716, 24 groups)
- error_recall_test: 0.42 [0.28, 0.70] (n 1716, 24 groups)

### `eval/retrieval.json`

- precision_at_k_test: 0.67 [0.58, 0.83] (n 132, 24 groups)

### `eval/deletion.json`

Source: `eval/deletion.json` · held-out test images · 20 random 3-region draws per image · drops are full read minus read with the regions removed and re-pooled; logit = centered calibrated logit (logit_target - mean over classes, divided by the temperature).

Exact contributions (`eval/tile_contrib.json`): constant plus region contributions reproduce each read's centered logit with max error 5.1e-08 over 2430 reads (tolerance 1e-06).

`uni2h512_lr` · predicted aggressive (n 34, 17 groups)

| Removed | Drop in P | Drop in logit | Minus random (P) | Minus random (logit) |
|---|---|---|---|---|
| 3 random regions | -0.001 [-0.004, 0.000] | -0.01 [-0.03, 0.01] |  |  |
| Top 3 tile-probe regions | 0.024 [0.002, 0.049] | 0.21 [0.13, 0.31] | 0.025 [0.001, 0.052] | 0.22 [0.13, 0.33] |
| Top 3 regions of the top concept | 0.002 [-0.016, 0.017] | -0.03 [-0.11, 0.03] | 0.003 [-0.012, 0.018] | -0.02 [-0.10, 0.04] |
| Top 3 regions by exact contribution | 0.032 [0.004, 0.060] | 0.45 [0.38, 0.54] | 0.034 [0.003, 0.063] | 0.46 [0.38, 0.56] |

`uni2h512_lr` · all eligible (n 132, 24 groups)

| Removed | Drop in P | Drop in logit | Minus random (P) | Minus random (logit) |
|---|---|---|---|---|
| 3 random regions | -0.001 [-0.002, 0.000] | 0.00 [-0.01, 0.01] |  |  |
| Top 3 tile-probe regions | 0.012 [0.001, 0.021] | 0.26 [0.18, 0.34] | 0.012 [0.002, 0.022] | 0.26 [0.18, 0.35] |
| Top 3 regions of the top concept | 0.001 [-0.006, 0.007] | -0.03 [-0.08, 0.00] | 0.002 [-0.005, 0.007] | -0.03 [-0.08, 0.00] |
| Top 3 regions by exact contribution | 0.017 [0.004, 0.028] | 0.54 [0.47, 0.60] | 0.017 [0.004, 0.029] | 0.54 [0.47, 0.60] |

### `eval/cells.json`

- direction_checks · features · neo_fraction · class_means · Normal: 0.17 [0.12, 0.23] (n 17, 6 groups)
- direction_checks · features · neo_fraction · class_means · Benign: 0.46 [0.15, 0.68] (n 40, 6 groups)
- direction_checks · features · neo_fraction · class_means · InSitu: 0.74 [0.66, 0.79] (n 29, 5 groups)
- direction_checks · features · neo_fraction · class_means · Invasive: 0.44 [0.34, 0.53] (n 28, 16 groups)
- direction_checks · features · neo_fraction · difference: 0.06 [-0.15, 0.29] (n 85, 21 groups)
- direction_checks · features · neo_density · class_means · Normal: 417.58 [154.77, 825.46] (n 17, 6 groups)
- direction_checks · features · neo_density · class_means · Benign: 1386.94 [252.77, 2174.13] (n 40, 6 groups)
- direction_checks · features · neo_density · class_means · InSitu: 2998.02 [2442.88, 3647.73] (n 29, 5 groups)
- direction_checks · features · neo_density · class_means · Invasive: 1235.00 [951.81, 1596.94] (n 28, 16 groups)
- direction_checks · features · neo_density · difference: 137.16 [-733.54, 1158.05] (n 85, 21 groups)
- direction_checks · features · nuc_area_mean · class_means · Normal: 49.57 [45.12, 54.09] (n 8, 6 groups)
- direction_checks · features · nuc_area_mean · class_means · Benign: 53.27 [49.52, 57.90] (n 28, 6 groups)
- direction_checks · features · nuc_area_mean · class_means · InSitu: 48.63 [43.38, 56.41] (n 29, 5 groups)
- direction_checks · features · nuc_area_mean · class_means · Invasive: 54.73 [48.50, 61.09] (n 27, 16 groups)
- direction_checks · features · nuc_area_mean · difference: 2.29 [-4.54, 9.56] (n 63, 21 groups)
- direction_checks · features · nuc_area_cv · class_means · Normal: 0.50 [0.45, 0.55] (n 8, 6 groups)
- direction_checks · features · nuc_area_cv · class_means · Benign: 0.50 [0.42, 0.52] (n 28, 6 groups)
- direction_checks · features · nuc_area_cv · class_means · InSitu: 0.56 [0.48, 0.64] (n 29, 5 groups)
- direction_checks · features · nuc_area_cv · class_means · Invasive: 0.61 [0.57, 0.64] (n 27, 16 groups)
- direction_checks · features · nuc_area_cv · difference: 0.10 [0.06, 0.17] (n 63, 21 groups)
- direction_checks · features · nuc_shape_cv · class_means · Normal: 0.13 [0.13, 0.14] (n 8, 6 groups)
- direction_checks · features · nuc_shape_cv · class_means · Benign: 0.14 [0.14, 0.15] (n 28, 6 groups)
- direction_checks · features · nuc_shape_cv · class_means · InSitu: 0.14 [0.13, 0.15] (n 29, 5 groups)
- direction_checks · features · nuc_shape_cv · class_means · Invasive: 0.13 [0.13, 0.14] (n 27, 16 groups)
- direction_checks · features · nuc_shape_cv · difference: -0.01 [-0.01, 0.00] (n 63, 21 groups)
- direction_checks · features · infl_fraction · class_means · Normal: 0.05 [0.03, 0.06] (n 17, 6 groups)
- direction_checks · features · infl_fraction · class_means · Benign: 0.05 [0.01, 0.11] (n 40, 6 groups)
- direction_checks · features · infl_fraction · class_means · InSitu: 0.02 [0.01, 0.03] (n 29, 5 groups)
- direction_checks · features · infl_fraction · class_means · Invasive: 0.01 [0.01, 0.02] (n 28, 16 groups)
- direction_checks · features · infl_fraction · difference: -0.04 [-0.08, 0.00] (n 85, 21 groups)
- direction_checks · features · til_proximity · class_means · Normal: 0.47 [0.37, 0.59] (n 16, 6 groups)
- direction_checks · features · til_proximity · class_means · Benign: 0.59 [0.35, 0.77] (n 34, 6 groups)
- direction_checks · features · til_proximity · class_means · InSitu: 0.73 [0.67, 0.91] (n 23, 5 groups)
- direction_checks · features · til_proximity · class_means · Invasive: 0.73 [0.54, 0.88] (n 22, 15 groups)
- direction_checks · features · til_proximity · difference: 0.18 [-0.05, 0.42] (n 72, 20 groups)
- direction_checks · features · dead_fraction · class_means · Normal: 0.02 [0.01, 0.04] (n 17, 6 groups)
- direction_checks · features · dead_fraction · class_means · Benign: 0.02 [0.01, 0.04] (n 40, 6 groups)
- direction_checks · features · dead_fraction · class_means · InSitu: 0.01 [0.00, 0.02] (n 29, 5 groups)
- direction_checks · features · dead_fraction · class_means · Invasive: 0.01 [0.00, 0.01] (n 28, 16 groups)
- direction_checks · features · dead_fraction · difference: -0.01 [-0.03, 0.00] (n 85, 21 groups)
- direction_checks · features · epi_fraction · class_means · Normal: 0.23 [0.10, 0.43] (n 17, 6 groups)
- direction_checks · features · epi_fraction · class_means · Benign: 0.16 [0.11, 0.35] (n 40, 6 groups)
- direction_checks · features · epi_fraction · class_means · InSitu: 0.13 [0.03, 0.30] (n 29, 5 groups)
- direction_checks · features · epi_fraction · class_means · Invasive: 0.45 [0.32, 0.56] (n 28, 16 groups)
- direction_checks · features · epi_fraction · difference: 0.27 [0.09, 0.40] (n 85, 21 groups)
- convergent_validity · pairs · nuc_area_cv~pleomorphism · tile: 0.34 [0.20, 0.49] (n 298, 24 groups)
- convergent_validity · pairs · nuc_area_cv~pleomorphism · case: 0.45 [0.25, 0.67] (n 92, 23 groups)
- convergent_validity · pairs · nuc_shape_cv~pleomorphism · tile: -0.04 [-0.22, 0.10] (n 298, 24 groups)
- convergent_validity · pairs · nuc_shape_cv~pleomorphism · case: -0.06 [-0.38, 0.22] (n 92, 23 groups)
- convergent_validity · pairs · infl_fraction~lymphocytes · tile: 0.10 [-0.03, 0.34] (n 382, 24 groups)
- convergent_validity · pairs · infl_fraction~lymphocytes · case: 0.05 [-0.11, 0.37] (n 114, 23 groups)
- ablation · rows[12] · log_loss: 0.48 [0.38, 0.59] (n 218, 49 groups)
- ablation · rows[12] · auroc_invasive: 0.990 [0.975, 0.999] (n 218, 49 groups)
- ablation · rows[12] · balanced_accuracy: 0.85 [0.80, 0.90] (n 218, 49 groups)
- ablation · rows[12] · macro_f1: 0.85 [0.79, 0.90] (n 218, 49 groups)
- ablation · rows[12] · macro_auroc: 0.956 [0.933, 0.975] (n 218, 49 groups)
- ablation · rows[13] · log_loss: 0.73 [0.64, 0.82] (n 218, 49 groups)
- ablation · rows[13] · auroc_invasive: 0.988 [0.973, 0.998] (n 218, 49 groups)
- ablation · rows[13] · balanced_accuracy: 0.76 [0.71, 0.82] (n 218, 49 groups)
- ablation · rows[13] · macro_f1: 0.76 [0.69, 0.82] (n 218, 49 groups)
- ablation · rows[13] · macro_auroc: 0.933 [0.905, 0.957] (n 218, 49 groups)
- ablation · rows[14] · log_loss: 1.32 [1.27, 1.38] (n 218, 49 groups)
- ablation · rows[14] · auroc_invasive: 0.68 [0.58, 0.78] (n 218, 49 groups)
- ablation · rows[14] · balanced_accuracy: 0.37 [0.31, 0.44] (n 218, 49 groups)
- ablation · rows[14] · macro_f1: 0.33 [0.26, 0.39] (n 218, 49 groups)
- ablation · rows[14] · macro_auroc: 0.62 [0.57, 0.67] (n 218, 49 groups)
- ablation · rows[19] · log_loss: 0.48 [0.38, 0.59] (n 218, 49 groups)
- ablation · rows[19] · auroc_invasive: 0.990 [0.975, 0.999] (n 218, 49 groups)
- ablation · rows[19] · balanced_accuracy: 0.85 [0.80, 0.90] (n 218, 49 groups)
- ablation · rows[19] · macro_f1: 0.85 [0.79, 0.90] (n 218, 49 groups)
- ablation · rows[19] · macro_auroc: 0.956 [0.933, 0.975] (n 218, 49 groups)
- ablation · rows[20] · log_loss: 0.73 [0.64, 0.82] (n 218, 49 groups)
- ablation · rows[20] · auroc_invasive: 0.988 [0.973, 0.998] (n 218, 49 groups)
- ablation · rows[20] · balanced_accuracy: 0.76 [0.71, 0.82] (n 218, 49 groups)
- ablation · rows[20] · macro_f1: 0.76 [0.69, 0.82] (n 218, 49 groups)
- ablation · rows[20] · macro_auroc: 0.933 [0.905, 0.957] (n 218, 49 groups)
- ablation · rows[21] · log_loss: 1.32 [1.27, 1.38] (n 218, 49 groups)
- ablation · rows[21] · auroc_invasive: 0.68 [0.58, 0.78] (n 218, 49 groups)
- ablation · rows[21] · balanced_accuracy: 0.37 [0.31, 0.44] (n 218, 49 groups)
- ablation · rows[21] · macro_f1: 0.33 [0.26, 0.39] (n 218, 49 groups)
- ablation · rows[21] · macro_auroc: 0.62 [0.57, 0.67] (n 218, 49 groups)

## Reviewed checks

- Concept top-tile and cell-overlay spot check: **AI-reviewed, not a pathologist** (reviews by Claude Opus 5.5 (AI, not a pathologist) × 200; Codex UI verification (AI, no clinical assessment) × 1) · `eval/spotcheck.json`. Plausibility of the top-scoring tiles per concept (acceptable / reviewed; the rest unsure or not acceptable):

| Feature | Acceptable |
|---|---|
| benign_stroma_fat | 20 / 20 |
| invasion | 19 / 20 |
| tubules | 18 / 20 |
| pleomorphism | 18 / 20 |
| duct_confined | 12 / 20 |
| lymphocytes | 8 / 20 |
| necrosis | 3 / 20 |
| mitoses | 0 / 20 |
| normal_lobules | 0 / 20 |
| cell segmentation overlays | 15 / 21 |

- duct_confined: e.g. "detached epithelial fragments with an ink/dust artifact"
- invasion: e.g. "solid nest of clear cells; infiltrative pattern not evident in this tile"
- lymphocytes: e.g. "grey background with granular debris; no lymphocytes"
- mitoses: e.g. "high-grade solid tumor where mitoses are expected; individual mitotic figures not resolvable at review resolution (256-384 px tiles)"
- necrosis: e.g. "cracked eosinophilic material: necrosis vs fibrin/blood clot not distinguishable"
- normal_lobules: e.g. "fat and fibrous stroma only; no terminal duct-lobular unit or two-layered epithelium in view"
- overlay: e.g. "dense dark nuclei; contours hard to see"
- pleomorphism: e.g. "mostly stroma; too few tumor nuclei to judge pleomorphism"
- tubules: e.g. "stroma and vessels; no tubule in view"

- Cell segmentation spike overlays: no record.
- No result here is clinically validated; every review above is labelled by who did it.

## Limitations

- Research prototype on a small public dataset (BACH, CC BY-NC-ND 4.0); not a clinical device and not for clinical use.
- BACH labels are diagnostic categories, not Nottingham grades; 'likely aggressive' means 'likely invasive carcinoma'.
- Small test set (THUNDER split, 132 images) with partial patient IDs: pseudo-patient groups, wide intervals.
- The surgical lane is simulated: image borders stand in for inked margins and frozen-section artifacts are simulated degradations; composite fields are synthetic constructions from test images.
- Concept scores are features associated with the classes (text-prompt and caption-consistent), not causes and not validated grade components; caption consistency measures caption co-mention, not tissue truth.
- Stain rules are illustrative and require clinical review; the agent only drafts orders.
- The queue simulation is a model with stated assumptions, not a measured workflow outcome.
