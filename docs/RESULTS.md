# PathQueue results

> Research prototype — not for clinical use. Every number below is quoted from the eval file named beside it, with its 95% cluster-bootstrap CI (patient / pseudo-patient groups) unless stated.

Generated 2026-10-18T13:54:01Z from 16 eval files (git 19b15557e966c050e4a0cc0df67c0546d00464a6, 4154f7693e436121aeffecbf8743c5af029a6968, 700a3b61b0f999a12e7a1a7b54058d6495b93689, 7ab2ffcbddd91931b194a5dec698c13622fef8d7).

## Tissue quality control (fixed in this run)

Source: `eval/qc.json` (pipeline/01q_requalify_qc). The original tissue mask required HSV value < 0.92; HSV value is the brightest RGB channel, so pale H&E tissue was rejected. With saturation > 0.07 alone, contract-tile QC pass rate went 75.1% -> 98.7% and insufficient-tissue images 51 -> 0 (before: Normal 8, Benign 3, InSitu 8, Invasive 32). Tile fractions were recomputed on display JPEGs; the old rule recomputed the same way reproduces the full-resolution Delta values (r = 0.998, mean |diff| = 0.012). Every tile had been embedded on Delta, so no new GPU work was needed; steps 04-08 and the evaluation were re-run.

## Classification (test split)

Source: `eval/metrics.json` · test n = 132, 24 patient / pseudo-patient groups · 2248 simulated variants excluded · production model `uni2h_lr`.

| Metric | Estimate [95% CI] |
|---|---|
| Invasive vs rest AUROC | 0.998 [0.994, 1.000] (n 132, 24 groups) |
| Macro AUROC | 0.947 [0.852, 0.992] (n 132, 24 groups) |
| Macro-F1 | 0.83 [0.61, 0.93] (n 132, 24 groups) |
| Balanced accuracy | 0.84 [0.62, 0.94] (n 132, 24 groups) |
| Accuracy | 0.84 [0.70, 0.94] (n 132, 24 groups) |
| Invasive sensitivity | 1.000 [1.000, 1.000] (n 132, 24 groups) |
| Invasive specificity | 0.970 [0.936, 1.000] (n 132, 24 groups) |

Model A `uni2h_lr` vs Model B `cb_text` (Invasive-vs-rest AUROC (uncalibrated scores)): A 0.998 [0.994, 1.000] (n 132, 24 groups); B 1.000 [1.000, 1.000] (n 132, 24 groups); paired difference A − B 0.00 [-0.01, 0.00] (n 132, 24 groups).

In situ ↔ invasive confusion: 3 of 30 in situ cases called invasive; 0 of 31 invasive cases called in situ.

## Calibration

Source: `eval/calibration.json` · temperature 0.501 fit on train_oof · 10 equal-mass ECE bins (biased at small n).

| | ECE | Brier | NLL |
|---|---|---|---|
| before temperature | 0.10 [0.07, 0.19] (n 132, 24 groups) | 0.28 [0.14, 0.47] (n 132, 24 groups) | 0.56 [0.29, 1.00] (n 132, 24 groups) |
| after temperature | 0.05 [0.04, 0.22] (n 132, 24 groups) | 0.27 [0.11, 0.53] (n 132, 24 groups) | 0.66 [0.20, 1.56] (n 132, 24 groups) |

## Conformal prediction sets

Source: `eval/conformal.json` · alpha 0.1 (target coverage 90%) · tiers use aps_marginal_nonempty.

| Method | Coverage | Mean set size | Empty sets |
|---|---|---|---|
| lac_marginal | 72.7% [60.7%, 84.4%] (n 132, 24 groups) | 0.81 [0.76, 0.89] (n 132, 24 groups) | 18.9% |
| aps | 87.9% [73.7%, 95.7%] (n 132, 24 groups) | 1.29 [1.15, 1.36] (n 132, 24 groups) | 2.3% |
| mondrian | 53.0% [38.7%, 73.6%] (n 132, 24 groups) | 0.61 [0.48, 0.81] (n 132, 24 groups) | 39.4% |
| aps_marginal_nonempty | 90.2% [75.7%, 98.1%] (n 132, 24 groups) | 1.31 [1.18, 1.38] (n 132, 24 groups) | 0.0% |

Per-class coverage of the tier set (aps_marginal_nonempty): Normal 82.6% [60.9%, 100.0%] (n 23, 6 groups); Benign 87.5% [52.9%, 100.0%] (n 48, 6 groups); InSitu 90.0% [66.7%, 94.7%] (n 30, 6 groups); Invasive 100.0% [100.0%, 100.0%] (n 31, 17 groups).

Per-class LAC coverage (pre-registered): Normal 56.5% [26.3%, 80.0%] (n 23, 6 groups); Benign 70.8% [44.4%, 88.2%] (n 48, 6 groups); InSitu 66.7% [33.3%, 79.0%] (n 30, 6 groups); Invasive 93.5% [83.3%, 100.0%] (n 31, 17 groups).

**Post-hoc change, disclosed:** marginal LAC was the pre-registered tier set. On the held-out test split it returned empty sets for 18.9% of cases and no multi-class sets, so it gave the reader no differential; the 50-image calibration split is far easier than the test split, which likely violates the exchangeability the guarantee needs. Tiers now use the set named above (APS, Romano et al. 2020, with the argmax class added to empty sets). The switch was made after viewing test results, so its observed coverage is optimistic and is not an independent confirmation.

Under exchangeability and continuous scores, coverage conditional on a random calibration set follows ~Beta(46, 5) (mean 0.902, SD 0.041); the 1 - alpha = 0.90 guarantee is marginal over calibration sets, not a guarantee for this test cohort. Compare the observed held-out coverage with the target.

## Benchmark anchor (THUNDER split)

Source: `eval/thunder.json` · **same split, different protocol (published: whole-image resize; ours: tiles + TTA)** · THUNDER official leaderboards, BACH F1 scores; verified 2026-10-17 at https://mics-lab.github.io/thunder/leaderboards/.

| Model | Macro-F1 (%) | Balanced accuracy (%) |
|---|---|---|
| published linear probe · conch | 84.4 (no CI) | — |
| published linear probe · uni2h | 84.7 (no CI) | — |
| published linear probe · musk | 69.9 (no CI) | — |
| published zero-shot · conch | 56.1 (no CI) | — |
| published zero-shot · musk | 62.5 (no CI) | — |
| ours · uni2h_lr (production (calibrated)) | 83.0 [60.9, 92.7] | 83.5 [61.6, 94.0] |
| ours · abmil (evaluated model) | 81.0 [63.4, 90.1] | 80.9 [63.8, 91.6] |
| ours · cb_cells (evaluated model) | 24.4 [15.7, 33.4] | 34.1 [24.0, 40.7] |
| ours · cb_text (evaluated model) | 70.8 [53.0, 82.3] | 74.2 [58.2, 86.5] |
| ours · cb_text_cells (evaluated model) | 66.7 [56.5, 76.1] | 74.0 [65.7, 79.5] |
| ours · conch_probe_lr (evaluated model) | 78.4 [63.2, 88.0] | 81.9 [67.1, 90.7] |
| ours · fusion_top3 (evaluated model) | 80.7 [62.6, 89.3] | 81.1 [62.6, 90.1] |
| ours · hoptimus0_lr (evaluated model) | 78.2 [59.9, 87.1] | 78.6 [60.7, 88.6] |
| ours · musk_lr (evaluated model) | 69.4 [56.2, 81.7] | 70.5 [57.5, 86.2] |
| ours · uni2h_lr (evaluated model) | 83.0 [61.1, 92.2] | 83.5 [62.2, 93.6] |

- Same split, different protocol: published THUNDER probes resize the whole image; we embed tiles with test-time augmentation and select models by grouped CV inside train.
- Not compared with the 87% ICIAR 2018 challenge result, which used a different, hidden test set.
- Published numbers carry no CI here; ours are cluster-bootstrap 95% CIs in percent.

## Queue simulation (workflow benefit and cost)

Source: `eval/queue_sim.json` · 1000 simulated days · minutes, mean [2.5–97.5 percentile over days].

| Policy | Invasive: mean first look | Invasive: P90 | Unflagged: mean wait | Overtime |
|---|---|---|---|---|
| FIFO (arrival order) | 20.2 [3.5, 60.1] | 37.7 [8.1, 100.8] | 20.0 [4.0, 59.5] | 29.3 [0.0, 104.3] |
| PathQueue tiers | 6.0 [1.9, 12.9] | 13.0 [4.8, 31.5] | 45.0 [5.6, 151.9] | 29.3 [0.0, 104.3] |
| PathQueue tiers + aging guard (configured targets) | 6.0 [1.9, 12.9] | 13.0 [4.8, 31.5] | 45.0 [5.6, 151.9] | 29.3 [0.0, 104.3] |
| Sensitivity: tiers + aging, day-scale targets | 6.0 [1.9, 12.9] | 13.0 [4.8, 31.5] | 44.8 [5.6, 151.2] | 29.3 [0.0, 104.3] |

PathQueue tiers vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 25.0 [0.7, 93.2] min**; false negatives: not available (no simulated day had a truly invasive case outside tiers 1-2).

PathQueue tiers + aging guard (configured targets) vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 25.0 [0.7, 93.2] min**; false negatives: not available (no simulated day had a truly invasive case outside tiers 1-2).

Sensitivity: tiers + aging, day-scale targets vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 24.8 [0.7, 92.7] min**; false negatives: not available (no simulated day had a truly invasive case outside tiers 1-2).

Aging guard: share of reads promoted per day — PathQueue tiers + aging guard (configured targets) 0.0%; Sensitivity: tiers + aging, day-scale targets 2.7%.

Assumptions: arrivals: Poisson, 6 per hour for 8 h; cases drawn uniformly with replacement from the clean test split; reader: one reader, fixed 9 min per case, non-preemptive, works overtime until empty; utilization: 0.9; tiers: core.tiers (surgical 1; Invasive top or in the aps_marginal_nonempty conformal set 2; set size != 1, reliability flag or insufficient tissue 3; else 4); common_random_numbers: True; seed: 44; interval: 2.5-97.5 percentiles over simulated days (simulation uncertainty only); truth_use: true labels only score the waits; they never drive the queue.

- Prioritizing flagged cases delays non-flagged ones: the vs_fifo rows report that cost (and the delay of false negatives) next to the benefit (cf. Annarumma 2019).
- A simulation with stated assumptions, not a measured workflow outcome.
- At the configured aging targets (tier 1: 20 min, tier 2: 1440 min, tier 3: 1440 min, tier 4: 2880 min) no case waits long enough to be promoted within a 8 h day, so tiers and tiers_aging coincide: the aging guard has NO measured effect here and must not be claimed to protect unflagged cases.
- tiers_aging_day_scale is a sensitivity run with illustrative day-scale aging targets (settings queue_sim.aging_scenarios), not the configured system: 2.7% of reads promoted; unflagged mean wait 45.0 -> 44.8 min vs plain tiers.

## Simulated margin read

Source: `eval/margin.json` · threshold P = 0.99982 [0.84738, 0.99989] (n 30, 6 groups), from 30 benign calibration images.

- Benign test images, false tumor-at-margin per margin (pooled over 4 margins): 0.0% [0.0%, 0.0%] (n 284, 7 groups) (the threshold targets ≤ 5% per margin; 71 of 71 benign test images scored). Image-level, any of 4 margins: 0.0% [0.0%, 0.0%] (n 71, 7 groups) (no 5% target at image level).
- Carcinoma test images with ≥ 1 margin called clear: 47.5% [25.0%, 64.7%] (n 61, 19 groups); Spearman(max margin score, case risk) 0.54 [0.05, 0.84] (n 61, 19 groups).
- Composite fields (64 of 64 scored): exact-negative margin called clear 100.0% [100.0%, 100.0%] (n 64, 41 groups); benign-half segments below threshold 100.0% [100.0%, 100.0%] (n 768, 41 groups); carcinoma-half segments above threshold 40.0% [30.8%, 50.9%] (n 768, 41 groups); tumor-side margin involved 70.3% [57.7%, 82.4%] (n 64, 41 groups). An echo of the diagnosis would score ~0% clear.
- Annotated WSI windows (84, 10 patients): sensitivity 45.4% [25.4%, 65.2%] (n 141, 10 groups), specificity 98.5% [96.7%, 100.0%] (n 195, 10 groups).

- Simulated margins: image borders stand in for inked margins (top superior, bottom inferior, left medial, right lateral); anterior/posterior are not representable in 2D.
- Raw-score evaluation: the reliability defer rule is a separate safety layer.
- Composite fields are built from test images only and never enter headline metrics.

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
- cv_metrics · hoptimus0_lr · log_loss: 0.32 [0.25, 0.40] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · auroc_invasive: 0.997 [0.991, 1.000] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · balanced_accuracy: 0.910 [0.867, 0.952] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · macro_f1: 0.913 [0.865, 0.952] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · macro_auroc: 0.990 [0.978, 0.998] (n 218, 49 groups)
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
- cv_metrics · abmil · log_loss: 0.26 [0.12, 0.43] (n 218, 49 groups)
- cv_metrics · abmil · auroc_invasive: 1.000 [0.999, 1.000] (n 218, 49 groups)
- cv_metrics · abmil · balanced_accuracy: 0.921 [0.881, 0.959] (n 218, 49 groups)
- cv_metrics · abmil · macro_f1: 0.922 [0.877, 0.957] (n 218, 49 groups)
- cv_metrics · abmil · macro_auroc: 0.991 [0.982, 0.998] (n 218, 49 groups)
- cv_metrics · fusion_top3 · log_loss: 0.21 [0.13, 0.32] (n 218, 49 groups)
- cv_metrics · fusion_top3 · auroc_invasive: 0.998 [0.995, 1.000] (n 218, 49 groups)
- cv_metrics · fusion_top3 · balanced_accuracy: 0.942 [0.905, 0.977] (n 218, 49 groups)
- cv_metrics · fusion_top3 · macro_f1: 0.943 [0.903, 0.976] (n 218, 49 groups)
- cv_metrics · fusion_top3 · macro_auroc: 0.991 [0.978, 0.998] (n 218, 49 groups)
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

- accuracy_vs_degradation · curve · blur · 0: 0.84 [0.68, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 1: 0.70 [0.55, 0.86] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 2: 0.73 [0.56, 0.89] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 3: 0.69 [0.57, 0.84] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 0: 0.84 [0.68, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 1: 0.70 [0.56, 0.90] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 2: 0.70 [0.54, 0.90] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 3: 0.67 [0.56, 0.80] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 0: 0.84 [0.68, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 1: 0.70 [0.54, 0.84] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 2: 0.69 [0.54, 0.84] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 3: 0.70 [0.57, 0.83] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 0: 0.84 [0.68, 0.94] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 1: 0.69 [0.54, 0.84] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 2: 0.74 [0.61, 0.88] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 3: 0.73 [0.64, 0.85] (n 132, 24 groups)
- error_auroc_test: 0.70 [0.59, 0.86] (n 1716, 24 groups)
- feature_auroc_test · ood_percentile: 0.49 [0.40, 0.58] (n 1716, 24 groups)
- feature_auroc_test · set_size: 0.70 [0.64, 0.79] (n 1716, 24 groups)
- feature_auroc_test · conch_musk_disagree: 0.61 [0.51, 0.75] (n 1716, 24 groups)
- error_recall_test: 0.46 [0.29, 0.61] (n 1716, 24 groups)

### `eval/retrieval.json`

- precision_at_k_test: 0.67 [0.58, 0.83] (n 132, 24 groups)

### `eval/deletion.json`

- models · uni2h_lr · predicted_aggressive · mean_drop_random: 0.00 [0.00, 0.00] (n 34, 17 groups)
- models · uni2h_lr · predicted_aggressive · mean_drop_risk: 0.02 [0.00, 0.04] (n 34, 17 groups)
- models · uni2h_lr · predicted_aggressive · mean_drop_concept: 0.00 [-0.02, 0.01] (n 34, 17 groups)
- models · uni2h_lr · predicted_aggressive · concept_minus_random: 0.00 [-0.02, 0.01] (n 34, 17 groups)
- models · uni2h_lr · predicted_aggressive · risk_minus_random: 0.02 [0.00, 0.05] (n 34, 17 groups)
- models · uni2h_lr · all_eligible · mean_drop_random: 0.00 [0.00, 0.00] (n 132, 24 groups)
- models · uni2h_lr · all_eligible · mean_drop_risk: 0.01 [0.00, 0.02] (n 132, 24 groups)
- models · uni2h_lr · all_eligible · mean_drop_concept: 0.00 [-0.01, 0.01] (n 132, 24 groups)
- models · uni2h_lr · all_eligible · concept_minus_random: 0.00 [-0.01, 0.01] (n 132, 24 groups)
- models · uni2h_lr · all_eligible · risk_minus_random: 0.01 [0.00, 0.02] (n 132, 24 groups)

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

_Eval files not present at generation time: finetune._
