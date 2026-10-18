# PathQueue results

> Research prototype — not for clinical use. Every number below is quoted from the eval file named beside it, with its 95% cluster-bootstrap CI (patient / pseudo-patient groups) unless stated.

Generated 2026-10-18T04:36:17Z from 15 eval files (git 1ca8e3d81eb084cf330b290ab7b73b0db97790b9).

## Classification (test split)

Source: `eval/metrics.json` · test n = 132, 24 patient / pseudo-patient groups · 2248 simulated variants excluded · production model `uni2h_lr`.

| Metric | Estimate [95% CI] |
|---|---|
| Invasive vs rest AUROC | 1.00 [1.00, 1.00] (n 132, 24 groups) |
| Macro AUROC | 0.95 [0.86, 0.99] (n 132, 24 groups) |
| Macro-F1 | 0.82 [0.60, 0.92] (n 132, 24 groups) |
| Balanced accuracy | 0.82 [0.60, 0.94] (n 132, 24 groups) |
| Accuracy | 0.83 [0.69, 0.94] (n 132, 24 groups) |
| Invasive sensitivity | 1.00 [1.00, 1.00] (n 132, 24 groups) |
| Invasive specificity | 0.96 [0.92, 1.00] (n 132, 24 groups) |

Model A `uni2h_lr` vs Model B `cb_text` (Invasive-vs-rest AUROC (uncalibrated scores)): A 1.00 [1.00, 1.00] (n 132, 24 groups); B 0.99 [0.96, 1.00] (n 132, 24 groups); paired difference A − B 0.01 [-0.00, 0.04] (n 132, 24 groups).

In situ ↔ invasive confusion: 4 of 30 in situ cases called invasive; 0 of 31 invasive cases called in situ.

## Calibration

Source: `eval/calibration.json` · temperature 0.622 fit on train_oof · 10 equal-mass ECE bins (biased at small n).

| | ECE | Brier | NLL |
|---|---|---|---|
| before temperature | 0.12 [0.09, 0.22] (n 132, 24 groups) | 0.29 [0.15, 0.47] (n 132, 24 groups) | 0.55 [0.30, 0.93] (n 132, 24 groups) |
| after temperature | 0.08 [0.05, 0.22] (n 132, 24 groups) | 0.28 [0.13, 0.52] (n 132, 24 groups) | 0.57 [0.24, 1.19] (n 132, 24 groups) |

## Conformal prediction sets

Source: `eval/conformal.json` · alpha 0.1 (target coverage 90%) · tiers use lac_marginal.

| Method | Coverage | Mean set size | Empty sets |
|---|---|---|---|
| lac_marginal | 67.4% [56.8%, 79.0%] (n 132, 24 groups) | 0.77 [0.72, 0.85] (n 132, 24 groups) | 23.5% |
| aps | 85.6% [71.1%, 94.3%] (n 132, 24 groups) | 1.28 [1.14, 1.36] (n 132, 24 groups) | 2.3% |
| mondrian | 47.7% [32.6%, 74.7%] (n 132, 24 groups) | 0.55 [0.41, 0.80] (n 132, 24 groups) | 45.5% |

Per-class LAC coverage: Normal 52.2% [23.8%, 78.3%] (n 23, 6 groups); Benign 66.7% [42.6%, 85.3%] (n 48, 6 groups); InSitu 53.3% [16.7%, 63.2%] (n 30, 6 groups); Invasive 93.5% [83.3%, 100.0%] (n 31, 17 groups).

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
| ours · uni2h_lr (production (calibrated)) | 81.9 [59.5, 92.1] | 82.1 [59.7, 93.7] |
| ours · abmil (evaluated model) | 78.1 [57.3, 87.6] | 78.2 [58.3, 88.9] |
| ours · cb_cells (evaluated model) | 24.4 [15.7, 33.4] | 34.1 [24.0, 40.7] |
| ours · cb_text (evaluated model) | 69.4 [47.7, 78.9] | 74.2 [51.4, 82.0] |
| ours · cb_text_cells (evaluated model) | 63.1 [43.6, 71.6] | 71.1 [49.5, 76.9] |
| ours · conch_probe_lr (evaluated model) | 73.6 [57.7, 82.4] | 77.5 [59.7, 84.8] |
| ours · fusion_top3 (evaluated model) | 78.8 [54.7, 89.8] | 79.4 [57.3, 90.4] |
| ours · hoptimus0_lr (evaluated model) | 76.1 [54.3, 86.0] | 76.4 [55.2, 87.2] |
| ours · musk_lr (evaluated model) | 66.2 [51.2, 78.9] | 68.1 [53.6, 80.9] |
| ours · uni2h_lr (evaluated model) | 81.9 [58.0, 91.6] | 82.1 [60.3, 93.2] |

- Same split, different protocol: published THUNDER probes resize the whole image; we embed tiles with test-time augmentation and select models by grouped CV inside train.
- Not compared with the 87% ICIAR 2018 challenge result, which used a different, hidden test set.
- Published numbers carry no CI here; ours are cluster-bootstrap 95% CIs in percent.

## Queue simulation (workflow benefit and cost)

Source: `eval/queue_sim.json` · 1000 simulated days · minutes, mean [2.5–97.5 percentile over days].

| Policy | Invasive: mean first look | Invasive: P90 | Unflagged: mean wait | Overtime |
|---|---|---|---|---|
| FIFO (arrival order) | 20.2 [3.5, 60.1] | 37.7 [8.1, 100.8] | 20.0 [4.1, 58.3] | 29.3 [0.0, 104.3] |
| PathQueue tiers | 6.0 [1.9, 12.9] | 12.9 [4.8, 32.0] | 44.0 [5.6, 149.6] | 29.3 [0.0, 104.3] |
| PathQueue tiers + aging guard (configured targets) | 6.0 [1.9, 12.9] | 12.9 [4.8, 32.0] | 44.0 [5.6, 149.6] | 29.3 [0.0, 104.3] |
| Sensitivity: tiers + aging, day-scale targets | 6.0 [1.9, 12.9] | 12.9 [4.8, 32.0] | 43.8 [5.6, 148.0] | 29.3 [0.0, 104.3] |

PathQueue tiers vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 24.0 [0.8, 92.5] min**; false negatives n/a min.

PathQueue tiers + aging guard (configured targets) vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 24.0 [0.8, 92.5] min**; false negatives n/a min.

Sensitivity: tiers + aging, day-scale targets vs FIFO: invasive first look -14.2 [-51.3, 0.0] min; **delay imposed on unflagged cases 23.8 [0.8, 89.7] min**; false negatives n/a min.

Aging guard: share of reads promoted per day — PathQueue tiers + aging guard (configured targets) 0.0%; Sensitivity: tiers + aging, day-scale targets 2.8%.

Assumptions: arrivals: Poisson, 6 per hour for 8 h; cases drawn uniformly with replacement from the clean test split; reader: one reader, fixed 9 min per case, non-preemptive, works overtime until empty; utilization: 0.9; tiers: core.tiers (surgical 1; Invasive top or in the LAC set 2; set size != 1, reliability flag or insufficient tissue 3; else 4); common_random_numbers: True; seed: 44; interval: 2.5-97.5 percentiles over simulated days (simulation uncertainty only); truth_use: true labels only score the waits; they never drive the queue.

- Prioritizing flagged cases delays non-flagged ones: the vs_fifo rows report that cost (and the delay of false negatives) next to the benefit (cf. Annarumma 2019).
- A simulation with stated assumptions, not a measured workflow outcome.
- At the configured aging targets (tier 1: 20 min, tier 2: 1440 min, tier 3: 1440 min, tier 4: 2880 min) no case waits long enough to be promoted within a 8 h day, so tiers and tiers_aging coincide: the aging guard has NO measured effect here and must not be claimed to protect unflagged cases.
- tiers_aging_day_scale is a sensitivity run with illustrative day-scale aging targets (settings queue_sim.aging_scenarios), not the configured system: 2.8% of reads promoted; unflagged mean wait 44.0 -> 43.8 min vs plain tiers.

## Simulated margin read

Source: `eval/margin.json` · threshold 1.000 (CI {'lo': 0.876189, 'hi': 0.999759, 'ci': 0.95, 'n_boot': 2000, 'n_valid': 2000, 'n': 30, 'n_groups': 6}, from 30 benign calibration images).

- Benign test images, false tumor-at-margin per margin (pooled over 4 margins): 0.0% [0.0%, 0.0%] (n 284, 7 groups) (the threshold targets ≤ 5% per margin; 71 of 71 benign test images scored). Image-level, any of 4 margins: 0.0% [0.0%, 0.0%] (n 71, 7 groups) (no 5% target at image level).
- Carcinoma test images with ≥ 1 margin called clear: 47.5% [21.9%, 62.4%] (n 61, 19 groups); Spearman(max margin score, case risk) 0.45 [0.04, 0.75] (n 61, 19 groups).
- Composite fields (64 of 64 scored): exact-negative margin called clear 100.0% [100.0%, 100.0%] (n 64, 41 groups); benign-half segments below threshold 100.0% [100.0%, 100.0%] (n 768, 41 groups); carcinoma-half segments above threshold 35.4% [27.1%, 45.3%] (n 768, 41 groups); tumor-side margin involved 70.3% [59.0%, 82.0%] (n 64, 41 groups). An echo of the diagnosis would score ~0% clear.
- Annotated WSI windows (84, 10 patients): sensitivity 36.2% [20.3%, 51.0%] (n 141, 10 groups), specificity 100.0% [100.0%, 100.0%] (n 195, 10 groups).

- Simulated margins: image borders stand in for inked margins (top superior, bottom inferior, left medial, right lateral); anterior/posterior are not representable in 2D.
- Raw-score evaluation: the reliability defer rule is a separate safety layer.
- Composite fields are built from test images only and never enter headline metrics.

## Other evaluations (concepts, reliability, retrieval, deletion, cells)

### `eval/model_choice.json`

- cv_metrics · conch_probe_lr · log_loss: 0.32 [0.24, 0.41] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · auroc_invasive: 0.99 [0.98, 1.00] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · balanced_accuracy: 0.90 [0.86, 0.94] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · macro_f1: 0.90 [0.85, 0.94] (n 218, 49 groups)
- cv_metrics · conch_probe_lr · macro_auroc: 0.98 [0.97, 0.99] (n 218, 49 groups)
- cv_metrics · uni2h_lr · log_loss: 0.31 [0.23, 0.41] (n 218, 49 groups)
- cv_metrics · uni2h_lr · auroc_invasive: 1.00 [1.00, 1.00] (n 218, 49 groups)
- cv_metrics · uni2h_lr · balanced_accuracy: 0.93 [0.89, 0.97] (n 218, 49 groups)
- cv_metrics · uni2h_lr · macro_f1: 0.93 [0.89, 0.97] (n 218, 49 groups)
- cv_metrics · uni2h_lr · macro_auroc: 0.99 [0.97, 1.00] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · log_loss: 0.36 [0.28, 0.45] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · auroc_invasive: 1.00 [0.99, 1.00] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · balanced_accuracy: 0.90 [0.85, 0.94] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · macro_f1: 0.90 [0.84, 0.94] (n 218, 49 groups)
- cv_metrics · hoptimus0_lr · macro_auroc: 0.98 [0.97, 1.00] (n 218, 49 groups)
- cv_metrics · musk_lr · log_loss: 0.35 [0.26, 0.46] (n 218, 49 groups)
- cv_metrics · musk_lr · auroc_invasive: 0.98 [0.95, 1.00] (n 218, 49 groups)
- cv_metrics · musk_lr · balanced_accuracy: 0.88 [0.83, 0.93] (n 218, 49 groups)
- cv_metrics · musk_lr · macro_f1: 0.89 [0.83, 0.93] (n 218, 49 groups)
- cv_metrics · musk_lr · macro_auroc: 0.98 [0.96, 0.99] (n 218, 49 groups)
- cv_metrics · cb_text · log_loss: 0.49 [0.39, 0.61] (n 218, 49 groups)
- cv_metrics · cb_text · auroc_invasive: 0.99 [0.97, 1.00] (n 218, 49 groups)
- cv_metrics · cb_text · balanced_accuracy: 0.79 [0.73, 0.85] (n 218, 49 groups)
- cv_metrics · cb_text · macro_f1: 0.79 [0.72, 0.85] (n 218, 49 groups)
- cv_metrics · cb_text · macro_auroc: 0.95 [0.93, 0.97] (n 218, 49 groups)
- cv_metrics · cb_text_cells · log_loss: 0.75 [0.65, 0.84] (n 218, 49 groups)
- cv_metrics · cb_text_cells · auroc_invasive: 0.98 [0.96, 1.00] (n 218, 49 groups)
- cv_metrics · cb_text_cells · balanced_accuracy: 0.74 [0.69, 0.79] (n 218, 49 groups)
- cv_metrics · cb_text_cells · macro_f1: 0.73 [0.66, 0.79] (n 218, 49 groups)
- cv_metrics · cb_text_cells · macro_auroc: 0.92 [0.89, 0.95] (n 218, 49 groups)
- cv_metrics · cb_cells · log_loss: 1.32 [1.27, 1.38] (n 218, 49 groups)
- cv_metrics · cb_cells · auroc_invasive: 0.68 [0.58, 0.78] (n 218, 49 groups)
- cv_metrics · cb_cells · balanced_accuracy: 0.37 [0.31, 0.44] (n 218, 49 groups)
- cv_metrics · cb_cells · macro_f1: 0.33 [0.26, 0.39] (n 218, 49 groups)
- cv_metrics · cb_cells · macro_auroc: 0.62 [0.57, 0.67] (n 218, 49 groups)
- cv_metrics · abmil · log_loss: 0.28 [0.12, 0.49] (n 218, 49 groups)
- cv_metrics · abmil · auroc_invasive: 1.00 [1.00, 1.00] (n 218, 49 groups)
- cv_metrics · abmil · balanced_accuracy: 0.93 [0.89, 0.97] (n 218, 49 groups)
- cv_metrics · abmil · macro_f1: 0.93 [0.88, 0.96] (n 218, 49 groups)
- cv_metrics · abmil · macro_auroc: 0.99 [0.97, 1.00] (n 218, 49 groups)

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
- nct · concepts · lymphocytes · text: 1.00 [1.00, 1.00] (n 2700, 2700 groups)
- nct · concepts · lymphocytes · cav: 0.32 [0.29, 0.35] (n 2700, 2700 groups)
- nct · concepts · necrosis · text: 0.94 [0.93, 0.95] (n 2700, 2700 groups)
- nct · concepts · necrosis · cav: 0.95 [0.94, 0.96] (n 2700, 2700 groups)
- nct · concepts · pleomorphism · text: 0.59 [0.56, 0.62] (n 2700, 2700 groups)
- nct · concepts · pleomorphism · cav: 0.61 [0.58, 0.65] (n 2700, 2700 groups)
- nct · concepts · benign_stroma_fat · text: 0.87 [0.85, 0.88] (n 2700, 2700 groups)
- nct · concepts · benign_stroma_fat · cav: 0.80 [0.78, 0.82] (n 2700, 2700 groups)

### `eval/concepts_bach.json`

- concepts · tubules · class_means · Normal: 1.06 [0.85, 1.30] (n 23, 6 groups)
- concepts · tubules · class_means · Benign: 1.10 [0.96, 1.34] (n 48, 6 groups)
- concepts · tubules · class_means · InSitu: 0.07 [-0.78, 0.59] (n 30, 6 groups)
- concepts · tubules · class_means · Invasive: -0.67 [-1.03, -0.27] (n 31, 17 groups)
- concepts · tubules · auroc_Invasive_vs_rest: 0.11 [0.02, 0.21] (n 132, 24 groups)
- concepts · pleomorphism · class_means · Normal: 0.34 [0.17, 0.45] (n 23, 6 groups)
- concepts · pleomorphism · class_means · Benign: -0.02 [-0.30, 0.38] (n 48, 6 groups)
- concepts · pleomorphism · class_means · InSitu: 0.39 [-0.11, 1.34] (n 30, 6 groups)
- concepts · pleomorphism · class_means · Invasive: 1.61 [1.29, 1.92] (n 31, 17 groups)
- concepts · pleomorphism · auroc_Invasive_vs_rest: 0.93 [0.87, 0.97] (n 132, 24 groups)
- concepts · mitoses · class_means · Normal: 0.33 [0.05, 0.53] (n 23, 6 groups)
- concepts · mitoses · class_means · Benign: 0.23 [-0.16, 0.53] (n 48, 6 groups)
- concepts · mitoses · class_means · InSitu: 1.07 [0.68, 1.52] (n 30, 6 groups)
- concepts · mitoses · class_means · Invasive: 1.49 [1.18, 1.77] (n 31, 17 groups)
- concepts · mitoses · auroc_Invasive_vs_rest: 0.83 [0.73, 0.95] (n 132, 24 groups)
- concepts · invasion · class_means · Normal: 0.34 [-0.10, 0.63] (n 23, 6 groups)
- concepts · invasion · class_means · Benign: 0.10 [-0.31, 0.59] (n 48, 6 groups)
- concepts · invasion · class_means · InSitu: 0.32 [0.07, 0.54] (n 30, 6 groups)
- concepts · invasion · class_means · Invasive: 1.24 [1.02, 1.44] (n 31, 17 groups)
- concepts · invasion · auroc_Invasive_vs_rest: 0.93 [0.84, 0.98] (n 132, 24 groups)
- concepts · duct_confined · class_means · Normal: -0.21 [-0.35, -0.06] (n 23, 6 groups)
- concepts · duct_confined · class_means · Benign: 0.30 [-0.35, 0.82] (n 48, 6 groups)
- concepts · duct_confined · class_means · InSitu: 0.97 [0.81, 1.12] (n 30, 6 groups)
- concepts · duct_confined · class_means · Invasive: 0.01 [-0.16, 0.21] (n 31, 17 groups)
- concepts · duct_confined · auroc_Invasive_vs_rest: 0.35 [0.14, 0.66] (n 132, 24 groups)
- concepts · normal_lobules · class_means · Normal: 1.40 [1.20, 1.54] (n 23, 6 groups)
- concepts · normal_lobules · class_means · Benign: 0.72 [0.27, 1.27] (n 48, 6 groups)
- concepts · normal_lobules · class_means · InSitu: 0.23 [-0.42, 0.43] (n 30, 6 groups)
- concepts · normal_lobules · class_means · Invasive: -0.05 [-0.24, 0.11] (n 31, 17 groups)
- concepts · normal_lobules · auroc_Invasive_vs_rest: 0.20 [0.04, 0.34] (n 132, 24 groups)
- concepts · benign_stroma_fat · class_means · Normal: 1.15 [0.86, 1.31] (n 23, 6 groups)
- concepts · benign_stroma_fat · class_means · Benign: 0.65 [0.27, 1.14] (n 48, 6 groups)
- concepts · benign_stroma_fat · class_means · InSitu: -0.01 [-1.22, 0.28] (n 30, 6 groups)
- concepts · benign_stroma_fat · class_means · Invasive: -0.26 [-0.58, 0.02] (n 31, 17 groups)
- concepts · benign_stroma_fat · auroc_Invasive_vs_rest: 0.22 [0.06, 0.36] (n 132, 24 groups)
- concepts · lymphocytes · class_means · Normal: 0.72 [0.29, 1.06] (n 23, 6 groups)
- concepts · lymphocytes · class_means · Benign: 0.65 [0.50, 1.00] (n 48, 6 groups)
- concepts · lymphocytes · class_means · InSitu: 0.67 [0.41, 1.03] (n 30, 6 groups)
- concepts · lymphocytes · class_means · Invasive: 0.48 [0.21, 0.76] (n 31, 17 groups)
- concepts · lymphocytes · auroc_Invasive_vs_rest: 0.42 [0.25, 0.60] (n 132, 24 groups)

### `eval/zeroshot.json`

- encoders · conch · macro_f1: 0.48 [0.35, 0.56] (n 132, 24 groups)
- encoders · conch · balanced_accuracy: 0.54 [0.46, 0.60] (n 132, 24 groups)
- encoders · conch · accuracy: 0.63 [0.44, 0.77] (n 132, 24 groups)
- encoders · musk · macro_f1: 0.50 [0.40, 0.61] (n 132, 24 groups)
- encoders · musk · balanced_accuracy: 0.55 [0.51, 0.67] (n 132, 24 groups)
- encoders · musk · accuracy: 0.64 [0.48, 0.77] (n 132, 24 groups)
- encoders · pathgenclip · macro_f1: 0.54 [0.44, 0.67] (n 132, 24 groups)
- encoders · pathgenclip · balanced_accuracy: 0.56 [0.51, 0.70] (n 132, 24 groups)
- encoders · pathgenclip · accuracy: 0.64 [0.49, 0.76] (n 132, 24 groups)

### `eval/reliability.json`

- accuracy_vs_degradation · curve · blur · 0: 0.82 [0.66, 0.92] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 1: 0.70 [0.57, 0.86] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 2: 0.73 [0.60, 0.86] (n 132, 24 groups)
- accuracy_vs_degradation · curve · blur · 3: 0.65 [0.53, 0.80] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 0: 0.82 [0.66, 0.92] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 1: 0.67 [0.52, 0.85] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 2: 0.65 [0.50, 0.84] (n 132, 24 groups)
- accuracy_vs_degradation · curve · jpeg · 3: 0.66 [0.56, 0.79] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 0: 0.82 [0.66, 0.92] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 1: 0.70 [0.57, 0.84] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 2: 0.70 [0.56, 0.83] (n 132, 24 groups)
- accuracy_vs_degradation · curve · stain · 3: 0.70 [0.57, 0.84] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 0: 0.82 [0.66, 0.92] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 1: 0.73 [0.59, 0.86] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 2: 0.71 [0.59, 0.86] (n 132, 24 groups)
- accuracy_vs_degradation · curve · resample · 3: 0.64 [0.53, 0.78] (n 132, 24 groups)
- error_auroc_test: 0.67 [0.54, 0.81] (n 1716, 24 groups)
- feature_auroc_test · ood_percentile: 0.53 [0.44, 0.61] (n 1716, 24 groups)
- feature_auroc_test · set_size: 0.42 [0.36, 0.45] (n 1716, 24 groups)
- feature_auroc_test · conch_musk_disagree: 0.62 [0.51, 0.76] (n 1716, 24 groups)
- error_recall_test: 0.52 [0.32, 0.70] (n 1716, 24 groups)

### `eval/retrieval.json`

- precision_at_k_test: 0.65 [0.56, 0.82] (n 132, 24 groups)

### `eval/deletion.json`

- models · uni2h_lr · predicted_aggressive · mean_drop_random: 0.00 [-0.00, 0.00] (n 34, 17 groups)
- models · uni2h_lr · predicted_aggressive · mean_drop_risk: 0.02 [0.00, 0.04] (n 29, 15 groups)
- models · uni2h_lr · predicted_aggressive · mean_drop_concept: 0.02 [-0.00, 0.04] (n 32, 16 groups)
- models · uni2h_lr · predicted_aggressive · concept_minus_random: 0.02 [-0.00, 0.04] (n 32, 16 groups)
- models · uni2h_lr · predicted_aggressive · risk_minus_random: 0.02 [-0.00, 0.04] (n 29, 15 groups)
- models · uni2h_lr · all_eligible · mean_drop_random: -0.00 [-0.00, 0.00] (n 130, 24 groups)
- models · uni2h_lr · all_eligible · mean_drop_risk: 0.01 [0.00, 0.01] (n 125, 23 groups)
- models · uni2h_lr · all_eligible · mean_drop_concept: 0.00 [-0.00, 0.00] (n 128, 24 groups)
- models · uni2h_lr · all_eligible · concept_minus_random: 0.00 [-0.00, 0.00] (n 128, 24 groups)
- models · uni2h_lr · all_eligible · risk_minus_random: 0.01 [0.00, 0.02] (n 125, 23 groups)

### `eval/cells.json`

- direction_checks · features · neo_fraction · class_means · Normal: 0.17 [0.12, 0.23] (n 17, 6 groups)
- direction_checks · features · neo_fraction · class_means · Benign: 0.46 [0.15, 0.68] (n 40, 6 groups)
- direction_checks · features · neo_fraction · class_means · InSitu: 0.74 [0.66, 0.79] (n 29, 5 groups)
- direction_checks · features · neo_fraction · class_means · Invasive: 0.44 [0.34, 0.53] (n 28, 16 groups)
- direction_checks · features · neo_fraction · difference: 0.06 [-0.15, 0.29] (n 85, 21 groups)
- direction_checks · features · neo_density · class_means · Normal: 636.09 [179.50, 1376.00] (n 17, 6 groups)
- direction_checks · features · neo_density · class_means · Benign: 1769.17 [293.59, 2784.93] (n 40, 6 groups)
- direction_checks · features · neo_density · class_means · InSitu: 3989.03 [3304.27, 6054.76] (n 29, 5 groups)
- direction_checks · features · neo_density · class_means · Invasive: 1808.45 [1339.18, 2439.17] (n 28, 16 groups)
- direction_checks · features · neo_density · difference: 377.22 [-822.71, 1820.99] (n 85, 21 groups)
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
- direction_checks · features · infl_fraction · difference: -0.04 [-0.08, -0.00] (n 85, 21 groups)
- direction_checks · features · til_proximity · class_means · Normal: 0.47 [0.37, 0.59] (n 16, 6 groups)
- direction_checks · features · til_proximity · class_means · Benign: 0.59 [0.35, 0.77] (n 34, 6 groups)
- direction_checks · features · til_proximity · class_means · InSitu: 0.73 [0.67, 0.91] (n 23, 5 groups)
- direction_checks · features · til_proximity · class_means · Invasive: 0.73 [0.54, 0.88] (n 22, 15 groups)
- direction_checks · features · til_proximity · difference: 0.18 [-0.05, 0.42] (n 72, 20 groups)
- direction_checks · features · dead_fraction · class_means · Normal: 0.02 [0.01, 0.04] (n 17, 6 groups)
- direction_checks · features · dead_fraction · class_means · Benign: 0.02 [0.01, 0.04] (n 40, 6 groups)
- direction_checks · features · dead_fraction · class_means · InSitu: 0.01 [0.00, 0.02] (n 29, 5 groups)
- direction_checks · features · dead_fraction · class_means · Invasive: 0.01 [0.00, 0.01] (n 28, 16 groups)
- direction_checks · features · dead_fraction · difference: -0.01 [-0.03, -0.00] (n 85, 21 groups)

## Reviewed checks

- Concept top-tile and cell-overlay spot check: **AI-reviewed (Claude), not a pathologist** · `eval/spotcheck.json` · summary {'concept': {}, 'cells': {'n': 1, 'acceptable': 0}, 'orphaned': 0}.
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
