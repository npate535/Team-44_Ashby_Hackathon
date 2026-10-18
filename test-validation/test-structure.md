# PathQueue Validation Test Structure

Team 44 · 2026 Ashby Prize Hackathon · Research prototype, not for clinical use

This document specifies **five strategies** to validate PathQueue's accuracy and its comparative advantage over **CONCH** and **MUSK**, plus **one strategy** to measure time savings in a simulated diagnostic pathology workflow. Every strategy states its hypothesis, protocol, metrics, pass criteria, and the test files that implement it.

---

## 0. Ground rules (apply to every strategy)

### 0.1 Systems under comparison

| Arm | Role | Encoder | Head |
|---|---|---|---|
| **PathQueue** | System under test | UNI2-h | Linear probe + calibration + APS conformal |
| **CONCH-probe** | Primary comparator | CONCH image tower | Identical head and pipeline |
| **MUSK-probe** | Primary comparator | MUSK image tower | Identical head and pipeline |
| **CONCH-zeroshot** | Secondary comparator | CONCH + text prompts | None (zero-shot) |
| **MUSK-zeroshot** | Secondary comparator | MUSK + text prompts | None (zero-shot) |
| **Floor** | Sanity | Majority class / ImageNet ViT | Identical head |

Optional reference arms already in the repo: H-optimus-0 and PathGen-CLIP.

**The only variable that may differ between arms is the encoder (or, for zero-shot arms, the absence of a head).** Splits, head type, hyperparameter search budget, calibration method, and conformal procedure are held constant. Any deviation invalidates the comparison.

### 0.2 Pre-registration

Before running anything, commit `tests/validation/PREREG.md` containing:

- **Primary metric:** balanced accuracy (4-class BACH: normal, benign, in situ, invasive)
- **Primary safety metric:** invasive recall at a fixed operating point
- **Primary comparison:** PathQueue vs. the *better* of CONCH-probe and MUSK-probe
- **Decision margin:** non-inferiority margin Δ (suggest 2 percentage points) and superiority threshold (paired 95% CI excludes 0)
- **Multiple comparison handling:** Holm correction across the two primary comparators

The README already discloses that the switch to argmax-added APS was post-hoc. Report it as post-hoc here as well, and always show pre-registered LAC alongside.

### 0.3 Leakage gate (blocks all other tests)

These run first. If any fail, no downstream result is reportable.

| Test | Pass condition |
|---|---|
| Patient/slide-level split integrity | No patient or slide appears in more than one partition (where BACH metadata allows) |
| Near-duplicate scan | No train/test pair with perceptual-hash distance below threshold or embedding cosine above 0.98 |
| Label-shuffle control | Balanced accuracy within the 95% null interval of chance (≈ 0.25) for all arms |
| Calibration/test disjointness | Conformal calibration set never overlaps the evaluation set |
| Synthetic-data guard | Any path containing `data_synthetic` or `artifacts/qa` aborts the run |

### 0.4 Statistical conventions

- All metrics reported with **95% CIs** (bootstrap, 10,000 resamples, stratified by class).
- Model comparisons are **paired** on identical images.
- Repeated stratified k-fold (5×10) rather than a single split. BACH is small, so single-split numbers are not trustworthy.
- Log commit hash, seeds, artifact SHA-256 checksums, and package versions with every run.

### 0.5 Repository layout

```
tests/validation/
├── PREREG.md
├── conftest.py                     # fixtures: embeddings, splits, arm registry
├── gates/
│   └── test_leakage.py
├── s1_headtohead/
│   ├── test_linear_probe.py
│   ├── test_zeroshot.py
│   └── test_label_efficiency.py
├── s2_calibration/
│   └── test_calibration_selective.py
├── s3_conformal/
│   └── test_coverage_efficiency.py
├── s4_robustness/
│   ├── test_perturbations.py
│   └── test_external_shift.py
├── s5_safety/
│   └── test_operating_point_errors.py
├── s6_timesavings/
│   ├── workflow_sim.py
│   ├── params.yaml
│   └── test_time_savings.py
├── validation_io.py                # RunContext: write_csv, save_jpg, manifest, chart helpers
├── test_outputs_schema.py          # asserts every expected CSV/JPG exists with required columns
└── report/
    └── build_report.py             # emits docs/VALIDATION.md from the CSVs

results/<run_id>/                   # gitignored
├── csv/                            # all numeric outputs
├── charts/                         # all figures, .jpg
└── manifest.csv                    # file, kind, rows, bytes, sha256
```

Make targets: `make validate` (all), `make validate-gates`, `make validate-s1` … `make validate-s6`.

---

## Strategy 1: Frozen-protocol head-to-head accuracy

**Question:** Does UNI2-h match or beat CONCH and MUSK on BACH under an identical protocol?

**Hypotheses**
- H1a (non-inferiority): PathQueue balanced accuracy ≥ best comparator − Δ.
- H1b (superiority): PathQueue's paired CI for the difference excludes 0.

**Protocol**
1. Extract embeddings for every arm using the same tile set and preprocessing (resize and normalization as each model requires, documented per arm).
2. Train the same logistic-regression probe per arm with the same C-grid, selected by nested CV.
3. Evaluate on identical held-out folds (5×10 repeated stratified).
4. Run zero-shot arms (CONCH, MUSK) with a fixed, pre-registered prompt set. Test prompt sensitivity with 3 prompt templates and report the range, not the best.
5. **Label-efficiency curve:** retrain probes with 5, 10, 25, 50, 100% of training labels. A model that wins only at full data has a different story than one that wins at low data.

**Metrics**
- Balanced accuracy, macro-F1, per-class recall, macro AUROC (one-vs-rest)
- Paired bootstrap CI on the difference vs. each comparator
- McNemar's test on per-image correctness; DeLong for AUROC
- Area under the label-efficiency curve

**Pass criteria**
- H1a satisfied against both comparators after Holm correction → "non-inferior"
- H1b satisfied against a comparator → "better than that comparator." If not, state "no demonstrated difference" and do not claim advantage.

**Implements:** `s1_headtohead/*`

---

## Strategy 2: Calibration and selective-prediction quality

**Question:** When PathQueue says it is 90% confident, is it right about 90% of the time, and does that beat CONCH/MUSK under the same calibration procedure?

**Protocol**
1. Apply the identical calibration method (e.g., temperature scaling fit on a held-out calibration fold) to every arm.
2. Compute reliability diagrams and scalar scores on the test fold, before and after calibration.
3. Build **risk-coverage curves**: sort test images by confidence, abstain on the least confident, and measure error among those retained.

**Metrics**
- ECE (15 bins, plus adaptive bins), NLL, multiclass Brier score
- Per-class calibration for invasive
- AURC (area under risk-coverage curve) and error at 80/90/95% coverage

**Pass criteria**
- Calibration reduces ECE and NLL relative to uncalibrated for PathQueue (internal sanity)
- PathQueue AURC is lower than CONCH/MUSK with a paired-bootstrap CI excluding 0, or reported as tied
- Confidence ranks errors above correct predictions: AUROC of confidence as an error detector ≥ 0.7

**Why this matters:** the worklist tiers depend on confidence. A better encoder with worse calibration can be a *worse* triage system.

**Implements:** `s2_calibration/*`

---

## Strategy 3: Conformal validity and efficiency per encoder

**Question:** Are PathQueue's conformal sets valid, and are they *smaller* (more informative) than those from CONCH/MUSK at the same coverage?

**Protocol**
1. For each arm, run **1,000+ random calibration/test partitions** with the same α.
2. Compute empirical coverage per partition. Examine the distribution, not just the mean.
3. Run APS, LAC, and the argmax-added APS variant for every arm.
4. Compute class-conditional coverage (especially invasive).

**Metrics**
- Marginal coverage: mean and 5th percentile across partitions
- Class-conditional coverage per class
- Mean set size, singleton rate, empty-set rate (before argmax fix), full-set rate
- Fraction of invasive cases whose set contains invasive (the tier-2 trigger)

**Pass criteria**
- **Validity:** mean coverage ≥ 1 − α − 0.01 for every arm and method (validity is a property of the procedure, so failure here is a bug, not a comparison loss)
- Argmax-added variant coverage ≥ plain APS coverage in every partition (it can only add labels)
- **Comparative advantage:** PathQueue has smaller mean set size or higher singleton rate than both comparators at equal coverage, with paired CI excluding 0
- Class-conditional invasive coverage reported with CI; flag if below 1 − α even when marginal passes

**Implements:** `s3_conformal/*`

---

## Strategy 4: Robustness and distribution shift

**Question:** Does PathQueue degrade more gracefully than CONCH/MUSK, and how much does the conformal guarantee erode?

**Protocol**
1. **Synthetic perturbations** applied to test images only (never train/calibration), at 5 severities each:
   - Gaussian blur, JPEG compression, additive noise, brightness/contrast shift
   - Stain jitter (H&E augmentation) and stain-normalization mismatch
2. Re-embed perturbed images per arm, re-run the frozen probe, calibration, and conformal procedure. **No refitting.**
3. **External dataset (if licensing permits):** run the frozen pipeline on an independent breast set (e.g., BRACS) with label mapping documented. NCT-CRC-HE-100K stays a cross-organ sanity check and is not counted as breast validation.

**Metrics**
- Balanced accuracy and invasive recall versus severity (slope of degradation)
- Coverage versus severity (quantifies the README's caveat that degraded images do not inherit the clean guarantee)
- Set-size inflation versus severity (does uncertainty rise appropriately?)
- External set: drop in balanced accuracy and coverage versus in-distribution

**Pass criteria**
- PathQueue's degradation slope is not steeper than the best comparator's (paired over perturbation types)
- Set size increases monotonically with severity (the system "knows" it is unsure)
- External-set drop is reported in full even if large. A big drop is a finding, not a failure to hide.

**Implements:** `s4_robustness/*`

---

## Strategy 5: Safety-critical operating-point and discordance analysis

**Question:** At a clinically sensible operating point, which system misses fewer invasive cancers, and *where* do the systems disagree?

**Protocol**
1. **Fixed-sensitivity comparison:** for each arm, set the threshold so invasive recall = 95% on calibration data, then evaluate specificity and workload (flag rate) on test data.
2. **Tier-level audit for PathQueue:** after assigning tiers 2 / likely invasive / low-confidence / unflagged, tabulate true class by tier.
3. **Discordance analysis:** collect images where PathQueue is correct and CONCH or MUSK is wrong (and vice versa). Review a sample of each using the existing human/AI spot-check workflow.
4. **Complementarity check:** does a simple CONCH+UNI2-h or MUSK+UNI2-h ensemble beat PathQueue alone? If it does, say so.

**Metrics**
- Specificity and flag rate at 95% invasive recall
- **False-reassurance rate:** true invasive cases in the unflagged tier (primary safety number)
- Confusion matrix (in situ ↔ invasive is the clinically important boundary)
- Discordant-pair counts with McNemar p-value
- Inter-rater agreement (Cohen's κ) between human and Claude spot checks on discordant cases

**Pass criteria**
- False-reassurance rate for PathQueue ≤ comparators' and ideally 0 on the test set, with a CI (rule of three: 0 of n gives an upper bound ≈ 3/n)
- Every discordant-case category has a documented hypothesis (e.g., stain artifact, ambiguous border) rather than being left as an unexplained residual
- If human and AI spot checks disagree (κ < 0.4), report AI review as unreliable for that category

**Implements:** `s5_safety/*`

---

## Strategy 6: Time savings in a simulated diagnostic pathology workflow

**Question:** Under a realistic arrival process, how much sooner do high-priority cases get reviewed, and how much pathologist time does confidence-aware triage save compared with standard ordering and with CONCH/MUSK-driven ordering?

**Important framing:** the README states that frozen sections, arrivals, margins, and operating-room assignments are simulated. This strategy validates **ordering logic under stated assumptions**. It does not demonstrate clinical time savings. All timing parameters are assumptions and must be labeled as such in every output.

### 6.1 Simulator design

A discrete-event simulation (`workflow_sim.py`, SimPy or hand-rolled) with:

- **Arrivals:** Poisson process, with rate varied across low / normal / surge load. Each arrival is a real BACH test case with its true label and each arm's prediction attached.
- **Servers:** 1 to 3 pathologists, each with review-time distribution by case type.
- **Frozen-section branch:** surgical cases trigger a margin call with a turnaround deadline, and an operating room waits on the result.
- **Interruptions:** a case in review is not preempted; a new surgical case jumps the queue only at the next free slot.

### 6.2 Queue policies compared

| Policy | Ordering rule |
|---|---|
| FIFO | Arrival order (standard baseline) |
| Random | Seeded shuffle |
| **PathQueue** | Surgical → likely invasive → low confidence → unflagged (UNI2-h + APS tiers) |
| CONCH-driven | Same tier logic fed by CONCH-probe predictions |
| MUSK-driven | Same tier logic fed by MUSK-probe predictions |
| Oracle | True labels (upper bound) |

### 6.3 Parameters (`params.yaml`)

All values below are placeholders to be replaced with cited literature values or clinician-interview estimates. Each must carry a source field or the flag `assumption: true`.

| Parameter | Baseline | Sensitivity range |
|---|---|---|
| Review time, unflagged benign/normal (min) | 5 | 3–8 |
| Review time, invasive / in situ (min) | 12 | 8–20 |
| Review time reduction when AI evidence is shown | 0% | 0–25% |
| Pathologists | 2 | 1–3 |
| Arrival rate (cases/hour) | 6 | 3–12 |
| Frozen-section turnaround target (min) | 20 | 15–30 |

The baseline for "AI-assisted review time reduction" is **0%**. Report the pure *reordering* benefit first, and show assistance effects only as a separate sensitivity layer.

### 6.4 Experiments

1. **Main run:** 500 seeded replications per policy per load level, same arrival sequences across policies (common random numbers) so differences are paired.
2. **Sensitivity sweep:** one-at-a-time and Latin-hypercube sweeps over the ranges above.
3. **Error injection:** run each AI-driven policy with its *actual* predictions, so misclassified invasive cases are penalized realistically (a mis-tiered case waits longer).
4. **Break-even analysis:** at what false-reassurance rate does PathQueue stop beating FIFO?

### 6.5 Metrics

- **Time-to-review of true invasive cases:** mean, median, and 90th percentile (minutes)
- **Time-to-margin-call** for frozen-section cases and fraction meeting the turnaround target
- **OR idle time** waiting for margin calls (minutes per case)
- **Total queue wait** and fraction of cases waiting more than 30 minutes
- **Pathologist-minutes saved** (only meaningful under the assistance sensitivity layer)
- **Cost of being wrong:** additional wait for mis-tiered invasive cases versus FIFO

### 6.6 Pass criteria

- PathQueue reduces mean and p90 time-to-review of invasive cases versus FIFO and Random, paired CI excluding 0, at normal and surge load
- PathQueue is within the oracle-to-FIFO gap by a reported percentage ("closes X% of the gap")
- Versus CONCH-driven and MUSK-driven policies: report the difference with CI. Any advantage should trace back to Strategy 3's set-size and Strategy 5's false-reassurance results; if it does not, treat it as suspect.
- Benefit shrinks toward zero at low load (idle pathologists make ordering irrelevant). If it does not, the simulator has a bug.
- Results include every parameter value used, and the report states: *"Simulated; not a clinical time-savings claim."*

**Implements:** `s6_timesavings/*`

---

## 7. Summary matrix

| Strategy | Primary question | Key metric | Comparative claim supported |
|---|---|---|---|
| Gate | Is the data clean? | Leakage/shuffle checks | None; blocks all others |
| S1 | Is it as accurate? | Balanced accuracy, paired CI | Accuracy parity or superiority |
| S2 | Is confidence trustworthy? | ECE, NLL, AURC | Better-calibrated uncertainty |
| S3 | Are uncertainty sets valid and tight? | Coverage, set size | Same validity, more informative sets |
| S4 | Does it hold under shift? | Degradation slope, external drop | Robustness |
| S5 | Does it miss fewer cancers? | False-reassurance rate | Safety at fixed sensitivity |
| S6 | Does it save time? | Time-to-review of invasive cases | Workflow benefit (simulated) |

## 8. Output artifacts: CSV tables and JPG charts

Every strategy writes its numbers to **CSV** and its figures to **JPG** through `validation_io.RunContext`. `build_report.py` reads only from these files, so every figure and claim in `docs/VALIDATION.md` can be traced to a CSV row.

### 8.1 Conventions

- **Location:** `results/<run_id>/csv/` and `results/<run_id>/charts/`. `results/` is gitignored, consistent with the README's rule against committing BACH-derived assets.
- **CSV:** UTF-8, tidy long format (one row per arm × metric × condition), floats to 6 decimals. `run_id`, `commit`, and `seed` are prepended to every file. Strategy 6 files also carry `simulated=True`.
- **JPG:** 200 dpi, quality 95, white background, a provenance footer (run, commit, seed), error bars or CI bands wherever a point estimate is plotted, and *n* in the axis label or caption. Strategy 6 charts are watermarked **SIMULATED - NOT A CLINICAL CLAIM**.
- **No placeholders:** an empty result raises an error instead of writing a file. Unavailable evaluations stay unavailable.
- **Synthetic guard:** `RunContext` refuses to start if the data root contains `data_synthetic` or `artifacts/qa`.
- **Manifest:** `manifest.csv` lists every file with row count, size, and SHA-256.

### 8.2 Files produced per strategy

| Strategy | CSV files (key columns) | JPG charts |
|---|---|---|
| **Gate** | `gates_leakage.csv` (`test, arm, statistic, threshold, passed`) | `gate_label_shuffle_null.jpg` |
| **S1** | `s1_metrics_per_fold.csv` (`arm, repeat, fold, balanced_acc, macro_f1, auroc_macro, recall_normal, recall_benign, recall_insitu, recall_invasive`)<br>`s1_paired_differences.csv` (`arm_a, arm_b, metric, diff, ci_low, ci_high, p_raw, p_holm, verdict`)<br>`s1_label_efficiency.csv` (`arm, train_frac, metric, mean, ci_low, ci_high`)<br>`s1_zeroshot_prompts.csv` (`arm, prompt_template, balanced_acc`) | `s1_balanced_accuracy_forest.jpg`<br>`s1_per_class_recall.jpg`<br>`s1_label_efficiency.jpg`<br>`s1_zeroshot_prompt_range.jpg` |
| **S2** | `s2_calibration_summary.csv` (`arm, stage, ece, ece_adaptive, nll, brier, ci_low, ci_high`)<br>`s2_reliability_bins.csv` (`arm, stage, bin, confidence, accuracy, count`)<br>`s2_risk_coverage.csv` (`arm, coverage, risk, ci_low, ci_high`) | `s2_reliability_diagrams.jpg`<br>`s2_risk_coverage.jpg`<br>`s2_ece_nll_comparison.jpg` |
| **S3** | `s3_coverage_partitions.csv` (`arm, method, partition, coverage, set_size, singleton, empty, full, invasive_coverage`)<br>`s3_coverage_summary.csv` (`arm, method, mean_cov, p05_cov, mean_set_size, singleton_rate, ci_low, ci_high`)<br>`s3_class_conditional.csv` (`arm, method, class, coverage, ci_low, ci_high`) | `s3_coverage_distribution.jpg`<br>`s3_set_size_comparison.jpg`<br>`s3_class_conditional_coverage.jpg` |
| **S4** | `s4_perturbation_results.csv` (`arm, perturbation, severity, balanced_acc, invasive_recall, coverage, set_size`)<br>`s4_external_shift.csv` (`arm, dataset, balanced_acc, invasive_recall, coverage, drop_vs_bach`) | `s4_accuracy_vs_severity.jpg`<br>`s4_coverage_vs_severity.jpg`<br>`s4_set_size_vs_severity.jpg`<br>`s4_external_drop.jpg` (only if an external set ran) |
| **S5** | `s5_operating_point.csv` (`arm, target_recall, threshold, specificity, flag_rate`)<br>`s5_tier_audit.csv` (`tier, true_class, n`)<br>`s5_false_reassurance.csv` (`arm, n_invasive, missed, rate, ci_low, ci_high`)<br>`s5_discordance.csv` (`case_id, true_label, pred_uni2h, pred_conch, pred_musk, category, spot_check_human, spot_check_ai`)<br>`s5_confusion_<arm>.csv` (4×4 counts) | `s5_confusion_<arm>.jpg` (one per arm)<br>`s5_false_reassurance.jpg`<br>`s5_tier_composition.jpg`<br>`s5_specificity_at_95_recall.jpg` |
| **S6** | `s6_replications.csv` (`policy, load, replication, mean_ttr_invasive, p90_ttr_invasive, ttm_met_frac, or_idle_min, mean_wait`)<br>`s6_summary.csv` (`policy, load, metric, mean, ci_low, ci_high, diff_vs_fifo`)<br>`s6_sensitivity.csv` (`parameter, value, policy, metric, mean`)<br>`s6_breakeven.csv` (`false_reassurance_rate, diff_vs_fifo, ci_low, ci_high`)<br>`s6_params_used.csv` (`parameter, value, source, assumption`) | `s6_time_to_review_box.jpg`<br>`s6_load_sweep.jpg`<br>`s6_sensitivity_tornado.jpg`<br>`s6_breakeven.jpg`<br>`s6_or_idle_time.jpg` |
| **Summary** | `headline_summary.csv` (`strategy, comparison, metric, estimate, ci_low, ci_high, verdict`)<br>`manifest.csv` | `headline_forest.jpg` (all primary comparisons on one chart) |

### 8.3 Example usage inside a strategy test

```python
from validation_io import RunContext, forest_plot, line_ci

ctx = RunContext(seed=SEED, data_root=DATA)           # simulated=True in Strategy 6

ctx.write_csv("s1_paired_differences", paired_df,
              required_cols=["arm_a", "arm_b", "metric", "diff", "ci_low", "ci_high"])
fig = forest_plot(paired_df, label="comparison", est="diff", lo="ci_low", hi="ci_high",
                  xlabel="Balanced accuracy difference vs. comparator (95% CI)")
ctx.save_jpg(fig, "s1_balanced_accuracy_forest",
             title="UNI2-h vs CONCH / MUSK (paired, n repeats x folds)")
ctx.write_manifest()
```

### 8.4 Output tests (`test_outputs_schema.py`)

- Every CSV listed in 8.2 for a strategy that ran exists, is non-empty, and contains its required columns.
- Every chart exists, is a valid JPEG, and exceeds a minimum pixel size (catches blank figures).
- `manifest.csv` checksums match the files on disk.
- Strategy 6 CSVs all contain `simulated=True`.
- No CSV contains `NaN` in an estimate or CI column unless the row's `verdict` is `unavailable`.

---

## 9. Reporting template

Each strategy writes to `docs/VALIDATION.md`, generated from the CSVs in section 8:

1. Hypothesis and pre-registered decision rule
2. Table of all arms, **including arms that beat PathQueue**
3. Point estimate, 95% CI, and n for every number, linked to its source CSV and chart
4. Verdict in one of four forms: *superior*, *non-inferior*, *no demonstrated difference*, *inferior*
5. Limitations: single-institution BACH, small test set, post-hoc argmax switch, simulated workflow parameters

## 10. Known limitations to state up front

- BACH is small and single-source, so all in-distribution numbers are optimistic.
- BACH labels are diagnostic categories, not Nottingham grades.
- Conformal coverage assumes exchangeability; perturbed and simulated data break it.
- The workflow simulation depends on assumed timings and demonstrates logic, not clinical impact.
- CONCH and MUSK are evaluated through the same frozen-probe protocol; their published numbers use different protocols and are not directly comparable.
- Model licensing and gated access apply to all checkpoints; do not publish embedding bundles.
