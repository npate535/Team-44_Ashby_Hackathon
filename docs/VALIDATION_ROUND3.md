# Validation round 3: pre-registration

Written Sun 2026-10-18, 14:10 CDT, **before any analysis below was run.** Nothing here depends on a result it
describes. Every statistic listed is reported whatever its value, and no model, threshold or tier rule changes as a
result of these checks.

All three analyses read the production model from `artifacts/eval/model_choice.json` at run time, so they apply
unchanged if a later round replaces the production model.

## 1. Conformal coverage across random re-splits

**Why.** The APS set that drives the worklist tiers (APS with the top class added to empty sets) was chosen after
viewing test coverage on the single THUNDER split: 91.7% against the 90% target, while the pre-registered LAC covered
72.0% with many empty sets. This check measures how each method's coverage varies across many calibration/test
partitions of the same held-out images, with the methods fixed in advance.

- **Model.** The production model and its fixed temperature (fit on training out-of-fold logits). Calibrated
  probabilities come from `data/predictions.csv` (`pcal__*`). Nothing is refit.
- **Images.** The 182 held-out clean images (50 calibration + 132 test). Training images are never used.
- **Partitions.** 1,000 random partitions, seed 20261018. Patient groups (`group` in `data/cases.csv`) stay whole:
  groups are shuffled and added to the calibration part until it holds at least 50 images; the remaining groups form
  the test part.
- **Methods.** The four methods in `artifacts/eval/conformal.json`, with the same α = 0.1 and settings: LAC
  (marginal), APS, Mondrian (class-wise LAC), and APS never empty (the tier set).
- **Statistics, per method.** Test coverage (mean, and the 2.5th, 50th and 97.5th percentiles across partitions);
  share of partitions with coverage ≥ 90%; mean set size; share of empty sets; share of invasive test images with
  Invasive in the set; share of non-invasive test images with Invasive in the set.
- **Reference.** The same statistics on the original THUNDER partition, reported alongside.

## 2. Deferral (risk–coverage) curve

**Why.** To measure how accuracy changes on the reads Sepia keeps when the least reliable reads are deferred to closer
review first.

- **Primary ranking.** The reliability score in `data/reliability.csv`; the lowest score is deferred first, ties
  broken by read ID.
- **Secondary rankings, reported alongside.** Calibrated top-class probability (lowest first); conformal set size
  (largest first, ties by top-class probability); random deferral (mean over 1,000 random orders, seed 20261018).
- **Populations.** (a) The 132 clean test images. (b) All test reads: clean plus the simulated degraded variants
  (blur, JPEG, stain shift, resampling), excluding composites. Each read is scored with the model that produced it in
  `data/predictions.csv`, as in the app.
- **Deferral fractions.** 0% to 50% in steps of 5%.
- **Statistics at each fraction.** Four-class accuracy on kept reads; share of all errors that were deferred; count
  of invasive reads called non-invasive among kept reads; count of in situ reads called invasive among kept reads.
- **Headline, fixed now.** Accuracy on kept reads at 20% deferral in population (b), with a 95% patient-grouped
  bootstrap interval (2,000 resamples), against 0% deferral and against random deferral at 20%.

## 3. Exact region contributions

**Why.** The case heatmap comes from a separate tile-level probe, and the "Why" rows come from the concept model, not
from the production read. In the deletion test, removing the highlighted regions barely changed P(Invasive): risk
regions 0.025 more than random regions [0.001, 0.052], concept regions indistinguishable from random.

- **Method.** The production read is a multinomial logistic regression on the mean of standardized tile embeddings.
  Its calibrated invasive logit, centered on the mean over classes, therefore equals a constant plus one additive
  term per tile. Those terms, summed within each displayed region, are the region contributions.
- **Acceptance criterion.** For every read, the constant plus the sum of contributions reproduces the centered logit
  computed from the model's own probabilities to within 1e-6.
- **Deletion test, added arm.** Remove the k regions with the largest contributions (k as in the existing test),
  re-pool, re-predict. Compare with removing k random regions and the k highest tile-probe regions. Report the change
  in P(Invasive) and in the centered logit, with 95% patient-grouped bootstrap intervals, on the populations the
  existing test uses.
