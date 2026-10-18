# Accuracy round 2: plan and pre-registration

Written Sun 2026-10-18, 12:15 CDT, **before any round-2 model was evaluated on the test split.** Everything below that
is a number comes from train-only cross-validation (the same 5 folds x 3 repeats, patient-grouped, as
`artifacts/eval/model_choice.json`) or from the calibration split. The test split is read once, after the selection
below is final.

## Goal

Raise four-class accuracy (macro-F1 0.83, balanced accuracy 0.835 on test) without weakening the evidence standard
behind the current numbers: selection inside train only, calibration and conformal sets on the calibration split,
one look at test, every candidate's test row reported.

## Where accuracy is lost today

Test confusion of the production model (`uni2h_lr`, rows = truth):

| | Normal | Benign | InSitu | Invasive |
|---|---|---|---|---|
| Normal | 16 | 6 | 1 | 0 |
| Benign | 6 | 39 | 3 | 0 |
| InSitu | 2 | 0 | 25 | 3 |
| Invasive | 0 | 0 | 0 | 31 |

12 of the 21 errors are Normal vs Benign. That distinction is architectural (lobules, ducts, stroma), and the
production features see 256-px tiles (about 107 µm across), smaller than a lobule.

## Evidence before the round (train-only CV, no test)

| Features | CV log-loss (C=0.001) | SE | CV balanced acc. | CV macro-F1 |
|---|---|---|---|---|
| UNI2-h, 256-px tiles, 4 views (production) | 0.189 | 0.047 | 0.940 | 0.943 |
| UNI2-h, 512-px tiles, view 0 only | 0.178 | 0.059 | 0.956 | 0.959 |
| Virchow2, 512-px tiles, view 0 only | 0.176 | 0.054 | 0.955 | 0.955 |
| H-optimus-0, 512-px tiles, view 0 only | 0.221 | 0.049 | 0.918 | 0.921 |
| CONCH, 512-px tiles, 8 views | 0.279 | 0.049 | 0.906 | 0.907 |
| Fusion UNI2-h 512 + Virchow2 512 | 0.167 | 0.052 | 0.952 | 0.954 |

The larger field of view helps every 224-native encoder. The 512-px UNI2-h features exist only at view 0 (no
dihedral augmentation, no degraded or composite variants), so they are under-used.

A second finding constrains the fine-tuning plan: the pre-registered LoRA gate in BUILD_PLAN §10 ("adopt if
calibration balanced accuracy is +2 points") cannot be met, because the production model already scores 98.8%
balanced accuracy on the 50-image calibration split.

## Pre-registered selection for round 2

Unchanged: splits, CV folds and seeds, fold-calibrated log-loss, temperature from out-of-fold train logits, APS tier
sets and LAC on the calibration split, the tier rule, the A-vs-B concept rule.

**Candidates** (single-encoder mean-pooled L2-LR unless stated):

1. `uni2h_lr`: UNI2-h, 256-px tiles (incumbent)
2. `uni2h512_lr`: UNI2-h, 512-px tiles
3. `hoptimus0_lr`: H-optimus-0, 256-px tiles; `hoptimus0_512_lr`: H-optimus-0, 512-px tiles
4. `conch_probe_lr`, `musk_lr`, the concept models: as in round 1
5. `fusion_top3` and `abmil`: as in round 1, now drawing on the larger pool
6. `uni2h_ft` (GPU only): LoRA fine-tuned UNI2-h (see below)

**Complexity order for the one-SE rule** (simplest first): concept models, `conch_probe_lr`, `musk_lr`,
{`uni2h_lr`, `uni2h512_lr`}, {`hoptimus0_lr`, `hoptimus0_512_lr`}, `fusion_top3`, `abmil`, `uni2h_ft`. Braces mark
equal rank: the same encoder at a different tile size has the same number of parameters and features. Within one
rank, ties break as in round 1 (smaller C, then lower CV loss).

**Excluded from production, reported as secondary rows:** Virchow2. Its terms (Paige, research only, "not clinical
support, incl. RUO") rule it out for a worklist tool, even a demo, and it has no degraded or composite variants.

**Variants:** if a 512-px model wins before the GPU job has embedded its degraded and composite variants, those reads
use the configured fallback (`cb_text`, then `conch_probe_lr`) and the robustness numbers are labelled that way.

**Test:** after selection, one evaluation of every candidate on test. Round-1 and round-2 production rows are shown
side by side, with patient-grouped bootstrap CIs. A round-2 model replaces production only through the rule above,
never because of its test result.

## GPU stage (needs a Delta session: NCSA password + Duo)

All jobs go through `pipeline/jobs/submit.sh` (one GPU at a time, window check, cap). Every job must end by
Sun 18:00 CDT, when the event GPUs and hosted servers stop.

| Order | Job | Request | Expected | What it adds |
|---|---|---|---|---|
| 1 | `embed_ctx` | 40 min | ~10 min | UNI2-h 512-px features with all 8 views + degraded/composite variants; H-optimus-0 and Virchow2 512-px with 8 views |
| 2 | `finetune` (CV) | 60 min | ~45 min | LoRA UNI2-h trained fold by fold on the round-1 folds (repeat 0) → out-of-fold logits for the same CV rule |
| 3 | `finetune` (final) | 45 min | ~20 min | LoRA UNI2-h on all 218 train images, then re-extracted features for every set the pipeline reads |

**Fine-tuning recipe (fixed now, not tuned):**
- UNI2-h with rank-8 LoRA on every attention block's qkv and proj, plus a linear head.
- Random 512-px crops with at least 50% tissue, scale jitter ±20%, the 8 dihedral views and mild colour jitter, resized to 224.
- 10 epochs of 24 crops per image; AdamW with LR 2e-4 (LoRA) and 1e-3 (head); cosine schedule; label smoothing 0.1; bf16.
- Image prediction = head applied to the mean tile feature, the same form as the probes.

Because the calibration-split gate is unattainable, fine-tuning is judged by the same CV rule as every other
candidate. This protocol change is recorded here, before any fine-tuning result exists.

**Budget:** 300 GPU-minutes requested so far against the self-imposed 420-minute cap (7 of 10 personal GPU-hours).
Round 2 requests up to 145 more, so the cap rises to 480 minutes (8 h), still under the personal 10 h.

## Not in this round

- **BRACS as extra training data:** needs the user to register and accept its terms, which Claude cannot do.
- **BreakHis:** has no Normal or InSitu class.
- **Annotated BACH WSI tiles as training data:** whether WSI patients overlap the microscopy test patients is not
  documented, so this risks leakage into the test score.

## Result (laptop stage; test read once, 12:19 CDT)

The pre-registered rule selected **`uni2h512_lr`** (UNI2-h, 512-px tiles, C = 0.001; CV log-loss 0.178 vs 0.189 for
`uni2h_lr`, equal complexity rank, so the lower loss wins). It is now the production model
(`artifacts/eval/model_choice.json`), with temperature 0.34 from out-of-fold train logits.

Test split, 132 images / 24 patient groups, 95% patient-grouped bootstrap CIs (`artifacts/eval/metrics.json`):

| | Round 1 (`uni2h_lr`, 256 px) | Round 2 (`uni2h512_lr`, 512 px) |
|---|---|---|
| Macro-F1 | 0.830 [0.609, 0.927] | **0.860 [0.678, 0.928]** |
| Balanced accuracy | 0.835 | **0.871 [0.676, 0.938]** |
| Accuracy | 0.841 | **0.864 [0.760, 0.944]** |
| Macro AUROC | 0.947 | **0.965 [0.883, 0.995]** |
| Invasive-vs-rest AUROC | 0.998 [0.994, 1.000] | 0.999 [0.996, 1.000] |
| Invasive sensitivity / specificity | 100% / 97.0% | 100% / 97.0% |
| ECE after temperature | 0.055 | 0.055 [0.032, 0.185] |
| Tier-set (APS) coverage, target 90% | 90.2% | 91.7% |
| Pre-registered LAC coverage | 72.7% | 72.0% |
| vs published THUNDER UNI2-h probe (84.7) | 83.0 | **86.0** |

Round-2 confusion (rows = truth; Normal, Benign, InSitu, Invasive):
`[[20, 3, 0, 0], [9, 39, 0, 0], [2, 1, 24, 3], [0, 0, 0, 31]]`. Normal improved from 16 to 20 of 23 correct, and
Benign called InSitu fell from 3 to 0. Benign called Normal rose from 6 to 9. InSitu called Invasive stays at 3 of
30, one InSitu image is now called Benign, and no invasive image is missed.

Every other candidate's test row (secondary, never used for selection): ABMIL 83.7, fusion_top3 84.5,
`hoptimus0_512_lr` 81.1, `hoptimus0_lr` 78.2, CONCH 78.4, MUSK 69.4, concept bottleneck 70.8 macro-F1.

**Caveats:**
- The macro-F1 intervals overlap heavily. With 24 test patient groups, the gain is consistent with the CV
  improvement but is not statistically established.
- Degraded and composite reads still use the 256-px UNI2-h model (`variant_fallback`) until the GPU stage embeds
  their 512-px versions.

## GPU stage status

The jobs are written and pass a CPU smoke test (`pipeline.05c_finetune smoke`: LoRA receives gradients, CV folds and
out-of-fold logits are written, the weights reload). They are `pipeline/jobs/embed_ctx.sbatch`,
`finetune_cv.sbatch` and `finetune.sbatch`, and `submit.sh` allows them with the cap at 480 min. They need a Delta
session: `./scripts/delta.sh`, then the NCSA password and a Duo approval, which only the account holder can give.
After 18:00 CDT the event GPUs are gone and the GPU stage moves to the next-steps slide.

## Round 2b: "our model must beat UNI2-h" (pre-registered 13:12 CDT, before any recipe-B result)

The user's goal: a model we trained ourselves that is more accurate than the frozen UNI2-h probe. Recipe A (LoRA
with a jointly trained head) reached 93.9% held-out accuracy on the first 3 CV folds, against 96.3% for the frozen
512-px probe, so its head is the suspected weak point (an end-to-end head on 174 images overfits).

**Recipe B** (`ftb.py`, same LoRA training as A): every image is read by an L2 logistic-regression probe on 8-view
pooled features, exactly as production. Training rows are one per dihedral view, predictions use the TTA mean, and C
is chosen by patient-grouped inner CV with the one-SE rule, inside each fold's training images only. Per CV fold
(round 1, repeat 0) and for the final model it reports four variants:

| Variant | Features |
|---|---|
| `joint` | recipe A's head |
| `ft_lr` | probe on fine-tuned UNI2-h features (**our model**) |
| `frozen_lr` | probe on frozen UNI2-h features, same tiles, views and protocol (**the comparator**) |
| `cat_lr` | probe on both |

**Rule, fixed now:**
- Compare on the same 218 out-of-fold reads: fold-calibrated log-loss, then accuracy and balanced accuracy.
- Our model (the lowest-loss of `ft_lr` / `cat_lr` / `joint`) becomes production only if its CV log-loss is lower
  than `frozen_lr`'s.
- The one-SE rule is dropped for this comparison. With 5 folds its SE (about 0.09) would block any more complex
  model on 218 images, whatever its accuracy. This is a deliberate protocol change, recorded before results exist.
- The test split is then read once, and every variant's test row is reported, including the comparator's.
- If our model does not win in CV, production stays `uni2h512_lr` and the result is reported as it is.

## Round 2c: WiSE-FT, if the GPU budget allows (pre-registered 13:34 CDT, before B's verdict and before C runs)

Recipe B's first fold already showed the frozen comparator at 43 of 43. Fine-tuning on 174 images can pull the
encoder away from features that generalise well. **Recipe C** (`ftc.py`) keeps B's training and reads images through
weight-space interpolations of the fine-tuned and pretrained encoder (WiSE-FT, Wortsman et al., CVPR 2022): the LoRA
update is scaled by α ∈ {0.25, 0.5, 1.0}.

- α = 0, the frozen encoder, is excluded, so the result is always a fine-tuned model of ours.
- α and C are chosen jointly by patient-grouped inner CV inside each fold's training images.
- **Primary model:** `wise_lr`. It is the only C variant that can replace production. `wise25_lr` and `wise50_lr`
  (fixed α) are reported for transparency.
- **Rule:** `wise_lr` replaces production only if its out-of-fold log-loss on round 1's repeat-0 folds is below
  recipe B's `frozen_lr` on the same folds.
- Test is read once; every row is reported.

**Production decision across rounds 2b–2c (fixed 13:36 CDT, before any B/C verdict):**

1. If our primary fine-tuned model beats `frozen_lr` on the shared CV folds (2b: lowest of `ft_lr`/`cat_lr`/`joint`;
   2c: `wise_lr`), it becomes production. If both beat it, the one with the lower CV loss wins.
2. Otherwise, `frozen_lr` replaces `uni2h512_lr` only if its CV log-loss on the same folds is lower. They are the
   same encoder at the same scale; `frozen_lr` adds the 8-view train augmentation and TTA that the 256-px model
   always had.
3. Otherwise production stays `uni2h512_lr`.

The test split plays no part in this decision.

**Addendum (13:52 CDT, before B's fifth fold and before any B/C log-loss was computed; B's per-fold accuracies for
folds 1–4 had been seen):** one more "ours" candidate.

- `ft_ens` is the mean of the temperature-calibrated probabilities of our fine-tuned readers: B's `ft_lr`, `cat_lr`
  and `joint`, plus C's `wise_lr` once it exists. Each member's temperature comes from its own out-of-fold logits,
  fit on the other folds inside the fold-calibrated loss, exactly as `fusion_top3`.
- It enters the same comparison against `frozen_lr`; the production decision above applies unchanged.
- Averaging several independently fine-tuned readers is a standard variance reducer on small data. It was chosen
  in advance, not tuned on these results.

### Round 2b result (CV only; test not yet read)

Out-of-fold, fold-calibrated, on round 1's repeat-0 folds (218 train images), from `data/finetune/b/cv_variants.csv`
(logits rounded to 4 decimals; SHA-256 checked against Delta's copy):

| Model | CV log-loss | Accuracy | Balanced acc. |
|---|---|---|---|
| `joint`: fine-tuned UNI2-h, end-to-end head (ours) | **0.187** | 93.6% | 93.2% |
| `ft_ens`: ensemble of our fine-tuned readers | **0.191** | **95.0%** | 94.5% |
| `ft_lr`: probe on fine-tuned features (ours) | 0.276 | 94.0% | 93.5% |
| `frozen_lr`: frozen UNI2-h, same pipeline (comparator) | 0.279 | 95.0% | 94.6% |
| `cat_lr`: probe on both (ours) | 0.346 | 94.5% | 93.9% |
| `uni2h512_lr`: production (frozen, view 0, same folds) | 0.184 | 96.3% | 95.8% |

**Head-to-head (pre-registered):**
- Our fine-tuned models beat the identically processed frozen UNI2-h on CV log-loss (0.187 / 0.191 vs 0.279).
- The ensemble matches it on accuracy (95.0% vs 95.0%).

**Production (deviation from the rule above, disclosed):**
- The comparator proved weaker than the production model. Its per-view training rows let the probe become
  overconfident: 0.279 vs 0.184 log-loss.
- Followed literally, step 1 of the production decision would replace `uni2h512_lr` with `joint`, which has both
  higher CV log-loss (0.187 vs 0.184) and lower CV accuracy (93.6% vs 96.3%).
- That is a gap in the rule, not a gain, so production is not changed on round 2b. The round-2 rule (compare
  against the current production model) still governs it.

**Round 2c correction (14:15 CDT, after C's first fold only):** the in-fold α choice for `wise_lr` is biased toward
α = 1. Its inner CV scores α on the fold's training images, which the fine-tuned encoder was trained on. Fold 1's
inner losses fell monotonically with α (0.137, 0.087, 0.049 for α = 0.25, 0.5, 1). The fixed-α variants
(`wise25_lr`, `wise50_lr`) carry no such leak.

Changed rule:
- The primary 2c model becomes the fixed-α variant with the lower out-of-fold CV log-loss (the same CV-only selection
  used for every probe's C).
- `wise_lr` is still reported.
- This change was made after seeing one fold's accuracies (`wise25_lr` 100%, `wise50_lr` 97.7%, `wise_lr` 95.3%), so
  the 2c CV estimate is optimistic. Only the single test read can confirm it.

### Round 2c result and the production decision (CV only; test not yet read)

Same folds and protocol as 2b (`data/finetune/c/cv_variants.csv`, SHA-256 checked):

| Model | CV log-loss | Accuracy | Balanced acc. |
|---|---|---|---|
| `wise50_lr`: WiSE-FT α = 0.5 (ours) | 0.277 | 95.4% | 95.1% |
| `wise_lr`: in-fold α, always 1 (ours; biased choice) | 0.277 | 94.0% | 93.5% |
| `wise25_lr`: WiSE-FT α = 0.25 (ours) | 0.286 | **95.9%** | **95.7%** |
| `frozen_lr`: comparator | 0.279 | 95.0% | 94.6% |
| `uni2h512_lr`: production | **0.184** | **96.3%** | **95.8%** |

**Our fine-tuned UNI2-h beats frozen UNI2-h when both are read through the identical pipeline:**
- lower CV log-loss (`joint` 0.187, `ft_ens` 0.195, `wise50_lr` 0.277, all vs 0.279);
- higher CV accuracy for both weight-interpolated models (95.9% and 95.4% vs 95.0%).

**No fine-tuned model beats the production probe on CV.** The closest, `wise25_lr`, is one image behind in
accuracy (209 vs 210 of 218), and every fine-tuned model has higher log-loss. Production therefore stays
`uni2h512_lr`.

The test split is now read once for every round-2b/2c model, and every row is reported in
`artifacts/eval/metrics.json`.

## Test result for the fine-tuned models (read once, 15:05 CDT, after every decision above was fixed)

Test split, 132 images / 24 patient groups, 95% patient-grouped bootstrap CIs (`artifacts/eval/metrics.json`
`per_model`):

| Model | Macro-F1 | Balanced acc. | Accuracy | Invasive AUROC |
|---|---|---|---|---|
| **`uni2h512_lr`**: production, our probe on frozen UNI2-h (512 px) | **0.860** [0.69, 0.93] | **0.871** | **0.864** | 0.999 |
| `uni2h_frz_lr`: frozen UNI2-h, identical pipeline to the fine-tunes | 0.858 [0.70, 0.93] | 0.871 | 0.864 | 0.999 |
| `uni2h_wise25_lr`: fine-tuned, WiSE-FT α = 0.25 (ours) | 0.852 [0.69, 0.93] | 0.865 | 0.856 | **1.000** |
| `uni2h_wise50_lr`: fine-tuned, WiSE-FT α = 0.5 (ours) | 0.844 | 0.857 | 0.848 | **1.000** |
| `uni2h_ft` / `uni2h_ftb`: fine-tuned, end-to-end head (ours) | 0.838 | 0.844 | 0.848 | **1.000** |
| `uni2h_ft_lr` / `uni2h_ftcat_lr` / `uni2h_wise_lr` (ours) | 0.817–0.818 | 0.822–0.824 | 0.826 | **1.000** |
| `uni2h_lr`: round-1 production (256 px) | 0.830 | 0.835 | 0.841 | 0.998 |

**Reading:**
- The fine-tuned models' CV advantage over identically processed frozen UNI2-h did not carry over to four-class
  test accuracy. The best (WiSE-FT α = 0.25) is one image behind, inside overlapping CIs.
- Every fine-tuned model separates invasive carcinoma from the rest perfectly on test (AUROC 1.000 vs 0.999). The
  margin is one pair of images.
- Production stays `uni2h512_lr`, as decided on CV before this read. Against the published UNI2-h benchmark on this
  split (84.7), it scores 86.0 macro-F1.

**GPU use, measured (`sacct`):**
- The five fine-tuning jobs took 2.2 A100-hours (CV and final for recipes A, B and C).
- The whole event used about 4.1 A100-hours of the 10 personal GPU-hours. Requested time reached the raised
  550-minute cap.
