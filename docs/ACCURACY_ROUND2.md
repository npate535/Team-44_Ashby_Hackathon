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
