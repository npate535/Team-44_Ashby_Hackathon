# External test on TCGA-BRCA, round 2: production model (pre-registration)

Written and committed on 2026-10-18, before the production model is run on any TCGA crop in this round.
The decisions below are fixed. Anything that proves impossible is recorded in a dated "Deviations" section at the end;
nothing above that section changes after results are seen.

## Why a second round

Round 1 (`eval/external_tcga.py` → `artifacts/eval/external_tcga.json`, kept unchanged) answered a different question:

- It scored the LoRA fine-tune `uni2h_ft`, not the production model.
- It used 8 patients, while the protocol (`TCGA_TODO.md`) specified 25.
- It built conformal sets with deterministic APS (U = 1) against a q-hat fit with randomized APS.

Round 2 tests the production model with exactly its BACH calibration/test recipe, on the protocol's 25 patients. The
round-1 crops are also scored, so the two models can be compared on identical crops.

## Model and recipe

Every item was checked against the code and the stored artifacts on 2026-10-18. The run re-checks the starred items
and aborts on a mismatch.

| Item | Value | Source |
|---|---|---|
| Model ID* | read at run time from `artifacts/eval/model_choice.json` `production` (today `uni2h512_lr`); must be a linear model on `uni2h` / `bach512` / `probe` features | `pipeline/05_risk_model.py`, `config/settings.yaml` `risk_model.embedding_candidates` |
| Encoder | UNI2-h (`MahmoodLab/UNI2-h`, timm, revision `d517a8dd47902dd7c308b3c36f63bce47e7b9a43`), frozen, class token, 1536-d | `config/encoders.yaml`, `data/emb/uni2h/meta.json` |
| Preprocessing | 512-px region → bilinear resize to 224 → ImageNet mean/std; bf16 autocast (the encoder spec) | `pipeline/encoders.py` `Preprocess`, `Encoder.embed` |
| Grid | `g512`: 4 × 3 regions of 512 px on the 2048 × 1536 crop | `settings.tiles.grids.g512` |
| Views* | **1 (view 0, identity)** | see "Views" below |
| Stored precision | each region embedding rounded to float16 before pooling, as `02_embed` stores it | `pipeline/02_embed.py` |
| Tissue QC | the `01q` rule BACH QC uses: render the display image (long side 1536, LANCZOS, JPEG quality 90), `common.tissue_mask` (HSV saturation > 0.07; brightness test off), region tissue fraction on the display mask; a region passes at ≥ 0.5 | `pipeline/01q_requalify_qc.py`, `settings.qc` |
| Pooling | mean of the QC-passing regions; all 12 regions if none passes (flagged `qc_fallback`) | `pipeline/riskmodels.pool_tiles` |
| Head | `data/models/<model>.npz`: median imputation, standardization, multinomial logistic regression | `pipeline/riskmodels.LinearModel` |
| Temperature* | `artifacts/eval/calibration.json` `temperature` (0.341542…, fit on training out-of-fold logits); must equal the model file and `model_choice.json` | `settings.calibration` |
| Conformal q-hat* | APS, alpha 0.1: `model_choice.json` `conformal[<model>].qhat_aps` (0.930810…), recomputed from `data/predictions.csv` calibration rows by `test_production_qhat_matches_app` | `core/conformal.py` |
| Conformal sets | as `05_risk_model` builds BACH test sets: randomized APS, one U per row from `default_rng(conformal.seed + 1)` = 45, one call over the cohort's crop matrix sorted by `crop_id`, then the argmax class added to empty sets (`conformal.nonempty`) | `pipeline/05_risk_model.add_aps_and_tier_sets` |
| Tier | `core.tiers.assign_tier`: lane `biopsy`, no aging clock, `reliability_flag` false, `qc_tiles_kept` as measured | `core/tiers.py` |
| Reliability | forced false. CONCH and MUSK are not on this laptop, so neither the OOD percentile nor the CONCH–MUSK disagreement can be computed | disclosed in the output |
| Device | Apple M3 Max laptop, PyTorch MPS | measured, recorded in the output |

### Views

The production model reads **one** view per region, so TCGA is read with one view.

- `data/emb/uni2h/meta.json` records `bach512`: `n_views` 1, `n_rows` 4,800 (400 images × 12 regions), no `_views` file.
- `data/emb/uni2h/bach512_probe_index.csv` has `n_views` = 1 for every region.
- `pipeline/jobs/embed_extra.sbatch` embedded `bach512` for UNI2-h with `--max-views bach512=1`.
- `docs/ACCURACY_ROUND2.md` lists `uni2h512_lr` as "production (frozen, view 0)".

`settings.encoders.views.g512: 8` sets the views written to the embedding manifest; CONCH and MUSK use them. They are
not the views behind the UNI2-h 512-px features the production model was trained, calibrated and tested on. The
judge-feedback plan assumed 8 views; that does not hold for `uni2h512_lr`. An 8-view read is reported as sensitivity
analysis S2 below.

## Parity gate (BACH, before any TCGA scoring)

The local recipe must reproduce the stored Delta embeddings and predictions first.

- **Images:** 5 BACH test images, fixed now: the first test case by `case_id` in each class (BACH_0110 Normal, BACH_0006
  Benign, BACH_0004 InSitu, BACH_0005 Invasive) plus the case example BACH_0087 (InSitu).
- **Source:** the original TIFFs, read from the public BACH archive on Zenodo (record 3632035, CC BY-NC-ND 4.0) with
  HTTP range requests into `data/tcga_brca_prod/bach_parity/`. They are never committed. If the archive cannot be
  read, the display JPEGs are used instead, labelled as such, and the gate becomes informational.
- **Compared:**
  - each region's local embedding against `data/emb/uni2h/bach512_probe.npy` (cosine similarity);
  - local QC against `data/tiles.csv`;
  - local calibrated probabilities against `data/predictions.csv` `pcal__*`.
- **Pass:** every region cosine ≥ 0.99 and every |Δ calibrated probability| ≤ 0.02.
- **On failure:** stop, investigate precision (bf16 on MPS) and preprocessing, and record the result under
  Deviations. TCGA is not scored with embeddings that fail the gate.

## Cohorts

**A. `same8`:** the 132 round-1 crops in `data/tcga_brca_test`, unchanged.

- 8 patients; 24 slides selected; 23 slides have at least one crop.
- This is the head-to-head with round 1.

**B. `protocol25`:** `select()` from `TCGA_script.py`, with the round-1 arguments except the patient count.

- Arguments: `--patients 25 --seed 44 --fovs 6 --target-mpp 0.42 --min-tissue 0.75 --max-slide-gb 2.0`.
- A dry run on 2026-10-18 selected 73 slides: 25 frozen tumor, 25 frozen normal and 23 FFPE tumor. Its first 24 slides
  are exactly the round-1 selection.
- Crops come from the new `--source tiles` path:
  - pixels from the GDC DeepZoom tile API at full resolution;
  - the tissue thumbnail from the DeepZoom level that matches the slide level openslide would use;
  - µm/px from the SVS `ImageDescription`, read with HTTP range requests.
- Everything else (selection, tissue mask, FOV sampling and seed, LANCZOS resize to 2048 × 1536) is the openslide
  path's code.

**Reuse rule for the 8 round-1 patients.** Their stored crops are reused in `protocol25` only if both checks pass:

- (a) for each of the 24 round-1 slides, the tile path picks the same FOV coordinates as `manifest.csv` (and none
  where round 1 had none);
- (b) re-extracting crop `_00` of every round-1 slide that has one (23 crops) through tiles, from the manifest
  coordinates, gives a maximum mean absolute difference < 3 (0–255 units) against the stored PNG.

Otherwise all 25 patients are extracted through tiles. Either way, the equivalence numbers are reported. Slides that
fail to extract or have fewer than 6 tissue-rich fields are listed, never replaced.

## Analyses (each cohort, slide level)

These are exactly `TCGA_TODO.md`. A slide's invasive score is the maximum calibrated P(Invasive) over its crops. A
slide is called invasive when any crop's top class is Invasive. Every interval is a 95% percentile bootstrap: 2,000
resamples, seed 44, clustered by patient (`core.stats.bootstrap_ci` through `eval/external_tcga.py` `analyze`).

**Primary**

1. Sensitivity on FFPE tumor slides.
2. Frozen tumor vs frozen normal: AUROC of the slide invasive score, plus sensitivity and specificity at the
   called-invasive rule.
3. FFPE vs frozen tumor sensitivity, and their difference.

**Secondary**

4. Reliability flag rate: **not computable** (needs CONCH and MUSK). Reported as such. The BACH clean test flag rate
   from `eval/reliability.json` is quoted for context.
5. OOD percentile: **not computable** (needs CONCH). Reported as such.
6. Tier placement: share of tumor slides with at least one crop in tier 1–2.
7. Crop-level P(Invasive) histograms by group (10 equal bins), for context.

**Pre-registered sensitivity analyses** (reported beside the primary results, never replacing them)

- **S1:** primary 1–3 and secondary 6 restricted to slides with exactly 6 crops. The protocol fixes 6 per slide so
  that normals do not get more chances to be flagged.
- **S2:** 8-view read. Each region's embedding is the mean of its 8 dihedral views (`encoders.dihedral` 0–7; float16
  views averaged as `02_embed.reduce_views` does). Everything else is unchanged. The production model was not trained
  on such features.
- **S3:** tier 1–2 rate with deterministic APS sets (U = 1), the rule round 1 used, for a like-for-like comparison on
  the same crops.

**Head-to-head on `same8`:** round-1 (`uni2h_ft`) estimates next to production estimates, plus per-slide agreement
on "called invasive".

**BACH reference:** BACH test images (single fields, not slides). Values come from the existing eval files where they
exist:

- invasive sensitivity, from `eval/metrics.json`;
- clean reliability flag rate, from `eval/reliability.json`.

The rest are computed from `data/predictions.csv` test rows, with the same bootstrap clustered by BACH patient group:

- specificity on Normal images (top class not Invasive);
- Invasive-vs-Normal AUROC of calibrated P(Invasive);
- tier 1–2 rate among Invasive images (top class Invasive, or Invasive in `set_tier`).

**Latency:** wall time per crop, from reading the PNG to the tier. It covers QC, the 12 regions, preprocessing, one
bf16 MPS batch of 12, pooling and scoring, under the primary recipe on `protocol25`. The first 2 crops are warm-up and
excluded. Reported: median, IQR and n; embedding-only time; and a hardware string (CPU, macOS, torch, device, dtype).

## Reporting

- Everything above goes to `artifacts/eval/external_tcga_prod.json` and `docs/RESULTS.md`, whatever the result.
- Nothing is retrained, and no threshold or rule is changed.
- The output states that TCGA has no benign or in situ labels.

## Limitations to state

- Labels are slide-level (sample type); a tumor-slide crop may show only stroma or fat.
- There are no benign or in situ labels, so specificity is measured only against frozen normal tissue adjacent to tumor.
- The sample is small (25 patients at most) and the intervals are wide.
- The reliability flag and OOD percentile are not computed (CONCH and MUSK are not local).
- `protocol25` crops come from GDC DeepZoom tiles, not from local openslide decoding of the slide file. The equivalence
  numbers above quantify the difference.
- BACH reference values are per image, TCGA values are per slide (maximum over up to 6 fields), so the comparison is
  indicative.

## Credit

The results here are based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga. Accessed via
the NCI Genomic Data Commons. BACH: Aresta G. et al., Medical Image Analysis (2019), doi:10.1016/j.media.2019.05.010.

## Deviations (reporting)

**2026-10-18, at the team's request.** The analyses above are unchanged. Only their presentation changes:

- The app, the deck and `docs/RESULTS.md` report only the production model.
- The round-1 fine-tune result is archived, not presented:
  - it moved to `artifacts/archive/eval/external_tcga_round1.json`, which is neither served nor packaged;
  - `eval/external_tcga.py` now writes only there.
- The served `artifacts/eval/external_tcga.json` holds this test. Its top level keeps the layout the app already reads,
  for the 25-patient cohort. The `same8` cohort and the other blocks sit alongside.
- The head-to-head comparison on the `same8` crops is therefore not reported. Production results on those crops are
  reported as the `same8` cohort.
- Sensitivity analyses S2 (8-view read) and S3 (deterministic APS sets) are computed and kept in the JSON, but not shown
  in `docs/RESULTS.md`. S1 is shown.
