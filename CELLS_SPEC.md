# Sepia — Cell Segmentation Spec

Extends `Plan.md`. Read `Plan.md` first, especially Section 2 (compute rules) and
Section 16 (rules for Claude Code); all of them apply here.

**Goal:** answer the judges' question "what cells caused this?" by segmenting and classifying
nuclei in the tiles that drove each prediction, turning them into measurable, named features,
and showing them in the explanation.

**Principle:** the segmentation model does not reason. It produces measurements (cell counts,
nuclear size and shape variation, lymphocyte proximity) that the interpretable model uses and
the summary cites. Every number shown to the user comes from a stored measurement.

**Feature flag:** `cells.enabled` in `config/settings.yaml`. The rest of Sepia must work
unchanged when it is `false`. If this module fails, turn it off and move on.

---

## 1. Where it fits

```
04_concepts ─► 04b_cells (NEW) ─► 05_risk_model ─► ... ─► 08_build_case_json
                  │                    ▲
                  └── cell features ───┘ (added to the concept bottleneck)
```

New and changed files:

```
config/settings.yaml          # add the `cells:` block (Section 9)
config/cell_features.yaml     # feature definitions, display names, expected directions
pipeline/04a_select_tiles.py  # picks top tiles per case (out-of-fold)
pipeline/04b_cells.py         # runs segmentation on selected tiles (GPU job)
pipeline/04c_cell_features.py # nuclei -> per-tile and per-case features (CPU)
pipeline/jobs/cells.sbatch
eval/cells_eval.py            # sanity checks, ablation, spot-check sheet
tests/test_cell_features.py
```

`05_risk_model.py` and `08_build_case_json.py` gain optional inputs (Sections 6 and 7).

---

## 2. Model choice and the feasibility spike

Run a **time-boxed spike (60 minutes total)** before building anything else. Try options in
order and stop at the first that works on 5 BACH tiles.

| Order | Option | Why | Watch out for |
|---|---|---|---|
| 1 | **HoVer-Net, PanNuke checkpoint, tile mode** (`vqdang/hover_net`, `run_infer.py tile`, `--model_mode=fast`, `--nr_types=6`) | Processes plain image tiles directly; writes per-nucleus JSON (contour, centroid, type) and an overlay PNG | Older code; dependency conflicts with the `pytorch-conda/2.8` module are possible. Use a separate venv if needed. PanNuke checkpoints **must** use `fast` mode |
| 2 | **CellViT** via the `cellvit` PyPI package (`TIO-IKIM/CellViT-Inference`) | Stronger model, actively maintained | Built for whole-slide images: needs openslide/libvips system libraries, ≥ 24 GB GPU memory, and downloads weights into `CELLVIT_CACHE` at runtime. Compute nodes have no internet, so weights must be cached on the login node first |
| 3 | Drop the module | — | Set `cells.enabled: false`; the product still works |

Spike checklist (a human runs the GPU parts):
1. Login node: install into a dedicated environment; download the checkpoint into
   `/work/nvme/bfqi/teams/<team>/pathqueue/weights/`. Confirm the license
   (CellViT is Apache 2.0 with Commons Clause; check HoVer-Net's checkpoint terms) allows
   non-commercial hackathon use, and record it in README Credits.
2. Short interactive GPU allocation (≤ 30 min): run on 5 tiles from different BACH classes.
3. Look at the overlays. Pass if nuclei outlines are mostly on nuclei and tumor vs lymphocyte
   labels look plausible to a team member. Write the result to `artifacts/eval/cells_spike.md`.

PanNuke cell types (both options): **neoplastic, inflammatory, connective/soft tissue, dead,
non-neoplastic epithelial** (plus background). Map them to display names in
`config/cell_features.yaml`.

---

## 3. Resolution matching

Segmentation models are trained at a specific resolution. PanNuke-trained models expect
about **0.25 µm/px (40x)**.

- Read BACH's pixel size from `artifacts/inventory.json` (Plan.md 5.2). It is commonly reported as
  about 0.42 µm/px; **use the measured value, not this number.**
- Rescale each selected tile by `bach_mpp / model_mpp` before inference
  (e.g. 0.42 / 0.25 ≈ 1.68, so a 512 px tile becomes ~860 px). Use bicubic interpolation.
- Store the scale factor. Convert all areas and distances back to **µm and µm²** using the model
  resolution, so features are in physical units and independent of rescaling.
- If CellViT offers a 20x checkpoint closer to BACH's resolution, the spike may compare both.

---

## 4. Which tiles to segment (`04a_select_tiles.py`)

Segment the **top 3 QC-passing tiles per case**, for **all splits** (train features are needed to
train the model). About 400 cases × 3 = ~1,200 tiles.

- Rank tiles by tile-level risk from the tile probe in Plan.md 6.6.
- **Avoid leakage:** for train-split cases, use **out-of-fold** tile risk (5-fold cross-validation
  over train cases), so a case's own label never chose its own tiles. Calibration and test cases use
  the probe trained on the full train split.
- Ties or cases with fewer than 3 QC tiles: take what exists, record `n_tiles`.
- Output: `cell_tiles.csv` (case_id, tile_idx, split, rank, tile_risk, path).

---

## 5. Running segmentation (`04b_cells.py`, `jobs/cells.sbatch`)

- One GPU batch job, `--time=01:00:00`, ≤ 16 CPUs, ≤ 62 GB memory. Must not cross Sun 08:00
  or Sun 18:00. Includes `module load pytorch-conda/2.8` and `unset PYTHONNOUSERSITE`
  (or activates the spike's venv).
- Resumable: skip tiles whose output exists; write outputs atomically.
- `--limit N` for testing. Claude Code writes the script and the exact `sbatch` command;
  **a human submits it.**
- Per tile output: `cells/<case_id>/<tile_idx>.json` containing, per nucleus:
  `centroid_um`, `area_um2`, `perimeter_um`, `eccentricity`, `solidity`, `type`, `type_prob`,
  and a simplified contour (for overlays). Also save `cells/<case_id>/<tile_idx>_overlay.png`
  (contours drawn on the original-resolution tile, colored by type).
- Overlay colors: neoplastic `#B0335C`, inflammatory `#4A3B8F`, connective `#C9A227`,
  dead `#5A5A5A`, non-neoplastic epithelial `#2E8BC0`. Colors differ in lightness, not only
  hue, and the UI always shows a legend.

---

## 6. Features (`04c_cell_features.py`, `config/cell_features.yaml`)

CPU only. Computed per tile, then aggregated per case (mean over the selected tiles, weighted by
nucleus count where noted).

| id | Definition | Display name | Expected direction (Invasive vs benign/normal) |
|---|---|---|---|
| `neo_fraction` | neoplastic nuclei / all nuclei | Tumor cell fraction | higher |
| `neo_density` | neoplastic nuclei per mm² of tissue | Tumor cell density | higher |
| `nuc_area_mean` | mean area of neoplastic nuclei (µm²) | Nuclear size | higher |
| `nuc_area_cv` | coefficient of variation of neoplastic nuclear area | Variation in nuclear size | higher |
| `nuc_shape_cv` | CV of eccentricity (and of solidity) of neoplastic nuclei | Variation in nuclear shape | higher |
| `infl_fraction` | inflammatory nuclei / all nuclei | Immune cell fraction | context |
| `til_proximity` | fraction of inflammatory nuclei within 30 µm of a neoplastic nucleus | Immune cells near tumor | context |
| `dead_fraction` | dead nuclei / all nuclei | Dead cell fraction | higher |
| `epi_fraction` | non-neoplastic epithelial / all nuclei | Benign epithelial fraction | lower |

Rules:
- **Minimum counts:** morphology features (`nuc_*`) need ≥ 20 neoplastic nuclei in the case;
  otherwise set to missing and add `insufficient_tumor_nuclei: true`. Ratios need ≥ 50 total
  nuclei. In modelling, impute missing values with the train median and add a missing-indicator.
- **Reference percentiles:** for each feature, store the case's percentile against **train cases
  labeled Normal or Benign**. This powers explanations like "variation in nuclear size is at the
  95th percentile of benign reference cases."
- `nuc_area_cv` and `nuc_shape_cv` are the measurable counterpart of the `pleomorphism` text
  concept. Keep both; their agreement is a validity check (Section 8).
- **Mitoses are out of scope** for this spec (PanNuke has no mitosis class). The text-prompt
  `mitoses` concept stays, marked weak. A mitosis detector (e.g. from the MIDOG challenge) is a
  stretch goal only.
- Every feature's display name and expected direction live in `config/cell_features.yaml`;
  no hard-coded strings in the UI.

---

## 7. Model integration and outputs

### 7.1 Risk model (changes to `05_risk_model.py`)

When `cells.enabled`, train and report three concept-bottleneck variants on the same splits:

| Variant | Inputs |
|---|---|
| B | Text-prompt concepts only (Plan.md 6.6) |
| B+C | Text-prompt concepts + cell features |
| C | Cell features only |

- Standardize features on train; L2-regularized logistic regression; choose regularization on the
  calibration split.
- Report AUROC (Invasive vs rest), InSitu↔Invasive confusion, and ECE for each, with bootstrap 95%
  CIs, alongside Model A (black box).
- **Pick the production model on the calibration split, not the test split.** Record the choice in
  `settings.yaml` and on the Metrics page. If B+C doesn't help, keep B for prediction and still show
  cell features as supporting evidence ("measured features consistent with the read").
- Per-feature contribution = coefficient × standardized value, exactly as for text concepts, so the
  "Why" chart shows both kinds side by side, labeled by source (text-prompt criterion vs measured
  cell feature).
- Calibration and conformal steps run unchanged on whichever model is chosen.

### 7.2 Case JSON additions (changes to `08_build_case_json.py`)

Add a `cells` block. Absent or `{"status": "disabled"}` when the flag is off. Never include it in
`eval_only` logic.

```json
"cells": {
  "status": "ok",
  "model": "hovernet_pannuke",
  "mpp_model": 0.25,
  "tiles": [
    {"idx": 5, "overlay": "cells/BACH_0123_5_overlay.png",
     "counts": {"neoplastic": 212, "inflammatory": 31, "connective": 18, "dead": 4, "epithelial": 2}}
  ],
  "features": {
    "nuc_area_cv": {"value": 0.48, "unit": "", "ref_percentile": 0.95, "contribution": 0.21},
    "neo_density": {"value": 3150, "unit": "per mm²", "ref_percentile": 0.91, "contribution": 0.18},
    "til_proximity": {"value": 0.22, "unit": "", "ref_percentile": 0.40, "contribution": 0.0}
  },
  "flags": {"insufficient_tumor_nuclei": false}
}
```

Add one finding per feature with a non-trivial contribution or an extreme reference percentile
(≥ 0.9 or ≤ 0.1), for example:

```json
{"id": "F5", "kind": "cells",
 "data": {"feature": "nuc_area_cv", "value": 0.48, "ref_percentile": 0.95, "tiles": [5, 6]}}
```

The grounded summary (Plan.md 8.2) may cite these findings. Its validator already rejects numbers
that aren't in cited findings; extend the template generator with one sentence pattern per feature,
e.g. "Variation in nuclear size is higher than in 95% of benign reference cases (regions R5, R6)."

### 7.3 UI (changes to the case view, Plan.md 11)

- New **Cells** tab in the case view: the top tiles with the overlay toggled on/off, a type legend,
  per-tile cell counts, and the feature list with reference-percentile bars.
- In the **Why** chart, measured cell features appear with a distinct marker ("measured") next to
  text-prompt criteria ("estimated").
- Clicking a cell finding in the summary opens the Cells tab on the relevant tile.
- Wording: "measured features associated with invasive carcinoma"; never "this cell caused".

---

## 8. Validation (`eval/cells_eval.py`)

All on the test split unless noted, with bootstrap 95% CIs. Write `artifacts/eval/cells.json`
and plots for the Metrics page.

1. **Visual spot check:** a team member reviews 20 overlays (5 per class) and marks each
   acceptable/not. Report the fraction. Generate a simple review sheet (HTML grid with checkboxes
   or a CSV) to make this fast.
2. **Expected-direction checks:** per-class means of each feature; e.g. `neo_fraction` and
   `nuc_area_cv` should be higher in InSitu/Invasive than in Normal/Benign. Report any feature
   that goes the "wrong" way instead of hiding it.
3. **Convergent validity:** Spearman correlation between `nuc_area_cv`/`nuc_shape_cv` and the
   `pleomorphism` text concept, and between `infl_fraction` and the `lymphocytes` concept.
   Positive correlation supports both; weak correlation is worth reporting honestly.
4. **Ablation:** the B / B+C / C table from 7.1.
5. **Faithfulness:** extend Plan.md's deletion test to cell features: setting a case's top cell
   feature to the benign-reference median should lower its risk more than resetting a random
   feature.

---

## 9. Configuration block

```yaml
cells:
  enabled: true
  backend: hovernet          # hovernet | cellvit
  checkpoint: /work/nvme/bfqi/teams/<team>/pathqueue/weights/<file>
  model_mpp: 0.25
  tiles_per_case: 3
  oof_folds: 5
  min_neoplastic_nuclei: 20
  min_total_nuclei: 50
  til_radius_um: 30
  reference_classes: [Normal, Benign]
  findings_percentile_extremes: [0.1, 0.9]
```

---

## 10. Milestones

These slot into Plan.md Section 13 after M4 and must finish before the Sunday GPU cutoff.

| # | Milestone | Done when |
|---|---|---|
| C0 | Spike (≤ 60 min) | `cells_spike.md` records the chosen backend, or the module is disabled |
| C1 | Tile selection + 5-tile run | overlays and nucleus JSON for 5 tiles look right |
| C2 | Full segmentation run (~1,200 tiles) | all selected tiles have outputs; job under 1 hour |
| C3 | Features + validation + ablation | `cells.json` written; production model chosen on calibration split |
| C4 | Case JSON + UI Cells tab | demo case shows overlays, features, and a cited cell finding |

If C2 has not finished by **Sun 10:00**, disable the module and present it as future work, using the
spike overlays as an illustration.

---

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Install or dependency conflicts | Separate venv; then try the other backend; then disable |
| Weights download needs Google Drive auth | Download on a laptop and copy to Delta with `scp`/OnDemand |
| GPU memory (CellViT wants ≥ 24 GB) | Use HoVer-Net, or smaller batch size / mixed precision |
| Resolution mismatch → poor outlines | Recheck mpp from inventory; compare rescale factors on 5 tiles |
| Too few tumor nuclei in some cases | Missing-value handling (Section 6); flag shown in UI |
| Cell features don't improve accuracy | Keep them as supporting evidence; report the ablation honestly |
| Running out of time | Disable the flag; show spike overlays as illustration |

---

## 12. Credits to add to the README and deck

- HoVer-Net: Graham et al., *Medical Image Analysis*, 2019.
- PanNuke dataset (training data for the checkpoints): Gamper et al.
- CellViT: Hörst et al., *Medical Image Analysis*, 2024 (if used).
- Verify all citations before the final presentation.
