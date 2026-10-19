# Whole slides, other organs, and the studies that would validate Sepia

Research prototype, not for clinical use. This file explains how Sepia would move from single fields to whole slides, which parts would change for a different organ, and how a reader study and a frozen-section study would be run. Everything below the first section is proposed and has not been built.

## Today

- **Read unit.** One 2048 × 1536 microscopy field at 0.42 µm/px, read as 12 regions of 512 px. Each region is embedded once with UNI2-h. The case read pools the regions that pass the tissue check.
- **Whole slides in use now.**
  - **BACH:** 84 annotated windows from 10 BACH whole slides test the simulated margin read (`artifacts/eval/margin.json`).
  - **TCGA:** each slide is sampled as 6 random tissue-rich fields at BACH resolution. The slide score is the maximum calibrated P(invasive) over its fields (`docs/EXTERNAL_TCGA_R2.md`).
- **Region evidence.** Each region carries an exact contribution to the read's invasive logit; constant plus contributions reproduce the read to within 1e-6. These contributions pass a deletion test, so they stay meaningful when regions are aggregated.

## Proposed whole-slide path

1. **Tissue detection** on a low-resolution thumbnail, using the same saturation rule as the field-level tissue check.
2. **Tiling.** Tile all tissue at 0.42 µm/px into 512-px regions on a fixed grid, keeping slide coordinates. A 15 × 15 mm tissue area holds about 4,900 such regions (arithmetic below).
3. **Embedding.** UNI2-h once per region, as today.
4. **Aggregation.** Region → slide → specimen:
   - **Region to slide:** the mean-pool linear read applied to sliding fields of 12 regions (the unit it was trained on), followed by the maximum over fields. This is the rule the TCGA test already uses with 6 fields. Attention pooling is an alternative.
   - **Slide to specimen:** the maximum over the specimen's slides.
   - **Recalibration:** temperature and conformal threshold refit on a slide-level calibration set, because field-level calibration does not transfer to slide-level maxima.
5. **Evidence navigation.** Store the top-contribution regions with slide coordinates. The case view opens them in a tiled viewer, using DeepZoom tiles as the GDC tile service does, so the pathologist can jump from the flag to the regions behind it.
6. **Throughput.** Measured laptop latency per field is reported in `docs/RESULTS.md` (TCGA section). Slide-level cost scales with the number of regions; it is estimated there from the measured per-region cost, not measured on whole slides.

Region arithmetic: a 512-px region at 0.42 µm/px covers 215 × 215 µm, about 0.046 mm². A 15 × 15 mm (225 mm²) tissue area therefore holds about 4,900 regions.

## Other organs

UNI2-h, CONCH and MUSK were pretrained on many organs, but Sepia's trained parts are breast-specific. These are the swap points:

| Part | File | What changes for a new organ |
|---|---|---|
| Class list and training labels | `config/settings.yaml` (`classes`, `aggressive_class`), `data/cases.csv` | A labeled dataset for the organ; retrain the linear probe |
| Feature descriptions | `config/concepts.yaml` | Organ-specific descriptors (for example from CAP cancer protocols), re-mined caption examples, a new spot check |
| Follow-up order rules | `config/stain_rules.yaml` | Organ-specific stains and rules, reviewed by a pathologist |
| Tier rule | `core/tiers.py` (reads `aggressive_class`) | Which class is escalated |
| Calibration and conformal threshold | refit per organ | A calibration split from the new organ |

How unevenly the features transfer is already measured. On colorectal tiles (NCT-CRC-HE-100K), the text features for lymphocytes and necrosis separate tissue types well (AUROC 0.998 and 0.939), but pleomorphism does not (0.59). Source: `artifacts/eval/concepts.json`. No accuracy is claimed outside breast.

## Proposed reader study

- **Design.** Multi-reader, multi-case crossover. Pathologists read the same cases with and without Sepia, with a washout of at least four weeks, in randomized order.
- **Cases.** Consecutive breast biopsies from at least two sites, enriched for atypia and in situ carcinoma, where human disagreement concentrates (`docs/HUMAN_BASELINE.md`).
- **Reference standard.** Expert panel consensus.
- **Endpoints.**
  - Agreement with the reference standard, per category.
  - Time per case.
  - Appropriate reliance: the share of correct AI suggestions accepted and of incorrect ones rejected.
  - Stain orders placed.
  - Inter-reader agreement.
- **Analysis.** Obuchowski–Rockette or a mixed model with reader and case as random effects.
- **Safety.** A data monitor reviews cases where a reader changed a correct call to an incorrect one after seeing Sepia.

## Proposed frozen-section study

- **Silent mode first.** Sepia scores real intraoperative frozen sections in parallel with standard care. Results are hidden from the pathologist and the surgeon.
- **Timestamps.** Specimen received, slide scanned, Sepia score ready, pathologist call. The aim is to measure where time goes, not to claim a saving in advance.
- **Endpoints.** Agreement of Sepia's flag with the frozen-section diagnosis and with the permanent-section diagnosis. The share of frozen sections Sepia defers for low reliability.
- **No autonomous margin calls.** Any visible use would only reorder the queue and show evidence. The pathologist makes every call.
