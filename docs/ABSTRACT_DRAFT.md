# Abstract: draft for team review

Numbers checked on 2026-10-18 against `artifacts/eval/*.json` (fact sheet, metrics, calibration, conformal,
deletion, spot check, queue simulation, human baseline, external TCGA test). Every number has a source in the table
below.

## Primary version (≤ 300 words)

**Sepia: a confidence-aware breast pathology worklist built on pathology foundation models**

**Problem.** Biopsies are read in arrival order, so invasive carcinoma can wait behind benign cases, and AI flags
rarely state their evidence or confidence.

**Approach.** Sepia reads 400 public BACH breast microscopy images (single 2048 × 1536 fields; Aresta et al., 2019) as
12 regions embedded by the frozen UNI2-h foundation model. A linear probe trained on 218 images suggests one of four
categories; after temperature scaling, a conformal set fit on 50 calibration images lists the categories that cannot
be excluded. Images with invasive carcinoma in their set move up the worklist. Each read shows exact region
contributions, similar training cases and nine named features scored with CONCH. Three mirror Nottingham components
(tubule formation, nuclear pleomorphism, mitotic count); no grade is computed.

**Results.** On 132 held-out images (95% patient-grouped bootstrap), invasive-vs-rest AUROC was 0.999 [0.996, 1.000]
and macro one-vs-rest AUROC 0.965 [0.883, 0.995]; all 31 invasive images were called invasive (exact 95% CI 88.8–100%;
specificity 97.0% [93.6, 100]). Four-class accuracy was 86.4% [76.0, 94.4] (four BACH pathologists: 73–96%, different
images). ECE fell from 0.184 to 0.055. Conformal sets covered 91.7% against a 90% target (method chosen after viewing
test results; pre-registered LAC: 72.0%). In images called invasive, removing three highest-contribution regions
lowered the invasive logit 0.46 [0.38, 0.56] more than random regions. An AI spot check (Claude, not a pathologist)
supported invasion (19/20 regions) but not mitoses (0/20). In a simulated queue, invasive cases were first read after
6.0 instead of 20.2 minutes; unflagged cases waited 21.4 minutes longer. On 65 TCGA-BRCA slides from other hospitals,
the unchanged model called 95.7% of FFPE and 84.0% of frozen tumor slides invasive.

**Significance.** Calibrated, set-valued reads with region-level evidence can order a worklist transparently. BACH is
curated and lacks atypia, the workflow is simulated, and no clinical validation is claimed.

*Research prototype, not for clinical use. 2026 Ashby Prize in Computational Science Hackathon.*

Word count (Problem through Significance): 299.

## Where each number comes from

| Number in the abstract | Source |
|---|---|
| 400 images, 2048 × 1536, 12 regions per image | `artifacts/eval/fact_sheet.json` `bach` |
| 218 / 50 / 132 images, 24 test groups | `fact_sheet.json` `splits` |
| Invasive-vs-rest AUROC 0.999 [0.996, 1.000]; macro one-vs-rest AUROC 0.965 [0.883, 0.995] | `fact_sheet.json` `test.auroc_conventions` (`metrics.json` `production`) |
| 31 of 31 invasive images called invasive, exact 95% CI 88.8–100%; specificity 97.0% [93.6, 100] (98 of 101) | `fact_sheet.json` `test.invasive_sensitivity` (denominator: invasive test images from the confusion matrix; exact Clopper–Pearson interval from `core.stats.exact_boundary_ci`, because the bootstrap in `metrics.json` collapses at 100%), `test.invasive_specificity` |
| Accuracy 86.4% [76.0, 94.4] | `artifacts/eval/human_baseline.json` `sepia.accuracy` (= `metrics.json` `production.accuracy`) |
| Four pathologists 73–96% on the BACH hidden test set | `human_baseline.json` `bach_readers.accuracy` (Aresta et al., 2019, Table 4; `config/literature.yaml`) |
| ECE 0.184 → 0.055 | `calibration.json` `before.ece`, `after.ece` |
| Coverage 91.7%; LAC 72.0%; 90% target | `conformal.json` `methods.aps_marginal_nonempty`, `methods.lac_marginal`, `target_coverage` |
| Deletion: 0.46 [0.38, 0.56] in the 34 images called invasive (pre-registered primary subset; 0.54 [0.47, 0.60] over all 132) | `deletion.json` `models.uni2h512_lr.predicted_aggressive.contribution_minus_random_logit` (`settings.primary_subset`; all 132: `all_eligible`) |
| Spot check 19/20, 0/20 | `spotcheck.json` (reviewer kind: AI) |
| 20.2 → 6.0 min; +21.4 min | `queue_sim.json` `results.fifo` / `results.tiers`; `analytics.json` `queue.vs_fifo` |
| Three of nine features mirror Nottingham components | `config/concepts.yaml` (`group: nottingham`) |
| 65 TCGA-BRCA slides; 95.7% FFPE and 84.0% frozen tumor slides called invasive | `artifacts/eval/external_tcga.json` `n_slides`, `primary.ffpe_tumor_sensitivity`, `primary.frozen_tumor_sensitivity` (production model, protocol `docs/EXTERNAL_TCGA_R2.md`) |

## Numbers corrected from the previous draft

- Reliability error AUROC is 0.69 [0.58, 0.87] (`reliability.json` `error_auroc_test`); the previous draft's 0.71
  was stale. It is left out of the primary version for length.
- Unflagged cases wait 21.4 minutes longer in the simulated queue, not 25.
- "All 31 invasive images" now carries its exact 95% interval (88.8–100%): the bootstrap interval collapses to
  [100, 100] at a perfect result. Accuracy is quoted as 86.4% [76.0, 94.4] to match the app and the fact sheet.
- The deletion test is quoted on the pre-registered subset (images called invasive, 0.46); the previous draft quoted
  the all-image value (0.54) without saying so.
- "Nine named histologic features" now states that three mirror Nottingham components and no grade is computed.
- Case summaries in the shipped case files are written by a template from structured findings; the previous draft
  said an LLM writes them.
- Sepia's 132-image test split and the BACH reader study's 100 hidden images are different images; the comparison is
  stated as context.
