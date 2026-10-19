# Abstract: draft for team review

Numbers checked on 2026-10-19 against `artifacts/eval/*.json` (fact sheet, metrics, analytics, calibration,
conformal, spot check, queue simulation, human baseline, external TCGA test) and the case files. Every number has a
source in the table below. The answers behind the caveats are in `docs/JUDGE_QA.md`.

## Primary version (≤ 300 words)

**Sepia: a confidence-aware breast pathology worklist built on pathology foundation models**

**Problem.** Biopsies are read in arrival order, so invasive carcinoma can wait behind benign cases, and AI flags
rarely state their evidence or confidence.

**Approach.** Sepia reads 400 public BACH breast microscopy images (single 2048 × 1536 fields; Aresta et al., 2019) as
12 regions embedded by the frozen UNI2-h foundation model. A linear probe trained on 218 images suggests one of four
categories; after temperature scaling, a conformal set fit on 50 calibration images lists the categories that cannot
be excluded. Images with invasive carcinoma in their set move up the worklist. Each read shows exact region
contributions, similar training cases and nine named features from a separate CONCH-based concept model; three mirror
Nottingham components, and no grade is computed.

**Results.** On 132 held-out images from 24 patients (95% patient bootstrap), invasive-vs-rest AUROC was 0.999
[0.996, 1.000]; all 31 invasive images, from 17 patients, were called invasive (exact 95% CI 88.8–100% by image,
80.5–100% by patient; specificity 97.0%). Four-class accuracy was 86.4% [76.0, 94.4] (four BACH pathologists: 73–96%,
different images). ECE fell from 0.184 to 0.055. Conformal sets covered 91.7% against a 90% target (method chosen
after viewing test results; pre-registered LAC: 72.0%). An AI spot check (Claude, not a pathologist) supported
invasion (19/20 regions) but not mitoses (0/20). In a simulated queue with a 90%-busy reader, invasive cases were
first read after 6.0 instead of 20.2 minutes; unflagged cases waited 21.4 minutes longer. On 65 TCGA-BRCA slides from
25 patients, the unchanged model called 95.7% of FFPE and 84.0% of frozen tumor slides invasive, and no frozen normal
slide (0/17).

**Significance.** Calibrated, set-valued reads with region-level evidence can order a worklist transparently. BACH is
curated, lacks atypia and has six patients per non-invasive class; the workflow is simulated, and no clinical
validation is claimed.

*Research prototype, not for clinical use. 2026 Ashby Prize in Computational Science Hackathon.*

Word count (Problem through Significance, `wc -w`, labels included): 296.

## Where each number comes from

| Number in the abstract | Source |
|---|---|
| 400 images, 2048 × 1536, 12 regions per image | `artifacts/eval/fact_sheet.json` `bach` |
| 218 / 50 / 132 images | `fact_sheet.json` `splits` |
| 24 test patients; invasive images from 17; six patients for each of normal, benign and in situ | `fact_sheet.json` `splits` (test groups 24, identified patients 24); `artifacts/eval/analytics.json` `groups_by_class` |
| Nine features from a separate concept model; three mirror Nottingham components | `artifacts/eval/model_choice.json` `contributions_model` (`cb_text`); `config/concepts.yaml` (`group: nottingham`) |
| Invasive-vs-rest AUROC 0.999 [0.996, 1.000] | `fact_sheet.json` `test.auroc_conventions` (`metrics.json` `production`) |
| 31 of 31 invasive images, exact 95% CI 88.8–100% by image; 80.5–100% by patient | `fact_sheet.json` `test.invasive_sensitivity`; patient-level bound: exact Clopper–Pearson for 17 of 17 patients with every invasive image called invasive (`docs/JUDGE_QA.md` appendix A3) |
| Specificity 97.0% (98 of 101) | `fact_sheet.json` `test.invasive_specificity` |
| Accuracy 86.4% [76.0, 94.4] | `artifacts/eval/human_baseline.json` `sepia.accuracy` (= `metrics.json` `production.accuracy`) |
| Four pathologists 73–96% on the BACH hidden test set | `human_baseline.json` `bach_readers.accuracy` (Aresta et al., 2019, Table 4; `config/literature.yaml`) |
| ECE 0.184 → 0.055 | `calibration.json` `before.ece`, `after.ece` |
| Coverage 91.7%; LAC 72.0%; 90% target | `conformal.json` `methods.aps_marginal_nonempty`, `methods.lac_marginal`, `target_coverage` |
| Spot check 19/20, 0/20 | `spotcheck.json` (reviewer kind: AI) |
| 90%-busy reader; 20.2 → 6.0 min; +21.4 min | `queue_sim.json` `assumptions.utilization`, `results.fifo` / `results.tiers`; `analytics.json` `queue.vs_fifo` |
| 65 TCGA-BRCA slides, 25 patients; 95.7% FFPE and 84.0% frozen tumor slides called invasive; 0 of 17 frozen normal slides | `artifacts/eval/external_tcga.json` `n_slides`, `n_patients`, `primary.ffpe_tumor_sensitivity`, `primary.frozen_tumor_sensitivity`, `primary.frozen_tumor_vs_normal.specificity` (production model, protocol `docs/EXTERNAL_TCGA_R2.md`) |

## Changes from the previous draft

- The deletion-test sentence is removed. For the production read, which is linear in the mean of its region
  embeddings, removing the highest-contribution regions must lower the logit more than random regions, so the result
  (0.46 [0.38, 0.56]) cannot fail and is not evidence of faithfulness beyond exactness (`docs/JUDGE_QA.md` question 28).
- The nine features are now attributed to the separate concept model that scores them (`cb_text`), not to the
  production read.
- Patient counts are added: 24 test patients, invasive images from 17, and six patients for each other class. The
  invasive sensitivity carries a patient-level exact interval (80.5–100%) beside the image-level one.
- The queue sentence states the simulated reader's 90% utilization. At lower load the gain shrinks (60%: 5.7 → 3.3
  minutes; `docs/JUDGE_QA.md` appendix A7).
- TCGA adds the normal comparator (0 of 17 frozen normal slides called invasive) and the patient count.
- Kept from the previous corrections: reliability error AUROC is 0.69, not 0.71 (left out for length); unflagged cases
  wait 21.4 minutes longer, not 25; case summaries in the shipped case files come from a template; the 132-image test
  split and the BACH reader study's 100 hidden images are different images.
