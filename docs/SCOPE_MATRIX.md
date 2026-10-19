# What Sepia does today: implemented, evaluated, simulated, proposed

Research prototype, not for clinical use. Each feature gets one status from the list below. Numbers come from `artifacts/eval/*.json` (see `docs/RESULTS.md` and `docs/FACT_SHEET.md`); numbers marked "computed" are explained in the appendix of `docs/JUDGE_QA.md`.

**Statuses**

- **Evaluated (BACH):** implemented and measured on the 132 held-out BACH test images (24 patients: invasive from 17, normal, benign and in situ from 6 each).
- **Evaluated (external):** implemented and measured on TCGA-BRCA tissue that the model never saw.
- **Simulated:** implemented, but exercised only on simulated inputs (simulated arrivals, degraded images standing in for frozen sections, image borders standing in for inked margins).
- **App only:** works in the app. It is not connected to any clinical system and has no outcome measurement.
- **Proposed:** not built.

## Reading an image

| Feature | Status | Evidence | What would validate it |
|---|---|---|---|
| Four-class read (Normal, Benign, In situ, Invasive) from UNI2-h region embeddings and a linear probe | Evaluated (BACH) | Macro-F1 0.860 [0.678, 0.928]; accuracy 86.4% [76.0, 94.4]; invasive-vs-rest AUROC 0.999 [0.996, 1.000]; 31 of 31 invasive images called invasive (exact 95% CI 88.8–100% by image, 80.5–100% by patient, 17 patients). In situ recall 24/30 [0.33, 0.87], from 6 patients | Multi-site test sets with all four classes and many more patients per class; reader study |
| Same read on tissue from other hospitals, scanners and real frozen sections | Evaluated (external) | 25 TCGA-BRCA patients, 65 slides: FFPE tumor slides called invasive 95.7% (22 of 23), frozen tumor 84.0% (21 of 25), frozen normal not called 17 of 17 (exact 95% CI 80.5–100%), but 7 of 17 frozen normal slides had a field in tiers 1–2 (computed). FFPE-to-frozen gap 11.7 points, patient bootstrap −4.7 to +28.0 (computed). TCGA has no benign or in situ labels and no FFPE normal slides | Larger external cohorts with benign and in situ cases and FFPE normal tissue |
| Calibrated class percentages (temperature scaling) | Evaluated (BACH) | ECE 0.184 → 0.055 [0.032, 0.185]; Brier 0.285 → 0.226; log-loss unchanged (0.571 → 0.575). Five errors at ≥ 97% confidence (computed) | Calibration on each new site before use |
| Conformal set ("cannot be excluded") and the safety rule | Evaluated (BACH) | Tier-set coverage 91.7% (method chosen after viewing test results, disclosed); pre-registered LAC 72.0%, which is unlikely under exchangeability (P ≈ 0.0005, computed); 94.5% mean over 1,000 re-splits of the same images | Prospective coverage monitoring on exchangeable cases from the deployment site |
| Reliability flag and deferral | Evaluated (BACH), including simulated degradations | Error AUROC 0.69 [0.58, 0.87]; flags 42% of test errors against an 80% target; its out-of-distribution and second-model (MUSK) parts are at chance (AUROC 0.46, 0.49). Deferring the least reliable 20% of all test reads (132 clean + 1,584 simulated degraded) raises kept accuracy 85.8% → 89.8%; top-class probability alone reaches 91.3% | Real artifacts (frozen, poor stain) with errors adjudicated |
| Region contributions (heatmap) | Evaluated (BACH) | Exact for the linear read (constant plus contributions reproduce the invasive logit within 1e-6). Removing the top 3 regions lowers the centered logit 0.46 [0.38, 0.56] more than random regions in the 34 images called invasive; for a linear mean-pooled read this arm passes by construction, so it shows concentration, not clinical relevance | Pathologist agreement with highlighted regions |
| Nine named features (3 related to Nottingham components + 6 others) | Evaluated (BACH), as associations | Scored by a separate concept model (`cb_text`, test macro-F1 0.708; same top class as production on 95 of 132, computed). AI spot check (Claude, not a pathologist): invasion 19/20, tubules 18/20, pleomorphism 18/20, benign stroma or fat 20/20; mitoses 0/20, normal lobules 0/20. Concept regions fail the deletion test (−0.02 [−0.10, +0.04]) | Pathologist annotation of feature regions |
| Similar training cases and the look-alike with a different label | Evaluated (BACH) | Precision@5 0.67 [0.58, 0.83] (benign 47%, in situ 55%) | Reader study: does the comparison change decisions? |
| Look-alike contrast table (which features differ, by how much, spot-check status) | App only | Computed from the same concept-model feature scores; 48 of 328 look-alikes have two differences that passed the spot check, 170 one, 110 none | Same as above |
| Cell segmentation features (HoVer-Net) | Evaluated (BACH), weak | Cells-only model macro-F1 0.33 on CV; shown as measured context only | — |

## Worklist and workflow

| Feature | Status | Evidence | What would validate it |
|---|---|---|---|
| Priority tiers in the worklist | Simulated | Simulated day with a 90%-busy reader: invasive first look 20.2 → 6.0 min; unflagged cases wait 21.4 min longer. At 60% load: 5.7 → 3.3 min, +3.5 min (computed). Only invasive carcinoma is escalated: 12 of 23 in situ biopsy images sit in tier 4, and one in situ carcinoma read as normal at 99% (BACH_0112) is unflagged | Silent-mode deployment with real timestamps |
| Aging guard (promotes long-waiting cases) | Simulated | No measured effect at the configured targets | Real queue data |
| Frozen-section lane and operating-room board with elapsed-time timer (amber at 15 min, red at 20 min) | Simulated | Arrivals are replayed; 33 of 132 test images are assigned to the frozen lane, and frozen-section appearance is simulated by blur, JPEG, stain shift and resampling, which change accuracy little (81.8–87.1%). Real frozen tissue is tested only in the TCGA external test, not in the OR workflow | Intraoperative silent trial (see `docs/WSI_AND_EXTENSION.md`) |
| Simulated margin read (image borders as inked margins) | Simulated | Composite fields: clear margins kept clear in 64 of 64 (exact 95% CI 94.4–100%); benign test images: no margin wrongly called involved, 0 of 71 (exact 95% CI 0–5.1%); annotated WSI windows: sensitivity 45.4% [25.4, 65.2], specificity 98.5% | Real inked margins with permanent-section outcome |
| Follow-up orders (p63/calponin, ER/PR/HER2/Ki-67, deeper levels, recuts) | App only | Drafted by rules. The pathologist approves or edits them; orders are approved in Sepia, and nothing is sent to a laboratory or ordering system. p63/calponin fires when in situ and invasive share the set: on test, 6 biopsy images, all in situ; it was drafted for 1 of the 3 in situ images read as invasive | Pathologist review of the rules; integration test against a sandboxed laboratory information system |
| Case summary text | App only | Written by a deterministic template from structured findings and checked sentence by sentence | — |
| "In plain language" explanation | App only | Written by a language model through a local CLI; see the model chip on each text. Every number is checked against the evidence. The deployed site serves cached explanations only. It summarizes evidence and is not evidence | — |
| Oncology inbox | App only (placeholder) | Receives the signed-out summary of an invasive read; concept features whose spot check failed are omitted from the listed features and only named underneath | — |

## Not built

| Item | Status | Plan |
|---|---|---|
| Whole-slide reading for the four-class read | Proposed | `docs/WSI_AND_EXTENSION.md`. Today: single fields; BACH whole slides only for margin windows and feature checks; TCGA slides sampled as up to 6 fields |
| Other organs | Proposed | Swap points listed in `docs/WSI_AND_EXTENSION.md`; nothing validated beyond breast |
| Prognosis (recurrence, survival, metastasis) | Not claimed | BACH labels are diagnostic categories. Displayed percentages are not risks of progression |
| Reader study with pathologists | Proposed | `docs/WSI_AND_EXTENSION.md` |
