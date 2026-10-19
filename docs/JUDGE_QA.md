# Answers to the judges' questions

Research prototype, not for clinical use. Each answer is short enough to say in about 20 seconds and ends with its source. Numbers were checked on 2026-10-19 against `artifacts/eval/*.json`, `docs/FACT_SHEET.md`, the case files and `data/*.csv`. Numbers marked "computed" are not in an eval file; the appendix says how each was computed. When an answer here disagrees with something said during judging, this file is correct.

Questions 1–43 are grouped by theme. Sixteen were asked in earlier sessions (marked "asked"); the rest are the hardest questions a biostatistician or a breast pathologist could add.

## Where the honest answer is weak

Prepare these first. Each is a real limitation, not a wording problem.

- **Few patients per class.** Normal, benign and in situ test images each come from only 6 patients; 26 of the 30 in situ images come from 2 (questions 3, 9, 10).
- **Leakage cannot be fully excluded** for the 255 training and calibration images without patient IDs (question 4).
- **The test split influenced two choices:** the tier set, and the decision to try 512-px regions (question 12).
- **Calibration is rough:** log-loss did not improve, and five errors had at least 97% confidence, one of them an in situ carcinoma read as normal and placed in the lowest tier (questions 14, 19).
- **The reliability flag catches 42% of test errors**; its out-of-distribution and second-model parts are at chance (question 30).
- **The deletion test cannot fail for the production read.** The nine features explain a different, weaker model, and their spot check was done by an AI (questions 27–29).
- **The p63/calponin rule fired for 1 of the 3** in situ images read as invasive (question 20).
- **TCGA:** the FFPE-to-frozen gap is not statistically established, 7 of 17 normal slides still reached tiers 1–2, and FFPE specificity is unmeasured (questions 32–34).
- **The queue gain needs a 90%-busy reader** (question 36). There are no prospective data (question 37).
- **No atypia** (question 18). The human comparison is not head-to-head (question 40).

## Corrections to earlier answers

- The FFPE-to-frozen drop was called "the measured cost of frozen-section artifacts". Its interval includes zero (question 34).
- The heatmap was said to "pass a faithfulness test". For a linear mean-pooled read that arm passes by construction (question 28).
- The plain-language tab was described with fixed model counts. The texts are being regenerated; each shows its model (question 43).
- "Sepia's main error is calling in situ invasive." The most frequent error is benign called normal (9 of 18). The most dangerous error is an in situ carcinoma read as normal (question 19).
- The Quilt near-duplicate distance was given "for scale" as if it showed separation. The closest Quilt image is about as close as an ordinary different BACH image (question 5).
- "No patient appears in two splits" holds only for images with patient IDs (question 4).

## A. Data, splits and leakage

**1. Are the 400 items original microscopy images, patches or whole-slide images? How many patients?** (asked)
They are 400 original BACH microscopy images, 100 per class, each one 2048 × 1536 field at 0.42 µm per pixel. They are not whole slides. Each image is read as 12 regions of 512 px. Patient IDs exist for 145 images from 29 patients. All 132 test images have IDs, from 24 patients. The 10 annotated BACH whole slides are used only for simulated margins and feature checks.
Source: `docs/FACT_SHEET.md` §Data units; `artifacts/eval/fact_sheet.json` `bach`, `splits`.

**2. Which images went into training, calibration and test? Can one patient appear on both sides?** (asked)
We use the published THUNDER split: 218 training, 50 calibration and 132 test images. The 213 and 238 said aloud were wrong. Test has 23 normal, 48 benign, 30 in situ and 31 invasive images. No identified patient appears in two splits, and all 12 regions of an image stay with it. The patient check is complete for the test split but not for training (question 4).
Source: `fact_sheet.json` `splits`, `disjointness`.

**3. How many patients actually stand behind each test class?**
Fewer than the image counts suggest. Invasive images come from 17 patients, but normal, benign and in situ come from 6 each. One patient contributes 36 test images. Two patients supply 26 of the 30 in situ images. One patient holds 7 of the 18 errors; without that patient, accuracy would be 90.9%. Per-class results describe a handful of patients.
Source: `artifacts/eval/analytics.json` `groups_by_class`; computed from `data/cases.csv` and `data/predictions.csv` (appendix A1).

**4. Images without patient IDs are grouped in blocks of 5. Could a test patient's images be in training?**
We cannot fully exclude it. All 132 test images have workbook patient IDs, but 255 training and calibration images do not. BACH numbers each patient's files consecutively. Five test patients sit next to unidentified training files, so overlap at those edges is possible if the workbook is incomplete. Blocks of 5 can also split one patient across folds. That may explain why cross-validation (macro-F1 0.955) beat test (0.860).
Source: `data/cases.csv` (file numbers by patient, appendix A2); `pipeline/01_build_cases.py`; `artifacts/eval/model_choice.json` `cv_metrics`.

**5. How were the ~27,000 Quilt images selected? Could a test image be in Quilt?** (asked)
Of 1,017,708 Quilt-1M caption rows, 26,837 mention breast. Caption patterns picked 5,436 image–feature rows. They fit the explanation features only; Quilt never trains the classifier or enters evaluation. The closest Quilt–BACH pair has CONCH similarity 0.935, about as close as an ordinary different BACH image (median 0.914). No BACH region has a Quilt neighbor above 0.95. This is a threshold screen, not a pixel-level duplicate search.
Source: `fact_sheet.json` `corpora`, `quilt_near_duplicates`.

**6. Were the demo cases cherry-picked?**
They are picked by a fixed rule (`demo.heroes` in `config/settings.yaml`) that prefers correct reads. No correct biopsy read matched the rule. So the case we show, BACH_0167, is one of the three in situ images misread as invasive. All 132 test images are in the app, and all 18 misreads are listed in the fact sheet.
Source: `config/settings.yaml` `demo`; `data/meta/08_build_case_json.json` `heroes`; `docs/FACT_SHEET.md`.

## B. Statistics

**7. Did you run k-fold cross-validation, or a single split?** (asked)
We did both. Patient-grouped 5-fold cross-validation, repeated 3 times inside the 218 training images, chose the model. A single held-out split of 132 images reports it. The temperature was fit on out-of-fold training predictions and the conformal threshold on the 50 calibration images. The test split never picked the model, but it was read more than once (question 12).
Source: `artifacts/eval/model_choice.json` `protocol`.

**8. What does 98–99% AUROC measure in a four-class task? How many invasive test images? What mistakes?** (asked)
The headline, invasive-vs-rest AUROC 0.999 [0.996, 1.000], ranks invasive images against all others by P(invasive). Macro one-vs-rest AUROC is 0.965. All 31 invasive images were called invasive, and specificity was 97.0% (98 of 101). Accuracy was 86.4% (114 of 132). Of the 18 misreads, 9 were benign called normal, 3 normal called benign, 3 in situ called invasive and 3 in situ called normal or benign.
Source: `fact_sheet.json` `test`; `docs/FACT_SHEET.md` misread table; appendix A4.

**9. With 132 images from 24 patients, how wide are the intervals?**
They are wide. Accuracy is 86.4% [76.0, 94.4]; macro-F1 is 0.860 [0.678, 0.928]. By class, with the patient bootstrap, recall is 0.80 [0.33, 0.87] for in situ, 0.81 [0.42, 1.00] for benign and 0.87 [0.64, 1.00] for normal. One image moves a class's recall by 2–4 points. These intervals rule out a useless model, not a mediocre one. They cannot separate Sepia from pathologists.
Source: `artifacts/eval/metrics.json` `production`; `artifacts/eval/human_baseline.json` `sepia.recall`.

**10. Images from one patient are correlated. Are your exact intervals, which assume independence, too optimistic?**
Yes. Correctness clusters within patients (intraclass correlation 0.15, design effect 2.4). The 132 images carry about as much information as 55 independent ones. For accuracy, an independence interval is 79.3–91.7%; a cluster-robust one is 77.3–95.4%, close to our patient bootstrap. For 31 of 31 invasive, the exact 88.8–100% counts images; counting the 17 patients gives 80.5–100%.
Source: computed (appendix A3); `metrics.json`.

**11. You scored about 19 models on the test split. Isn't the best one a fluke?**
The production model was fixed by cross-validation before the test read. It also happens to rank first of 19 on test, but no difference is significant. Paired macro-F1 differences: +0.002 [−0.041, +0.040] against the same encoder in the fine-tuning pipeline, +0.023 [−0.044, +0.073] against attention MIL and +0.030 [−0.013, +0.118] against the round-1 model. Our 86.0 against THUNDER's published 84.7 is not a win either: different protocol, wide interval.
Source: `metrics.json` `per_model`; `artifacts/eval/thunder.json`; computed (appendix A5).

**12. Did test results influence any decision?**
Yes, twice, and both are disclosed. First, the pre-registered conformal set (LAC) covered 72.0% on test, so we switched the tiers to APS-never-empty (91.7%); that number is therefore not an independent confirmation. Second, round-1 test errors prompted the 512-px round, though cross-validation made the choice. Across 1,000 re-splits of the same held-out images, the tier set averages 94.5% and LAC 89.3%.
Source: `artifacts/eval/conformal.json`, `conformal_resplit.json`; `docs/ACCURACY_ROUND2.md`; `docs/RESULTS.md`.

**13. What do the displayed percentages mean? Are they calibrated?** (asked)
They are temperature-scaled class probabilities for this image, not risks of progression or recurrence. For BACH_0087 they are normal 14.9%, benign 1.7%, in situ 42.7% and invasive 40.7%; the missing 2–3% in the conversation was benign. Its conformal set {normal, in situ, invasive} keeps invasive, so the case goes to tier 2 with p63/calponin drafted. Feature scores are in standard deviations from the training-image average.
Source: `fact_sheet.json` `case_examples`; `artifacts/eval/calibration.json`.

**14. Is calibration measured on 132 images meaningful?**
Only roughly. Calibration error (ECE) fell from 0.184 to 0.055, but its interval reaches 0.185. Log-loss did not improve (0.571 to 0.575); the Brier score did (0.285 to 0.226). Reads at 95% or more confidence were right 68 of 73 times. Five errors had at least 97% confidence: four benign images called normal and one in situ carcinoma called normal at 99% (BACH_0112).
Source: `calibration.json`; computed from `data/predictions.csv` (appendix A3).

**15. What does the 90% conformal guarantee assume, and does it hold on TCGA?**
It assumes the calibration and new cases are exchangeable, and it holds on average over calibration sets. With 50 calibration images, one draw can land anywhere from about 81% to 97%. Our split broke the assumption: under exchangeability, a coverage as low as LAC's 72.0% has probability about 0.0005. TCGA has a different site, preparation and scanner, so we claim no coverage there. Its frozen-section sets average 2.1 classes, against 1.30 on BACH.
Source: `conformal.json` `finite_sample` (Beta(46, 5)); `artifacts/eval/external_tcga.json` `crops`; computed (appendix A6).

**16. Why did the attention model do worse? What was controlled?** (asked)
It did not do clearly worse. Embeddings, training images, folds and selection rule were the same as for the linear probe. Its cross-validated log-loss was 0.183 ± 0.051, against 0.178 ± 0.059 for the probe. That is within one standard error, so the rule took the simpler model. On test, macro-F1 was 0.837 vs 0.860, a paired difference of −0.023 [−0.073, +0.044]. "Too complex for 218 images" is a hypothesis, not a finding.
Source: `artifacts/eval/attention_comparison.json`; `docs/ATTENTION_COMPARISON.md`.

## C. Pathology

**17. Isn't BACH too clean? Can a pathologist tell these apart easily?** (asked)
Yes, and that is the main limitation. BACH discarded images its two annotators disagreed on and confirmed doubtful ones by immunohistochemistry, so hard cases are under-represented. Two signs of optimism: cross-validation macro-F1 was 0.955 against 0.860 on test, and the calibration split was easier than test. That is why we tested on TCGA. A reader study on uncurated material is the next step.
Source: `config/literature.yaml` `aresta2019_bach`, `bach_dataset_page`; `docs/HUMAN_BASELINE.md`.

**18. BACH has no atypia class. What does Sepia do with atypical ductal hyperplasia?**
Something untested. It must assign one of its four classes, and nothing marks a lesion as outside them. Its out-of-distribution signal does not predict errors (question 30). Atypia is where pathologists agree least: 48% concordance with the expert reference in B-Path. Sepia offers no evidence there. An atypia claim would need a labeled dataset that contains atypia.
Source: `literature.yaml` `elmore2015_bpath`; `artifacts/eval/reliability.json`.

**19. In situ vs invasive is the hard boundary. What did the model get wrong?**
Three of 30 in situ images were called invasive: BACH_0167, BACH_0364 and BACH_0389, from two of the six in situ patients. No invasive image was called in situ. The more dangerous error runs the other way. Three in situ images were called normal or benign. One of them, BACH_0112, was called normal at 99% with no flag and landed in the lowest tier.
Source: `docs/FACT_SHEET.md` misread table; `artifacts/cases/BACH_0112.json`.

**20. Does the p63/calponin rule catch those errors?**
It caught one of three. The rule drafts p63 and calponin when both in situ and invasive are in the set. On test it fired for 6 biopsy images, all truly in situ. Among the three over-calls, only BACH_0167 got it. BACH_0364's set was invasive alone, so only ER/PR/HER2/Ki-67 was drafted. BACH_0389 is in the simulated frozen lane, where stains are never drafted. The real safeguard is the pathologist's read.
Source: `config/stain_rules.yaml` `breast_insitu_vs_invasive`; `artifacts/cases/*.json` `agent.orders` (appendix A4).

**21. Is Nottingham grading three components or nine features? Which did you implement?** (asked)
Nottingham grading scores tubule formation, nuclear pleomorphism and mitotic count, 1–3 each, for a total of 3–9. Sepia computes no grade, because BACH labels are diagnostic categories. Three of its nine named features mirror those components, and the app labels them "Nottingham components (features, not a grade)". The AI spot check accepted tubules and pleomorphism (18 of 20 each) but not mitoses (0 of 20).
Source: `config/concepts.yaml` (`group: nottingham`); `artifacts/eval/spotcheck.json`.

**22. What do "Nottingham components" mean on an in situ case, and can one field show mitoses?**
On an in situ case they mean only that the image scores high or low on text descriptions of those features. Nottingham grading applies to invasive carcinoma. A BACH image covers 0.86 × 0.65 mm. The encoder sees each 512-px region resized to 224 px, about 0.96 µm per pixel. That is less area and resolution than a mitotic count needs, consistent with the 0-of-20 spot check.
Source: `fact_sheet.json` `bach`; `docs/EXTERNAL_TCGA_R2.md` (preprocessing); `spotcheck.json`.

**23. Frozen sections in the app are simulated. How?**
A quarter of the test images (33 of 132) are assigned to a simulated frozen-section lane. Degraded copies add blur, JPEG compression, stain shift and resampling at three strengths. These are not freezing artifacts such as ice crystals, folds or thick sections. They barely change accuracy (81.8–87.1%); stain shift changed nothing. The only real frozen tissue we tested is TCGA, where tumor-slide sensitivity was 84.0%.
Source: `config/settings.yaml` `lanes`, `degrade`; `reliability.json` `accuracy_vs_degradation`; `external_tcga.json`.

**24. How are margins assessed?**
They are simulated. The four image borders stand in for inked margins: there is no ink, no anterior or posterior margin and no specimen orientation. Clear margins stayed clear in 64 of 64 composite fields, and no benign image had a margin wrongly called involved (0 of 71). On annotated whole-slide windows, though, sensitivity was only 45.4% [25.4, 65.2]. It is a demonstration, not a margin assessment.
Source: `artifacts/eval/margin.json`; `docs/RESULTS.md` §Simulated margin read.

**25. Why does a confidently called DCIS sit in the lowest tier?**
The tier rule escalates only invasive carcinoma, the time-critical call in this design. A confident in situ read with a one-class set gets tier 4; 12 of the 23 in situ images in the biopsy lane did. In the simulation, tier-4 cases wait 41.4 minutes on average, against 20.0 in arrival order. Whether DCIS should be escalated is a clinical policy question we have not tested.
Source: `core/tiers.py`; `artifacts/cases/*.json` `flag.tier` (appendix A4); `artifacts/eval/queue_sim.json`.

## D. Explanations and confidence

**26. Why is the benign look-alike different? Does Sepia name the distinguishing features?** (asked)
Each look-alike shows the two features that differ most, with their values and spot-check status. A feature that failed its spot check is greyed out and not offered as a reason. Across 328 cases, 48 look-alikes have two passing differences, 170 one and 110 none. For BACH_0087 the passing difference is benign stroma or fat, +1.06 SD against −1.95 SD. These are associations in the concept model (question 27).
Source: `artifacts/cases/*.json` `contrast_case.detail`; `core/contract.py` `SPOTCHECK_RELIABLE_MIN` (0.75).

**27. Do the region heatmap and the nine features explain the same model?**
No. The heatmap shows exact region contributions of the production read, a linear probe on mean-pooled UNI2-h embeddings; constant plus contributions reproduces its invasive logit to within 1e-6. The nine features come from a separate concept-bottleneck model, `cb_text`. Its test macro-F1 is 0.708, and its top class matches production on only 95 of 132 test images. The features describe that model, not the production read.
Source: `artifacts/eval/tile_contrib.json`; `model_choice.json` `contributions_model`; `metrics.json` `per_model.cb_text`; computed agreement (appendix A3).

**28. Is your deletion test a valid faithfulness test?**
For the production read it is a sanity check, not strong evidence. The read is linear in the mean of its region embeddings. Removing the three highest-contribution regions must therefore lower the logit at least as much as removing any other three, so that arm cannot fail. It shows the evidence is concentrated: 0.46 logit, or 3.4 probability points. The separate tile-probe heatmap passes (0.22 [0.13, 0.33]); concept-feature regions fail (−0.02 [−0.10, +0.04]).
Source: `artifacts/eval/deletion.json` `models.uni2h512_lr.predicted_aggressive`; `eval/deletion_test.py`.

**29. Who checked that each feature shows what its name says?**
An AI, not a pathologist. Claude Opus 5.5 reviewed the 20 top-scoring test regions per feature, without labels: 98 of 180 acceptable, 36 not, 46 unsure. Four features reached the 75% cut-off: tubules, pleomorphism, invasion and benign stroma or fat. Five failed, including mitoses and normal lobules at 0 of 20; they are never given as reasons, and the oncology inbox omits them. The check removes reasons; it does not validate them.
Source: `spotcheck.json` (`reviewer_kind: ai`); `core/contract.py`.

**30. Does the reliability flag catch errors?**
Partly. Its error AUROC is 0.69 [0.58, 0.87]. It catches 42% of test errors, against an 80% design target and 87% on the calibration data it was fit on. Its out-of-distribution and CONCH–MUSK parts do not predict errors on test (AUROC 0.46 and 0.49); set size carries the signal. At the pre-registered 20% deferral, plain top-class probability did better (91.3% vs 89.8% kept accuracy).
Source: `reliability.json` `error_recall_test`, `feature_auroc_test`; `artifacts/eval/deferral.json`; `docs/RESULTS.md` §Deferral.

**31. What does the "window of five" mean?** (asked)
It is the five nearest training images by cosine similarity of their mean CONCH embedding. The library holds only the 218 training images, so a test image never retrieves itself or another test image. On average, 67% of the five share the test image's label (precision@5 0.67 [0.58, 0.83]). By class that is 92% for normal, 91% for invasive, 55% for in situ and 47% for benign.
Source: `artifacts/eval/retrieval.json`.

## E. External validity

**32. What happens on a different site, scanner, stain or protocol?** (asked)
Under a pre-registered protocol, we ran the production model unchanged on TCGA-BRCA slides from other hospitals and scanners: 25 patients, 65 slides, up to 6 random tissue-rich fields each. Sepia called 22 of 23 FFPE tumor slides and 21 of 25 frozen tumor slides invasive. It called none of 17 frozen normal slides invasive, but 7 of those 17 still had a field in tiers 1–2.
Source: `external_tcga.json` `primary`, `slides`; `docs/EXTERNAL_TCGA_R2.md`.

**33. What does TCGA actually test?**
Only invasive tumor against adjacent normal tissue. Labels are slide-level sample types (Primary Tumor or Solid Tissue Normal) with no tumor outlines, so a "tumor" field may be stroma. TCGA has no benign or in situ cases, so four-class accuracy and the in situ boundary are untested. Normal tissue exists only as frozen slides, so FFPE specificity is unmeasured. The cohort is 25 patients, and 8 normal slides had no usable field.
Source: `external_tcga.json` `limitations`, `slides_without_crops`.

**34. Is the 11.7-point drop from FFPE to frozen the cost of frozen-section artifacts?**
It is consistent with that, but not established. The patient-bootstrap interval for the gap runs from −4.7 to +28.0 points. Among 23 patients with both slide types, 3 were caught only on FFPE and 1 only on frozen (exact McNemar p = 0.63). In the first 8 patients the gap went the other way (−12.5 points). FFPE and frozen slides also come from different sections. Our earlier wording overstated this.
Source: `external_tcga.json` `primary`, `cohorts.same8`; computed (appendix A6).

## F. Workflow

**35. Does "20 to 6 minutes" refer to model runtime, a simulation, or a hospital measurement?** (asked)
It is a simulation: one reader at 9 minutes per case, 6 arrivals per hour over 8 hours, cases drawn from the test split, 1,000 simulated days. The mean wait before an invasive case is first read falls from 20.2 to 6.0 minutes. Unflagged cases wait 21.4 minutes longer (20.0 to 41.4). Model runtime is separate (0.73 s per image on a laptop), and the closeness to the 20-minute frozen-section benchmark is coincidence.
Source: `queue_sim.json`; `external_tcga.json` `latency_s_per_image`; `literature.yaml` (Novis 1997).

**36. How much does that depend on your assumptions?**
A lot. The simulated reader is 90% busy (6 cases × 9 minutes each hour). At 75% load, the invasive wait falls from 10.6 to 4.6 minutes and unflagged cases wait 8.9 minutes longer. At 60% it is 5.7 to 3.3, with 3.5 minutes added; at 45%, 3.2 to 2.3, with 1.4 added. A quarter of simulated cases are frozen sections that go first whatever the model says.
Source: computed by re-running `eval/queue_sim.py` at 3–6 arrivals per hour (appendix A7).

**37. Do you have any prospective or real-workflow data?**
No. We have no real arrival or reading times, no pathologist used Sepia on live cases, and no outcome was measured. Every workflow result is simulated or app-only. The next step is a silent-mode study: score real cases in parallel with the output hidden, and record timestamps and agreement before any visible use.
Source: `docs/SCOPE_MATRIX.md`; `docs/WSI_AND_EXTENSION.md`.

**38. Are the stain orders and frozen-section timers connected to real systems?** (asked)
No. Rules draft the stain orders, and the pathologist approves or edits them. Orders are approved in Sepia, and nothing is sent to a laboratory or ordering system. The frozen-section timer runs on simulated arrivals. It turns amber at 15 minutes and red at 20, following the CAP 20-minute benchmark.
Source: `docs/SCOPE_MATRIX.md`; `config/stain_rules.yaml`.

**39. How would this handle whole slides, and other organs?** (asked)
For whole slides, tile all tissue at the same resolution and read 12-region fields as now. Then take the maximum over fields and slides, and refit calibration at slide level. TCGA shows why the refit is needed: 7 of 17 normal slides had at least one field in tiers 1–2. For another organ, change the labels, feature descriptions, stain rules and calibration. Nothing beyond breast is validated.
Source: `docs/WSI_AND_EXTENSION.md`.

## G. Human baseline

**40. Your human comparison uses different images and readers who saw one field. Is it fair?**
It is context, not a comparison. In the BACH reader study, four pathologists read the 100 hidden test images, 25 per class from different patients, seeing only the field. They scored 73–96% accuracy, mean 85%. Sepia read 132 different public images with unequal classes, so its balanced accuracy, 0.871 [0.676, 0.938], is the closer analogue. Its interval overlaps most of the readers' range. No reader study was run on Sepia's images.
Source: `human_baseline.json`; `literature.yaml` `aresta2019_bach` (Table 4); `docs/HUMAN_BASELINE.md`.

## H. Engineering and provenance

**41. What did the team train, and what was reused?** (asked)
Reused and frozen: UNI2-h, the production encoder; CONCH, for feature scores, similar cases and out-of-distribution distance; MUSK, for the agreement check. H-optimus-0, Virchow2, Phikon-v2 and PathGen-CLIP were only evaluated. We trained the linear probe, temperature and conformal threshold, the feature vectors and concept model, the reliability combiner, the tile and margin probes, and unused attention-MIL and LoRA variants. Tiers and orders are rules. "UDIP2" in the recording is UNI2-h.
Source: `fact_sheet.json` `components`.

**42. What compute did you use?**
Delta A100 GPUs did the embedding with seven encoders, HoVer-Net cell segmentation and fine-tuning. That came to about 4.1 A100-hours measured with `sacct`, 2.2 of them fine-tuning, within one person's 10 GPU-hours. Everything else ran on an Apple M3 Max laptop: the probes, calibration, evaluation and TCGA scoring, at 0.73 s per field.
Source: `docs/RESOURCE_USAGE.md`; `docs/ACCURACY_ROUND2.md`; `external_tcga.json` `latency_s_per_image`.

**43. What did language models do?**
None of them makes the diagnosis. All 328 case summaries come from a deterministic template. The optional plain-language tab is written by a language model through a local CLI; every number is checked against the evidence, and a model chip names the model on each text. The site serves cached texts only. Claude Opus 5.5 did the spot check, and AI coding agents wrote most of the code under our direction (285 of 327 commits).
Source: `fact_sheet.json` `components`; `artifacts/interpretations/case/`; `spotcheck.json`; `git log` (at commit 3a0e4d1).

## Appendix: reference numbers and how the computed ones were made

All "computed" numbers use the production model's test rows in `data/predictions.csv` (top class `top`, calibrated `pcal__*`) joined to `data/cases.csv` (`label`, `group`), with the 132 clean test images only. They can be reproduced with a few lines of pandas.

**A1. Test images by patient group** (`data/cases.csv`, split test; groups are real patients in the test split).

| Patient | Images | Classes | Errors |
|---|---|---|---|
| p14 | 36 | 21 benign, 12 in situ, 3 invasive | 4 (2 benign→normal, 2 in situ→invasive: BACH_0167, BACH_0389) |
| p23 | 14 | 14 in situ | 2 (BACH_0034 → benign, BACH_0112 → normal) |
| p17 | 18 | 14 benign, 4 normal | 2 benign→normal |
| p21 | 11 | 6 benign, 5 normal | 7 (5 benign→normal, 2 normal→benign) |
| 20 other patients | 53 | 28 invasive, 14 normal, 7 benign, 4 in situ | 3 (BACH_0219, BACH_0293, BACH_0364) |

Patients per class: normal 6, benign 6, in situ 6, invasive 17 (`analytics.json` `groups_by_class`).

**A2. File-number boundaries.** Within each class, a patient's file numbers are consecutive. Test patients p14 (b091, is070, iv074), p16 (n045), p21 (n069), p23 (is039) and p27 (iv040) border training files that have no patient ID. Computed by listing, for each test patient and class, the files just outside its number range and their split.

**A3. Clustering and calibration.**

- **Accuracy intervals.** Independence: Clopper–Pearson 79.3–91.7%. Cluster-robust: a ratio-estimator standard error over the 24 patients, 0.046 against 0.030 binomial, giving 77.3–95.4% (normal) or 76.8–95.9% (t with 23 df). Design effect 2.4; effective n about 55.
- **Intraclass correlation.** 0.15, the one-way ANOVA estimator for unequal clusters.
- **Patient-level invasive bound.** 17 of 17 patients with every invasive image called invasive gives an exact lower bound of 80.5%. Rule of thumb: an all-correct result needs 72 patients for an exact lower bound of 95%.
- **In situ called invasive.** 3 of 30 images (exact 2.1–26.5%), from 2 of 6 patients.
- **Confident reads.** 73 reads had top-class probability ≥ 0.95, and 68 were correct.
- **Concept model vs production.** The `cb_text` top class matched the production top class on 95 of 132 test images.

**A4. Tiers and orders on the test split** (`artifacts/cases/*.json`, 132 clean test images).

- **Lanes.** 33 are in the simulated frozen lane (tier 1); 99 are biopsies (tier 2: 28, tier 3: 22, tier 4: 49).
- **Invasive in the set.** 40 images had Invasive as the top class or in the set: 31 invasive and 9 in situ, with no benign or normal image.
- **p63/calponin.** It was drafted for 6 biopsy images, all in situ.
- **In situ in tier 4.** 12 of the 23 in situ biopsy images were in tier 4.

| True \ Predicted | Normal | Benign | In situ | Invasive |
|---|---|---|---|---|
| Normal (23) | 20 | 3 | 0 | 0 |
| Benign (48) | 9 | 39 | 0 | 0 |
| In situ (30) | 2 | 1 | 24 | 3 |
| Invasive (31) | 0 | 0 | 0 | 31 |

**A5. Paired test differences.** Macro-F1 of production minus each comparator: a cluster bootstrap over the 24 patients (2,000 resamples, seed 44) on the same images.

| Comparator | Difference [95% CI] |
|---|---|
| `uni2h_frz_lr` (same encoder, fine-tuning pipeline) | +0.002 [−0.041, +0.040] |
| `uni2h_wise25_lr` (fine-tuned, WiSE-FT α = 0.25) | +0.008 [−0.040, +0.057] |
| `fusion_top3` | +0.015 [−0.044, +0.057] |
| `abmil` | +0.023 [−0.044, +0.073] |
| `uni2h_lr` (round 1, 256 px) | +0.030 [−0.013, +0.118] |
| `hoptimus0_512_lr` | +0.049 [−0.001, +0.098] |

**A6. Conformal and TCGA.**

- **Calibration-draw spread.** Coverage for one random calibration draw follows Beta(46, 5): 2.5th–97.5th percentiles 80.8–96.7%; P(coverage ≤ 0.72) = 0.0005.
- **TCGA set sizes.** Mean 1.41 classes on FFPE tumor fields, 2.13 on frozen tumor and 2.11 on frozen normal, against 1.30 on BACH test.
- **TCGA field reads.** Frozen tumor fields were read as invasive 70, in situ 69 and normal 6 times (of 145). Frozen normal fields were read as normal 60 and in situ 10 times (of 70).
- **Normal slides in tiers 1–2.** 7 of 17 frozen normal slides had a field in tiers 1–2 (exact 18.4–67.1%). With deterministic sets (pre-registered S3) no frozen normal field reached tier 2.
- **FFPE vs frozen gap.** Patient bootstrap of the slide-level difference (4,000 resamples, seed 44): −4.7 to +28.0 points. Exact McNemar test on the 23 patients with both slide types: 3 vs 1 discordant, p = 0.63.

**A7. Queue sensitivity.** `eval/queue_sim.py` `simulate()`, run with the production pool and 1,000 days per setting. Only the arrival rate changes; the read time stays 9 minutes.

| Arrivals per hour (load) | Invasive first read, FIFO → tiers (min) | Extra wait for unflagged cases (min) |
|---|---|---|
| 3 (45%) | 3.2 → 2.3 | 1.4 |
| 4 (60%) | 5.7 → 3.3 | 3.5 |
| 5 (75%) | 10.6 → 4.6 | 8.9 |
| 6 (90%, configured) | 20.2 → 6.0 | 21.4 |

**A8. TCGA results** (`external_tcga.json`, slide level, 95% bootstrap over patients).

| Measure | Result |
|---|---|
| FFPE tumor slides called invasive | 95.7% [87.0, 100] (22 of 23) |
| Frozen tumor slides called invasive | 84.0% [68.0, 96.0] (21 of 25) |
| Frozen normal slides not called invasive | 17 of 17 (exact 80.5–100%) |
| Frozen tumor vs frozen normal AUROC | 1.000 (all 25 above all 17; no interval) |
| Tumor slides with a field in tiers 1–2 | 97.9% [93.6, 100] (47 of 48) |
| Frozen normal slides with a field in tiers 1–2 | 7 of 17 (computed) |
