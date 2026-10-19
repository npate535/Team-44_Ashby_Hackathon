# Human reader baseline and clinical timing benchmarks

> Research prototype, not for clinical use. Published figures below come from `config/literature.yaml`, which records for each figure the abstract, section or table it was read from. Sepia figures come from `artifacts/eval/metrics.json` (production model, held-out THUNDER test split of the BACH microscopy images, n = 132 images, 24 patient / pseudo-patient groups) and `artifacts/eval/queue_sim.json`. No human figure on this page was measured by the team.

## Human readers on the same diagnostic categories

### Two published reader studies

**B-Path (Elmore 2015).** 115 US pathologists in 8 states each read 60 breast biopsy cases from a set of 240 (1 H&E slide per case, 6900 interpretations), with no time limit. The reference diagnosis was the consensus of three expert breast pathologists. The 240 cases were 23 invasive carcinoma, 73 DCIS, 72 atypia and 72 benign without atypia, a higher prevalence of difficult cases than typical practice (Elmore 2016). Overall concordance with the reference was 75.3% (95% CI 73.4–77.0%). Even the three experts agreed unanimously on only 75% of cases before consensus.

**BACH reader study (Aresta 2019, Section 4.1.1 and Table 4).** Four pathologists classified the 100 hidden BACH test images (25 per class), seeing only the field image, which is the same information the algorithms had. One of them (E1) had co-labelled BACH and re-read the images one month later. Their four-class accuracies were 96% (E1), 94%, 78% and 73%. The best challenge algorithms reached 87% on the same images.

### B-Path categories next to Sepia

B-Path reports, for each reference category, how often a pathologist's call was the same (concordant), higher (overinterpreted) or lower (underinterpreted) on the ordered scale benign without atypia < atypia < DCIS < invasive. Sepia's counterparts below use the same ordering, with BACH Normal and Benign together standing in for "benign without atypia". They are counted from the test confusion matrix (`metrics.json`, `production.confusion`).

| Reference category | B-Path pathologists vs consensus: concordant / over / under (95% CI) | B-Path experts, independent reads vs consensus: concordant | BACH counterpart | Sepia, test split: concordant / over / under |
|---|---|---|---|---|
| Benign without atypia | 87% (85–89) / 13% (11–15) / – | 91% | Normal + Benign (71 images) | 71/71 (exact 95% CI 94.9–100%) / 0/71 (exact 95% CI 0–5.1%) / – (within this pooled category Sepia swapped Normal and Benign in 12 of 71; B-Path has no such split) |
| Atypia | 48% (44–52) / 17% (15–21) / 35% (31–39) | 80% | No BACH class | Not measured |
| DCIS | 84% (82–86) / 3% (2–4) / 13% (12–15) | 97% | In situ carcinoma (30) | 24/30 (80.0%) / 3/30 (10.0%) called invasive / 3/30 (10.0%) called Normal or Benign |
| Invasive carcinoma | 96% (94–97) / – / 4% (3–6) | 97% | Invasive carcinoma (31) | 31/31 (100%; exact 95% CI 88.8–100%) / – / 0/31 |

B-Path columns: Elmore 2015, abstract and Table 3 (pathologists, 663–2097 interpretations per category); Table 1 (three experts, 23–73 cases per category).

### BACH four-class sensitivity: pathologists and Sepia

| Class | BACH pathologists E1 / E2 / E3 / E4 (hidden test set, 25 images per class) | Mean ± SD of the four | Sepia recall (THUNDER test split) |
|---|---|---|---|
| Normal | 0.96 / 0.96 / 0.88 / 0.40 | 0.80 ± 0.23 | 20/23 (87.0%) |
| Benign | 0.92 / 0.88 / 0.76 / 0.84 | 0.85 ± 0.06 | 39/48 (81.3%) |
| In situ carcinoma | 1.0 / 1.0 / 0.56 / 0.76 | 0.83 ± 0.18 | 24/30 (80.0%) |
| Invasive carcinoma | 0.96 / 0.92 / 0.92 / 0.92 | 0.93 ± 0.02 | 31/31 (100%; exact 95% CI 88.8–100%) |
| Four-class accuracy | 0.96 / 0.94 / 0.78 / 0.73 | 0.85 ± 0.10 | 0.864 [0.760, 0.944] (114/132) |

The pathologist columns are from Aresta 2019, Table 4. Sepia's accuracy interval is the 95% cluster bootstrap from `metrics.json`. For 31/31 the bootstrap collapses to a single point, so the exact Clopper–Pearson interval is given (`core.stats.exact_boundary_ci`). Sepia's invasive-vs-rest AUROC is 0.999 [0.996, 1.000] and its macro-F1 is 0.860 [0.678, 0.928].

### Why these are not directly comparable

- **Different unit of reading.** B-Path pathologists read a whole glass slide per case at the microscope. BACH and Sepia use a single curated 2048 × 1536 microscopy field per image, at 0.42 µm per pixel.
- **BACH labels are curated.** The official dataset page says images where the two annotators disagreed were discarded. The paper says images with Normal-versus-Benign disagreement were discarded and the remaining doubtful cases were confirmed by immunohistochemistry. Each image is labelled by its predominant type. So BACH under-represents the hard cases where pathologists disagree. B-Path did the opposite: it kept disputed cases, resolved them by consensus, and enriched the set for atypia and DCIS.
- **Atypia is not a BACH class.** Atypia is where B-Path pathologists agreed least (48%). Sepia has never seen it, and the comparison says nothing about it.
- **Different reference standards.** B-Path used a three-expert consensus by modified Delphi. BACH labels came from two experts who could see other regions of the patient's tissue and, potentially, immunohistochemistry. The BACH readers saw only the field.
- **Category mapping is approximate.** B-Path's "benign without atypia" covers both non-proliferative and proliferative lesions; BACH splits Normal from Benign. B-Path's DCIS is ductal; BACH's class is "in situ carcinoma".
- **Different images.** The BACH reader study used the 100 hidden test images. Sepia's test split is 132 of the 400 public images. The source and protocol match, but no image is shared.
- **Wide intervals at n = 132.** Sepia's classes have 23–48 test images, so one image moves a class's recall by 2.1–4.3 percentage points. The accuracy interval spans 0.760 to 0.944. The BACH reader study had 25 images per class; B-Path had 663–2097 interpretations per category.
- **Different reading conditions.** B-Path readers had no time limit and read a whole slide. BACH readers saw only the field image.

### What the comparison shows and does not show

It shows:

- On BACH's own images and classes, single-field reading by pathologists is not near-perfect. Accuracy ranged from 73% to 96% across four pathologists. In situ sensitivity ranged from 0.56 to 1.0 and Normal sensitivity from 0.40 to 0.96.
- For both pathologists and Sepia, invasive carcinoma is the most consistently recognised class: BACH readers 0.92–0.96, B-Path 96%, Sepia 31/31 (exact 95% CI 88.8–100%). Sepia's invasive-vs-rest AUROC of 0.999 is measured on the easiest boundary.
- Human disagreement concentrates in atypia (48% concordance) and, to a lesser extent, DCIS (84%). BACH excludes atypia, and it labels in situ carcinoma with curation (disagreements discarded, doubtful cases confirmed by immunohistochemistry).
- Sepia's main error is calling in situ carcinoma invasive: 3 of 30 (10.0%). B-Path pathologists overcalled DCIS as invasive in 3% (2–4%) of interpretations. The point estimate is higher for Sepia, but it rests on 30 images in a different setting.

It does not show:

- That Sepia matches or exceeds pathologists. No reader study was run on Sepia's test images, and the units, images and reference standards differ.
- Anything about atypia, microinvasion or whole-slide and whole-case reading. In B-Path, one invasive case was predominantly DCIS with a focus of microinvasion that 2 of the 3 expert panelists initially missed (Elmore 2015, Results).

In summary, BACH is an easier, curated task than breast biopsy practice, and human disagreement concentrates in the categories BACH mostly excludes. Sepia's high invasive-vs-rest AUROC is evidence that it orders clear invasive fields first. It is not evidence that it resolves the borderline cases where pathologists differ. Showing that would take a reader study on uncurated material.

## Where the clinical time goes

### Published benchmarks

| Interval | Benchmark | Source |
|---|---|---|
| Intraoperative frozen section, from the pathologist receiving the specimen to communicating the result to the surgeon | 90% of 32,868 frozen-section blocks were completed within 20 minutes (700 institutions, 1995). 70% of hospitals completed 90% of their frozen sections within 20 minutes. | Novis 1997, abstract |
| CAP accreditation checklist item ANP.11820, as quoted in September 2008 | Note: 90% of frozen sections should be completed within 20 minutes. Applies only to single-specimen, single-block cases during regular business hours. | CAP Today 2008 |
| Intraoperative frozen-section margin assessment in breast-conserving surgery | Mean 27 minutes (4 studies). Range 10–50 minutes (11 of 19 studies). | Esbona 2012, abstract and Table 3; Garcia 2021, Results |
| Routine surgical pathology, from accessioning to report completion | The CAP accreditation standard was 2 working days. Routine biopsies: 79% within 1 working day, 95% within 2, 98% within 3. Complex specimens: 56%, 81%, 91% and 95% within 1, 2, 3 and 4 working days; mean of participants' median turnaround 1.5 days. | Zarbo 1996, abstract |

### Margins in breast-conserving surgery

- **Margin standard.** For invasive cancer, an adequate margin is "no ink on tumor"; a positive margin doubles the risk of ipsilateral breast tumor recurrence (Moran 2014). For DCIS treated with whole-breast irradiation, the standard is 2 mm (Morrow 2016).
- **When the result arrives.** Without intraoperative assessment, margin status comes from permanent sections reported after the operation, and a positive margin usually means a second operation: 85.9% of initially positive margins were re-excised in McCahill 2012. Intraoperative frozen-section analysis "permits immediate re-excision, avoiding reoperation" (Boughey 2014).
- **How often surgery is repeated.**
  - Before the 2014 guideline, about 1 in 4 women attempting breast-conserving therapy had a re-excision (Moran 2014, Introduction).
  - In 2206 women at four US institutions (2003–2008), 22.9% (21.2–24.7%) underwent re-excision (McCahill 2012).
  - Pooled final re-excision rates were 35% with permanent sections alone and 10% with intraoperative frozen section (Esbona 2012).
  - The 30-day reoperation rate was 13.2% nationally (NSQIP) and 3.6% at an institution that routinely uses frozen-section margins (Boughey 2014).
- **Practice varies.** There is no consensus on intraoperative margin assessment (Garcia 2021, Introduction). Frozen-section margins had pooled sensitivity 0.81 and specificity 0.97 (Garcia 2021), and they add operating time (above).
- **On the "1–2 days" estimate.** None of the verified sources gives a measured turnaround for permanent-section margin reports on breast lumpectomies specifically. The nearest figure is general: in the CAP Q-Probes study, the mean of participants' median turnaround for complex specimens was 1.5 days, and 81% were reported within 2 working days (Zarbo 1996). That is consistent with results arriving after the operation, but it is not a breast-margin measurement.

### What Sepia's 20.2 → 6.0 minute figure is

Sepia's figure comes from a simulated queue, not a measured turnaround (`queue_sim.json`). The setup: Poisson arrivals at 6 per hour over an 8-hour day, one reader, a fixed 9 minutes per read, cases drawn with replacement from the test split, repeated over 1000 simulated days.

- Under arrival order, invasive test images waited a mean of 20.2 minutes before first being read. Under Sepia's tiers, they waited 6.0 minutes.
- The cost falls on everything else: the mean wait of unflagged cases rose from 20.0 to 41.4 minutes.
- The intervals (20.2 [3.5, 60.1] vs 6.0 [1.9, 12.9]) reflect simulation variability only.

This figure measures one interval: time waiting in a single reader's queue. It leaves out grossing, processing, sectioning, staining, transport and reporting, which are all inside the published turnaround benchmarks above. It cannot be compared with them.

The 20.2-minute arrival-order wait and the 20-minute frozen-section benchmark are numerically close by coincidence; they measure different things. Frozen sections and margins in Sepia are simulated.

## Second opinions

- **Staradub 2002.** Second-opinion review of 346 breast cancers (340 patients, 1997–2001) produced some change in 80%, mostly added prognostic information (40%). Major changes that altered surgical therapy occurred in 7.8%. Changes were more common in in situ than in invasive carcinoma (P = .004).
- **Khazai 2015.** Second review of 1970 breast pathology referrals in one year found a significant discrepancy, defined as one that affected patient care, in 226 cases (11.47%). Histologic category was the most common area of discrepancy (66 cases, 33%). Intraductal lesions were among the most problematic categories.
- **Elmore 2016.** On the B-Path data, a second opinion on every case lowered overall misclassification from 24.7% to 18.1%. Atypia remained misclassified in more than 34% of cases under every strategy. Second opinions limited to invasive cancer did not significantly reduce misclassification; every other strategy did.

Expert review changes breast diagnoses most often in in situ and borderline lesions. Those are the categories where Sepia's evidence is weakest: atypia is absent from BACH, and in situ recall is 24/30.

Sepia should therefore be described as a confirmation and ordering aid. It puts likely invasive fields first and shows the evidence for each read. It is not a second pathologist, it renders no diagnosis, and it has not been validated as a second reader.

## Different-task reference: CAMELYON16

Ehteshami Bejnordi 2017 compared algorithms with pathologists on a different task: detecting breast cancer metastases in lymph node whole-slide images. The test set had 129 slides, 49 with metastases and 80 without.

- The best algorithm reached AUC 0.994 (0.983–0.999).
- Eleven pathologists reading under a time constraint (a flexible 2-hour session simulating routine workflow) reached a mean AUC of 0.810 (range 0.738–0.884).
- One pathologist without a time constraint reached 0.966 (0.927–0.998), comparable to the top five algorithms (mean 0.960, range 0.923–0.994).

In a related reader study, Steiner 2018 had six pathologists review 70 lymph node slides with and without algorithm assistance:

- Sensitivity for micrometastases rose from 83% to 91%.
- Mean review time per image fell from 116 to 61 seconds for micrometastases and from 137 to 111 seconds for negative images.

These results do not bear directly on BACH. The task is binary metastasis detection on whole slides, with a different tissue and reference standard. They are relevant in two ways. First, pathologists reading under a time constraint scored well below the one reading without it, so any human baseline must state the reading conditions. Second, measured gains from AI assistance came from multireader studies with a wash-out period, which is the design Sepia would need before any claim about reading time or accuracy in practice.

## References

- Aresta G, et al. BACH: Grand challenge on breast cancer histology images. *Med Image Anal*. 2019;56:122-139. doi:10.1016/j.media.2019.05.010. PMID 31226662.
- BACH challenge organizers. Dataset. ICIAR 2018 Grand Challenge on BreAst Cancer Histology images. https://iciar2018-challenge.grand-challenge.org/Dataset/. Accessed October 18, 2026.
- Boughey JC, et al. Impact of analysis of frozen-section margin on reoperation rates in women undergoing lumpectomy for breast cancer: evaluation of the National Surgical Quality Improvement Program data. *Surgery*. 2014;156(1):190-197. doi:10.1016/j.surg.2014.03.025. PMID 24929768.
- College of American Pathologists. Q and A: Laboratory Accreditation News. *CAP Today*. September 2008. https://www.captodayonline.com/Archives/0908/0908_QA_LAP_1.html. Accessed October 18, 2026.
- Ehteshami Bejnordi B, et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. *JAMA*. 2017;318(22):2199-2210. doi:10.1001/jama.2017.14585. PMID 29234806.
- Elmore JG, et al. Diagnostic concordance among pathologists interpreting breast biopsy specimens. *JAMA*. 2015;313(11):1122-1132. doi:10.1001/jama.2015.1405. PMID 25781441.
- Elmore JG, et al. Evaluation of 12 strategies for obtaining second opinions to improve interpretation of breast histopathology: simulation study. *BMJ*. 2016;353:i3069. doi:10.1136/bmj.i3069. PMID 27334105.
- Esbona K, Li Z, Wilke LG. Intraoperative imprint cytology and frozen section pathology for margin assessment in breast conservation surgery: a systematic review. *Ann Surg Oncol*. 2012;19(10):3236-3245. doi:10.1245/s10434-012-2492-2. PMID 22847119.
- Garcia MT, et al. Accuracy of frozen section in intraoperative margin assessment for breast-conserving surgery: a systematic review and meta-analysis. *PLoS One*. 2021;16(3):e0248768. doi:10.1371/journal.pone.0248768. PMID 33735315.
- Khazai L, et al. Breast pathology second review identifies clinically significant discrepancies in over 10% of patients. *J Surg Oncol*. 2015;111(2):192-197. doi:10.1002/jso.23788. PMID 25273328.
- McCahill LE, et al. Variability in reexcision following breast conservation surgery. *JAMA*. 2012;307(5):467-475. doi:10.1001/jama.2012.43. PMID 22298678.
- Moran MS, et al. Society of Surgical Oncology–American Society for Radiation Oncology consensus guideline on margins for breast-conserving surgery with whole-breast irradiation in stages I and II invasive breast cancer. *Ann Surg Oncol*. 2014;21(3):704-716. doi:10.1245/s10434-014-3481-4. PMID 24515565. Also published in *Int J Radiat Oncol Biol Phys*. 2014;88(3):553-564. doi:10.1016/j.ijrobp.2013.11.012.
- Morrow M, et al. Society of Surgical Oncology–American Society for Radiation Oncology–American Society of Clinical Oncology consensus guideline on margins for breast-conserving surgery with whole-breast irradiation in ductal carcinoma in situ. *Ann Surg Oncol*. 2016;23(12):3801-3810. doi:10.1245/s10434-016-5449-z. PMID 27527714.
- Novis DA, Zarbo RJ. Interinstitutional comparison of frozen section turnaround time: a College of American Pathologists Q-Probes study of 32868 frozen sections in 700 hospitals. *Arch Pathol Lab Med*. 1997;121(6):559-567. PMID 9199619.
- Staradub VL, et al. Changes in breast cancer therapy because of pathology second opinions. *Ann Surg Oncol*. 2002;9(10):982-987. doi:10.1007/BF02574516. PMID 12464590.
- Steiner DF, et al. Impact of deep learning assistance on the histopathologic review of lymph nodes for metastatic breast cancer. *Am J Surg Pathol*. 2018;42(12):1636-1646. doi:10.1097/PAS.0000000000001151. PMID 30312179.
- Zarbo RJ, Gephardt GN, Howanitz PJ. Intralaboratory timeliness of surgical pathology reports: results of two College of American Pathologists Q-Probes studies of biopsies and complex specimens. *Arch Pathol Lab Med*. 1996;120(3):234-244. PMID 8629897.
