# Deck outline — DRAFT for team review (≤ 20 main slides; due Wed Oct 21)

Numbers are quoted from docs/RESULTS.md (held-out THUNDER test split, 132 images, 95% cluster-bootstrap CIs).
Figures: `artifacts/eval/*.png` (git-ignored; regenerate with `make eval`) or screenshots of the app's Model evaluation
page (`#/metrics`), which has the presentation-quality charts. Screens: the running app at `http://127.0.0.1:8000`.
BACH imagery is CC BY-NC-ND 4.0: cite Aresta et al. 2019 on every slide that shows it.

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | PathQueue | One line: an explainable, confidence-aware breast pathology worklist. Team 44. "Research prototype, not for clinical use." | App worklist screenshot |
| 2 | The problem | Cases are read in arrival order; AI flags rarely say why or how sure. Frozen sections need answers in ~20 min. | — |
| 3 | What we built | Two lanes (biopsy, simulated frozen section), four tiers, safety rule, drafted orders, routing to surgeon and oncologist. | Architecture diagram (plan §3) |
| 4 | Data and models | BACH 400 images, THUNDER patient-safe split (218 / 50 / 132); 7 encoders embedded on NCSA Delta (UNI2-h production; CONCH, MUSK, H-optimus-0, Virchow2, Phikon-v2, PathGen-CLIP); selection by grouped CV inside train only. | Model-choice forest plot |
| 5 | Triage accuracy | Invasive vs rest AUROC 0.998 [0.994, 1.000]; sensitivity 100%, specificity 97% [94, 100]; macro-F1 0.83 [0.61, 0.93] vs published UNI2-h probe 84.7 (same split, different protocol). | ROC + confusion matrix |
| 6 | The clinically important error | 3 of 30 in situ images called invasive, 0 of 31 invasive called in situ; why the p63/calponin rule and the safety rule exist. | Probability histograms by class |
| 7 | Honest confidence | ECE 0.10 → 0.05 after temperature scaling. APS set covers 90.2% [75.7, 98.1] vs 90% target (**chosen after viewing test results**); pre-registered LAC 72.7% with 19% empty sets — say why we switched. | Reliability diagram + coverage-by-method chart |
| 8 | Safety rule in action | All 31 invasive test images land in tiers 1–2; 13 in situ images carry Invasive in their set, so the safety rule sends them to tier 2 (tier 1 if they arrive in the simulated surgical lane). | Worklist tier placement heatmap |
| 9 | Demo (live or video) | DEMO_SCRIPT.md: BACH_0167 jumps the queue, {InSitu, Invasive} → p63/calponin + ER/PR/HER2/Ki-67 → sign-out → oncology inbox. | Live app |
| 10 | Why: named features | 9 breast criteria from text prompts + caption-derived concept vectors; levels in SD vs training cases; signed pushes; regions on the image. | Case "Why" panel screenshot |
| 11 | Are the explanations real? | Concept × class heatmap (all 8 directional concepts move the expected way); NCT-CRC cross-organ check; caption consistency; AI spot check: invasion 19/20, pleomorphism 18/20, tubules 18/20 but normal lobules 0/20 and mitoses 0/20; deletion test shows weak faithfulness. | Spot-check chart + heatmap |
| 12 | Grounded summaries | LLM (gemma-4-31b-it via Lumen) sees findings only; validator rejects un-cited numbers and contradictions (e.g. "margins clear" on a deferred read); required margin / quality sentences. | Summary panel with finding chips |
| 13 | Frozen-section lane (simulated) | Image borders as inked margins; log-odds margin signal vs a threshold set on benign calibration images; composites keep 100% of tumor-free margins clear, tumor-side margin found 70%; WSI windows sensitivity 45%, specificity 98%. | Surgical case view |
| 14 | When not to trust the read | Degradation curves (accuracy 84% → 67–73% at the worst severity); reliability score error AUROC 0.71; defer to permanent with a plain-language reason. | Degradation small multiples + defer panel |
| 15 | Workflow benefit and cost | Queue simulation: invasive first look 20.2 → 6.0 min; unflagged cases wait 25 min longer; aging guard has no measured effect at configured targets. | Queue-simulation boxplots |
| 16 | Data quality matters | A tissue-QC bug rejected pale tissue in 51 of 400 images (32 invasive); fixing it raised CV balanced accuracy 0.93 → 0.95 and test accuracy 83.3% → 84.1%. | QC before/after panel |
| 17 | Extending beyond breast | Organ-specific: classes, criteria, rules, labels; everything else reused (plan §17). | Table |
| 18 | Limitations | Small public dataset; simulated frozen lane, margins, arrivals; post-hoc conformal choice; illustrative stain rules; AI (not pathologist) spot check; no clinical validation. | — |
| 19 | Credits | BACH, THUNDER, CONCH, MUSK, UNI2-h, H-optimus-0, Virchow2, Phikon-v2, PathGen-CLIP, NCT-CRC-HE-100K, Quilt-1M / PathGen captions, HoVer-Net / PanNuke, Lumen (gemma-4-31b-it), NCSA Delta, `ncsa/ashby-hackathon-2026`. | — |

Before presenting: a team member should record a human spot check on the Evidence review page (`#/spotcheck`) so
slide 11 can show human and AI reviews side by side.
