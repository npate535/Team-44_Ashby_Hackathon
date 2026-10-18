# Abstract — DRAFT for team review (numbers checked against `artifacts/eval/*.json`, 2026-10-18)

**Sepia: an explainable, confidence-aware breast pathology worklist built on pathology foundation models**

*Team 44 · 2026 Ashby Prize in Computational Science Hackathon · research prototype, not for clinical use*

**Problem.** Pathologists read cases in arrival order, so an invasive carcinoma can wait behind benign biopsies, and
AI triage tools rarely say why they flagged a case or how far to trust the flag.

**Approach.** Sepia reads each BACH breast microscopy field (Aresta et al., 2019) as a grid of tiles embedded by
pathology foundation models on NCSA Delta (UNI2-h for the production read; CONCH and MUSK for text-aligned concepts
and agreement). A calibrated linear probe on 512-pixel regions suggests one of four categories; split-conformal prediction (APS) returns
the set of categories that cannot be excluded, and a safety rule sends any case with invasive carcinoma in that set
to the top of the worklist, behind only simulated frozen sections. Each read is explained by nine named histologic
features (e.g. tumor infiltrating stroma, nuclear pleomorphism, tubule formation) scored from text prompts and
caption-derived concept vectors, linked to image regions and similar reference cases. A rule engine drafts
follow-up orders (p63/calponin, ER/PR/HER2/Ki-67, deeper levels) that the pathologist approves; an LLM writes a
summary from structured findings only, checked sentence by sentence against those findings.

**Results (held-out THUNDER test split, 132 images, 95% cluster-bootstrap CIs).** Invasive-vs-rest AUROC 0.999
[0.996, 1.000]; invasive sensitivity 100% and specificity 97% [94, 100]; four-class macro-F1 0.86 [0.68, 0.93],
above the published UNI2-h linear probe on the same split (84.7; different protocol). After temperature
scaling, ECE 0.055. APS sets covered the true class in 91.7% [80.8, 98.1] of images against a 90% target (method
chosen after viewing test results; the pre-registered LAC reached 72.0%). All 31 invasive images landed in the top
two tiers. In a simulated reading day the tiers cut mean time-to-first-look for invasive cases from 20.2 to 6.0
minutes, at the cost of unflagged cases waiting 25 minutes longer. A reliability score ranked likely misreads across
clean and simulated-degradation reads (error AUROC 0.71 [0.59, 0.87]), and the simulated margin read kept 100% of tumor-free composite
margins clear while finding 45% of involved margins on annotated whole-slide windows. An AI spot check (Claude, not a
pathologist) of top-scoring regions supported invasion, pleomorphism and tubule features (18–19 of 20) but not the normal-lobule or mitosis
features (0 of 20).

**Significance.** Sepia shows how foundation-model reads can be made actionable in a worklist with calibrated,
set-valued confidence, explicit safety rules and grounded explanations, while stating what is simulated and what is
not validated: frozen sections, margins and arrivals are simulated, the dataset is small, and no clinical validation
is claimed.

---

## Condensed version (296 words, for a 300-word limit)

The draft above is 411 words (Problem through Significance), over a typical 300-word abstract limit; no official
limit was found in the inspected organizer material, so check the submission form. Same verified numbers, condensed:

**Problem.** Pathologists read cases in arrival order, so invasive carcinoma can wait behind benign biopsies, and AI
triage tools rarely say why they flagged a case or how far to trust the flag.

**Approach.** Sepia embeds each BACH breast microscopy image (Aresta et al., 2019) as tiles with pathology
foundation models on NCSA Delta (UNI2-h for the prediction; CONCH and MUSK for concepts and agreement). A calibrated
linear probe on 512-pixel regions suggests one of four categories; split-conformal APS sets list the categories that cannot be excluded,
and any case with invasive carcinoma in its set goes to the top of the worklist, behind only simulated frozen
sections. Nine named histologic features, scored from text prompts and caption-derived concept vectors, explain each
read with linked regions and similar reference cases. Rules draft follow-up orders for pathologist approval; an LLM
summary is checked sentence by sentence against structured findings.

**Results (held-out THUNDER split, 132 images, 95% cluster-bootstrap CIs).** Invasive-vs-rest AUROC 0.999
[0.996, 1.000]; invasive sensitivity 100%, specificity 97% [94, 100]; macro-F1 0.86 [0.68, 0.93]. ECE after
temperature scaling 0.055. APS sets covered 91.7% [80.8, 98.1] of images against a 90% target (method chosen after
viewing test results; pre-registered LAC: 72.0%). All 31 invasive images reached the top two tiers. In a simulated
reading day, mean time to first look for invasive cases fell from 20.2 to 6.0 minutes while unflagged cases waited
25 minutes longer. A reliability score ranked likely misreads with error AUROC 0.71. An AI spot check (not a
pathologist) supported the invasion, pleomorphism and tubule features but not the mitosis or normal-lobule features.

**Significance.** Foundation-model reads become actionable in a worklist through set-valued confidence, explicit
safety rules and grounded explanations. Frozen sections, margins and arrivals are simulated, the dataset is small,
and no clinical validation is claimed.
