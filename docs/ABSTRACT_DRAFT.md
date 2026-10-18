# Abstract — DRAFT for team review (numbers from docs/RESULTS.md, 2026-10-18)

**PathQueue: an explainable, confidence-aware breast pathology worklist built on pathology foundation models**

*Team 44 · 2026 Ashby Prize in Computational Science Hackathon · research prototype, not for clinical use*

**Problem.** Pathologists read cases in arrival order, so an invasive carcinoma can wait behind benign biopsies, and
AI triage tools rarely say why they flagged a case or how far to trust the flag.

**Approach.** PathQueue reads each BACH breast microscopy field (Aresta et al., 2019) as a grid of tiles embedded by
pathology foundation models on NCSA Delta (UNI2-h for the production read; CONCH and MUSK for text-aligned concepts
and agreement). A calibrated linear probe suggests one of four categories; split-conformal prediction (APS) returns
the set of categories that cannot be excluded, and a safety rule sends any case with invasive carcinoma in that set
to the top of the worklist, behind only simulated frozen sections. Each read is explained by nine named histologic
features (e.g. tumor infiltrating stroma, nuclear pleomorphism, tubule formation) scored from text prompts and
caption-derived concept vectors, linked to image regions and similar reference cases. A rule engine drafts
follow-up orders (p63/calponin, ER/PR/HER2/Ki-67, deeper levels) that the pathologist approves; an LLM writes a
summary from structured findings only, checked sentence by sentence against those findings.

**Results (held-out THUNDER test split, 132 images, 95% cluster-bootstrap CIs).** Invasive-vs-rest AUROC 0.998
[0.994, 1.000]; invasive sensitivity 100% and specificity 97% [94, 100]; four-class macro-F1 0.83 [0.61, 0.93],
in line with the published UNI2-h linear probe on the same split (84.7; different protocol). After temperature
scaling, ECE 0.055. APS sets covered the true class in 90.2% [75.7, 98.1] of images against a 90% target (method
chosen after viewing test results; the pre-registered LAC reached 72.7%). All 31 invasive images landed in the top
two tiers. In a simulated reading day the tiers cut mean time-to-first-look for invasive cases from 20.2 to 6.0
minutes, at the cost of unflagged cases waiting 25 minutes longer. A reliability score flags degraded
(simulated frozen-section) reads (error AUROC 0.71), and the simulated margin read kept 100% of tumor-free composite
margins clear while finding 45% of involved margins on annotated whole-slide windows. An AI spot check of top-scoring
regions supported invasion, pleomorphism and tubule features (18–19 of 20) but not the normal-lobule or mitosis
features (0 of 20).

**Significance.** PathQueue shows how foundation-model reads can be made actionable in a worklist with calibrated,
set-valued confidence, explicit safety rules and grounded explanations, while stating what is simulated and what is
not validated: frozen sections, margins and arrivals are simulated, the dataset is small, and no clinical validation
is claimed.
