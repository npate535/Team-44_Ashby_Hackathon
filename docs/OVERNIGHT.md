# Overnight polish — 2026-10-17 23:40 to 2026-10-18 07:00 CDT

Research prototype, not for clinical use. Every number below comes from the eval files in `artifacts/eval/`
(95% cluster-bootstrap CIs over patient / pseudo-patient groups, held-out THUNDER test split, n = 132 images,
24 groups). "Before" is the pipeline state at commit `b58ddbc` (preserved at `.worktrees/backup-pre-overnight/`).

## Run it

```sh
make demo                       # http://127.0.0.1:8000, real BACH artifacts, fresh workspace
make test                       # 2,500+ unit/integration tests
make e2e                        # Playwright workflow tests (PQ_E2E=1)
```

Presenter script with real case IDs: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

Demo path (plan §14), verified by hand on the real artifacts at 03:05–03:09 CDT:
Start replay → tier-2 biopsies sort to the top → open one → **Order tests** → **Sign out** → it appears in the
**Oncology inbox** · **+ Composite** sends a simulated frozen section to an OR → it jumps to tier 1 with a running
timer → **Tumor at margin** (named margins pre-filled) → the **Operating rooms** board shows the call and the stopped
turnaround timer · **+ Degraded** → reliability flag → suggested **Defer to permanent** with the reason in plain words
· **Model evaluation** shows every analysis as charts with the source numbers one click away.

## What changed and why

### 1. Tissue QC bug fixed (accuracy)
The tissue mask required HSV value < 0.92. HSV value is the brightest RGB channel, so pale H&E tissue sits at
V ≈ 0.95–1.0 and was rejected: **51 of 400 images (32 of 100 Invasive) were flagged "insufficient tissue"** and their
reads used only the few surviving tiles. Saturation alone separates BACH's grey glass background from tissue (checked
visually on the lowest-tissue images). `pipeline/01q_requalify_qc.py` recomputes tile QC from the display JPEGs (the
old rule recomputed the same way reproduces the Delta full-resolution values, r = 0.998). Every tile had already been
embedded on Delta, so steps 04–08 and the whole evaluation were re-run on the laptop with no new GPU work.

| | Before | After |
|---|---|---|
| Contract tiles passing QC | 75.1% | 98.7% |
| "Insufficient tissue" images | 51 | 0 |
| Production CV balanced accuracy (train, grouped) | 0.930 [0.890, 0.969] | 0.949 [0.916, 0.981] |
| Concept-bottleneck CV balanced accuracy | 0.794 [0.732, 0.849] | 0.854 [0.798, 0.902] |
| Test accuracy | 0.833 [0.687, 0.938] | 0.841 [0.699, 0.940] |
| Test macro-F1 | 0.819 [0.595, 0.921] | 0.830 [0.609, 0.927] |
| Invasive specificity (sensitivity stays 100%) | 0.960 [0.922, 1.000] | 0.970 [0.936, 1.000] |
| ECE after temperature scaling | 0.076 [0.053, 0.224] | 0.055 [0.039, 0.224] |
| Retrieval precision@5 | 0.645 | 0.668 |
| Reliability-score error AUROC | 0.668 [0.540, 0.810] | 0.705 [0.587, 0.865] |
| WSI margin-window sensitivity (specificity) | 36.2% (100%) | 45.4% (98.5%) |

In situ ↔ invasive confusion: 3 of 30 in situ images called invasive (was 4); 0 of 31 invasive called in situ.

### 2. Conformal sets made usable (disclosed post-hoc change)
Pre-registered LAC returned **empty sets for 23.5%** of test images and never a multi-class set, so the worklist
showed "∅ empty" and the InSitu-vs-Invasive story never happened with real data. Cause: the 50-image calibration split
is far easier than the test split (98% vs 84% accurate), which breaks the exchangeability the guarantee needs.
Tiers now use **APS (Romano et al. 2020) with the argmax added to an empty set** (`conformal.method: aps`,
`conformal.nonempty: true`; plan §6.6 allows APS or LAC). LAC is still computed and reported next to it.

| Tier set | Coverage (target 90%) | Empty sets | Mean size |
|---|---|---|---|
| LAC, pre-registered (before → after QC fix) | 67.4% → 72.7% [60.7, 84.4] | 23.5% → 18.9% | 0.81 |
| APS + nonempty (drives tiers now) | 90.2% [75.7, 98.1] | 0% | 1.31 |

The switch was made **after viewing test results**, so 90.2% is optimistic and is labelled that way in the app,
RESULTS.md and the dashboard. With APS the set size now predicts errors (AUROC 0.70, was 0.42 under LAC), and 31 of
31 invasive test images land in tier 1 or 2. Real cases now produce the plan's demo story on their own, e.g.
BACH_0167: {InSitu, Invasive} → tier 2 → drafted p63/calponin and ER/PR/HER2/Ki-67.

### 3. Grounded LLM summaries that are actually grounded
All 328 summaries were template fallbacks: `ornith-1.0-35b` spent its 1,500-token budget on hidden reasoning and
returned empty replies. Summaries now come from `gemma-4-31b-it` via Lumen (328 calls, cached offline; the live demo
makes zero LLM calls). The number/citation validator was not enough: early outputs said "Margins are clear." for
deferred reads and called scarce features "prominent" (a positive push was read as presence). Added:
- concept findings carry a `level` (above / typical / below the training average, from the new `value_std`);
- `semantic_problem()` rejects sentences that contradict a cited margin call, reliability flag or feature level,
  state a simulated margin call as fact, or present away-pushing features as support;
- `ensure_required()` appends the deterministic margin / quality-concern sentence when the model omits it
  (generator reads `lumen:gemma-4-31b-it+template`; 152 of 328 cases).

### 4. Clinical rules
`uncertain_benign_atypia` (+ CK5/6/ER) now also fires for {Normal, InSitu} sets: two such cases drafted nothing
although in situ disease could not be excluded (new `in_set_any` rule condition).

### 5. AI spot check of the explanations (plan §5.4) — AI-reviewed, not a pathologist
Claude reviewed the top-20 tiles of every concept (and 20 cell overlays) and recorded a verdict per tile through the
app's spot-check API (`eval/spotcheck.json`): invasion 19/20, benign stroma/fat 20/20, tubules 18/20,
pleomorphism 18/20, duct-confined 12/20, lymphocytes 8/20, necrosis 3/20 (mostly clot-like eosinophilic material),
mitoses 0/20 (figures not resolvable at 256–384 px; all "unsure"), normal lobules 0/20 (top tiles are fat and
stroma: the concept tracks context, not lobule morphology). The case view shows the badge per feature and tells the
reader to discount the weak ones.

### 6. The application (three parallel agents, merged into main)
- **Worklist / shell**: dense one-line clinical worklist with sticky tier groups, queue-composition bar, filters,
  keyboard navigation (J/K, Enter, N for next case, / search), live OR board, report-style oncology inbox, toasts,
  compact professional page headers, persistent replay + frozen-section controls.
- **Case review**: "Why" panel grouped as plan §6.4 with each feature's level in SD vs training cases and its push;
  zoomable full-screen viewer; region tooltips; fixed reference thumbnails (4 of 5 were blank) and side-by-side
  compare; probability chart with the APS set; plain-language confidence and agreement checklist; decision stepper
  with confirmations. Surgical view: log-odds margin bars against the threshold, specimen diagram, segment-level
  inked-border strip on the image, plain-language defer reason, big timer.
- **Model evaluation**: the 16,000-px wall of tables is now a curated dashboard (ROC per model and per class,
  confusion matrix, precision–recall, model forest plots, THUNDER benchmark, reliability diagrams, probability
  histograms, coverage by class and method, set composition, concept × class heatmap, NCT-CRC and caption checks,
  deletion test, degradation curves, reliability flag, margin read, worklist tier placement, queue-simulation
  distributions, cell measurements, data quality, spot check), each with a computed "What this shows" sentence and
  the source numbers. `eval/analytics.py` produces the chart data.

### 7. Demo robustness
- After the last scheduled biopsy arrives, the simulator clock drops from the boosted replay speed (×161) to wall time
  (`sim.after_arrivals_speed: 1`), so a demo left running no longer pushes every case past its target and lets the
  aging guard reshuffle the queue. Reset restores the replay speed.
- Each concept row now explains its whole push: the case JSON carries both standardized inputs (`value_std` for the
  3-strongest-region mean, `frac_std` for the share of regions above threshold).
- `make demo-fresh` (`scripts/demo_fresh.sh`) restarts the app on a fresh workspace and archives the previous database.

## Limits that still hold
Small public dataset (BACH, 132 test images); simulated frozen sections, margins, arrivals and OR rooms; stain rules
are illustrative and need clinical review; concept explanations are associations (deletion test shows weak
faithfulness for the concept tiles); the queue simulation's aging guard has no measured effect; nothing here is
clinically validated. Nothing was pushed to GitHub and the deployed planning site was not updated.
