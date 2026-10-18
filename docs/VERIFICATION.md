# PathQueue verification

Verified locally on October 18, 2026, 03:00–07:00 (America/Chicago). Research prototype, not a clinically validated
system. Overnight changes and their measured effect: [OVERNIGHT.md](OVERNIGHT.md). Numbers: [RESULTS.md](RESULTS.md).

## Code and workflows

- Full Python unit/integration suite on merged `main`: **2,533 passed**, 1 skipped (the opt-in browser suite), ~2 min
  (final run 04:18 CDT).
- Browser suite (`PQ_E2E=1`, Playwright, desktop/light and phone/dark, labelled synthetic temporary workspaces):
  **23 passed** (final run 04:20 CDT, after the last change). It covers the replay, tier ordering, drafted-order approval with inline confirmation, sign-out to the
  oncology inbox, frozen-section injection, named-margin and defer calls, the OR board timer stop, keyboard
  next-case navigation, reload persistence, the evaluation and evidence-review pages.
- Contract → TypeScript consistency (`make types-check`) passes after the `value_std` and generator changes.
- **Hand-driven demo on the real artifacts** at `http://127.0.0.1:8000/` (03:05–03:09): arrival replay; a tier-2
  biopsy (BACH_0364) → Order tests (confirm) → Sign out (confirm) → oncology report card; composite frozen section
  COMP_026 → OR-3 tier 1 with running timer → Tumor at superior/medial/lateral (pre-filled) → OR board shows the call,
  turnaround and stopped timer, plus a toast; degraded frozen section BACH_0301_FSIM_resample3 → reliability flag →
  plain-language defer reason → Defer to permanent → routed. After the full replay the planned hero BACH_0167 sits in
  tier 2 with the set {InSitu, Invasive} and drafted p63/calponin + ER/PR/HER2/Ki-67. All decisions were made by the
  operator for testing, not clinical assessments; the workspace was then archived and reset for the morning.
  A second hand pass (04:13–04:17, throwaway workspace) covered the Cells tab, Edit orders (Ki-67 removed → order
  "edited", drafted items kept), Take a closer look first → Dismiss flag (with confirmation) → sign-out unlocked, the
  live OR timer, and keyboard navigation (J/K, /, N opens the next case in priority order).
- Accessibility: axe-core 4.10 on the worklist, a biopsy case, a surgical case, the OR board, the oncology inbox, the
  evaluation and evidence-review pages, at 1440×900 and 390×844, light and dark: **0 serious or critical violations**
  (after fixing a chart role, four sub-4.5:1 text colours, the dark tier-1 badge and phone table focus); 0 moderate
  findings after the notice strip and case navigation became landmarks.
- The three UI agents each screenshot-checked their pages at 1440×900 and 390×844 in light and dark: no page errors,
  no horizontal overflow, no broken images (scripts and screenshots under `.worktrees/ov-*-env/shots/`, git-ignored
  because they contain BACH-derived imagery).

## Data and compute

- Default workspace: **328 contract-valid cases** (132 held-out test images, 132 degraded simulated frozen sections,
  64 composite simulated fields), **0 rejected**, **0 synthetic**; **903 of 903** referenced media assets present.
- Tile QC re-run on the laptop with the corrected mask (`pipeline/01q_requalify_qc.py`): insufficient-tissue images
  51 → 0; display-JPEG recomputation of the old rule reproduces the Delta full-resolution values (r = 0.998). The
  original columns are kept in `data/meta/tiles_qc_v1.csv`. No new GPU work was needed: every tile had been embedded on
  Delta. Requested Delta GPU time is unchanged at 300 of the 420 GPU-minutes the submission guard allows (Slurm time limits
  requested, not measured GPU use).
- Laptop steps 04 → 08 and the whole evaluation suite were re-run (`make analyze` order) after the QC fix and the
  conformal change; `eval/analytics.py` reproduces every AUROC in `metrics.json` exactly from its ROC points.
- Summaries: 328 Lumen calls to `gemma-4-31b-it`, cached under `data/llm_cache/`; 176 summaries are fully LLM-written,
  152 carry a required deterministic margin / quality sentence (`lumen:gemma-4-31b-it+template`). Every sentence passed
  the citation, number and semantic checks; the live demo makes zero LLM calls.
- AI spot check (Claude, not a pathologist): 200 tiles reviewed through the app's API (180 concept tiles, 20 cell
  overlays), recorded in `eval/spotcheck.json` with per-tile notes, plus one cell-overlay check from the Codex UI
  verification (also AI); no human review has been recorded yet.
- The workspace databases from earlier sessions are archived under `artifacts/archive/` (never deleted).

## Results and limits

Held-out test accuracy is **84.1% on 132 original images** (macro-F1 0.83); invasive sensitivity 100%, specificity
97%. The tier-driving APS set covers **90.2%** against a 90% target, but that method was chosen after viewing test
results, so the figure is optimistic; the pre-registered LAC covers 72.7%. Arrivals, operating rooms, frozen
artifacts, image-border margins and queue-benefit estimates are simulations. Stain rules are illustrative and need
clinical review. Hospital order entry, clinical deployment, expert validation and a coverage guarantee are not claimed.
