# Sepia submission upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the four approved upgrades for the Ashby submission: exact region contributions for the production
read (A), conformal coverage across random re-splits (B), a deferral curve (C), and a rebuilt Sepia deck (D).

**Architecture:** A, B and C are laptop-only analyses in the existing pipeline/eval/frontend layout: each writes a
tracked `artifacts/eval/*.json` (A also extends the case JSON contract), adds a section to `docs/RESULTS.md` through
`eval/results_md.py`, and adds a figure to the Model evaluation page (`js/views/metrics.js`). D rebuilds the deck from
`outputs/deck/` with numbers read from `artifacts/eval/*.json`, so it re-renders when A–C land.

**Tech Stack:** Python 3.13 (uv), numpy/pandas/scikit-learn, pytest; vanilla ES-module frontend; Playwright;
pptxgenjs deck builder (`outputs/deck/build.js`) rendered to PDF with LibreOffice (`soffice`).

**Spec:** the four items approved in chat on 2026-10-18 (deck rebuild; exact contributions; conformal re-split;
deferral curve) and the pre-registration `docs/VALIDATION_ROUND3.md`, which fixes every analysis choice.

## Global Constraints

- Repo root `/Users/m3-max/Documents/GitHub/Team-44_Ashby_Hackathon`. Large inputs live in the git-ignored
  `data/` there; from a worktree set `PATHQUEUE_DATA_ROOT=/Users/m3-max/Documents/GitHub/Team-44_Ashby_Hackathon/data`
  and `PATHQUEUE_ARTIFACTS_ROOT=<worktree>/artifacts`. Never symlink `data` into a worktree.
- Do not edit `pipeline/05_risk_model.py`, `pipeline/05c_finetune.py`, `docs/ACCURACY_ROUND2.md` or
  `tests/test_finetune_ingest.py` (another session's in-flight work). Do not overwrite existing files in `data/`;
  new files there are allowed.
- No GPU, Delta, Slurm, Lumen or Cursor calls. Case regeneration must use cached LLM output only.
- Read the production model ID from `artifacts/eval/model_choice.json` / the `model` column of `data/predictions.csv`.
  Never hard-code `uni2h512_lr`: a fine-tuned model may replace it later today.
- Analysis choices are fixed by `docs/VALIDATION_ROUND3.md`. Do not change them after seeing results. Report every
  statistic it lists.
- Intervals: 95% patient-grouped bootstrap (`core/stats.bootstrap_ci`, groups = `group` in `data/cases.csv`).
- JSON outputs follow the existing eval files: `name`, `created_utc`, `config_hash`, `git_sha`, `synthetic`; no NaN
  (use null).
- Copy is plain and professional: descriptive titles, no slogans or taglines, no team number in app chrome, no
  decorative ↗ on in-app links. Say "simulated" wherever data is simulated.
- Never fabricate numbers. Every number in the UI, RESULTS.md or the deck comes from an eval JSON.
- `make test` (`uv run pytest -q`) and `PQ_E2E=1 uv run pytest -q tests/e2e` must pass before a workstream is done.
- Commit on the workstream branch with messages ending in `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Never push.

## Review Focus

1. **Variant reads use a different model.** Degraded and composite reads use the fallback model on 256-px tiles.
   Their contributions must come from that model and be summed into the 512-px contract regions. A read whose model
   has no per-tile embeddings gets `contribution: null`, never zeros, and the UI hides the layer.
   Test: `test_variant_contributions_aggregate_to_contract_grid` and `test_missing_embeddings_give_null` (Task A1).
2. **QC-failed regions.** Regions excluded from pooling have `contribution: null` and stay hatched in the heatmap.
   Test: `test_qc_failed_tiles_are_null` (Task A1); static render check in Task A4.
3. **Large patient groups in re-splits.** Calibration may overshoot 50 images, a group must never straddle the two
   parts, and Mondrian with few calibration images per class must not crash.
   Test: `test_groups_never_straddle` and `test_mondrian_sparse_class_is_reported` (Task B1).
4. **Saturated probabilities.** Many reads have P(Invasive) ≈ 1. Logit-space numbers must come from logits, never
   from `log(p)` of rounded probabilities. Test: `test_logit_delta_finite_when_saturated` (Task A3).
5. **Case regeneration drift.** Re-running `08_build_case_json` must change only the new contribution fields. Test:
   the diff check in Task A2 Step 5 must report zero changes outside those fields.

---

### Task A1: Exact region contributions step

**Files:**
- Create: `pipeline/05d_tile_contrib.py`
- Modify: `Makefile` (`pipeline-laptop`: run `05d_tile_contrib` after `06_reliability`, before `07_retrieval`)
- Test: `tests/test_tile_contrib.py`

**Interfaces:**
- Consumes: `pipeline.riskmodels.LinearModel.from_npz`, `LinearModel.standardize`, `pool_tiles`, `load_emb`; the
  feature-set mapping that `05_risk_model` uses for each model ID (reuse it, do not duplicate the table).
- Produces:
  - `read_contributions(model: LinearModel, rows: np.ndarray, tile_of_row: np.ndarray, temperature: float, target: int) -> tuple[pd.Series, float]`:
    per-tile contributions (index = tile key) and the constant. For pooled `z = rows.mean(0)`:
    `constant + sum(contributions) == (logit_target(z) - mean_k logit_k(z)) / temperature`.
  - `to_contract_grid(contrib: pd.Series, native_tiles: pd.DataFrame, contract_tiles: pd.DataFrame) -> pd.Series`:
    sums native-tile terms into contract-grid regions by spatial containment.
  - `data/tile_contrib.csv` with columns `read_id, model, tile_idx, contribution, constant, target, temperature, n_rows`
    (`tile_idx` on the contract grid; QC-failed regions are absent).

- [ ] **Step 1: Write failing tests** in `tests/test_tile_contrib.py`:
  `test_sum_matches_centered_logit` (random `LinearModel`, 4 classes, 12 tiles × 2 views: `abs(constant + sum - centered_logit) < 1e-9`);
  `test_variant_contributions_aggregate_to_contract_grid` (48 native 256-px tiles → 12 contract regions; sums preserved);
  `test_qc_failed_tiles_are_null`; `test_missing_embeddings_give_null`.
- [ ] **Step 2:** `uv run pytest tests/test_tile_contrib.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement. Pool exactly as the production read does (`pool_tiles`, same QC and views). For each read
  use the model and temperature recorded for that read in `data/predictions.csv`. The target class is the configured
  aggressive class.
- [ ] **Step 4:** `uv run pytest tests/test_tile_contrib.py -v`. Expected: PASS.
- [ ] **Step 5: Real-data identity check.** Run the step on the real data root. For every read, the reconstructed
  centered calibrated logit must match `log(pcal_target) - mean_k log(pcal_k)` from `data/predictions.csv` within
  1e-6, wherever all `pcal` > 1e-12. Print the max error, the number of reads checked and any skipped reads.
- [ ] **Step 6:** Commit.

### Task A2: Case JSON contract and regeneration

**Files:**
- Modify: `core/contract.py`, `pipeline/08_build_case_json.py` (tiles block reads `data/tile_contrib.csv`),
  `backend/dev_cases.py` (synthetic cases carry the fields), `scripts/gen_types_ts.py` output `frontend/src/types.ts`,
  `fixtures/` (via `make types fixtures`)
- Test: `tests/test_contract.py`, `tests/test_build_case_json.py`

**Interfaces:**
- Produces (case JSON): `tiles[i].contribution: number | null` (rounded to 4 dp) and
  `prediction.attribution: {method: "exact_linear", target: string, constant: number, unit: "centered calibrated logit"} | null`.

- [ ] **Step 1:** Failing tests: contract accepts both fields and `null`; `08` writes them from a fixture `tile_contrib.csv`; absent file → `null` (no crash).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement; `make types fixtures`; `make types-check`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Regenerate and diff.** Run `08_build_case_json` with cached summaries only, so it makes no LLM calls
  and keeps the committed summaries. Then diff every `artifacts/cases/*.json` against `HEAD` with keys
  `tiles[*].contribution` and `prediction.attribution` removed. Expected: zero differences. If anything else changed,
  stop and report; do not commit drifted case files.
- [ ] **Step 6:** Commit code and regenerated cases separately.

### Task A3: Deletion test arm

**Files:**
- Modify: `eval/deletion_test.py`, `eval/results_md.py`, `team44-triage-frontend/app/js/views/metrics.js` (`figDeletion`)
- Test: `tests/test_deletion.py`

**Interfaces:**
- Produces: in `artifacts/eval/deletion.json`, per model and population, `mean_drop_contribution`,
  `contribution_minus_random`, and logit-space twins of every drop (`mean_drop_logit_random|risk|concept|contribution`,
  `contribution_minus_random_logit`), each a bootstrap-CI block as today.

- [ ] **Step 1:** Failing tests: `test_contribution_arm_present`; `test_logit_delta_finite_when_saturated` (P = 1 − 1e-12 read gives finite logit delta).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement the arm with the same k and populations as the existing arms; compute logit deltas from re-pooled logits.
- [ ] **Step 4:** Run tests; run `python -m eval.deletion_test` on real data; then `eval.results_md`.
- [ ] **Step 5:** The figure shows four arms (random, tile probe, concept, exact contribution) in probability and
  logit units, with a computed takeaway sentence and no question-style title.
- [ ] **Step 6:** Commit code and regenerated `artifacts/eval/deletion.json`, `docs/RESULTS.md`.

### Task A4: Contribution layer in the case view

**Files:**
- Modify: `team44-triage-frontend/app/js/views/heatmap.js`, `team44-triage-frontend/app/js/views/why.js`,
  `team44-triage-frontend/app/case.css` (only if a new style is needed)
- Test: `tests/test_case_view_static.py`

- [ ] **Step 1:** Failing static tests:
  - with contributions present, the layer select offers "Region contribution (exact)" and selects it by default;
  - pink means toward invasive and blue away, scaled to the case's max |contribution|;
  - QC-failed regions stay hatched;
  - without contributions, the layer is absent.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement the layer.
  - Legend copy: "Each region's push toward (pink) or away from (blue) invasive carcinoma. The pushes plus a constant add up exactly to the model's invasive score."
  - Tooltip: `Region 7 · +0.42 toward invasive`.
  - In the Why card, add "Regions with the largest push": the top three positive regions, each with its contribution and its two highest concept z-scores. Each row links to its region with the existing region-focus mechanism.
- [ ] **Step 4:** `uv run pytest -q tests/test_case_view_static.py` and `PQ_E2E=1 uv run pytest -q tests/e2e`. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task B1: Conformal re-split analysis

**Files:**
- Create: `eval/conformal_resplit.py`; Test: `tests/test_conformal_resplit.py`
- Modify: `Makefile` (`eval`: after `metrics`), `core/schemas.py` (`EVAL_FILES`, if expected files are listed there)

**Interfaces:**
- Consumes: `core.conformal` (`fit_lac`, `lac_sets`, `fit_aps`, `aps_sets`, `mondrian_quantiles`, `mondrian_sets`,
  `nonempty_sets`, `coverage`, `mean_set_size`) with the settings used for `artifacts/eval/conformal.json`.
- Produces:
  - `grouped_resplits(groups: np.ndarray, n_cal: int, n_splits: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]`
  - `split_stats(probs: np.ndarray, labels: np.ndarray, cal: np.ndarray, test: np.ndarray, alpha: float, target: int, seed: int) -> dict[str, dict[str, float]]`
  - `artifacts/eval/conformal_resplit.json`: `{n_images, n_groups, n_splits, n_cal_target, alpha, seed, model, methods: {<method>: {coverage: {mean, p2_5, p50, p97_5}, share_at_or_above_target, mean_set_size, empty_share, invasive_in_set_given_invasive, invasive_in_set_given_not_invasive}}, original_split: {<method>: {...same}}}`. Method keys match `conformal.json`.

- [ ] **Step 1:** Failing tests:
  - `test_groups_never_straddle`;
  - `test_calibration_reaches_target` (≥ 50, and stops at the first group that reaches it);
  - `test_deterministic_with_seed`;
  - `test_coverage_near_target_on_exchangeable_data` (synthetic Dirichlet probs, 400 images, 200 splits: mean coverage within 0.03 of 0.9 for APS and LAC);
  - `test_mondrian_sparse_class_is_reported` (no crash; null where undefined).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement per `docs/VALIDATION_ROUND3.md` §1 (1,000 splits, seed 20261018, α 0.1).
- [ ] **Step 4:** Tests pass; run on real data; check the `original_split` coverage equals `conformal.json` for each method (same model, same split).
- [ ] **Step 5:** Commit code and `artifacts/eval/conformal_resplit.json`.

### Task B2: Re-split results on the evaluation page and in RESULTS.md

**Files:** Modify `team44-triage-frontend/app/js/views/metrics.js` (new `figResplit` in the Confidence section, after
`figCoverage`), `eval/results_md.py` (`section_conformal_resplit`), `tests/test_metrics_view.py`.

- [ ] **Step 1:** Failing test: the metrics view renders the figure from a fixture JSON, shows the 90% target line, and degrades to the standard "Not available in this run" card when the file is missing.
- [ ] **Step 2:** Implement.
  - The figure shows the coverage range across re-splits per method: the 2.5th–97.5th percentile bar with the median, and the original split marked.
  - Title: "Conformal coverage across 1,000 random re-splits" (the count comes from the JSON).
  - The takeaway sentence is computed from the JSON.
  - The APS summary tile keeps its "chosen after test results" disclosure and adds the re-split mean.
- [ ] **Step 3:** `uv run pytest -q`, regenerate `docs/RESULTS.md`, commit.

### Task C1: Deferral analysis

**Files:**
- Create: `eval/deferral.py`; Test: `tests/test_deferral.py`
- Modify: `Makefile` (`eval`: after `conformal_resplit`), `core/schemas.py` if needed

**Interfaces:**
- Produces:
  - `risk_coverage(correct: np.ndarray, defer_first: np.ndarray, fractions: Sequence[float]) -> list[dict[str, float]]` (`defer_first` is the sort key: smallest deferred first; ties by read ID upstream)
  - `artifacts/eval/deferral.json`: `{fractions, populations: {clean_test, all_test_reads: {n_reads, n_groups, rankings: {reliability, top_prob, set_size, random: [{fraction, kept, accuracy, errors_deferred_share, invasive_missed_kept, insitu_overcalled_kept}]}}}, headline: {population: "all_test_reads", fraction: 0.2, accuracy_kept: ci, accuracy_all: ci, accuracy_random_20: number}}`

- [ ] **Step 1:** Failing tests:
  - `test_perfect_ranking_removes_errors_first`;
  - `test_random_ranking_is_flat_on_average`;
  - `test_fraction_zero_matches_overall_accuracy`;
  - `test_ties_are_deterministic`;
  - `test_composites_excluded`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement per `docs/VALIDATION_ROUND3.md` §2.
  - True labels: the source case's label.
  - Predicted class: the `top` of that read in `data/predictions.csv`.
  - Reliability: `reliability` in `data/reliability.csv`, joined on read ID. Fail loudly on unmatched reads.
- [ ] **Step 4:** Tests pass; run on real data.
- [ ] **Step 5:** Commit code and `artifacts/eval/deferral.json`.

### Task C2: Deferral curve on the evaluation page and in RESULTS.md

**Files:** Modify `team44-triage-frontend/app/js/views/metrics.js` (new `figDeferral` in the Frozen-section
reliability section, after `figReliabilityScore`), `eval/results_md.py` (`section_deferral`), `tests/test_metrics_view.py`.

- [ ] **Step 1:** Failing test: renders from fixture JSON; missing file degrades to the standard card.
- [ ] **Step 2:** Implement.
  - The figure plots accuracy on kept reads (y) against share deferred (x, 0–50%) for the reliability ranking in both populations, with random deferral as a dashed reference.
  - Title: "Accuracy on kept reads as the least reliable reads are deferred".
  - The takeaway is computed from `headline`.
- [ ] **Step 3:** `uv run pytest -q`, regenerate `docs/RESULTS.md`, commit.

### Task D1: Deck data and screenshots

**Files:** Create `outputs/deck/extract_data.py` (reads `artifacts/eval/*.json`, writes `outputs/deck/data.json`);
fresh screenshots in `outputs/deck/assets/`.

- [ ] **Step 1:** Write the extractor.
  - It runs again whenever eval files change.
  - It fails with a clear message when a required file or key is missing.
  - `--allow-pending` marks the A–C slides as pending for drafting only. The final build refuses pending slides.
- [ ] **Step 2: Screenshots.**
  - Start a throwaway server on port 8011 from a copy of `artifacts/` in the scratchpad (never touch the :8000 server or `artifacts/pathqueue.sqlite`).
  - Drive it with Playwright: start the replay, open BACH_0167 and a degraded frozen section, and inject a composite.
  - Capture the worklist, the case view (image, Why, Prediction and confidence, orders), the OR board, the oncology inbox and the evaluation charts.
  - Light theme, 1440×900, device scale 2.

### Task D2: Deck build

**Files:** Modify `outputs/deck/build.js` → output `outputs/deck/Sepia_Team44.pptx` and `.pdf`; update
`docs/DECK_OUTLINE.md` to the new outline.

Structure (≤ 20 main slides, then backup slides titled "Backup"). It follows the organizers' required sections:
aims, methods, results, challenges overcome and remaining, references.

1. **Title.** "Sepia: confidence-aware triage for breast pathology"; "Team 44 · 2026 Ashby Prize in Computational Science Hackathon"; "Research prototype; not for clinical use"; worklist screenshot.
2. **Aim and clinical problem.** Arrival-order queues; frozen sections need an answer in about 20 minutes; AI flags rarely give a reason or a calibrated confidence.
3. **One case, start to finish.** BACH_0167 arrives last and is read early. Its set is {In situ, Invasive}, so it goes to tier 2. p63/calponin is drafted, approved and signed out, and the result reaches the oncology inbox.
4. **System overview.** Delta, then the laptop analytics, then the app; two lanes; four tiers; the safety rule.
5. **Data and split.** BACH, THUNDER 218/50/132, 24 test patient groups, tissue QC.
6. **Foundation models and model selection.** Encoders embedded on Delta; grouped CV inside train; one-SE rule; the pre-registered 512-px round.
7. **Calibration and conformal sets.** Temperature scaling, APS tier set, safety rule.
8. **Triage accuracy.** AUROC, sensitivity, specificity, macro-F1 vs published; confusion matrix.
9. **The clinically important error.** In situ vs invasive.
10. **Calibrated confidence.** ECE; coverage on the THUNDER split (post-hoc disclosure) and across re-splits (B).
11. **Safety rule in the worklist.** Tier by true class.
12. **When to defer.** Deferral curve (C); reliability AUROC; degradation accuracy.
13. **Region contributions.** Contribution heatmap and deletion comparison (A).
14. **Named features.** Concept direction; spot check with the failures stated.
15. **Frozen-section lane (simulated).** Margin read numbers; the defer path.
16. **Workflow simulation.** Invasive first look and the cost to unflagged cases.
17. **Challenges overcome.** QC bug, LAC empty sets, PathMMU/BACH mismatch, one GPU per team, partial patient IDs, LLM summary validation.
18. **Remaining challenges and next steps.** External validation, pathologist review, whole slides, silent prospective trial.
19. **Limitations.**
20. **References.** Formal citations; every model, dataset and method used.

Copy and output rules:
- Assertion titles backed by a number are fine. No slogans or question titles.
- Footer: "Sepia · Research prototype, not for clinical use", plus the page number.
- BACH credit on every slide that shows BACH imagery.
- Speaker notes on every main slide: the 8-minute talk track.

- [ ] **Step 1:** Build with `--allow-pending`, render to PDF and per-slide JPGs with `soffice`, and inspect every slide image for overflow, overlap and clipping. Fix and repeat.
- [ ] **Step 2:** Report slide count, pending slides and the QA notes.

### Task D3 (master, after A–C merge)

- [ ] Re-run the extractor and build without `--allow-pending`; re-capture the case-view screenshot with the contribution layer.
- [ ] Update `docs/ABSTRACT_DRAFT.md` with the final numbers.
- [ ] Final full-deck visual QA.
