# Judge feedback response: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap the two judging conversations (recordings 153 and 155) exposed. Each claim the team made aloud gets an exact, sourced number or a correction in the deck, abstract, app and docs. Add the missing evidence: a production-model external test on TCGA-BRCA, a matched human-reader baseline from the literature, and a contrastive explanation for the look-alike case.

**Architecture:** Most judge items are already measured in `artifacts/eval/*.json`; the team mis-stated them aloud.
So the core is a generated **fact sheet** that every downstream artifact (deck, abstract, Q&A) quotes. The new evidence comes from three independent workstreams in separate git worktrees:

- **T3:** external TCGA test of the production model (laptop MPS, GDC tile API).
- **T4:** contrastive look-alike detail (pipeline 07 + case view).
- **T2:** literature baseline.

The deck and abstract are rebuilt last, from the generated files.

**Tech Stack:** Python 3.13 (uv), numpy/pandas/torch-MPS/timm, pytest; vanilla ES-module frontend; pptxgenjs deck (`outputs/deck/`, git-ignored, rendered with LibreOffice).

**Spec:** the judge-feedback register pasted on 2026-10-18 (sections 2–6: P0.1–P0.12, P1.1–P1.9, P2.1–P2.6, the 16 Q&A questions).

## What the repo already answers (the team mis-stated it aloud)

| Judge item | Measured truth already in the repo |
|---|---|
| 400 "tiles" / WSIs | 400 BACH microscopy images (2048×1536, 100/class), not WSIs; each read as a 4×3 grid of 512-px regions × 8 dihedral TTA views |
| 218/238/213 split | THUNDER split 218 train / 50 calibration / 132 test = 400 |
| "Cross-validation" | Grouped CV inside the 218 training images, used only to choose the model; one held-out test split |
| "98–99% AUROC" | Invasive-vs-rest AUROC 0.999 [0.996, 1.000]; macro one-vs-rest 0.965; macro-F1 0.86; 31/31 invasive test images flagged; 3/30 in situ called invasive |
| 15/42/40 case | BACH_0087 (true in situ): Normal 14.9, Benign 1.7, In situ 42.7, Invasive 40.7; set {Normal, In situ, Invasive}; tier 2; p63/calponin drafted |
| 20 → 6 min | `queue_sim.json`: simulated day, Poisson 6/h, 9 min/read, one reader. Invasive first look 20.2 → 6.0 min; unflagged wait +21.4 min |
| Attention model "worse" | ABMIL on the same features: test macro-F1 83.7 vs 86.0 (CV 0.946 vs 0.955) |
| "Gemini" summaries | Actually `gemma-4-31b-it` via NCSA Lumen (summaries) and Claude via cursor-agent ("In plain language" tab) |
| "Window of five" | k = 5 nearest training cases, by cosine similarity on CONCH embeddings |

## Real gaps this plan closes

1. **No single provenance/fact source.** Quilt counts, reference distributions, leakage checks and model ownership are scattered or undocumented.
2. **External test used the wrong model on fewer patients than pre-registered.** `external_tcga.json` scored the LoRA fine-tune (`uni2h_ft`), not production, on 8 patients; the protocol said 25.
3. **No human-reader baseline or clinical turnaround benchmarks with citations.**
4. **The look-alike only lists concept names.** It doesn't show direction or magnitude, whether the concept passed its spot check, or where the concept sits in each image.
5. **"Nottingham = nine features" wording.** It has to become: 3 Nottingham components mirrored + 6 custom features, with no grade computed.
6. **The deck lacks several pieces.** Missing: a provenance figure, the human baseline, production-model external validation, and an implemented / simulated / proposed matrix. It also needs an "End of presentation" divider, and the inbox should move to the appendix.
7. **No Q&A sheet** for the 16 questions the judges asked or implied.
8. **Outcome association (P2.1/2.2) was never attempted.** It needs per-patient concept scores, which need CONCH locally.

## Global Constraints

- Work in worktrees under `.worktrees/` on branches `jf/<task>`. Another agent has uncommitted edits to `team44-triage-frontend/app/{app.js,index.html,metrics.css,js/views/tcga.js}` in the main tree: never touch those four files from a worktree. If T3 needs UI changes there, wait until the other agent's work is committed.
- Data root: `PATHQUEUE_DATA_ROOT=/Users/m3-max/Documents/GitHub/Team-44_Ashby_Hackathon/data` and `PATHQUEUE_ARTIFACTS_ROOT=<worktree>/artifacts`. Never symlink `data/` into a worktree. Never overwrite existing files in `data/`; new files are fine.
- Do not change the production model, temperature, conformal rule, tier rule or any pre-registered analysis. New analyses are pre-registered in writing, committed **before** they run, and reported whatever the result.
- Never fabricate numbers or citations. Every number in deck, abstract, app or docs comes from an eval JSON or the fact sheet. Every citation is verified (DOI/PMID resolved) before use.
- Copy is plain and professional: descriptive or number-backed assertion titles. No slogans, taglines or question titles, and no team number in the app chrome. Write "simulated" wherever data is simulated and "AI-reviewed" wherever Claude stands in for a human.
- No Delta/Slurm or event GPU (the event GPU window ended Sun 18:00 CDT). Laptop MPS is allowed.
- No LLM calls when regenerating case JSONs: use cached summaries only (`08_build_case_json` cached path).
- `uv run pytest -q` passes before any merge. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never push and never change repo visibility. Pushing `main` deploys to Vercel and the repo belongs to a teammate, so both go back to the user.

## Review Focus

1. **Tile-API crops differ from openslide crops.** The new TCGA crops come from GDC DeepZoom tiles, not `openslide.read_region`. Expected: re-extracting 6 existing crops from the same coordinates gives a mean absolute pixel difference < 3/255. Test: `test_tile_api_crop_matches_manifest_crop` (T3), run against the network once and recorded in `run_info.json`.
2. **TTA mismatch.** BACH cal/test reads are means over 8 dihedral views per region. A TCGA read embedded with 1 view is a protocol deviation. Test: `test_tcga_views_match_bach_recipe` asserts that the view count and transforms equal `settings.risk_model.views.g512` and the 02_embed dihedral set (T3).
3. **Conformal threshold drift.** The TCGA run must use exactly the APS q-hat the app uses for production. Test: `test_production_qhat_matches_app` recomputes q-hat from `predictions.csv` calibration rows and compares it with the value behind `set_tier` (T3).
4. **Contrast concept with a failed spot check shown as evidence.** A differs-most-in concept whose spot check failed (mitoses 0/20, normal lobules 0/20, necrosis 3/20) must render as "not reliable", never as a reason. Test: `test_contrast_detail_marks_failed_spotcheck` (T4) and the static render check.
5. **Contrast detail missing for variants and composites.** Degraded and composite reads have a contrast case but may lack library features. Expected: `null` detail, the UI falls back to chips, and nothing crashes. Test: `test_contrast_detail_null_when_features_missing` (T4).

---

### Task T1: Fact sheet (P0.1–P0.3, P0.7, P0.11, P0.12, P1.7)

**Files:**
- Create: `eval/fact_sheet.py`, `tests/test_fact_sheet.py`
- Output: `artifacts/eval/fact_sheet.json`, `docs/FACT_SHEET.md` (generated)

**Interfaces:**
- Produces: `artifacts/eval/fact_sheet.json` with these keys: `bach` (source images per class, size, mpp, regions per image, views, total regions); `splits` (per split: images per class, patient groups, known vs pseudo patient IDs); `disjointness` (group overlap between splits, must be 0); `model_selection` (CV K×R, seed, what it chose); `test_metrics` (AUROC conventions, 4×4 confusion matrix, per-class sensitivity/specificity, listed errors); `calibration_conformal`; `corpora` (each corpus: role, count, used-for, never-used-for); `reference_distributions`; `components` (pretrained vs trained by team vs LLM, exact IDs); `latency`; `timing_claims` (queue-sim vs model latency vs clinical TAT, kept separate); `case_examples` (BACH_0087, BACH_0167, BACH_0084).
- T5 and T6 quote only this file and the eval JSONs.

- [ ] **Step 1:** Write failing tests:
  - `test_splits_sum_to_400_and_are_disjoint`: 218 + 50 + 132, zero group overlap.
  - `test_confusion_matrix_matches_metrics_json`.
  - `test_auroc_conventions_labelled`: invasive_vs_rest 0.999, macro_ovr 0.965, each with its definition string.
  - `test_case_example_probs_sum_to_one`: BACH_0087 probs sum to 1 ± 1e-3, Benign 0.0166 present.
- [ ] **Step 2:** Run `uv run pytest tests/test_fact_sheet.py -v` → FAIL.
- [ ] **Step 3:** Implement `build_fact_sheet(data_root, art_root) -> dict` and `render_md(sheet) -> str`.
  - Sources: `data/cases.csv`, `data/tiles.csv`, `data/predictions.csv`, the eval JSONs, `config/settings.yaml`, `config/concepts.yaml`, and the mined-caption manifests (exact Quilt/PathGen counts per concept and polarity).
  - Leakage: if CONCH align embeddings exist for both the mined Quilt set and BACH test, report the max cosine similarity between any mined image and any BACH test image as a near-duplicate check; otherwise record "not computed".
  - Latency: per-image seconds from T3's measured MPS run, when present. Keep the A100 embedding throughput from Delta job logs only if such a log exists in `data/`.
- [ ] **Step 4:** Tests pass. Run `uv run python -m eval.fact_sheet` and read `docs/FACT_SHEET.md` top to bottom against the judge questions.
- [ ] **Step 5:** Commit.

### Task T2: Human-reader baseline and clinical timing literature (P0.6, P1.8)

**Files:**
- Create: `docs/HUMAN_BASELINE.md`, `config/literature.yaml` (verified citations: authors, year, title, journal, DOI/PMID, the exact figures used, the page or table they come from)

- [ ] **Step 1:** Find and verify sources for six areas:
  - Breast-biopsy inter-pathologist concordance by category (B-Path; Elmore et al., JAMA 2015).
  - How BACH labels were made (Aresta et al., 2019).
  - Frozen-section turnaround benchmark (CAP Q-Probes).
  - Routine surgical-pathology turnaround.
  - Intraoperative breast-margin practice and re-excision rates.
  - Second-opinion diagnostic changes in breast pathology, plus the human-vs-AI reader study for breast metastases (CAMELYON16) as a different-task reference.

  Resolve each citation through LibKey when the connector is authenticated; otherwise verify the DOI resolves and the figure appears in the abstract or full text.
- [ ] **Step 2:** Write the matched comparison: Elmore category concordance next to Sepia per-class test recall (from T1). State every mismatch: glass slides vs single fields; whole cases vs curated images; BACH discarded images the two annotators disagreed on, which makes BACH easier than practice. No invented human numbers.
- [ ] **Step 3:** Commit. Output feeds T5 and T6.

### Task T3: Production-model external test on TCGA-BRCA (P1.1, P1.2, P0.7)

**Files:**
- Create: `docs/EXTERNAL_TCGA_R2.md` (pre-registration, committed before any run), `eval/external_tcga_prod.py`, `tests/test_external_tcga_prod.py`
- Modify: `TCGA_script.py` (add `--source tiles`: crops from GDC DeepZoom tiles with the same FOV sampling, level choice and resize as the openslide path; read mpp from the SVS header over an HTTP range request)
- Output: `artifacts/eval/external_tcga_prod.json`. The existing `external_tcga.json` (fine-tune, 8 patients) is kept and reported unchanged.

**Interfaces:**
- Produces: `external_tcga_prod.json`. It has the same `primary`, `slides`, `crops` and `limitations` keys as `external_tcga.json` plus `model`, `n_views`, `latency_s_per_image` (MPS), `cohort` (`protocol25` and `same8`) and `bach_reference` (the same metrics on BACH test, for the side-by-side table).

- [ ] **Step 1: Pre-register.** Write and commit `docs/EXTERNAL_TCGA_R2.md`:
  - Model: production `uni2h512_lr`, read from `model_choice.json`.
  - Embedding: frozen UNI2-h at 512-px regions, 8 dihedral views, BACH temperature, the app's APS q-hat with nonempty, the same tier rule.
  - Reliability: forced off, because CONCH/MUSK are not local. Disclose this.
  - Cohort: the original protocol's 25 patients, seed 44, 6 FOVs, target 0.42 µm/px. Report the 8-patient subset alongside, so it can be compared with the fine-tune run.
  - Analyses: identical to `TCGA_TODO.md` primary and secondary.
- [ ] **Step 2:** Write failing tests:
  - `test_tile_api_crop_matches_manifest_crop` (marked `network`).
  - `test_tcga_views_match_bach_recipe`.
  - `test_production_qhat_matches_app`.
  - `test_slide_score_is_max_over_crops`.
- [ ] **Step 3:** Implement the tile source and the production scorer. Reuse `eval/external_tcga.py` helpers (`tile_case`, `pool_qc`, `analyze`, `slide_rows`) by import, not copy. Time each image's embed + score on MPS.
- [ ] **Step 4:** Run the extraction with `--patients 25 --source tiles`, then the evaluation. Tests pass.
- [ ] **Step 5:** Add a "TCGA-BRCA external test (production model)" section to `eval/results_md.py` → `docs/RESULTS.md`, with the BACH-vs-TCGA table.
- [ ] **Step 6:** Commit code, the pre-registration (already committed), the JSON and RESULTS.md.

### Task T4: Contrastive look-alike detail (P1.3)

**Files:**
- Modify: `pipeline/07_retrieval.py` (write `contrast_detail` to `data/retrieval.json`), `pipeline/08_build_case_json.py`, `core/contract.py` (optional `contrast_case.detail`), `team44-triage-frontend/app/js/views/case_ui.js` (render), types via `make types fixtures`
- Test: `tests/test_retrieval.py`, `tests/test_contract.py`, `tests/test_case_view_static.py`

**Interfaces:**
- Produces (case JSON): `contrast_case.detail: [{concept: str, label: str, query_z: float, contrast_z: float, delta: float, spotcheck: {acceptable: int, reviewed: int} | null, reliable: bool, query_region: int | null, contrast_region: int | null}] | null`.
  - Ordered as `differs_most_in`.
  - `reliable = acceptable / reviewed ≥ 0.75`. The cut-off is fixed now, before looking at which concepts pass.

- [ ] **Step 1:** Write failing tests:
  - `test_contrast_detail_matches_differs_most_in`: same concepts, same order, |delta| equal to the ranking key.
  - `test_contrast_detail_marks_failed_spotcheck`.
  - `test_contrast_detail_null_when_features_missing`.
  - Contract accepts `detail` and `null`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement in 07, carry through 08, then `make types fixtures`.
- [ ] **Step 4:** Render a compact table under the look-alike: concept, this case, look-alike, difference (in SD of training cases), and spot check. Unreliable rows are greyed and labelled "spot check failed; not used as a reason". Add one line: "Concept scores are similarity to text descriptions, z-scored on training cases; they are associations, not a pathologist's reasoning."
- [ ] **Step 5:** Regenerate cases with cached summaries. Diff against HEAD with `contrast_case.detail` removed: expect zero other changes.
- [ ] **Step 6:** Read `backend/vercel.py` and `docs/DEPLOYMENT.md`, and prepare the workspace-version bump the case change requires (documented; not deployed).
- [ ] **Step 7:** `uv run pytest -q` and `PQ_E2E=1 uv run pytest -q tests/e2e`; then commit.

### Task T5: Corrections, scope matrix, Q&A (P0.4, P0.5, P0.12, P1.4–P1.6, P1.9, P2.4–P2.6)

**Files:**
- Create:
  - `docs/JUDGE_QA.md`: the 16 questions; each answer gives a number and its source file.
  - `docs/SCOPE_MATRIX.md`: every feature marked implemented / demonstrated on data / simulated / proposed, and what would validate it.
  - `docs/WSI_AND_EXTENSION.md`: WSI path, disease-agnostic swap points, reader-study and frozen-section test plans.
  - `docs/ATTENTION_COMPARISON.md`.
- Modify:
  - `config/concepts.yaml` display text and the case-view concept labels: say "related to Nottingham component X" only for tubules, pleomorphism and mitoses; never "Nottingham grade".
  - README (exact site URL, data/model credits).

- [ ] **Step 1:** `ATTENTION_COMPARISON.md`. Run a paired grouped bootstrap of the test macro-F1 difference (`abmil` − `uni2h512_lr`) from `predictions.csv`. State what was held fixed (features, splits, folds, selection rule). The complexity explanation is a hypothesis.
- [ ] **Step 2:** Grep app, docs and deck sources for "Nottingham", "nine", "grade" and "Gemini", then fix each one. Test: `test_no_nottingham_grade_claims` scans `team44-triage-frontend/app` and `docs/` for "Nottingham grade" and "nine Nottingham".
- [ ] **Step 3:** Write SCOPE_MATRIX, WSI_AND_EXTENSION and JUDGE_QA, quoting T1–T4 outputs.
- [ ] **Step 4:** Verify the public site logged-out with Playwright (worklist, a case, evaluation page, TCGA page). Record the URLs. Record repo visibility as a user decision.
- [ ] **Step 5:** Commit.

### Task T6: Deck and abstract (P0.8, P0.9, P0.5)

**Files:**
- Modify: `outputs/deck/extract_data.py`, `outputs/deck/build.js`, `docs/DECK_OUTLINE.md`, `docs/ABSTRACT_DRAFT.md`

The 20 main slides follow the judges' spine:

- **Need and case:**
  1. Title.
  2. Problem and who uses Sepia.
  3. BACH_0087 walk-through: uncertain read → set → tier → drafted stain.
- **Mechanism:**
  4. Data and pipeline figure, with each corpus's role and counts.
  5. Splits, units and leakage controls.
  6. Model selection and what the team trained versus reused.
  7. Calibration → conformal set → tier.
- **Evidence:**
  8. Classification with AUROC conventions and the confusion matrix.
  9. In situ vs invasive errors.
  10. Calibration and coverage.
  11. Human-reader baseline, matched with caveats.
  12. External TCGA, production model.
  13. Deferral.
  14. Region contributions and the deletion test.
  15. Concepts: 3 Nottingham-related + 6 custom, with spot checks and the look-alike contrast.
- **Workflow:**
  16. Queue simulation, labelled simulated, next to the cited clinical turnaround benchmarks.
  17. Implemented / simulated / proposed matrix (frozen sections, stain orders, OR board).
- **Credibility and future:**
  18. Limitations, with the clean-data caveat first.
  19. Next steps: WSI, other sites, reader study, other organs.
  20. References.

Then an "End of presentation" divider, followed by the appendix:
- every model's test score;
- the attention comparison;
- simulated margins;
- the oncology inbox;
- summaries and their validator;
- the degradation curves;
- the fine-tune TCGA run.

- [ ] **Step 1:** Extend `extract_data.py` to read `fact_sheet.json`, `external_tcga_prod.json` and `literature.yaml`. Fail on missing keys.
- [ ] **Step 2:** Rebuild the slides in `build.js` per the spine and render the PDF. Visually inspect every page image for overflow and illegible text.
- [ ] **Step 3:** Rewrite the abstract within 300 words. Add the external result and the human baseline sentence. Fix the "nine named histologic features" wording.
- [ ] **Step 4:** Commit tracked files (the deck lives in git-ignored `outputs/`; report its path).

### Task T7 (stretch): Outcome association on TCGA-BRCA (P2.1, P2.2)

Run only if CONCH weights can be loaded locally with the user's existing Hugging Face access, and T1–T6 are done.

- [ ] **Step 1:** Pre-register `docs/OUTCOME_TCGA.md` before any run:
  - Cohort: FFPE DX slides, 6 FOVs each.
  - Score: slide-level mean z of the two Nottingham-related concepts that passed spot check (pleomorphism minus tubules).
  - Endpoints: pN+ (AUROC) and overall survival (Cox HR per SD; KM split at the median; log-rank).
  - Power: state the expected number of events.
  - Framing: the BACH classifier is diagnostic, not prognostic; a null result is reported as such.
- [ ] **Step 2:** Implement with the T3 tile source; run; report in RESULTS.md and the appendix only.

---

## Execution order

T1, T2, T3 and T4 start in parallel: T3 and T4 in separate worktrees by subagents, T2 by a subagent, T1 by the controller. T5 follows T1 and partly T3 and T4. T6 runs last. T7 runs only if time and weights allow. Merge order: T1 → T4 → T3 → T5 → T6. Pushing and the repo visibility decision go back to the user.
