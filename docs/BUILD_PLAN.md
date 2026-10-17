# PathQueue — one-shot build plan

## Context
Team 44 (Rishabh, Aarushi Das, Cheryl Tsai, Nadia Figueroa, Neev Patel) builds **PathQueue** as specified in
`plan.md` + `CELLS_SPEC.md` (sources of truth: breast/BACH only, Case JSON contract §7, tier rules §1, Rules for
Claude Code §16, cell module behind `cells.enabled`). The repo holds only specs, `resources/` (official NCSA tooling
snapshot + decks), `scripts/` (Delta SSH + hosted-model helpers) and `site/`; nothing is built. The user asked for
the best achievable results by fully leveraging the event resources (Delta GPUs across all 5 members' 10 GPU-h
budgets, CONCH/MUSK + other foundation encoders, Quilt-1M/PathGen, Lumen), no shortcuts, and authorized me to pick
the best option wherever the spec is weak. The build should be **parallelized across strong-reasoning subagents**
(§C). The user takes **no manual steps beyond approving Duo pushes on their phone**: I log in to Delta myself through
Open OnDemand in the Aside browser and submit the jobs under strict guardrails (§E). Hard clock: reserved GPUs, hosted servers and Lumen all end **Sun Oct 18
18:00 CDT**; GPU jobs must not cross Sun 07:59/08:00 or 18:00; deliverable (abstract + ≤20-slide stand-alone deck)
due Wed Oct 21, so real numbers + screenshots are needed by Sun evening.

This is a **research prototype evaluated on a small public dataset**, not a clinical device. That framing ("suggested",
"associated with", "simulated", "research use only") is built into every surface and is what makes results defensible.

---

## A. Decisions (spec refinements; each recorded in README "Design notes")

**Data & evaluation integrity**
1. **Use the official BACH release (Zenodo 3632035, 10.4 GB, CC BY-NC-ND), not only `pathmmu/BACH`.** PathMMU stores
   BACH as JPEG-converted multiple-choice items with labels only in question JSON. The official zip has lossless
   2048×1536 TIFFs (0.42 µm/px), the **patient-ID file** (`ICIAR2018_BACH_dataset_micro_patient.xlsx`), and 30
   training WSIs (10 with XML region annotations, 0.467 µm/px). A CPU job downloads it to the team dir (organizers ask
   for downloads in CPU jobs; 10.4 GB is well under the 200 GB ask-first threshold; the user authorized the controller
   to act in place of plan §5.2's "human downloads"). `00_inventory.py` still
   inventories `pathmmu/BACH` first and records the comparison.
2. **Splits = the eva/THUNDER fixed patient-safe index ranges: train 218 / calibration 50 / test 132.** The patient
   file identifies only 145 of 400 images, and patients span classes (one has 36 images from benign to invasive),
   so a home-made 60/20/20 split would leak. THUNDER's ranges over the sorted per-class file lists are asserted
   leakage-free against the patient file, and we re-run that assertion. Our test set *is* the THUNDER leaderboard
   test set, which makes results directly comparable. This replaces plan.md's 60/20/20 for a stated reason.
   CV inside train groups by known patient ID, else by contiguous file-number block.
3. **Model/hyperparameter selection by patient-grouped CV inside train; calibration split reserved for temperature,
   conformal, reliability and margin thresholds.** You first chose the spec wording (select on calibration), then
   said to use my recommendation where something is weak. With the expanded search below (several encoders × heads ×
   ensembles), choosing on the 50 calibration cases would overfit the choice and inflate the conformal coverage we
   claim. So selection moves to CV, and the same ruling covers CELLS_SPEC §7.1's "choose on calibration".
   - **Pre-registered candidates (≤6 + B variants)**, chosen by the **one-standard-error rule**: CV SE (3–5 points)
     exceeds published encoder gaps, so among configs within 1 SE of the best, the simplest wins, and concept
     bottleneck B wins any tie.
   - **Temperature** is fit on the **out-of-fold train logits** CV already produces (218 cases instead of 50). The
     conformal score function is then fixed before the calibration split is seen, which keeps conformal
     exchangeability exact.
   - The test split is touched once. *(Veto at plan approval if you want the literal spec.)*
   - CELLS_SPEC 04a ranks tiles with the **microscopy-only** out-of-fold tile probe, as specified.
4. **Benchmark anchor comes free:** our test split is THUNDER's. We report macro-F1 + balanced accuracy beside the
   published linear probes (CONCH 84.4, UNI2-h 84.7, MUSK 69.9), labeled "same split, different protocol" (tiles +
   TTA vs whole-image resize). No comparison is made to the 87% challenge result, which was measured on a
   different, hidden test set.
5. **Statistics:**
   - **Cluster bootstrap** (2000) over patient / pseudo-patient groups for every CI, since image-level resampling is
     too narrow with few patients; n_patients reported per split.
   - Brier + NLL alongside ECE (10 equal-mass bins, flagged as biased at small n).
   - **Conformal default = marginal split LAC** (plan.md): k=⌈(n+1)(1−α)⌉ = 46 of 50 at α=0.1; coverage given the
     calibration set is ~Beta(46,5), SD ≈0.04, stated on the Metrics page. APS and Mondrian (≈12/class, near-max,
     conservative) are secondary, with per-class coverage on test.
   - No coverage claim is made for degraded or composite variants.
   - **Tier safety:** argmax = Invasive *or* Invasive ∈ set ⇒ tier 2, always; an empty set only adds reason
     `empty_set` (tier 3 when argmax isn't Invasive).

**Models — "best results" stack (all selection by CV, §A3)**
6. **One embedding job per stage on our own `gpuA100x4` allocation** (not the shared hosted server). Text prompts must be
   encoded locally anyway; image and text then come from the identical adapter (`resources/official/models.py`).
   Hosted `/embed` is used only for a 5-tile equivalence check.
   - **Reuse (plan §2):** `models.py` adapters, `submit.sbatch`'s environment pattern, and `train_probe.py`'s
     grouped-split idea. We don't reuse `run_embeddings.py`'s JSON-in-CSV output: it re-reads the whole CSV on
     every flush, which is far too slow for ~500k embeddings. Its resume semantics are kept, with npy chunks.
7. **Multi-scale, augmented features at each encoder's native scale:**
   - CONCH/MUSK use the 512-px grid (3×4, the contract's tiles; ≈0.48 µm/px after the 448 resize).
   - 224-input encoders (UNI2-h, H-optimus-0, …) use the 256-px fine grid (6×8, ≈0.48 µm/px).
   - CONCH is *not* used on 256-px crops, which would upsample to ≈0.24 µm/px.
   - Dihedral views (×8 at 512, ×4 at 256) for train augmentation; cal/test are TTA-averaged on the node before
     export.
   - For CONCH and MUSK we save "align" embeddings (hosted-equivalent: CONCH `proj_contrast=True` normalized 512-d;
     MUSK `with_head,out_norm` 1024-d) *and* linear-probe features (CONCH `proj_contrast=False`; MUSK `ms_aug=True`
     2048-d), from one trunk forward pass where the API allows.
   - OOD/retrieval neighbors use TTA-averaged train embeddings, so augmented copies never count as neighbors.
8. **Extra encoders for the black-box ceiling** (HACKATHON_COMPUTE §4 lists them as "strong for classifiers on
   embeddings"). Rishabh's HF access, checked live:
   - **granted:** CONCH, MUSK, PathGen-CLIP (a third text-aligned model; best published BACH zero-shot);
   - **automatic gate, one click each** (I request these via Aside, §E): UNI2-h, Virchow2, H-optimus-0 (Apache-2.0);
   - **open:** Phikon-v2;
   - H-optimus-1 is a manual gate and is skipped.

   Each loader follows its model card exactly and is verified in the smoke job. Priority order is UNI2-h and
   H-optimus-0, then the others as time allows.

   **Pre-registered case-level candidates:**
   1. CONCH-probe mean-pooled L2-LR (512);
   2. UNI2-h mean-pooled L2-LR (256);
   3. H-optimus-0 mean-pooled L2-LR (256);
   4. ABMIL on the best single encoder (gives attention heatmaps too);
   5. late-fusion ensemble of the top-3 encoders' calibrated probabilities;
   6. MUSK mean-pooled L2-LR (agreement model; selectable).

   Plus concept bottlenecks B, B+C, C. Different encoders are never mixed in one index; fusion happens at the
   probability level only.
9. **Clean tile labels from BACH's annotated WSIs:** tiles extracted (CPU job, openslide) from annotated regions,
   resampled to 0.42 µm/px. Uses: (a) the tile probe (heatmaps + margins) gets true tile labels instead of propagated
   case labels; (b) **pixel-level margin ground truth**: 2048×1536 windows from held-out annotated WSIs, with
   patient-disjoint WSI folds. WSI↔microscopy patient overlap can't be fully checked (most microscopy images are
   unidentified), and scanner shift and "unannotated ≠ normal" make WSI labels imperfect. So **anything scored on
   the microscopy test split uses WSI-free models**: case-level candidates, 04a, margin reads, composites. The
   WSI-enhanced tile probe is reported as an ablation, drives the UI heatmap only if it wins on calibration, and
   is evaluated on held-out WSI windows.
10. **LoRA fine-tuning (optional, first item cut):** to avoid the leakage of fine-tuning on train and then
    CV-selecting on the same images, it trains on the **train split only** and is compared against the frozen champion
    in a **single pre-registered comparison on the calibration split**: adopt if balanced accuracy is +2 points
    with no NLL regression. It is disclosed as one calibration-split decision. Its test result is always reported
    as a separate row. Concepts stay on frozen CONCH. Weights stay private (CC BY-NC-ND).
11. **Data-driven concept vectors from Quilt-1M + PathGen**, honestly labeled:
    - caption mining (regex synonym lists per concept, breast/H&E filters) → CONCH-embedded positive/negative images
      → linear concept vectors;
    - held-out AUROC with splits **by video/slide**, so near-duplicate Quilt frames don't leak;
    - the label is "**caption-consistent**", not "validated": it measures caption co-mention, not tissue truth.
      `invasion` / `duct_confined` are additionally checked on WSI invasive vs in-situ regions;
    - PathGen is used only if inventory shows patch images (it may be captions + 2k slides only);
    - Quilt images never appear in the deck;
    - concepts carry `source` (text / cav / cell) and `components {text_z, cav_z}`, so the contract's `value`
      keeps one meaning.

    Text prompts follow the CONCH recipe (pathologist-style synonyms × ~6–22 templates, ensemble-averaged) and
    **never use negation** ("no X"), which CONCH can't parse. Contrast prompts describe the contrasting tissue
    affirmatively.

**Product logic**
12. **Margin read localizes instead of echoing the diagnosis** (side-agent note: with 512-px tiles, 10 of 12 tiles touch
    an edge). Each margin = outermost **256-px strip (≈107 µm at the measured mpp)**, 24 segments per image.
    - **Fine-scale probe:** binary carcinoma (InSitu ∪ Invasive, since DCIS at ink also matters) on a 224-native
      encoder's 256-px embeddings (UNI2-h or H-optimus-0, whichever passes smoke; CONCH 512-px windows at stride 256
      as fallback). Trained WSI-free with propagated labels; the WSI-label variant is an ablation.
    - **Calls:** per-margin score = max over segments. The threshold is the (1−`max_fpr`) quantile of **image-level**
      per-margin maxima on benign calibration images (≈12 images, reported with a CI). Every benign margin is a true
      negative, so the false "tumor at margin" rate is controlled. Defer if reliability flag / tumor_at_margin (named
      margins + involved length in µm) / clear.

    **Validation (`eval/margin_eval.py`):**
    - per-margin false-positive rate on benign test images, with a CI;
    - **composite-field test**: 64 fields built from test images only, half carcinoma + half benign, scored **per
      segment**. Every benign-half segment is an exact negative inside a carcinoma case: an echo scores 0% clear
      there, a localizer scores high;
    - real-image echo statistics: P(≥1 clear margin | carcinoma), and correlation with case risk;
    - per-margin sensitivity/specificity on **annotated-WSI windows**.

    UI draws per-segment colored strips. Margin names: top=superior, bottom=inferior, left=medial, right=lateral;
    anterior/posterior aren't representable in 2D, as the UI notes. Surgical timer goes amber at 15 min and red at
    20 min (CAP Q-Probes: 90% of frozen sections within 20 min).
13. **Contract additions** (§16: fixture + pydantic + TS types change together):
    - `schema_version`;
    - `margin.{strip_um, threshold, segments[{x,y,size,p}]}`; a `margin` finding kind;
    - `cells` block (CELLS_SPEC §7.2); optional `attention` per tile (ABMIL);
    - `concepts[].{source, components, caption_consistent}`;
    - `simulated_variant` + `source_case_ids` (degraded / composite cases);
    - `flag.reasons` as an enum (`aggressive_top`, `aggressive_in_conformal_set`, `set_size_gt_1`, `empty_set`,
      `low_reliability`, `insufficient_tissue`, `aging_promoted`, `surgical_frozen_section`);
    - `summary.generator` badge (LLM model or template).
14. **Stain rules** (header "illustrative; requires clinical review", guideline citations in comments):
    - `breast_insitu_vs_invasive` → p63 + calponin (plan §9/§14 wording; SMMHC noted as the CAP-preferred
      alternative; ≥2 markers including a nuclear one; normal ducts as internal control);
    - `breast_invasive` → ER, PR, HER2 (IHC, reflex ISH if 2+), Ki-67 (IKWG: prognostic use in ER+/HER2−);
    - `breast_insitu` → ER (ASCO/CAP 2020 recommends ER testing for DCIS);
    - `uncertain_benign_atypia` → deeper levels + CK5/6 and ER;
    - `low_quality` → recut / deeper levels;
    - `insufficient` → request new sample.
15. **Shared pure logic in `core/`** (config, contract, tiers, conformal, calibration, margin) imported by pipeline,
    eval, agent and backend. Tier assignment lives in one function, as plan.md requires.
16. Fixture in `fixtures/demo_0001.json` (committed, synthetic placeholder image). Surgical timers use wall clock;
    arrivals use the simulator clock. Demo hero cases are auto-picked by step 08 and placed late; metrics always use
    the full test split.
17. **LLM:** Lumen profile (live): 10.00/10 coins, refill only +0.10/h; `ornith-1.0-35b` (cheapest, no consent needed)
    is the default, then `gemma-4-31b-it`; `qwen3-coder-next` needs a consent click, so it isn't required. One probe
    call measures the coin cost, which sets the cap; organizers have reset exhausted coins for teams on request.
    Tool calling and JSON mode are probed once, with fallback to a JSON prompt, then the template. Calls are capped
    (`--max-llm-calls`, flagged/hero cases first) and run before Sun 18:00 (Lumen resets). The cache is keyed by a
    hash of each case's findings, so any rerun of 05–08 falls back to the template instead of showing stale text.
18. **Cells (CELLS_SPEC)** built in full behind the flag.
    - The spike is a `--limit 5` batch job on arbitrary tiles, run early, which validates the install (interactive
      partitions bill double).
    - 04a runs *inside* `cells.sbatch` from on-disk embeddings: its out-of-fold logistic regression takes seconds,
      so there is no laptop round trip.
    - Overlays get an AI review (§E). Backend/install per §F; hard stop **Sun 10:00**.
19. **Compute etiquette:**
    - All GPU work runs on Delta `gpuA100x4`. **The controller submits jobs itself through OOD** (user-authorized,
      §E) via the guarded `pipeline/jobs/submit.sh`: one GPU at a time, window/deadline checks, a 7 h cap, a log,
      no auto-resubmission.
    - **DeltaAI:** organizers posted "Don't use DeltaAI" (Oct 16 18:46) and said it has only "a few hours total GPU
      time" (Oct 17 10:42). It is used for read-only file inspection only, if a CLI session ever exists. It is
      aarch64, so nothing is installed from it.
    - Login nodes 01/04 (organizers: 02/03 rebooting).
    - Downloads run in a CPU job per organizer guidance, with a login-node `nice wget` fallback.
20. **Secrets and licensed weights:** keep `HF_HOME` in the private home cache (~17 GB of weights fits the 100 GB home).
    - This deliberately departs from CLAUDE.md's and organizers' "HF_HOME on personal NVMe". `/work/nvme` is
      event-readable, so it would expose the HF token *and* non-shareable gated weights. If the token is already
      there, it is moved to home and the directory gets `chmod 700`.
    - Job scripts set `HF_HOME` explicitly. Keys are only read from `~/.config/ashby-2026/*.env` or env vars,
      never printed.
21. **Teammate frontend** (`team44-triage-frontend/`, vanilla JS, mock colon data from the earlier multi-organ plan;
    by a teammate) **stays untouched**. plan.md, revised after it, specifies React. The new `frontend/` ports its best
    ideas, credited in the README:
    - the **arrival-order vs AI-priority toggle** (the clearest demo of the reorder);
    - an **aging guard**: a case that has used ≥75% of its tier's turnaround target is promoted one tier (never above
      tier 2; reason `aging_promoted`; config `aging.enabled`). Its effect is measured in the queue sim (FIFO vs tiers
      vs tiers+aging), the starvation risk Baltruschat 2021 warns about;
    - an **OR board** (surgeon screen with room + countdown);
    - **visibly blocked agent actions** (e.g. stains are never drafted in the frozen lane).
22. **Package isolation:** organizer-verified `pip install --user` covers CONCH/MUSK only. Everything else lives in
    `--system-site-packages` venvs with a constraints file pinning the module's torch/numpy/timm/transformers, so
    MUSK/CONCH can't break. The smoke job prints all versions. sklearn objects are never pickled across machines:
    models export weights as npz.
23. **Licensing of derived images:** BACH is CC BY-NC-ND. Tiles, overlays, composites, degraded variants and cell
    overlays are adaptations: never committed, never hosted publicly (the app runs locally; no GitHub Pages). The
    deck uses original fields with attribution, and showing overlays there is flagged for the team in the final
    report.

---

## B. Repository layout (repo root = plan.md's `pathqueue/`; existing README/scripts/site preserved)
```
pyproject.toml uv.lock Makefile .env.example README.md   (.gitignore += artifacts/ data/ data_synthetic/ .venv/
                                                            node_modules/ frontend/dist/ logs/ *.npy *.npz docs/screenshots/)
config/   settings.yaml concepts.yaml stain_rules.yaml cell_features.yaml encoders.yaml caption_mining.yaml
core/     config.py contract.py tiers.py conformal.py calibration.py margin.py io.py stats.py
pipeline/ common.py 00_inventory.py 01_build_cases.py degrade.py composites.py wsi_extract.py mine_captions.py
          02_embed.py 03_text.py make_synthetic.py 04_concepts.py 04_cav.py 04a_select_tiles.py 04b_cells.py
          04c_cell_features.py 05_risk_model.py 05b_abmil.py 05c_finetune.py 06_reliability.py 07_retrieval.py
          08_build_case_json.py
          jobs/ prep_a.sbatch prep_b.sbatch embed_core.sbatch embed_extra.sbatch embed_aux.sbatch cells.sbatch finetune.sbatch
                submit.sh   (scripts/ood/: OOD bridge — controller-only)
eval/     metrics.py deletion_test.py queue_sim.py margin_eval.py cells_eval.py benchmark_thunder.py figures.py
          results_md.py
agent/    rules.py agent.py lumen.py
backend/  app.py db.py simulator.py summary.py
team44-triage-frontend/ teammate's vanilla-JS app, connected to the API and extended (ruling R6); frontend/src/types.ts = contract mirror
fixtures/ demo_0001.json demo_0001.jpg (synthetic)
scripts/  sync-delta.sh (tar-over-ssh via delta.sh, or `ssh deltaai` fallback; push code / pull data)
tests/    test_tiers test_conformal test_calibration test_stats test_summary_validator test_rules test_margin
          test_contract test_degrade test_composites test_cell_features test_api test_embed_local test_pipeline_synthetic
docs/     RESULTS.md (generated) screenshots/
```
Delta: code `/u/rbansal3/pathqueue`; data `/work/nvme/bfqi/teams/team44/pathqueue/` (event-readable, no secrets;
`team.dir` in config so it is one edit if the team prefers another name); official BACH `/work/nvme/bfqi/teams/team44/bach/`.

---

## C. Parallel execution protocol (subagent-driven, per `superpowers:subagent-driven-development`)

**Roles.** I am the controller/integrator: I write no feature code myself. I dispatch implementer subagents, review
their work, merge it, run the Delta runbook (§E: OOD login via Aside, installs, guarded job submission, pulls), and
make rulings. Every
implementer and reviewer is dispatched with an explicit model (the user asked for the strongest reasoning):

| Role | Model | Effort |
|---|---|---|
| Correctness-critical implementers (Tasks 1, 2, 3, 4, 8, 10, 11, 12) | `opus` | `max` |
| Other implementers (Tasks 5, 6, 7, 9, 13) | `opus` | `xhigh` |
| Per-task reviewers | `opus` | `high` |
| Statistics reviewer (Tasks 4, 10, 11) and final whole-branch reviewer | `opus` | `max` |

**Isolation for parallelism.** The skill's default is one implementer at a time, to avoid conflicts. The user asked
for parallel subagents, so conflicts are prevented structurally instead:
- each implementer runs with `isolation: "worktree"` on branch `task/<N>-<slug>` cut from `build/pathqueue`
  *after Task 1 merges*;
- each task owns a disjoint set of paths (listed per task) and edits nothing else;
- shared files (`pyproject.toml`, `Makefile`, `config/settings.yaml`, `core/contract.py`, `core/schemas.py`,
  `frontend/src/types.ts`) belong to the controller. A task that needs a dependency, Makefile target or config key
  lists it under "Requested shared changes" in its report, and I apply it at merge;
- cross-task interfaces are frozen as signatures + docstrings in Task 1 (stubs). Later tasks code against them and
  test with `data_synthetic/`.

**Waves and merge order.**
- **Wave 0:** Task 1 alone. In parallel, I run §E steps 1–2 (OOD bridge, Delta login, HF access + token), which
  need no repo code.
- **Wave 1** (≤5 concurrent; priority order 2, 3, 5, 4, 7, then 6, 8 as slots free): Tasks 2–8. Tasks 2, 3 and 5
  merge first because they unblock Delta (`prep_a`, embed jobs, and the prompt bank encoded by embed_core).
  The §E runbook starts the moment they merge, while the rest of the waves continue.
- **Wave 2** (parallel): Tasks 9, 10, 11.
- **Wave 3:** Task 12 (integration). Then the real-data run (§E), then Task 13.

**Per task (skill loop):**
1. Record BASE, then `task-brief docs/BUILD_PLAN.md N`. The plan is copied into the repo at start.
2. Dispatch with: brief path; Global Constraints (below, verbatim); the interfaces it consumes; its report path.
   Implementers never spawn subagents.
3. On DONE: build the review package and dispatch the task reviewer (+ statistics reviewer for Tasks 4, 10, 11; +
   `pr-review-toolkit:silent-failure-hunter` for Tasks 2, 3, 6, 8).
4. Fix loop ≤5 rounds (1–3 resume the implementer; 4–5 a fresh `opus`/`max`).
5. Merge into `build/pathqueue`, run full `make test`, append to the ledger.

Ledger: `.superpowers/sdd/<plan>/progress.md`, git-ignored. Rulings are logged as
`Ruling: … — why — cost if wrong` and listed in the final message. Final whole-branch review (`opus`/`max`) +
`security-review` of secrets handling, then one fix wave. Waiting: bounded 5–10 min stretches with a one-line
status, never tight polling.

### C.2 Global Constraints (pasted verbatim into every dispatch and review)
- Binding specs: `plan.md`, `CELLS_SPEC.md`, and §A of `docs/BUILD_PLAN.md`; where they differ, §A states the
  ruling. Scope: breast cancer on BACH only.
- Implementers and reviewers never submit Slurm jobs, never SSH to Delta/DeltaAI, never drive the Aside browser
  (only the controller does, §E), and never run model inference on a login node. BACH-derived images (tiles,
  overlays, composites) are never committed; fixtures use generated non-tissue images. Job scripts
  start with `module load pytorch-conda/2.8` then `unset PYTHONNOUSERSITE`. GPU jobs: `--account=bfqi-delta-gpu
  --partition=gpuA100x4 --gpus-per-node=1 --cpus-per-task≤16 --mem≤60G --time≤01:00:00`; CPU jobs:
  `bfqi-delta-cpu`/`cpu`. Compute nodes are offline: `HF_HUB_OFFLINE=1`, weights pre-cached in the home HF cache.
- No secrets in code, logs, tests, fixtures or reports. Keys come only from env vars or `~/.config/ashby-2026/*.env`;
  never print them. Never commit data, embeddings, artifacts, `.env`.
- Every class name, threshold, alpha, path and seed comes from `config/`; no magic constants. The aggressive class
  is `Invasive`.
- Ground truth lives only in `eval_only` / eval files and never reaches the API or UI. UI wording: "suggested",
  "associated with", "simulated", "research use only"; the model never "diagnoses".
- Edit only your owned paths. Request shared changes in your report.
- Delta-side code (`pipeline/00–03`, `wsi_extract`, `mine_captions`, `04b`) must run on Python ≥3.10 with only the
  `pytorch-conda/2.8` packages plus those §E installs. Laptop code targets Python 3.13 via `uv`.
- Every step: idempotent, resumable, atomic writes (`core/io.py`), `--limit N`, and a `meta.json` (inputs, config
  hash, git SHA, versions).
- Tests: pytest, no network, deterministic seeds. TDD for logic (`superpowers:test-driven-development`). Done means
  `make test` passes for your paths and you've shown the output.
- Commit on your branch with clear messages ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## D. Tasks (each `### Task N` block is a self-contained brief)

### Task 1: Foundation — environment, config, contracts, schemas, interface stubs (Wave 0)
**Owns:** `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore` (+ artifacts/ data/ data_synthetic/ .venv/
node_modules/ frontend/dist/ logs/ *.npy *.npz docs/screenshots/ .superpowers/), `.env.example`,
`config/settings.yaml`, `core/__init__.py`, `core/config.py`, `core/io.py`, `core/contract.py`, `core/schemas.py`,
`core/tiers.py` (full), stubs for `core/conformal.py`, `core/calibration.py`, `core/stats.py`, `core/margin.py`,
`pipeline/degrade.py` (signatures + docstrings + `NotImplementedError`), `fixtures/` (demo_0001 biopsy + demo_0002
surgical JSON + a generated non-tissue placeholder JPG), `frontend/src/types.ts`, `tests/test_contract.py`,
`tests/test_tiers.py`, `tests/test_config.py`.
- `uv` project on Python 3.13: numpy, pandas, scikit-learn, scipy, matplotlib, pyyaml, pillow, pydantic 2, fastapi,
  uvicorn, httpx, openai, requests, pytest, openpyxl, torch (CPU/MPS wheel). `make setup|test|synthetic|analyze|demo|e2e`
  targets, with stubs where later tasks fill them.
- `settings.yaml` declares every key later tasks need: organ, dataset, classes, aggressive class; `team.dir:
  team44`; Delta + local paths (`PATHQUEUE_DATA_ROOT` override); `split.scheme: thunder` with the exact index ranges
  of Task 2; tile sizes 512/256; QC thresholds; lanes; `alpha: 0.1`; `conformal.{method: lac, mode: marginal}`;
  concept z-threshold 1.0; `model_choice` margins (AUROC 0.03, BAcc 0.05); reliability (k=10, target error recall
  0.8); margin (`strip_px: 256`, `max_fpr: 0.05`, timer amber 15 min / red 20 min); sim speed 30; the `llm` block;
  the CELLS_SPEC §9 `cells` block; demo hero rules.
- `core/contract.py`: pydantic models exactly matching plan §7 + CELLS_SPEC §7.2 + every §A13 addition
  (`schema_version`, margin segments, `margin` finding kind, `cells`, `tiles[].attention?`,
  `concepts[].{source,components,caption_consistent}`, `simulated_variant`, `source_case_ids`, `flag.reasons` enum,
  `summary.generator`, top-level `synthetic?`). `CasePublic` excludes `eval_only`. Fixtures validate. `types.ts`
  mirrors it field for field.
- `core/schemas.py`: column/shape specs + path helpers for every inter-task file: `cases.csv`, `tiles.csv`,
  `embed_manifest.csv`, `emb/<encoder>/<set>_<head>.npy` + `_index.csv` + `meta.json`, `text/<model>.npz`,
  `concepts_tile.npz`, `case_features.csv`, `tile_risk.csv` (+ `_oof`), `predictions.csv`, `retrieval.json`,
  `reliability.csv`, `margins.json`, `cell_tiles.csv`, `cells/…`, `eval/*.json`.
- `core/tiers.py` (full): `assign_tier(case, cfg, now=None) -> (tier, reasons)` and `sort_key`.
  - surgical → 1;
  - aggressive top **or** aggressive in conformal set → 2, *even when the set is empty*;
  - set size ≠ 1, reliability flag, or insufficient tissue → 3;
  - else 4;
  - optional aging guard (§A21): ≥75% of the tier's turnaround target used → promote one tier, never above 2;
  - within a tier: risk descending, then arrival.

  Tests cover the safety rule, empty sets with argmax Invasive, aging promotion caps, and ties.

### Task 2: Delta data preparation (Wave 1, critical path)
**Owns:** `pipeline/common.py`, `pipeline/00_inventory.py`, `pipeline/01_build_cases.py`, `pipeline/degrade.py`
(implement), `pipeline/composites.py`, `pipeline/wsi_extract.py`, `pipeline/mine_captions.py`,
`pipeline/jobs/prep_a.sbatch`, `pipeline/jobs/prep_b.sbatch`, `scripts/sync-delta.sh`, `tests/test_inventory.py`,
`tests/test_split.py`, `tests/test_degrade.py`, `tests/test_composites.py`, `tests/test_wsi_extract.py`,
`tests/synthetic_tree.py` (builds fake BACH/NCT/Quilt trees in tmp dirs).
- `00_inventory.py` (login-node safe: listings, image headers, `nrows`-limited CSV peeks only):
  - `pathmmu/BACH`: structure, label source, counts, formats;
  - official BACH: `Photos/<Class>/*.tif`, WSI `.svs` + `.xml`, patient xlsx;
  - measured mpp; NCT class folders; Quilt CSV columns + breast-row estimate; PathGen layout.

  Writes `inventory.json` + summary. Gate: ≥80 labeled per class.
- `01_build_cases.py`:
  - **Split:** classes sorted, files sorted, assert 100 per class, then index ranges (eva/THUNDER):
    - train `(0,15),(25,41),(59,60),(90,110),(120,139),(169,200),(210,240),(258,260),(273,305),(315,345),(368,380),(390,400)`;
    - cal `(15,25),(110,120),(200,210),(305,315),(380,390)`;
    - test `(41,59),(60,90),(139,169),(240,258),(260,273),(345,368)`.

    Assert zero patient overlap via the xlsx (Histology→Patient). Within train, CV groups = patient ID, else
    contiguous 5-number blocks.
  - **Lanes:** ~25% of test cases surgical, covering Invasive, InSitu and Normal/Benign.
  - **QC:** tissue fraction from saturation + brightness thresholds.
  - **Tiles:** 512 (3×4) and 256 (6×8) PNGs; display JPEG (long side 1536) for all 400 cases.
  - **NCT sample:** 300 per class, seeded.
  - **Composites:** 64, test images only, half carcinoma + half benign, 4 orientations, per-margin truth table
    (`exact_negative` / `tumor_side` / `mixed`), display JPEG.
  - **Degraded displays:** test-surgical cases × 4 types at severity 3.
  - **Manifest:** `embed_manifest.csv` with sets `bach512, bach256, nct, deg512 (cal+test × 4 types × 3 severities),
    comp256, wsi256, wsi_margin256, quilt, pathgen` and dihedral `view` 0–7.
- `degrade.py` (pure PIL/numpy, deterministic per seed): Gaussian blur, JPEG quality, H&E optical-density stain
  shift (Ruifrok matrix), down-up sampling; 3 severities each from config.
- `wsi_extract.py`:
  - openslide; parse XML polygons into masks, resample 0.467→0.42 µm/px;
  - labeled 512/256 tiles with ≥90% label purity;
  - held-out 2048×1536 margin windows stratified into tumor-at-specific-edge / central-only / none, with per-margin
    truth from the 256-px strips;
  - patient-disjoint WSI folds.
- `mine_captions.py`: chunked `usecols` reading of the Quilt CSV + PathGen captions; regex synonym lists from
  `config/caption_mining.yaml` (owned by Task 5); organ filters; per-concept positive/negative caps; seeded.
- `prep_a.sbatch`: Zenodo download (exits fast if offline), unzip, inventory, `01 --limit 5`, full `01` (splits,
  tiles, displays, core manifest). `prep_b.sbatch`: composites, WSI extraction (activates the `wsi` venv) and caption
  mining, appending the aux manifest.
- `sync-delta.sh push|pull [--via deltaai]`: tar over `scripts/delta.sh` (default `ASHBY_DELTA_HOST=dt-login04…`);
  excludes data/artifacts/.venv/node_modules/.env.

### Task 3: GPU embedding, text encoding, job tooling, synthetic data generator (Wave 1, critical path)
**Owns:** `pipeline/encoders.py`, `pipeline/02_embed.py`, `pipeline/03_text.py`, `pipeline/export.py`,
`pipeline/make_synthetic.py`, `config/encoders.yaml`, `pipeline/jobs/{smoke,embed_core,embed_extra,embed_aux,text_cpu}.sbatch`,
`pipeline/jobs/submit.sh`, `tests/test_embed_local.py`, `tests/test_submit.py`, `tests/test_make_synthetic.py`.
- Efficiency:
  - one trunk forward pass feeds both CONCH heads (pooled → projection);
  - fp16 autocast;
  - views: ×8 at 512 and ×4 at 256, train split only; cal/test reduced to TTA means on the node;
  - `smoke.sbatch` runs on NCT + any `pathmmu/BACH` images, so it never waits for the official download;
  - `export.py` packs the TTA-reduced float16 arrays + CSVs + display JPEGs into one tgz for the OOD download.
- `encoders.py`:
  - CONCH and MUSK "align" embeddings come from `resources/official/models.py` (`resolve_adapter`); assert our align
    output equals the adapter's;
  - "probe" heads: CONCH `proj_contrast=False, normalize=False`; MUSK `ms_aug=True` (2048-d);
  - UNI2-h, Virchow2 (CLS ⊕ mean patch tokens), H-optimus-0, Phikon-v2 (CLS) and PathGen-CLIP
    (`openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt`, or the card's file) are loaded exactly per their
    model cards, with each card's URL and revision recorded in `meta.json`.
- `02_embed.py`:
  - DataLoader (14 workers, pinned memory) over manifest subsets; dihedral views + `degrade.apply` on the fly;
    fp16 on GPU;
  - float16 chunk files with resume, then a final aligned `.npy` + `_index.csv`;
  - aborting checks: L2 norms (align), dims, align==adapter, NCT zero-shot accuracy above the
    `encoders.yaml` floors (organizer reference ≈63% CONCH / ≈77% MUSK; near chance means misaligned).
- `03_text.py`: every prompt in `concepts.yaml` (Task 5) × templates, class prompts, and `prompts_nct.json`
  → `text/<model>.npz` keyed by prompt id. It also runs as the `text_cpu.sbatch` CPU job, so prompt edits cost no
  GPU time.
- `submit.sh <job> [--after JOBID] [--print-only]` (run by the controller on the login node):
  - accepts only an allow-list of reviewed job scripts and runs `sbatch --test-only` first;
  - refuses if now + `--time` crosses Sun Oct 18 07:59/08:00 or 18:00 CDT, if another GPU job of ours is running or
    pending outside the dependency chain, or if cumulative requested GPU time would exceed 7 h;
  - adds `--deadline` overnight and `--dependency=afterok:` when chained;
  - appends every submission to `logs/submissions.log`;
  - never loops or auto-resubmits.
- `make_synthetic.py`: writes `data_synthetic/` in exactly the Task 1 schemas: cases/tiles/manifest, every
  encoder/head/set embedding (small dims), text npz matching `concepts.yaml` ids, NCT, degraded, composites, WSI
  sets. It plants class and concept structure so downstream metrics are non-trivial. Everything is stamped
  `synthetic: true`.
- `test_embed_local.py` runs `02_embed.py` end to end on CPU with the official `test` adapter over a synthetic tree
  (chunking, resume, alignment).

### Task 4: Core statistics, conformal, calibration, margin logic (Wave 1)
**Owns:** `core/conformal.py`, `core/calibration.py`, `core/stats.py`, `core/margin.py` (implement Task 1 stubs),
`tests/test_conformal.py`, `tests/test_calibration.py`, `tests/test_stats.py`, `tests/test_margin.py`.
- **Conformal:**
  - LAC score 1−p̂_y; q̂ = k-th smallest score with k=⌈(n+1)(1−α)⌉, and k>n ⇒ q̂=∞ (full set);
  - set = {y: 1−p̂_y ≤ q̂}; empty sets are allowed and flagged;
  - also APS (randomized, seeded) and Mondrian (per-class q̂);
  - tests: synthetic empirical coverage ≈1−α within the Beta band; edge cases n small.
- **Calibration:** temperature by NLL minimisation (scipy, bounded) on given logits, argmax-invariant; ECE (10
  equal-mass bins); Brier; NLL; reliability-diagram data.
- **Stats:** stratified bootstrap (2000, seeded) percentile CIs for any metric fn; AUROC (one-vs-rest + macro);
  macro-F1; balanced accuracy; confusion matrix; paired-difference CI helper.
- **Margin:**
  - strip geometry: outermost `strip_px` band, 256×256 segments, 8 per long edge and 6 per short edge, corners
    shared, mapped top/bottom/left/right → superior/inferior/medial/lateral;
  - per-margin score = max segment p;
  - threshold = (1−`max_fpr`) quantile of per-margin maxima over benign calibration images;
  - call: defer if reliability flag, else tumor_at_margin (involved margins + involved length in µm via mpp),
    else clear.

### Task 5: Clinical content, agent, grounded summaries (Wave 1)
**Owns:** `config/concepts.yaml`, `config/stain_rules.yaml`, `config/caption_mining.yaml`, `agent/rules.py`,
`agent/lumen.py`, `agent/agent.py`, `backend/summary.py`, `tests/test_concepts_yaml.py`, `tests/test_rules.py`,
`tests/test_summary_validator.py`, `tests/test_agent.py`.
- `concepts.yaml`:
  - the 9 plan §6.4 concepts, with group, label, expected direction, and a `weak` flag (mitoses);
  - 3–5 positive and 2–3 contrast phrasings each, in breast-pathology language using CAP descriptors (§F), all
    affirmative (lint test rejects "no", "without", "absence", "not");
  - ~6–22 templates (CONCH + MUSK styles, §F);
  - header: "draft; requires pathologist review".
- `caption_mining.yaml`: per-concept positive regex synonym lists, exclusion lists, organ filters, caps.
- `stain_rules.yaml`: the six rules in §A14, with guideline citations in comments and an "illustrative; requires
  clinical review" header.
- `rules.py`: deterministic; order list with `rule_id` + template rationale; never invents items.
- `lumen.py`:
  - OpenAI client on `https://lumen.ncsa.illinois.edu/v1`; key from env or `~/.config/ashby-2026/lumen.env`, never
    printed;
  - model picked from `llm.model_preference` ∩ live `/v1/models`;
  - disk cache keyed by (model, prompt hash); retries with backoff; `--max-calls` budget; coin-cost probe helper.
- `agent.py`: tool calling (`get_case_findings`, `lookup_rules`, `draft_order`) → JSON-prompt fallback → rules
  + template. Chosen items must be a subset of rule outputs.
- `summary.py`:
  - findings-only prompt: ≤4 sentences, JSON `[{text, findings}]`, "features associated with", no diagnosis as
    fact;
  - validator drops sentences with no finding ID, an unknown ID, or numbers absent from the cited findings;
  - fewer than 2 survivors ⇒ deterministic template (prediction, concept, cell, margin, quality findings).
  - All tests mock the LLM.

### Task 6: Backend API + simulator (Wave 1)
**Owns:** `backend/__init__.py`, `backend/app.py`, `backend/db.py`, `backend/simulator.py`, `tests/test_api.py`.
- FastAPI + SQLite (tables `cases`, `decisions`, `orders`, `routed_results`, `events`), loading
  `artifacts/cases/*.json` (or `fixtures/` via env).
- Endpoints exactly per plan §10, plus:
  - `POST /api/sim/inject_surgical {variant: normal|degraded|composite}`;
  - `POST /api/spotcheck` → `artifacts/eval/spotcheck.json`;
  - `GET /api/health`, `GET /api/config` (UI-relevant settings: timer thresholds, labels, aging), and
    `GET /api/cases/{id}/events`;
  - `GET /api/worklist?order=ai|arrival` for the teammate-inspired toggle.
- Invalid state transitions return 409. Reset re-seeds every table. Static routes for images, cells overlays and
  heatmaps.
- Every response passes through `CasePublic`, so `eval_only` never leaves. Tiers come from `core.tiers`.
- Agent state machine `none→drafted→approved|edited_and_approved|deferred_closer_look|dismissed`, every
  transition timestamped in `events`. Sign-out routes to the oncologist; `margin_call` routes to the surgeon and
  stops the timer. Timers use wall clock; arrivals use the sim clock (speed ×N); reset restores drafted orders.
- Serves `artifacts/images`, `artifacts/cells` and `frontend/dist`.
- Tests: sorting incl. safety rule, `eval_only` absent everywhere, each decision path, routing, timers, sim controls.

### Task 7: Frontend — connect and extend the teammate's app (Wave 1)
**Owns:** `team44-triage-frontend/**` (the teammate's vanilla-JS app; the user asked to connect it rather than build a
new React app — ruling R6), `tests/e2e/frontend_smoke.*` if added. Read
`.superpowers/sdd/BUILD_PLAN/research/frontend_connect.md` first if present.
- Keep its no-build vanilla JS/HTML/CSS structure, visual language, light/dark styles and the author's ideas
  (arrival vs AI-priority toggle, aging promotion at 75% of target, OR board, blocked agent actions). Extend, don't
  rewrite; credit the author in its README.
- Data layer: replace `data/cases.json` mock reads with the FastAPI endpoints of plan §10 (+ `/api/config`,
  `/api/cases/{id}/events`, `/api/worklist?order=ai|arrival`), mapping the Case JSON contract (`core/contract.py`,
  mirrored in `frontend/src/types.ts` → move/keep a JSDoc typedef copy in the app) to its view model. Keep a
  `?mock=1` mode that loads `fixtures/*.json` so it runs without the backend. Remove colon/multi-organ mock data
  from defaults (breast/BACH only); keep the old file as `data/legacy_mock.json` untouched for reference.
- Views to support (plan §11 + CELLS_SPEC §7.3): worklist (tiers, timers with amber/red thresholds from config,
  reason badges, conformal-set chips, risk, orders-drafted marker, arrival animation, order toggle); biopsy case
  (tile heatmap over the display image with selector risk / each concept / attention; Why chart with signed
  contributions marked measured vs estimated and caption-consistent; How-sure; Compared-to with contrast case;
  summary sentences with clickable finding refs and template/LLM badge; agent panel with Order tests / Edit /
  Take a closer look / Dismiss / Sign out; Cells tab with overlay toggle + legend); surgical case (big timer,
  segmented margin strips, reliability, three calls, "Simulated frozen section" banner, anterior/posterior note);
  OR/surgeon board; oncologist inbox; Metrics (charts from `/api/metrics`, plain SVG or a CDN-free inline approach);
  Spot-check view. Persistent "Research prototype — not for clinical use" banner; explicit loading / backend-down /
  empty / cells-disabled / `margin: null` states; ground truth never rendered.
- `data-testid`s on everything Playwright drives; works at 390 px and desktop; no external CDNs (offline demo).
  The backend (Task 6) serves this folder at `/`.

### Task 8: Cells module (Wave 1)
**Owns:** `config/cell_features.yaml`, `pipeline/04a_select_tiles.py`, `pipeline/04b_cells.py`,
`pipeline/04c_cell_features.py`, `pipeline/jobs/cells.sbatch`, `pipeline/jobs/cells_env.sh` (login-node venv
recipe, printed for the controller), `eval/cells_eval.py`, `tests/test_cell_features.py`, `tests/test_select_tiles.py`.
- Implements CELLS_SPEC fully, with §F backend order:
  1. TIAToolbox 2.1.3 `hovernet_fast-pannuke`, run with an explicit weights path, `torch_compile_mode="disable"`,
     tiles pre-resized by measured mpp / 0.25;
  2. vqdang/hover_net with the `imgaug` line removed;
  3. HoVer-NeXt.
- Outputs are normalized to CELLS_SPEC §5 per-nucleus JSON in µm/µm² + overlay PNGs with §5 colors.
- `04a` ranks tiles by the **microscopy-only out-of-fold** tile risk. It runs *inside* `cells.sbatch` before
  segmentation: a 5-fold grouped L2-LR on the on-disk CONCH train tile embeddings with case labels propagated (out
  of fold for train; full-train fit for cal/test), writing `cell_tiles.csv`. Task 10 reuses the same function. Tests
  use synthetic data.
- The `cells_spike` mode segments 5 arbitrary tiles from different classes and writes overlays for the controller's
  AI review.
- `04c` computes features, minimum counts, missing indicators and benign-reference percentiles; `cells_eval.py`
  covers CELLS_SPEC §8 (convergent validity, direction checks, review sheet).
- `--limit`, resume, and a `stretch: margin_strips` option.

### Task 9: Concept scoring + data-driven concept vectors (Wave 2)
**Owns:** `pipeline/04_concepts.py`, `pipeline/04_cav.py`, `tests/test_concepts.py`, `tests/test_cav.py`.
- Tile concept score = mean cos(positives) − mean cos(contrasts) over the template ensemble (normalized prompt
  embeddings averaged per phrasing), z-scored on train tiles; spread = std across phrasings.
- Concept vectors: L2 logistic regression on CONCH align embeddings of mined Quilt/PathGen positives vs negatives;
  held-out AUROC; `validated = AUROC ≥ 0.75`; final concept value = mean of z(text) and z(vector) when validated,
  else z(text).
- Case features: top-3 mean + fraction above z>1.0 (18 features).
- Eval outputs: NCT cross-organ AUROCs (lymphocytes→LYM, necrosis→DEB, pleomorphism→TUM,
  benign_stroma_fat→STR+ADI); BACH per-class means (test); zero-shot BACH 4-class baseline per text-aligned model →
  `eval/concepts.json`, `eval/concepts_bach.json`, `eval/zeroshot.json`.

### Task 10: Risk models, calibration/conformal application, reliability, retrieval, fine-tune (Wave 2)
**Owns:** `pipeline/05_risk_model.py`, `pipeline/05b_abmil.py`, `pipeline/05c_finetune.py`,
`pipeline/jobs/finetune.sbatch`, `pipeline/06_reliability.py`, `pipeline/07_retrieval.py`, `tests/test_risk_model.py`,
`tests/test_reliability.py`, `tests/test_retrieval.py`.
- **Candidates (pre-registered, §A8):**
  1. CONCH-probe LR (512);
  2. UNI2-h LR (256);
  3. H-optimus-0 LR (256);
  4. ABMIL (torch, MPS/CPU) on the best single encoder;
  5. top-3 late-fusion ensemble;
  6. MUSK LR.

  Plus concept bottleneck B, B+C, C. All are WSI-free.
- **Selection:** patient-grouped 5-fold CV inside train only (repeat ×3) by log-loss, with the **one-SE rule**: the
  simplest config within 1 SE of the best wins, and B wins any tie. The A-vs-B rule (margins from settings) decides
  what drives predictions. The choice and the full candidate table go to `eval/model_choice.json`.
- **Calibration and conformal:** temperature from out-of-fold train logits; marginal LAC on calibration (+ APS,
  Mondrian secondary); risk = calibrated P(Invasive).
- **Contributions:** centered contributions (W_Invasive − mean_k W_k)·x_std per concept/cell feature.
- **Tile probes:**
  - microscopy-only (out-of-fold for train → `tile_risk_oof.csv`, shared function with 04a): primary for heatmaps;
  - WSI-enhanced: an ablation, used for the UI heatmap only if it wins on calibration;
  - the fine-scale binary carcinoma probe on a 224-native encoder's 256-px embeddings for margin segments
    (WSI-free; the WSI variant is an ablation).
- **`05c` + `finetune.sbatch` (optional):** LoRA (peft, `ft` venv) on the best image encoder, train split only;
  single pre-registered calibration-split comparison (+2 BAcc points, no NLL regression); re-extracts features. The
  test result is always a separate row.
- **`06`:** OOD = mean over tiles of mean cosine distance to the k=10 nearest **TTA-averaged** train tiles →
  percentile vs cal clean. A 3-feature combiner fit on cal (clean + degraded) → P(error); flag threshold from
  `target_error_recall`. No coverage claims on degraded sets.
- **`07`:** top-5 train neighbors, neighbor agreement vs *predicted* class, contrast case, `differs_most_in`
  (top-2 |Δz|).

### Task 11: Case assembly + evaluation suite (Wave 2)
**Owns:** `pipeline/08_build_case_json.py`, `eval/metrics.py`, `eval/deletion_test.py`, `eval/queue_sim.py`,
`eval/margin_eval.py`, `eval/benchmark_thunder.py`, `eval/figures.py`, `eval/results_md.py`,
`tests/test_build_case_json.py`, `tests/test_eval.py`.
- **`08`:**
  - one JSON per test case + degraded and composite surgical variants (`<id>_FSIM_<type><sev>`, `<id>_COMP_<k>`);
  - findings F1…; flag via `core.tiers`; margin via `core.margin`; agent drafts and summaries (Task 5, cached);
  - seeded Poisson arrival offsets with auto-picked hero cases placed late;
  - `eval_only`; validate with `core.contract`; copy display images (test + train refs) and cell overlays.
- **Eval** (test split only; cluster-bootstrap CIs over patient/pseudo-patient groups everywhere; degraded and
  composite variants never enter headline metrics):
  - Invasive-vs-rest AUROC A vs B; macro AUROC/F1, BAcc; confusion + InSitu↔Invasive counts;
  - ECE/Brier/NLL pre/post temperature; reliability diagram; coverage + set size (marginal, per class, APS,
    Mondrian);
  - THUNDER comparison table;
  - deletion test on the **production model and Model A**, not on B, whose top-3 aggregate is defined by the
    deleted tiles: concept-ranked vs risk-ranked vs random (×20) top-3 tile removal, re-aggregated in-distribution;
    paired per-case bootstrap of the risk drop;
  - queue sim: Poisson day, fixed read time, single reader, 1000 reps; FIFO vs tiers vs tiers+aging guard; wait per
    class, P90 tail, delay to non-flagged and false negatives; assumptions in JSON;
  - margin eval: benign FPR, composite exact-negative clear rate, echo statistics, WSI-window sensitivity and
    specificity;
  - PNG figures for the deck; `docs/RESULTS.md` generator.

### Task 12: Integration + end-to-end (Wave 3)
**Owns:** any file, minimal changes, each listed in the report; plus `tests/e2e/`, `README.md`, `docs/`.
- Make `make setup test synthetic analyze demo` pass end to end on `data_synthetic/` and fix seams between tasks.
- Playwright e2e covering the plan §14 demo script, the composite margin case, the Cells tab and Spot-check. It
  asserts no `eval_only`/true labels in the DOM or network responses, and runs at 390 px and desktop widths.
- README: run instructions, §A design notes, limitations, Credits.

### Task 13: Real-data results + deliverable assets (after §E real run)
**Owns:** `docs/RESULTS.md`, `docs/screenshots/`, `docs/demo.webm`, `docs/deck_facts.md`.
- From real `artifacts/eval/*`: figures, RESULTS.md (every number with CI + source file), Playwright screenshots of
  every view, a ≤2-min demo video, and a deck-facts sheet (aims, methods, results, limitations, citations) for the
  ≤20-slide deck.

### Task 99 — end of task list (marker so the brief extractor stops here; not a task)

## E. Controller runbook — Delta via Open OnDemand in Aside (no human steps except Duo approvals)

The user does nothing except approve Duo pushes on their phone. Delta disables SSH keys (password + Duo always),
and no SSH master is live, so the Mac's `scripts/delta.sh` can't authenticate unattended. **All Delta work goes
through Open OnDemand (OOD) in the Aside browser**, driven only by me (never by subagents):
- **Login:** `aside exec` opens `https://openondemand.delta.ncsa.illinois.edu` and signs in with the browser's
  saved NCSA credentials (Aside never echoes them). Before each Duo push I send a `PushNotification` ("Duo push
  coming — please approve"). Re-login happens only when the OOD session expires.
- **OOD bridge** (built and verified by me first, ~30 min; `scripts/ood/` via `aside repl`, after reading
  `aside guide repl`):
  - `ood_run`: types a command into the OOD "Delta Shell Access" terminal (login node 01/04), wraps it with unique
    start/end sentinels plus an exit code, and returns the text between them;
  - `ood_put` / `ood_get`: upload a code tarball and download export archives through the OOD Files app, using the
    page's authenticated session.

  Large outputs go to files and are fetched, never scraped from the screen. **Fallbacks** if terminal automation
  proves brittle:
  1. `aside exec` drives the same OOD shell agentically (slower, same guardrails);
  2. OOD's Job Composer submits the reviewed `.sbatch` files, with the Files app for transfers.

  If OOD itself is down, `#compute-issues` status is checked via Aside (read-only) and work continues on synthetic
  data until it returns.
- **Login-node work** is limited to light tasks (listings, pip installs, weight downloads, tar/untar, `sbatch`,
  `squeue`, `sacct`). Everything heavy runs in jobs.
- **Job submission** (user-authorized override of plan.md §16 "humans submit", logged as a ruling):
  - only via `pipeline/jobs/submit.sh <job> [--after ID]`, which runs `sbatch --test-only` first;
  - it refuses any job that crosses Sun 07:59/08:00 or 18:00, any GPU job while another of ours is running or
    pending outside a dependency chain, and anything that would push total requested GPU time past **7 h** of
    Rishabh's 10;
  - it adds `--deadline` for the overnight block and logs every submission to `logs/submissions.log` + the ledger;
  - there are no resubmission loops: a failed job is diagnosed from its log before any resubmit.
- **Human-review steps become AI reviews, labeled as such:** cell-overlay spike check (CELLS_SPEC §2/§8.1), concept
  top-tile spot check (plan §5.4), optional margin-strip review. I view the images with the Read tool and record
  results as "reviewed by Claude (AI), not a pathologist". The Spot-check UI stays available for a real reviewer later.
- **Browser actions taken in the user's name** (approving this plan authorizes them):
  - request access to UNI2-h, Virchow2 and H-optimus-0 on Hugging Face (automatic gates; accepting their
    non-commercial license terms truthfully as Rishabh/UIUC, hackathon research use);
  - create a read-scoped HF token and paste it into `hf auth login` inside the OOD terminal. Aside is instructed
    never to output the token value, so it never enters this transcript or any file except the private
    `~/.cache/huggingface/token` (chmod 600).
- **DeltaAI:** no live session and the same password + Duo barrier, so it isn't used. If its CLI session ever
  exists, it is read-only: no `pip`/venv from DeltaAI, since it is aarch64 and could put ARM wheels in the shared
  home.

**Sequence** (each step's output is checked before the next):
1. Build and verify the OOD bridge. Log in (Duo). Run `hostname; accounts; quota; squeue -u rbansal3`; read
   `~/.bashrc` for `HF_HOME` (if it points at `/work/nvme`, check token placement and `chmod 700`); `mkdir -p` the
   team dir; fetch the 15 KB patient xlsx.
2. HF access requests + token via Aside (as above).
3. Upload the code (after Tasks 1–3 + 5 merge). Login-node installs: organizer-verified `pip install --user`
   (open_clip_torch, CONCH, MUSK, fairscale, sentencepiece). Separate `--system-site-packages` venvs on NVMe with a
   constraints file pinning the module's torch/numpy/timm/transformers: `enc` (newer timm only if an encoder needs
   it), `wsi` (openslide-python + openslide-bin), `cells` (TIAToolbox 2.1.3), `ft` (peft). Pre-download all weights
   into the home HF cache. Run `00_inventory.py`.
4. Submit the **smoke job immediately** on NCT tiles + whatever `pathmmu/BACH` images exist (it doesn't wait for the
   official download). Submit `prep_a` (CPU). If compute nodes are offline, its download step exits at once and I run
   `nice wget` on login node 04 instead.
5. Check the smoke log (versions, norms, dims, align==adapter, NCT zero-shot) + the hosted 5-tile cosine check.
   After `prep_a` passes the inventory gate, submit `prep_b` (CPU) and the GPU chain `embed_core → embed_extra →
   embed_aux` (`afterok`; `embed_aux` also after `prep_b`). Then submit `cells_spike` (5 arbitrary tiles,
   validates the install) and, once its overlays pass AI review, `cells` (04a runs *inside* the job from the on-disk
   microscopy embeddings, so there's no laptop round trip).
6. Pull TTA-reduced exports (`ood_get`). Laptop analytics; then `finetune` only if every core item is done and it
   fits before Sun 09:00.
7. **Freeze models by Sun 10:00.** Then case JSONs → LLM summaries (cache keyed by a hash of the findings; a
   mismatch falls back to the template) → Task 13 assets before Sun 18:00.

---

## F. Verified facts the build depends on (sources gathered this session)
- **CONCH:** 448-px bicubic resize + center crop; `encode_image(proj_contrast=True, normalize=True)` → 512-d normalized;
  `proj_contrast=False` → pre-projection features recommended for linear probes; `encode_text` normalizes by default;
  128-token context; zero-shot ensembles average normalized prompt embeddings over synonyms × 22 templates.
  The hosted `/classify` softmax omits `logit_scale`, so its scores are ≈1/C and never used.
- **MUSK:** 384 px, Inception mean/std, `with_head+out_norm` 1024-d for text-aligned work, `ms_aug=True` 2048-d
  for probes, 100-token XLM-R text, fp16 on GPU. Benchmark templates: "a histopathology slide showing {c}",
  "histopathology image of {c}", "pathology tissue showing {c}", "presence of {c} tissue on image".
- **BACH:** 400 TIFFs (100/class), 2048×1536, 0.42 µm/px, H&E, Leica DM2000; partial patient IDs in
  `ICIAR2018_BACH_dataset_micro_patient.xlsx`; 30 train WSIs (.svs, 0.467 µm/px; 10 with XML annotations; patient
  IDs for all); test labels never released, so we use our own patient-safe split. THUNDER's split is 218/50/132.
  Published linear probes: CONCH F1 84.4, UNI2-h 84.7, MUSK 69.9; zero-shot CONCH 56.1, MUSK 62.5.
- **Delta:** compute nodes have no internet (pre-download weights, `HF_HUB_OFFLINE=1`); hosted server cap is 32
  images/request; `pytorch-conda/2.8` includes torch, timm, transformers, huggingface_hub, sklearn, pandas,
  fastapi; `unset PYTHONNOUSERSITE` after every module load.
- **Cells (CELLS_SPEC §2 refined by research):**
  1. **TIAToolbox 2.1.3** `hovernet_fast-pannuke` (same HoVer-Net PanNuke model). Torch ≥2.5 and numpy ≥2
     compatible; public HF weights `TIACentre/TIAToolbox_pretrained_weights/hovernet_fast-pannuke.pth`, passed as a
     file path offline. Set `rcParam["torch_compile_mode"]="disable"` on A100s; pre-resize tiles to 0.25 mpp
     (512→~860 px); zarr output per tile.
  2. **vqdang/hover_net** fallback: delete the `imgaug` import in `run_utils/utils.py` (NumPy 2 break), weights from
     Google Drive via gdown, `--model_mode=fast --nr_types=6`; `type_prob` is a vote fraction, not a softmax.
  3. HoVer-NeXt (own environment).
  4. CellViT is skipped: WSI-only inputs, numpy<2 pins.

  All in a venv `--system-site-packages` on NVMe. PanNuke-trained weights are CC BY-NC-SA 4.0 (non-commercial use
  OK, cite).
- **Clinical wording** (CAP/ASCO): myoepithelial panel ≥2 markers including a nuclear one (p63 + SMMHC/calponin), with
  pitfalls (patchy p63, vessels, sclerosing lesions); ER Low Positive 1–10%; HER2 0/1+/2+/3+ with reflex ISH at 2+
  ("HER2-low" is not an ASCO/CAP category); Ki-67 IKWG (≤5% or ≥30% prognostic in ER+/HER2− T1–2 N0–1); Nottingham
  pleomorphism descriptors from CAP protocol 4.11; positive margin = ink on invasive carcinoma or DCIS;
  frozen-section TAT benchmark 20 min.
- **Worklist evidence for the deck:** reordering the worklist (not notification widgets) cut ICH wait 15.75→12.01 min
  (O'Neill 2021). Simulated CXR triage cut critical delays, but non-flagged waits grew (Annarumma 2019). We report
  the cost to non-flagged cases too.

## G. GPU schedule & budget (one GPU per team; controller submits via guardrailed `submit.sh`)
| Job | Account/partition | Limit | Contents | Est. use |
|---|---|---|---|---|
| prep_a | bfqi-delta-cpu / cpu | 1 h | download + unzip BACH, inventory, `01` microscopy splits/tiles/displays → core manifest | CPU only |
| prep_b | bfqi-delta-cpu / cpu | 1 h | composites, WSI tiles + margin windows (`wsi` venv), Quilt/PathGen caption mining → aux manifest | CPU only |
| smoke | gpu / gpuA100x4 | 20 min | `--limit 5` every encoder on NCT + any pathmmu BACH; version/norm/dim/equivalence/zero-shot checks | 0.3 GPU-h |
| embed_core | gpu | 1 h | CONCH + MUSK, one forward pass feeding both heads, fp16 autocast: 512×8 views (train) / TTA-mean (cal, test), NCT, degraded cal+test, prompts | ~0.8 |
| embed_extra | gpu | 1 h | UNI2-h, H-optimus-0 (priority), then Virchow2, Phikon-v2, PathGen-CLIP: 256×4 views + 512×1; reduces over TTA on node | ~0.9 |
| embed_aux | gpu | 1 h | WSI tiles, margin windows, composite segments, Quilt/PathGen concept images | ~0.5 |
| cells_spike → cells | gpu | 20 min → 1 h | TIAToolbox HoVer-Net: 5 arbitrary tiles → 04a (in-job) + ~1,200 tiles (+ margin strips, stretch) | ~1.1 |
| finetune (optional) | gpu | 1 h | LoRA on the best image encoder (train split only), re-extract features | ~1.0 |

Total ≈4.6 GPU-h used, under `submit.sh`'s 7 h cap. All jobs run under Rishabh's account and budget (the CONCH/MUSK/
encoder licenses are personal). GPU jobs chain with `--dependency=afterok:<prev>`; overnight jobs carry
`--deadline=2026-10-18T07:59`; nothing crosses 08:00 or 18:00. Hosted servers: equivalence check only.

**Target timeline (CDT):**
- Sat 15:00–18:00: Task 1, then Wave 1; OOD bridge + login; HF access + token; installs; smoke + `prep_a`.
- Sat 18:00–23:00: `prep_b` + embed chain; Wave 1 merges; Wave 2 on synthetic data.
- Sat 23:00–Sun 03:00: pull exports; real analytics; cells spike → cells; Task 12 integration.
- Sun 08:00–10:00: reruns buffer, optional finetune; **models frozen at 10:00** (cells hard stop at 10:00 too).
- Sun 10:00–14:00: case JSONs, LLM summaries, Task 13 assets.
- Sun 14:00–15:30: final review + fixes; demo freeze 15:30.

## H. Verification (each phase ends green before the next)
- `make test`: pytest covering
  - tiers (incl. the conformal safety rule and empty sets);
  - conformal (finite-sample k, Mondrian, empirical coverage on synthetic);
  - calibration (T reduces NLL; argmax unchanged);
  - bootstrap CIs; summary validator (uncited / unknown-ID / number mismatch); rules engine (each rule fires
    exactly when specified, never invents stains);
  - margin (strip geometry, corner sharing, benign-FPR threshold); composites truth table; degrade determinism;
    cell features (min counts, µm units, percentiles);
  - contract (fixture + every artifact validates; `eval_only` never in API output); API flows (worklist sort,
    decisions, routing, timers, sim controls);
  - `02_embed.py` end-to-end locally with the official `test` adapter on synthetic tiles (chunking, resume, alignment).
- `make synthetic && make analyze DATA=data_synthetic && make demo`: the full app runs before Delta returns.
- Delta checks: inventory gate; `--limit 5` outputs inspected before full runs; in-job assertions (versions, norms,
  dims, align==adapter, NCT zero-shot ≈ organizer numbers); hosted-vs-job cosine ≈1 on 5 tiles; cell overlays
  AI-reviewed (labeled as such); every submission in `logs/submissions.log` with its guard checks.
- Extra tests from review: patient-disjoint splits; empty set with argmax Invasive stays tier 2; composites and
  degraded variants excluded from headline metrics; no BACH-derived image tracked in git (scan `git ls-files`);
  missing Lumen key ⇒ template path; summary cache invalidates when the findings hash changes.
- Real-data acceptance: every number on the Metrics page traces to `artifacts/eval/*.json` with CIs; the test split is
  used once; selection logs show CV-only choices; ground truth absent from UI DOM (Playwright asserts).
- Frontend: `tsc --noEmit`, `vite build`, Playwright run of the full demo script with screenshots, checked at
  phone and desktop widths, in both themes if supported.

## I. Cut order (if the clock runs short: drop from the top; never drop the bottom list)
1. LoRA fine-tune.
2. Extra encoders beyond UNI2-h + H-optimus-0 (Virchow2, Phikon-v2, PathGen-CLIP).
3. PathGen caption mining, then Quilt concept vectors entirely.
4. ABMIL / attention maps.
5. Dihedral views → TTA only.
6. WSI clean labels and WSI margin windows (composites stay).
7. Full cells run: keep spike overlays as "future work" (hard stop Sun 10:00).
8. LLM summaries: templates only.

**Never cut:** THUNDER patient-safe split; CONCH + MUSK; Models A and B; temperature + conformal; tiers with the
safety rule; margin read + composite evaluation; degraded reliability; template summaries; the end-to-end demo; a
Metrics page with CIs.

## J. After execution
The final report lists every `Ruling:` from the ledger, the cut items (if any), and the AI-reviewed (not
human-reviewed) checks. It also flags for the team the overlay-in-deck licensing question and the teammate-frontend
merge question. Then I save memories:
- the user prefers full autonomy (Duo approvals only);
- strongest-reasoning parallel subagents;
- best results over speed;
- deviations allowed when justified.
