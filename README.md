# Sepia

Sepia is a research prototype of a breast-pathology review worklist that orders biopsy images by a calibrated foundation-model read and shows the evidence behind each read. It connects precomputed model evidence to a persistent review worklist, drafted follow-up orders, simulated surgical margin calls, and routed results. Built by Team 44 for the 2026 Ashby Prize in Computational Science Hackathon.

- **Live application:** https://sepia-ashby.xyz (also https://sepia-ashby.vercel.app)
- **Run locally** (Python 3.13 and [uv](https://docs.astral.sh/uv/)): `uv sync`, then `make demo`, then open http://127.0.0.1:8000 ([details](#run-the-application))
- **Documentation:** [docs/README.md](docs/README.md) groups every document by purpose

**Research prototype. Not for clinical use.**

- **Real:** 400 public BACH microscopy images (ICIAR 2018), each a single 2048 x 1536 field, not a whole-slide image. Model reads, calibrated probabilities, confidence sets, region heatmaps and accuracy results were computed from these images. The external test uses real TCGA-BRCA slides, including frozen sections.
- **Simulated:** case arrivals and the reading-day queue, frozen sections in the worklist (degraded microscopy images), surgical margins (image borders stand in for inked margins), and operating-room assignments.
- **Not connected:** drafted orders, approvals and the oncology inbox are stored in Sepia's own database; no hospital or laboratory system is connected. BACH labels are diagnostic categories, not Nottingham grades, and “likely invasive” is a triage suggestion.

## Headline results

Production model `uni2h512_lr`: a linear probe on UNI2-h embeddings of 512-px regions (encoder not fine-tuned), with temperature scaling and an APS conformal set. Held-out THUNDER test split of 132 BACH images. Brackets are 95% bootstrap intervals over patient groups; a 100% result shows the exact Clopper–Pearson interval.

| Measure | Result |
|---|---|
| Macro-F1, four classes | 0.860 [0.678, 0.928] |
| Accuracy | 86.4% (114 of 132) [76.0, 94.4] |
| Invasive images called invasive | 31 of 31 (exact 95% CI 88.8–100%) |
| Non-invasive images not called invasive | 97.0% (98 of 101) [93.6, 100] |
| External TCGA-BRCA: FFPE tumor slides called invasive | 95.7% (22 of 23) [87.0, 100.0] |
| External TCGA-BRCA: frozen tumor slides called invasive | 84.0% (21 of 25) [68.0, 96.0] |

The TCGA-BRCA test (25 patients, 65 slides) ran the unchanged production model under a protocol written before the run ([docs/EXTERNAL_TCGA_R2.md](docs/EXTERNAL_TCGA_R2.md)); TCGA has no benign or in situ labels. With 24 test patient groups the BACH intervals are wide. Every metric, the confusion matrix and every misread: [docs/FACT_SHEET.md](docs/FACT_SHEET.md) and [docs/RESULTS.md](docs/RESULTS.md), both generated from `artifacts/eval/`.

## Evidence and reviewer documents

The full index, including validation rounds, deployment and superseded logs, is [docs/README.md](docs/README.md).

| Document | Contents |
|---|---|
| [docs/FACT_SHEET.md](docs/FACT_SHEET.md) | Data units, splits, test metrics with AUROC conventions, every test misread, dataset roles, reference distributions, which parts are pretrained or trained (generated) |
| [docs/RESULTS.md](docs/RESULTS.md) | Every evaluation with 95% patient-grouped intervals (generated) |
| [docs/HUMAN_BASELINE.md](docs/HUMAN_BASELINE.md) | Published pathologist reader studies and clinical turnaround benchmarks, with their limits as comparisons |
| [docs/EXTERNAL_TCGA_R2.md](docs/EXTERNAL_TCGA_R2.md) | Pre-registered external test of the production model on TCGA-BRCA |
| [docs/SCOPE_MATRIX.md](docs/SCOPE_MATRIX.md) | What is evaluated, simulated, app-only or proposed |
| [docs/ATTENTION_COMPARISON.md](docs/ATTENTION_COMPARISON.md) | Attention-based MIL vs the production linear probe on identical inputs (generated) |
| [docs/WSI_AND_EXTENSION.md](docs/WSI_AND_EXTENSION.md) | Whole-slide path, other organs, reader-study and frozen-section study designs |
| [docs/JUDGE_QA.md](docs/JUDGE_QA.md) | Answers to the judges' questions with sources |

## Run the application

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
make demo
```

Open `http://127.0.0.1:8000`. The default data source is `artifacts/cases/`. The app does not silently fall back to synthetic data. Start the arrival replay to introduce the precomputed cases; replay controls change arrivals, not model predictions.

If the case folder is missing, build or import the real pipeline outputs first. You can point the app at another completed artifact root:

```sh
make demo ART=/absolute/path/to/artifacts
```

For isolated **synthetic workflow development**, generate an explicitly labeled separate workspace:

```sh
uv run python -m backend.dev_cases --out artifacts/qa
make demo ART=artifacts/qa PORT=8001
```

`?mock=1` is an explicit in-memory fixture mode. It is labeled synthetic and does not exercise backend persistence. The normal application uses FastAPI and SQLite.

## Review flow

1. Start the arrival replay. The queue sorts surgical cases first, followed by likely invasive, low confidence, and unflagged cases. Invasive anywhere in the conformal set receives tier 2. Search by case ID or flag; filter by lane and review status.
2. Open a biopsy. Inspect the image heatmap, cited summary, contributions, calibrated probabilities, conformal set, and reference cases. Contribution buttons highlight their supporting regions. Cell findings open measured evidence when segmentation outputs exist.
3. Approve or edit drafted orders, defer for a closer look, or dismiss the flag. Sign-out becomes available only after the order decision. Signed-out results appear in the oncology inbox.
4. Introduce a simulated frozen section. Choose clear margins, named involved margins, or defer to permanent sections. The recorded call stops the timer and appears on the operating-room board.
5. Inspect actual evaluation outputs and record human or AI spot checks. Missing evaluations and cell measurements remain unavailable rather than becoming placeholder results.

Decisions, order edits, routed results, and events persist in `artifacts/pathqueue.sqlite`. **Reset** clears the current replay and its decisions; use it only to start a fresh demonstration. Browser refresh alone preserves recorded work.

## Data and compute

The source plans are [plan.md](plan.md), [CELLS_SPEC.md](CELLS_SPEC.md), and the refinements in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md), all written before the build. PathQueue was the working name; it remains in the `PATHQUEUE_*` environment variables and `artifacts/pathqueue.sqlite`. The implementation uses the existing teammate’s vanilla JavaScript frontend, with shared view adapters and server-authoritative state; no frontend framework migration is required to run it.

Delta runs data preparation, model embeddings, prompt encoding, and cell segmentation. The laptop runs downstream analytics and serves cached case artifacts, so reviewing a case needs no live GPU or LLM call. Use Delta, not DeltaAI, for event GPU work. Never run model inference on a login node. Job submissions use the guarded `pipeline/jobs/submit.sh`, respecting one GPU per team, the recorded budget, and event windows.

Exports use a manifest with per-file sizes and SHA-256 checksums:

```sh
# After downloading a pipeline.export bundle from Delta:
uv run python -m pipeline.export --unpack /absolute/path/export.tgz --dest data
# Once all required feature outputs are present:
make analyze DATA=data ART=artifacts
```

Tissue QC on the laptop: `uv run python -m pipeline.01q_requalify_qc` recomputes the tissue fraction of each
sub-image tile from the display JPEGs with the corrected mask (the original value cut rejected pale tissue; dated
log in [docs/OVERNIGHT.md](docs/OVERNIGHT.md)).
Worklist tiers use the APS conformal set with the argmax added to empty sets (`conformal.method`/`nonempty` in
`config/settings.yaml`); pre-registered LAC is still reported, and the switch is disclosed as post-hoc. The shipped
case summaries are written by a deterministic template from structured findings. An optional language-model path
through NCSA Lumen (`08_build_case_json --max-llm-calls N`, cached) checks any generated summary against the
findings for citations, numbers and categorical facts (margin call, reliability flag, feature level); it was not
used for the shipped case files.

Analytics steps can report missing candidates, skip unavailable optional modules, or fail on required inputs. Read their logs and metadata before calling the experiment complete. Do not use `data_synthetic` to produce real-data results. Limit smoke runs with `--limit 5`; rerun complete steps with `--force` after inputs change.

The app’s `/api/health` reports validated case counts, rejected files, prediction model IDs, and missing referenced media. A connected service alone does not establish model quality. The sidebar distinguishes empty, synthetic, incomplete, and populated artifact states.

Credentials live in private configuration outside Git. See `.env.example` for variable names. Public BACH-derived assets and licensed model weights are kept in ignored local data/artifact directories; do not commit or publish these bundles.

## Verify

```sh
make test                  # Python/unit/integration suite, no network
make types-check           # contract → TypeScript consistency
uv run playwright install chromium
make e2e                   # desktop/light and phone/dark workflow tests
```

Browser tests start their own temporary backend using labeled synthetic cases. They verify the workflow and persistence boundaries, not model accuracy. Screenshots are generated in the ignored `docs/screenshots/` directory. The production artifact root is never reset by those tests. API contracts and endpoints are documented in [docs/API.md](docs/API.md).

Results with confidence intervals: [docs/RESULTS.md](docs/RESULTS.md). The overnight changes on 2026-10-18 are logged
in [docs/OVERNIGHT.md](docs/OVERNIGHT.md); its numbers describe the earlier 256-px model and are superseded.

## Credits and limits

- BACH: Aresta et al., *BACH: Grand challenge on breast cancer histology images*, Medical Image Analysis (2019), ICIAR 2018 dataset; CC BY-NC-ND 4.0. Cite the original release in presentations.
- CONCH: Mahmood Lab; MUSK: Stanford Li Lab. Model checkpoints remain subject to their respective licenses and individual gated access.
- UNI2-h (Mahmood Lab) is the production prediction encoder; since accuracy round 2 ([docs/ACCURACY_ROUND2.md](docs/ACCURACY_ROUND2.md)) the production read is a linear probe on its embeddings of 512-px regions (`uni2h512_lr`). H-optimus-0 was a model-selection candidate and PathGen-CLIP a zero-shot comparison. Virchow2 and Phikon-v2 were embedded on Delta but are not production candidates (Virchow2's terms exclude clinical-support use). LoRA fine-tunes of UNI2-h (`pipeline/05c_finetune.py`, weights kept private under CC BY-NC-ND) were judged by the same cross-validation rule; they beat the frozen probe on cross-validation but not on the test split, so they are not the production model and no reported result comes from them. They appear only as a labeled comparison in [docs/ACCURACY_ROUND2.md](docs/ACCURACY_ROUND2.md). The selected prediction model is recorded in each case and in the model-choice evaluation.
- [THUNDER](https://mics-lab.github.io/thunder/leaderboards/): split/benchmark reference; published comparison values were checked against the authors’ leaderboard. See [verification](docs/research/benchmark-verification.md).
- NCT-CRC-HE-100K provides cross-organ concept sanity checks; it is not breast validation.
- PathMMU: the shared `pathmmu/BACH` folder was inventoried, but it is the 2015 Bioimaging breast set, not ICIAR 2018 BACH, so the official BACH release is the dataset of record.
- Quilt and PathGen provide caption-based concept candidates when their actual feature outputs are available. Caption consistency is not pathologist validation.
- HoVer-Net / PanNuke and TIAToolbox support the optional cell module; model and weight licensing is recorded in configuration.
- Case summaries: all 328 shipped case files use a deterministic template that restates structured findings. The optional "In plain language" tab holds cached explanations generated through Cursor (Gemini 4 for 330 cases, Claude for 2), with every number checked against the evidence; it is not evidence. An optional summary path through `gemma-4-31b-it` on NCSA Lumen uses the same checks but produced none of the shipped text. Counts: [docs/FACT_SHEET.md](docs/FACT_SHEET.md).
- AI spot check of concept regions and cell overlays: Claude (Anthropic), recorded as AI review, not a pathologist.
- Conformal prediction: APS (Romano, Sesia & Candès, NeurIPS 2020) and LAC (Sadinle, Lei & Wasserman, JASA 2019).
- NCSA Delta, Lumen, and the MIT-licensed `ncsa/ashby-hackathon-2026` tooling under `resources/official/`.
- Original teammate worklist by adas286, extended with the API adapter, workflow views, and cohesive Sepia workspace.

This is a small-dataset research experiment. Conformal coverage depends on exchangeability; degraded and simulated composite fields do not inherit a clean-test coverage guarantee. Explanations describe model associations. Follow-up rules require clinical review. No clinical validation or external hospital integration is claimed.
