# Sepia

Sepia is Team 44’s research workspace for confidence-aware breast pathology triage, built for the 2026 Ashby Prize Hackathon. It connects precomputed model evidence to a persistent review worklist, drafted follow-up orders, simulated surgical margin calls, and routed results.

**Research prototype. Not for clinical use.** BACH labels are diagnostic categories, not Nottingham grades. “Likely invasive” is a triage suggestion. Frozen sections, image-border margins, arrivals, and operating-room assignments are simulated. Approval records and inbox routing are local to this application; no hospital order system is connected.

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

The source plans are [plan.md](plan.md), [CELLS_SPEC.md](CELLS_SPEC.md), and the refinements in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md). The implementation uses the existing teammate’s vanilla JavaScript frontend, with shared view adapters and server-authoritative state; no frontend framework migration is required to run it.

Delta runs data preparation, model embeddings, prompt encoding, and cell segmentation. The laptop runs downstream analytics and serves cached case artifacts, so reviewing a case needs no live GPU or LLM call. Use Delta, not DeltaAI, for event GPU work. Never run model inference on a login node. Job submissions use the guarded `pipeline/jobs/submit.sh`, respecting one GPU per team, the recorded budget, and event windows.

Exports use a manifest with per-file sizes and SHA-256 checksums:

```sh
# After downloading a pipeline.export bundle from Delta:
uv run python -m pipeline.export --unpack /absolute/path/export.tgz --dest data
# Once all required feature outputs are present:
make analyze DATA=data ART=artifacts
```

Tile QC on the laptop: `uv run python -m pipeline.01q_requalify_qc` recomputes tissue fractions from the display
JPEGs with the corrected mask (the original value cut rejected pale tissue; see [docs/OVERNIGHT.md](docs/OVERNIGHT.md)).
Worklist tiers use the APS conformal set with the argmax added to empty sets (`conformal.method`/`nonempty` in
`config/settings.yaml`); pre-registered LAC is still reported, and the switch is disclosed as post-hoc. Grounded
summaries are generated once with `08_build_case_json --max-llm-calls N` (Lumen, cached) and checked against the
findings for citations, numbers and categorical facts (margin call, reliability flag, feature level).

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

Results with confidence intervals: [docs/RESULTS.md](docs/RESULTS.md). What changed overnight on 2026-10-18 and the
measured effect: [docs/OVERNIGHT.md](docs/OVERNIGHT.md).

## Credits and limits

- BACH: Aresta et al., *BACH: Grand challenge on breast cancer histology images*, Medical Image Analysis (2019), ICIAR 2018 dataset; CC BY-NC-ND 4.0. Cite the original release in presentations.
- CONCH: Mahmood Lab; MUSK: Stanford Li Lab. Model checkpoints remain subject to their respective licenses and individual gated access.
- UNI2-h (Mahmood Lab) is the production prediction encoder. H-optimus-0 was a model-selection candidate and PathGen-CLIP a zero-shot comparison; Virchow2 and Phikon-v2 were embedded on Delta but are not among the pre-registered candidates or any reported model. The selected prediction model is recorded in each case and in the model-choice evaluation.
- [THUNDER](https://mics-lab.github.io/thunder/leaderboards/): split/benchmark reference; published comparison values were checked against the authors’ leaderboard. See [verification](docs/research/benchmark-verification.md).
- NCT-CRC-HE-100K provides cross-organ concept sanity checks; it is not breast validation.
- PathMMU: the shared `pathmmu/BACH` folder was inventoried, but it is the 2015 Bioimaging breast set, not ICIAR 2018 BACH, so the official BACH release is the dataset of record.
- Quilt and PathGen provide caption-based concept candidates when their actual feature outputs are available. Caption consistency is not pathologist validation.
- HoVer-Net / PanNuke and TIAToolbox support the optional cell module; model and weight licensing is recorded in configuration.
- Summaries: `gemma-4-31b-it` via NCSA Lumen, generated offline from structured findings only and validated.
- AI spot check of concept tiles and cell overlays: Claude (Anthropic), recorded as AI review, not a pathologist.
- Conformal prediction: APS (Romano, Sesia & Candès, NeurIPS 2020) and LAC (Sadinle, Lei & Wasserman, JASA 2019).
- NCSA Delta, Lumen, and the MIT-licensed `ncsa/ashby-hackathon-2026` tooling under `resources/official/`.
- Original teammate worklist by adas286, extended with the API adapter, workflow views, and cohesive Sepia workspace.

This is a small-dataset research experiment. Conformal coverage depends on exchangeability; degraded and simulated composite fields do not inherit a clean-test coverage guarantee. Explanations describe model associations. Follow-up rules require clinical review. No clinical validation or external hospital integration is claimed.
