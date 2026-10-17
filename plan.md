# PathQueue — Build Plan

An AI-prioritized pathology worklist with explainable, confidence-aware case reads.
Built for the 2026 Ashby Prize Hackathon (NCSA Delta, Sat–Sun Oct 17–18, 2026).

**Scope: breast cancer, using the BACH dataset.** We validate the workflow on one organ and
build it so another organ can be added by swapping configuration (Section 17).

This file is the source of truth for Claude Code. Read it fully before writing code.
Section 2 lists hard constraints from the event's compute setup; never violate them.

---

## 1. What we are building

A pathologist opens a worklist. Cases arrive from two lanes:

- **Surgical lane (frozen sections from the OR).** Always top priority, with a running timer.
  The model reads the sample, says whether tumor appears at a margin, and reports how reliable
  its read is. The pathologist makes the call, and the result goes back to the surgeon as one of
  three answers: **margins clear**, **tumor at a named margin (more tissue needed)**, or
  **cannot determine, defer to permanent sections**.
- **Biopsy lane.** The model estimates risk, explains why using named histologic criteria,
  and attaches a calibrated confidence. Cases are flagged if **likely aggressive** or
  **low confidence**. For flagged cases, an **agent drafts follow-up orders** (stains, recuts,
  new-sample requests) before the pathologist opens the case. **Nothing is ordered until the
  pathologist approves.** The pathologist can order the drafted tests, take a closer look first,
  or dismiss the flag. The signed-out result is routed to the oncologist.

### Why breast, and what "aggressive" means here

- **Why one organ:** grading systems are organ-specific (Nottingham for breast, Gleason for
  prostate, WHO/ISUP for kidney, and so on), and the shared data on Delta has labeled cases for
  very few cancers. A validated single-organ system is a stronger result than an unvalidated
  pan-cancer one.
- **Why breast and BACH:** BACH's four classes (**Normal, Benign, InSitu, Invasive**) map directly
  onto triage, and in situ vs invasive is a real, clinically meaningful ambiguity. The Nottingham
  grading criteria (tubule formation, nuclear pleomorphism, mitotic count) give a natural set of
  named criteria for explanations, and the follow-up stains (ER, PR, HER2, Ki-67; p63/calponin for
  in situ vs invasive) are standard.
- **Label semantics — state this honestly everywhere:** BACH labels are **diagnostic categories,
  not Nottingham grades.** In this project, **"likely aggressive" means "likely invasive
  carcinoma."** In situ carcinoma is the main differential. The Nottingham-style criteria are used
  as **explanatory features**, not as validated grade components; we do not predict Nottingham grade.

### Priority tiers (worklist sort order)

1. Surgical frozen sections (timer running).
2. Likely aggressive (likely **Invasive**). **Safety rule:** if `Invasive` is anywhere in the
   case's conformal prediction set, the case is tier 2 even if it is not the top prediction.
3. Low confidence (conformal set has more than one class, or the reliability flag is raised).
4. Unflagged, in arrival order.

Within a tier: sort by risk score (descending), then arrival time.

### Explainability requirements (from judge feedback)

Every conclusion must show:
- **Why:** contributions of named criteria (e.g. nuclear pleomorphism, invasion into stroma,
  gland formation, lymphocytic infiltrate, necrosis), each linked to the image regions that drove it.
- **Where:** tile-level heatmaps per criterion.
- **Compared to what:** similar reference cases, including the nearest case with a different label.
- **How sure:** calibrated probability, a conformal prediction set, and agreement signals
  (CONCH vs MUSK, nearest-neighbor label agreement, prompt-ensemble spread).

The natural-language summary is written by an LLM **only from structured findings**; it never
sees the image. Every sentence must cite finding IDs, or it is dropped.

---

## 2. Hard constraints (event compute rules)

Source: `ncsa/ashby-hackathon-2026` (`HACKATHON_COMPUTE.md`, `STUDENT_GUIDE.md`, `README.md`).
Clone it next to this project; reuse its `client.py`, `models.py`, `run_embeddings.py`,
`submit.sbatch`, and `train_probe.py` rather than rewriting them.

**Cluster and jobs**
- Cluster: NCSA Delta. Accounts: `bfqi-delta-gpu` (GPU), `bfqi-delta-cpu` (CPU).
- **Never run model inference on the login node** (`serve.py`, `run_embeddings.py`, or anything
  that loads CONCH/MUSK). The process gets silently killed. Use `srun` or `sbatch`.
  Lightweight HTTP clients, `pip install`, and file edits are fine on the login node.
- **One GPU per team at a time. Each person has 10 GPU-hours.** Request realistic `--time`.
  Do not exceed 16 CPUs or ~62 GB memory per GPU (billing uses the largest). Interactive
  partitions cost double.
- Reserved A100 nodes (partition `gpuA100x4`, no `--reservation` flag needed) run in two blocks:
  **Sat 07:30 – Sun 07:59** and **Sun 08:00 – 18:00**. A job whose time limit crosses
  Sun 08:00 or Sun 18:00 falls into the regular queue (days). Keep jobs to 1–2 hours.
- **Claude Code must never submit `sbatch`/`srun` jobs on its own.** Write the job script,
  show the command, and let a human run it. Never write auto-resubmitting scripts.
- Compute nodes have no internet. Download/install on the login node.
- Environment setup (login node, once):
  ```bash
  module load pytorch-conda/2.8
  unset PYTHONNOUSERSITE          # required after every module load
  pip install --user "open_clip_torch>=2.24"
  pip install --user git+https://github.com/mahmoodlab/CONCH.git
  pip install --user git+https://github.com/lilab-stanford/MUSK.git fairscale sentencepiece
  ```
  Every sbatch script we write must include those two first lines
  (`module load pytorch-conda/2.8` then `unset PYTHONNOUSERSITE`).

**Hosted model servers**
- Organizers host CONCH, MUSK, QuiltNet-B-32, and PLIP on an A40 node until **Sun 18:00**.
  Endpoints and API keys are posted in Slack #compute-issues. Callable **only from inside Delta**.
- API: `POST /embed {"images":[b64,...]}` returns L2-normalized image embeddings.
  `POST /classify` returns **only the top label and score**, so it cannot give per-criterion
  scores. We therefore embed images via `/embed` and compute text-prompt embeddings ourselves
  (Section 6.3).
- Throughput is ~60 images/s **shared by ~60 teams**. Send batches of 32, retry with backoff,
  and use it for up to a few thousand tiles. For more, run our own GPU job with `submit.sbatch`.
- **Each team member using CONCH or MUSK must have their own approved Hugging Face access**
  (licenses forbid sharing weights). QuiltNet and PLIP are open.
- Embeddings from different models are not comparable; never mix them in one index.

**Storage**
- Code: `/u/<username>` (home, 100 GB) and GitHub.
- Data and outputs: `/work/nvme/bfqi/<username>` and team space
  `/work/nvme/bfqi/teams/<team>` (create with `mkdir -p`). Not backed up.
  **Readable by all participants** — never store keys or anything private there.
- Shared read-only datasets (read in place, never copy wholesale):
  ```
  /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/  # 100k CRC tiles, 9 class folders
  /projects/bfqi/shared/datasets/pathgen_dataset/               # PathGen-1.6M captions + 2k-slide sample
  /projects/bfqi/shared/datasets/pathmmu/                       # ARCH, BACH, MHIST, NCT-CRC, Osteo
  /work/nvme/bfqi/shared/datasets/quilt/                        # Quilt-1M (115 GB) + lookup CSV
  ```
- Avoid `/work/hdd` this weekend.

**LLMs**
- Lumen (NCSA), OpenAI-compatible: base URL `https://lumen.ncsa.illinois.edu/v1`,
  key in env var `LUMEN_API_KEY` (never in code, never committed).
- Coins are limited (start at 10, slow refill). Costs differ ~40× by model. Default to
  `qwen3-coder-next` or `ornith-1.0-35b`; avoid `glm-5.3`. Check `/v1/models` for current IDs.
- **Generate summaries and agent rationales once, offline, and cache them in the case JSON.**
  The live demo must work with zero LLM calls (template fallback).
- Use only public, de-identified data. Cite every model, dataset, and tool in the final deck.

### Consequences for the design (deviations from the pitch deck)

- **No whole-slide images are provided** and TITAN/slide-level models are impractical this
  weekend. We work at **tile level**: each case is one large field-of-view image, cut into a
  grid of tiles, embedded, then aggregated to a case-level read. Heatmaps are per tile.
- **No frozen-section data is provided.** The surgical lane is a **clearly labeled simulation**:
  image borders stand in for inked margins, and frozen-section artifacts are simulated with
  controlled degradations to test the reliability flag. Real TCGA frozen slides are a stretch goal.
- The **demo app runs locally** on a laptop from precomputed artifacts copied off Delta, so it
  does not depend on the hosted servers (which stop Sun 18:00) or on GPU availability.

---

## 3. Architecture

```
            ┌──────────────── Delta (offline pipeline) ────────────────┐
            │                                                           │
 datasets ──► 00 inventory ─► 01 build cases + tiles ─► 02 embed tiles   │
            │                                    (hosted /embed or job) │
            │  03 encode prompts (GPU/CPU job) ──┐                      │
            │                                    ▼                      │
            │  04 concept scores ─► 05 risk model + calibration +       │
            │                       conformal ─► 06 reliability/OOD ─►  │
            │                       07 retrieval ─► 08 case JSON +      │
            │                       summaries (Lumen, cached)           │
            └───────────────────────────┬───────────────────────────────┘
                                        │ scp / OnDemand download
                                        ▼  artifacts/ (JSON + thumbnails)
            ┌──────────────── Laptop (demo app) ───────────────────────┐
            │ simulator ─► FastAPI + SQLite ─► React worklist/case UI  │
            │                 │                                         │
            │                 └─► agent (rules + optional Lumen) ──►    │
            │                     pathologist decision ─► routing        │
            │                     (surgeon screen / oncologist inbox)    │
            └──────────────────────────────────────────────────────────┘
```

---

## 4. Repository layout

```
pathqueue/
  Plan.md
  README.md                    # how to run pipeline + app; credits/citations
  .gitignore                   # artifacts/, data/, *.csv, *.npy, .env, logs/
  .env.example                 # LUMEN_API_KEY=, CONCH_ENDPOINT=, CONCH_KEY=, MUSK_ENDPOINT=, MUSK_KEY=
  config/
    settings.yaml              # organ: breast, dataset: bach, classes, aggressive class, paths, tile size, alpha, thresholds
    concepts.yaml              # criterion prompt bank (positive + negative phrasings)
    stain_rules.yaml           # suspected finding -> drafted orders (illustrative, needs clinical review)
  pipeline/                    # runs on Delta
    common.py                  # config loading, paths, hosted-server client w/ batching+retry+resume
    00_inventory.py            # verifies BACH on disk
    01_build_cases.py
    02_embed_tiles.py
    03_encode_prompts.py
    04_concepts.py
    05_risk_model.py
    06_reliability.py
    07_retrieval.py
    08_build_case_json.py
    jobs/
      encode_prompts.sbatch
      embed_tiles.sbatch       # fallback if hosted server is slow/down
  eval/
    metrics.py                 # AUROC, ECE, reliability diagram, conformal coverage, concept AUROC
    queue_sim.py               # FIFO vs PathQueue waiting-time simulation
    deletion_test.py           # explanation faithfulness
  agent/
    rules.py                   # deterministic rule engine over stain_rules.yaml
    agent.py                   # Lumen tool-calling wrapper (optional) with rules fallback
    lumen.py                   # OpenAI-compatible client, caching, retries
  backend/
    app.py                     # FastAPI
    db.py                      # SQLite: cases, decisions, orders, routed_results
    simulator.py               # replays cases as live arrivals
    summary.py                 # grounded-summary validator + template fallback
  frontend/                    # Vite + React + TypeScript + Tailwind
  artifacts/                   # git-ignored; produced by pipeline, consumed by app
    cases/<case_id>.json
    images/<case_id>.jpg       # downscaled display image
    reference_index.json
    eval/                      # metrics JSON + plots for the Metrics page
  tests/
```

---

## 5. Data plan (BACH)

### 5.1 About BACH

BACH (ICIAR 2018 Grand Challenge on Breast Cancer Histology) contains H&E microscopy images of
breast tissue in four classes: **Normal, Benign, InSitu (carcinoma in situ), Invasive (invasive
carcinoma)**. The microscopy part is commonly described as 400 images (100 per class) of
2048×1536 pixels; the full release also has a small set of annotated whole-slide images.
**Verify all of this against what is actually on disk** (5.2) rather than trusting these numbers.

### 5.2 Phase 0 — verify BACH on Delta (do this first, ~30 min)

`pipeline/00_inventory.py` (login node is fine; it only lists files and reads image headers):
- List `/projects/bfqi/shared/datasets/pathmmu/BACH`: subfolders, file counts, extensions,
  3 example paths per folder.
- Find how labels are stored (class folders, a CSV, or VQA question files). Map them to the four
  canonical names `Normal`, `Benign`, `InSitu`, `Invasive`.
- Open 5 images per class with PIL and record size and magnification metadata if present.
- Check whether any whole-slide images or region annotations are included (useful for 6.9).
- Check for any patient or slide identifiers usable for group splits.
- Write `artifacts/inventory.json` plus a short summary: images per class, image size, label source.

**Decision:**
- **≥ 80 images per class with labels** → proceed.
- **Fewer, or labels only embedded in VQA questions** → parse labels from the question files if
  unambiguous. If still incomplete, a human downloads the official BACH release on the login node
  into `/work/nvme/bfqi/teams/<team>/bach/` (check its license and size first; ask in
  #compute-issues if it exceeds ~200 GB). Claude Code must not start large downloads on its own.
- Do **not** switch organs. The project scope is breast.

### 5.3 Cases and tiles

- **Case = one BACH microscopy image.** Cut into a non-overlapping grid of 512×512 tiles
  (about 4×3 for a 2048×1536 image). Discard tiles that are mostly background
  (simple saturation/brightness threshold — this is our QC step).
- Store tile crops under `/work/nvme/bfqi/teams/<team>/pathqueue/tiles/<case_id>/<r>_<c>.png`.
- Split cases into **train 60% / calibration 20% / test 20%**, stratified by class, fixed seed.
  If patient IDs exist, split by patient. If not, note in the README and on the Metrics page that
  images from the same patient may appear in more than one split, so results may be optimistic.
- Lanes for the demo: test-split cases become simulated arrivals. Assign ~25% to the surgical lane
  and the rest to biopsy. Surgical cases are clearly marked "simulated frozen section" in the UI.
  Make sure the surgical subset includes Invasive, InSitu, and Normal/Benign cases so all three
  margin calls can occur.

### 5.4 Reference and validation sets

- **Retrieval reference library:** all train-split BACH cases (embeddings, labels, thumbnails).
- **Concept sanity check:** NCT-CRC-HE-100K, **300 tiles per class**, fixed seed. It is colorectal,
  not breast, but its tissue classes (TUM, LYM, STR, DEB, ADI) test whether the tissue-level
  criteria (tumor epithelium, lymphocytes, stroma, necrosis, fat) fire on the right tissue.
  Report it as a cross-organ sanity check, not a breast validation.
- Breast-specific concept validation is limited by available labels. Where possible, do a small
  **manual spot check**: a team member reviews the top-scoring tiles for each criterion and records
  how many look right (e.g. 20 tiles per criterion). Report the result as a qualitative check.

---

## 6. Model pipeline (Delta)

Run steps in order. Every step is **idempotent and resumable**: it skips work whose outputs exist,
and writes outputs atomically. Test every step on 5 cases (`--limit 5`) before running in full.

### 6.1 `01_build_cases.py`
Inputs: BACH images and labels found in 5.2. Outputs: `cases.csv` (case_id, image_path, label, split, lane, grid),
`tiles.csv` (case_id, tile_idx, row, col, x, y, path, qc_pass), downscaled display JPEG per case.

### 6.2 `02_embed_tiles.py`
- Embed all QC-passing tiles with **CONCH** (primary) and **MUSK** (agreement signal).
- Default: hosted `/embed`, batch size 32, exponential backoff, resume from partial CSV,
  at most ~4 concurrent requests from our team.
- Fallback: `pipeline/jobs/embed_tiles.sbatch` wraps the event repo's `run_embeddings.py`
  (`--model conch` / `--model musk`), 1-hour limit.
- Output: `emb_conch.npy`, `emb_musk.npy` aligned with `tiles.csv`; also embed the NCT-CRC
  validation tiles with CONCH.

### 6.3 `03_encode_prompts.py` (+ `jobs/encode_prompts.sbatch`)
- The hosted API can't return text embeddings, so encode prompts ourselves using the event repo's
  `models.resolve_adapter("conch")` / `("musk")` and their `encode_texts`. Text encoding is light;
  a 15-minute GPU job (or a CPU job if it loads) is enough.
- Verify text embeddings are L2-normalized and live in the same space as `/embed` image
  embeddings (the adapters normalize; confirm in `models.py`).
- Encode **every prompt in `config/concepts.yaml`** (several phrasings per concept, positive and
  negative) and **class prompts** for zero-shot baselines.
- Output: `text_conch.npz`, `text_musk.npz` keyed by prompt ID.

### 6.4 `config/concepts.yaml` — breast criterion prompt bank

Each concept has 3–5 positive phrasings and 2–3 negative/contrast phrasings, written in breast
pathology language (e.g. "invasive ductal carcinoma cells infiltrating desmoplastic stroma").
Concept score per tile = mean cosine similarity to positives minus mean similarity to negatives,
z-scored against the train split.

Concepts are grouped so the UI can present them the way a breast pathologist thinks:

| Group | id | Criterion | Expected direction |
|---|---|---|---|
| **Nottingham-style** | `tubules` | Well-formed tubules/glands with preserved architecture | lower risk |
| | `pleomorphism` | Marked nuclear pleomorphism: enlarged, irregular nuclei, prominent nucleoli | higher risk |
| | `mitoses` | Frequent mitotic figures | higher risk (expect a weak signal at tile scale) |
| **In situ vs invasive** | `invasion` | Tumor cells infiltrating stroma as irregular nests, cords or single cells | higher risk (Invasive) |
| | `duct_confined` | Atypical cells confined within ducts, intact myoepithelial layer | in situ pattern |
| **Benign / normal** | `normal_lobules` | Normal terminal duct lobular units with two-layered epithelium | lower risk |
| | `benign_stroma_fat` | Benign fibrous stroma or adipose tissue | lower risk |
| **Microenvironment** | `lymphocytes` | Dense lymphocytic infiltrate (TILs) | context, not ranked |
| | `necrosis` | Tumor necrosis | higher risk |

The model learns the actual direction and weight of each concept; "expected direction" is only a
sanity check. If a learned coefficient contradicts it (e.g. `tubules` pushing toward Invasive),
surface that on the Metrics page rather than hiding it.

Wording rule for the UI: concepts are "**features associated with** invasive carcinoma",
never claimed as biological causes, and never presented as a Nottingham grade.

### 6.5 `04_concepts.py`
- Per tile: concept scores (6.4). Per case: aggregate each concept with **mean of top-3 tiles**
  and **fraction of tiles above threshold**.
- Prompt-ensemble spread: std of concept score across phrasings (feeds confidence).
- Concept sanity check on NCT-CRC (cross-organ): AUROC of `lymphocytes` for LYM vs rest,
  `necrosis` for DEB, `pleomorphism` for TUM, `benign_stroma_fat` for STR+ADI. Write
  `eval/concepts.json`.
- Breast concept check on BACH: per-class mean of each concept (e.g. `invasion` should be highest
  in Invasive, `duct_confined` in InSitu, `normal_lobules` in Normal) plus the manual spot check
  from 5.4. Write `eval/concepts_bach.json`.

### 6.6 `05_risk_model.py`
Train on the train split, tune on calibration, report on test.
- **Model A — black-box ceiling:** logistic regression on mean-pooled CONCH tile embeddings
  (case level). Also a tile-level probe (case label propagated to tiles) for per-tile risk heatmaps.
- **Model B — concept bottleneck (the explainable one):** logistic regression on the aggregated
  concept features only (~16 features). Its coefficients × feature values are the per-criterion
  contributions shown in the UI.
- **Report both.** If B is within a reasonable margin of A, B drives predictions. If B is much worse,
  A drives predictions and B is shown as "associated features" with its agreement with A reported.
  Record the choice in `config/settings.yaml` and state it honestly on the Metrics page.
- Same pipeline with MUSK embeddings for the agreement signal.
- **Calibration:** temperature scaling fitted on the calibration split. Report ECE and a
  reliability diagram before/after.
- **Conformal prediction:** split conformal (APS or LAC — implement directly, ~30 lines, or use
  MAPIE if installed) on the calibration split at `alpha = 0.1` (configurable). Report empirical
  coverage and average set size on test. Note: with ~80 calibration cases, coverage is noisy;
  say so.
- **Risk score** = calibrated P(Invasive). The aggressive class is set in config as `Invasive`.
- Also report the **Invasive vs rest** confusion and, specifically, how often InSitu and Invasive
  are confused; this is the clinically important error and the reason for the p63/calponin rule.

### 6.7 `06_reliability.py` — simulated frozen-section artifacts and OOD
- Degradations at 3 severities: Gaussian blur, JPEG compression, stain/color shift,
  downsample-upsample. Apply to test-case tiles, re-embed (hosted server; keep the count small,
  e.g. 20 cases × 3 severities × 4 types), and rerun the read.
- **Reliability score** combines: OOD distance (mean cosine distance from each tile to its k=10
  nearest train tiles, calibrated to a percentile), conformal set size, and CONCH–MUSK disagreement.
- **Reliability flag** threshold chosen on the calibration split.
- Evaluate: accuracy vs severity curve, and AUROC of the reliability score for predicting errors.
  This is the evidence behind "defer to permanent".

### 6.8 `07_retrieval.py`
- Case embedding = mean of tile embeddings. Cosine kNN over the train library.
- For each test case: top-5 neighbors with labels and similarity, **neighbor label agreement**,
  and the **nearest neighbor with a different label** (contrast case).
- Report precision@5 (neighbor label = query label).

### 6.9 Surgical lane — margin read (simulation)
- Treat the image's four borders as inked margins: **top = superior, bottom = inferior,
  left = medial, right = lateral** (labeled as a simulation in the UI).
- A margin is "involved" if any tile touching that border has tile-level tumor probability above a
  threshold tuned on calibration data.
- Suggested call: `tumor_at_margin` (with margin names) / `clear` / `defer` (if reliability flag).
- The pathologist always makes the final call in the UI.

### 6.10 `08_build_case_json.py`
Assemble one JSON per test case (schema in Section 7), including structured `findings`, then
generate the grounded summary (Section 8.2) and the agent's draft orders (Section 9) **once**,
caching them in the JSON. Copy `artifacts/` to the laptop.

---

## 7. Case JSON contract

This contract lets frontend, backend, agent, and pipeline work in parallel. **Write a fixture
`artifacts/cases/demo_0001.json` by hand on day one** so UI work can start before the pipeline
finishes. Ground-truth labels live under `eval_only` and must never be shown in the worklist or
case view.

```json
{
  "case_id": "BACH_0123",
  "lane": "biopsy",
  "simulated": false,
  "arrived_offset_s": 340,
  "image": {"display": "images/BACH_0123.jpg", "width": 2048, "height": 1536,
            "tile_size": 512, "grid": [3, 4]},
  "quality": {"qc_tiles_kept": 11, "ood_percentile": 0.42, "reliability": 0.81,
              "reliability_flag": false},
  "prediction": {
    "model": "concept_bottleneck",
    "classes": ["Normal", "Benign", "InSitu", "Invasive"],
    "probs": [0.02, 0.05, 0.21, 0.72],
    "risk_score": 0.72,
    "conformal_alpha": 0.1,
    "conformal_set": ["InSitu", "Invasive"]
  },
  "concepts": [
    {"id": "invasion", "label": "Tumor infiltrating stroma", "value": 1.9,
     "contribution": 0.44, "direction": "up", "spread": 0.06, "top_tiles": [5, 6]},
    {"id": "duct_confined", "label": "Cells confined within ducts", "value": 0.8,
     "contribution": -0.12, "direction": "down", "spread": 0.09, "top_tiles": [2]}
  ],
  "tiles": [
    {"idx": 5, "row": 1, "col": 1, "x": 512, "y": 512, "risk": 0.88,
     "concepts": {"invasion": 2.3, "pleomorphism": 1.4, "lymphocytes": -0.2}}
  ],
  "similar_cases": [
    {"case_id": "BACH_0007", "label": "Invasive", "similarity": 0.93, "image": "images/BACH_0007.jpg"}
  ],
  "contrast_case": {"case_id": "BACH_0044", "label": "InSitu", "similarity": 0.88,
                    "differs_most_in": ["invasion"]},
  "agreement": {"conch_musk_same_top": true, "neighbor_agreement": 0.8},
  "margin": null,
  "flag": {"flagged": true, "tier": 2, "reasons": ["aggressive_in_conformal_set", "set_size_gt_1"]},
  "findings": [
    {"id": "F1", "kind": "prediction", "data": {"top": "Invasive", "p": 0.72, "set": ["InSitu", "Invasive"]}},
    {"id": "F2", "kind": "concept", "data": {"concept": "invasion", "contribution": 0.44, "tiles": [5, 6]}},
    {"id": "F3", "kind": "retrieval", "data": {"agree": 4, "of": 5, "label": "Invasive"}},
    {"id": "F4", "kind": "quality", "data": {"reliability_flag": false}}
  ],
  "summary": {
    "generator": "lumen:qwen3-coder-next",
    "sentences": [
      {"text": "Most likely invasive carcinoma, though in situ disease cannot be excluded at 90% confidence.", "findings": ["F1"]},
      {"text": "The strongest driver is tumor infiltrating stroma in two central regions.", "findings": ["F2"]}
    ]
  },
  "agent": {
    "status": "drafted",
    "orders": [
      {"type": "stain", "items": ["p63", "calponin"], "rule_id": "breast_insitu_vs_invasive",
       "rationale": "Myoepithelial markers help distinguish in situ from invasive disease."},
      {"type": "stain", "items": ["ER", "PR", "HER2", "Ki-67"], "rule_id": "breast_invasive",
       "rationale": "Standard biomarker panel if invasion is confirmed."}
    ]
  },
  "eval_only": {"true_label": "Invasive", "split": "test"}
}
```

Surgical cases set `"lane": "surgical"`, `"simulated": true`, and fill `margin`:
```json
"margin": {"suggested_call": "tumor_at_margin", "involved": ["superior"],
           "per_margin": {"superior": 0.81, "inferior": 0.07, "medial": 0.12, "lateral": 0.04}}
```

---

## 8. Explanation and summary

### 8.1 What the case view shows
- Image with a **tile heatmap overlay**; a selector switches between overall risk and each concept.
- **"Why" panel:** concept contributions as a signed bar chart (pushes toward / away from the
  aggressive class). Clicking a bar highlights its `top_tiles`.
- **"How sure" panel:** calibrated probability, conformal set as chips (one chip = confident),
  CONCH–MUSK agreement, neighbor agreement, reliability flag.
- **"Compared to" panel:** top-5 similar cases and the contrast case with "differs most in".
- **Summary:** sentences rendered with clickable finding references.

### 8.2 Grounded summary generation (`backend/summary.py`, run offline in step 08)
- Prompt the Lumen model with the `findings` array only, and require JSON output:
  `[{"text": ..., "findings": ["F1", ...]}]`, max 4 sentences, no numbers that aren't in findings,
  no diagnosis stated as fact, phrase concepts as "features associated with".
- **Validator:** drop any sentence citing no finding or an unknown finding ID; reject numbers not
  present in the cited findings. If fewer than 2 sentences survive, use the template generator.
- **Template generator** (always available, used in tests and as fallback) produces the same
  structure deterministically.
- Cache results; never call the LLM during the live demo by default.

---

## 9. Agent (`agent/`)

- **Deterministic core (`rules.py`):** reads case findings and `config/stain_rules.yaml`.
  Rules are data, e.g. for BACH:
  - `breast_insitu_vs_invasive`: InSitu and Invasive both in conformal set → p63, calponin
    (myoepithelial markers).
  - `breast_invasive`: top class Invasive → ER, PR, HER2, Ki-67.
  - `low_quality`: reliability flag → recut / deeper levels.
  - `uncertain_benign_atypia`: Benign and InSitu both in set → deeper levels.
  - `insufficient`: few QC-passing tiles → request new sample.
  Mark the file "illustrative; requires clinical review" in a header comment and in the UI.
- **LLM layer (`agent.py`, optional):** Lumen tool-calling with tools `get_case_findings`,
  `lookup_rules`, `draft_order`. The LLM may only choose among rule outputs and write the rationale;
  it cannot invent stains. If tool-calling fails or coins run out, fall back to rules +
  template rationale.
- **States:** `none → drafted → approved | edited_and_approved | deferred_closer_look | dismissed`.
  Orders are never "placed" without a pathologist action. Every transition is logged with a timestamp.

---

## 10. Backend (`backend/`, FastAPI + SQLite, runs on laptop)

- On startup, load `artifacts/cases/*.json` into SQLite (tables: `cases`, `decisions`, `orders`,
  `routed_results`, `events`).
- `simulator.py`: replays test cases as arrivals using `arrived_offset_s` (speed multiplier,
  e.g. 30×). Controls: start, pause, reset, inject a surgical case now.
- Endpoints:
  - `GET  /api/worklist?lane=all|surgical|biopsy` → sorted by tier rules (Section 1)
  - `GET  /api/cases/{id}` → full case JSON minus `eval_only`
  - `POST /api/cases/{id}/decision` → `{action: approve_orders|edit_orders|closer_look|dismiss_flag|sign_out|margin_call, payload}`
  - `GET  /api/routed?to=surgeon|oncologist` → results sent after sign-out / margin call
  - `POST /api/sim/{start|pause|reset|inject_surgical}`
  - `GET  /api/metrics` → contents of `artifacts/eval/*.json`
  - static files for `artifacts/images/`
- Surgical timers are computed server-side from arrival time; stop on `margin_call`.
- Tier assignment lives in one function with unit tests covering the safety rule.

---

## 11. Frontend (`frontend/`, Vite + React + TypeScript + Tailwind)

Visual language from the pitch deck: ink `#1B1A2E`, paper `#F5F2EC`, hematoxylin purple
`#4A3B8F` (biopsy), eosin pink `#B0335C` (surgical). Poll the backend every 2 s.

Views:
1. **Worklist** — rows grouped by tier; surgical rows show a live timer; badges for flag reasons;
   conformal-set chips; risk score; "orders drafted" indicator. New arrivals animate in.
2. **Case view (biopsy)** — Section 8.1 panels plus the agent panel: drafted orders with
   rationale and rule ID; buttons **Order tests**, **Edit**, **Take a closer look first**,
   **Dismiss flag**, **Sign out**.
3. **Case view (surgical)** — large timer, image with the four margins drawn and colored by
   involvement, reliability indicator, and three buttons: **Margins clear**,
   **Tumor at margin** (pick margins), **Defer to permanent**. Banner: "Simulated frozen section".
4. **Surgeon screen** — receives margin calls in real time.
5. **Oncologist inbox** — receives signed-out biopsy summaries.
6. **Metrics** — calibration diagram, conformal coverage/set size, concept sanity AUROCs,
   reliability-vs-degradation curve, retrieval precision@5, queue simulation result.

---

## 12. Evaluation (`eval/`)

All metrics on the **test split** only; write JSON + PNG plots to `artifacts/eval/`.
The BACH test split is small, so report a **bootstrap 95% confidence interval** for every metric.

| Claim | Metric | Script |
|---|---|---|
| Triage separates invasive cases | AUROC (Invasive vs rest), Model A vs Model B; InSitu↔Invasive confusion | `metrics.py` |
| Confidence is honest | ECE + reliability diagram (pre/post temperature); conformal coverage and mean set size | `metrics.py` |
| Explanations are real | Concept AUROCs on NCT-CRC; **deletion test**: removing a case's top-3 tiles for its top concept lowers risk more than removing 3 random tiles (report mean drop ± CI) | `metrics.py`, `deletion_test.py` |
| Reliability flag works | Accuracy vs degradation severity; AUROC of reliability score for predicting errors | `06_reliability.py` |
| Retrieval is meaningful | precision@5 | `07_retrieval.py` |
| Workflow benefit | **Queue simulation:** replay a day of arrivals (Poisson, configurable rate) with a fixed read time per case; compare mean/median time-to-first-look for aggressive cases, FIFO vs PathQueue tiers. State assumptions on the Metrics page. | `queue_sim.py` |

---

## 13. Milestones and owners

Adapt names to the team. Time boxes assume work continues from Saturday afternoon.
**All Delta GPU/hosted-server work must finish by Sun ~12:00** (hosted servers stop at 18:00;
leave margin). Demo freeze **Sun ~15:30**; final presentation before 17:00.

| # | Milestone | Owner | Done when |
|---|---|---|---|
| M0 | Env + access: every member has HF access to CONCH/MUSK, Lumen key, team dir, event repo cloned, packages installed | all | `example_student_usage.py` succeeds against the CONCH endpoint |
| M1 | BACH verified on Delta; label mapping written | model | `inventory.json` shows ≥ 80 labeled images per class (or the human-run download is done) |
| M2 | Hand-written fixture JSON + backend skeleton + worklist UI against fixture | app | worklist renders and sorts the fixture |
| M3 | Cases/tiles built; CONCH+MUSK tile embeddings done | model | `.npy` files aligned with `tiles.csv` (Sat evening) |
| M4 | Prompt embeddings; concept scores; NCT-CRC concept check | model | `eval/concepts.json` |
| M5 | Risk models A/B, calibration, conformal, retrieval | model | metrics JSON + test predictions (Sun morning) |
| M6 | Reliability/degradation experiment; margin read | model | reliability curve + margin fields |
| M7 | Case JSONs + cached summaries + agent drafts; copied to laptop | model + agent | `artifacts/cases/*.json` validate against the contract |
| M8 | Full UI: case views, agent panel, surgeon/oncologist screens, simulator | app | end-to-end demo path works with real artifacts |
| M9 | Metrics page + queue simulation | eval | plots render in the app |
| M10 | Deck updated with real numbers, honest limitations, citations; rehearsal | all | demo rehearsed twice |

Parallelism: M2 and M8 depend only on the Section 7 contract, not on pipeline results.

---

## 14. Demo script (target 3 minutes)

1. Start the simulator. Biopsies arrive; one jumps to tier 2 on arrival despite arriving last.
2. Open it: heatmap → "Why" panel (tumor infiltrating stroma drives the call) → conformal set shows
   {InSitu, Invasive} → agent has drafted p63/calponin and ER/PR/HER2/Ki-67 → click **Order tests**.
   Sign out → appears in the oncologist inbox.
3. Inject a surgical case: it goes straight to the top with a timer. Open it: superior margin
   highlighted → **Tumor at margin: superior** → surgeon screen updates, timer stops.
4. Inject a degraded surgical case: reliability flag raised → **Defer to permanent**.
5. Metrics page: queue simulation, calibration, concept checks, reliability curve. Close on
   limitations: simulated frozen lane, small dataset, rules need clinical review.

---

## 15. Risks and fallbacks

| Risk | Fallback |
|---|---|
| BACH incomplete in `pathmmu` | Parse labels from VQA files; otherwise a human downloads the official BACH release (5.2). Do not switch organs |
| Few BACH images (small test set) | Report confidence intervals (bootstrap) on every metric; say the dataset is small |
| InSitu vs Invasive confusion is high | Make it the story: the conformal set captures it, the case goes to tier 2, and the agent drafts p63/calponin |
| Hosted server slow/overloaded | Own 1-hour GPU job via `jobs/embed_tiles.sbatch`; cache aggressively |
| CONCH text encoder won't load | Use QuiltNet or PLIP for both image and text sides (open models, `openclip:`/`hfclip:` specs); never mix models |
| Concept bottleneck much weaker than black box | Predict with Model A; show concepts as associated features with measured agreement |
| Lumen coins exhausted | Template summaries and rule-based agent rationales (already the fallback) |
| GPU budget running low | Embed fewer degradation variants; reduce MUSK to agreement on a subset |
| Reserved-node window boundaries | Never request `--time` that crosses Sun 08:00 or Sun 18:00 |
| Delta unreachable at demo time | App runs locally from `artifacts/` |

---

## 16. Rules for Claude Code

- Read Section 2 before running anything on Delta. Never run inference on the login node.
- **Never submit Slurm jobs yourself.** Write the script and the exact command; a human submits it.
- Never write or print secrets. Read `LUMEN_API_KEY` and endpoint keys from environment variables
  or a git-ignored `.env`. Never commit `.env`, data, embeddings, or artifacts.
- Test every pipeline step with `--limit 5` and show the output before a full run.
- Keep the Section 7 JSON contract stable; if it must change, update the fixture, backend models,
  and frontend types in the same change.
- Read class names, aggressive class, thresholds, alpha, and paths from `config/`; no magic constants.
- Keep ground-truth labels out of every user-facing view.
- Stay in scope: breast cancer on BACH. Don't add other organs or datasets unless a human asks
  (Section 17 describes how that would work).
- UI wording: "suggested", "associated with", "simulated"; the model never "diagnoses".
- Write small unit tests for: tier assignment (including the conformal safety rule), conformal set
  construction, summary validator, rules engine, margin involvement.
- Track every model, dataset, and tool used in README "Credits" for the final presentation:
  CONCH, MUSK, (QuiltNet/PLIP if used), BACH (Aresta et al., ICIAR 2018 challenge, Medical Image Analysis 2019), NCT-CRC-HE-100K, PathMMU as the source folder,
  the `ncsa/ashby-hackathon-2026` tooling, Lumen and the specific LLM, NCSA Delta.

---

## 17. Extending beyond breast (pitch claim, not build scope)

The system is organ-specific in four places and organ-agnostic everywhere else. Everything
organ-specific lives in `config/`:

| Organ-specific (swap per organ) | Organ-agnostic (reused as is) |
|---|---|
| Class names and the aggressive class (`settings.yaml`) | Tiling, QC, embedding, retrieval |
| Grading criteria prompt bank (`concepts.yaml`) | Concept scoring and concept bottleneck model |
| Follow-up order rules (`stain_rules.yaml`) | Calibration, conformal sets, reliability flag |
| Labeled cases for training, calibration and testing | Worklist tiers, agent states, UI, routing |

Each organ needs its own grading criteria because grading is organ-specific (e.g. Gleason patterns
for prostate, nucleolar grade for clear cell kidney cancer, pattern-based grading for lung
adenocarcinoma), and some cancers rely more on other measures (e.g. Breslow thickness for melanoma).

Pitch wording: "Validated on breast. Adding an organ means swapping its grading criteria, follow-up
rules, and labeled data; the pipeline, confidence methods and worklist stay the same."

**Stretch evidence only if time remains after M10:** the NCT-CRC concept sanity check (5.4) already
shows the tissue-level criteria (lymphocytes, necrosis, tumor, stroma) behaving sensibly on
colorectal tissue. Present it as a small piece of evidence that the explanation layer transfers,
not as colorectal support.
