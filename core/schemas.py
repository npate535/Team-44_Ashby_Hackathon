"""Column/shape specs and path helpers for every inter-task file. Task 1 freezes these; Tasks 2-13 code to them.

Read this module top to bottom once: it is the data dictionary of the pipeline. If a task needs a new column or
file, it asks the controller ("Requested shared changes") instead of inventing a format.

Conventions (apply to every file):
  * Roots. Pipeline files live under the DATA ROOT (core.config.data_root: Delta
    /work/nvme/bfqi/teams/<team>/pathqueue; laptop ./data; synthetic ./data_synthetic). App-facing files live under
    the ARTIFACTS ROOT (core.config.artifacts_root: ./artifacts). Committed fixtures live flat in ./fixtures.
  * Paths stored inside files are POSIX strings relative to the data root for files under it, absolute otherwise
    (source datasets on Delta). resolve(root, p) turns a stored path into a Path.
  * Vocabulary: splits train | calibration | test; lanes surgical | biopsy; classes and their order come from
    settings.classes; grids g512 (3x4, the contract tiles) and g256 (6x8).
  * Tile identity is (case_id, grid, tile_idx): tile_idx = row * cols + col, x = col * size, y = row * size, in
    source-image pixels (2048x1536 for BACH). Display JPEGs are downscaled; consumers rescale.
  * Case IDs are opaque (settings split.ids): "<prefix>_<n:04d>" from a seeded permutation, so an ID never reveals
    the label or the class-sorted index. Derived IDs: variant_case_id(), composite_case_id().
  * Booleans in CSVs are 0/1 integers (True/False is accepted on read). Empty string or NaN means missing.
  * Ground truth lives in cases.csv `label`, manifest `label`, composites/wsi truth columns, eval files and the case
    JSON `eval_only` block only. It never reaches API responses or the UI.
  * Every writer uses core.io (atomic) and writes meta/<step>.json; synthetic data carries synthetic=true.

Portable: Python >= 3.10, stdlib + pandas (frame checks only). Must not import pydantic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

# ---- vocabularies ----------------------------------------------------------------------------------------------
SPLITS = ("train", "calibration", "test")
LANES = ("surgical", "biopsy")
GRIDS = ("g512", "g256")
HEADS = ("align", "probe")
SOURCES = ("bach_official", "pathmmu", "synthetic")
DEGRADE_TYPES = ("blur", "jpeg", "stain", "resample")
MARGINS = ("superior", "inferior", "medial", "lateral")
MARGIN_TRUTH = ("exact_negative", "tumor_side", "mixed")  # composites: per-margin ground truth
WSI_MARGIN_TRUTH = ("tumor", "clear")
CELL_TYPES = ("neoplastic", "inflammatory", "connective", "dead", "epithelial")  # PanNuke ids 1..5
CONCEPT_DIRECTIONS = ("up", "down", "context")

# Embedding sets in embed_manifest.csv (BUILD_PLAN Task 2). Core sets first; the rest are auxiliary.
EMBED_SETS = (
    "bach512",        # all 400 cases, g512 tiles (contract tiles); CONCH/MUSK
    "bach256",        # all 400 cases, g256 tiles; 224-input encoders, margin probe training
    "nct",            # NCT-CRC-HE-100K sample (nct.per_class per class), whole 224-px tiles
    "deg512",         # calibration + test cases x degrade.types x severities, g512 tiles (reliability)
    "deg256",         # optional: degraded test-surgical cases at degrade.display_severity, g256 (variant margins)
    "comp512",        # optional: composite fields, g512 tiles (case-level read of composite variants)
    "comp256",        # composite fields, g256 tiles (margin segments, §A12 composite test)
    "wsi512",         # optional: labeled WSI tiles at 512 (WSI-enhanced tile-probe ablation)
    "wsi256",         # labeled WSI tiles at 256 (>= wsi.min_label_purity)
    "wsi_margin256",  # held-out WSI margin windows, g256 tiles
    "quilt",          # mined Quilt-1M concept images (label = "<concept_id>:pos|neg")
    "pathgen",        # mined PathGen patches, only if inventory finds patch images
)

# Required key in inventory.json read by core.config.effective_mpp.
INVENTORY_MPP_KEY = "measured_mpp"

# Eval outputs under <artifacts_root>/eval/<name>.json (GET /api/metrics serves them all). Each JSON carries the
# EVAL_ENVELOPE keys (core.io.provenance gives all but `name`) plus its payload.
EVAL_ENVELOPE = ("name", "created_utc", "config_hash", "git_sha", "synthetic")
EVAL_FILES: dict[str, str] = {
    "fact_sheet": "Judge response · data units, splits, metrics, corpora, reference distributions, component roles",
    "attention_comparison": "Judge response · ABMIL vs production probe on identical inputs: CV and paired test",
    "human_baseline": "Judge response · BACH pathologist reader study and B-Path concordance next to Sepia per-class recall",
    "qc": "01q · tissue-QC fix: old vs new rule, pass rates, insufficient-tissue counts, display-vs-full-res validation",
    "concepts": "Task 9 · NCT cross-organ concept AUROCs; per concept {caption_consistent, cav_auroc}",
    "concepts_bach": "Task 9 · BACH per-class concept means (test) and spot-check summary",
    "zeroshot": "Task 9 · zero-shot BACH 4-class baseline per text-aligned encoder",
    "model_choice": "Task 10 · pre-registered candidate CV table, one-SE choice, A-vs-B decision",
    "reliability": "Task 10 · accuracy vs degradation severity; AUROC of reliability for errors",
    "retrieval": "Task 10 · precision@5 of the train reference library",
    "finetune": "Task 10 · optional LoRA single calibration-split comparison",
    "metrics": "Task 11 · AUROC A vs B, macro AUROC/F1, BAcc, confusion, InSitu<->Invasive counts",
    "calibration": "Task 11 · ECE/Brier/NLL before/after temperature + reliability-diagram data",
    "conformal": "Task 11 · coverage and set size: marginal, per class, APS, Mondrian",
    "conformal_resplit": "Round 3 · conformal coverage across 1,000 patient-grouped calibration/test re-splits",
    "thunder": "Task 11 · comparison with published THUNDER probes (same split, different protocol)",
    "deletion": "Task 11 · deletion-test faithfulness",
    "tile_contrib": "05d · exact region contributions: identity check against predictions.csv, reads without "
                    "per-tile embeddings",
    "queue_sim": "Task 11 · FIFO vs tiers vs tiers+aging queue simulation",
    "deferral": "Round 3 · deferral (risk-coverage) curve: accuracy on kept test reads as the least reliable are deferred",
    "margin": "Task 11 · margin eval: benign FPR, composite exact-negative clear rate, echo stats, WSI windows",
    "cells": "Task 8 · cell-feature validation (CELLS_SPEC §8)",
    "spotcheck": "Task 6 · spot-check results recorded in the UI",
    "analytics": "Dashboard · chart-ready test-split aggregates: ROC/PR curves, confusion, reliability bins, conformal "
                 "set composition, tiers by class, queue-wait distributions",
    "external_tcga": "External · TCGA-BRCA generalization of the BACH-trained production probe (TCGA_TODO.md)",
}

# Patterns shared with core.contract (case IDs; media refs relative to the artifacts root).
CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,127}$"
IMAGE_REF_PATTERN = r"^images/[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$"
CELL_REF_PATTERN = r"^cells/[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$"

_CASE_ID_RE = re.compile(CASE_ID_PATTERN)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_CLASS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
_MEDIA_REF_RE = re.compile(r"^(images|cells)/[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


# ---- spec types -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Col:
    """One CSV column. kind: str | int | float | bool. required = the column must exist (values may still be
    empty unless the column is part of the table key). choices = allowed non-empty values."""

    name: str
    kind: str
    desc: str
    required: bool = True
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Table:
    filename: str  # path under the data root ("<...>" marks a pattern)
    producer: str
    consumers: str
    key: tuple[str, ...]
    columns: tuple[Col, ...]
    notes: str = ""

    @property
    def required(self) -> list[str]:
        return [c.name for c in self.columns if c.required]


@dataclass(frozen=True)
class JsonSpec:
    filename: str
    producer: str
    consumers: str
    required_keys: tuple[str, ...]
    doc: str

    def problems(self, obj: Any) -> list[str]:
        if not isinstance(obj, Mapping):
            return [f"{self.filename}: top level must be an object"]
        missing = [k for k in self.required_keys if k not in obj]
        return [f"{self.filename}: missing keys {missing}"] if missing else []


@dataclass(frozen=True)
class ArraySpec:
    filename: str
    producer: str
    consumers: str
    arrays: tuple[tuple[str, str], ...]  # (array name, dtype + shape + meaning)
    doc: str

    def problems(self, arrays: Mapping[str, Any]) -> list[str]:
        missing = [name for name, _ in self.arrays if name not in arrays and not name.startswith("(")]
        return [f"{self.filename}: missing arrays {missing}"] if missing else []


# ---- CSV tables ---------------------------------------------------------------------------------------------------
def _c(name: str, kind: str, desc: str, **kw: Any) -> Col:
    return Col(name, kind, desc, **kw)


_CASE = _c("case_id", "str", "case ID (see conventions)")
_SPLIT = _c("split", "str", "train | calibration | test", choices=SPLITS)
_GRID = _c("grid", "str", "g512 | g256", choices=GRIDS)

CASES = Table(
    "cases.csv", "Task 2 · 01_build_cases (Delta); Task 3 · make_synthetic", "Tasks 8-11, eval",
    key=("case_id",),
    columns=(
        _CASE,
        _c("source", "str", "origin of the image", choices=SOURCES),
        _c("source_file", "str", "path relative to the dataset root, e.g. Photos/Benign/b001.tif"),
        _c("source_index", "int", "0..n_images-1 index in the class-folder-sorted, file-name-sorted list (THUNDER ranges)"),
        _c("image_path", "str", "readable source image (absolute on Delta)"),
        _c("label", "str", "ground-truth class (settings.classes); eval-only downstream"),
        _SPLIT,
        _c("patient_id", "str", "patient ID from the BACH patient xlsx; empty when unidentified"),
        _c("group", "str", "patient-level group for grouped CV and cluster bootstrap: 'p<patient_id>' when known, "
                           "else 'b<source_index // split.cv.block_size>' (contiguous file-number block)"),
        _c("lane", "str", "surgical | biopsy; surgical only for test cases (settings.lanes)", choices=LANES),
        _c("width", "int", "source width in pixels"),
        _c("height", "int", "source height in pixels"),
        _c("display_path", "str", "display JPEG, data-root relative: display/<case_id>.jpg"),
    ),
)

TILES = Table(
    "tiles.csv", "Task 2 · 01_build_cases; Task 3 · make_synthetic", "Tasks 3, 8 (04a), 9, 10, 11",
    key=("case_id", "grid", "tile_idx"),
    columns=(
        _CASE, _GRID,
        _c("tile_idx", "int", "row * cols + col"),
        _c("row", "int", "grid row (0 = top)"),
        _c("col", "int", "grid column (0 = left)"),
        _c("x", "int", "left edge in source pixels = col * size"),
        _c("y", "int", "top edge in source pixels = row * size"),
        _c("size", "int", "tile edge in source pixels (512 or 256)"),
        _c("tissue_frac", "float", "fraction of tissue pixels (settings.qc thresholds)"),
        _c("qc_pass", "bool", "1 when tissue_frac >= qc.min_tissue_frac"),
        _c("path", "str", "tile PNG, data-root relative: tiles/<grid>/<case_id>/<row>_<col>.png"),
    ),
    notes="Every tile of both grids is listed, QC pass or not. Contract tiles[] use grid g512 with qc_pass = 1.",
)

EMBED_MANIFEST = Table(
    "embed_manifest.csv", "Task 2 · 01_build_cases (core sets), prep_b (aux sets appended)", "Task 3 · 02_embed",
    key=("row_id",),
    columns=(
        _c("row_id", "str", "unique row: '<item_id>|v<view>'"),
        _c("item_id", "str", "identity shared by all views of one image (make_item_id); TTA means are per item_id"),
        _c("set", "str", "embedding set", choices=EMBED_SETS),
        _c("case_id", "str", "BACH/composite/variant case ID, NCT file stem, WSI tile or window ID, Quilt/PathGen image ID"),
        _c("split", "str", "split of the BACH case; empty for non-BACH sets", choices=SPLITS),
        _c("grid", "str", "g512 | g256 for tiled sets; empty for whole images", choices=GRIDS),
        _c("tile_idx", "int", "tile index within the grid; -1 for whole images"),
        _c("path", "str", "image file to read (data-root relative or absolute)"),
        _c("x", "int", "crop box left within the image at `path`"),
        _c("y", "int", "crop box top"),
        _c("w", "int", "crop box width; (0, 0, image width, image height) = whole image"),
        _c("h", "int", "crop box height"),
        _c("view", "int", "dihedral view 0-7: rotate 90 deg counter-clockwise (view % 4) times, then mirror "
                          "left-right if view >= 4; 0 = identity"),
        _c("degrade_type", "str", "empty = clean, else blur | jpeg | stain | resample (pipeline.degrade)", choices=DEGRADE_TYPES),
        _c("degrade_severity", "int", "0 = clean, 1-3"),
        _c("degrade_seed", "int", "seed passed to pipeline.degrade.apply"),
        _c("label", "str", "ground truth for non-BACH sets: NCT class, WSI region class, '<concept_id>:pos|neg' for "
                           "quilt/pathgen; empty for BACH (labels live in cases.csv)"),
        _c("group", "str", "leakage group for held-out splits: WSI patient, Quilt video, PathGen slide; empty for BACH"),
    ),
    notes="Views: train items get settings encoders.views[grid] rows (views 0..n-1); calibration/test items get the "
          "same views and are reduced to their TTA mean on the node; whole-image sets use view 0 only; degraded sets "
          "(deg512, deg256) use view 0 only (ruling M9).",
)

EMB_INDEX = Table(
    "emb/<encoder>/<set>_<head>_index.csv", "Task 3 · 02_embed / export", "Tasks 8, 9, 10, 11",
    key=("emb_row",),
    columns=(
        _c("emb_row", "int", "row number in the aligned .npy (0..N-1)"),
        _c("item_id", "str", "manifest item_id (unique in the per-item file)"),
        _c("set", "str", "embedding set", choices=EMBED_SETS),
        _CASE,
        _c("split", "str", "manifest split", choices=SPLITS),
        _c("grid", "str", "manifest grid", choices=GRIDS),
        _c("tile_idx", "int", "manifest tile_idx"),
        _c("view", "int", "-1 in the per-item file (mean of n_views views); 0-7 in the _views file"),
        _c("n_views", "int", "number of views averaged into the row (1 = no TTA)"),
        _c("degrade_type", "str", "manifest degrade_type", choices=DEGRADE_TYPES),
        _c("degrade_severity", "int", "manifest degrade_severity"),
        _c("label", "str", "manifest label"),
        _c("group", "str", "manifest group"),
    ),
    notes="Two files per (encoder, set, head): <set>_<head>.npy + _index.csv with one row per item_id (TTA mean; "
          "the only file most consumers read), and <set>_<head>_views.npy + _views_index.csv with one row per "
          "(item, view) for every split of the clean multi-view sets (bach512, bach256): train rows feed augmentation, "
          "calibration/test view-0 rows are the like-for-like clean baseline for degraded reads (ruling M9: deg512 / "
          "deg256 are embedded at view 0 only).",
)

COMPOSITES = Table(
    "composites.csv", "Task 2 · composites (prep_b)", "Tasks 3, 10, 11",
    key=("case_id",),
    columns=(
        _c("case_id", "str", "opaque composite case ID: composite_case_id(carcinoma_case_id, k) = COMP_<k:03d>, k = 0..composites.n-1"),
        _c("carcinoma_case_id", "str", "test case giving the carcinoma half (InSitu or Invasive)"),
        _c("benign_case_id", "str", "test case giving the benign half (Normal or Benign)"),
        _c("orientation", "int", "0..composites.orientations-1"),
        _c("carcinoma_side", "str", "half that is carcinoma", choices=("left", "right", "top", "bottom")),
        _c("width", "int", "pixels"),
        _c("height", "int", "pixels"),
        _c("path", "str", "composite image, data-root relative: composites/<case_id>.png"),
        _c("display_path", "str", "display JPEG: display/<case_id>.jpg"),
        *(_c(f"truth_{m}", "str", f"{m} margin truth", choices=MARGIN_TRUTH) for m in MARGINS),
    ),
)

WSI_WINDOWS = Table(
    "wsi_windows.csv", "Task 2 · wsi_extract (prep_b)", "Tasks 3, 10, 11 (margin eval)",
    key=("window_id",),
    columns=(
        _c("window_id", "str", "case-like ID of the held-out window, e.g. WSI_A03_07"),
        _c("wsi_id", "str", "source slide, e.g. A03"),
        _c("patient_id", "str", "from ICIAR2018_BACH_dataset_WSI_patient.xlsx"),
        _c("fold", "int", "patient-disjoint WSI fold"),
        _c("x", "int", "window left in slide pixels resampled to wsi.target_mpp"),
        _c("y", "int", "window top"),
        _c("width", "int", "window width (wsi.window)"),
        _c("height", "int", "window height"),
        _c("stratum", "str", "tumor at a specific edge / central only / none", choices=("edge_tumor", "central_only", "none")),
        _c("path", "str", "window image, data-root relative"),
        *(_c(f"truth_{m}", "str", f"{m} margin truth from annotated 256-px strips", choices=WSI_MARGIN_TRUTH) for m in MARGINS),
    ),
)

CASE_FEATURES = Table(
    "case_features.csv", "Task 9 · 04_concepts", "Task 10 · 05_risk_model; Task 11",
    key=("case_id",),
    columns=(_CASE, _SPLIT),
    notes="Dynamic columns per concept id c (settings concepts.ids order), k = concepts.top_k: c__top<k> (mean of the "
          "top-k QC tile z) and c__frac (fraction of QC tiles with z > concepts.z_threshold) are the model features "
          "(concept_feature_columns, 18 for 9 concepts); c__spread is the case prompt-ensemble spread "
          "(concept_spread_columns); optional components c__text_top<k>, c__cav_top<k>.",
)

_TILE_RISK_COLUMNS = (
    _CASE, _SPLIT, _GRID,
    _c("tile_idx", "int", "tile index within the grid"),
    _c("risk", "float", "P(aggressive class) for the tile, in [0, 1]"),
    _c("probe", "str", "name of the tile probe that produced the score"),
    _c("fold", "int", "-1 = probe fit on the full train split; >= 0 = out-of-fold fold index (train cases)"),
)
TILE_RISK = Table(
    "tile_risk.csv", "Task 10 · 05_risk_model", "Task 11 · 08 (contract tiles[].risk heatmap)",
    key=("case_id", "grid", "tile_idx"), columns=_TILE_RISK_COLUMNS,
    notes="Production heatmap probe (microscopy-only unless the WSI-enhanced probe wins on calibration, §A9).",
)
TILE_RISK_OOF = Table(
    "tile_risk_oof.csv", "Task 8 · 04a (shared function, inside cells.sbatch); Task 10", "Task 8 · 04a; Task 10",
    key=("case_id", "grid", "tile_idx"), columns=_TILE_RISK_COLUMNS,
    notes="Microscopy-only probe: train rows out-of-fold (fold >= 0), calibration/test rows from the full-train fit "
          "(fold = -1), CELLS_SPEC §4.",
)

PREDICTIONS = Table(
    "predictions.csv", "Task 10 · 05_risk_model", "Task 11 · 08 and eval",
    key=("case_id",),
    columns=(
        _CASE, _SPLIT,
        _c("model", "str", "production model name (settings model_choice)"),
        _c("temperature", "float", "temperature applied to the production model"),
        _c("set_lac", "str", "marginal LAC conformal set at settings.alpha (format_set); '' = empty set"),
        _c("qhat_lac", "float", "LAC threshold q-hat"),
        _c("set_size", "int", "number of classes in set_lac"),
        _c("risk", "float", "calibrated P(aggressive class) = pcal__<aggressive_class>"),
        _c("top", "str", "argmax class of the calibrated probabilities"),
        _c("set_aps", "str", "secondary APS set (format_set)", required=False),
        _c("set_mondrian", "str", "secondary Mondrian set (format_set)", required=False),
        _c("set_tier", "str", "tier-driving set: the conformal.method x mode set, argmax added when empty if "
                              "conformal.nonempty (core.conformal.tier_set_source); absent => set_lac", required=False),
        _c("set_tier_size", "int", "number of classes in set_tier", required=False),
        _c("agree_top", "bool", "production top == agreement model top (contract agreement.conch_musk_same_top)", required=False),
    ),
    notes="Dynamic: pcal__<class> for every class (cal_prob_col; calibrated production probabilities summing to 1) and "
          "p__<model>__<class> for every evaluated model (prob_col; uncalibrated). Rows: calibration and test cases "
          "(train rows optional, out-of-fold, split=train).",
)

RELIABILITY = Table(
    "reliability.csv", "Task 10 · 06_reliability", "Task 11 · 08 (contract quality block)",
    key=("case_id",),
    columns=(
        _c("case_id", "str", "case ID; degraded variants use variant_case_id()"),
        _c("source_case_id", "str", "clean case the row derives from (= case_id for clean rows)"),
        _c("variant", "str", "'clean' or '<degrade_type><severity>'"),
        _SPLIT,
        _c("ood_distance", "float", "mean over tiles of the mean cosine distance to the reliability.k nearest TTA-averaged train tiles"),
        _c("ood_percentile", "float", "percentile of ood_distance among clean calibration cases, in [0, 1]"),
        _c("set_size", "int", "conformal set size"),
        _c("conch_musk_disagree", "bool", "1 when the CONCH and MUSK models disagree on the top class"),
        _c("p_error", "float", "combiner P(error)"),
        _c("reliability", "float", "1 - p_error (contract quality.reliability)"),
        _c("reliability_flag", "bool", "1 when p_error >= threshold"),
        _c("threshold", "float", "p_error threshold chosen on calibration for reliability.target_error_recall"),
    ),
)

TILE_ATTENTION = Table(
    "tile_attention.csv", "Task 10 · 05b_abmil (Agent 3; optional, cut order §I item 4)",
    "Task 11 · 08 (contract tiles[].attention; Agent 4)",
    key=("case_id", "grid", "tile_idx"),
    columns=(
        _c("case_id", "str", "case ID; variants use their variant IDs"),
        _SPLIT, _GRID,
        _c("tile_idx", "int", "tile index within the grid"),
        _c("attention", "float", "ABMIL attention weight (>= 0; sums to 1 over a case's QC tiles)"),
        _c("model", "str", "ABMIL model name, e.g. abmil_uni2h"),
    ),
    notes="Absent when ABMIL is cut or not trained; 08 then omits tiles[].attention.",
)

CONTRIBUTIONS = Table(
    "contributions.csv", "Task 10 · 05_risk_model (Agent 3)",
    "Task 11 · 08 (contract concepts[].contribution, cells.features[].contribution; Agent 4)",
    key=("case_id", "feature"),
    columns=(
        _c("case_id", "str", "case ID; degraded / composite variants use their variant IDs"),
        _SPLIT,
        _c("model", "str", "concept-bottleneck model the contributions come from (cb_text | cb_text_cells | cb_cells)"),
        _c("feature", "str", "model feature column, e.g. invasion__top3, invasion__frac, nuc_area_cv, nuc_area_cv__missing"),
        _c("concept", "str", "concept id (settings concepts.ids) or cell feature id the feature belongs to"),
        _c("source", "str", "text | cav | cell (contract concepts[].source)", choices=("text", "cav", "cell")),
        _c("value", "float", "raw feature value (z units for concepts, physical units for cell features)"),
        _c("value_std", "float", "standardized value (train mean / std)"),
        _c("contribution", "float", "(W_aggressive - mean_k W_k) * value_std: signed push toward (+) / away from (-) "
                                    "the aggressive class"),
    ),
    notes="Rows for calibration and test cases (and their variants) of the concept-bottleneck model that drives or "
          "explains predictions (eval/model_choice.json). The contract contribution of a concept is the sum over its "
          "features; 08 reads this file and never refits a model.",
)

CELL_TILES = Table(
    "cell_tiles.csv", "Task 8 · 04a_select_tiles (inside cells.sbatch)", "Task 8 · 04b/04c; Task 11",
    key=("case_id", "tile_idx"),
    columns=(
        _CASE,
        _c("tile_idx", "int", "g512 (contract) tile index"),
        _SPLIT,
        _c("rank", "int", "1 = highest tile risk"),
        _c("tile_risk", "float", "out-of-fold for train, full-train fit for calibration/test (CELLS_SPEC §4)"),
        _c("path", "str", "tile PNG, data-root relative"),
        _c("n_tiles", "int", "tiles selected for the case (<= cells.tiles_per_case)"),
    ),
)

CELL_FEATURES = Table(
    "cell_features.csv", "Task 8 · 04c_cell_features", "Task 10 (B+C, C); Task 11 (contract cells block)",
    key=("case_id",),
    columns=(
        _CASE, _SPLIT,
        _c("n_tiles", "int", "segmented tiles"),
        _c("n_nuclei", "int", "typed nuclei over the tiles"),
        _c("n_neoplastic", "int", "neoplastic nuclei"),
        _c("insufficient_tumor_nuclei", "bool", "1 when n_neoplastic < cells.min_neoplastic_nuclei"),
    ),
    notes="Dynamic per feature f of config/cell_features.yaml: f (value in physical units), f__missing (0/1), "
          "f__ref_pct (percentile among train Normal/Benign cases, in [0, 1]).",
)

CELL_FEATURES_TILE = Table(
    "cell_features_tile.csv", "Task 8 · 04c_cell_features", "Task 11 (contract cells.tiles[].counts)",
    key=("case_id", "tile_idx"),
    columns=(
        _CASE,
        _c("tile_idx", "int", "g512 tile index"),
        _c("n_nuclei", "int", "typed nuclei in the tile"),
        *(_c(f"n_{t}", "int", f"{t} nuclei") for t in CELL_TYPES),
        _c("overlay", "str", "overlay PNG, data-root relative: cells/<case_id>/<tile_idx>_overlay.png"),
    ),
    notes="Plus the per-tile feature values f (same ids as cell_features.csv).",
)

TILE_CONTRIB = Table(
    "tile_contrib.csv", "05d_tile_contrib", "08 (contract tiles[].contribution, prediction.attribution)",
    key=("read_id", "tile_idx"),
    columns=(
        _c("read_id", "str", "predictions.csv case_id of the read (variants use their variant IDs)"),
        _c("model", "str", "model that produced the read (predictions.csv model)"),
        _c("tile_idx", "int", "region index on the contract grid (settings tiles.contract_grid)"),
        _c("contribution", "float", "region's additive term of the centered calibrated logit of `target`"),
        _c("constant", "float", "read constant: centered intercept / temperature (same on every row of a read)"),
        _c("target", "str", "class whose centered logit is decomposed (settings aggressive_class)"),
        _c("temperature", "float", "the read's temperature (predictions.csv)"),
        _c("n_rows", "int", "embedding rows pooled by the read"),
    ),
    notes="constant + sum of a read's contributions = log pcal__target - mean_k log pcal__k. Regions excluded from "
          "pooling (QC-failed) and reads whose model has no per-tile embeddings have no rows (contract null).",
)

TABLES = (
    CASES, TILES, EMBED_MANIFEST, EMB_INDEX, COMPOSITES, WSI_WINDOWS, CASE_FEATURES, TILE_RISK, TILE_RISK_OOF,
    PREDICTIONS, RELIABILITY, CONTRIBUTIONS, TILE_ATTENTION, CELL_TILES, CELL_FEATURES, CELL_FEATURES_TILE,
    TILE_CONTRIB,
)

# ---- JSON files -----------------------------------------------------------------------------------------------------
INVENTORY = JsonSpec(
    "inventory.json", "Task 2 · 00_inventory", "Task 2 · 01; core.config.effective_mpp; Task 8",
    required_keys=("created_utc", "synthetic", INVENTORY_MPP_KEY, "mpp_source", "label_source", "n_images",
                   "per_class_counts", "image_size", "bach_root", "patient_file", "wsi", "pathmmu_bach", "gate_passed"),
    doc="measured_mpp: float or null (null -> settings dataset.mpp); mpp_source: where it came from; per_class_counts: "
        "{class: int}; image_size: [w, h] most common; wsi: {n_svs, n_xml}; pathmmu_bach: comparison summary; "
        "gate_passed: >= dataset.min_labeled_per_class labeled images per class.",
)
EMB_META = JsonSpec(
    "emb/<encoder>/meta.json", "Task 3 · 02_embed", "Tasks 8-11",
    required_keys=("encoder", "model_card", "revision", "heads", "sets", "preprocess", "dtype", "created_utc",
                   "git_sha", "config_hash", "versions", "synthetic"),
    doc="heads: {head: {dim: int, normalized: bool}}; sets: {set: {n_items: int, n_views: int, views_file: bool}}; "
        "preprocess: {resize, crop, mean, std, input_px, mpp}; dtype: 'float16'; revision: resolved HF commit.",
)
RETRIEVAL = JsonSpec(
    "retrieval.json", "Task 10 · 07_retrieval", "Task 11 · 08",
    required_keys=("encoder", "head", "k", "cases", "created_utc", "config_hash", "synthetic"),
    doc="cases: {case_id: {neighbors: [{case_id, label, similarity}] (k train references, cosine), predicted, agree, "
        "of, neighbor_agreement (= agree / of, neighbor labels vs the predicted class), contrast: {case_id, label, "
        "similarity, differs_most_in: [concept ids]} or null}}.",
)
MARGINS_JSON = JsonSpec(
    "margins.json", "Task 10 · 05_risk_model (fine-scale probe)", "Task 11 · 08 (core.margin call) and margin eval",
    required_keys=("probe", "strip_px", "segment_px", "max_fpr", "threshold", "threshold_ci", "n_benign_cal_images",
                   "cases", "created_utc", "config_hash", "synthetic"),
    doc="threshold: core.margin.fit_threshold on benign calibration images; cases: {case_id: {width, height, "
        "segments: [{margin, x, y, size, p}] (core.margin.strip_segments order; corner squares appear once per "
        "adjacent margin), per_margin: {superior, inferior, medial, lateral: max segment p}}} for every surgical, "
        "composite and degraded-variant case.",
)
CELL_JSON = JsonSpec(
    "cells/<case_id>/<tile_idx>.json", "Task 8 · 04b_cells", "Task 8 · 04c; Task 11",
    required_keys=("case_id", "tile_idx", "backend", "model", "mpp_model", "mpp_source", "scale", "n_nuclei",
                   "n_untyped", "nuclei"),
    doc="nuclei: [{centroid_um: [x, y], area_um2, perimeter_um, eccentricity, solidity, type (CELL_TYPES), type_prob, "
        "edge: bool (truncated at the tile border), contour_px: [[x, y], ...] in original tile pixels}]; "
        "scale = model px / original px = mpp_source / mpp_model; µm use mpp_model (CELLS_SPEC §3, §5).",
)
STEP_META = JsonSpec(
    "meta/<step>.json", "every step (core.io.write_meta)", "humans; docs/RESULTS.md",
    required_keys=("step", "inputs", "outputs", "args", "created_utc", "git_sha", "git_dirty", "config_hash",
                   "versions", "platform", "synthetic"),
    doc="One per pipeline step run; args include --limit.",
)
SPOTCHECK_ITEMS = JsonSpec(
    "spotcheck/items.json", "Task 11 · 08 (Agent 4)", "Task 6 · backend /api/spotcheck (Agent 5)",
    required_keys=("created_utc", "synthetic", "items"),
    doc="ARTIFACTS-root file. items: [{item_id, kind: concept | cells, target: concept id or 'overlay', label: display "
        "name, case_id, tile_idx, image: artifacts-relative ref ('spotcheck/<file>.png' crop or "
        "'cells/<case_id>_<idx>_overlay.png'), score: float or null}], 20 per concept (plan §5.4) and 20 overlays "
        "(5 per class, CELLS_SPEC §8.1). Class labels are used only to pick items and are never written here. Reviews "
        "go to eval/spotcheck.json: {created_utc, synthetic, reviewer_kind: human | ai, reviews: [{item_id, reviewer, "
        "verdict: acceptable | not_acceptable | unsure, note, at}], summary: {concept: {<id>: {n, acceptable}}, "
        "cells: {n, acceptable}}}.",
)
JSON_SPECS = (INVENTORY, EMB_META, RETRIEVAL, MARGINS_JSON, CELL_JSON, STEP_META, SPOTCHECK_ITEMS)

# ---- arrays ---------------------------------------------------------------------------------------------------------
EMB_NPY = ArraySpec(
    "emb/<encoder>/<set>_<head>.npy", "Task 3 · 02_embed / export", "Tasks 8, 9, 10, 11",
    arrays=(("(array)", "float16 [N, D]; row i = emb_row i of <set>_<head>_index.csv"),),
    doc="One row per item_id (mean over n_views TTA views). Align-head rows are re-normalized to unit L2 norm after "
        "averaging (cosine = dot); probe rows are raw means. '_views' variant: one row per (item, view) for every "
        "split of bach512/bach256 (ruling M9). "
        "Never mix encoders in one index.",
)
TEXT_NPZ = ArraySpec(
    "text/<model>.npz", "Task 3 · 03_text", "Task 9 · 04_concepts and zero-shot",
    arrays=(
        ("ids", "str [P] prompt IDs (expand_prompts ids, plus nct_prompt_id for prompts_nct.json)"),
        ("texts", "str [P] exact text encoded"),
        ("emb", "float32 [P, D] L2-normalized text embeddings in the encoder's align space"),
        ("model", "str [] encoder key"),
        ("synthetic", "bool []"),
    ),
    doc="Look up by prompt ID: dict(zip(ids, emb)).",
)
CONCEPTS_TILE_NPZ = ArraySpec(
    "concepts_tile.npz", "Task 9 · 04_concepts", "Tasks 10, 11 (tiles[].concepts, top_tiles), Task 8 (validity)",
    arrays=(
        ("case_id", "str [N]"),
        ("tile_idx", "int32 [N] g512 tile index (QC-passing tiles)"),
        ("grid", "str [] 'g512'"),
        ("concept_ids", "str [C] settings concepts.ids order"),
        ("z", "float32 [N, C] final concept value per tile, z-scored on train tiles"),
        ("z_text", "float32 [N, C] text-prompt component"),
        ("z_cav", "float32 [N, C] concept-vector component; NaN where the concept is not caption-consistent"),
        ("spread", "float32 [N, C] std across phrasings (prompt-ensemble spread)"),
        ("encoder", "str []"),
        ("synthetic", "bool []"),
    ),
    doc="Rows cover every case of every split.",
)
ARRAY_SPECS = (EMB_NPY, TEXT_NPZ, CONCEPTS_TILE_NPZ)


# ---- frame validation ------------------------------------------------------------------------------------------------
def frame_problems(table: Table, df: Any, extra_required: Sequence[str] = ()) -> list[str]:
    """Problems of a pandas DataFrame against a Table spec: missing columns, empty or duplicate keys, values outside
    a column's vocabulary, non-numeric values in int/float columns, non-boolean values in bool columns."""
    import pandas as pd

    problems: list[str] = []
    present_cols = {str(c) for c in df.columns}
    missing = [c.name for c in table.columns if c.required and c.name not in present_cols]
    missing += [c for c in extra_required if c not in present_cols]
    if missing:
        problems.append(f"{table.filename}: missing columns {missing}")
    if all(k in present_cols for k in table.key) and len(df):
        key = list(table.key)
        if df[key].isna().any().any():
            problems.append(f"{table.filename}: empty values in key columns {key}")
        dup = df.duplicated(subset=key, keep=False)
        if dup.any():
            first = df.loc[dup, key].iloc[0].tolist()
            problems.append(f"{table.filename}: {int(dup.sum())} rows share a duplicate key {key}, e.g. {first}")
    for col in table.columns:
        if col.name not in present_cols:
            continue
        values = df[col.name]
        values = values[values.notna()]
        values = values[values.astype(str) != ""]
        if col.choices is not None:
            bad = sorted(set(values.astype(str)) - set(col.choices))
            if bad:
                problems.append(f"{table.filename}: column {col.name} has values outside {list(col.choices)}: {bad[:5]}")
        if col.kind in ("int", "float"):
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.isna().any():
                problems.append(f"{table.filename}: column {col.name} must be numeric, got "
                                f"{values[numeric.isna()].astype(str).tolist()[:3]}")
        elif col.kind == "bool":
            ok = values.astype(str).str.lower().isin(["0", "1", "true", "false", "0.0", "1.0"])
            if not ok.all():
                problems.append(f"{table.filename}: column {col.name} must be 0/1, got {values[~ok].astype(str).tolist()[:3]}")
    return problems


def validate_frame(table: Table, df: Any, extra_required: Sequence[str] = ()) -> None:
    """Raise ValueError listing every frame_problems() entry; use on write (producers) and read (consumers)."""
    problems = frame_problems(table, df, extra_required)
    if problems:
        raise ValueError("\n".join(problems))


# ---- names and IDs ----------------------------------------------------------------------------------------------------
def check_case_id(case_id: str) -> str:
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise ValueError(f"invalid case_id {case_id!r}: letters, digits, '_' or '-', starting with a letter or digit")
    return case_id


def check_name(name: str, what: str = "name") -> str:
    """Encoder, set, head, model, grid, step and eval names: lowercase letters, digits and '_'."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid {what} {name!r}: lowercase letters, digits and '_' only")
    return name


def _index(value: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} must be a non-negative integer, got {value!r}")
    return value


def variant_case_id(case_id: str, degrade_type: str, severity: int) -> str:
    """Degraded simulated-frozen-section variant of a test case: '<case_id>_FSIM_<type><severity>'."""
    if degrade_type not in DEGRADE_TYPES:
        raise ValueError(f"unknown degrade type {degrade_type!r}; expected one of {DEGRADE_TYPES}")
    if _index(severity, "severity") < 1:
        raise ValueError("severity must be >= 1 for a degraded variant")
    return f"{check_case_id(case_id)}_FSIM_{degrade_type}{severity}"


def composite_case_id(carcinoma_case_id: str, k: int) -> str:
    """Opaque ID of composite field k (global index): 'COMP_<k:03d>'.

    The carcinoma source case is validated but deliberately NOT embedded: '<case_id>_COMP_<k>' would tell anyone
    viewing the worklist that <case_id> is carcinoma (a ground-truth leak through the ID; master ruling M10). The
    sources live in composites.csv and the case JSON's source_case_ids (sorted, so the carcinoma half is not marked).
    """
    check_case_id(carcinoma_case_id)
    return f"COMP_{_index(k, 'k'):03d}"


def make_item_id(set_name: str, case_id: str, tile_idx: int | None = None, degrade_type: str | None = None,
                 severity: int | None = None) -> str:
    """Manifest item ID '<set>:<case_id>[:<tile_idx>][:<type><severity>]'. Readers treat it as opaque."""
    parts = [check_name(set_name, "set"), check_case_id(case_id)]
    if tile_idx is not None:
        parts.append(str(_index(tile_idx, "tile_idx")))
    if degrade_type:
        if degrade_type not in DEGRADE_TYPES:
            raise ValueError(f"unknown degrade type {degrade_type!r}")
        if severity is None or _index(severity, "severity") < 1:
            raise ValueError("a degraded item needs severity >= 1")
        parts.append(f"{degrade_type}{severity}")
    return ":".join(parts)


# ---- predictions.csv column names and set encoding -----------------------------------------------------------------------
def _class_name(cls: str) -> str:
    if not isinstance(cls, str) or not _CLASS_RE.fullmatch(cls) or "__" in cls:
        raise ValueError(f"invalid class name {cls!r}")
    return cls


def prob_col(model: str, cls: str) -> str:
    """Uncalibrated probability column 'p__<model>__<class>' (model names must not contain '__')."""
    if "__" in model:
        raise ValueError(f"model name {model!r} must not contain '__'")
    return f"p__{check_name(model, 'model')}__{_class_name(cls)}"


def cal_prob_col(cls: str) -> str:
    """Calibrated production probability column 'pcal__<class>'."""
    return f"pcal__{_class_name(cls)}"


def format_set(members: Sequence[str], classes: Sequence[str]) -> str:
    """Encode a conformal set as classes joined by '|' in settings.classes order; '' = empty set."""
    unknown = [m for m in members if m not in classes]
    if unknown:
        raise ValueError(f"set members {unknown} are not classes {list(classes)}")
    return "|".join(c for c in classes if c in members)


def parse_set(text: Any) -> list[str]:
    """Inverse of format_set; '' / None / NaN (pandas reads '' as NaN) = empty set."""
    if text is None or (isinstance(text, float) and math.isnan(text)) or text == "":
        return []
    return str(text).split("|")


def concept_feature_columns(cfg: Mapping[str, Any]) -> list[str]:
    """The concept-bottleneck model features in case_features.csv: c__top<k>, c__frac per concept id."""
    k = int(cfg["concepts"]["top_k"])
    return [col for cid in cfg["concepts"]["ids"] for col in (f"{cid}__top{k}", f"{cid}__frac")]


def concept_spread_columns(cfg: Mapping[str, Any]) -> list[str]:
    return [f"{cid}__spread" for cid in cfg["concepts"]["ids"]]


# ---- data-root paths ---------------------------------------------------------------------------------------------------------
def resolve(root: str | Path, stored: str) -> Path:
    """Path of a path stored in a file: absolute as is, else relative to `root`."""
    p = Path(stored)
    return p if p.is_absolute() else Path(root) / p


def inventory_json(root: str | Path) -> Path:
    return Path(root) / "inventory.json"


def cases_csv(root: str | Path) -> Path:
    return Path(root) / "cases.csv"


def tiles_csv(root: str | Path) -> Path:
    return Path(root) / "tiles.csv"


def embed_manifest_csv(root: str | Path) -> Path:
    return Path(root) / "embed_manifest.csv"


def composites_csv(root: str | Path) -> Path:
    return Path(root) / "composites.csv"


def wsi_windows_csv(root: str | Path) -> Path:
    return Path(root) / "wsi_windows.csv"


def tile_png(root: str | Path, grid: str, case_id: str, row: int, col: int) -> Path:
    return Path(root) / "tiles" / check_name(grid, "grid") / check_case_id(case_id) / f"{_index(row, 'row')}_{_index(col, 'col')}.png"


def display_jpg(root: str | Path, case_id: str) -> Path:
    return Path(root) / "display" / f"{check_case_id(case_id)}.jpg"


def emb_dir(root: str | Path, encoder: str) -> Path:
    return Path(root) / "emb" / check_name(encoder, "encoder")


def emb_npy(root: str | Path, encoder: str, set_name: str, head: str, *, views: bool = False) -> Path:
    suffix = "_views" if views else ""
    return emb_dir(root, encoder) / f"{check_name(set_name, 'set')}_{check_name(head, 'head')}{suffix}.npy"


def emb_index_csv(root: str | Path, encoder: str, set_name: str, head: str, *, views: bool = False) -> Path:
    suffix = "_views" if views else ""
    return emb_dir(root, encoder) / f"{check_name(set_name, 'set')}_{check_name(head, 'head')}{suffix}_index.csv"


def emb_meta_json(root: str | Path, encoder: str) -> Path:
    return emb_dir(root, encoder) / "meta.json"


def text_npz(root: str | Path, model: str) -> Path:
    return Path(root) / "text" / f"{check_name(model, 'model')}.npz"


def concepts_tile_npz(root: str | Path) -> Path:
    return Path(root) / "concepts_tile.npz"


def case_features_csv(root: str | Path) -> Path:
    return Path(root) / "case_features.csv"


def tile_risk_csv(root: str | Path, *, oof: bool = False) -> Path:
    return Path(root) / ("tile_risk_oof.csv" if oof else "tile_risk.csv")


def predictions_csv(root: str | Path) -> Path:
    return Path(root) / "predictions.csv"


def retrieval_json(root: str | Path) -> Path:
    return Path(root) / "retrieval.json"


def reliability_csv(root: str | Path) -> Path:
    return Path(root) / "reliability.csv"


def margins_json(root: str | Path) -> Path:
    return Path(root) / "margins.json"


def contributions_csv(root: str | Path) -> Path:
    return Path(root) / "contributions.csv"


def tile_attention_csv(root: str | Path) -> Path:
    return Path(root) / "tile_attention.csv"


def tile_contrib_csv(root: str | Path) -> Path:
    return Path(root) / "tile_contrib.csv"


def cell_tiles_csv(root: str | Path) -> Path:
    return Path(root) / "cell_tiles.csv"


def cell_json(root: str | Path, case_id: str, tile_idx: int) -> Path:
    return Path(root) / "cells" / check_case_id(case_id) / f"{_index(tile_idx, 'tile_idx')}.json"


def cell_overlay_png(root: str | Path, case_id: str, tile_idx: int) -> Path:
    return Path(root) / "cells" / check_case_id(case_id) / f"{_index(tile_idx, 'tile_idx')}_overlay.png"


def cell_features_csv(root: str | Path, *, per_tile: bool = False) -> Path:
    return Path(root) / ("cell_features_tile.csv" if per_tile else "cell_features.csv")


def step_meta_json(root: str | Path, step: str) -> Path:
    return Path(root) / "meta" / f"{check_name(step, 'step')}.json"


# ---- artifacts-root paths and contract references ---------------------------------------------------------------------------
def case_json(artifacts_root: str | Path, case_id: str) -> Path:
    """Case JSON (with eval_only; the API strips it through core.contract.CasePublic)."""
    return Path(artifacts_root) / "cases" / f"{check_case_id(case_id)}.json"


def case_image(artifacts_root: str | Path, case_id: str) -> Path:
    """Display JPEG for a case or a train reference case; contract ref display_ref(case_id)."""
    return Path(artifacts_root) / "images" / f"{check_case_id(case_id)}.jpg"


def case_cell_overlay(artifacts_root: str | Path, case_id: str, tile_idx: int) -> Path:
    return Path(artifacts_root) / "cells" / f"{check_case_id(case_id)}_{_index(tile_idx, 'tile_idx')}_overlay.png"


def eval_json(artifacts_root: str | Path, name: str) -> Path:
    if name not in EVAL_FILES:
        raise ValueError(f"unknown eval file {name!r}; register it in core.schemas.EVAL_FILES first")
    return Path(artifacts_root) / "eval" / f"{name}.json"


def eval_asset(artifacts_root: str | Path, filename: str) -> Path:
    """Non-JSON eval outputs (PNG figures, cells_spike.md) under <artifacts_root>/eval/."""
    if not isinstance(filename, str) or not _FILE_RE.fullmatch(filename) or ".." in filename:
        raise ValueError(f"invalid eval asset name {filename!r}")
    return Path(artifacts_root) / "eval" / filename


def spotcheck_items_json(artifacts_root: str | Path) -> Path:
    """Spot-check review items (SPOTCHECK_ITEMS); crops live beside it under <artifacts_root>/spotcheck/."""
    return Path(artifacts_root) / "spotcheck" / "items.json"


def reference_index_json(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root) / "reference_index.json"


def display_ref(case_id: str) -> str:
    """Contract image reference (image.display, similar_cases[].image), relative to the artifacts root."""
    return f"images/{check_case_id(case_id)}.jpg"


def cell_overlay_ref(case_id: str, tile_idx: int) -> str:
    """Contract reference for cells.tiles[].overlay, relative to the artifacts root."""
    return f"cells/{check_case_id(case_id)}_{_index(tile_idx, 'tile_idx')}_overlay.png"


def fixture_case_json(fixtures_root: str | Path, case_id: str) -> Path:
    """fixtures/ is flat: <fixtures>/<case_id>.json."""
    return Path(fixtures_root) / f"{check_case_id(case_id)}.json"


def fixture_media(fixtures_root: str | Path, ref: str) -> Path:
    """A contract media ref ('images/x.jpg', 'cells/y.png') resolved by basename in the flat fixtures folder."""
    if not isinstance(ref, str) or not _MEDIA_REF_RE.fullmatch(ref) or ".." in ref:
        raise ValueError(f"invalid media ref {ref!r}: expected images/<file> or cells/<file>")
    return Path(fixtures_root) / PurePosixPath(ref).name


# ---- config/concepts.yaml schema (ruling R1) -----------------------------------------------------------------------------------
# Top level (owned by Task 5; validated by validate_concept_bank, loaded by core.config.load_concept_bank):
#
#   templates:      [str]   each with exactly one "{}" placeholder and no other braces, unique, >= 1
#                           e.g. "an H&E image of {}." (CONCH and MUSK styles; a phrasing fills the "{}")
#   concepts:       [concept]   exactly the settings concepts.ids, each once:
#     id:           str     one of settings concepts.ids
#     group:        str     the group settings concepts.groups assigns to that id
#     label:        str     UI display label, e.g. "Tumor infiltrating stroma"
#     direction:    up | down | context    expected direction vs the aggressive class (sanity reference only)
#     weak:         bool    true for signals expected to be weak at tile scale (mitoses)
#     positives:    [str]   affirmative phrasings, count within settings concepts.positives [min, max]
#     contrasts:    [str]   affirmative contrast phrasings (never negations), count within concepts.contrasts
#   class_prompts:  {class: [str]}   keys exactly settings.classes; zero-shot class phrasings (>= 1 each)
#   version, notes: optional metadata
#
# Phrasings must be non-empty, unique within their list, and contain no "{" or "}".
# Prompt IDs (keys of text/<model>.npz): see expand_prompts().
CONCEPT_KEYS = ("id", "group", "label", "direction", "weak", "positives", "contrasts")
BANK_REQUIRED = ("templates", "concepts", "class_prompts")
BANK_OPTIONAL = ("version", "notes")


@dataclass(frozen=True)
class Concept:
    id: str
    group: str
    label: str
    direction: str
    weak: bool
    positives: tuple[str, ...]
    contrasts: tuple[str, ...]


@dataclass(frozen=True)
class ConceptBank:
    templates: tuple[str, ...]
    concepts: tuple[Concept, ...]  # settings concepts.ids order
    class_prompts: Mapping[str, tuple[str, ...]]  # settings.classes order

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.concepts]

    def by_id(self) -> dict[str, Concept]:
        return {c.id: c for c in self.concepts}


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    kind: str          # concept | class
    target: str        # concept id or class name
    polarity: str      # pos | con for concepts, "" for classes
    phrasing_idx: int
    template_idx: int


def _phrases(value: Any, where: str, bounds: Sequence[int] | None, problems: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        problems.append(f"{where} must be a list of non-empty strings")
        return ()
    if len(set(value)) != len(value):
        problems.append(f"{where} has duplicate phrasings")
    if bounds is not None and not (bounds[0] <= len(value) <= bounds[1]):
        problems.append(f"{where}: {len(value)} phrasings, expected {bounds[0]}-{bounds[1]}")
    braces = [v for v in value if "{" in v or "}" in v]
    if braces:
        problems.append(f"{where}: phrasings must not contain '{{' or '}}': {braces}")
    return tuple(value)


def validate_concept_bank(data: Any, cfg: Mapping[str, Any]) -> ConceptBank:
    """Validate parsed concepts.yaml against the schema above and settings; raise ValueError listing every problem."""
    problems: list[str] = []
    if not isinstance(data, Mapping):
        raise ValueError("concepts.yaml: top level must be a mapping")
    missing = [k for k in BANK_REQUIRED if k not in data]
    extra = [k for k in data if k not in BANK_REQUIRED + BANK_OPTIONAL]
    if missing:
        problems.append(f"concepts.yaml: missing top-level keys {missing}")
    if extra:
        problems.append(f"concepts.yaml: unexpected top-level keys {extra}")

    templates = data.get("templates")
    if not isinstance(templates, list) or not templates or not all(isinstance(t, str) for t in templates):
        problems.append("templates must be a non-empty list of strings")
        templates = []
    for i, template in enumerate(templates):
        if template.count("{}") != 1 or template.replace("{}", "").count("{") or template.replace("{}", "").count("}"):
            problems.append(f"templates[{i}] must contain exactly one '{{}}' placeholder and no other braces: {template!r}")
    if len(set(templates)) != len(templates):
        problems.append("templates must be unique")

    ids = list(cfg["concepts"]["ids"])
    group_of = {cid: name for name, spec in cfg["concepts"]["groups"].items() for cid in spec["ids"]}
    pos_bounds, con_bounds = cfg["concepts"]["positives"], cfg["concepts"]["contrasts"]
    parsed: dict[str, Concept] = {}
    raw_concepts = data.get("concepts")
    if not isinstance(raw_concepts, list):
        problems.append("concepts must be a list")
        raw_concepts = []
    for i, item in enumerate(raw_concepts):
        if not isinstance(item, Mapping):
            problems.append(f"concepts[{i}] must be a mapping")
            continue
        where = f"concepts[{i}] ({item.get('id', '?')})"
        missing_keys = [k for k in CONCEPT_KEYS if k not in item]
        extra_keys = [k for k in item if k not in CONCEPT_KEYS]
        if missing_keys:
            problems.append(f"{where}: missing keys {missing_keys}")
        if extra_keys:
            problems.append(f"{where}: unexpected keys {extra_keys}")
        cid = item.get("id")
        if cid not in ids:
            problems.append(f"{where}: unknown concept id {cid!r}; settings concepts.ids are {ids}")
            continue
        if cid in parsed:
            problems.append(f"{where}: concept id {cid!r} appears twice")
            continue
        if item.get("group") != group_of[cid]:
            problems.append(f"{where}: group {item.get('group')!r} does not match settings group {group_of[cid]!r}")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            problems.append(f"{where}: label must be a non-empty string")
        if item.get("direction") not in CONCEPT_DIRECTIONS:
            problems.append(f"{where}: direction must be one of {list(CONCEPT_DIRECTIONS)}, got {item.get('direction')!r}")
        if not isinstance(item.get("weak"), bool):
            problems.append(f"{where}: weak must be true or false, got {item.get('weak')!r}")
        positives = _phrases(item.get("positives"), f"{where} positives", pos_bounds, problems)
        contrasts = _phrases(item.get("contrasts"), f"{where} contrasts", con_bounds, problems)
        parsed[cid] = Concept(cid, str(item.get("group")), str(label), str(item.get("direction")),
                              bool(item.get("weak")), positives, contrasts)
    absent = [cid for cid in ids if cid not in parsed]
    if absent and raw_concepts:
        problems.append(f"concepts: missing concepts {absent}")

    classes = list(cfg["classes"])
    raw_prompts = data.get("class_prompts")
    class_prompts: dict[str, tuple[str, ...]] = {}
    if not isinstance(raw_prompts, Mapping):
        problems.append("class_prompts must be a mapping {class: [phrasings]}")
    else:
        if sorted(raw_prompts) != sorted(classes):
            problems.append(f"class_prompts keys must be exactly the classes {classes}; got {sorted(raw_prompts)}")
        for cls in classes:
            if cls in raw_prompts:
                class_prompts[cls] = _phrases(raw_prompts[cls], f"class_prompts.{cls}", None, problems)

    if problems:
        raise ValueError("concepts.yaml is invalid:\n- " + "\n- ".join(problems))
    return ConceptBank(tuple(templates), tuple(parsed[cid] for cid in ids), class_prompts)


def concept_prompt_id(concept_id: str, polarity: str, phrasing_idx: int, template_idx: int) -> str:
    if polarity not in ("pos", "con"):
        raise ValueError(f"polarity must be 'pos' or 'con', got {polarity!r}")
    return f"concept:{concept_id}:{polarity}:{phrasing_idx}:{template_idx}"


def class_prompt_id(class_name: str, phrasing_idx: int, template_idx: int) -> str:
    return f"class:{class_name}:{phrasing_idx}:{template_idx}"


def nct_prompt_id(nct_class: str) -> str:
    """Organizer NCT prompts (resources/official/prompts_nct.json) are complete sentences, encoded untemplated."""
    return f"nct:{nct_class}"


def expand_prompts(bank: ConceptBank) -> list[Prompt]:
    """Every (phrasing x template) prompt, in a fixed order: per concept its positives then its contrasts, then the
    class prompts per class; within a phrasing, templates in order. Writers (03_text) and readers (04_concepts) both
    use this, so prompt IDs never drift. Text = template with "{}" replaced by the phrasing."""
    prompts: list[Prompt] = []
    for concept in bank.concepts:
        for polarity, phrasings in (("pos", concept.positives), ("con", concept.contrasts)):
            for i, phrase in enumerate(phrasings):
                for t, template in enumerate(bank.templates):
                    prompts.append(Prompt(concept_prompt_id(concept.id, polarity, i, t), template.replace("{}", phrase),
                                          "concept", concept.id, polarity, i, t))
    for cls, phrasings in bank.class_prompts.items():
        for i, phrase in enumerate(phrasings):
            for t, template in enumerate(bank.templates):
                prompts.append(Prompt(class_prompt_id(cls, i, t), template.replace("{}", phrase), "class", cls, "", i, t))
    return prompts
