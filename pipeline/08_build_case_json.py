"""Step 08: assemble one contract case JSON per test case and simulated surgical variant (plan §6.10, §7).

    python -m pipeline.08_build_case_json [--limit N] [--max-llm-calls N] [--llm-agent] [--force] [--clean]

Laptop step. It NEVER refits a model: every number comes from the producers' files (core.schemas):
    cases.csv, tiles.csv, composites.csv (Agent 1)        case_features.csv, concepts_tile.npz (Agent 2)
    predictions.csv, tile_risk.csv, contributions.csv,      cell_features.csv, cell_features_tile.csv,
    tile_attention.csv (optional), reliability.csv,         cells/<case>/<idx>_overlay.png (Agent 1, optional)
    retrieval.json, margins.json (Agent 3)                  config/concepts.yaml, config/cell_features.yaml
A missing required input, a clean test case missing from one, or a blank/NaN value in a required field (flags,
probabilities, contributions, tile scores) is an error — blanks never become False/0. A simulated variant that fails
any of these is skipped and listed in meta/08_build_case_json.json ("skipped"). Every case is validated against the
contract before any LLM call or write. Retrieval references must be train-split cases.

Cases written (artifacts root):
  * every TEST case (lane from cases.csv: biopsy or surgical);
  * degraded surgical variants: every test-surgical case x every settings degrade.types at degrade.display_severity,
    id schemas.variant_case_id, simulated_variant "degraded", source_case_ids [source];
  * composite surgical variants: every composites.csv row, id = its case_id, source_case_ids [carcinoma, benign].
  Variants are meant for on-demand injection by the simulator; their arrival offsets come after every clean case.

Blocks follow core.contract exactly (fixtures/demo_0001.json, demo_0002.json):
  quality      qc_tiles_kept = QC-passing g512 tiles; ood_percentile, reliability, flag from reliability.csv
  prediction   calibrated probs (pcal__<class>), risk, settings alpha, tier set (predictions.csv set_tier, else set_lac)
  concepts     per concept: value (case_features c__top<k>), value_std (contributions.csv <c>__top<k> value_std),
               contribution = sum of its contributions.csv rows,
               expected direction/label (concept bank), spread (c__spread), top_tiles (top-k tiles by z with
               z >= concepts.z_threshold, at least one), source/components/caption_consistent; measured cell
               features (contributions.csv source "cell") appear as source "cell" entries
  tiles        QC g512 tiles: risk (tile_risk.csv), concept z (concepts_tile.npz), optional ABMIL attention
  similar_cases / contrast_case / agreement   retrieval.json (train reference labels may be shown)
  margin       surgical only: margins.json segments + per_margin, core.margin.suggest_call(per_margin, threshold,
               reliability_flag), threshold (kept for defer calls too, ruling M8), strip_um = margin.strip_px x core.config.effective_mpp, involved_length_um
  cells        CELLS_SPEC §7.2 block, or {"status": "disabled"} (cells.enabled false or the module did not run)
  flag         core.tiers.flag_for(case, cfg) with now=None (static)
  findings     F1 prediction; concepts (|contribution| >= findings.min_abs_contribution, top findings.max_concepts);
               retrieval; quality; cells (CELLS_SPEC §7.2 extreme-percentile / non-trivial-contribution rule); margin
  summary      backend.summary.generate (template unless --max-llm-calls > 0 and a Lumen key exists)
  agent        agent.agent.draft (rules; LLM rationales only with --llm-agent and budget)
  eval_only    true_label, split, source ids, group (+ composite margin truth) — the ONLY place ground truth appears

Arrivals: a seeded Poisson process (settings sim.arrival) over a seeded order of the clean cases; hero cases (settings
demo.heroes, picked from these cases; labels may break ties offline when demo.prefer_correct): the biopsy hero arrives
last of all clean cases; surgical heroes take the earliest offsets of their kind, because the backend holds surgical
cases for injection and injects the earliest best match (see arrival_offsets). Metrics never depend on this.

Also writes artifacts/images/*.jpg (every written case and every referenced train case), artifacts/cells/*.png,
artifacts/reference_index.json and artifacts/spotcheck/{items.json, *.png} (settings spotcheck).

Outputs are rewritten only when their content changes (so a rerun after upstream changes never leaves stale text);
--force rewrites everything; --clean removes case JSONs in artifacts/cases/ that this run did not produce.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agent import agent as agent_mod
from agent import rules as rules_mod
from agent.lumen import LumenClient
from backend import summary as summary_mod
from core import cli, config, contract, io, margin, schemas, tiers

STEP = "08_build_case_json"
CELL_FEATURES_YAML = "config/cell_features.yaml"   # Agent 1's file (requested settings key cells.features_file)
CELL_DIRECTIONS = {"higher": "up", "lower": "down", "context": "context", "up": "up", "down": "down"}


# ---- small helpers -----------------------------------------------------------------------------------------------
def rnd(x: Any, n: int) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, n)


def is_missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x == "")


class InputError(ValueError):
    """Invalid or missing input for step 08."""


def as_bool(x: Any, what: str = "flag") -> bool:
    """0/1/true/false -> bool; a missing value raises (a blank flag must never read as False)."""
    if is_missing(x):
        raise InputError(f"{what} is missing")
    if isinstance(x, str):
        text = x.strip().lower()
        if text not in ("0", "1", "true", "false", "0.0", "1.0"):
            raise InputError(f"{what} must be 0/1, got {x!r}")
        return text in ("1", "true", "1.0")
    return bool(x)


def req_float(x: Any, what: str) -> float:
    if is_missing(x):
        raise InputError(f"{what} is missing")
    value = float(x)
    if not math.isfinite(value):
        raise InputError(f"{what} is not finite ({value})")
    return value


def sum_contrib(rows: pd.DataFrame, what: str) -> float:
    values = pd.to_numeric(rows["contribution"], errors="coerce")
    if values.isna().any():
        raise InputError(f"{what}: missing contribution values")
    return float(values.sum())


def _need(path: Path, what: str) -> Path:
    if not path.is_file():
        raise InputError(f"missing input {what}: {path}")
    return path


def _read_table(table: schemas.Table, path: Path, extra: Sequence[str] = ()) -> pd.DataFrame:
    df = io.read_csv(_need(path, table.filename), keep_default_na=True)
    schemas.validate_frame(table, df, extra)
    return df


# ---- inputs ----------------------------------------------------------------------------------------------------------
@dataclass
class CellInputs:
    features: pd.DataFrame                    # cell_features.csv indexed by case_id
    tiles: pd.DataFrame                       # cell_features_tile.csv
    meta: dict[str, dict[str, Any]]           # feature id -> {label, unit, direction}


@dataclass
class Inputs:
    cases: pd.DataFrame                       # indexed by case_id
    tiles: pd.DataFrame                       # g512 rows
    predictions: pd.DataFrame                 # indexed by case_id
    reliability: pd.DataFrame                 # indexed by case_id
    contributions: pd.DataFrame
    case_features: pd.DataFrame | None        # indexed by case_id
    tile_risk: pd.DataFrame                   # g512 rows
    tile_attention: pd.DataFrame | None
    concepts_tile: dict[str, dict[int, dict[str, np.ndarray]]]   # case -> tile -> {z, z_text, z_cav, spread}
    concept_ids: list[str]
    retrieval: dict[str, Any]
    margins: dict[str, Any] | None
    composites: pd.DataFrame | None
    cells: CellInputs | None
    bank: schemas.ConceptBank
    concepts_eval: dict[str, Any]
    inputs_used: list[str] = field(default_factory=list)


def load_cell_meta(path: Path) -> dict[str, dict[str, Any]]:
    """Feature id -> {label, unit, direction} from config/cell_features.yaml.

    Accepts `features:` as a list of {id, label|display_name, unit, direction} or a mapping {id: {...}}
    (direction higher|lower|context, or up|down|context)."""
    data = yaml.safe_load(_need(path, "cell_features.yaml").read_text(encoding="utf-8"))
    raw = data.get("features") if isinstance(data, Mapping) else None
    if isinstance(raw, Mapping):
        raw = [{"id": k, **(v or {})} for k, v in raw.items()]
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{path}: expected a non-empty `features` list or mapping")
    out: dict[str, dict[str, Any]] = {}
    for item in raw:
        fid = item.get("id")
        label = item.get("label") or item.get("display_name")
        direction = CELL_DIRECTIONS.get(str(item.get("direction", "")).lower())
        if not fid or not label or direction is None:
            raise InputError(f"{path}: feature {item!r} needs id, label/display_name and direction")
        out[str(fid)] = {"label": str(label), "unit": str(item.get("unit") or ""), "direction": direction}
    return out


def load_inputs(cfg: Mapping[str, Any], root: Path, artifacts: Path, *, concept_bank_path: Path | None = None,
                cell_features_path: Path | None = None) -> Inputs:
    used: list[str] = []

    def track(p: Path) -> Path:
        used.append(str(p))
        return p

    grid = cfg["tiles"]["contract_grid"]
    cases = _read_table(schemas.CASES, track(schemas.cases_csv(root))).set_index("case_id", drop=False)
    tiles = _read_table(schemas.TILES, track(schemas.tiles_csv(root)))
    tiles = tiles[tiles["grid"] == grid]
    classes = list(cfg["classes"])
    predictions = _read_table(schemas.PREDICTIONS, track(schemas.predictions_csv(root)),
                              [schemas.cal_prob_col(c) for c in classes]).set_index("case_id", drop=False)
    reliability = _read_table(schemas.RELIABILITY, track(schemas.reliability_csv(root))).set_index("case_id", drop=False)
    contributions = _read_table(schemas.CONTRIBUTIONS, track(schemas.contributions_csv(root)))
    tile_risk = _read_table(schemas.TILE_RISK, track(schemas.tile_risk_csv(root)))
    tile_risk = tile_risk[tile_risk["grid"] == grid]
    case_features = None
    if schemas.case_features_csv(root).is_file():
        case_features = _read_table(schemas.CASE_FEATURES, track(schemas.case_features_csv(root))).set_index(
            "case_id", drop=False)
    tile_attention = None
    if schemas.tile_attention_csv(root).is_file():
        tile_attention = _read_table(schemas.TILE_ATTENTION, track(schemas.tile_attention_csv(root)))
        tile_attention = tile_attention[tile_attention["grid"] == grid]

    npz = io.load_npz(_need(track(schemas.concepts_tile_npz(root)), "concepts_tile.npz"))
    problems = schemas.CONCEPTS_TILE_NPZ.problems(npz)
    if problems:
        raise InputError("; ".join(problems))
    concept_ids = [str(c) for c in npz["concept_ids"]]
    if concept_ids != list(cfg["concepts"]["ids"]):
        raise InputError(f"concepts_tile.npz concept_ids {concept_ids} differ from settings concepts.ids")
    concepts_tile: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for i, (cid, tidx) in enumerate(zip(npz["case_id"], npz["tile_idx"])):
        per_case = concepts_tile.setdefault(str(cid), {})
        if int(tidx) in per_case:
            raise InputError(f"concepts_tile.npz has a duplicate row for ({cid}, {int(tidx)})")
        per_case[int(tidx)] = {k: npz[k][i] for k in ("z", "z_text", "z_cav", "spread")}

    retrieval = io.read_json(_need(track(schemas.retrieval_json(root)), "retrieval.json"))
    problems = schemas.RETRIEVAL.problems(retrieval)
    margins = None
    if schemas.margins_json(root).is_file():
        margins = io.read_json(track(schemas.margins_json(root)))
        problems += schemas.MARGINS_JSON.problems(margins)
        for key in ("strip_px", "segment_px"):
            if key in margins and int(margins[key]) != int(cfg["margin"][key]):
                problems.append(f"margins.json {key}={margins[key]} differs from settings margin.{key}={cfg['margin'][key]}")
    if problems:
        raise InputError("; ".join(problems))
    composites = None
    if schemas.composites_csv(root).is_file():
        composites = _read_table(schemas.COMPOSITES, track(schemas.composites_csv(root)))

    cells = None
    cell_path = schemas.cell_features_csv(root)
    if cfg["cells"]["enabled"] and cell_path.is_file():
        meta = load_cell_meta(cell_features_path or config.resolve_repo_path(CELL_FEATURES_YAML))
        cell_feats = _read_table(schemas.CELL_FEATURES, track(cell_path)).set_index("case_id", drop=False)
        cell_tiles = _read_table(schemas.CELL_FEATURES_TILE, track(schemas.cell_features_csv(root, per_tile=True)))
        missing = [f for f in meta if f not in cell_feats.columns]
        if missing:
            raise InputError(f"cell_features.csv lacks features {missing} listed in cell_features.yaml")
        cells = CellInputs(cell_feats, cell_tiles, meta)

    bank = config.load_concept_bank(concept_bank_path, cfg)
    concepts_eval: dict[str, Any] = {}
    eval_path = schemas.eval_json(artifacts, "concepts")
    if eval_path.is_file():
        concepts_eval = io.read_json(track(eval_path))
    return Inputs(cases, tiles, predictions, reliability, contributions, case_features, tile_risk, tile_attention,
                  concepts_tile, concept_ids, retrieval, margins, composites, cells, bank, concepts_eval, used)


# ---- the case plan ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    lane: str
    variant: str | None                 # None | degraded | composite
    degradation: tuple[str, int] | None
    source_ids: tuple[str, ...]
    qc_source: str | None               # case whose tiles.csv QC applies (itself, the degraded source, or None)
    display_src: Path
    width: int
    height: int
    true_label: str
    split: str
    group: str
    extra_eval: tuple[tuple[str, Any], ...] = ()


def case_plan(inputs: Inputs, cfg: Mapping[str, Any], root: Path, limit: int | None) -> list[CaseSpec]:
    cases = inputs.cases
    test = cases[cases["split"] == "test"]
    ids = cli.limit_case_ids(test["case_id"], limit)
    specs: list[CaseSpec] = []
    for cid in ids:
        row = cases.loc[cid]
        specs.append(CaseSpec(cid, str(row["lane"]), None, None, (), cid, schemas.resolve(root, row["display_path"]),
                              int(row["width"]), int(row["height"]), str(row["label"]), "test", str(row["group"])))
    severity = int(cfg["degrade"]["display_severity"])
    for cid in ids:
        row = cases.loc[cid]
        if row["lane"] != "surgical":
            continue
        for dtype in cfg["degrade"]["types"]:
            vid = schemas.variant_case_id(cid, dtype, severity)
            specs.append(CaseSpec(vid, "surgical", "degraded", (dtype, severity), (cid,), cid,
                                  schemas.display_jpg(root, vid), int(row["width"]), int(row["height"]),
                                  str(row["label"]), "test", str(row["group"])))
    if inputs.composites is not None:
        test_all = set(test["case_id"])
        for _, comp in inputs.composites.sort_values("case_id").iterrows():
            carc, benign = str(comp["carcinoma_case_id"]), str(comp["benign_case_id"])
            if carc not in ids:
                if carc not in test_all:
                    raise InputError(f"composite {comp['case_id']}: carcinoma case {carc} is not a test case")
                continue                                   # left out by --limit
            if benign not in test_all:
                raise InputError(f"composite {comp['case_id']}: benign case {benign} is not a test case")
            truth_values = {m: comp[f"truth_{m}"] for m in schemas.MARGINS}
            if any(is_missing(v) for v in truth_values.values()):
                raise InputError(f"composite {comp['case_id']}: missing margin truth")
            truth = tuple((f"margin_truth_{m}", str(v)) for m, v in truth_values.items())
            groups = sorted({str(cases.loc[c, "group"]) for c in (carc, benign)})
            # Top-level source ids are sorted so their order never tells which half is carcinoma (eval_only does).
            specs.append(CaseSpec(str(comp["case_id"]), "surgical", "composite", None, tuple(sorted((carc, benign))), None,
                                  schemas.resolve(root, comp["display_path"]), int(comp["width"]), int(comp["height"]),
                                  str(cases.loc[carc, "label"]), "test", "+".join(groups),
                                  (("carcinoma_case_id", carc), ("benign_case_id", benign),
                                   ("carcinoma_side", str(comp["carcinoma_side"])), *truth)))
    return specs


# ---- blocks ------------------------------------------------------------------------------------------------------------
def qc_tiles(spec: CaseSpec, inputs: Inputs) -> list[int]:
    if spec.qc_source is not None:
        rows = inputs.tiles[(inputs.tiles["case_id"] == spec.qc_source)]
        if rows.empty:
            raise InputError(f"{spec.case_id}: no g512 tiles for {spec.qc_source} in tiles.csv")
        keep = [as_bool(v, f"{spec.qc_source} tiles.csv qc_pass") for v in rows["qc_pass"]]
        return sorted(int(t) for t, k in zip(rows["tile_idx"], keep) if k)
    rows = inputs.tile_risk[inputs.tile_risk["case_id"] == spec.case_id]
    if rows.empty:
        raise InputError(f"{spec.case_id}: no g512 rows in tile_risk.csv")
    return sorted(int(t) for t in rows["tile_idx"])


def prediction_block(spec: CaseSpec, inputs: Inputs, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if spec.case_id not in inputs.predictions.index:
        raise InputError(f"{spec.case_id}: missing from predictions.csv")
    row = inputs.predictions.loc[spec.case_id]
    classes = list(cfg["classes"])
    probs = [req_float(row[schemas.cal_prob_col(c)], f"{spec.case_id} {schemas.cal_prob_col(c)}") for c in classes]
    total = sum(probs)
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise InputError(f"{spec.case_id}: calibrated probabilities sum to {total}")
    probs = [round(p / total, 4) for p in probs]
    risk = req_float(row["risk"], f"{spec.case_id} predictions.csv risk")
    aggressive = req_float(row[schemas.cal_prob_col(cfg["aggressive_class"])], f"{spec.case_id} pcal aggressive")
    if abs(risk - aggressive) > 1e-3:
        raise InputError(f"{spec.case_id}: risk {risk} differs from {schemas.cal_prob_col(cfg['aggressive_class'])} {aggressive}")
    set_col, size_col = ("set_tier", "set_tier_size") if "set_tier" in row.index else ("set_lac", "set_size")
    members = schemas.parse_set(row[set_col])
    if len(members) != int(req_float(row[size_col], f"{spec.case_id} {size_col}")):
        raise InputError(f"{spec.case_id}: {set_col} {members} does not match {size_col} {row[size_col]}")
    return {"model": str(row["model"]), "classes": classes, "probs": probs,
            "risk_score": round(risk, 4), "conformal_alpha": float(cfg["alpha"]), "conformal_set": members}


def quality_block(spec: CaseSpec, inputs: Inputs, n_tiles: int) -> dict[str, Any]:
    if spec.case_id not in inputs.reliability.index:
        raise InputError(f"{spec.case_id}: missing from reliability.csv")
    row = inputs.reliability.loc[spec.case_id]
    return {"qc_tiles_kept": n_tiles,
            "ood_percentile": round(req_float(row["ood_percentile"], f"{spec.case_id} ood_percentile"), 3),
            "reliability": round(req_float(row["reliability"], f"{spec.case_id} reliability"), 3),
            "reliability_flag": as_bool(row["reliability_flag"], f"{spec.case_id} reliability_flag")}


def tiles_block(spec: CaseSpec, inputs: Inputs, cfg: Mapping[str, Any], qc: Sequence[int]) -> list[dict[str, Any]]:
    gspec = cfg["tiles"]["grids"][cfg["tiles"]["contract_grid"]]
    size, cols = int(gspec["size"]), int(gspec["cols"])
    risk = inputs.tile_risk[inputs.tile_risk["case_id"] == spec.case_id].set_index("tile_idx")["risk"]
    attention = None
    if inputs.tile_attention is not None:
        att = inputs.tile_attention[inputs.tile_attention["case_id"] == spec.case_id]
        attention = att.set_index("tile_idx")["attention"] if not att.empty else None
    ctile = inputs.concepts_tile.get(spec.case_id, {})
    out = []
    for idx in qc:
        if idx not in risk.index:
            raise InputError(f"{spec.case_id}: tile {idx} missing from tile_risk.csv")
        if idx not in ctile:
            raise InputError(f"{spec.case_id}: tile {idx} missing from concepts_tile.npz")
        z = [req_float(v, f"{spec.case_id} tile {idx} concept z") for v in ctile[idx]["z"]]
        tile = {"idx": idx, "row": idx // cols, "col": idx % cols, "x": (idx % cols) * size, "y": (idx // cols) * size,
                "risk": round(req_float(risk[idx], f"{spec.case_id} tile {idx} risk"), 3),
                "concepts": {c: round(v, 2) for c, v in zip(inputs.concept_ids, z)}}
        if attention is not None and idx in attention.index:
            tile["attention"] = round(req_float(attention[idx], f"{spec.case_id} tile {idx} attention"), 4)
        out.append(tile)
    return out


def _top_tiles(ctile: Mapping[int, Mapping[str, np.ndarray]], qc: Sequence[int], ci: int, cfg: Mapping[str, Any]
               ) -> list[int]:
    k, thr = int(cfg["concepts"]["top_k"]), float(cfg["concepts"]["z_threshold"])
    ranked = sorted(qc, key=lambda t: (-float(ctile[t]["z"][ci]), t))[:k]
    keep = [t for t in ranked if float(ctile[t]["z"][ci]) >= thr]
    return sorted(keep or ranked[:1])


def concept_blocks(spec: CaseSpec, inputs: Inputs, cfg: Mapping[str, Any], qc: Sequence[int],
                   cell_tiles: Sequence[int]) -> list[dict[str, Any]]:
    contrib = inputs.contributions[inputs.contributions["case_id"] == spec.case_id]
    if contrib.empty:
        raise InputError(f"{spec.case_id}: missing from contributions.csv")
    k = int(cfg["concepts"]["top_k"])
    feats = None
    if inputs.case_features is not None:
        if spec.case_id not in inputs.case_features.index:
            raise InputError(f"{spec.case_id}: missing from case_features.csv")
        feats = inputs.case_features.loc[spec.case_id]
    ctile = inputs.concepts_tile.get(spec.case_id, {})
    by_id = inputs.bank.by_id()
    per_concept_eval = (inputs.concepts_eval.get("concepts") or {}) if isinstance(inputs.concepts_eval, Mapping) else {}
    out = []
    for ci, cid in enumerate(inputs.concept_ids):
        rows = contrib[(contrib["concept"] == cid) & (contrib["source"] != "cell")]
        if rows.empty:
            raise InputError(f"{spec.case_id}: no contributions.csv rows for concept {cid}")
        top_col = f"{cid}__top{k}"
        if feats is not None and top_col in feats.index:
            value = feats[top_col]
        else:
            match = rows[rows["feature"] == top_col]
            if match.empty:
                raise InputError(f"{spec.case_id}: no value for {top_col} in case_features.csv or contributions.csv")
            value = match["value"].iloc[0]
        source = "cav" if (rows["source"] == "cav").any() else "text"
        entry: dict[str, Any] = {
            "id": cid, "label": by_id[cid].label, "value": round(req_float(value, f"{spec.case_id} {top_col}"), 3),
            "contribution": round(sum_contrib(rows, f"{spec.case_id} {cid}"), 3), "direction": by_id[cid].direction,
            "top_tiles": _top_tiles(ctile, qc, ci, cfg) if qc else [], "source": source,
        }
        std_row = rows[rows["feature"] == top_col]
        if not std_row.empty and not is_missing(std_row["value_std"].iloc[0]):
            entry["value_std"] = round(float(std_row["value_std"].iloc[0]), 3)
        spread = feats.get(f"{cid}__spread") if feats is not None else None
        if is_missing(spread) and qc:
            spread = float(np.mean([float(ctile[t]["spread"][ci]) for t in qc]))
        if not is_missing(spread) and math.isfinite(float(spread)):
            entry["spread"] = max(0.0, round(float(spread), 3))
        if feats is not None and (f"{cid}__text_top{k}" in feats.index or f"{cid}__cav_top{k}" in feats.index):
            entry["components"] = {"text_z": rnd(feats.get(f"{cid}__text_top{k}"), 3),
                                   "cav_z": rnd(feats.get(f"{cid}__cav_top{k}"), 3)}
        evald = per_concept_eval.get(cid) if isinstance(per_concept_eval, Mapping) else None
        if isinstance(evald, Mapping) and "caption_consistent" in evald:
            entry["caption_consistent"] = bool(evald["caption_consistent"])
        elif source == "cav":
            entry["caption_consistent"] = True
        out.append(entry)
    if inputs.cells is not None:
        cell_rows = contrib[contrib["source"] == "cell"]
        cell_row = inputs.cells.features.loc[spec.case_id] if spec.case_id in inputs.cells.features.index else None
        for fid, rows in cell_rows.groupby("concept", sort=False):
            meta = inputs.cells.meta.get(str(fid))
            if meta is None:
                if not str(fid).endswith("__missing"):
                    raise InputError(f"contributions.csv cell feature {fid!r} is not in cell_features.yaml")
                continue
            if cell_row is None or is_missing(cell_row.get(fid)):
                continue
            fid_rows = cell_rows[cell_rows["concept"].isin([fid, f"{fid}__missing"])]
            out.append({"id": str(fid), "label": meta["label"], "value": round(float(cell_row[fid]), 3),
                        "contribution": round(sum_contrib(fid_rows, f"{spec.case_id} {fid}"), 3),
                        "direction": meta["direction"], "top_tiles": list(cell_tiles), "source": "cell"})
    out.sort(key=lambda c: (-abs(c["contribution"]), c["id"]))
    return out


def retrieval_blocks(spec: CaseSpec, inputs: Inputs) -> tuple[list, dict | None, float | None, dict | None]:
    entry = (inputs.retrieval.get("cases") or {}).get(spec.case_id)
    if entry is None:
        if spec.variant is None:
            raise InputError(f"{spec.case_id}: missing from retrieval.json")
        return [], None, None, None
    refs = [n["case_id"] for n in entry["neighbors"]] + ([entry["contrast"]["case_id"]] if entry.get("contrast") else [])
    for ref in refs:
        if ref not in inputs.cases.index or inputs.cases.loc[ref, "split"] != "train":
            raise InputError(f"{spec.case_id}: retrieval reference {ref} is not a train-split case")
    similar = [{"case_id": n["case_id"], "label": n["label"], "similarity": rnd(n["similarity"], 3),
                "image": schemas.display_ref(n["case_id"])} for n in entry["neighbors"]]
    contrast = None
    if entry.get("contrast"):
        c = entry["contrast"]
        contrast = {"case_id": c["case_id"], "label": c["label"], "similarity": rnd(c["similarity"], 3),
                    "differs_most_in": list(c.get("differs_most_in") or []), "image": schemas.display_ref(c["case_id"])}
    return similar, contrast, rnd(entry.get("neighbor_agreement"), 3), entry


def margin_block(spec: CaseSpec, inputs: Inputs, cfg: Mapping[str, Any], reliability_flag: bool, mpp: float
                 ) -> dict[str, Any]:
    if inputs.margins is None:
        raise InputError("margins.json is required for surgical cases")
    entry = (inputs.margins.get("cases") or {}).get(spec.case_id)
    if entry is None:
        raise InputError(f"{spec.case_id}: missing from margins.json")
    threshold = req_float(inputs.margins["threshold"], "margins.json threshold")
    # p and the threshold stay unrounded so the JSON call always agrees with margin eval on the raw scores.
    segments = [{"margin": s["margin"], "x": int(s["x"]), "y": int(s["y"]), "size": int(s["size"]),
                 "p": req_float(s["p"], f"{spec.case_id} segment p")} for s in entry["segments"]]
    probs = [s["p"] for s in segments]
    geometry = [{k: s[k] for k in ("margin", "x", "y", "size")} for s in segments]
    per_margin = margin.per_margin_scores(geometry, probs)
    call, involved = margin.suggest_call(per_margin, threshold, reliability_flag)
    block: dict[str, Any] = {"suggested_call": call, "involved": involved,
                             "per_margin": {m: float(per_margin[m]) for m in schemas.MARGINS},
                             "strip_um": round(float(cfg["margin"]["strip_px"]) * mpp, 2),
                             "threshold": threshold, "segments": segments}
    if call == "tumor_at_margin":
        lengths = margin.involved_length_um(geometry, probs, threshold, mpp)
        block["involved_length_um"] = {m: round(v, 1) for m, v in lengths.items()}
    return block


def cells_block(spec: CaseSpec, inputs: Inputs, cfg: Mapping[str, Any], qc: Sequence[int]
                ) -> tuple[dict[str, Any], list[int], str | None]:
    """(cells block, cell tile idxs, warning)."""
    if inputs.cells is None:
        return {"status": "disabled"}, [], None
    if spec.case_id not in inputs.cells.features.index:
        if spec.variant:            # cells are segmented on clean cases only (CELLS_SPEC §4)
            return {"status": "disabled"}, [], None
        return {"status": "failed"}, [], f"{spec.case_id}: no cell features (cells status failed)"
    row = inputs.cells.features.loc[spec.case_id]
    tile_rows = inputs.cells.tiles[inputs.cells.tiles["case_id"] == spec.case_id]
    contrib = inputs.contributions[(inputs.contributions["case_id"] == spec.case_id)
                                   & (inputs.contributions["source"] == "cell")]
    tiles = []
    for _, t in tile_rows.sort_values("tile_idx").iterrows():
        idx = int(t["tile_idx"])
        if idx not in qc:
            continue
        tiles.append({"idx": idx, "overlay": schemas.cell_overlay_ref(spec.case_id, idx),
                      "counts": {ct: int(t[f"n_{ct}"]) for ct in schemas.CELL_TYPES}})
    features = {}
    for fid, meta in inputs.cells.meta.items():
        rows = contrib[contrib["concept"].isin([fid, f"{fid}__missing"])]
        features[fid] = {"value": rnd(row.get(fid), 4), "unit": meta["unit"],
                         "ref_percentile": rnd(row.get(f"{fid}__ref_pct"), 3),
                         "contribution": round(sum_contrib(rows, f"{spec.case_id} {fid}"), 3) if not rows.empty else None}
    block = {"status": "ok", "model": str(cfg["cells"]["model"]), "mpp_model": float(cfg["cells"]["model_mpp"]),
             "tiles": tiles, "features": features,
             "flags": {"insufficient_tumor_nuclei": as_bool(row["insufficient_tumor_nuclei"],
                                                            f"{spec.case_id} insufficient_tumor_nuclei")}}
    return block, [t["idx"] for t in tiles], None


# ---- findings ----------------------------------------------------------------------------------------------------------
def build_findings(case: Mapping[str, Any], cfg: Mapping[str, Any], retrieval_entry: Mapping[str, Any] | None,
                   cell_meta: Mapping[str, Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(kind: str, data: dict[str, Any]) -> None:
        findings.append({"id": f"F{len(findings) + 1}", "kind": kind, "data": data})

    pred = case["prediction"]
    top = rules_mod.top_class(pred, cfg["aggressive_class"])
    add("prediction", {"top": top, "p": round(pred["probs"][pred["classes"].index(top)], 3),
                       "set": list(pred["conformal_set"]), "confidence": round(1.0 - float(cfg["alpha"]), 4),
                       "risk": round(pred["risk_score"], 3)})
    min_abs = float(cfg["findings"]["min_abs_contribution"])
    ranked = [c for c in case["concepts"] if c["source"] != "cell" and abs(c["contribution"]) >= min_abs]
    for c in ranked[: int(cfg["findings"]["max_concepts"])]:
        add("concept", {"concept": c["id"], "label": c["label"], "contribution": c["contribution"],
                        "tiles": list(c["top_tiles"]), "value": c["value"], "direction": c["direction"]})
    if retrieval_entry is not None:
        add("retrieval", {"agree": int(retrieval_entry["agree"]), "of": int(retrieval_entry["of"]),
                          "label": str(retrieval_entry["predicted"])})
    q = case["quality"]
    add("quality", {"reliability_flag": q["reliability_flag"], "reliability": round(q["reliability"], 2),
                    "ood_percentile": round(q["ood_percentile"], 2), "qc_tiles_kept": q["qc_tiles_kept"],
                    "min_tiles_kept": int(cfg["qc"]["min_tiles_kept"])})
    cells = case.get("cells") or {}
    if cells.get("status") == "ok":
        lo, hi = cfg["cells"]["findings_percentile_extremes"]
        cell_tiles = [t["idx"] for t in cells["tiles"]]
        for fid, feat in cells["features"].items():
            if feat["value"] is None:
                continue
            pct, contribution = feat["ref_percentile"], feat["contribution"]
            extreme = pct is not None and (pct <= lo or pct >= hi)
            nontrivial = contribution is not None and abs(contribution) >= min_abs
            if extreme or nontrivial:
                label = (cell_meta or {}).get(fid, {}).get("label", fid)
                add("cells", {"feature": fid, "label": label, "value": feat["value"], "unit": feat["unit"],
                              "ref_percentile": pct, "tiles": cell_tiles})
    if case.get("margin") is not None:
        m = case["margin"]
        data = {"suggested_call": m["suggested_call"], "involved": list(m["involved"]),
                "per_margin": {k: round(v, 2) for k, v in m["per_margin"].items()}}
        if m.get("involved_length_um"):
            data["involved_length_um"] = {k: round(v) for k, v in m["involved_length_um"].items()}
        add("margin", data)
    return findings


# ---- assembly ----------------------------------------------------------------------------------------------------------
def assemble(spec: CaseSpec, inputs: Inputs, cfg: Mapping[str, Any], rules: rules_mod.RuleSet, mpp: float,
             synthetic: bool) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """(case dict without arrival/summary/LLM agent, retrieval entry, warnings). Raises InputError."""
    warnings: list[str] = []
    qc = qc_tiles(spec, inputs)
    cells, cell_tiles, warn = cells_block(spec, inputs, cfg, qc)
    if warn:
        warnings.append(warn)
    tiles = tiles_block(spec, inputs, cfg, qc)
    quality = quality_block(spec, inputs, len(tiles))
    prediction = prediction_block(spec, inputs, cfg)
    similar, contrast, neighbor_agreement, retrieval_entry = retrieval_blocks(spec, inputs)
    if retrieval_entry is None and spec.variant is None:
        warnings.append(f"{spec.case_id}: not in retrieval.json (no similar cases)")
    agree_top = inputs.predictions.loc[spec.case_id].get("agree_top")
    if is_missing(agree_top):
        disagree = inputs.reliability.loc[spec.case_id].get("conch_musk_disagree")
        agree_top = None if is_missing(disagree) else not as_bool(disagree, "conch_musk_disagree")
    else:
        agree_top = as_bool(agree_top, "agree_top")
    gspec = cfg["tiles"]["grids"][cfg["tiles"]["contract_grid"]]
    case: dict[str, Any] = {
        "schema_version": contract.SCHEMA_VERSION,
        "synthetic": bool(synthetic),
        "case_id": spec.case_id,
        "lane": spec.lane,
        "simulated": spec.lane == "surgical",
        "simulated_variant": spec.variant,
        "degradation": {"type": spec.degradation[0], "severity": spec.degradation[1]} if spec.degradation else None,
        "source_case_ids": list(spec.source_ids),
        "arrived_offset_s": 0.0,
        "image": {"display": schemas.display_ref(spec.case_id), "width": spec.width, "height": spec.height,
                  "tile_size": int(gspec["size"]), "grid": [int(gspec["rows"]), int(gspec["cols"])]},
        "quality": quality,
        "prediction": prediction,
        "concepts": concept_blocks(spec, inputs, cfg, qc, cell_tiles),
        "tiles": tiles,
        "similar_cases": similar,
        "contrast_case": contrast,
        "agreement": {"conch_musk_same_top": agree_top, "neighbor_agreement": neighbor_agreement},
        "margin": margin_block(spec, inputs, cfg, quality["reliability_flag"], mpp) if spec.lane == "surgical" else None,
        "cells": cells,
    }
    case["flag"] = tiers.flag_for(case, cfg)
    case["findings"] = build_findings(case, cfg, retrieval_entry, inputs.cells.meta if inputs.cells else None)
    case["agent"] = rules_mod.draft_orders(case, cfg, rules)
    case["summary"] = summary_mod.template_summary(case["findings"], cfg)
    eval_only = {"true_label": spec.true_label, "split": spec.split, "group": spec.group,
                 "source_case_ids": list(spec.source_ids) or [spec.case_id]}
    eval_only.update(dict(spec.extra_eval))
    case["eval_only"] = eval_only
    try:
        contract.Case.model_validate(case)        # fail before any LLM call or write
    except ValueError as exc:
        raise InputError(f"{spec.case_id}: assembled case violates the contract: {exc}") from None
    return case, retrieval_entry or {}, warnings


# ---- heroes and arrivals ------------------------------------------------------------------------------------------------
def _hero_kind(case: Mapping[str, Any], kind: str) -> bool:
    if kind == "biopsy":
        return case["lane"] == "biopsy"
    if kind == "surgical":
        return case["lane"] == "surgical" and case["simulated_variant"] is None
    return case["lane"] == "surgical" and case["simulated_variant"] == kind.split("_", 1)[1]


def _criteria(case: Mapping[str, Any], spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[bool]:
    out = []
    pred = case["prediction"]
    m = case.get("margin") or {}
    for key, want in spec.items():
        if key == "arrival":
            continue
        if key == "top_class":
            out.append(rules_mod.top_class(pred, cfg["aggressive_class"]) == want)
        elif key == "conformal_set":
            out.append(sorted(pred["conformal_set"]) == sorted(want))
        elif key == "reliability_flag":
            out.append(case["quality"]["reliability_flag"] == want)
        elif key == "agent_status":
            out.append(case["agent"]["status"] == want)
        elif key == "suggested_call":
            out.append(m.get("suggested_call") == want)
        elif key == "involved":
            out.append(sorted(m.get("involved") or []) == sorted(want))
        elif key == "simulated_variant":
            out.append(case["simulated_variant"] == want)
        else:
            raise ValueError(f"demo.heroes: unknown criterion {key!r}")
    return out


def _correct(case: Mapping[str, Any], kind: str, cfg: Mapping[str, Any]) -> bool:
    truth = case["eval_only"]["true_label"]
    if kind == "biopsy":
        return rules_mod.top_class(case["prediction"], cfg["aggressive_class"]) == truth
    m = case.get("margin") or {}
    if kind == "surgical_composite":
        return all(case["eval_only"].get(f"margin_truth_{name}") == "tumor_side" for name in m.get("involved") or [])
    carcinoma = truth in cfg["carcinoma_classes"]
    if m.get("suggested_call") == "tumor_at_margin":
        return carcinoma
    return True


def pick_heroes(cases: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """{hero kind: {case_id, exact, matched, of}} per settings demo.heroes (closest match when none is exact)."""
    demo = cfg["demo"]
    out: dict[str, dict[str, Any]] = {}
    taken: set[str] = set()
    for kind, spec in demo["heroes"].items():
        pool = [c for c in cases if _hero_kind(c, kind) and c["case_id"] not in taken]
        if not pool:
            continue

        def rank(c: Mapping[str, Any]) -> tuple:
            hits = _criteria(c, spec, cfg)
            correct = _correct(c, kind, cfg) if demo["prefer_correct"] else False
            return (-sum(hits), not correct, -float(c["prediction"]["risk_score"]), c["case_id"])

        best = min(pool, key=rank)
        hits = _criteria(best, spec, cfg)
        if not all(hits) and demo["fallback"] != "closest_match":
            continue
        taken.add(best["case_id"])
        out[kind] = {"case_id": best["case_id"], "exact": all(hits), "matched": sum(hits), "of": len(hits)}
    return out


def arrival_offsets(cases: Sequence[Mapping[str, Any]], heroes: Mapping[str, Mapping[str, Any]],
                    cfg: Mapping[str, Any]) -> dict[str, float]:
    """Seeded Poisson arrivals over one clock. Order: clean surgical heroes first, then the shuffled clean non-hero
    cases, then the biopsy hero (arrives last of the replayed cases), then the variant heroes, then the other variants.

    The backend holds surgical cases for injection and injects, per variant, the best demo.heroes match with the
    EARLIEST offset (backend.simulator.inject_surgical); several cases can match every criterion, so surgical heroes
    get the earliest surgical offsets of their kind and the injection picks the hero 08 chose (08 may prefer a correct
    read offline; the backend never sees labels)."""
    arrival = cfg["sim"]["arrival"]
    if arrival["process"] != "poisson":
        raise ValueError(f"sim.arrival.process {arrival['process']!r} is not supported (poisson)")
    rng = random.Random(int(arrival["seed"]))
    hero_ids = [h["case_id"] for k, h in heroes.items() if k != "biopsy"]
    biopsy_hero = [heroes["biopsy"]["case_id"]] if "biopsy" in heroes else []
    special = set(hero_ids) | set(biopsy_hero)
    clean = sorted(c["case_id"] for c in cases if c["simulated_variant"] is None and c["case_id"] not in special)
    rng.shuffle(clean)
    clean_heroes = [h for h in hero_ids if any(c["case_id"] == h and c["simulated_variant"] is None for c in cases)]
    variant_heroes = [h for h in hero_ids if h not in clean_heroes]
    variants = sorted(c["case_id"] for c in cases if c["simulated_variant"] is not None and c["case_id"] not in special)
    order = clean_heroes + clean + biopsy_hero + variant_heroes + variants
    mean = float(arrival["mean_interarrival_s"])
    t, out = 0.0, {}
    for cid in order:
        t += rng.expovariate(1.0 / mean)
        out[cid] = round(t, 1)
    return out


# ---- LLM pass ---------------------------------------------------------------------------------------------------------------
def llm_order(cases: Sequence[Mapping[str, Any]], heroes: Mapping[str, Mapping[str, Any]]) -> list[int]:
    """Indices of cases in LLM priority: heroes, then flagged / low tiers, then the rest."""
    hero_ids = {h["case_id"] for h in heroes.values()}
    return sorted(range(len(cases)), key=lambda i: (cases[i]["case_id"] not in hero_ids,
                                                    not cases[i]["flag"]["flagged"], cases[i]["flag"]["tier"],
                                                    cases[i]["case_id"]))


# ---- media ------------------------------------------------------------------------------------------------------------------
def copy_file(src: Path, dst: Path, force: bool) -> bool:
    if not src.is_file():
        raise InputError(f"missing media file {src}")
    data = src.read_bytes()
    if dst.is_file() and not force and dst.stat().st_size == len(data) and dst.read_bytes() == data:
        return False
    io.atomic_write_bytes(dst, data)
    return True


def write_if_changed(path: Path, obj: Any, force: bool) -> bool:
    if path.is_file() and not force:
        try:
            if io.read_json(path) == obj:
                return False
        except ValueError:
            pass
    io.write_json(path, obj)
    return True


def spotcheck_items(cases_df: pd.DataFrame, inputs: Inputs, cfg: Mapping[str, Any], root: Path, art: Path,
                    synthetic: bool, force: bool) -> dict[str, Any]:
    """SPOTCHECK_ITEMS: top settings spotcheck.per_concept test tiles per concept (crops cut from the display JPEG)
    and spotcheck.overlays_per_class cell overlays per class (labels used only to pick; never written)."""
    from PIL import Image

    sc = cfg["spotcheck"]
    gspec = cfg["tiles"]["grids"][cfg["tiles"]["contract_grid"]]
    size = int(gspec["size"])
    test_ids = set(cases_df.loc[cases_df["split"] == "test", "case_id"])
    rows = [(cid, t, d["z"]) for cid, tiles in inputs.concepts_tile.items() if cid in test_ids for t, d in tiles.items()]
    qc_ok = {(r.case_id, int(r.tile_idx)) for r in inputs.tiles.itertuples() if as_bool(r.qc_pass)}
    rows = [r for r in rows if (r[0], r[1]) in qc_ok]
    labels = inputs.bank.by_id()
    items: list[dict[str, Any]] = []
    opened: dict[str, Any] = {}
    for ci, cid in enumerate(inputs.concept_ids):
        ranked = sorted(rows, key=lambda r: (-float(r[2][ci]), r[0], r[1]))[: int(sc["per_concept"])]
        for case_id, tidx, z in ranked:
            name = f"{cid}_{case_id}_{tidx}.png"
            dst = art / "spotcheck" / name
            if force or not dst.is_file():
                if case_id not in opened:
                    opened[case_id] = Image.open(schemas.resolve(root, cases_df.loc[case_id, "display_path"])).convert("RGB")
                img = opened[case_id]
                scale = img.width / float(cases_df.loc[case_id, "width"])
                col, row = tidx % int(gspec["cols"]), tidx // int(gspec["cols"])
                box = tuple(int(round(v * scale)) for v in (col * size, row * size, (col + 1) * size, (row + 1) * size))
                crop = img.crop(box).resize((int(sc["crop_px"]), int(sc["crop_px"])))
                io.save_image(dst, crop)
            items.append({"item_id": f"concept:{cid}:{case_id}:{tidx}", "kind": "concept", "target": cid,
                          "label": labels[cid].label, "case_id": case_id, "tile_idx": int(tidx),
                          "image": f"spotcheck/{name}", "score": rnd(z[ci], 3)})
    overlay_items: list[dict[str, Any]] = []
    if inputs.cells is not None:
        rng = random.Random(int(sc["seed"]))
        tiles = inputs.cells.tiles[inputs.cells.tiles["case_id"].isin(test_ids)]
        for cls in cfg["classes"]:
            ids = sorted(set(tiles["case_id"]) & set(cases_df.loc[cases_df["label"] == cls, "case_id"]))
            rng.shuffle(ids)
            for case_id in ids[: int(sc["overlays_per_class"])]:
                t = tiles[tiles["case_id"] == case_id].sort_values("tile_idx").iloc[0]
                idx = int(t["tile_idx"])
                copy_file(schemas.resolve(root, t["overlay"]), schemas.case_cell_overlay(art, case_id, idx), force)
                overlay_items.append({"item_id": f"cells:overlay:{case_id}:{idx}", "kind": "cells", "target": "overlay",
                                      "label": "Cell segmentation overlay", "case_id": case_id, "tile_idx": idx,
                                      "image": schemas.cell_overlay_ref(case_id, idx), "score": None})
        rng.shuffle(overlay_items)            # picked per class; the order must not reveal the class
    items.extend(overlay_items)
    return {"created_utc": io.utc_now(), "synthetic": bool(synthetic), "items": items}


# ---- main -------------------------------------------------------------------------------------------------------------------
def build(ctx: cli.StepContext, *, max_llm_calls: int = 0, llm_agent: bool = False, clean: bool = False,
          concept_bank_path: Path | None = None, cell_features_path: Path | None = None,
          client: LumenClient | None = None) -> dict[str, Any]:
    cfg, root, art = ctx.cfg, ctx.data_root, ctx.artifacts_root
    inputs = load_inputs(cfg, root, art, concept_bank_path=concept_bank_path, cell_features_path=cell_features_path)
    rules = rules_mod.load_rules(cfg)
    mpp = config.effective_mpp(cfg, schemas.inventory_json(root))
    specs = case_plan(inputs, cfg, root, ctx.limit)
    cases: list[dict[str, Any]] = []
    spec_of: dict[str, CaseSpec] = {}
    warnings: list[str] = []
    skipped: list[dict[str, str]] = []
    for spec in specs:
        try:
            case, _, warn = assemble(spec, inputs, cfg, rules, mpp, ctx.synthetic)
        except (InputError, ValueError) as exc:
            if spec.variant is None:
                raise
            skipped.append({"case_id": spec.case_id, "reason": str(exc)})
            continue
        warnings.extend(warn)
        cases.append(case)
        spec_of[spec.case_id] = spec

    heroes = pick_heroes(cases, cfg)
    offsets = arrival_offsets(cases, heroes, cfg)
    for case in cases:
        case["arrived_offset_s"] = offsets[case["case_id"]]

    if client is None and max_llm_calls > 0:
        client = LumenClient.from_settings(cfg, cache_dir=root / cfg["llm"]["cache_dir"], max_calls=max_llm_calls)
        if not client.has_key:
            warnings.append("no Lumen key: summaries and rationales use the templates")
            client = None
    summary_paths: dict[str, int] = {}
    agent_paths: dict[str, int] = {}
    llm_cases: list[dict[str, Any]] = []
    for i in llm_order(cases, heroes):
        case = cases[i]
        if client is not None:
            summary, info = summary_mod.generate(case["findings"], cfg, client=client)
            case["summary"] = summary
            summary_paths[info["path"]] = summary_paths.get(info["path"], 0) + 1
            entry = {"case_id": case["case_id"], "summary": info["path"], "summary_reason": info["reason"],
                     "findings_hash": info["findings_hash"]}
            if llm_agent:
                agent_block, ainfo = agent_mod.draft(case, cfg, rules=rules, client=client)
                case["agent"] = agent_block
                agent_paths[ainfo["path"]] = agent_paths.get(ainfo["path"], 0) + 1
                entry["agent"] = ainfo["path"]
            llm_cases.append(entry)
        else:
            summary_paths["template"] = summary_paths.get("template", 0) + 1

    written, unchanged = 0, 0
    out_ids: set[str] = set()
    image_ids: set[str] = set()
    for case in cases:
        model = contract.Case.model_validate(case)
        obj = model.model_dump(mode="json")
        path = schemas.case_json(art, case["case_id"])
        if write_if_changed(path, obj, ctx.force):
            written += 1
        else:
            unchanged += 1
        out_ids.add(path.name)
        spec = spec_of[case["case_id"]]
        copy_file(spec.display_src, schemas.case_image(art, case["case_id"]), ctx.force)
        image_ids.add(case["case_id"])
        refs = [s["case_id"] for s in case["similar_cases"]] + ([case["contrast_case"]["case_id"]]
                                                               if case["contrast_case"] else [])
        for ref in refs:
            if ref not in image_ids:
                if ref not in inputs.cases.index:
                    raise InputError(f"{case['case_id']}: reference {ref} is not in cases.csv")
                copy_file(schemas.resolve(root, inputs.cases.loc[ref, "display_path"]), schemas.case_image(art, ref),
                          ctx.force)
                image_ids.add(ref)
        if case["cells"] and case["cells"]["status"] == "ok" and inputs.cells is not None:
            tile_rows = inputs.cells.tiles[inputs.cells.tiles["case_id"] == case["case_id"]].set_index("tile_idx")
            for t in case["cells"]["tiles"]:
                copy_file(schemas.resolve(root, tile_rows.loc[t["idx"], "overlay"]),
                          schemas.case_cell_overlay(art, case["case_id"], t["idx"]), ctx.force)

    removed: list[str] = []
    if clean:
        for path in sorted((art / "cases").glob("*.json")):
            if path.name not in out_ids:
                path.unlink()
                removed.append(path.name)

    refs_all = sorted({r for c in cases for r in [s["case_id"] for s in c["similar_cases"]]
                       + ([c["contrast_case"]["case_id"]] if c["contrast_case"] else [])})
    reference_index = {
        "created_utc": io.utc_now(), "synthetic": bool(ctx.synthetic),
        "encoder": inputs.retrieval.get("encoder"), "head": inputs.retrieval.get("head"), "k": inputs.retrieval.get("k"),
        "cases": {r: {"label": str(inputs.cases.loc[r, "label"]), "split": str(inputs.cases.loc[r, "split"]),
                      "image": schemas.display_ref(r)} for r in refs_all},
    }
    io.write_json(schemas.reference_index_json(art), reference_index)
    io.write_json(schemas.spotcheck_items_json(art),
                  spotcheck_items(inputs.cases, inputs, cfg, root, art, ctx.synthetic, ctx.force))

    extra = {
        "n_cases": len(cases), "written": written, "unchanged": unchanged, "removed": removed,
        "by_kind": {"test": sum(c["simulated_variant"] is None for c in cases),
                    "degraded": sum(c["simulated_variant"] == "degraded" for c in cases),
                    "composite": sum(c["simulated_variant"] == "composite" for c in cases)},
        "heroes": heroes, "skipped": skipped, "warnings": warnings,
        "variants_without_retrieval": sum(c["simulated_variant"] is not None and not c["similar_cases"] for c in cases),
        "llm_calls": client.calls if client is not None else 0,
        "llm_cache_hits": client.cache_hits if client is not None else 0,
        "llm_model": client.model if client is not None and client.calls + client.cache_hits > 0 else None,
        "summary_paths": summary_paths, "agent_paths": agent_paths, "llm_cases": llm_cases, "mpp": mpp,
    }
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg, inputs=inputs.inputs_used,
                  outputs=[str(art / "cases"), str(art / "images"), str(schemas.reference_index_json(art)),
                           str(schemas.spotcheck_items_json(art))],
                  args={"limit": ctx.limit, "force": ctx.force, "max_llm_calls": max_llm_calls,
                        "llm_agent": llm_agent, "clean": clean},
                  synthetic=ctx.synthetic, extra=extra)
    return extra


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Assemble contract case JSONs, images, references and spot-check items.")
    parser.add_argument("--max-llm-calls", type=int, default=0,
                        help="Lumen call budget for summaries (and --llm-agent rationales); 0 = templates only")
    parser.add_argument("--llm-agent", action="store_true", help="also let the LLM write agent rationales")
    parser.add_argument("--clean", action="store_true", help="remove case JSONs this run did not produce")
    args = parser.parse_args(argv)
    if args.max_llm_calls < 0:
        parser.error("--max-llm-calls must be >= 0")
    ctx = cli.context(args)
    try:
        extra = build(ctx, max_llm_calls=args.max_llm_calls, llm_agent=args.llm_agent, clean=args.clean)
    except (InputError, ValueError) as exc:
        print(f"08_build_case_json: {exc}", file=sys.stderr)
        return 1
    print(f"08_build_case_json: {extra['n_cases']} cases ({extra['by_kind']}), written {extra['written']}, "
          f"unchanged {extra['unchanged']}, skipped {len(extra['skipped'])}, LLM calls {extra['llm_calls']} "
          f"(cache hits {extra['llm_cache_hits']}), heroes {sorted(h['case_id'] for h in extra['heroes'].values())}")
    for w in extra["warnings"][:20]:
        print(f"  warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
