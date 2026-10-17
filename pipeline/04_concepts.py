"""Named-criterion concept scores per tile and case, concept sanity checks and zero-shot baselines (Task 9, §A11).

    python -m pipeline.04_concepts [--limit 5] [--force]          # laptop, after export --unpack (or make analyze)

Tile score (plan §6.4/§6.5, BUILD_PLAN §A11), in the CONCH align space (settings concepts.encoder):
    phrasing embedding  p = normalize(mean over templates of the prompt embeddings)    (text/<encoder>.npz)
    s_i(tile)           = cos(tile, p_i) - mean_j cos(tile, q_j)   for positive phrasings p_i, contrasts q_j
    text score          = mean_i s_i  ( = mean cos to positives - mean cos to contrasts )
    z_text              = (text score - train mean) / train std       train = clean QC tiles of train cases
    spread              = std_i s_i / train std                       prompt-ensemble spread, in z units
    z_cav               = z-scored concept-vector logit (04_cav concept_vectors.npz), only for caption-consistent
                          concepts, else NaN
    z                   = mean(z_text, z_cav) when caption-consistent, else z_text   (source "cav" / "text")
Case features (core.schemas CASE_FEATURES, concept_feature_columns): c__top<k> = mean of the top-k QC tile z,
c__frac = fraction of QC tiles with z > concepts.z_threshold, c__spread = mean spread of those top-k tiles, plus
components c__text_top<k> / c__cav_top<k>, and n_tiles / variant / source_case_id. Rows: every clean case of every
split plus degraded (deg512 -> schemas.variant_case_id) and composite (comp512 -> composite IDs) variants when
their embeddings exist, z-scored with the clean train statistics; split = the source case's split. Variants are
recognized by ID and never enter headline metrics. concepts_tile.npz (CONCEPTS_TILE_NPZ) holds every tile row.

Eval (core.schemas eval_json; NaN -> null):
    concepts.json       section "nct": cross-organ sanity check on NCT-CRC (settings nct.concept_checks), AUROC
                        with stratified (cluster-free) bootstrap CIs; section "concepts": per concept source and
                        caption-consistency (04_cav writes section "cav" of the same file; each step keeps the other's)
    concepts_bach.json  BACH test split: per-class means of c__top<k> with cluster-bootstrap CIs, Invasive-vs-rest
                        AUROC per concept vs its expected direction, expected-top-class table; contradictions are
                        listed, never hidden; spot-check summary when eval/spotcheck.json exists
    zeroshot.json       zero-shot BACH 4-class baseline per text-aligned encoder: case embedding = normalized mean
                        of its QC tile align embeddings; class embedding = normalized mean over class phrasings x
                        templates; macro-F1 / balanced accuracy / accuracy with cluster-bootstrap CIs on test

Wording: concepts are "features associated with" invasive carcinoma, never causes or a Nottingham grade.
Laptop step (Python 3.13 under uv): numpy, pandas, scipy via core.stats.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core import cli, config, io, schemas, stats

STEP = "pipeline.04_concepts"
META_STEP = "04_concepts"
GRID = "g512"
CONCEPT_VECTORS = "concept_vectors.npz"   # written by 04_cav (format in pipeline/04_cav.py)


def concept_vectors_npz(root: str | Path) -> Path:
    return Path(root) / CONCEPT_VECTORS


def _l2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def _num(x: Any) -> Any:
    """JSON-safe number: NaN / inf -> None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _clean_ci(ci: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (_num(v) if isinstance(v, (float, np.floating)) else v) for k, v in ci.items()}


# ---- prompt embeddings -------------------------------------------------------------------------------------------------
def load_text(root: Path, encoder: str) -> dict[str, np.ndarray]:
    path = schemas.text_npz(root, encoder)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run pipeline.03_text first")
    data = io.load_npz(path)
    problems = schemas.TEXT_NPZ.problems(data)
    if problems:
        raise ValueError("; ".join(problems))
    return dict(zip(data["ids"].tolist(), np.asarray(data["emb"], np.float32)))


def phrasing_embeddings(bank: schemas.ConceptBank, text: Mapping[str, np.ndarray]
                        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """concept id -> (positives [P, D], contrasts [Q, D]); each row = normalized mean over the templates."""
    n_t = len(bank.templates)
    out = {}
    for c in bank.concepts:
        mats = []
        for polarity, phrasings in (("pos", c.positives), ("con", c.contrasts)):
            rows = []
            for i in range(len(phrasings)):
                ids = [schemas.concept_prompt_id(c.id, polarity, i, t) for t in range(n_t)]
                missing = [pid for pid in ids if pid not in text]
                if missing:
                    raise KeyError(f"text npz lacks prompt ids {missing[:3]}: re-run 03_text for this concepts.yaml")
                rows.append(_l2(np.mean([_l2(text[pid]) for pid in ids], axis=0)))
            mats.append(np.stack(rows))
        out[c.id] = (mats[0], mats[1])
    return out


def class_embeddings(bank: schemas.ConceptBank, text: Mapping[str, np.ndarray], classes: Sequence[str]) -> np.ndarray:
    """[C, D]: normalized mean over every phrasing x template of each class (CONCH zero-shot ensembling)."""
    rows = []
    for cls in classes:
        ids = [schemas.class_prompt_id(cls, i, t) for i in range(len(bank.class_prompts[cls]))
               for t in range(len(bank.templates))]
        rows.append(_l2(np.mean([_l2(text[pid]) for pid in ids], axis=0)))
    return np.stack(rows)


def raw_scores(x: np.ndarray, phr: Mapping[str, tuple[np.ndarray, np.ndarray]], ids: Sequence[str]
               ) -> tuple[np.ndarray, np.ndarray]:
    """(score [N, C], phrasing std [N, C]) in cosine units."""
    x = _l2(x)
    score = np.zeros((len(x), len(ids)), np.float32)
    std = np.zeros_like(score)
    for k, cid in enumerate(ids):
        pos, con = phr[cid]
        s_i = x @ pos.T - (x @ con.T).mean(1, keepdims=True)
        score[:, k] = s_i.mean(1)
        std[:, k] = s_i.std(1)
    return score, std


# ---- concept vectors (04_cav) -------------------------------------------------------------------------------------------
def load_vectors(root: Path, ids: Sequence[str], dim: int) -> dict[str, Any] | None:
    path = concept_vectors_npz(root)
    if not path.is_file():
        return None
    data = io.load_npz(path)
    order = {c: i for i, c in enumerate(data["concept_ids"].tolist())}
    if data["w"].shape[1] != dim:
        raise ValueError(f"{path}: vector dim {data['w'].shape[1]} != embedding dim {dim} (different encoder?)")
    use = np.array([bool(data["caption_consistent"][order[c]]) if c in order else False for c in ids])
    w = np.stack([data["w"][order[c]] if c in order else np.full(dim, np.nan, np.float32) for c in ids])
    b = np.array([data["b"][order[c]] if c in order else np.nan for c in ids], np.float32)
    return {"use": use, "w": w.astype(np.float32), "b": b, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# ---- data ------------------------------------------------------------------------------------------------------------------
def read_csv_str(path: Path, **kw: Any) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype={"case_id": str, "split": str, "label": str,
                                                           "group": str, "item_id": str, "degrade_type": str}, **kw)


def load_tile_embeddings(root: Path, encoder: str, set_name: str) -> tuple[pd.DataFrame, np.ndarray] | None:
    npy, idx = schemas.emb_npy(root, encoder, set_name, "align"), schemas.emb_index_csv(root, encoder, set_name, "align")
    if not io.outputs_exist(npy, idx):
        return None
    index = read_csv_str(idx)
    schemas.validate_frame(schemas.EMB_INDEX, index)
    arr = np.load(npy).astype(np.float32)
    if len(arr) != len(index):
        raise ValueError(f"{npy}: {len(arr)} rows but {len(index)} index rows")
    return index, arr


def tile_rows(cases: pd.DataFrame, tiles: pd.DataFrame, index: pd.DataFrame, kind: str, cfg: Mapping[str, Any],
              root: Path) -> pd.DataFrame:
    """Frame of embedded g512 tiles to score: emb_row, case_id (output ID), source_case_id, split, tile_idx, variant,
    qc. Clean and degraded tiles take the source case's QC flags; composite tiles are all used (no tiles.csv rows)."""
    df = index[index["grid"].isin([GRID, ""])].copy()
    df["tile_idx"] = df["tile_idx"].astype(int)
    df = df[df["tile_idx"] >= 0]
    split_of = dict(zip(cases["case_id"], cases["split"]))
    qc = tiles[tiles["grid"] == GRID][["case_id", "tile_idx", "qc_pass"]].copy()
    qc["qc_pass"] = qc["qc_pass"].astype(str).str.lower().isin(["1", "true", "1.0"])
    qc_map = {(c, int(t)): bool(q) for c, t, q in qc.itertuples(index=False, name=None)}
    if kind == "clean":
        df = df[df["degrade_type"] == ""]
        df["source_case_id"] = df["case_id"]
        df["variant"] = "clean"
    elif kind == "degraded":
        df = df[df["degrade_type"] != ""]
        df["source_case_id"] = df["case_id"]
        df["case_id"] = [schemas.variant_case_id(c, t, int(s)) for c, t, s in
                         zip(df["case_id"], df["degrade_type"], df["degrade_severity"])]
        df["variant"] = "degraded"
    else:
        comp_path = schemas.composites_csv(root)
        comps = read_csv_str(comp_path) if comp_path.is_file() else pd.DataFrame(columns=["case_id", "carcinoma_case_id"])
        src = dict(zip(comps["case_id"], comps.get("carcinoma_case_id", comps["case_id"])))
        df["source_case_id"] = [src.get(c, c) for c in df["case_id"]]
        df["variant"] = "composite"
    df["split"] = [s if s else split_of.get(src_id, cfg["composites"]["source_split"] if kind == "composite" else "")
                   for s, src_id in zip(df["split"], df["source_case_id"])]
    if kind == "composite":
        df["qc"] = True
    else:
        df["qc"] = [qc_map.get((c, t), True) for c, t in zip(df["source_case_id"], df["tile_idx"])]
    df = df[df["split"].isin(schemas.SPLITS)]
    return df[["emb_row", "case_id", "source_case_id", "split", "tile_idx", "variant", "qc"]].reset_index(drop=True)


# ---- case aggregation ------------------------------------------------------------------------------------------------------
def aggregate_cases(meta: pd.DataFrame, z: np.ndarray, z_text: np.ndarray, z_cav: np.ndarray, spread: np.ndarray,
                    ids: Sequence[str], top_k: int, z_threshold: float, log: Any = print) -> pd.DataFrame:
    """One row per case: c__top<k>, c__frac, c__spread, c__text_top<k>, c__cav_top<k> over the case's QC tiles (all
    embedded tiles when none passes QC, logged)."""
    rows = []
    for case_id, group in meta.groupby("case_id", sort=True):
        sel = group.index[group["qc"]].to_numpy()
        if len(sel) == 0:
            log(f"note: {case_id} has no QC-passing embedded tile; aggregating all {len(group)} tiles")
            sel = group.index.to_numpy()
        first = group.iloc[0]
        row: dict[str, Any] = {"case_id": case_id, "split": first["split"], "variant": first["variant"],
                               "source_case_id": first["source_case_id"], "n_tiles": int(len(sel))}
        k = min(top_k, len(sel))
        for j, cid in enumerate(ids):
            zz = z[sel, j]
            top = np.argsort(-zz, kind="stable")[:k]
            row[f"{cid}__top{top_k}"] = float(zz[top].mean())
            row[f"{cid}__frac"] = float((zz > z_threshold).mean())
            row[f"{cid}__spread"] = float(spread[sel, j][top].mean())
            row[f"{cid}__text_top{top_k}"] = float(np.sort(z_text[sel, j])[::-1][:k].mean())
            cav = z_cav[sel, j]
            row[f"{cid}__cav_top{top_k}"] = float(np.sort(cav)[::-1][:k].mean()) if np.isfinite(cav).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---- eval ------------------------------------------------------------------------------------------------------------------
def envelope(name: str, cfg: Mapping[str, Any], synthetic: bool) -> dict[str, Any]:
    prov = io.provenance(cfg, synthetic=synthetic)
    return {"name": name, **{k: prov[k] for k in ("created_utc", "config_hash", "git_sha", "synthetic")}}


def merge_eval_json(artifacts_root: Path, name: str, cfg: Mapping[str, Any], synthetic: bool,
                    sections: Mapping[str, Any]) -> Path:
    """Read-modify-write an eval JSON so 04_concepts and 04_cav each keep the other's sections."""
    path = schemas.eval_json(artifacts_root, name)
    data = io.read_json(path) if path.is_file() else {}
    data.update(envelope(name, cfg, synthetic))
    data["synthetic"] = bool(synthetic)
    data.update(sections)
    return io.write_json(path, data)


def boot(cfg: Mapping[str, Any], metric: Any, arrays: Sequence[np.ndarray], groups: np.ndarray | None,
         strata: np.ndarray | None) -> dict[str, Any]:
    b = cfg["bootstrap"]
    return _clean_ci(stats.bootstrap_ci(metric, arrays, groups, strata, int(b["n"]), float(b["ci"]), int(b["seed"])))


def nct_check(root: Path, encoder: str, phr: Mapping[str, Any], ids: Sequence[str], cfg: Mapping[str, Any],
              vectors: dict[str, Any] | None) -> dict[str, Any]:
    loaded = load_tile_embeddings(root, encoder, "nct")
    out: dict[str, Any] = {"label": "cross-organ sanity check (NCT-CRC-HE-100K colorectal tiles; not breast validation)",
                           "encoder": encoder, "concepts": {}}
    if loaded is None:
        out["status"] = "skipped: no nct embeddings"
        return out
    index, arr = loaded
    labels = index["label"].to_numpy()
    score, _ = raw_scores(arr, phr, ids)
    out.update(status="ok", n_tiles=int(len(arr)), per_class_counts=index["label"].value_counts().sort_index().to_dict())
    for cid, positives in cfg["nct"]["concept_checks"].items():
        j = list(ids).index(cid)
        y = np.isin(labels, positives).astype(int)
        entry: dict[str, Any] = {"positive_classes": list(positives), "n": int(len(y)), "n_pos": int(y.sum()),
                                 "text": boot(cfg, stats.auroc, [y, score[:, j]], None, y)}
        if vectors is not None and np.isfinite(vectors["w"][j]).all():
            logit = arr @ vectors["w"][j] + vectors["b"][j]
            entry["cav"] = boot(cfg, stats.auroc, [y, logit], None, y)
        out["concepts"][cid] = entry
    return out


def bach_check(features: pd.DataFrame, cases: pd.DataFrame, bank: schemas.ConceptBank, cfg: Mapping[str, Any]
               ) -> dict[str, Any]:
    k = int(cfg["concepts"]["top_k"])
    classes = list(cfg["classes"])
    aggressive = cfg["aggressive_class"]
    test = features[(features["variant"] == "clean") & (features["split"] == "test")].merge(
        cases[["case_id", "label", "group"]], on="case_id", how="left")
    out: dict[str, Any] = {"split": "test", "feature": f"<concept>__top{k}", "n_cases": int(len(test)),
                           "n_groups": int(test["group"].nunique()) if len(test) else 0,
                           "per_class_n": {c: int((test["label"] == c).sum()) for c in classes}, "concepts": {},
                           "contradictions": [],
                           "note": "Features associated with each class on the held-out test split; expected "
                                   "directions are a sanity reference, not validation; small n."}
    if test.empty:
        out["status"] = "skipped: no clean test cases"
        return out
    groups = test["group"].replace("", np.nan).fillna(test["case_id"]).to_numpy()
    y_agg = (test["label"] == aggressive).astype(int).to_numpy()
    expected_top = cfg["concepts"].get("expected_top_class", {})
    for c in bank.concepts:
        v = test[f"{c.id}__top{k}"].to_numpy(float)
        means = {}
        for cls in classes:
            mask = (test["label"] == cls).to_numpy()
            means[cls] = boot(cfg, np.mean, [v[mask]], groups[mask], None) if mask.any() else None
        top_class = max((cls for cls in classes if means[cls]), key=lambda cls: means[cls]["estimate"])
        auc = boot(cfg, stats.auroc, [y_agg, v], groups, y_agg)
        entry = {"label": c.label, "group": c.group, "direction": c.direction, "weak": c.weak,
                 "class_means": means, "highest_class": top_class, f"auroc_{aggressive}_vs_rest": auc}
        if c.direction in ("up", "down") and auc["estimate"] is not None:
            agrees = (auc["estimate"] > 0.5) == (c.direction == "up")
            entry["direction_consistent"] = bool(agrees)
            if not agrees:
                out["contradictions"].append({"concept": c.id, "kind": "direction", "expected": c.direction,
                                              "auroc": auc["estimate"]})
        if c.id in expected_top:
            ok = top_class in expected_top[c.id]
            entry["expected_highest"] = list(expected_top[c.id])
            entry["highest_as_expected"] = bool(ok)
            if not ok:
                out["contradictions"].append({"concept": c.id, "kind": "highest_class",
                                              "expected": list(expected_top[c.id]), "observed": top_class})
        out["concepts"][c.id] = entry
    out["status"] = "ok"
    return out


def spotcheck_summary(artifacts_root: Path) -> dict[str, Any] | None:
    path = schemas.eval_json(artifacts_root, "spotcheck")
    if not path.is_file():
        return None
    data = io.read_json(path)
    return {"reviewer_kind": data.get("reviewer_kind"), "summary": data.get("summary", {}).get("concept")}


def zero_shot(root: Path, cases: pd.DataFrame, tiles: pd.DataFrame, bank: schemas.ConceptBank,
              cfg: Mapping[str, Any], limit_ids: set[str] | None) -> dict[str, Any]:
    classes = list(cfg["classes"])
    out: dict[str, Any] = {"method": "case embedding = normalized mean of QC g512 tile align embeddings; class "
                                     "embedding = normalized mean over class phrasings x templates; argmax cosine",
                           "split": "test", "encoders": {}}
    test_cases = cases[cases["split"] == "test"]
    if limit_ids is not None:
        test_cases = test_cases[test_cases["case_id"].isin(limit_ids)]
    for enc in cfg["encoders"]["text_aligned"]:
        if not schemas.text_npz(root, enc).is_file():
            out["encoders"][enc] = {"status": "skipped: no text npz"}
            continue
        loaded = load_tile_embeddings(root, enc, "bach512")
        if loaded is None:
            out["encoders"][enc] = {"status": "skipped: no bach512 align embeddings"}
            continue
        index, arr = loaded
        meta = tile_rows(cases, tiles, index, "clean", cfg, root)
        meta = meta[meta["case_id"].isin(test_cases["case_id"])]
        text = load_text(root, enc)
        cls_emb = class_embeddings(bank, text, classes)
        y, pred, grp = [], [], []
        label_of = dict(zip(cases["case_id"], cases["label"]))
        group_of = dict(zip(cases["case_id"], cases["group"]))
        for case_id, g in meta.groupby("case_id", sort=True):
            sel = g[g["qc"]] if g["qc"].any() else g
            emb = _l2(arr[sel["emb_row"].to_numpy()].mean(0))
            pred.append(int(np.argmax(cls_emb @ emb)))
            y.append(classes.index(label_of[case_id]))
            grp.append(group_of.get(case_id) or case_id)
        if not y:
            out["encoders"][enc] = {"status": "skipped: no test cases embedded"}
            continue
        y_arr, p_arr, g_arr = np.array(y), np.array(pred), np.array(grp)
        n_c = len(classes)
        out["encoders"][enc] = {
            "status": "ok", "n_cases": int(len(y)), "n_groups": int(len(set(grp))),
            "macro_f1": boot(cfg, lambda a, b: stats.macro_f1(a, b, n_c), [y_arr, p_arr], g_arr, None),
            "balanced_accuracy": boot(cfg, lambda a, b: stats.balanced_accuracy(a, b, n_c), [y_arr, p_arr], g_arr, None),
            "accuracy": boot(cfg, lambda a, b: float(np.mean(a == b)), [y_arr, p_arr], g_arr, None),
            "confusion": stats.confusion_matrix(y_arr, p_arr, n_c).tolist(), "classes": classes,
        }
    return out


# ---- main ------------------------------------------------------------------------------------------------------------------
def inputs_fingerprint(paths: Sequence[Path], extra: str) -> str:
    h = hashlib.sha256(extra.encode("utf-8"))
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
    return h.hexdigest()[:20]


def main(argv: Sequence[str] | None = None, log: Any = print) -> int:
    parser = cli.step_parser(STEP, "Concept scores per tile and case + concept checks + zero-shot (plan §6.5).")
    parser.add_argument("--encoder", default=None, help="align space for concepts (default settings concepts.encoder)")
    parser.add_argument("--concepts", default=None, help="concepts.yaml path")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root, art = ctx.cfg, ctx.data_root, ctx.artifacts_root
    encoder = args.encoder or cfg["concepts"]["encoder"]
    bank = config.load_concept_bank(args.concepts, cfg)
    ids = bank.ids
    k, zt = int(cfg["concepts"]["top_k"]), float(cfg["concepts"]["z_threshold"])

    for need in (schemas.cases_csv(root), schemas.tiles_csv(root), schemas.text_npz(root, encoder)):
        if not need.is_file():
            log(f"ABORT: missing input {need}")
            return 2
    cases = read_csv_str(schemas.cases_csv(root))
    schemas.validate_frame(schemas.CASES, cases)
    tiles = read_csv_str(schemas.tiles_csv(root))
    schemas.validate_frame(schemas.TILES, tiles)
    clean = load_tile_embeddings(root, encoder, "bach512")
    if clean is None:
        log(f"ABORT: missing emb/{encoder}/bach512_align.npy (+ index)")
        return 2
    inputs = [p for p in [schemas.cases_csv(root), schemas.tiles_csv(root), schemas.text_npz(root, encoder),
                          concept_vectors_npz(root)]
              + [schemas.emb_npy(root, encoder, s, "align") for s in ("bach512", "deg512", "comp512", "nct")]
              if p.is_file()]
    fp = inputs_fingerprint(inputs, json.dumps([config.config_hash(cfg), ids, bank.templates, ctx.limit,
                                                schemas.expand_prompts(bank)[-1].id], default=str))
    meta_path = schemas.step_meta_json(root, META_STEP)
    outs = [schemas.case_features_csv(root), schemas.concepts_tile_npz(root)]
    if not ctx.force and io.outputs_exist(*outs, meta_path) and io.read_json(meta_path).get("fingerprint") == fp:
        log(f"04_concepts: outputs are up to date for these inputs (fingerprint {fp}); use --force to redo")
        return 0

    text = load_text(root, encoder)
    phr = phrasing_embeddings(bank, text)
    vectors = load_vectors(root, ids, clean[1].shape[1])
    limit_ids = set(cli.limit_case_ids(cases["case_id"], ctx.limit)) if ctx.limit else None

    frames, embs = [], []
    for kind, set_name in (("clean", "bach512"), ("degraded", "deg512"), ("composite", "comp512")):
        loaded = clean if set_name == "bach512" else load_tile_embeddings(root, encoder, set_name)
        if loaded is None:
            log(f"04_concepts: no {set_name} embeddings: {kind} variants not scored")
            continue
        index, arr = loaded
        meta = tile_rows(cases, tiles, index, kind, cfg, root)
        if limit_ids is not None:
            meta = meta[meta["source_case_id"].isin(limit_ids)].reset_index(drop=True)
        frames.append(meta)
        embs.append(arr[meta["emb_row"].to_numpy()])
    meta = pd.concat(frames, ignore_index=True)
    x = np.concatenate(embs)
    score, phr_std = raw_scores(x, phr, ids)

    train = ((meta["variant"] == "clean") & (meta["split"] == "train") & meta["qc"]).to_numpy()
    if train.sum() < 2:
        log("ABORT: fewer than 2 clean train QC tiles: cannot z-score")
        return 2
    mu, sd = score[train].mean(0), score[train].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z_text = (score - mu) / sd
    spread = phr_std / sd
    z_cav = np.full_like(z_text, np.nan)
    source = ["text"] * len(ids)
    cav_stats: dict[str, Any] = {}
    if vectors is not None:
        for j, cid in enumerate(ids):
            if not vectors["use"][j]:
                continue
            logit = x @ vectors["w"][j] + vectors["b"][j]
            m, s = float(logit[train].mean()), float(logit[train].std()) or 1.0
            z_cav[:, j] = (logit - m) / s
            source[j] = "cav"
            cav_stats[cid] = {"train_mean": m, "train_std": s}
    z = np.where(np.isfinite(z_cav), (z_text + z_cav) / 2.0, z_text).astype(np.float32)

    features = aggregate_cases(meta, z, z_text, z_cav, spread, ids, k, zt, log)
    feat_cols = schemas.concept_feature_columns(cfg)
    order = (["case_id", "split", "variant", "source_case_id", "n_tiles"] + feat_cols
             + schemas.concept_spread_columns(cfg)
             + [f"{c}__{part}_top{k}" for c in ids for part in ("text", "cav")])
    features = features[order]
    schemas.validate_frame(schemas.CASE_FEATURES, features, extra_required=feat_cols)
    if features[feat_cols].isna().any().any():
        raise ValueError("case_features has NaN model features")
    io.write_csv(schemas.case_features_csv(root), features)
    io.save_npz(schemas.concepts_tile_npz(root),
                case_id=np.asarray(meta["case_id"].tolist(), dtype=str),
                tile_idx=meta["tile_idx"].to_numpy(np.int32), grid=np.asarray(GRID),
                concept_ids=np.asarray(ids, dtype=str), z=z, z_text=z_text.astype(np.float32),
                z_cav=z_cav.astype(np.float32), spread=spread.astype(np.float32), encoder=np.asarray(encoder),
                synthetic=np.asarray(bool(ctx.synthetic)), qc_pass=meta["qc"].to_numpy(bool),
                variant=np.asarray(meta["variant"].tolist(), dtype=str))
    counts = features["variant"].value_counts().to_dict()
    log(f"04_concepts: {len(meta)} tiles, cases by variant {counts}; sources {dict(zip(ids, source))}")

    concepts_section = {c.id: {"label": c.label, "group": c.group, "direction": c.direction, "weak": c.weak,
                               "source": source[j], "text_train_mean": float(mu[j]), "text_train_std": float(sd[j]),
                               **({"cav_train": cav_stats[c.id]} if c.id in cav_stats else {})}
                        for j, c in enumerate(bank.concepts)}
    merge_eval_json(art, "concepts", cfg, ctx.synthetic,
                    {"encoder": encoder, "nct": nct_check(root, encoder, phr, ids, cfg, vectors),
                     "concepts": concepts_section,
                     "vectors_used": None if vectors is None else vectors["sha256"][:16]})
    bach = bach_check(features, cases, bank, cfg)
    bach["spotcheck"] = spotcheck_summary(art)
    io.write_json(schemas.eval_json(art, "concepts_bach"), {**envelope("concepts_bach", cfg, ctx.synthetic), **bach})
    io.write_json(schemas.eval_json(art, "zeroshot"),
                  {**envelope("zeroshot", cfg, ctx.synthetic), **zero_shot(root, cases, tiles, bank, cfg, limit_ids)})
    if bach.get("contradictions"):
        log(f"04_concepts: expected-direction contradictions (surfaced, not hidden): {bach['contradictions']}")
    io.write_meta(meta_path, step=META_STEP, cfg=cfg, inputs=inputs, synthetic=ctx.synthetic,
                  outputs=outs + [schemas.eval_json(art, n) for n in ("concepts", "concepts_bach", "zeroshot")],
                  args={"encoder": encoder, "limit": ctx.limit, "force": ctx.force},
                  extra={"fingerprint": fp, "sources": dict(zip(ids, source)), "cases_by_variant": counts})
    return 0


if __name__ == "__main__":
    sys.exit(main())
