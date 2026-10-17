"""Data-driven concept vectors from mined Quilt-1M / PathGen images (Task 9, BUILD_PLAN §A11; Agent 2).

    python -m pipeline.04_cav [--force]          # laptop; run BEFORE 04_concepts so the vectors are used

Per concept c (settings concepts.ids) on the CONCH align embeddings of the mined concept images (embedding sets
settings concepts.cav.sets, manifest label "<c>:pos" / "<c>:neg", group = Quilt video / PathGen slide):
    * L2 logistic regression (C = concepts.cav.C) on standardized features, folded back into raw-space (w, b);
    * held-out AUROC from out-of-fold scores of a StratifiedGroupKFold split BY GROUP (concepts.cav.folds), so
      near-duplicate frames of one video or patches of one slide never sit on both sides; cluster-bootstrap CI
      over groups;
    * caption_consistent = held-out AUROC >= concepts.cav.min_auroc. The label is "caption-consistent": it measures
      agreement with caption co-mentions, not tissue truth, and is never called "validated";
    * invasion / duct_confined are also scored on annotated WSI regions (wsi512, region classes from
      concepts.cav.wsi_checks) when those embeddings exist.
Too few mined images (concepts.cav.min_per_class) or groups (min_groups), or no mined sets at all, leave the concept
text-only (cut order §I item 3); the reason is recorded.

Output concept_vectors.npz (data root; read by 04_concepts):
    concept_ids str [C] · w float32 [C, D] (NaN rows = no vector) · b float32 [C] · caption_consistent bool [C] ·
    auroc float32 [C] (held-out, NaN when not fit) · n_pos, n_neg, n_groups int32 [C] · encoder str [] · synthetic bool []
and section "cav" of eval/concepts.json (04_concepts keeps it; this step keeps 04_concepts' sections).
Laptop step: numpy, pandas, scikit-learn, core.stats.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core import cli, config, io, schemas, stats

STEP = "pipeline.04_cav"
META_STEP = "04_cav"
concepts_step = importlib.import_module("pipeline.04_concepts")


def grouped_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, folds: int, c: float, seed: int
                ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Out-of-fold decision scores from a StratifiedGroupKFold split by group; returns (scores, [(train, test)])."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = int(min(folds, len(np.unique(groups))))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    splits = []
    for tr, te in splitter.split(x, y, groups):
        if set(groups[tr]) & set(groups[te]):  # pragma: no cover - guaranteed by the splitter; checked anyway
            raise AssertionError("a group appears on both sides of a held-out split")
        if len(np.unique(y[tr])) < 2:
            continue
        w, b = fit_vector(x[tr], y[tr], c=c, seed=seed)
        oof[te] = x[te] @ w + b
        splits.append((tr, te))
    return oof, splits


def fit_vector(x: np.ndarray, y: np.ndarray, *, c: float, seed: int) -> tuple[np.ndarray, float]:
    """L2 logistic regression on standardized features, returned in raw space: logit = x @ w + b."""
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    model = LogisticRegression(C=c, max_iter=5000, random_state=seed).fit((x - mu) / sd, y)
    w = model.coef_[0] / sd
    return w.astype(np.float32), float(model.intercept_[0] - (model.coef_[0] * mu / sd).sum())


def _norm_class(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def load_mined(root: Path, encoder: str, sets: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frames, arrays, used = [], [], []
    for set_name in sets:
        loaded = concepts_step.load_tile_embeddings(root, encoder, set_name)
        if loaded is None:
            continue
        index, arr = loaded
        frames.append(index.assign(_set=set_name))
        arrays.append(arr)
        used.append(set_name)
    if not frames:
        return pd.DataFrame(columns=["label", "group", "item_id", "_set"]), np.zeros((0, 0), np.float32), used
    return pd.concat(frames, ignore_index=True), np.concatenate(arrays), used


def wsi_check(root: Path, encoder: str, cid: str, classes_pair: Sequence[str], vec: tuple[np.ndarray, float] | None,
              text_score: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    loaded = concepts_step.load_tile_embeddings(root, encoder, "wsi512")
    if loaded is None:
        return {"status": "skipped: no wsi512 embeddings"}
    index, arr = loaded
    lab = index["label"].map(_norm_class).to_numpy()
    pos, neg = _norm_class(classes_pair[0]), _norm_class(classes_pair[1])
    keep = np.isin(lab, [pos, neg])
    if keep.sum() == 0 or len(set(lab[keep])) < 2:
        return {"status": f"skipped: wsi512 lacks {classes_pair[0]} and {classes_pair[1]} regions"}
    y = (lab[keep] == pos).astype(int)
    groups = index["group"].replace("", np.nan).fillna(index["item_id"]).to_numpy()[keep]
    out: dict[str, Any] = {"status": "ok", "positive": classes_pair[0], "negative": classes_pair[1],
                           "n": int(keep.sum()), "n_pos": int(y.sum()), "n_groups": int(len(set(groups)))}
    out["text"] = concepts_step.boot(cfg, stats.auroc, [y, text_score(arr[keep])], groups, y)
    if vec is not None:
        out["cav"] = concepts_step.boot(cfg, stats.auroc, [y, arr[keep] @ vec[0] + vec[1]], groups, y)
    return out


def main(argv: Sequence[str] | None = None, log: Any = print) -> int:
    parser = cli.step_parser(STEP, "Concept vectors from mined Quilt/PathGen images (BUILD_PLAN §A11).")
    parser.add_argument("--encoder", default=None, help="default settings concepts.encoder")
    parser.add_argument("--concepts", default=None, help="concepts.yaml path (for the WSI text-score comparison)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    cav = cfg["concepts"]["cav"]
    encoder = args.encoder or cfg["concepts"]["encoder"]
    ids = list(cfg["concepts"]["ids"])
    index, x, used = load_mined(root, encoder, cav["sets"])
    out_path = concepts_step.concept_vectors_npz(root)
    inputs = [p for s in list(cav["sets"]) + ["wsi512", "bach512"] for p in (schemas.emb_npy(root, encoder, s, "align"),)
              if p.is_file()] + [p for p in [schemas.text_npz(root, encoder)] if p.is_file()]
    fp = concepts_step.inputs_fingerprint(inputs, json.dumps([config.config_hash(cfg), encoder], default=str))
    meta_path = schemas.step_meta_json(root, META_STEP)
    if not ctx.force and io.outputs_exist(out_path, meta_path) and io.read_json(meta_path).get("fingerprint") == fp:
        log(f"04_cav: up to date (fingerprint {fp}); use --force to redo")
        return 0

    dim = int(x.shape[1]) if len(x) else None
    if dim is None:
        bach = concepts_step.load_tile_embeddings(root, encoder, "bach512")
        dim = int(bach[1].shape[1]) if bach is not None else 1
    w = np.full((len(ids), dim), np.nan, np.float32)
    b = np.full(len(ids), np.nan, np.float32)
    auroc = np.full(len(ids), np.nan, np.float32)
    consistent = np.zeros(len(ids), bool)
    n_pos, n_neg, n_groups = (np.zeros(len(ids), np.int32) for _ in range(3))
    report: dict[str, Any] = {}

    text_scorer = None
    if schemas.text_npz(root, encoder).is_file():
        bank = config.load_concept_bank(args.concepts, cfg)
        phr = concepts_step.phrasing_embeddings(bank, concepts_step.load_text(root, encoder))
        text_scorer = lambda cid: (lambda arr: concepts_step.raw_scores(arr, phr, [cid])[0][:, 0])  # noqa: E731

    labels = index["label"].astype(str).to_numpy() if len(index) else np.array([], dtype=str)
    for j, cid in enumerate(ids):
        mask = np.isin(labels, [f"{cid}:pos", f"{cid}:neg"])
        y = (labels[mask] == f"{cid}:pos").astype(int)
        groups = (index.loc[mask, "group"].replace("", np.nan).fillna(index.loc[mask, "item_id"]).to_numpy()
                  if mask.any() else np.array([]))
        n_pos[j], n_neg[j], n_groups[j] = int(y.sum()), int(len(y) - y.sum()), int(len(set(groups)))
        entry: dict[str, Any] = {"n_pos": int(n_pos[j]), "n_neg": int(n_neg[j]), "n_groups": int(n_groups[j])}
        if not used:
            entry["status"] = "text only: no mined concept-image embeddings (quilt/pathgen sets absent)"
        elif min(n_pos[j], n_neg[j]) < int(cav["min_per_class"]):
            entry["status"] = f"text only: fewer than {cav['min_per_class']} mined positives or negatives"
        elif n_groups[j] < int(cav["min_groups"]):
            entry["status"] = f"text only: fewer than {cav['min_groups']} distinct videos/slides"
        else:
            xc = x[mask]
            oof, splits = grouped_oof(xc, y, groups, folds=int(cav["folds"]), c=float(cav["C"]), seed=int(cav["seed"]))
            ok = np.isfinite(oof)
            ci = concepts_step.boot(cfg, stats.auroc, [y[ok], oof[ok]], groups[ok], y[ok])
            w[j], b[j] = fit_vector(xc, y, c=float(cav["C"]), seed=int(cav["seed"]))
            auroc[j] = np.nan if ci["estimate"] is None else ci["estimate"]
            consistent[j] = bool(ci["estimate"] is not None and ci["estimate"] >= float(cav["min_auroc"]))
            entry.update(status="fit", heldout_auroc=ci, n_folds=len(splits), caption_consistent=bool(consistent[j]))
        if cid in cav["wsi_checks"]:
            vec = (w[j], float(b[j])) if np.isfinite(w[j]).all() else None
            entry["wsi_check"] = (wsi_check(root, encoder, cid, cav["wsi_checks"][cid], vec, text_scorer(cid), cfg)
                                  if text_scorer is not None else {"status": "skipped: no text npz"})
        entry.setdefault("caption_consistent", False)
        report[cid] = entry

    io.save_npz(out_path, concept_ids=np.asarray(ids, dtype=str), w=w, b=b, caption_consistent=consistent,
                auroc=auroc, n_pos=n_pos, n_neg=n_neg, n_groups=n_groups, encoder=np.asarray(encoder),
                synthetic=np.asarray(bool(ctx.synthetic)))
    section = {"label": "caption-consistent: held-out agreement with caption co-mentions (grouped by video/slide); "
                        "not tissue truth, not validation",
               "encoder": encoder, "sets_used": used, "min_auroc": float(cav["min_auroc"]), "concepts": report,
               "n_caption_consistent": int(consistent.sum())}
    concepts_step.merge_eval_json(ctx.artifacts_root, "concepts", cfg, ctx.synthetic, {"cav": section})
    log(f"04_cav: sets {used or 'none'}; caption-consistent {[c for c, k in zip(ids, consistent) if k] or 'none'}")
    io.write_meta(meta_path, step=META_STEP, cfg=cfg, inputs=inputs, outputs=[out_path], synthetic=ctx.synthetic,
                  args={"encoder": encoder, "force": ctx.force}, extra={"fingerprint": fp, "sets_used": used})
    return 0


if __name__ == "__main__":
    sys.exit(main())
