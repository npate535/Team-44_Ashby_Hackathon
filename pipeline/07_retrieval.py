"""Step 07: similar-case retrieval over the train reference library (plan §6.8, BUILD_PLAN Task 10).

    python -m pipeline.07_retrieval [--data-root D] [--artifacts-root A] [--limit N] [--force]

Case embedding = re-normalized mean of the case's QC-passing CONCH align tile embeddings (TTA means; settings
retrieval.encoder / head). Library = train cases (retrieval.library_split). For every calibration/test case and every
degraded / composite variant: cosine top-k neighbors (retrieval.k), neighbor agreement with the PREDICTED class
(predictions.csv `top`, never the true label), the contrast case = nearest library case whose label differs from the
predicted class, and differs_most_in = the retrieval.differs_most_in_k concepts with the largest |delta z| between
query and contrast case features (case_features.csv c__top<k>, z-scored on train cases; [] when absent).
Writes retrieval.json (data root) and eval/retrieval.json: precision@k on the test split (neighbor label == query
true label) with a cluster-bootstrap CI over patient groups.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from core import cli, io, schemas, stats
from pipeline import riskmodels as rm

STEP = "07_retrieval"


def case_embeddings(root, encoder: str, head: str, tiles: pd.DataFrame) -> pd.DataFrame:
    """Unit-norm mean of QC tile embeddings per read (clean cases, degraded and composite variants)."""
    frames = []
    for set_name, comp in (("bach512", False), ("deg512", False), ("comp512", True)):
        if not rm.emb_exists(root, encoder, set_name, head):
            if set_name == "bach512":
                raise FileNotFoundError(f"emb/{encoder}/{set_name}_{head} missing")
            continue
        x, idx = rm.load_emb(root, encoder, set_name, head)
        frames.append(rm.pool_tiles(x, idx, None if comp else tiles, composite=comp).frame())
    emb = pd.concat(frames)
    v = emb.to_numpy(np.float64)
    return pd.DataFrame(v / np.linalg.norm(v, axis=1, keepdims=True), index=emb.index)


def run(ctx: cli.StepContext) -> dict[str, Any]:
    cfg, root = ctx.cfg, ctx.data_root
    rcfg = cfg["retrieval"]
    k, kd = int(rcfg["k"]), int(rcfg["differs_most_in_k"])
    cases = rm.read_cases(root)
    tiles = rm.read_tiles(root)
    label = dict(zip(cases.case_id, cases.label))
    group = dict(zip(cases.case_id, cases.group))
    pred = rm.read_csv(schemas.predictions_csv(root))
    schemas.validate_frame(schemas.PREDICTIONS, pred)
    predicted = dict(zip(pred.case_id, pred.top))
    emb = case_embeddings(root, rcfg["encoder"], rcfg["head"], tiles)
    lib_ids = sorted(cases[cases.split == rcfg["library_split"]].case_id)
    lib = emb.loc[lib_ids].to_numpy()
    lib_labels = np.array([label[i] for i in lib_ids])

    feats = rm.concept_frame(root, cfg)
    concept_ids = list(cfg["concepts"]["ids"])
    top_cols = [f"{c}__top{int(cfg['concepts']['top_k'])}" for c in concept_ids]
    if feats is not None:
        z = feats[top_cols].astype(float)
        train_rows = z.loc[[i for i in lib_ids if i in z.index]]
        z = (z - train_rows.mean()) / train_rows.std(ddof=0).replace(0, 1.0)
    else:
        z = None

    comp_src: dict[str, str] = {}
    comp_csv = schemas.composites_csv(root)
    if comp_csv.is_file():
        comp_df = rm.read_csv(comp_csv)
        comp_src = dict(zip(comp_df["case_id"], comp_df["carcinoma_case_id"]))

    queries = [i for i in emb.index if i not in set(lib_ids) and i in predicted]
    if ctx.limit is not None:
        keep = set(cli.limit_case_ids(cases[cases.split.isin(["calibration", "test"])].case_id, ctx.limit))
        queries = [q for q in queries if comp_src.get(q, q.split("_FSIM_")[0].split("_COMP_")[0]) in keep]
    sims = emb.loc[queries].to_numpy() @ lib.T
    out: dict[str, Any] = {}
    for qi, q in enumerate(queries):
        order = np.argsort(-sims[qi], kind="stable")
        top = order[:k]
        neighbors = [{"case_id": lib_ids[j], "label": str(lib_labels[j]), "similarity": round(float(sims[qi, j]), 6)}
                     for j in top]
        p = predicted[q]
        agree = int(sum(n["label"] == p for n in neighbors))
        contrast = None
        diff = order[lib_labels[order] != p]
        if len(diff):
            j = int(diff[0])
            dz: list[str] = []
            if z is not None and q in z.index and lib_ids[j] in z.index:
                delta = (z.loc[q] - z.loc[lib_ids[j]]).abs().to_numpy()
                dz = [concept_ids[t] for t in np.argsort(-delta, kind="stable")[:kd]]
            contrast = {"case_id": lib_ids[j], "label": str(lib_labels[j]), "similarity": round(float(sims[qi, j]), 6),
                        "differs_most_in": dz}
        out[q] = {"neighbors": neighbors, "predicted": p, "agree": agree, "of": len(neighbors),
                  "neighbor_agreement": agree / len(neighbors), "contrast": contrast}
    prov = io.provenance(cfg, synthetic=ctx.synthetic)
    doc = {"encoder": rcfg["encoder"], "head": rcfg["head"], "k": k, "library_split": rcfg["library_split"],
           "n_library": len(lib_ids), "cases": out, "created_utc": prov["created_utc"],
           "config_hash": prov["config_hash"], "synthetic": bool(ctx.synthetic)}
    problems = schemas.RETRIEVAL.problems(doc)
    if problems:
        raise ValueError(problems)
    io.write_json(schemas.retrieval_json(root), rm.jsonable(doc))

    # precision@k on test (true labels; evaluation only)
    test_ids = [q for q in queries if q in label and cases.set_index("case_id").split[q] == "test"]
    prec = np.array([np.mean([n["label"] == label[q] for n in out[q]["neighbors"]]) for q in test_ids])
    agree_correct = np.array([out[q]["neighbor_agreement"] for q in test_ids])
    boot = cfg["bootstrap"]
    groups = np.array([group[q] for q in test_ids])
    ev = {"name": "retrieval", "created_utc": prov["created_utc"], "config_hash": prov["config_hash"],
          "git_sha": prov["git_sha"], "synthetic": bool(ctx.synthetic), "k": k, "encoder": rcfg["encoder"],
          "library": f"{len(lib_ids)} {rcfg['library_split']} cases",
          "precision_at_k_test": stats.bootstrap_ci(lambda v: float(np.mean(v)), [prec], groups, None, int(boot["n"]),
                                                     float(boot["ci"]), int(boot["seed"])),
          "mean_neighbor_agreement_with_prediction_test": float(agree_correct.mean()) if len(test_ids) else None,
          "per_class_precision_at_k_test": {c: float(prec[[label[q] == c for q in test_ids]].mean())
                                            for c in cfg["classes"] if any(label[q] == c for q in test_ids)},
          "note": "precision@k = share of the k train neighbors whose label equals the query's true label (test split)"}
    io.write_json(schemas.eval_json(ctx.artifacts_root, "retrieval"), rm.jsonable(ev))
    return ev


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Similar-case retrieval over the train library.")
    ctx = cli.context(parser.parse_args(argv))
    outs = [schemas.retrieval_json(ctx.data_root), schemas.eval_json(ctx.artifacts_root, "retrieval")]
    if io.outputs_exist(*outs) and not ctx.force:
        print(f"{STEP}: outputs exist; use --force to recompute")
        return 0
    if not schemas.predictions_csv(ctx.data_root).is_file():
        print(f"{STEP}: run pipeline.05_risk_model first", file=sys.stderr)
        return 2
    ev = run(ctx)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg,
                  inputs=[schemas.predictions_csv(ctx.data_root), "emb/"], outputs=outs,
                  args={"limit": ctx.limit, "force": ctx.force}, synthetic=ctx.synthetic)
    print(f"{STEP}: precision@{ev['k']} (test) = {ev['precision_at_k_test']['estimate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
