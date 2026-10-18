"""Step 06: reliability score and flag for every read (plan §6.7, BUILD_PLAN Task 10, ruling M9).

    python -m pipeline.06_reliability [--data-root D] [--artifacts-root A] [--limit N] [--force]

Features per read (settings reliability.features):
  * ood_percentile: OOD distance = mean over the read's QC tiles of the mean cosine distance to the reliability.k
    nearest TTA-averaged train tiles (CONCH align; settings reliability.ood_reference), as a percentile among clean
    calibration cases read the same way (TTA reads vs clean-calibration TTA distances; view-0 reads vs clean-
    calibration view-0 distances).
  * set_size: size of the read's tier-driving conformal set (predictions.csv set_tier_size; set_size = LAC when absent).
  * conch_musk_disagree: 1 when the production model's top class differs from the agreement model's (MUSK) top.
Combiner: standardized logistic regression P(error) fit on the calibration split, clean view-0 reads + degraded reads
(view 0, ruling M9: like-for-like single-view reads). The flag threshold is the largest p_error that still flags
reliability.target_error_recall of the calibration errors. Production reliability.csv rows are clean calibration/test
TTA reads plus every degraded and composite variant.

eval/reliability.json (test split): accuracy vs degradation type x severity (severity 0 = clean view-0 read) with
cluster-bootstrap CIs, the same curve for model A when the production model is a concept bottleneck without view-0
features (so the curve is not inflated by 8-view clean reads), AUROC of p_error for predicting errors with a CI, the
threshold and achieved recall. No coverage claim is made on degraded inputs.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import cli, io, schemas, stats
from pipeline import riskmodels as rm

STEP = "06_reliability"
VIEW0_PRED = "predictions_view0.csv"


def knn_distance(tiles: np.ndarray, ref: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """Per tile: mean cosine distance (1 - dot; unit vectors) to its k nearest reference tiles."""
    out = np.empty(len(tiles))
    for s in range(0, len(tiles), chunk):
        sims = tiles[s:s + chunk] @ ref.T
        top = np.partition(sims, -k, axis=1)[:, -k:]
        out[s:s + chunk] = 1.0 - top.mean(axis=1)
    return out


def read_distances(x: np.ndarray, index: pd.DataFrame, qc: pd.DataFrame | None, ref: np.ndarray, k: int, *,
                   composite: bool = False) -> pd.Series:
    """Mean tile kNN distance per read id (QC tiles; all tiles if none pass; composites keep every tile)."""
    ids, bags, _ = rm.tile_bags(x, index, qc, composite=composite)
    allt = np.concatenate(bags)
    d = knn_distance(allt / np.linalg.norm(allt, axis=1, keepdims=True), ref, k)
    out, pos = {}, 0
    for rid, b in zip(ids, bags):
        out[rid] = float(d[pos:pos + len(b)].mean())
        pos += len(b)
    return pd.Series(out, name="ood_distance")


def percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)


def recall_threshold(p_error: np.ndarray, is_error: np.ndarray, target: float) -> float:
    """Largest threshold t such that flagging p_error >= t catches at least `target` of the errors."""
    errs = np.sort(np.asarray(p_error)[np.asarray(is_error, dtype=bool)])[::-1]
    if len(errs) == 0:
        raise ValueError("no calibration errors: cannot set a recall-based flag threshold")
    k = math.ceil(target * len(errs) - 1e-12)
    return float(errs[max(k, 1) - 1])


class Combiner:
    """Standardized logistic regression on the reliability features -> P(error)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Combiner":
        from sklearn.linear_model import LogisticRegression

        if len(set(np.asarray(y).tolist())) < 2:
            raise ValueError("combiner needs both correct and erroneous calibration reads")
        self.mean = x.mean(0)
        sd = x.std(0)
        self.scale = np.where(sd > 1e-12, sd, 1.0)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit((x - self.mean) / self.scale, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba((x - self.mean) / self.scale)[:, 1]


def variant_of(rid: str) -> str:
    if "_FSIM_" in rid:
        return rid.split("_FSIM_")[1]
    return "composite" if ("_COMP_" in rid or rid.startswith("COMP_")) else "clean"


def source_of(rid: str, comp_src: Mapping[str, str] | None = None) -> str:
    for sep in ("_FSIM_", "_COMP_"):
        if sep in rid:
            return rid.split(sep)[0]
    if comp_src and rid in comp_src:
        return comp_src[rid]
    return rid


def run(ctx: cli.StepContext) -> dict[str, Any]:
    cfg, root = ctx.cfg, ctx.data_root
    rcfg = cfg["reliability"]
    classes = list(cfg["classes"])
    k = int(rcfg["k"])
    enc = cfg["encoders"]["primary"]
    cases = rm.read_cases(root)
    tiles = rm.read_tiles(root)
    comp_src: dict[str, str] = {}
    comp_csv = schemas.composites_csv(root)
    if comp_csv.is_file():
        comp_df = rm.read_csv(comp_csv)
        comp_src = dict(zip(comp_df["case_id"], comp_df["carcinoma_case_id"]))
    label = dict(zip(cases.case_id, cases.label))
    group = dict(zip(cases.case_id, cases.group))
    split = dict(zip(cases.case_id, cases.split))
    pred = rm.read_csv(schemas.predictions_csv(root))
    schemas.validate_frame(schemas.PREDICTIONS, pred)
    pred = pred[pred.split != "train"].set_index("case_id")
    v0 = rm.read_csv(root / VIEW0_PRED).set_index("case_id")
    choice = io.read_json(schemas.eval_json(ctx.artifacts_root, "model_choice"))

    # OOD reference: TTA-averaged train QC tiles
    x, idx = rm.load_emb(root, enc, "bach512", "align")
    q = tiles[tiles.qc_pass][rm.TILE_KEY]
    tr = idx.reset_index().merge(q, on=rm.TILE_KEY)
    ref = x[tr.loc[tr.split == "train", "index"].to_numpy()]
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    clean_idx = idx[idx.split.isin(["calibration", "test"])].reset_index(drop=True)
    d_tta = read_distances(x[idx.split.isin(["calibration", "test"]).to_numpy()], clean_idx, tiles, ref, k)
    xv, iv = rm.load_emb(root, enc, "bach512", "align", views=True)
    sel = (iv.view == 0) & iv.split.isin(["calibration", "test"])
    d_v0 = read_distances(xv[sel.to_numpy()], iv[sel].reset_index(drop=True), tiles, ref, k)
    parts = []
    if rm.emb_exists(root, enc, "deg512", "align"):
        xd, idd = rm.load_emb(root, enc, "deg512", "align")
        parts.append(read_distances(xd, idd, tiles, ref, k))
    d_deg = pd.concat(parts) if parts else pd.Series(dtype=float)
    d_comp = pd.Series(dtype=float)
    if rm.emb_exists(root, enc, "comp512", "align"):
        xc, ic = rm.load_emb(root, enc, "comp512", "align")
        d_comp = read_distances(xc, ic, None, ref, k, composite=True)
    cal_ids = sorted(c for c in d_tta.index if split[c] == "calibration")
    ref_tta, ref_v0 = d_tta.loc[cal_ids].to_numpy(), d_v0.loc[[c for c in cal_ids if c in d_v0.index]].to_numpy()

    def feats(rows: pd.DataFrame, dist: pd.Series, reference: np.ndarray) -> pd.DataFrame:
        ids = [i for i in rows.index if i in dist.index]
        r = rows.loc[ids]
        agree = pd.to_numeric(r["agree_top"], errors="coerce")
        out = pd.DataFrame({"ood_distance": dist.loc[ids].to_numpy(),
                            "ood_percentile": percentile(dist.loc[ids].to_numpy(), reference),
                            "set_size": r["set_tier_size" if "set_tier_size" in r.columns else "set_size"].astype(int).to_numpy(),
                            "conch_musk_disagree": np.where(agree.isna(), 0, 1 - agree.fillna(1)).astype(int),
                            "agree_known": agree.notna().astype(int).to_numpy(),
                            "top": r["top"].to_numpy()}, index=pd.Index(ids, name="case_id"))
        out["source_case_id"] = [source_of(i, comp_src) for i in ids]
        out["variant"] = [variant_of(i) for i in ids]
        out["split"] = [split.get(s, cfg["composites"]["source_split"] if v == "composite" else "test")
                        for s, v in zip(out["source_case_id"], out["variant"])]
        out["correct"] = [int(t == label[s]) if (v != "composite" and s in label) else -1 for t, s, v in
                          zip(out["top"], out["source_case_id"], out["variant"])]
        return out

    clean_rows = pred.loc[[i for i in pred.index if i in d_tta.index]]
    deg_rows = pred.loc[[i for i in pred.index if "_FSIM_" in i]]
    comp_rows = pred.loc[[i for i in pred.index if "_COMP_" in i or i.startswith("COMP_")]]
    f_clean = feats(clean_rows, d_tta, ref_tta)
    f_v0 = feats(v0, d_v0, ref_v0)
    f_deg = feats(deg_rows, d_deg, ref_v0)
    f_comp = feats(comp_rows, d_comp, ref_tta)
    names = list(rcfg["features"])
    fit = pd.concat([f_v0[f_v0.split == "calibration"], f_deg[f_deg.split == "calibration"]])
    dropped: list[str] = []
    if "conch_musk_disagree" in names and not fit["agree_known"].all():
        # disagreement unknown for some fit reads (no MUSK view-0/degraded read): defaulting it to 0 there would give
        # the combiner a feature whose meaning differs between fit and use, so it is left out and recorded
        names = [n for n in names if n != "conch_musk_disagree"]
        dropped.append("conch_musk_disagree")

    y_fit = 1 - fit["correct"].to_numpy()
    comb = Combiner().fit(fit[names].to_numpy(float), y_fit)
    threshold = recall_threshold(comb.predict(fit[names].to_numpy(float)), y_fit, float(rcfg["target_error_recall"]))

    out = pd.concat([f_clean, f_deg, f_comp])
    out["p_error"] = comb.predict(out[names].to_numpy(float))
    out["reliability"] = 1.0 - out["p_error"]
    out["reliability_flag"] = (out["p_error"] >= threshold).astype(int)
    out["threshold"] = threshold
    if ctx.limit is not None:
        keep = set(cli.limit_case_ids(cases[cases.split.isin(["calibration", "test"])].case_id, ctx.limit))
        out = out[out.source_case_id.isin(keep)]
    table = out.reset_index()[[c.name for c in schemas.RELIABILITY.columns]].sort_values("case_id")
    schemas.validate_frame(schemas.RELIABILITY, table)
    io.write_csv(schemas.reliability_csv(root), table)

    report = evaluate(cfg, choice, f_v0, f_deg, f_clean, comb, threshold, names, fit, y_fit, v0, pred, label, group,
                      classes)
    prov = io.provenance(cfg, synthetic=ctx.synthetic)
    doc = {"name": "reliability", "created_utc": prov["created_utc"], "config_hash": prov["config_hash"],
           "git_sha": prov["git_sha"], "synthetic": bool(ctx.synthetic), **report,
           "combiner": {"features": names, "dropped_features": dropped, "coef": comb.clf.coef_[0].tolist(), "intercept": float(comb.clf.intercept_[0]),
                        "fit_on": "calibration split: clean view-0 reads + degraded reads (view 0, ruling M9)",
                        "n_fit": int(len(fit)), "n_fit_errors": int(y_fit.sum())},
           "threshold": threshold, "target_error_recall": float(rcfg["target_error_recall"]),
           "ood": {"encoder": enc, "head": "align", "k": k, "reference": "TTA-averaged train QC tiles",
                   "percentile_reference": "clean calibration cases read the same way (TTA or view 0)"},
           "coverage_note": "No coverage claim is made on degraded or composite inputs: conformal coverage holds only "
                            "for exchangeable clean reads."}
    io.write_json(schemas.eval_json(ctx.artifacts_root, "reliability"), rm.jsonable(doc))
    return doc


def evaluate(cfg, choice, f_v0, f_deg, f_clean, comb, threshold, names, fit, y_fit, v0, pred, label, group,
             classes) -> dict[str, Any]:
    boot = cfg["bootstrap"]

    def ci(metric, arrays, groups):
        return stats.bootstrap_ci(metric, arrays, groups, None, int(boot["n"]), float(boot["ci"]), int(boot["seed"]))

    test_v0 = f_v0[f_v0.split == "test"]
    test_deg = f_deg[f_deg.split == "test"]
    curve: dict[str, Any] = {}
    read_kind = str(v0["read"].iloc[0]) if len(v0) else "none"
    acc = lambda c: float(np.mean(c))  # noqa: E731
    clean_ci = ci(acc, [test_v0["correct"].to_numpy()], np.array([group[s] for s in test_v0.source_case_id]))
    for dtype in cfg["degrade"]["types"]:
        curve[dtype] = {"0": clean_ci}
        for sev in cfg["degrade"]["severities"]:
            part = test_deg[test_deg.variant == f"{dtype}{sev}"]
            if len(part):
                curve[dtype][str(sev)] = ci(acc, [part["correct"].to_numpy()],
                                            np.array([group[s] for s in part.source_case_id]))
    result: dict[str, Any] = {
        "accuracy_vs_degradation": {"model": choice["production"], "clean_read": read_kind, "curve": curve,
                                    "note": ("clean severity-0 reads are view-0 single-view reads (like-for-like with "
                                             "degraded view-0 reads, ruling M9)") if read_kind == "view0" else
                                            ("clean severity-0 reads are TTA means (the production model has no view-0 "
                                             "features yet); the drop may be overstated — see model_a_curve")},
    }
    a = choice.get("model_a")
    if a and a != choice["production"]:
        col = [schemas.prob_col(a, c) for c in classes]
        if all(c in v0.columns for c in col) and all(c in pred.columns for c in col):
            a_curve: dict[str, Any] = {}
            vv = v0.loc[v0[col].notna().all(axis=1)]
            correct0 = np.array([int(classes[int(np.argmax(r))] == label[i]) for i, r in zip(vv.index, vv[col].to_numpy())])
            mask = np.array([cfg_split == "test" for cfg_split in vv["split"]])
            groups0 = np.array([group[i] for i in vv.index[mask]])
            a0 = ci(acc, [correct0[mask]], groups0)
            dd = pred.loc[[i for i in pred.index if "_FSIM_" in i and pred.loc[i, "split"] == "test"]]
            dd = dd[dd[col].notna().all(axis=1)]
            for dtype in cfg["degrade"]["types"]:
                a_curve[dtype] = {"0": a0}
                for sev in cfg["degrade"]["severities"]:
                    ids = [i for i in dd.index if variant_of(i) == f"{dtype}{sev}"]
                    if ids:
                        c = np.array([int(classes[int(np.argmax(dd.loc[i, col].to_numpy(float)))] == label[source_of(i)])
                                      for i in ids])
                        a_curve[dtype][str(sev)] = ci(acc, [c], np.array([group[source_of(i)] for i in ids]))
            result["model_a_curve"] = {"model": a, "clean_read": "view0", "curve": a_curve,
                                       "note": "black-box model A, single-view clean vs single-view degraded reads"}
    # error prediction on test (clean TTA production reads + degraded reads; composites excluded: no case label)
    test = pd.concat([f_clean[f_clean.split == "test"], test_deg])
    p = comb.predict(test[names].to_numpy(float))
    err = 1 - test["correct"].to_numpy()
    groups = np.array([group[s] for s in test.source_case_id])
    result["error_auroc_test"] = ci(stats.auroc, [err, p], groups)
    result["feature_auroc_test"] = {n: ci(stats.auroc, [err, test[n].to_numpy(float)], groups) for n in names}
    flag = (p >= threshold).astype(int)
    result["error_recall_test"] = ci(lambda e, f: float(f[e == 1].mean()) if (e == 1).any() else math.nan,
                                     [err, flag], groups)
    result["flag_rate_test"] = {"clean": float(flag[: (f_clean.split == "test").sum()].mean()),
                                "degraded": float(flag[(f_clean.split == "test").sum():].mean())}
    p_fit = comb.predict(fit[names].to_numpy(float))
    result["error_recall_calibration"] = float((p_fit[y_fit == 1] >= threshold).mean())
    result["n_test_reads"] = int(len(test))
    result["n_test_groups"] = int(len(set(groups)))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Reliability score and flag (OOD, set size, CONCH-MUSK disagreement).")
    ctx = cli.context(parser.parse_args(argv))
    outs = [schemas.reliability_csv(ctx.data_root), schemas.eval_json(ctx.artifacts_root, "reliability")]
    if io.outputs_exist(*outs) and not ctx.force:
        print(f"{STEP}: outputs exist; use --force to recompute")
        return 0
    need = [schemas.predictions_csv(ctx.data_root), ctx.data_root / VIEW0_PRED,
            schemas.eval_json(ctx.artifacts_root, "model_choice")]
    missing = [str(p) for p in need if not Path(p).is_file()]
    if missing:
        print(f"{STEP}: run pipeline.05_risk_model first; missing {missing}", file=sys.stderr)
        return 2
    doc = run(ctx)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg, inputs=need, outputs=outs,
                  args={"limit": ctx.limit, "force": ctx.force}, synthetic=ctx.synthetic,
                  extra={"threshold": doc["threshold"]})
    print(f"{STEP}: threshold {doc['threshold']:.3f}; test error AUROC {doc['error_auroc_test']['estimate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
