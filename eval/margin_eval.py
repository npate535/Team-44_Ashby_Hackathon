"""Margin-read evaluation: does the simulated margin call localize tumor or just echo the diagnosis? (BUILD_PLAN §A12)

    python -m eval.margin_eval

Inputs (data root): margins.json (threshold + per-case segments/per_margin), cases.csv, predictions.csv (risk),
composites.csv (per-margin truth), optional wsi_windows.csv (+ margins.json entries for its window ids).
A margin is "involved" when its score (max segment p) is strictly above the threshold (core.margin.involved_margins).
These are raw-score evaluations: the reliability defer rule is not applied (it is a separate safety layer).

Writes <artifacts>/eval/margin.json:
  benign_fpr        clean TEST benign (settings benign_classes) images with margin scores: per-margin and any-margin
                    rate of a false "tumor at margin" (target <= settings margin.max_fpr); cluster CI over groups.
  composite         per-segment test on composite fields (half carcinoma, half benign, test images only): the margin
                    on the benign side is an EXACT negative inside a carcinoma case. Exact-negative margins called clear
                    (and exact-negative segments below threshold): an echo of the diagnosis scores ~0%, a localizer
                    scores high. Tumor-side margins called involved = sensitivity. Clustered by carcinoma-case group.
  echo              clean TEST carcinoma images: P(>= 1 clear margin), mean involved margins, and Spearman correlation
                    of the image's max margin score with the case risk (a pure echo correlates ~1).
  wsi_windows       per-margin sensitivity / specificity against annotated WSI 256-px strip truth (clustered by
                    patient), when windows are present.
Every rate carries a CI, n and n_groups.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core import cli, io, margin, schemas
from eval.metrics import ci, load_cases, load_predictions, write_eval

STEP = "eval_margin_eval"
MARGINS = schemas.MARGINS


def _rate(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else math.nan


def rate_ci(indicator: Sequence[bool], groups: Sequence[str], cfg: Mapping[str, Any], offset: int) -> dict[str, Any]:
    x = np.asarray(indicator, dtype=float)
    if not len(x):
        return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_groups": 0}
    return ci(_rate, [x], np.asarray(groups), cfg, offset)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan
    return float(spearmanr(a, b).statistic)


def _per_margin(entry: Mapping[str, Any]) -> dict[str, float]:
    geometry = [{k: s[k] for k in ("margin", "x", "y", "size")} for s in entry["segments"]]
    return margin.per_margin_scores(geometry, [float(s["p"]) for s in entry["segments"]])


def benign_block(cases, scores, threshold, cfg) -> dict[str, Any]:
    benign = cases[(cases["split"] == "test") & cases["label"].isin(cfg["benign_classes"])]
    have = benign[benign["case_id"].isin(scores)]
    groups = have["group"].astype(str).tolist()
    pm = [scores[c] for c in have["case_id"]]
    out: dict[str, Any] = {"n_benign_test": int(len(benign)), "n_scored": int(len(have)),
                           "target_max_fpr": float(cfg["margin"]["max_fpr"]), "per_margin": {}}
    for j, m in enumerate(MARGINS):
        out["per_margin"][m] = rate_ci([p[m] > threshold for p in pm], groups, cfg, 500 + j)
    out["any_margin"] = rate_ci([any(p[m] > threshold for m in MARGINS) for p in pm], groups, cfg, 505)
    if len(have) < len(benign):
        out["note"] = (f"only {len(have)} of {len(benign)} benign test images have margin scores in margins.json "
                       "(the producer scores surgical / variant cases); the rate is over those")
    return out


def composite_block(comps, cases, entries, threshold, cfg) -> dict[str, Any]:
    group_of = cases.set_index("case_id")["group"].astype(str).to_dict()
    neg_clear, neg_groups, seg_clear, seg_groups, tum_hit, tum_groups = [], [], [], [], [], []
    per_comp = []
    for _, row in comps.iterrows():
        cid = row["case_id"]
        if cid not in entries:
            continue
        entry = entries[cid]
        pm = _per_margin(entry)
        g = group_of.get(row["carcinoma_case_id"], str(row["carcinoma_case_id"]))
        summary = {"case_id": cid, "carcinoma_side": row["carcinoma_side"], "per_margin": pm, "truth": {}}
        for m in MARGINS:
            truth = row[f"truth_{m}"]
            summary["truth"][m] = truth
            if truth == "exact_negative":
                neg_clear.append(pm[m] <= threshold)
                neg_groups.append(g)
                for s in entry["segments"]:
                    if s["margin"] == m:
                        seg_clear.append(float(s["p"]) <= threshold)
                        seg_groups.append(g)
            elif truth == "tumor_side":
                tum_hit.append(pm[m] > threshold)
                tum_groups.append(g)
        per_comp.append(summary)
    return {
        "n_composites": len(per_comp), "n_groups": len(set(neg_groups) | set(tum_groups)),
        "exact_negative_margin_clear_rate": rate_ci(neg_clear, neg_groups, cfg, 510),
        "exact_negative_segment_clear_rate": rate_ci(seg_clear, seg_groups, cfg, 511),
        "tumor_side_margin_sensitivity": rate_ci(tum_hit, tum_groups, cfg, 512),
        "interpretation": "An echo of the case diagnosis calls every margin of a carcinoma field involved (clear rate "
                          "~0% on exact negatives); a localizer clears them (rate near 1).",
        "fields": per_comp,
    }


def echo_block(cases, preds, scores, threshold, cfg) -> dict[str, Any]:
    carc = cases[(cases["split"] == "test") & cases["label"].isin(cfg["carcinoma_classes"])]
    have = carc[carc["case_id"].isin(scores)]
    groups = have["group"].astype(str).to_numpy()
    pm = [scores[c] for c in have["case_id"]]
    risk = preds.set_index("case_id")["risk"]
    max_score = np.asarray([max(p.values()) for p in pm], dtype=float)
    risks = np.asarray([float(risk.get(c, math.nan)) for c in have["case_id"]], dtype=float)
    ok = ~np.isnan(risks)
    out = {
        "n_carcinoma_test": int(len(carc)), "n_scored": int(len(have)),
        "p_at_least_one_clear_margin": rate_ci([any(p[m] <= threshold for m in MARGINS) for p in pm],
                                               groups.tolist(), cfg, 520),
        "mean_involved_margins": (ci(lambda x: float(np.mean(x)), [np.asarray([sum(p[m] > threshold for m in MARGINS)
                                                                             for p in pm], dtype=float)],
                                     groups, cfg, 521) if len(pm) else None),
        "spearman_max_margin_vs_risk": (ci(_spearman, [max_score[ok], risks[ok]], groups[ok], cfg, 522)
                                        if ok.sum() >= 3 else {"estimate": None, "n": int(ok.sum())}),
        "interpretation": "A margin read that only echoes the diagnosis puts tumor at every margin of every carcinoma "
                          "image (P(>= 1 clear) ~ 0) and tracks the case risk (Spearman ~ 1).",
    }
    return out


def wsi_block(root, entries, threshold, cfg) -> dict[str, Any] | None:
    path = schemas.wsi_windows_csv(root)
    if not path.is_file():
        return None
    windows = io.read_csv(path)
    schemas.validate_frame(schemas.WSI_WINDOWS, windows)
    windows = windows[windows["window_id"].isin(entries)]
    if windows.empty:
        return {"n_windows": 0, "note": "wsi_windows.csv present but no window has margin scores in margins.json"}
    tp_g, tn_g, sens, spec = [], [], [], []
    per_margin: dict[str, dict[str, list]] = {m: {"sens": [], "sg": [], "spec": [], "cg": []} for m in MARGINS}
    for _, w in windows.iterrows():
        pm = _per_margin(entries[w["window_id"]])
        g = str(w["patient_id"])
        for m in MARGINS:
            hit = pm[m] > threshold
            if w[f"truth_{m}"] == "tumor":
                sens.append(hit)
                tp_g.append(g)
                per_margin[m]["sens"].append(hit)
                per_margin[m]["sg"].append(g)
            else:
                spec.append(not hit)
                tn_g.append(g)
                per_margin[m]["spec"].append(not hit)
                per_margin[m]["cg"].append(g)
    return {
        "n_windows": int(len(windows)), "n_patients": int(windows["patient_id"].nunique()),
        "sensitivity": rate_ci(sens, tp_g, cfg, 530), "specificity": rate_ci(spec, tn_g, cfg, 531),
        "per_margin": {m: {"sensitivity": rate_ci(v["sens"], v["sg"], cfg, 540 + j),
                           "specificity": rate_ci(v["spec"], v["cg"], cfg, 550 + j)} for j, (m, v) in enumerate(per_margin.items())},
        "note": "WSI labels are imperfect ('unannotated' is not 'normal'; scanner shift); windows are patient-disjoint "
                "from the probe's training data.",
    }


def run(ctx: cli.StepContext) -> Any:
    cfg, root = ctx.cfg, ctx.data_root
    mpath = schemas.margins_json(root)
    if not mpath.is_file():
        raise FileNotFoundError(f"missing input margins.json: {mpath}")
    margins = io.read_json(mpath)
    problems = schemas.MARGINS_JSON.problems(margins)
    if problems:
        raise ValueError("; ".join(problems))
    threshold = float(margins["threshold"])
    entries = margins["cases"]
    cases = load_cases(cfg, root)
    preds = load_predictions(cfg, root)
    clean_test = set(cases.loc[cases["split"] == "test", "case_id"])
    scores = {c: _per_margin(e) for c, e in entries.items() if c in clean_test}
    payload: dict[str, Any] = {
        "split": "test", "threshold": threshold, "threshold_ci": margins.get("threshold_ci"),
        "n_benign_cal_images": margins.get("n_benign_cal_images"), "probe": margins.get("probe"),
        "strip_px": margins.get("strip_px"), "max_fpr": margins.get("max_fpr"),
        "benign_fpr": benign_block(cases, scores, threshold, cfg),
        "echo": echo_block(cases, preds, scores, threshold, cfg),
    }
    cpath = schemas.composites_csv(root)
    if cpath.is_file():
        comps = io.read_csv(cpath)
        schemas.validate_frame(schemas.COMPOSITES, comps)
        payload["composite"] = composite_block(comps, cases, entries, threshold, cfg)
    wsi = wsi_block(root, entries, threshold, cfg)
    if wsi is not None:
        payload["wsi_windows"] = wsi
    payload["notes"] = ["Simulated margins: image borders stand in for inked margins (top superior, bottom inferior, "
                        "left medial, right lateral); anterior/posterior are not representable in 2D.",
                        "Raw-score evaluation: the reliability defer rule is a separate safety layer.",
                        "Composite fields are built from test images only and never enter headline metrics."]
    out = write_eval(ctx.artifacts_root, "margin", payload, cfg, ctx.synthetic)
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg,
                  inputs=[mpath, schemas.cases_csv(root), schemas.predictions_csv(root)], outputs=[out],
                  synthetic=ctx.synthetic)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ctx = cli.context(cli.step_parser("eval.margin_eval", "Margin localization evaluation (benign FPR, composites, echo, WSI).").parse_args(argv))
    try:
        out = run(ctx)
    except (FileNotFoundError, ValueError) as exc:
        print(f"eval.margin_eval: {exc}", file=sys.stderr)
        return 1
    print(f"eval.margin_eval: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
