"""Chart-ready aggregates for the Model evaluation dashboard (plan §11 item 6, §12; the app's #/metrics page).

    python -m eval.analytics [--config ...] [--data-root ...] [--artifacts-root ...] [--reps N]

Writes <artifacts>/eval/analytics.json: plot-ready curves and counts that the other eval JSONs do not already carry,
computed on the clean TEST split only (eval.metrics.load_test: cases.csv split == test joined with predictions.csv,
so degraded / composite variants never enter; `variants_excluded` counts them exactly as metrics.json does).

    roc            Invasive-vs-rest ROC points (fpr, tpr, threshold) for every model with p__ columns plus the
                   production calibrated read (pcal__); one-vs-rest ROC per class for production. Every curve carries
                   its rank AUROC (core.stats.auroc, as in metrics.json) and the trapezoid area of the stored points.
    pr             Invasive precision-recall points, average precision and prevalence, same models.
    confusion      production 4 x 4 confusion matrix (rows true class, columns predicted), counts and row-normalized.
    reliability    equal-mass reliability bins (settings calibration.ece_bins) before / after temperature: top-label
                   (core.calibration.reliability_diagram, as calibration.json) and for P(aggressive class).
    risk_histogram counts of calibrated P(aggressive class) in equal-width bins, per true class.
    conformal      per method (pre-registered LAC marginal, APS, Mondrian, and the tier-driving set_tier when it is a
                   distinct set, keyed by conformal.json `drives_tiers`): per-class coverage counts, set-size
                   distribution overall and per class, aggressive-class-in-set counts, set composition counts;
                   `tier_method_key` names the block that drives tiers and `post_hoc_switch` flags a non-pre-registered
                   choice (its test coverage is optimistic).
    data_quality   the tile-QC re-qualification summary (data/meta/01q_requalify_qc_summary.json) when present.
    triage         base worklist tier (core.tiers, static, as eval.queue_sim) by true class: where invasive cases land.
    queue          per-simulated-day distributions (quantiles) of the queue-simulation waits per policy and of the
                   paired difference vs FIFO, replaying eval.queue_sim's arrival stream (same seed and reps); the
                   means are checked against queue_sim.json when it exists (`queue.consistency`).

Aggregates only: no case IDs and no per-case labels are written. Every block records its population (n, groups).
Display constants (histogram bins, rounding, quantiles) are presentation choices, not model thresholds; class names,
the aggressive class, alpha and bins for calibration come from settings.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import calibration, cli, io, schemas, stats
from eval import metrics as M

STEP = "eval_analytics"
RISK_HIST_BINS = 10                     # equal-width bins of P(aggressive) over [0, 1] (display only)
ROUND = 4                               # decimals for curve coordinates (display only)
QUANTILES = (2.5, 10, 25, 50, 75, 90, 97.5)
QUEUE_KEYS = ("aggressive_mean", "aggressive_p90", "non_flagged_mean", "surgical_mean", "all_mean", "max_wait",
              "overtime")
QUEUE_DIFF_KEYS = ("aggressive_mean", "non_flagged_mean")
CONFORMAL_COLUMNS = {"lac_marginal": "set_lac", "aps": "set_aps", "mondrian": "set_mondrian"}
PRE_REGISTERED_SET = "lac_marginal"     # plan §6.6 / BUILD_PLAN: the pre-registered tier set (settings conformal comment)
QC_SUMMARY_STEP = "01q_requalify_qc_summary"   # data-root meta written by pipeline.01q_requalify_qc (when it ran)


def _r(values: Any) -> list[float | None]:
    out = []
    for v in np.asarray(values, dtype=float).tolist():
        out.append(round(v, ROUND) if math.isfinite(v) else None)
    return out


def _trapezoid(x: Sequence[float], y: Sequence[float]) -> float:
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0)) if len(xs) > 1 else math.nan


# ---- curves ------------------------------------------------------------------------------------------------------------
def roc_curve(y_bin: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """ROC points + rank AUROC (metrics.json convention) + trapezoid area of the stored (rounded) points."""
    from sklearn.metrics import roc_curve as _roc

    n_pos, n_neg = int(np.sum(y_bin == 1)), int(np.sum(y_bin == 0))
    out: dict[str, Any] = {"n": int(len(y_bin)), "n_pos": n_pos, "n_neg": n_neg}
    if n_pos == 0 or n_neg == 0:
        return {**out, "fpr": [], "tpr": [], "threshold": [], "auroc": None, "auroc_trapezoid": None}
    fpr, tpr, thr = _roc(y_bin, scores, drop_intermediate=True)
    # sklearn's first threshold is +inf (nothing flagged yet): stored as null
    out.update(fpr=_r(fpr), tpr=_r(tpr), threshold=[None] + _r(thr[1:]),
               auroc=stats.auroc(y_bin, scores), auroc_trapezoid=_trapezoid(fpr, tpr))
    return out


def pr_curve(y_bin: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    n_pos = int(np.sum(y_bin == 1))
    out: dict[str, Any] = {"n": int(len(y_bin)), "n_pos": n_pos, "prevalence": float(n_pos / len(y_bin)) if len(y_bin) else None}
    if n_pos == 0 or n_pos == len(y_bin):
        return {**out, "recall": [], "precision": [], "threshold": [], "average_precision": None}
    precision, recall, thr = precision_recall_curve(y_bin, scores)
    order = np.argsort(recall, kind="stable")                   # ascending recall for plotting
    thr_full = np.append(thr, np.nan)                            # last point (recall 0) has no threshold
    out.update(recall=_r(recall[order]), precision=_r(precision[order]), threshold=_r(thr_full[order]),
               average_precision=float(average_precision_score(y_bin, scores)))
    return out


def class_reliability(p: np.ndarray, y_bin: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Equal-mass bins of P(class): mean predicted probability vs observed frequency, with counts."""
    order = np.argsort(p, kind="stable")
    bins = []
    total = 0.0
    for idx in np.array_split(order, int(n_bins)):
        if idx.size == 0:
            continue
        mp, ob = float(p[idx].mean()), float(y_bin[idx].mean())
        bins.append({"lo": float(p[idx].min()), "hi": float(p[idx].max()), "count": int(idx.size),
                     "mean_predicted": mp, "observed": ob})
        total += idx.size * abs(ob - mp)
    return {"bins": bins, "ece": total / len(p) if len(p) else None, "n": int(len(p))}


# ---- blocks --------------------------------------------------------------------------------------------------------------
def _matrices(df: pd.DataFrame, cfg: Mapping[str, Any], models: Mapping[str, Any]) -> dict[str, np.ndarray]:
    classes = list(cfg["classes"])
    out = {"production": M.prob_matrix(df, [schemas.cal_prob_col(c) for c in classes])}
    for m in models["available"]:
        out[m] = M.prob_matrix(df, [schemas.prob_col(m, c) for c in classes])
    return out


def roc_block(y: np.ndarray, mats: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    k = classes.index(cfg["aggressive_class"])
    yb = (y == k).astype(int)
    return {
        "positive_class": cfg["aggressive_class"],
        "invasive_vs_rest": {name: roc_curve(yb, p[:, k]) for name, p in mats.items()},
        "production_one_vs_rest": {c: roc_curve((y == i).astype(int), mats["production"][:, i])
                                   for i, c in enumerate(classes)},
        "note": "production = calibrated probabilities (pcal__); every other entry uses that model's uncalibrated p__ "
                "scores, as metrics.json per_model. threshold = the score at which a case starts being flagged.",
    }


def pr_block(y: np.ndarray, mats: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> dict[str, Any]:
    k = list(cfg["classes"]).index(cfg["aggressive_class"])
    yb = (y == k).astype(int)
    return {"positive_class": cfg["aggressive_class"], "models": {n: pr_curve(yb, p[:, k]) for n, p in mats.items()}}


def confusion_block(y: np.ndarray, probs: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    cm = stats.confusion_matrix(y, probs.argmax(axis=1), len(classes))
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    return {"classes": classes, "rows": "true class", "columns": "predicted class (argmax of calibrated probabilities)",
            "counts": cm.astype(int).tolist(), "row_normalized": [[round(float(v), 6) for v in r] for r in norm],
            "row_totals": rows.ravel().astype(int).tolist(), "n": int(cm.sum())}


def reliability_block(y: np.ndarray, df: pd.DataFrame, mats: Mapping[str, np.ndarray], cfg: Mapping[str, Any],
                      production: str) -> dict[str, Any]:
    classes = list(cfg["classes"])
    k = classes.index(cfg["aggressive_class"])
    bins = int(cfg["calibration"]["ece_bins"])
    sides = {"after": mats["production"]}
    if production in mats:
        sides["before"] = mats[production]
    out: dict[str, Any] = {"n_bins": bins, "binning": "equal_mass", "model": production,
                           "temperature": float(df["temperature"].iloc[0]) if "temperature" in df else None,
                           "top_label": {}, "aggressive": {}}
    for side, p in sides.items():
        out["top_label"][side] = calibration.reliability_diagram(p, y, bins)
        out["aggressive"][side] = class_reliability(p[:, k], (y == k).astype(int), bins)
    out["note"] = ("top_label: confidence of the top class vs its accuracy (as calibration.json); aggressive: "
                   "P(aggressive class) vs the observed aggressive fraction. Equal-mass bins are biased at small n.")
    return out


def histogram_block(y: np.ndarray, probs: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    k = classes.index(cfg["aggressive_class"])
    edges = np.linspace(0.0, 1.0, RISK_HIST_BINS + 1)
    by_class = {c: np.histogram(probs[y == i, k], bins=edges)[0].astype(int).tolist() for i, c in enumerate(classes)}
    return {"probability": f"calibrated P({cfg['aggressive_class']})", "bin_edges": _r(edges), "by_class": by_class,
            "n_by_class": {c: int(np.sum(y == i)) for i, c in enumerate(classes)}}


def tier_set_name(cfg: Mapping[str, Any]) -> str:
    """Name of the tier-driving set as conformal.json `drives_tiers` (core.conformal.tier_set_name when present)."""
    from core import conformal

    fn = getattr(conformal, "tier_set_name", None)
    if callable(fn):
        return str(fn(cfg))
    name = f"{cfg['conformal']['method']}_{cfg['conformal']['mode']}"
    return name + ("_nonempty" if bool(cfg["conformal"].get("nonempty", False)) else "")


def conformal_columns(df: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, str]:
    """Method name -> predictions.csv set column, as eval/metrics.py: set_tier is added under the tier-set name when
    it is a distinct set (e.g. APS with the argmax added to empty sets)."""
    columns = dict(CONFORMAL_COLUMNS)
    name = tier_set_name(cfg)
    if "set_tier" in df.columns and name not in ("lac_marginal", "aps_marginal"):
        columns[name] = "set_tier"
    return {k: v for k, v in columns.items() if v in df.columns}


def conformal_block(df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    k = classes.index(cfg["aggressive_class"])
    alpha = float(cfg["alpha"])
    drives = tier_set_name(cfg)
    columns = conformal_columns(df, cfg)
    driver = drives if drives in columns else ("aps" if drives == "aps_marginal" and "aps" in columns else
                                               PRE_REGISTERED_SET)
    out: dict[str, Any] = {"alpha": alpha, "target_coverage": 1 - alpha, "drives_tiers": drives,
                           "tier_method_key": driver, "pre_registered": PRE_REGISTERED_SET,
                           "post_hoc_switch": driver != PRE_REGISTERED_SET, "methods": {}}
    for name, col in columns.items():
        if col not in df.columns:
            continue
        sets = M.set_matrix(df, col, classes)
        size = sets.sum(axis=1)
        covered = sets[np.arange(len(y)), y]
        per_class = {}
        for i, c in enumerate(classes):
            mask = y == i
            per_class[c] = {"n": int(mask.sum()), "n_groups": int(len(set(groups[mask].tolist()))),
                            "covered": int(covered[mask].sum()),
                            "coverage": float(covered[mask].mean()) if mask.any() else None,
                            "set_size": {str(s): int(np.sum(size[mask] == s)) for s in range(len(classes) + 1)},
                            "aggressive_in_set": int(sets[mask, k].sum())}
        comp = Counter(schemas.format_set([c for j, c in enumerate(classes) if row[j]], classes) for row in sets)
        out["methods"][name] = {
            "column": col, "n": int(len(y)), "covered": int(covered.sum()), "coverage": float(covered.mean()),
            "mean_set_size": float(size.mean()), "empty_sets": int(np.sum(size == 0)),
            "set_size": {str(s): int(np.sum(size == s)) for s in range(len(classes) + 1)},
            "per_class": per_class,
            "composition": [{"members": schemas.parse_set(key), "count": int(n)}
                            for key, n in sorted(comp.items(), key=lambda kv: (-kv[1], kv[0]))],
        }
    out["note"] = ("Counts on the clean test split. An empty set means no class reached the conformal threshold; "
                   "core.tiers sends it to tier 3 (low confidence). Coverage CIs are in conformal.json.")
    if out["post_hoc_switch"]:
        out["selection_note"] = (f"{driver} drives the tiers; {PRE_REGISTERED_SET} was pre-registered. The switch was "
                                 "made after viewing test results, so the tier set's test coverage is optimistic.")
    return out


def data_quality_block(root: Path) -> dict[str, Any] | None:
    """Tile-QC re-qualification summary (all 400 images, every split) when pipeline.01q_requalify_qc has run."""
    path = schemas.step_meta_json(root, QC_SUMMARY_STEP)
    if not path.is_file():
        return None
    src = io.read_json(path)
    keep = ("rule_old", "rule_new", "n_cases", "n_tiles", "validation_old_rule_display_vs_fullres", "qc_pass_rate",
            "insufficient_tissue_cases", "mean_contract_tiles_kept")
    return {**{k: src[k] for k in keep if k in src}, "source": f"data/meta/{path.name}",
            "population": "every BACH image (all splits); tile-level QC, not a model result"}


def triage_block(cfg: Mapping[str, Any], root: Path) -> dict[str, Any]:
    from core import tiers
    from eval import queue_sim

    pool = queue_sim.test_cases(cfg, root)
    classes = list(cfg["classes"])
    by_class = {c: {str(t): 0 for t in (1, 2, 3, 4)} for c in classes}
    reasons = {c: Counter() for c in classes}
    for p in pool:
        by_class[p["truth"]][str(p["tier"])] += 1
        reasons[p["truth"]].update(p["reasons"])
    aggressive = cfg["aggressive_class"]
    return {"n": len(pool), "tiers_by_class": by_class,
            "reasons_by_class": {c: {r: int(reasons[c][r]) for r in tiers.REASON_ORDER if reasons[c][r]}
                                 for c in classes},
            "aggressive_not_flagged": sum(p["truth"] == aggressive and p["tier"] > 2 for p in pool),
            "note": "Static base tiers (no aging) of the clean test cases, as eval.queue_sim's pool; tier 1 = cases "
                    "assigned to the simulated surgical lane."}


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if not len(arr):
        return {"n_days": 0}
    out = {"n_days": int(len(arr)), "mean": float(arr.mean())}
    out.update({f"p{q:g}".replace(".", "_"): float(np.percentile(arr, q)) for q in QUANTILES})
    return out


def queue_block(cfg: Mapping[str, Any], root: Path, art: Path, reps: int | None) -> dict[str, Any]:
    """Per-day distributions, replaying eval.queue_sim.simulate's arrival stream (same seed, same reps)."""
    from eval import queue_sim

    q = cfg["queue_sim"]
    reps = int(q["reps"]) if reps is None else int(reps)
    pool = queue_sim.test_cases(cfg, root)
    runs = queue_sim.policy_runs(cfg)
    rate, hours = float(q["arrivals_per_hour"]), float(q["day_hours"])
    rng = random.Random(int(q["seed"]))
    days: dict[str, list[dict[str, float]]] = {p: [] for p in runs}
    for _ in range(reps):
        t, arrivals = 0.0, []
        while True:
            t += rng.expovariate(rate / 60.0)
            if t > hours * 60.0:
                break
            arrivals.append((t, rng.randrange(len(pool))))
        for name, (policy, run_cfg) in runs.items():
            days[name].append(queue_sim.day_metrics(queue_sim.simulate_day(pool, arrivals, policy, run_cfg), pool, run_cfg))
    keys = list(QUEUE_KEYS) + [f"class_{c}_mean" for c in cfg["classes"]]
    out: dict[str, Any] = {"reps": reps, "seed": int(q["seed"]), "unit": "minutes",
                           "policies": {p: cfg["eval"]["queue_policies"].get(p, p) for p in runs},
                           "distributions": {}, "vs_fifo": {}}
    for name, per_day in days.items():
        out["distributions"][name] = {k: _quantiles([d[k] for d in per_day]) for k in keys if per_day and k in per_day[0]}
    if "fifo" in days:
        for name, per_day in days.items():
            if name == "fifo":
                continue
            block = {}
            for k in QUEUE_DIFF_KEYS:
                diffs = [d[k] - f[k] for d, f in zip(per_day, days["fifo"])]
                clean = [v for v in diffs if not math.isnan(v)]
                block[k] = {**_quantiles(diffs),
                            "share_days_lower": float(np.mean([v < 0 for v in clean])) if clean else None,
                            "share_days_higher": float(np.mean([v > 0 for v in clean])) if clean else None}
            out["vs_fifo"][name] = block
    out["consistency"] = _queue_consistency(out, art, reps)
    out["note"] = ("Quantiles over simulated days (simulation uncertainty only, not test-set sampling). Same arrival "
                   "stream as eval.queue_sim (common random numbers); see queue_sim.json for the stated assumptions.")
    return out


def _queue_consistency(block: Mapping[str, Any], art: Path, reps: int) -> dict[str, Any]:
    path = schemas.eval_json(art, "queue_sim")
    if not path.is_file():
        return {"checked": False, "reason": "queue_sim.json not present"}
    ref = io.read_json(path)
    if int(ref.get("reps", -1)) != reps:
        return {"checked": False, "reason": f"queue_sim.json used {ref.get('reps')} reps, analytics {reps}"}
    worst = 0.0
    for name, dist in block["distributions"].items():
        for k, d in dist.items():
            mean, other = d.get("mean"), ((ref.get("results") or {}).get(name) or {}).get(k, {}).get("mean")
            if mean is not None and other is not None:
                worst = max(worst, abs(mean - other))
    return {"checked": True, "max_abs_mean_difference_min": worst, "matches_queue_sim": worst < 1e-6}


# ---- main --------------------------------------------------------------------------------------------------------------
def payload(ctx: cli.StepContext, reps: int | None = None) -> tuple[dict[str, Any], list[Path]]:
    cfg, root, art = ctx.cfg, ctx.data_root, ctx.artifacts_root
    df, variants = M.load_test(cfg, root)
    classes = list(cfg["classes"])
    y, groups = df["y"].to_numpy(), df["group"].to_numpy()
    models = M.pick_models(df, cfg, art)
    mats = _matrices(df, cfg, models)
    inputs = [schemas.cases_csv(root), schemas.predictions_csv(root)]
    out: dict[str, Any] = {
        "split": "test", "n": int(len(df)), "n_groups": int(len(set(groups.tolist()))), "variants_excluded": variants,
        "classes": classes, "aggressive_class": cfg["aggressive_class"], "alpha": float(cfg["alpha"]),
        "n_by_class": {c: int(np.sum(y == i)) for i, c in enumerate(classes)},
        "groups_by_class": {c: int(len(set(groups[y == i].tolist()))) for i, c in enumerate(classes)},
        "models": {k: models.get(k) for k in ("production", "a", "b", "available", "source")},
        "roc": roc_block(y, mats, cfg),
        "pr": pr_block(y, mats, cfg),
        "confusion": confusion_block(y, mats["production"], cfg),
        "reliability": reliability_block(y, df, mats, cfg, models["production"]),
        "risk_histogram": histogram_block(y, mats["production"], cfg),
        "conformal": conformal_block(df, y, groups, cfg),
        "unavailable": {},
    }
    dq = data_quality_block(root)
    if dq is not None:
        out["data_quality"] = dq
        inputs.append(schemas.step_meta_json(root, QC_SUMMARY_STEP))
    for name, fn, extra in (("triage", lambda: triage_block(cfg, root), [schemas.reliability_csv(root), schemas.tiles_csv(root)]),
                            ("queue", lambda: queue_block(cfg, root, art, reps), [])):
        try:
            out[name] = fn()
            inputs += [p for p in extra if p not in inputs]
        except (FileNotFoundError, ValueError, KeyError) as exc:
            out["unavailable"][name] = f"{type(exc).__name__}: {exc}"
    used_eval = [n for n in ("model_choice", "queue_sim") if schemas.eval_json(art, n).is_file()]
    out["sources"] = {"data": [p.name for p in inputs], "eval": [f"eval/{n}.json" for n in used_eval],
                      "population": "clean test split (cases.csv split == test); simulated degraded / composite "
                                    "variants excluded as in eval/metrics.py"}
    out["notes"] = ["Aggregates for plotting; headline estimates and their cluster-bootstrap CIs live in metrics.json, "
                    "calibration.json and conformal.json.",
                    f"Small test set (n = {len(df)}, {out['n_groups']} patient / pseudo-patient groups)."]
    return out, inputs + [schemas.eval_json(art, n) for n in used_eval]


def run(ctx: cli.StepContext, reps: int | None = None) -> Path:
    body, inputs = payload(ctx, reps)
    path = M.write_eval(ctx.artifacts_root, "analytics", body, ctx.cfg, ctx.synthetic)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg, inputs=inputs, outputs=[path],
                  args={"reps": reps}, synthetic=ctx.synthetic)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.analytics", "Chart-ready test-split aggregates for the Model evaluation page.")
    parser.add_argument("--reps", type=int, default=None, help="queue replications (default settings queue_sim.reps)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    try:
        path = run(ctx, args.reps)
    except (FileNotFoundError, ValueError) as exc:
        print(f"eval.analytics: {exc}", file=sys.stderr)
        return 1
    print(f"eval.analytics: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
