"""Headline classification, calibration and conformal metrics on the TEST split (plan §12, BUILD_PLAN §A5).

    python -m eval.metrics [--config ...] [--data-root ...] [--artifacts-root ...]

Inputs (data root): cases.csv (label, split, group), predictions.csv (pcal__<class>, p__<model>__<class>, set_lac,
optional set_aps / set_mondrian, temperature); optional <artifacts>/eval/model_choice.json for the model names.
Only clean test cases (rows of cases.csv with split == test) are scored: degraded and composite variants never enter
these metrics (their count is recorded as `variants_excluded`).

Writes <artifacts>/eval/:
  metrics.json      Invasive-vs-rest AUROC for Model A (black-box ceiling) and Model B (concept bottleneck) with a
                    paired A-B difference CI; production model macro AUROC, macro-F1, balanced accuracy, accuracy,
                    Invasive sensitivity/specificity, confusion matrix, InSitu<->Invasive confusion counts; every
                    evaluated model's row.
  calibration.json  ECE (equal-mass bins; biased at small n), Brier and NLL before (uncalibrated p__<production>)
                    and after temperature (pcal__), reliability-diagram data for both.
  conformal.json    coverage and mean set size for marginal LAC (drives tiers), APS and Mondrian when present; per-class
                    coverage; empty-set rate; set-size histogram; the finite-sample note: with n calibration cases,
                    coverage given the calibration set ~ Beta(k, n + 1 - k), k = ceil((n + 1)(1 - alpha)).

Every interval is a percentile cluster bootstrap over cases.csv `group` (settings bootstrap: n, ci, seed; patient /
pseudo-patient groups, §A5) and carries estimate, lo, hi, ci, n, n_groups, n_boot, n_valid.

Model names: model_choice.json `model_a` / `model_b` / `production` when present; otherwise A = the first settings
model_choice.candidates with p__ columns, B = the first settings model_choice.concept_models with p__ columns, and
production = predictions.csv `model`. The choice and its source are recorded (`models.source`).

Helpers here (envelope, load_test, ci, write_eval) are shared by the other eval scripts.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import calibration, cli, io, schemas, stats

STEP = "eval_metrics"


# ---- shared helpers ------------------------------------------------------------------------------------------------
def envelope(name: str, cfg: Mapping[str, Any], synthetic: bool) -> dict[str, Any]:
    """EVAL_ENVELOPE keys (+ provenance extras) for an eval JSON."""
    prov = io.provenance(cfg, synthetic=synthetic)
    return {"name": name, "created_utc": prov["created_utc"], "config_hash": prov["config_hash"],
            "git_sha": prov["git_sha"], "synthetic": bool(synthetic), "git_dirty": prov["git_dirty"]}


def write_eval(art: Path, name: str, payload: Mapping[str, Any], cfg: Mapping[str, Any], synthetic: bool) -> Path:
    obj = {**envelope(name, cfg, synthetic), **payload}
    missing = [k for k in schemas.EVAL_ENVELOPE if k not in obj]
    if missing:
        raise ValueError(f"eval {name}: missing envelope keys {missing}")
    return io.write_json(schemas.eval_json(art, name), _finite(obj))


def _finite(obj: Any) -> Any:
    """NaN/inf -> None so strict JSON never fails (undefined metrics are reported as null)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        return _finite(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _finite(obj.tolist())
    return obj


def bootstrap_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    b = cfg["bootstrap"]
    return {"n_boot": int(b["n"]), "ci": float(b["ci"]), "seed": int(b["seed"]), "cluster": bool(b["cluster"])}


def ci(metric: Callable[..., float], arrays: Sequence[np.ndarray], groups: np.ndarray, cfg: Mapping[str, Any],
       offset: int = 0) -> dict[str, Any]:
    """Cluster-bootstrap CI (settings bootstrap); `offset` varies the seed per metric deterministically."""
    b = bootstrap_settings(cfg)
    return stats.bootstrap_ci(metric, arrays, groups if b["cluster"] else None, None, b["n_boot"], b["ci"],
                              b["seed"] + offset)


def load_cases(cfg: Mapping[str, Any], root: Path) -> pd.DataFrame:
    path = schemas.cases_csv(root)
    if not path.is_file():
        raise FileNotFoundError(f"missing input cases.csv: {path}")
    df = io.read_csv(path)
    schemas.validate_frame(schemas.CASES, df)
    return df


def load_predictions(cfg: Mapping[str, Any], root: Path) -> pd.DataFrame:
    path = schemas.predictions_csv(root)
    if not path.is_file():
        raise FileNotFoundError(f"missing input predictions.csv: {path}")
    df = io.read_csv(path)
    schemas.validate_frame(schemas.PREDICTIONS, df, [schemas.cal_prob_col(c) for c in cfg["classes"]])
    return df


def load_test(cfg: Mapping[str, Any], root: Path) -> tuple[pd.DataFrame, int]:
    """Clean test cases joined with their predictions (y = class index). Returns (frame, variants excluded)."""
    cases = load_cases(cfg, root)
    preds = load_predictions(cfg, root)
    test = cases[cases["split"] == "test"][["case_id", "label", "group", "lane"]]
    variants = int((preds["split"] == "test").sum() - preds["case_id"].isin(test["case_id"]).sum())
    df = test.merge(preds, on="case_id", how="left", validate="one_to_one")
    missing = df.loc[df[schemas.cal_prob_col(cfg["classes"][0])].isna(), "case_id"].tolist()
    if missing:
        raise ValueError(f"{len(missing)} test cases have no predictions, e.g. {missing[:3]}")
    index = {c: i for i, c in enumerate(cfg["classes"])}
    unknown = sorted(set(df["label"]) - set(index))
    if unknown:
        raise ValueError(f"cases.csv labels {unknown} are not settings classes")
    df["y"] = df["label"].map(index).astype(int)
    df["group"] = df["group"].astype(str)
    return df.reset_index(drop=True), max(variants, 0)


def prob_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    m = df[list(cols)].to_numpy(dtype=float)
    if np.isnan(m).any():
        raise ValueError(f"NaN probabilities in {list(cols)[:2]}...")
    return m


def set_matrix(df: pd.DataFrame, column: str, classes: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(df), len(classes)), dtype=bool)
    for i, text in enumerate(df[column]):
        for member in schemas.parse_set(text):
            out[i, list(classes).index(member)] = True
    return out


def available_models(df: pd.DataFrame, cfg: Mapping[str, Any]) -> list[str]:
    classes = list(cfg["classes"])
    models = sorted({c.split("__")[1] for c in df.columns if c.startswith("p__") and c.count("__") == 2})
    return [m for m in models if all(schemas.prob_col(m, c) in df.columns for c in classes)]


def pick_models(df: pd.DataFrame, cfg: Mapping[str, Any], art: Path) -> dict[str, Any]:
    models = available_models(df, cfg)
    production = sorted(set(df["model"].astype(str)))
    if len(production) != 1:
        raise ValueError(f"predictions.csv test rows carry several production models {production}")
    out: dict[str, Any] = {"production": production[0], "available": models}
    path = schemas.eval_json(art, "model_choice")
    choice = io.read_json(path) if path.is_file() else {}
    if choice.get("model_a") in models and choice.get("model_b") in models:
        out.update(a=choice["model_a"], b=choice["model_b"], source="eval/model_choice.json")
        return out
    a = next((m for m in cfg["model_choice"]["candidates"] if m in models), None)
    b = next((m for m in cfg["model_choice"]["concept_models"] if m in models), None)
    out.update(a=a, b=b, source="settings model_choice order (model_choice.json lacks model_a/model_b)")
    return out


# ---- metric functions ----------------------------------------------------------------------------------------------
def _invasive_auroc(k: int) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda y, p: stats.auroc((y == k).astype(int), p[:, k] if p.ndim == 2 else p)


def classification_block(y: np.ndarray, probs: np.ndarray, groups: np.ndarray, cfg: Mapping[str, Any],
                         offset: int) -> dict[str, Any]:
    classes = list(cfg["classes"])
    c = len(classes)
    k = classes.index(cfg["aggressive_class"])
    pred = probs.argmax(axis=1)
    sens = lambda yy, pp: float(np.mean(pp[yy == k] == k)) if np.any(yy == k) else math.nan       # noqa: E731
    spec = lambda yy, pp: float(np.mean(pp[yy != k] != k)) if np.any(yy != k) else math.nan       # noqa: E731
    return {
        "invasive_vs_rest_auroc": ci(_invasive_auroc(k), [y, probs], groups, cfg, offset + 1),
        "macro_auroc": ci(lambda yy, pp: stats.macro_auroc(yy, pp, c), [y, probs], groups, cfg, offset + 2),
        "macro_f1": ci(lambda yy, pp: stats.macro_f1(yy, pp, c), [y, pred], groups, cfg, offset + 3),
        "balanced_accuracy": ci(lambda yy, pp: stats.balanced_accuracy(yy, pp, c), [y, pred], groups, cfg, offset + 4),
        "accuracy": ci(lambda yy, pp: float(np.mean(yy == pp)), [y, pred], groups, cfg, offset + 5),
        "invasive_sensitivity": ci(sens, [y, pred], groups, cfg, offset + 6),
        "invasive_specificity": ci(spec, [y, pred], groups, cfg, offset + 7),
    }


def confusion_block(y: np.ndarray, probs: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    pred = probs.argmax(axis=1)
    cm = stats.confusion_matrix(y, pred, len(classes))
    i, v = classes.index("InSitu") if "InSitu" in classes else None, classes.index(cfg["aggressive_class"])
    out: dict[str, Any] = {"rows": "true class", "columns": "predicted class", "classes": classes,
                           "matrix": cm.astype(int).tolist()}
    if i is not None:
        out["insitu_invasive"] = {
            "insitu_called_invasive": int(cm[i, v]), "invasive_called_insitu": int(cm[v, i]),
            "n_insitu": int(cm[i].sum()), "n_invasive": int(cm[v].sum()),
            "note": "The clinically important error; it motivates the p63/calponin rule and the Invasive safety rule.",
        }
    return out


def metrics_payload(df: pd.DataFrame, variants: int, cfg: Mapping[str, Any], art: Path) -> dict[str, Any]:
    classes = list(cfg["classes"])
    y, groups = df["y"].to_numpy(), df["group"].to_numpy()
    k = classes.index(cfg["aggressive_class"])
    models = pick_models(df, cfg, art)
    production = prob_matrix(df, [schemas.cal_prob_col(c) for c in classes])
    out: dict[str, Any] = {
        "split": "test", "n": int(len(df)), "n_groups": int(len(set(groups))), "variants_excluded": variants,
        "classes": classes, "aggressive_class": cfg["aggressive_class"], "models": models,
        "bootstrap": bootstrap_settings(cfg),
        "production": {**classification_block(y, production, groups, cfg, 0), "confusion": confusion_block(y, production, cfg),
                       "probabilities": "calibrated (pcal__)"},
        "per_model": {},
    }
    for j, m in enumerate(models["available"]):
        p = prob_matrix(df, [schemas.prob_col(m, c) for c in classes])
        out["per_model"][m] = {**classification_block(y, p, groups, cfg, 100 * (j + 1)),
                               "probabilities": "uncalibrated (p__)"}
    a, b = models.get("a"), models.get("b")
    if a and b:
        pa = prob_matrix(df, [schemas.prob_col(a, c) for c in classes])[:, k]
        pb = prob_matrix(df, [schemas.prob_col(b, c) for c in classes])[:, k]
        bs = bootstrap_settings(cfg)
        diff = stats.paired_difference_ci(lambda yy, s: stats.auroc((yy == k).astype(int), s), y, pa, pb,
                                          groups if bs["cluster"] else None, None, bs["n_boot"], bs["ci"], bs["seed"] + 999)
        out["a_vs_b"] = {"a": a, "b": b, "metric": "Invasive-vs-rest AUROC (uncalibrated scores)",
                         "auroc_a": out["per_model"][a]["invasive_vs_rest_auroc"],
                         "auroc_b": out["per_model"][b]["invasive_vs_rest_auroc"],
                         "difference_a_minus_b": {**diff, "n": int(len(df)), "n_groups": int(len(set(groups)))}}
    out["notes"] = [
        "Test split only, used once for reporting; model selection happened by patient-grouped CV inside train (§A3).",
        "Intervals: percentile cluster bootstrap over patient / pseudo-patient groups (§A5).",
        f"Small test set (n = {len(df)}, {len(set(groups))} groups): intervals are wide.",
        "Degraded and composite variants are excluded from these metrics.",
    ]
    return out


def calibration_payload(df: pd.DataFrame, cfg: Mapping[str, Any], models: Mapping[str, Any]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    y, groups = df["y"].to_numpy(), df["group"].to_numpy()
    bins = int(cfg["calibration"]["ece_bins"])
    after = prob_matrix(df, [schemas.cal_prob_col(c) for c in classes])
    blocks = {"after": after}
    prod = models["production"]
    if prod in models["available"]:
        blocks["before"] = prob_matrix(df, [schemas.prob_col(prod, c) for c in classes])
    out: dict[str, Any] = {
        "split": "test", "n": int(len(df)), "n_groups": int(len(set(groups))), "model": prod,
        "temperature": float(df["temperature"].iloc[0]) if "temperature" in df else None,
        "temperature_fit_on": cfg["calibration"]["fit_on"], "ece_bins": bins,
        "ece_binning": cfg["calibration"]["ece_binning"],
    }
    for j, (name, p) in enumerate(sorted(blocks.items())):
        out[name] = {
            "ece": ci(lambda yy, pp: calibration.ece(pp, yy, bins), [y, p], groups, cfg, 300 + 10 * j),
            "brier": ci(lambda yy, pp: calibration.brier(pp, yy), [y, p], groups, cfg, 301 + 10 * j),
            "nll": ci(lambda yy, pp: calibration.nll(pp, yy), [y, p], groups, cfg, 302 + 10 * j),
            "reliability_diagram": calibration.reliability_diagram(p, y, bins),
        }
    if "before" not in out:
        out["before_note"] = f"no uncalibrated p__{prod}__* columns in predictions.csv; only 'after' is reported"
    out["notes"] = [f"ECE uses {bins} equal-mass bins and is biased upward at small n (n = {len(df)}); read it with "
                    "Brier and NLL.", "Temperature was fit on out-of-fold train logits, never on test (§A3)."]
    return out


def beta_note(n_cal: int, alpha: float) -> dict[str, Any]:
    k = math.ceil((n_cal + 1) * (1 - alpha))
    if n_cal <= 0 or k > n_cal:
        return {"n_calibration": n_cal, "k": k, "note": "too few calibration cases: every set is the full set"}
    a, b = k, n_cal + 1 - k
    mean = a / (a + b)
    sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return {"n_calibration": n_cal, "k": k, "beta": [a, b], "mean": mean, "sd": sd,
            "note": f"Coverage given this calibration set is ~Beta({a}, {b}) (mean {mean:.3f}, SD {sd:.3f}); the "
                    f"1 - alpha = {1 - alpha:.2f} guarantee holds on average over calibration sets."}


def conformal_payload(df: pd.DataFrame, preds: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    y, groups = df["y"].to_numpy(), df["group"].to_numpy()
    alpha = float(cfg["alpha"])
    n_cal = int((preds["split"] == "calibration").sum())
    out: dict[str, Any] = {"split": "test", "n": int(len(df)), "n_groups": int(len(set(groups))), "alpha": alpha,
                           "target_coverage": 1 - alpha, "drives_tiers": f"{cfg['conformal']['method']}_{cfg['conformal']['mode']}",
                           "finite_sample": beta_note(n_cal, alpha), "methods": {}}
    columns = {"lac_marginal": "set_lac", "aps": "set_aps", "mondrian": "set_mondrian"}
    for j, (name, col) in enumerate(columns.items()):
        if col not in df.columns:
            continue
        sets = set_matrix(df, col, classes)
        cover = lambda yy, ss: float(np.mean(ss[np.arange(len(yy)), yy]))       # noqa: E731
        size = lambda ss: float(np.mean(ss.sum(axis=1)))                        # noqa: E731
        block: dict[str, Any] = {
            "coverage": ci(cover, [y, sets], groups, cfg, 400 + 10 * j),
            "mean_set_size": ci(size, [sets], groups, cfg, 401 + 10 * j),
            "empty_set_rate": float(np.mean(sets.sum(axis=1) == 0)),
            "set_size_histogram": {str(s): int(np.sum(sets.sum(axis=1) == s)) for s in range(len(classes) + 1)},
            "per_class": {},
        }
        for ci_idx, cls in enumerate(classes):
            mask = y == ci_idx
            if mask.sum() == 0:
                block["per_class"][cls] = {"n": 0}
                continue
            block["per_class"][cls] = ci(cover, [y[mask], sets[mask]], groups[mask], cfg, 402 + 10 * j + ci_idx)
        out["methods"][name] = block
    out["notes"] = ["Marginal LAC drives the worklist tiers; APS and Mondrian are secondary and never drive tiers.",
                    "Mondrian uses about a dozen calibration cases per class, so its thresholds are near-maximal "
                    "and conservative.", "No coverage claim is made for degraded or composite variants."]
    return out


# ---- main --------------------------------------------------------------------------------------------------------------
def run(ctx: cli.StepContext) -> dict[str, Path]:
    cfg, root, art = ctx.cfg, ctx.data_root, ctx.artifacts_root
    df, variants = load_test(cfg, root)
    preds = load_predictions(cfg, root)
    metrics = metrics_payload(df, variants, cfg, art)
    outputs = {
        "metrics": write_eval(art, "metrics", metrics, cfg, ctx.synthetic),
        "calibration": write_eval(art, "calibration", calibration_payload(df, cfg, metrics["models"]), cfg, ctx.synthetic),
        "conformal": write_eval(art, "conformal", conformal_payload(df, preds, cfg), cfg, ctx.synthetic),
    }
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg,
                  inputs=[schemas.cases_csv(root), schemas.predictions_csv(root)], outputs=list(outputs.values()),
                  args={"limit": ctx.limit}, synthetic=ctx.synthetic)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.metrics", "Classification, calibration and conformal metrics on the test split.")
    ctx = cli.context(parser.parse_args(argv))
    try:
        outputs = run(ctx)
    except (FileNotFoundError, ValueError) as exc:
        print(f"eval.metrics: {exc}", file=sys.stderr)
        return 1
    print("eval.metrics: wrote " + ", ".join(str(p) for p in outputs.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
