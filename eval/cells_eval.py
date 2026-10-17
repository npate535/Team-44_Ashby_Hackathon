"""Cell-feature validation (CELLS_SPEC §8; BUILD_PLAN Task 8). Laptop step: python -m eval.cells_eval.

All statistics on the TEST split with cluster-bootstrap 95% CIs (core.stats.bootstrap_ci over cases.csv `group`,
settings.bootstrap). Writes <artifacts>/eval/cells.json (EVAL_ENVELOPE + sections) and PNGs via schemas.eval_asset:
  1. review_sheet      20 overlays (cells.review per class = 5 per class), picked with labels but shown in a seeded
                       shuffled order with no label: eval/cells_review.html (checkbox grid, overlays embedded) +
                       eval/cells_review.csv (item_id, case_id, tile_idx, overlay, verdict, note). Reviews come back
                       through the spot-check flow; cells.json records the sheet, not verdicts.
  2. direction_checks  per-class means of each feature (config/cell_features.yaml) with CIs, and the difference
                       Invasive - (Normal + Benign) with a CI vs the expected direction: "as_expected", "wrong_way"
                       (CI excludes 0 on the wrong side: reported, never hidden), "inconclusive" (CI covers 0), or
                       "context" (no expected direction).
  3. convergent_validity  Spearman, tile level on the same segmented tiles: nuc_area_cv / nuc_shape_cv vs the
                       `pleomorphism` concept z and infl_fraction vs `lymphocytes` (concepts_tile.npz, Agent 2), plus
                       the case-level version (case feature vs mean concept z over the selected tiles).
  4. ablation          B / B+C / C rows read from <artifacts>/eval/model_choice.json (Agent 3), copied verbatim.
  5. faithfulness      the cell-feature part of <artifacts>/eval/deletion.json (Agent 3), copied verbatim.
A missing input makes its section {"status": "pending", "reason": ...}; numbers are never invented. With
cells.enabled false, cells.json is {"status": "disabled"} plus the envelope. Wording: "measured features associated
with invasive carcinoma"; never "this cell caused".
"""

from __future__ import annotations

import base64
import html
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core import cli, io, schemas, stats
from pipeline import cells_convert as cc

STEP = "cells_eval"
CLASS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # dataviz reference categorical slots 1-4, fixed order
VALIDITY_PAIRS = (("nuc_area_cv", "pleomorphism"), ("nuc_shape_cv", "pleomorphism"), ("infl_fraction", "lymphocytes"))
CONCEPT_MODELS_KEYS = ("model", "name", "candidate", "id")


def pending(reason: str) -> dict[str, Any]:
    return {"status": "pending", "reason": reason}


def _finite(d: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe copy: NaN/inf -> None."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def _boot(metric: Any, arrays: Sequence[np.ndarray], groups: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    b = cfg["bootstrap"]
    return _finite(stats.bootstrap_ci(metric, arrays, groups if b["cluster"] else None, None, int(b["n"]),
                                      float(b["ci"]), int(b["seed"])))


# ---- 1. review sheet ---------------------------------------------------------------------------------------------
def review_sheet(sel: pd.DataFrame, cases: pd.DataFrame, root: Path, art: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    per_class = int(cfg["cells"]["review_per_class"])
    rng = np.random.default_rng(int(cfg["seed"]))
    test = sel[sel["split"] == "test"].merge(cases[["case_id", "label"]], on="case_id")
    test = test[[schemas.cell_overlay_png(root, c, int(i)).is_file() for c, i in zip(test["case_id"], test["tile_idx"])]]
    picked = []
    for cls in cfg["classes"]:
        top = test[(test["label"] == cls) & (test["rank"] == 1)].sort_values("case_id")
        if len(top):
            picked.append(top.iloc[sorted(rng.choice(len(top), size=min(per_class, len(top)), replace=False))])
    if not picked:
        return pending("no segmented test tiles with overlays")
    items = pd.concat(picked).iloc[rng.permutation(sum(len(p) for p in picked))].reset_index(drop=True)
    items["item_id"] = [f"cells_{k + 1:02d}" for k in range(len(items))]
    items["overlay"] = [schemas.cell_overlay_png(root, c, int(i)).relative_to(root).as_posix()
                        for c, i in zip(items["case_id"], items["tile_idx"])]
    items["verdict"], items["note"] = "", ""
    sheet = items[["item_id", "case_id", "tile_idx", "overlay", "verdict", "note"]]  # no label column
    io.write_csv(schemas.eval_asset(art, "cells_review.csv"), sheet)
    colors = cfg["cells"]["overlay_colors"]
    legend = " ".join(f'<span style="color:{v}">&#9632;</span> {html.escape(k)}' for k, v in colors.items())
    cards = []
    for r in sheet.itertuples(index=False):
        data = base64.b64encode((root / r.overlay).read_bytes()).decode("ascii")
        cards.append(f'<figure><img src="data:image/png;base64,{data}" alt="{r.item_id}">'
                     f'<figcaption>{r.item_id} · {html.escape(r.case_id)} tile {int(r.tile_idx)}<br>'
                     f'<label><input type="checkbox" name="{r.item_id}"> acceptable</label></figcaption></figure>')
    page = ("<!doctype html><meta charset='utf-8'><title>Cells overlay review</title>"
            "<style>body{font-family:sans-serif;margin:16px}main{display:grid;grid-template-columns:repeat(auto-fill,"
            "minmax(260px,1fr));gap:12px}img{width:100%}figure{margin:0}</style>"
            "<h1>Cells overlay review</h1><p>Research prototype — not for clinical use. Mark an overlay acceptable when "
            "the outlines are mostly on nuclei and tumor vs immune-cell colors look plausible. Class labels are hidden. "
            f"Legend: {legend}. Record results in cells_review.csv (verdict: acceptable | not_acceptable | unsure) or the "
            "Spot-check view; say whether the reviewer is a human or an AI.</p><main>" + "".join(cards) + "</main>")
    io.atomic_write_text(schemas.eval_asset(art, "cells_review.html"), page)
    return {"status": "ok", "n_items": int(len(sheet)), "per_class_target": per_class,
            "files": ["eval/cells_review.html", "eval/cells_review.csv"], "reviewed": False}


# ---- 2. direction checks -------------------------------------------------------------------------------------------
def direction_checks(df: pd.DataFrame, fcfg: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """df: test cases with label, group and feature columns."""
    agg, ref = cfg["aggressive_class"], list(cfg["cells"]["reference_classes"])
    out: dict[str, Any] = {"contrast": f"{agg} - ({' + '.join(ref)})", "features": {}, "wrong_way": []}
    for f in fcfg["features"]:
        fid = f["id"]
        sub = df[df[fid].notna()]
        means = {}
        for cls in cfg["classes"]:
            c = sub[sub["label"] == cls]
            means[cls] = _boot(np.mean, [c[fid].to_numpy(float)], c["group"].to_numpy(), cfg) if len(c) else None
        two = sub[sub["label"].isin([agg] + ref)]
        is_agg = (two["label"] == agg).to_numpy()
        diff = None
        if is_agg.any() and (~is_agg).any():
            diff = _boot(lambda v, a: float(v[a].mean() - v[~a].mean()) if a.any() and (~a).any() else math.nan,
                         [two[fid].to_numpy(float), is_agg], two["group"].to_numpy(), cfg)
        if f["direction"] == "context":
            verdict = "context"
        elif diff is None or diff["lo"] is None or diff["hi"] is None:
            verdict = "insufficient_data"
        else:
            sign = 1 if f["direction"] == "up" else -1
            lo, hi = sorted((sign * diff["lo"], sign * diff["hi"]))
            verdict = "as_expected" if lo > 0 else "wrong_way" if hi < 0 else "inconclusive"
        if verdict == "wrong_way":
            out["wrong_way"].append(fid)
        out["features"][fid] = {"display_name": f["display_name"], "unit": f["unit"], "expected": f["direction"],
                                "n_cases": int(len(sub)), "class_means": means, "difference": diff, "verdict": verdict}
    return out


# ---- 3. convergent validity ------------------------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(spearmanr(x, y).statistic)


def convergent_validity(tile_feats: pd.DataFrame, case_feats: pd.DataFrame, sel: pd.DataFrame, npz_path: Path,
                        cases: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not npz_path.is_file():
        return pending(f"{npz_path.name} not found (Agent 2's 04_concepts)")
    z = io.load_npz(npz_path)
    ids = [str(c) for c in z["concept_ids"]]
    ztab = pd.DataFrame(np.asarray(z["z"], dtype=float), columns=ids)
    ztab["case_id"] = [str(c) for c in z["case_id"]]
    ztab["tile_idx"] = np.asarray(z["tile_idx"]).astype(int)
    groups = cases.set_index("case_id")["group"]
    test_ids = set(cases.loc[cases["split"] == "test", "case_id"])
    out: dict[str, Any] = {"status": "ok", "encoder": str(z.get("encoder", "")), "pairs": {}}
    tiles = tile_feats[tile_feats["case_id"].isin(test_ids)].merge(ztab, on=["case_id", "tile_idx"], how="inner")
    sel_z = sel[sel["case_id"].isin(test_ids)].merge(ztab, on=["case_id", "tile_idx"], how="inner")
    case_z = sel_z.groupby("case_id")[[c for c in ids]].mean().reset_index()
    cases_m = case_feats[case_feats["case_id"].isin(test_ids)].merge(case_z, on="case_id", how="inner")
    for feature, concept in VALIDITY_PAIRS:
        if concept not in ids:
            out["pairs"][f"{feature}~{concept}"] = pending(f"concept {concept!r} not in concepts_tile.npz")
            continue
        res = {}
        for level, frame in (("tile", tiles), ("case", cases_m)):
            ok = frame[[feature, concept]].notna().all(axis=1) if len(frame) else pd.Series(dtype=bool)
            f = frame[ok]
            res[level] = (_boot(_spearman, [f[feature].to_numpy(float), f[concept].to_numpy(float)],
                                f["case_id"].map(groups).to_numpy(), cfg) if len(f) >= 3
                          else {"estimate": None, "n": int(len(f)), "note": "fewer than 3 paired values"})
        out["pairs"][f"{feature}~{concept}"] = res
    return out


# ---- 4/5. Agent 3's files -------------------------------------------------------------------------------------------
def _find_rows(obj: Any, names: Sequence[str]) -> list[dict[str, Any]]:
    found = []
    if isinstance(obj, Mapping):
        if any(obj.get(k) in names for k in CONCEPT_MODELS_KEYS):
            found.append(dict(obj))
        for k, v in obj.items():
            if k in names and isinstance(v, Mapping):
                found.append({"model": k, **v})
            else:
                found += _find_rows(v, names)
    elif isinstance(obj, list):
        for v in obj:
            found += _find_rows(v, names)
    return found


def ablation(art: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = schemas.eval_json(art, "model_choice")
    if not path.is_file():
        return pending("eval/model_choice.json not found (Agent 3's 05_risk_model)")
    names = list(cfg["model_choice"]["concept_models"])
    rows = _find_rows(io.read_json(path), names)
    if not rows:
        return {"status": "pending", "reason": f"no rows for {names} in eval/model_choice.json"}
    return {"status": "ok", "source": "eval/model_choice.json", "models": names, "rows": rows}


def faithfulness(art: Path) -> dict[str, Any]:
    path = schemas.eval_json(art, "deletion")
    if not path.is_file():
        return pending("eval/deletion.json not found (Agent 3's deletion test)")
    data = io.read_json(path)
    hits = {k: v for k, v in data.items() if "cell" in str(k).lower()} if isinstance(data, Mapping) else {}
    if not hits:
        return {"status": "pending", "reason": "eval/deletion.json has no cell-feature section"}
    return {"status": "ok", "source": "eval/deletion.json", **hits}


# ---- plot -----------------------------------------------------------------------------------------------------------
def plot_direction(checks: Mapping[str, Any], cfg: Mapping[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = [k for k, v in checks["features"].items() if any(m for m in v["class_means"].values())]
    if not feats:
        return False
    cols = 3
    rows = math.ceil(len(feats) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    classes = list(cfg["classes"])
    for ax, fid in zip(axes.ravel(), feats):
        info = checks["features"][fid]
        for i, cls in enumerate(classes):
            m = info["class_means"].get(cls)
            if not m or m["estimate"] is None:
                continue
            lo = m["lo"] if m["lo"] is not None else m["estimate"]
            hi = m["hi"] if m["hi"] is not None else m["estimate"]
            ax.errorbar([i], [m["estimate"]], yerr=[[m["estimate"] - lo], [hi - m["estimate"]]], fmt="o", ms=6,
                        color=CLASS_COLORS[i % len(CLASS_COLORS)], ecolor=CLASS_COLORS[i % len(CLASS_COLORS)],
                        elinewidth=2, capsize=0)
        ax.set_xticks(range(len(classes)), ["\n".join(cfg["ui"]["class_labels"][c].rsplit(" ", 1)) for c in classes],
                      fontsize=7)
        unit = f" ({info['unit']})" if info["unit"] else ""
        ax.set_title(f"{info['display_name']}{unit}\n{info['verdict'].replace('_', ' ')}", fontsize=8, color="#333333")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7, colors="#555555")
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("Measured cell features by class (test split, mean with 95% cluster-bootstrap CI)\n"
                 "Research prototype — measured features associated with invasive carcinoma, not causes",
                 fontsize=9, color="#222222")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.atomic_path(path) as tmp:
        fig.savefig(tmp, format="png", dpi=150)
    plt.close(fig)
    return True


# ---- main -----------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"eval.{STEP}", "Cell-feature validation (CELLS_SPEC §8).")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root, art = ctx.cfg, ctx.data_root, ctx.artifacts_root
    env = {"name": "cells", **{k: v for k, v in io.provenance(cfg, synthetic=ctx.synthetic).items()
                               if k in schemas.EVAL_ENVELOPE}}
    out_path = schemas.eval_json(art, "cells")
    if not cfg["cells"]["enabled"]:
        io.write_json(out_path, {**env, "status": "disabled"})
        print(json.dumps({"status": "disabled"}))
        return 0
    try:
        fcfg = cc.load_feature_config(cfg)
        cases = io.read_csv(schemas.cases_csv(root), dtype=str, keep_default_na=False)
        feats_path, tile_path = schemas.cell_features_csv(root), schemas.cell_features_csv(root, per_tile=True)
        if not feats_path.is_file() or not schemas.cell_tiles_csv(root).is_file():
            io.write_json(out_path, {**env, "status": "pending", "reason": "cell_features.csv / cell_tiles.csv not found"})
            print(json.dumps({"status": "pending"}))
            return 0
        case_feats = io.read_csv(feats_path, dtype={"case_id": str})
        tile_feats = io.read_csv(tile_path, dtype={"case_id": str})
        sel = io.read_csv(schemas.cell_tiles_csv(root), dtype={"case_id": str})
        keep = cli.limit_case_ids(case_feats["case_id"], ctx.limit)
        case_feats = case_feats[case_feats["case_id"].isin(keep)]
        df = case_feats.merge(cases[["case_id", "label", "group"]], on="case_id", how="left")
        test = df[df["split"] == "test"]
        checks = direction_checks(test, fcfg, cfg) if len(test) else pending("no test cases with cell features")
        result = {
            **env, "status": "ok", "split": "test", "n_test_cases": int(len(test)),
            "n_test_groups": int(test["group"].nunique()) if len(test) else 0,
            "bootstrap": {"n": cfg["bootstrap"]["n"], "ci": cfg["bootstrap"]["ci"], "cluster": cfg["bootstrap"]["cluster"]},
            "review_sheet": review_sheet(sel[sel["case_id"].isin(keep)], cases, root, art, cfg),
            "direction_checks": checks,
            "convergent_validity": convergent_validity(tile_feats, case_feats, sel, schemas.concepts_tile_npz(root),
                                                       cases, cfg),
            "ablation": ablation(art, cfg),
            "faithfulness": faithfulness(art),
            "insufficient_tumor_nuclei_test_cases": int(test["insufficient_tumor_nuclei"].sum()) if len(test) else 0,
            "wording": "measured features associated with invasive carcinoma; not causes; not a Nottingham grade",
        }
        if checks.get("features") and plot_direction(checks, cfg, schemas.eval_asset(art, "cells_direction.png")):
            result["figures"] = ["eval/cells_direction.png"]
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2
    io.write_json(out_path, result)
    print(json.dumps({k: (v.get("status") if isinstance(v, Mapping) else v) for k, v in result.items()
                      if k not in schemas.EVAL_ENVELOPE}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
