"""Deck-ready PNG figures for every eval JSON present in <artifacts>/eval/ (BUILD_PLAN Task 11).

    python -m eval.figures

One figure per available eval file (missing files are skipped and listed):
    calibration.json -> calibration.png    reliability diagram before / after temperature
    conformal.json   -> conformal.png      coverage (with target) and mean set size per method (two panels)
    metrics.json     -> ab_roc.png         Invasive-vs-rest ROC, Model A vs Model B, AUROC CIs in the legend
    thunder.json     -> thunder.png        published THUNDER macro-F1 vs ours (same split, different protocol)
    queue_sim.json   -> queue_sim.png      aggressive-case time to first look and non-flagged wait per policy
    margin.json      -> margin.png         margin-read checks (benign FPR, composite clear rate, echo, WSI)
    concepts.json, concepts_bach.json, reliability.json, cells.json, deletion.json, retrieval.json, zeroshot.json
                     -> <name>.png         generic dot + CI chart of every {estimate, lo, hi} block found (these
                                           files belong to Agents 1-3; the generic view needs no shape agreement)

Style (dataviz skill, reference palette): validated categorical slots blue / orange / aqua (all-pairs pass; aqua is
below 3:1 on the surface, so every series also gets a marker shape and a direct label or legend entry), one y-axis per
panel, recessive grid, text in ink tokens. Every figure carries the footer "Research prototype — not for clinical
use", and a SYNTHETIC DATA banner when the eval file is synthetic.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core import cli, io, schemas  # noqa: E402

STEP = "eval_figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # reference palette slots 1-3 (validated all-pairs, light)
MARKERS = ("o", "s", "^")
GENERIC = ("concepts", "concepts_bach", "reliability", "cells", "deletion", "retrieval", "zeroshot", "model_choice")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(n: int = 1, width: float = 6.4, height: float = 4.2):
    fig, axes = plt.subplots(1, n, figsize=(width * (n if n > 1 else 1) * (0.8 if n > 1 else 1), height),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes[0]:
        _style(ax)
    return fig, list(axes[0])


def _finish(fig, title: str, synthetic: bool, path: Path, dpi: int) -> Path:
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left", y=0.99)
    footer = "Research prototype — not for clinical use"
    fig.text(0.01, 0.01, footer, color=INK2, fontsize=8, ha="left", va="bottom")
    if synthetic:
        fig.text(0.99, 0.01, "SYNTHETIC DATA — not a result", color=INK, fontsize=9, ha="right", va="bottom",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.atomic_path(path) as tmp:
        fig.savefig(tmp, format="png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def _ci_text(block: Mapping[str, Any] | None, pct: bool = False, decimals: int = 2, pct_decimals: int = 0) -> str:
    if not block or block.get("estimate") is None:
        return "n/a"
    f = (lambda v: f"{100 * v:.{pct_decimals}f}%") if pct else (lambda v: f"{v:.{decimals}f}")
    lo, hi = block.get("lo"), block.get("hi")
    rng = f" [{f(lo)}, {f(hi)}]" if lo is not None and hi is not None else ""
    return f"{f(block['estimate'])}{rng}"


def _dots(ax, labels: Sequence[str], blocks: Sequence[Mapping[str, Any] | None], color: str | Sequence[str],
          marker: str | Sequence[str] = "o", pct: bool = False, decimals: int = 2, pct_decimals: int = 0) -> None:
    n = len(labels)
    colors = [color] * n if isinstance(color, str) else list(color)
    markers = [marker] * n if isinstance(marker, str) else list(marker)
    ys = list(range(len(labels)))[::-1]
    for y, block, color, marker in zip(ys, blocks, colors, markers):
        if not block or block.get("estimate") is None:
            ax.text(0.0, y, "  n/a", va="center", color=MUTED, fontsize=8)
            continue
        est, lo, hi = block["estimate"], block.get("lo"), block.get("hi")
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([est], [y], marker=marker, color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(_ci_text(block, pct, decimals, pct_decimals), (est, y), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK, fontsize=9)


# ---- specific figures ---------------------------------------------------------------------------------------------
def fig_calibration(data: Mapping[str, Any], path: Path, dpi: int) -> Path:
    fig, (ax,) = _figure()
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")
    for i, side in enumerate(("before", "after")):
        if side not in data:
            continue
        bins = [b for b in data[side]["reliability_diagram"]["bins"] if b.get("count")]
        xs = [b["confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        name = "before temperature" if side == "before" else f"after temperature (T = {data.get('temperature') or 0:.2f})"
        ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], linewidth=2, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=f"{name}: ECE {_ci_text(data[side]['ece'])}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted top-class confidence (equal-mass bins)", color=INK2)
    ax.set_ylabel("Observed accuracy", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.text(0.98, 0.04, f"test n = {data.get('n')}, {data.get('n_groups')} groups\nECE is biased at small n",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    return _finish(fig, "Calibration: reliability diagram before and after temperature scaling",
                   bool(data.get("synthetic")), path, dpi)


def fig_conformal(data: Mapping[str, Any], path: Path, dpi: int) -> Path:
    methods = list(data.get("methods") or {})
    names = {"lac_marginal": "LAC, marginal (drives tiers)", "aps": "APS", "mondrian": "Mondrian (per class)"}
    fig, (ax1, ax2) = _figure(2)
    labels = [names.get(m, m) for m in methods]
    _dots(ax1, labels, [data["methods"][m]["coverage"] for m in methods], SERIES[0], pct=True)
    ax1.axvline(data["target_coverage"], color=MUTED, linestyle="--", linewidth=1)
    ax1.text(data["target_coverage"], 1.01, f"target {100 * data['target_coverage']:.0f}%", color=INK2, fontsize=8,
             ha="right", transform=ax1.get_xaxis_transform())
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Empirical coverage on test", color=INK2)
    _dots(ax2, labels, [data["methods"][m]["mean_set_size"] for m in methods], SERIES[1], "s")
    ax2.set_xlabel("Mean prediction-set size (classes)", color=INK2)
    fs = data.get("finite_sample") or {}
    if fs.get("sd") is not None:
        fig.text(0.5, 0.06, f"Coverage given the {fs['n_calibration']} calibration cases ~ Beta({fs['beta'][0]}, "
                 f"{fs['beta'][1]}), SD {fs['sd']:.3f}", ha="center", fontsize=8, color=INK2)
    return _finish(fig, f"Conformal prediction sets at alpha = {data['alpha']} (test n = {data['n']})",
                   bool(data.get("synthetic")), path, dpi)


def fig_ab_roc(data: Mapping[str, Any], path: Path, dpi: int) -> Path | None:
    ab = data.get("a_vs_b")
    if not ab or not ab.get("roc"):
        return None
    fig, (ax,) = _figure(width=5.6, height=4.6)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    for i, (key, name) in enumerate((("a", f"Model A ({ab['a']})"), ("b", f"Model B ({ab['b']})"))):
        roc = ab["roc"][key]
        auc = ab["auroc_a"] if key == "a" else ab["auroc_b"]
        ax.plot(roc["fpr"], roc["tpr"], color=SERIES[i], linewidth=2, drawstyle="steps-post",
                label=f"{name}: AUROC {_ci_text(auc)}")
    diff = ab["difference_a_minus_b"]
    ax.text(0.98, 0.05, f"A − B: {_ci_text(diff)} (paired cluster bootstrap)", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("False-positive rate (not invasive called invasive)", color=INK2)
    ax.set_ylabel("Sensitivity for invasive carcinoma", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=INK)
    return _finish(fig, f"Invasive vs rest on the test split (n = {data['n']})", bool(data.get("synthetic")), path, dpi)


def fig_thunder(data: Mapping[str, Any], path: Path, dpi: int) -> Path:
    labels, blocks, colors, markers = [], [], [], []
    for kind, title in (("linear_probe", "linear probe"), ("zero_shot", "zero-shot")):
        for name, value in data["published"][kind].items():
            labels.append(f"THUNDER {title} · {name}")
            blocks.append({"estimate": value / 100.0})
            colors.append(SERIES[0])
            markers.append("o")
    for row in data["ours"]:
        role = "production, calibrated" if row["role"].startswith("production") else "uncalibrated"
        labels.append(f"ours · {row['model']} ({role})")
        mf = row["macro_f1"] or {}
        blocks.append({k: (None if mf.get(k) is None else mf[k] / 100.0) for k in ("estimate", "lo", "hi")})
        colors.append(SERIES[1])
        markers.append("s")
    fig, (ax,) = _figure(width=7.4, height=0.5 * len(labels) + 2.0)
    _dots(ax, labels, blocks, colors, markers, pct=True, pct_decimals=1)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro-F1 on the THUNDER BACH test split", color=INK2)
    ax.plot([], [], "o", color=SERIES[0], label="published (no CI)")
    ax.plot([], [], "s", color=SERIES[1], label="ours (95% cluster-bootstrap CI)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=INK)
    return _finish(fig, "Same split, different protocol: ours beside published THUNDER results",
                   bool(data.get("synthetic")), path, dpi)


def fig_queue(data: Mapping[str, Any], path: Path, dpi: int) -> Path:
    policies = list(data["results"])
    names = [data["policies"].get(p, p) for p in policies]
    fig, (ax1, ax2) = _figure(2)
    for ax, key, color, marker, title in (
            (ax1, "aggressive_mean", SERIES[0], "o", "Truly invasive cases: mean time to first look (min)"),
            (ax2, "non_flagged_mean", SERIES[1], "s", "Unflagged (tier 4) cases: mean wait (min)")):
        blocks = [{"estimate": data["results"][p][key]["mean"], "lo": data["results"][p][key]["lo"],
                   "hi": data["results"][p][key]["hi"]} for p in policies]
        _dots(ax, names, blocks, color, marker, decimals=1)
        ax.set_xlabel(title, color=INK2, fontsize=9)
        ax.set_xlim(left=0)
    a = data["assumptions"]
    fig.text(0.5, 0.06, f"{a['arrivals']}; {a['reader']}; utilization {a['utilization']}; {data['reps']} simulated days",
             ha="center", fontsize=7, color=INK2, wrap=True)
    return _finish(fig, "Queue simulation: benefit to invasive cases and cost to unflagged cases",
                   bool(data.get("synthetic")), path, dpi)


def fig_margin(data: Mapping[str, Any], path: Path, dpi: int) -> Path:
    rows: list[tuple[str, Mapping[str, Any] | None]] = [
        ("Benign images: false tumor-at-margin, per margin", data["benign_fpr"].get("pooled_per_margin")),
        ("Benign images: false call at any of 4 margins", data["benign_fpr"].get("any_margin")),
        ("Carcinoma images: >= 1 margin called clear", data["echo"].get("p_at_least_one_clear_margin")),
    ]
    if "composite" in data:
        c = data["composite"]
        rows += [("Composites: exact-negative margin called clear", c.get("exact_negative_margin_clear_rate")),
                 ("Composites: benign-half segments below threshold", c.get("benign_half_segment_clear_rate")),
                 ("Composites: tumor-side margin called involved", c.get("tumor_side_margin_sensitivity"))]
    if data.get("wsi_windows", {}).get("sensitivity"):
        rows += [("WSI windows: sensitivity", data["wsi_windows"]["sensitivity"]),
                 ("WSI windows: specificity", data["wsi_windows"]["specificity"])]
    fig, (ax,) = _figure(width=7.4, height=0.55 * len(rows) + 1.8)
    _dots(ax, [r[0] for r in rows], [r[1] for r in rows], SERIES[0], pct=True)
    ax.set_xlim(0, 1.05)
    if data.get("max_fpr"):
        ax.axvline(data["max_fpr"], color=MUTED, linestyle=":", linewidth=1)
        ax.text(data["max_fpr"], 1.01, f"per-margin FPR target {100 * data['max_fpr']:.0f}%", color=INK2, fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("Rate (95% cluster-bootstrap CI)", color=INK2)
    return _finish(fig, f"Simulated margin read: localizer or echo? (threshold {data['threshold']:.2f})",
                   bool(data.get("synthetic")), path, dpi)


# ---- generic dot + CI view ------------------------------------------------------------------------------------------
def ci_blocks(obj: Any, prefix: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    """Every {estimate, lo, hi} mapping inside obj, labelled by its key path."""
    out: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(obj, Mapping):
        if {"estimate", "lo", "hi"} <= set(obj) and isinstance(obj.get("estimate"), (int, float)):
            return [(prefix or "value", obj)]
        for k, v in obj.items():
            out += ci_blocks(v, f"{prefix} · {k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += ci_blocks(v, f"{prefix}[{i}]")
    return out


def fig_generic(name: str, data: Mapping[str, Any], path: Path, dpi: int, limit: int = 16) -> Path | None:
    blocks = ci_blocks({k: v for k, v in data.items() if k not in schemas.EVAL_ENVELOPE})[:limit]
    if not blocks:
        return None
    fig, (ax,) = _figure(width=7.6, height=0.42 * len(blocks) + 1.6)
    _dots(ax, [lab[-60:] for lab, _ in blocks], [b for _, b in blocks], SERIES[0])
    ax.set_xlabel("Estimate (95% CI as reported in the eval file)", color=INK2)
    return _finish(fig, f"{name}: reported estimates with CIs", bool(data.get("synthetic")), path, dpi)


SPECIFIC = {"calibration": ("calibration.png", fig_calibration), "conformal": ("conformal.png", fig_conformal),
            "metrics": ("ab_roc.png", fig_ab_roc), "thunder": ("thunder.png", fig_thunder),
            "queue_sim": ("queue_sim.png", fig_queue), "margin": ("margin.png", fig_margin)}


def run(ctx: cli.StepContext) -> dict[str, Any]:
    art = ctx.artifacts_root
    dpi = int(ctx.cfg["eval"]["figure_dpi"])
    written, skipped = {}, {}
    for name, (fname, fn) in SPECIFIC.items():
        path = schemas.eval_json(art, name)
        if not path.is_file():
            skipped[name] = "no eval file"
            continue
        out = fn(io.read_json(path), schemas.eval_asset(art, fname), dpi)
        if out is None:
            skipped[name] = "nothing to plot"
        else:
            written[name] = str(out)
    for name in GENERIC:
        path = schemas.eval_json(art, name)
        if not path.is_file():
            skipped[name] = "no eval file"
            continue
        out = fig_generic(name, io.read_json(path), schemas.eval_asset(art, f"{name}.png"), dpi)
        if out is None:
            skipped[name] = "no {estimate, lo, hi} blocks"
        else:
            written[name] = str(out)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg,
                  inputs=[str(art / "eval")], outputs=list(written.values()), synthetic=ctx.synthetic,
                  extra={"written": written, "skipped": skipped})
    return {"written": written, "skipped": skipped}


def main(argv: Sequence[str] | None = None) -> int:
    ctx = cli.context(cli.step_parser("eval.figures", "Deck-ready PNG figures for the eval JSONs.").parse_args(argv))
    try:
        result = run(ctx)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"eval.figures: {exc}", file=sys.stderr)
        return 1
    print(f"eval.figures: wrote {len(result['written'])} figures; skipped {sorted(result['skipped'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
