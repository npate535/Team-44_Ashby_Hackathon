"""Generate docs/RESULTS.md from <artifacts>/eval/*.json (BUILD_PLAN Task 11; written at real-run time).

    python -m eval.results_md [--out PATH]      # default docs/RESULTS.md (real) or <artifacts>/RESULTS_synthetic.md

Every number is printed with its 95% CI, n / n_groups and the eval file it came from; nothing is computed here
beyond formatting. Sections appear only for eval files that exist (missing ones are listed). The document opens with
a SYNTHETIC banner when any source file is synthetic, and always states the research-prototype framing, the
limitations, and which checks were AI-reviewed vs human-reviewed (eval/spotcheck.json `reviewer_kind`).
Eval files owned by Agents 1-3 (concepts, concepts_bach, zeroshot, model_choice, reliability, retrieval, deletion,
cells, finetune) are rendered generically: every {estimate, lo, hi} block they contain, labelled by its key path.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core import cli, config, io, schemas
from eval.figures import ci_blocks

STEP = "eval_results_md"
OWN = ("metrics", "calibration", "conformal", "thunder", "queue_sim", "margin")
FOREIGN = ("model_choice", "concepts", "concepts_bach", "zeroshot", "reliability", "retrieval", "deletion", "cells",
           "finetune", "spotcheck")


def fmt(block: Mapping[str, Any] | None, *, pct: bool = False, digits: int = 2) -> str:
    """'0.91 [0.85, 0.96] (n 132, 61 groups)'; 'n/a' when undefined."""
    if not block or block.get("estimate") is None:
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.{digits}f}")
    text = f(block["estimate"])
    if block.get("lo") is not None and block.get("hi") is not None:
        text += f" [{f(block['lo'])}, {f(block['hi'])}]"
    extra = [f"n {block['n']}" for _ in [0] if block.get("n") is not None]
    if block.get("n_groups") is not None:
        extra.append(f"{block['n_groups']} groups")
    return text + (f" ({', '.join(extra)})" if extra else "")


def sim_fmt(block: Mapping[str, Any] | None) -> str:
    if not block or block.get("mean") is None:
        return "n/a"
    return f"{block['mean']:.1f} [{block['lo']:.1f}, {block['hi']:.1f}]"


def section_metrics(m: Mapping[str, Any]) -> list[str]:
    p = m["production"]
    out = ["## Classification (test split)", "",
           f"Source: `eval/metrics.json` · test n = {m['n']}, {m['n_groups']} patient / pseudo-patient groups · "
           f"{m['variants_excluded']} simulated variants excluded · production model `{m['models']['production']}`.", "",
           "| Metric | Estimate [95% CI] |", "|---|---|"]
    for key, label in (("invasive_vs_rest_auroc", "Invasive vs rest AUROC"), ("macro_auroc", "Macro AUROC"),
                       ("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy"),
                       ("invasive_sensitivity", "Invasive sensitivity"), ("invasive_specificity", "Invasive specificity")):
        out.append(f"| {label} | {fmt(p.get(key))} |")
    ab = m.get("a_vs_b")
    if ab:
        out += ["", f"Model A `{ab['a']}` vs Model B `{ab['b']}` ({ab['metric']}): A {fmt(ab['auroc_a'])}; "
                    f"B {fmt(ab['auroc_b'])}; paired difference A − B {fmt(ab['difference_a_minus_b'])}."]
    ii = (p.get("confusion") or {}).get("insitu_invasive")
    if ii:
        out += ["", f"In situ ↔ invasive confusion: {ii['insitu_called_invasive']} of {ii['n_insitu']} in situ cases "
                    f"called invasive; {ii['invasive_called_insitu']} of {ii['n_invasive']} invasive cases called in situ."]
    return out + [""]


def section_calibration(c: Mapping[str, Any]) -> list[str]:
    temp = c.get("temperature")
    temp = "n/a" if temp is None else f"{float(temp):.3f}"
    out = ["## Calibration", "", f"Source: `eval/calibration.json` · temperature {temp} fit on "
           f"{c.get('temperature_fit_on')} · {c.get('ece_bins')} equal-mass ECE bins (biased at small n).", "",
           "| | ECE | Brier | NLL |", "|---|---|---|---|"]
    for side in ("before", "after"):
        if side in c:
            out.append(f"| {side} temperature | {fmt(c[side]['ece'])} | {fmt(c[side]['brier'])} | {fmt(c[side]['nll'])} |")
    return out + [""]


def section_conformal(c: Mapping[str, Any]) -> list[str]:
    out = ["## Conformal prediction sets", "", f"Source: `eval/conformal.json` · alpha {c['alpha']} (target coverage "
           f"{100 * c['target_coverage']:.0f}%) · tiers use {c['drives_tiers']}.", "",
           "| Method | Coverage | Mean set size | Empty sets |", "|---|---|---|---|"]
    for name, b in c["methods"].items():
        out.append(f"| {name} | {fmt(b['coverage'], pct=True)} | {fmt(b['mean_set_size'])} | "
                   f"{100 * b['empty_set_rate']:.1f}% |")
    lac = c["methods"].get("lac_marginal")
    if lac:
        out += ["", "Per-class LAC coverage: " + "; ".join(f"{k} {fmt(v, pct=True)}" for k, v in lac["per_class"].items()) + "."]
    fs = c.get("finite_sample") or {}
    if fs.get("note"):
        out += ["", fs["note"]]
    return out + [""]


def section_thunder(t: Mapping[str, Any]) -> list[str]:
    out = ["## Benchmark anchor (THUNDER split)", "", f"Source: `eval/thunder.json` · **{t['label']}** · {t['source']}.",
           "", "| Model | Macro-F1 (%) | Balanced accuracy (%) |", "|---|---|---|"]
    for kind, title in (("linear_probe", "published linear probe"), ("zero_shot", "published zero-shot")):
        for name, v in t["published"][kind].items():
            out.append(f"| {title} · {name} | {v:.1f} (no CI) | — |")
    for row in t["ours"]:
        mf, ba = row["macro_f1"] or {}, row["balanced_accuracy"] or {}
        f = lambda b: "n/a" if b.get("estimate") is None else (                                         # noqa: E731
            f"{b['estimate']:.1f}" + (f" [{b['lo']:.1f}, {b['hi']:.1f}]" if b.get("lo") is not None else ""))
        out.append(f"| ours · {row['model']} ({row['role']}) | {f(mf)} | {f(ba)} |")
    return out + [""] + [f"- {n}" for n in t["notes"]] + [""]


def section_queue(q: Mapping[str, Any]) -> list[str]:
    out = ["## Queue simulation (workflow benefit and cost)", "",
           f"Source: `eval/queue_sim.json` · {q['reps']} simulated days · minutes, mean [2.5–97.5 percentile over days].",
           "", "| Policy | Invasive: mean first look | Invasive: P90 | Unflagged: mean wait | Overtime |",
           "|---|---|---|---|---|"]
    for p, r in q["results"].items():
        out.append(f"| {q['policies'].get(p, p)} | {sim_fmt(r['aggressive_mean'])} | {sim_fmt(r['aggressive_p90'])} | "
                   f"{sim_fmt(r['non_flagged_mean'])} | {sim_fmt(r['overtime'])} |")
    for p, r in q["results"].items():
        if "vs_fifo" in r:
            v = r["vs_fifo"]
            out.append(f"\n{q['policies'].get(p, p)} vs FIFO: invasive first look {sim_fmt(v['aggressive_mean'])} min; "
                       f"**delay imposed on unflagged cases {sim_fmt(v['non_flagged_mean'])} min**; false negatives "
                       f"{sim_fmt(v['false_negative_mean'])} min.")
    out += ["", "Assumptions: " + "; ".join(f"{k}: {v}" for k, v in q["assumptions"].items() if k != "aging") + "."]
    return out + [""] + [f"- {n}" for n in q["notes"]] + [""]


def section_margin(m: Mapping[str, Any]) -> list[str]:
    out = ["## Simulated margin read", "", f"Source: `eval/margin.json` · threshold {m['threshold']:.3f} "
           f"(CI {m.get('threshold_ci')}, from {m.get('n_benign_cal_images')} benign calibration images).", "",
           f"- Benign test images, false tumor-at-margin per margin (pooled over 4 margins): "
           f"{fmt(m['benign_fpr'].get('pooled_per_margin'), pct=True)} (the threshold targets ≤ "
           f"{100 * m['benign_fpr']['target_max_fpr']:.0f}% per margin; {m['benign_fpr']['n_scored']} of "
           f"{m['benign_fpr']['n_benign_test']} benign test images scored). Image-level, any of 4 margins: "
           f"{fmt(m['benign_fpr'].get('any_margin'), pct=True)} (no 5% target at image level).",
           f"- Carcinoma test images with ≥ 1 margin called clear: {fmt(m['echo'].get('p_at_least_one_clear_margin'), pct=True)}; "
           f"Spearman(max margin score, case risk) {fmt(m['echo'].get('spearman_max_margin_vs_risk'))}."]
    if "composite" in m:
        c = m["composite"]
        out += [f"- Composite fields ({c['n_composites']} of {c['n_expected']} scored): exact-negative margin called clear "
                f"{fmt(c['exact_negative_margin_clear_rate'], pct=True)}; benign-half segments below threshold "
                f"{fmt(c['benign_half_segment_clear_rate'], pct=True)}; carcinoma-half segments above threshold "
                f"{fmt(c['carcinoma_half_segment_hit_rate'], pct=True)}; tumor-side margin involved "
                f"{fmt(c['tumor_side_margin_sensitivity'], pct=True)}. An echo of the diagnosis would score ~0% clear."]
    w = m.get("wsi_windows")
    if w and w.get("sensitivity"):
        out += [f"- Annotated WSI windows ({w['n_windows']}, {w['n_patients']} patients): sensitivity "
                f"{fmt(w['sensitivity'], pct=True)}, specificity {fmt(w['specificity'], pct=True)}."]
    return out + [""] + [f"- {n}" for n in m["notes"]] + [""]


def section_foreign(name: str, data: Mapping[str, Any]) -> list[str]:
    blocks = ci_blocks({k: v for k, v in data.items() if k not in schemas.EVAL_ENVELOPE})
    out = [f"### `eval/{name}.json`", ""]
    if not blocks:
        return out + ["(no estimate/CI blocks; see the file)", ""]
    return out + [f"- {label}: {fmt(b)}" for label, b in blocks[:40]] + [""]


def review_lines(art: Path) -> list[str]:
    path = schemas.eval_json(art, "spotcheck")
    lines = ["## Reviewed checks", ""]
    if path.is_file():
        sc = io.read_json(path)
        kind = sc.get("reviewer_kind", "unknown")
        who = "AI-reviewed (Claude), not a pathologist" if kind == "ai" else (
            "human-reviewed" if kind == "human" else f"reviewer kind {kind!r}")
        summary = sc.get("summary") or {}
        lines.append(f"- Concept top-tile and cell-overlay spot check: **{who}** · `eval/spotcheck.json` · summary {summary}.")
    else:
        lines.append("- Concept top-tile / cell-overlay spot check: not recorded yet (`eval/spotcheck.json` missing).")
    spike = schemas.eval_asset(art, "cells_spike.md")
    lines.append(f"- Cell segmentation spike overlays: {'see `eval/cells_spike.md` (AI review unless stated)' if spike.is_file() else 'no record'}.")
    return lines + ["- No result here is clinically validated; every review above is labelled by who did it.", ""]


LIMITATIONS = [
    "Research prototype on a small public dataset (BACH, CC BY-NC-ND 4.0); not a clinical device and not for clinical use.",
    "BACH labels are diagnostic categories, not Nottingham grades; 'likely aggressive' means 'likely invasive carcinoma'.",
    "Small test set (THUNDER split, 132 images) with partial patient IDs: pseudo-patient groups, wide intervals.",
    "The surgical lane is simulated: image borders stand in for inked margins and frozen-section artifacts are "
    "simulated degradations; composite fields are synthetic constructions from test images.",
    "Concept scores are features associated with the classes (text-prompt and caption-consistent), not causes and not "
    "validated grade components; caption consistency measures caption co-mention, not tissue truth.",
    "Stain rules are illustrative and require clinical review; the agent only drafts orders.",
    "The queue simulation is a model with stated assumptions, not a measured workflow outcome.",
]


def build(art: Path, cfg: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    present = {n: io.read_json(schemas.eval_json(art, n)) for n in OWN + FOREIGN if schemas.eval_json(art, n).is_file()}
    synthetic = any(bool(d.get("synthetic")) for d in present.values())
    lines = ["# PathQueue results", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA — these numbers come from generated test data and are NOT results.**", ""]
    lines += ["> Research prototype — not for clinical use. Every number below is quoted from the eval file named "
              "beside it, with its 95% cluster-bootstrap CI (patient / pseudo-patient groups) unless stated.", ""]
    sha = sorted({str(d.get("git_sha")) for d in present.values()})
    lines += [f"Generated {io.utc_now()} from {len(present)} eval files (git {', '.join(sha) or 'n/a'}).", ""]
    renderers = {"metrics": section_metrics, "calibration": section_calibration, "conformal": section_conformal,
                 "thunder": section_thunder, "queue_sim": section_queue, "margin": section_margin}
    for name in OWN:
        if name in present:
            lines += renderers[name](present[name])
    foreign = [n for n in FOREIGN if n in present and n != "spotcheck"]
    if foreign:
        lines += ["## Other evaluations (concepts, reliability, retrieval, deletion, cells)", ""]
        for name in foreign:
            lines += section_foreign(name, present[name])
    lines += review_lines(art)
    lines += ["## Limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    missing = [n for n in OWN + FOREIGN if n not in present]
    if missing:
        lines += [f"_Eval files not present at generation time: {', '.join(missing)}._", ""]
    return "\n".join(lines), {"present": sorted(present), "missing": missing, "synthetic": synthetic}


def run(ctx: cli.StepContext, out: Path | None = None) -> Path:
    """Default output: docs/RESULTS.md for real results; <artifacts>/RESULTS_synthetic.md for synthetic ones, so a
    synthetic run never overwrites the real document."""
    text, info = build(ctx.artifacts_root, ctx.cfg)
    if out is None:
        out = (ctx.artifacts_root / "RESULTS_synthetic.md") if (info["synthetic"] or ctx.synthetic) \
            else config.resolve_repo_path("docs/RESULTS.md")
    io.atomic_write_text(out, text)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg,
                  inputs=[str(ctx.artifacts_root / "eval")], outputs=[out], synthetic=info["synthetic"], extra=info)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.results_md", "Generate docs/RESULTS.md from the eval JSONs.")
    parser.add_argument("--out", default=None, help="output path (default docs/RESULTS.md)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    try:
        path = run(ctx, Path(args.out) if args.out else None)
    except (KeyError, ValueError) as exc:
        print(f"eval.results_md: {exc}", file=sys.stderr)
        return 1
    print(f"eval.results_md: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
