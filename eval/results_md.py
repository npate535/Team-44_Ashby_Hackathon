"""Generate docs/RESULTS.md from <artifacts>/eval/*.json (BUILD_PLAN Task 11; written at real-run time).

    python -m eval.results_md [--out PATH]      # default docs/RESULTS.md (real) or <artifacts>/RESULTS_synthetic.md

Every number is printed with its 95% CI, n / n_groups and the eval file it came from; nothing is computed here
beyond formatting. Sections appear only for eval files that exist (missing ones are listed). The document opens with
a SYNTHETIC banner when any source file is synthetic, and always states the research-prototype framing, the
limitations, and which checks were AI-reviewed vs human-reviewed (eval/spotcheck.json `reviewer_kind`).
Eval files owned by Agents 1-3 (concepts, concepts_bach, zeroshot, model_choice, reliability, retrieval, cells,
finetune) are rendered generically: every {estimate, lo, hi} block they contain, labelled by its key path. The
deletion test has its own table (four removal arms in probability and logit units), with the exact-contribution
identity check from tile_contrib.json beside it.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core import cli, config, io, schemas
from eval.figures import ci_blocks

STEP = "eval_results_md"
OWN = ("qc", "metrics", "calibration", "conformal", "conformal_resplit", "thunder", "queue_sim", "deferral",
       "margin", "external_tcga")
FOREIGN = ("model_choice", "concepts", "concepts_bach", "zeroshot", "reliability", "retrieval", "deletion", "cells",
           "finetune", "spotcheck", "tile_contrib")


def _num(value: float, digits: int, suffix: str = "") -> str:
    """Fixed-point text without a signed zero ('-0.00' -> '0.00')."""
    text = f"{value:.{digits}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text + suffix


def fmt(block: Mapping[str, Any] | None, *, pct: bool = False, digits: int | None = None) -> str:
    """'0.91 [0.85, 0.96] (n 132, 61 groups)'; 'n/a' when undefined. digits=None: 2 decimals, or 3 when a value is on
    the probability scale near 1 (an AUROC of 0.998 is not '1.00')."""
    if not block or block.get("estimate") is None:
        return "n/a"
    if digits is None:
        values = [block.get(k) for k in ("estimate", "lo", "hi") if isinstance(block.get(k), (int, float))]
        digits = 3 if any(0.95 <= v <= 1.0 for v in values) else 2
    f = (lambda v: _num(100 * v, 1, "%")) if pct else (lambda v: _num(v, digits))
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
    return f"{_num(block['mean'], 1)} [{_num(block['lo'], 1)}, {_num(block['hi'], 1)}]"


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
    # conformal.json keys plain marginal APS as "aps" (set_tier is only added when it is a distinct set)
    tier = c["methods"].get(c["drives_tiers"]) or (c["methods"].get("aps") if c["drives_tiers"] == "aps_marginal" else None)
    if tier and c["drives_tiers"] != "lac_marginal":
        out += ["", f"Per-class coverage of the tier set ({c['drives_tiers']}): "
                + "; ".join(f"{k} {fmt(v, pct=True)}" for k, v in tier["per_class"].items()) + "."]
    lac = c["methods"].get("lac_marginal")
    if lac:
        out += ["", "Per-class LAC coverage (pre-registered): "
                + "; ".join(f"{k} {fmt(v, pct=True)}" for k, v in lac["per_class"].items()) + "."]
    fs = c.get("finite_sample") or {}
    if c["drives_tiers"] != "lac_marginal" and lac:
        # every number and qualifier comes from conformal.json, so a re-run cannot leave a stale claim here
        hist = {int(k): int(v) for k, v in (lac.get("set_size_histogram") or {}).items()}
        n_multi, n_all = sum(v for k, v in hist.items() if k >= 2), sum(hist.values())
        multi = ("no multi-class sets, so it gave the reader no differential" if hist and not n_multi else
                 f"multi-class sets for {100 * n_multi / n_all:.1f}% of cases" if n_all else "")
        n_cal = fs.get("n_calibration")
        name = str(c["drives_tiers"])
        method = ("APS, Romano et al. 2020" if name.startswith("aps") else name) + (
            ", with the argmax class added to empty sets" if name.endswith("_nonempty") else "")
        out += ["", "**Post-hoc change, disclosed:** marginal LAC was the pre-registered tier set. On the held-out test "
                f"split it returned empty sets for {100 * lac['empty_set_rate']:.1f}% of cases"
                + (f" and {multi}" if multi else "") + "; the "
                + (f"{n_cal}-image " if n_cal is not None else "") + "calibration split is far easier than the test "
                "split, which likely violates the exchangeability the guarantee needs. Tiers now use the set named above "
                f"({method}). The switch was made after viewing test results, so its observed coverage is optimistic "
                "and is not an independent confirmation."]
    if fs.get("note"):
        out += ["", fs["note"]]
    return out + [""]


def section_conformal_resplit(r: Mapping[str, Any]) -> list[str]:
    """Coverage distribution across the pre-registered re-splits (docs/VALIDATION_ROUND3.md §1); every number from
    eval/conformal_resplit.json."""
    def pct(v: Any, digits: int = 1) -> str:
        return "n/a" if v is None else _num(100 * float(v), digits, "%")

    n = f"{int(r['n_splits']):,}"
    sizes = (r.get("partition_sizes") or {}).get("n_cal") or {}
    out = [f"## Conformal coverage across {n} random re-splits", "",
           f"Source: `eval/conformal_resplit.json` · pre-registered in `docs/VALIDATION_ROUND3.md` §1 · model "
           f"{r['model']} · {r['n_images']} held-out images (calibration + test), {r['n_groups']} patient groups kept "
           f"whole · calibration part at least {r['n_cal_target']} images"
           + (f" ({sizes['min']}–{sizes['max']}, median {sizes['p50']:g})" if sizes else "")
           + f" · alpha {r['alpha']} · seed {r['seed']}.", "",
           "Coverage percentiles describe variation across calibration/test partitions of the same images, not a "
           "confidence interval for a new cohort. The original split is the THUNDER partition the app uses.", "",
           "| Method | Coverage mean | Median | 2.5th–97.5th pct | Re-splits ≥ target | Mean set size | Empty sets "
           "| Invasive in set (invasive) | Invasive in set (other) | Original split coverage |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in r["methods"].items():
        q, o = b["coverage"], (r.get("original_split") or {}).get(name) or {}
        size = "n/a" if b.get("mean_set_size") is None else _num(b["mean_set_size"], 2)
        out.append(f"| {name} | {pct(q['mean'])} | {pct(q['p50'])} | {pct(q['p2_5'])}–{pct(q['p97_5'])} | "
                   f"{pct(b['share_at_or_above_target'])} | {size} | {pct(b.get('empty_share'))} | "
                   f"{pct(b.get('invasive_in_set_given_invasive'))} | {pct(b.get('invasive_in_set_given_not_invasive'))} | "
                   f"{pct(o.get('coverage'))} |")
    mond = r["methods"].get("mondrian") or {}
    if mond.get("unbounded_class_share") is not None:
        out += ["", f"Mondrian had at least one class with too few calibration images for a finite threshold in "
                    f"{pct(mond['unbounded_class_share'])} of re-splits; that class was then in every set."]
    check = (r.get("original_split_check") or {}).get("matches_conformal_json")
    if check is not None:
        out += ["", "Original-split coverage " + ("equals" if check else "does NOT equal")
                + " `eval/conformal.json` for every method."]
    tier = r.get("drives_tiers")
    if tier in r["methods"] and tier != "lac_marginal":
        out += ["", f"{tier} drives the worklist tiers and was chosen after viewing test coverage on the original "
                    "split; this check was pre-registered afterwards and applies the same rules to every method."]
    return out + [""]


def section_qc(q: Mapping[str, Any]) -> list[str]:
    it, rate = q["insufficient_tissue_cases"], q["qc_pass_rate"]
    v = q["validation_old_rule_display_vs_fullres"]
    by_old = ", ".join(f"{k} {n}" for k, n in it["old_by_class"].items())
    return ["## Tissue quality control (fixed in this run)", "",
            "Source: `eval/qc.json` (pipeline/01q_requalify_qc). The original tissue mask required HSV value < "
            f"{q['rule_old']['brightness_max']}; HSV value is the brightest RGB channel, so pale H&E tissue was "
            f"rejected. With saturation > {q['rule_new']['saturation_min']} alone, contract-tile QC pass rate went "
            f"{100 * rate['old'].get('g512', 0):.1f}% -> {100 * rate['new'].get('g512', 0):.1f}% and insufficient-tissue "
            f"images {it['old']} -> {it['new']} (before: {by_old}). Tile fractions were recomputed on display JPEGs; "
            f"the old rule recomputed the same way reproduces the full-resolution Delta values (r = {v['pearson_r']:.3f}, "
            f"mean |diff| = {v['mean_abs_diff']:.3f}). Every tile had been embedded on Delta, so no new GPU work was "
            "needed; steps 04-08 and the evaluation were re-run.", ""]


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
            fn = v.get("false_negative_mean") or {}
            fn_text = (f"false negatives {sim_fmt(fn)} min" if fn.get("mean") is not None else
                       "false negatives: not available (no simulated day had a truly invasive case outside tiers 1-2)")
            out.append(f"\n{q['policies'].get(p, p)} vs FIFO: invasive first look {sim_fmt(v['aggressive_mean'])} min; "
                       f"**delay imposed on unflagged cases {sim_fmt(v['non_flagged_mean'])} min**; {fn_text}.")
    promos = q.get("aging_promotions") or {}
    if promos:
        out += ["", "Aging guard: share of reads promoted per day — " + "; ".join(
            f"{q['policies'].get(k, k)} {100 * v:.1f}%" for k, v in promos.items()) + "."]
    out += ["", "Assumptions: " + "; ".join(f"{k}: {v}" for k, v in q["assumptions"].items() if k != "aging") + "."]
    return out + [""] + [f"- {n}" for n in q["notes"]] + [""]


def section_deferral(d: Mapping[str, Any]) -> list[str]:
    pop_names = {"all_test_reads": "All test reads (clean + simulated degraded)", "clean_test": "Clean test images"}
    rank_names = {"reliability": "Reliability score", "top_prob": "Top-class probability",
                  "set_size": "Conformal set size", "random": "Random (mean of orders)"}
    h = d["headline"]
    pct = lambda v: "n/a" if v is None else _num(100 * v, 1, "%")                       # noqa: E731
    count = lambda v: "n/a" if v is None else (str(v) if float(v).is_integer() else _num(v, 1))  # noqa: E731
    share = lambda v: _num(100 * v, 0, "%")                                              # noqa: E731
    gain = h.get("gain_vs_all") or {}
    out = ["## Deferral (risk–coverage) curve", "",
           f"Source: `eval/deferral.json` · pre-registered in {d.get('pre_registration', 'docs/VALIDATION_ROUND3.md')} "
           f"· production model `{d.get('production_model')}`. Each read is scored by the model that produced it in "
           "predictions.csv (counts per population below). Reads are deferred to closer review in ranking order; "
           "accuracy is four-class accuracy on the kept reads.", "",
           f"- **Headline.** {pop_names.get(h['population'], h['population'])}, {share(h['fraction'])} deferred by "
           f"reliability score ({h.get('deferred')} of {d['populations'][h['population']]['n_reads']} reads): "
           f"accuracy on kept reads {fmt(h['accuracy_kept'], pct=True)}, against "
           f"{fmt(h['accuracy_all'], pct=True)} with no deferral and {pct(h['accuracy_random_20'])} for random "
           f"deferral of {share(h['fraction'])}."]
    if gain.get("estimate") is not None:
        out.append(f"- Paired change, kept minus all (in addition to the pre-registered headline): "
                   f"{_num(100 * gain['estimate'], 1)} points [{_num(100 * gain['lo'], 1)}, {_num(100 * gain['hi'], 1)}].")
    out += ["", "Each cell: accuracy on kept reads · share of all errors deferred · invasive reads called non-invasive "
            "among kept reads · in situ reads called invasive among kept reads (random: means over orders)."]
    for key, pop in d["populations"].items():
        ranks = list(pop["rankings"])
        out += ["", f"### {pop_names.get(key, key)} ({pop['n_reads']} reads, {pop['n_groups']} groups)", "",
                f"Scoring models: {pop.get('scoring') or ', '.join(f'{n:,} by {m}' for m, n in pop['models'].items())}",
                "",
                "| Deferred | Kept | " + " | ".join(rank_names.get(r, r) for r in ranks) + " |",
                "|---|---|" + "---|" * len(ranks)]
        for i, base in enumerate(pop["rankings"][ranks[0]]):
            cells = [f"{pct(row['accuracy'])} · {pct(row['errors_deferred_share'])} · "
                     f"{count(row['invasive_missed_kept'])} · {count(row['insitu_overcalled_kept'])}"
                     for row in (pop["rankings"][r][i] for r in ranks)]
            out.append(f"| {share(base['fraction'])} | {base['kept']} | " + " | ".join(cells) + " |")
    return out + [""] + [f"- {n}" for n in d.get("notes", [])] + [""]


def section_margin(m: Mapping[str, Any]) -> list[str]:
    ci = m.get("threshold_ci") if isinstance(m.get("threshold_ci"), Mapping) else {}
    threshold = fmt({**ci, "estimate": m["threshold"]}, digits=5)    # P near 1: 0.99982, not 1.000
    out = ["## Simulated margin read", "", f"Source: `eval/margin.json` · threshold P = {threshold}, from "
           f"{m.get('n_benign_cal_images')} benign calibration images.", "",
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


TCGA_ROWS = (("FFPE tumor slides called invasive (sensitivity)", ("ffpe_tumor_sensitivity",)),
             ("Frozen tumor slides called invasive (sensitivity)", ("frozen_tumor_sensitivity",)),
             ("Frozen normal slides not called invasive (specificity)", ("frozen_tumor_vs_normal", "specificity")),
             ("Frozen tumor vs frozen normal, AUROC of the slide score", ("frozen_tumor_vs_normal", "auroc")),
             ("Tumor slides with a crop in tier 1–2", ("tumor_tier_1_2_rate",)))


def _get(d: Mapping[str, Any] | None, path: Sequence[str]) -> Any:
    for key in path:
        d = (d or {}).get(key)
    return d


def _tcga_cell(block: Mapping[str, Any] | None, auroc: bool) -> str:
    return fmt(block) if auroc else fmt(block, pct=True)


def section_external_tcga(d: Mapping[str, Any]) -> list[str]:
    """TCGA-BRCA external test of the production model: protocol25 (top level) and same8 (cohorts.same8), BACH vs
    TCGA, sensitivity analysis S1, parity, tile equivalence and latency, all quoted from eval/external_tcga.json.
    S2 and S3 stay in the JSON."""
    rec = d.get("recipe") or {}
    cohorts = {"protocol25": d, **{k: v for k, v in (d.get("cohorts") or {}).items() if k == "same8"}}
    names = [n for n in ("protocol25", "same8") if n in cohorts]
    label = {"protocol25": "25 patients (tile-API crops)", "same8": "first 8 patients (openslide crops)"}
    out = ["## TCGA-BRCA external test (production model)", "",
           f"Source: `eval/external_tcga.json` · pre-registered in `{d.get('protocol')}` before the run · model "
           f"`{d.get('model')}` (frozen UNI2-h, {d.get('n_views')} view per 512-px region, as in its BACH embeddings), "
           f"BACH temperature {_num(float(rec.get('temperature', 0)), 3)}, APS q-hat "
           f"{_num(float(rec.get('qhat_aps', 0)), 3)} (alpha {rec.get('alpha')}) with randomized sets as in the app, "
           "unchanged tier rule · no retraining and no threshold changes.", "",
           "TCGA-BRCA has no benign or in situ labels. Only invasive carcinoma (Primary Tumor slides) and normal "
           "tissue (Solid Tissue Normal slides, frozen) are tested, and labels are per slide, not per crop. A "
           "slide's score is the maximum calibrated P(Invasive) over its crops (up to 6 random tissue-rich fields); "
           "a slide is called invasive when any crop's top class is Invasive. Intervals: 95% bootstrap over patients. "
           "The reliability flag and the OOD percentile are not computed (CONCH and MUSK are not on this laptop), "
           "so `reliability_flag` is false for every crop.", ""]
    out += ["| Analysis | " + " | ".join(label[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for text, path in TCGA_ROWS:
        out.append(f"| {text} | " + " | ".join(_tcga_cell(_get(cohorts[n].get("primary"), path), "auroc" in path)
                                              for n in names) + " |")
    gaps = []
    for n in names:
        g = _get(cohorts[n].get("primary"), ("ffpe_vs_frozen_tumor_sensitivity_gap", "estimate"))
        gaps.append("n/a" if g is None else f"{_num(100 * g, 1)} points")
    out.append("| FFPE minus frozen tumor sensitivity | " + " | ".join(gaps) + " |")
    out.append("| Patients / slides / crops | " + " | ".join(
        f"{cohorts[n]['n_patients']} / {cohorts[n]['n_slides']} / {cohorts[n]['n_crops']}" for n in names) + " |")
    for n in names:
        c = cohorts[n]
        notes = []
        if c.get("slides_without_crops"):
            notes.append("no tissue-rich field: " + ", ".join(f"{s['patient_id']} {s['group']}"
                                                            for s in c["slides_without_crops"]))
        if c.get("slides_short_crops"):
            notes.append("fewer than 6 crops: " + ", ".join(f"{s['patient_id']} {s['group']} ({s['n_crops']})"
                                                         for s in c["slides_short_crops"]))
        if c.get("failed_crops"):
            notes.append(f"{len(c['failed_crops'])} crops failed to score")
        if notes:
            out += ["", f"{label[n].capitalize()}: " + "; ".join(notes) + "."]

    b = d.get("bach_reference") or {}
    if b:
        p = d.get("primary") or {}
        rows = (("Invasive / tumor called invasive", b.get("invasive_sensitivity"),
                 "FFPE " + fmt(p.get("ffpe_tumor_sensitivity"), pct=True) + "; frozen "
                 + fmt(p.get("frozen_tumor_sensitivity"), pct=True)),
                ("Normal not called invasive", b.get("normal_specificity"),
                 fmt(_get(p, ("frozen_tumor_vs_normal", "specificity")), pct=True)),
                ("Invasive vs normal AUROC", b.get("invasive_vs_normal_auroc"),
                 fmt(_get(p, ("frozen_tumor_vs_normal", "auroc")))),
                ("Invasive / tumor in tier 1–2", b.get("invasive_tier_1_2_rate"),
                 fmt(p.get("tumor_tier_1_2_rate"), pct=True)))
        out += ["", "### BACH test vs TCGA (25 patients)", "",
                f"BACH: {b.get('n_images')} held-out test images (unit: {b.get('unit')}); TCGA: slides, with frozen "
                "normal slides as the normal comparator. The units differ, so the comparison is indicative.", "",
                "| Statistic | BACH test | TCGA |", "|---|---|---|"]
        for text, bach, tcga in rows:
            auroc = "AUROC" in text
            out.append(f"| {text} | {fmt(bach) if auroc else fmt(bach, pct=True)} | {tcga} |")
        flag = _get(b, ("reliability_flag_rate_clean", "estimate"))
        out.append(f"| Reliability flag rate | {'n/a' if flag is None else _num(100 * flag, 1, '%')} (clean test) | "
                   "not computed |")

    out += ["", "### Pre-registered sensitivity analysis S1: slides with exactly 6 crops", "",
            "| Analysis | " + " | ".join(label[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for text, path in TCGA_ROWS:
        out.append(f"| {text} | " + " | ".join(
            _tcga_cell(_get(cohorts[n].get("sensitivity_s1_six_crop_slides"), path), "auroc" in path)
            for n in names) + " |")
    out.append("| Slides analysed | " + " | ".join(
        str(_get(cohorts[n], ("sensitivity_s1_six_crop_slides", "n_slides"))) for n in names) + " |")

    out += ["", "### Checks behind the run", ""]
    par = d.get("parity") or {}
    if par:
        out.append(f"- Parity gate ({par.get('image_source')}): {len(par.get('images', []))} BACH test images "
                   f"embedded on the laptop match the stored Delta embeddings with region cosine ≥ "
                   f"{_num(float(par['region_cosine_min']), 5)} (mean {_num(float(par['region_cosine_mean']), 5)}); "
                   f"calibrated probabilities differ from `predictions.csv` by at most "
                   f"{_num(float(par['max_abs_prob_diff']), 4)}; pass = {par.get('pass')}.")
    te = d.get("tile_equivalence") or {}
    if te:
        out.append(f"- Tile source: re-extracting {te.get('n_crops_compared')} openslide crops from their manifest "
                   f"coordinates through the GDC tile API gave a mean absolute pixel difference of at most "
                   f"{te.get('mad_max')} (0–255 units). The tile path picked the same fields of view on "
                   f"{te.get('n_slides_picks_identical')} of {te.get('n_slides')} slides; decision: {te.get('decision')}.")
        diag = te.get("thumbnail_diagnostic") or {}
        if diag:
            out.append(f"- Why picks can differ (one slide downloaded and checked): {diag.get('explanation')}")
    lat = d.get("latency_s_per_image") or {}
    e2e = lat.get("end_to_end_s") or {}
    if e2e.get("median") is not None:
        emb = lat.get("embedding_only_s") or {}
        out.append(f"- Latency per 2048 × 1536 crop ({lat.get('hardware')}): median {_num(e2e['median'], 2)} s "
                   f"(IQR {_num(e2e['p25'], 2)}–{_num(e2e['p75'], 2)} s, n {e2e['n']}, first "
                   f"{lat.get('warmup_excluded')} excluded); embedding alone median {_num(emb['median'], 2)} s. "
                   "Single fields, not whole slides.")
    if d.get("limitations"):
        out += ["", "Limitations of this test:", ""] + [f"- {x}" for x in d["limitations"]]
    return out + [""]


def section_foreign(name: str, data: Mapping[str, Any]) -> list[str]:
    blocks = ci_blocks({k: v for k, v in data.items() if k not in schemas.EVAL_ENVELOPE})
    out = [f"### `eval/{name}.json`", ""]
    if not blocks:
        return out + ["(no estimate/CI blocks; see the file)", ""]
    return out + [f"- {label}: {fmt(b)}" for label, b in blocks] + [""]           # every block, no truncation


DELETION_ARMS = (("random", "{n} random regions"), ("risk", "Top {n} tile-probe regions"),
                 ("concept", "Top {n} regions of the top concept"), ("contribution", "Top {n} regions by exact contribution"))


def section_deletion(d: Mapping[str, Any], tile_contrib: Mapping[str, Any] | None) -> list[str]:
    """Per model and population: drop in P(aggressive) and in the centered calibrated logit for each removal arm, and
    each arm minus random (paired), with 95% patient-grouped bootstrap CIs."""
    s = d.get("settings") or {}
    n = s.get("n_tiles", "k")
    out = ["### `eval/deletion.json`", "",
           f"Source: `eval/deletion.json` · held-out test images · {s.get('n_random', '?')} random {n}-region draws per "
           "image · drops are full read minus read with the regions removed and re-pooled; logit = centered "
           "calibrated logit (logit_target - mean over classes, divided by the temperature)."]
    ident = (tile_contrib or {}).get("identity") or {}
    if ident.get("max_abs_error") is not None:
        out += ["", f"Exact contributions (`eval/tile_contrib.json`): constant plus region contributions reproduce "
                f"each read's centered logit with max error {ident['max_abs_error']:.2g} over "
                f"{ident['n_reads_checked']} reads (tolerance {ident['tolerance']:g})."]
    for model, m in (d.get("models") or {}).items():
        if "skipped" in m:
            out += ["", f"`{model}`: skipped ({m['skipped']})."]
            continue
        for pop in ("predicted_aggressive", "all_eligible"):
            p = m.get(pop) or {}
            if not p.get("n"):
                continue
            out += ["", f"`{model}` · {pop.replace('_', ' ')} (n {p['n']}, {p.get('n_groups', '?')} groups)", "",
                    "| Removed | Drop in P | Drop in logit | Minus random (P) | Minus random (logit) |",
                    "|---|---|---|---|---|"]
            def cell(key: str, digits: int) -> str:     # n and groups are in the heading
                block = p.get(key)
                return fmt({k: block.get(k) for k in ("estimate", "lo", "hi")} if block else None, digits=digits)

            for arm, label in DELETION_ARMS:
                if f"mean_drop_{arm}" not in p:
                    continue
                diff = (cell(f"{arm}_minus_random", 3), cell(f"{arm}_minus_random_logit", 2)) if arm != "random" \
                    else ("", "")
                out.append(f"| {label.format(n=n)} | {cell(f'mean_drop_{arm}', 3)} | "
                           f"{cell(f'mean_drop_logit_{arm}', 2)} | {diff[0]} | {diff[1]} |")
        for note in ("contribution_arm", "caveat"):
            if m.get(note):
                out += ["", f"`{model}`: {m[note]}."]
    return out + [""]


def review_lines(art: Path) -> list[str]:
    path = schemas.eval_json(art, "spotcheck")
    lines = ["## Reviewed checks", ""]
    if path.is_file():
        sc = io.read_json(path)
        kind = sc.get("reviewer_kind", "unknown")
        who = "AI-reviewed, not a pathologist" if kind == "ai" else (
            "human-reviewed" if kind == "human" else "mixed AI and human review" if kind == "mixed"
            else f"reviewer kind {kind!r}")
        counts: dict[str, int] = {}                  # reviewer names as recorded, never assumed
        for r in sc.get("reviews", []):
            name = str(r.get("reviewer") or "unnamed reviewer")
            counts[name] = counts.get(name, 0) + 1
        by = "; ".join(f"{name} × {n}" for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        summary = sc.get("summary") or {}
        lines.append(f"- Concept top-tile and cell-overlay spot check: **{who}**" + (f" (reviews by {by})" if by else "")
                     + " · `eval/spotcheck.json`. Plausibility of the top-scoring tiles per concept (acceptable / "
                     "reviewed; the rest unsure or not acceptable):")
        lines += ["", "| Feature | Acceptable |", "|---|---|"]
        for cid, b in sorted((summary.get("concept") or {}).items(), key=lambda kv: -kv[1].get("acceptable", 0)):
            lines.append(f"| {cid} | {b.get('acceptable', 0)} / {b.get('n', 0)} |")
        cells = summary.get("cells") or {}
        if cells.get("n"):
            lines.append(f"| cell segmentation overlays | {cells.get('acceptable', 0)} / {cells['n']} |")
        notes = {}
        for r in sc.get("reviews", []):
            cid = r.get("item_id", "").split(":")[1] if r.get("item_id", "").count(":") >= 2 else ""
            if r.get("verdict") != "acceptable" and r.get("note") and cid not in notes:
                notes[cid] = r["note"]
        if notes:
            lines += [""] + [f"- {cid}: e.g. \"{note}\"" for cid, note in sorted(notes.items())]
        lines.append("")
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
    lines = ["# Sepia results", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA — these numbers come from generated test data and are NOT results.**", ""]
    lines += ["> Research prototype — not for clinical use. Every number below is quoted from the eval file named "
              "beside it, with its 95% cluster-bootstrap CI (patient / pseudo-patient groups) unless stated.", ""]
    sha = sorted({str(d["git_sha"]) for d in present.values() if d.get("git_sha") not in (None, "", "None")})
    lines += [f"Generated {io.utc_now()} from {len(present)} eval files (git {', '.join(sha) or 'n/a'}).", ""]
    renderers = {"qc": section_qc, "metrics": section_metrics, "calibration": section_calibration, "conformal": section_conformal,
                 "conformal_resplit": section_conformal_resplit,
                 "thunder": section_thunder, "queue_sim": section_queue, "margin": section_margin,
                 "deferral": section_deferral, "external_tcga": section_external_tcga}
    for name in OWN:
        if name in present:
            lines += renderers[name](present[name])
    foreign = [n for n in FOREIGN if n in present and n not in ("spotcheck", "tile_contrib")]
    if foreign:
        lines += ["## Other evaluations (concepts, reliability, retrieval, deletion, cells)", ""]
        for name in foreign:
            lines += (section_deletion(present[name], present.get("tile_contrib")) if name == "deletion"
                      else section_foreign(name, present[name]))
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
