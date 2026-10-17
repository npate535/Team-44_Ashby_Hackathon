"""Synthetic, contract-valid cases for UI development and end-to-end tests (never real data, never into fixtures/).

    uv run python -m backend.dev_cases --out artifacts/dev [--fill 6] [--mean-interarrival-s 120] [--force]
    PATHQUEUE_ARTIFACTS_ROOT=artifacts/dev make demo

Writes an artifacts-root layout under --out (default artifacts/dev, git-ignored):

    cases/DEV_*.json        Case JSONs (core.contract.Case, synthetic: true, eval_only kept so leak tests bite)
    images/*.jpg            generated striped placeholders labeled SYNTHETIC (non-tissue), incl. reference cases
    cells/*_overlay.png     generated dot-pattern overlays in the cells.overlay_colors
    spotcheck/items.json    spot-check items (core.schemas.SPOTCHECK_ITEMS) + generated crops
    eval/*.json             only with --with-eval: made-up metrics (synthetic: true) for Metrics-page development
    meta/dev_cases.json     provenance (core.io.write_meta, synthetic: true)

Cases are derived from fixtures/demo_0001.json (biopsy) and demo_0002.json (surgical) and cover the plan §14 demo
path: routine, low-confidence, low-reliability, insufficient-tissue, empty-set and confident-invasive biopsies; the
biopsy hero (Invasive, conformal set {InSitu, Invasive}, drafted p63/calponin + ER/PR/HER2/Ki-67, cells ok) arriving
last; and simulated frozen sections: the surgical hero (tumor at the superior margin), a clear one, a degraded
variant (reliability flag, defer) and a composite field. Flags come from core.tiers.flag_for. Biopsy arrivals follow
the settings sim.arrival Poisson process (seeded); surgical cases are held for injection by the simulator.

Rule ids and items mirror BUILD_PLAN §A14 for display only; the real drafts come from Agent 4's rules in step 08.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from core import cli, config, contract, io, schemas, tiers

PREFIX = "DEV"
SRC_W, SRC_H, TILE, ROWS, COLS = 2048, 1536, 512, 3, 4
SEG = 256
DISPLAY_W = 1024                  # placeholder JPEG width (contract coordinates stay in source pixels)
THRESHOLD = 0.5                   # synthetic margin threshold (the real one comes from calibration, §A12)
HUES = {"Normal": (92, 140, 110), "Benign": (120, 120, 170), "InSitu": (170, 120, 60), "Invasive": (176, 51, 92)}
LANE_HUES = {"biopsy": (74, 59, 143), "surgical": (176, 51, 92)}

ORDERS = {
    "breast_insitu_vs_invasive": ("stain", ["p63", "calponin"],
                                  "Myoepithelial markers help distinguish in situ from invasive disease."),
    "breast_invasive": ("stain", ["ER", "PR", "HER2", "Ki-67"], "Standard biomarker panel if invasion is confirmed."),
    "breast_insitu": ("stain", ["ER"], "ER status is recommended for ductal carcinoma in situ."),
    "uncertain_benign_atypia": ("deeper_levels", ["Deeper levels", "CK5/6", "ER"],
                                "Benign and in situ are both possible; deeper levels and CK5/6 + ER help."),
    "low_quality": ("recut", ["Recut", "Deeper levels"], "Image reliability is low; a recut may improve the read."),
    "insufficient": ("new_sample", ["New sample"], "Too little tissue passed quality control."),
}

# role, true label, probs (cfg class order), conformal set, reliability flag, qc tiles kept, rules, cells
BIOPSY_SCENARIOS: list[dict[str, Any]] = [
    {"role": "routine_normal", "truth": "Normal", "probs": [0.86, 0.10, 0.03, 0.01], "set": ["Normal"]},
    {"role": "routine_benign", "truth": "Benign", "probs": [0.12, 0.80, 0.06, 0.02], "set": ["Benign"]},
    {"role": "benign_vs_insitu", "truth": "Benign", "probs": [0.05, 0.52, 0.38, 0.05], "set": ["Benign", "InSitu"],
     "rules": ["uncertain_benign_atypia"]},
    {"role": "invasive_confident", "truth": "Invasive", "probs": [0.01, 0.02, 0.06, 0.91], "set": ["Invasive"],
     "rules": ["breast_invasive"], "cells": True},
    {"role": "low_reliability", "truth": "InSitu", "probs": [0.08, 0.17, 0.62, 0.13], "set": ["InSitu"],
     "reliability_flag": True, "rules": ["low_quality"]},
    {"role": "insufficient", "truth": "Benign", "probs": [0.30, 0.55, 0.10, 0.05], "set": ["Benign"],
     "qc_tiles": 2, "rules": ["insufficient"]},
    {"role": "empty_set", "truth": "Normal", "probs": [0.45, 0.40, 0.10, 0.05], "set": []},
]
HERO = {"role": "hero_biopsy", "truth": "Invasive", "probs": [0.02, 0.05, 0.21, 0.72], "set": ["InSitu", "Invasive"],
        "rules": ["breast_insitu_vs_invasive", "breast_invasive"], "cells": True}
FILL = [
    {"role": "fill_normal", "truth": "Normal", "probs": [0.78, 0.17, 0.04, 0.01], "set": ["Normal"]},
    {"role": "fill_benign", "truth": "Benign", "probs": [0.09, 0.84, 0.05, 0.02], "set": ["Benign"]},
]


def _id(n: int) -> str:
    return f"{PREFIX}_{n:04d}"


def _ref_id(n: int) -> str:
    return f"{PREFIX}REF_{n:02d}"


# ---- generated media (never tissue) --------------------------------------------------------------------------------------
def placeholder(path: Path, hue: tuple[int, int, int], label: str, size: tuple[int, int] = (DISPLAY_W, 768)) -> None:
    img = Image.new("RGB", size, (245, 242, 236))
    draw = ImageDraw.Draw(img)
    step = max(8, size[0] // 24)
    for i in range(-size[1], size[0], step * 2):
        draw.line([(i, 0), (i + size[1], size[1])], fill=hue, width=max(2, step // 3))
    for r in range(ROWS + 1):
        y = round(r * size[1] / ROWS)
        draw.line([(0, y), (size[0], y)], fill=(27, 26, 46), width=1)
    for c in range(COLS + 1):
        x = round(c * size[0] / COLS)
        draw.line([(x, 0), (x, size[1])], fill=(27, 26, 46), width=1)
    draw.rectangle([(6, 6), (min(size[0] - 6, 330), 34)], fill=(27, 26, 46))
    draw.text((12, 14), f"SYNTHETIC PLACEHOLDER - {label}", fill=(245, 242, 236))
    io.save_image(path, img, **({"quality": 85} if path.suffix == ".jpg" else {}))


def overlay(path: Path, colors: Mapping[str, str], seed: int, size: int = 256) -> None:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (245, 242, 236))
    draw = ImageDraw.Draw(img)
    palette = list(colors.values())
    for _ in range(140):
        x, y, r = rng.randrange(size), rng.randrange(size), rng.randrange(3, 7)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=palette[rng.randrange(len(palette))], width=2)
    draw.text((6, size - 16), "SYNTHETIC", fill=(27, 26, 46))
    io.save_image(path, img)


# ---- case construction ---------------------------------------------------------------------------------------------------
def _tiles(base: Sequence[Mapping[str, Any]], risk_scale: float, keep: int | None) -> list[dict[str, Any]]:
    tiles = []
    for tile in base:
        t = copy.deepcopy(dict(tile))
        t["risk"] = round(min(0.99, max(0.01, t["risk"] * risk_scale)), 3)
        tiles.append(t)
    if keep is not None:
        tiles = sorted(tiles, key=lambda t: -t["risk"])[:keep]
        tiles.sort(key=lambda t: t["idx"])
    return tiles


def _concepts(base: Sequence[Mapping[str, Any]], sign: float, kept: set[int], cells: bool) -> list[dict[str, Any]]:
    out = []
    for concept in base:
        c = copy.deepcopy(dict(concept))
        if c.get("source") == "cell" and not cells:
            continue
        c["contribution"] = round(c["contribution"] * sign, 3)
        c["top_tiles"] = [i for i in c["top_tiles"] if i in kept] or [min(kept)]
        out.append(c)
    return out


def _segments(hot: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    """24 margin segments: 8 per long edge, 6 per short edge; p from `hot[margin][k]` (default 0.03)."""
    segs = []
    for name, n, xy in (("superior", SRC_W // SEG, lambda k: (k * SEG, 0)),
                        ("inferior", SRC_W // SEG, lambda k: (k * SEG, SRC_H - SEG)),
                        ("medial", SRC_H // SEG, lambda k: (0, k * SEG)),
                        ("lateral", SRC_H // SEG, lambda k: (SRC_W - SEG, k * SEG))):
        ps = list(hot.get(name, ())) + [0.03] * n
        for k in range(n):
            x, y = xy(k)
            segs.append({"margin": name, "x": x, "y": y, "size": SEG, "p": round(ps[k], 3)})
    return segs


def _margin(hot: Mapping[str, Sequence[float]], defer: bool, mpp: float) -> dict[str, Any]:
    segs = _segments(hot)
    per = {m: max(s["p"] for s in segs if s["margin"] == m) for m in contract.MARGIN_NAMES}
    above = [m for m in contract.MARGIN_NAMES if per[m] > THRESHOLD]
    involved = [] if defer else above                 # ruling M8: a deferred read lists no involved margins
    call = "defer" if defer else ("tumor_at_margin" if involved else "clear")
    lengths = {m: round(sum(1 for s in segs if s["margin"] == m and s["p"] > THRESHOLD) * SEG * mpp, 1)
               for m in involved}
    return {"suggested_call": call, "involved": involved, "per_margin": per, "strip_um": round(SEG * mpp, 2),
            "threshold": THRESHOLD, "segments": segs, "involved_length_um": lengths or None}


def _findings_and_summary(case: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[list[dict], dict]:
    pred = case["prediction"]
    classes = pred["classes"]
    top_i = max(range(len(classes)), key=lambda i: pred["probs"][i])
    top = classes[top_i]
    labels = cfg["ui"]["class_labels"]
    findings = [{"id": "F1", "kind": "prediction",
                 "data": {"top": top, "p": pred["probs"][top_i], "set": pred["conformal_set"]}}]
    ranked = sorted((c for c in case["concepts"] if c.get("source") != "cell"), key=lambda c: -abs(c["contribution"]))
    for c in ranked[:2]:
        findings.append({"id": f"F{len(findings) + 1}", "kind": "concept",
                         "data": {"concept": c["id"], "contribution": c["contribution"], "tiles": c["top_tiles"]}})
    agree = sum(1 for s in case["similar_cases"] if s["label"] == top)
    findings.append({"id": f"F{len(findings) + 1}", "kind": "retrieval",
                     "data": {"agree": agree, "of": len(case["similar_cases"]), "label": top}})
    findings.append({"id": f"F{len(findings) + 1}", "kind": "quality",
                     "data": {"reliability_flag": case["quality"]["reliability_flag"]}})
    cells = case.get("cells") or {}
    if cells.get("status") == "ok":
        feat = cells["features"]["nuc_area_cv"]
        findings.append({"id": f"F{len(findings) + 1}", "kind": "cells",
                         "data": {"feature": "nuc_area_cv", "value": feat["value"],
                                  "ref_percentile": feat["ref_percentile"], "tiles": [t["idx"] for t in cells["tiles"]]}})
    if case.get("margin"):
        m = case["margin"]
        findings.append({"id": f"F{len(findings) + 1}", "kind": "margin",
                         "data": {"suggested_call": m["suggested_call"], "involved": m["involved"],
                                  "per_margin": m["per_margin"]}})
    by_kind = {f["kind"]: f for f in findings}
    in_set = [s for s in pred["conformal_set"] if s != top]
    first = f"Suggested category: {labels[top].lower()} (calibrated probability {pred['probs'][top_i]:.2f})"
    first += (", with " + " and ".join(labels[s].lower() for s in in_set) + " also in the confidence set."
              if in_set else ".")
    sentences = [{"text": first, "findings": ["F1"]}]
    concept_f = [f for f in findings if f["kind"] == "concept"]
    if concept_f:
        names = [next(c["label"] for c in case["concepts"] if c["id"] == f["data"]["concept"]).lower()
                 for f in concept_f]
        sentences.append({"text": f"{names[0].capitalize()} is the feature most associated with the suggestion"
                                  + (f", followed by {names[1]}." if len(names) > 1 else "."),
                          "findings": [f["id"] for f in concept_f]})
    if "cells" in by_kind:
        pct = round(by_kind["cells"]["data"]["ref_percentile"] * 100)
        sentences.append({"text": f"Measured variation in nuclear size is higher than in {pct}% of benign reference "
                                  "cases.", "findings": [by_kind["cells"]["id"]]})
    r = by_kind["retrieval"]["data"]
    sentences.append({"text": f"{r['agree']} of {r['of']} similar reference cases carry the same suggested category.",
                      "findings": [by_kind["retrieval"]["id"]]})
    if "margin" in by_kind:
        m = by_kind["margin"]["data"]
        text = {"clear": "Suggested margin call: margins clear (simulated frozen section).",
                "defer": "Suggested margin call: defer to permanent sections (low reliability, simulated).",
                "tumor_at_margin": "Suggested margin call: tumor at the " + " and ".join(m["involved"])
                                   + " margin (simulated frozen section)."}[m["suggested_call"]]
        sentences.append({"text": text, "findings": [by_kind["margin"]["id"]]})
    elif case["quality"]["reliability_flag"]:
        sentences.append({"text": "The reliability flag is raised; image quality may limit this read.",
                          "findings": [by_kind["quality"]["id"]]})
    return findings, {"generator": cfg["summary"]["template_generator"],
                      "sentences": sentences[: int(cfg["summary"]["max_sentences"])]}


def _agent(rules: Sequence[str], lane: str) -> dict[str, Any]:
    if lane == "surgical":
        return {"status": "none", "orders": [],
                "blocked": [{"action": "draft_stains", "reason": "Stains are not drafted for simulated frozen sections."}]}
    orders = [{"type": ORDERS[r][0], "items": list(ORDERS[r][1]), "rule_id": r, "rationale": ORDERS[r][2]}
              for r in rules]
    return {"status": "drafted" if orders else "none", "orders": orders}


def build_case(base: Mapping[str, Any], scenario: Mapping[str, Any], case_id: str, offset: float,
               cfg: Mapping[str, Any], refs: Sequence[str]) -> dict[str, Any]:
    classes = list(cfg["classes"])
    case = copy.deepcopy(dict(base))
    lane = scenario.get("lane", "biopsy")
    probs = [round(p, 4) for p in scenario["probs"]]
    risk = probs[classes.index(cfg["aggressive_class"])]
    keep = scenario.get("qc_tiles")
    tiles = _tiles(base["tiles"], risk / 0.72 if risk > 0 else 0.1, keep)
    kept = {t["idx"] for t in tiles}
    cells_on = bool(scenario.get("cells"))
    carcinoma = scenario["truth"] in cfg["carcinoma_classes"]
    case.update({
        "case_id": case_id, "lane": lane, "simulated": lane == "surgical", "synthetic": True,
        "arrived_offset_s": round(offset, 1),
        "image": {**base["image"], "display": schemas.display_ref(case_id)},
        "quality": {"qc_tiles_kept": len(tiles), "ood_percentile": 0.97 if scenario.get("reliability_flag") else 0.42,
                    "reliability": 0.31 if scenario.get("reliability_flag") else 0.81,
                    "reliability_flag": bool(scenario.get("reliability_flag"))},
        "prediction": {**base["prediction"], "classes": classes, "probs": probs, "risk_score": risk,
                       "conformal_alpha": cfg["alpha"], "conformal_set": list(scenario["set"])},
        "tiles": tiles,
        "concepts": _concepts(base["concepts"], 1.0 if carcinoma else -1.0, kept, cells_on),
    })
    top = classes[max(range(len(classes)), key=lambda i: probs[i])]
    other = "InSitu" if top == "Invasive" else ("Invasive" if top == "InSitu" else ("Benign" if top == "Normal"
                                                                                    else "Normal"))
    labels = [top, top, other, top, top] if scenario.get("set") and len(scenario["set"]) > 1 else [top] * 4 + [other]
    case["similar_cases"] = [{"case_id": refs[i], "label": labels[i], "similarity": round(0.93 - 0.02 * i, 2),
                              "image": schemas.display_ref(refs[i])} for i in range(5)]
    contrast = next(i for i, lab in enumerate(labels) if lab != top)
    case["contrast_case"] = {"case_id": refs[contrast], "label": labels[contrast],
                             "similarity": case["similar_cases"][contrast]["similarity"],
                             "differs_most_in": ["invasion", "duct_confined"],
                             "image": schemas.display_ref(refs[contrast])}
    agree = sum(1 for lab in labels if lab == top)
    case["agreement"] = {"conch_musk_same_top": not scenario.get("reliability_flag"), "neighbor_agreement": agree / 5}
    if cells_on:
        cells = copy.deepcopy(base.get("cells") or {})
        if cells.get("status") != "ok":
            raise ValueError("cells scenarios need the biopsy fixture's cells block")
        cells["tiles"] = [{**t, "overlay": schemas.cell_overlay_ref(case_id, t["idx"])} for t in cells["tiles"]
                          if t["idx"] in kept]
        case["cells"] = cells
    else:
        case["cells"] = {"status": "disabled"}
    case["margin"] = scenario.get("margin")
    for key in ("simulated_variant", "degradation", "source_case_ids"):
        case.pop(key, None)
    if scenario.get("variant"):
        case["simulated_variant"] = scenario["variant"]
        case["source_case_ids"] = list(scenario["sources"])
        if scenario["variant"] == "degraded":
            case["degradation"] = dict(scenario["degradation"])
    case["agent"] = _agent(scenario.get("rules", ()), lane)
    case["flag"] = tiers.flag_for(case, cfg)
    case["findings"], case["summary"] = _findings_and_summary(case, cfg)
    case["eval_only"] = {"true_label": scenario["truth"], "split": "test", "role": scenario["role"]}
    contract.Case.model_validate(case)            # fail loudly on drift
    return case


def build_dev_cases(cfg: Mapping[str, Any], out: Path, *, fill: int, mean_interarrival_s: float, seed: int,
                    fixtures: Path, with_eval: bool = False) -> list[Path]:
    """Write the dev case set under `out`; returns the case JSON paths."""
    biopsy_base = io.read_json(schemas.fixture_case_json(fixtures, "demo_0001"))
    surgical_base = io.read_json(schemas.fixture_case_json(fixtures, "demo_0002"))
    mpp = float(cfg["dataset"]["mpp"])
    rng = random.Random(seed)
    refs = [_ref_id(i + 1) for i in range(5)]
    for i, ref in enumerate(refs):
        placeholder(out / "images" / f"{ref}.jpg", HUES["Invasive" if i != 2 else "InSitu"], f"reference {ref}",
                    size=(256, 192))

    scenarios = list(BIOPSY_SCENARIOS) + [dict(FILL[i % len(FILL)], role=f"{FILL[i % len(FILL)]['role']}_{i}")
                                          for i in range(fill)]
    order = scenarios[:]
    rng.shuffle(order)
    order.append(HERO)                             # the biopsy hero arrives last (plan §14 step 1)
    written: list[Path] = []
    t = 0.0
    n = 0
    for scenario in order:
        t += rng.expovariate(1.0 / mean_interarrival_s)
        n += 1
        written.append(_write(build_case(biopsy_base, scenario, _id(n), t, cfg, refs), out, cfg))

    hero_s, clear_s = _id(n + 1), _id(n + 2)
    surgical = [
        (hero_s, {"role": "hero_surgical", "truth": "Invasive", "lane": "surgical", "probs": [0.02, 0.05, 0.21, 0.72],
                  "set": ["InSitu", "Invasive"], "margin": _margin({"superior": [0.03, 0.03, 0.03, 0.81, 0.64]},
                                                                   False, mpp)}),
        (clear_s, {"role": "surgical_clear", "truth": "Benign", "lane": "surgical", "probs": [0.15, 0.79, 0.04, 0.02],
                   "set": ["Benign"], "margin": _margin({}, False, mpp)}),
        (schemas.variant_case_id(hero_s, "blur", 3),
         {"role": "surgical_degraded", "truth": "Invasive", "lane": "surgical", "probs": [0.06, 0.24, 0.33, 0.37],
          "set": ["Benign", "InSitu", "Invasive"], "reliability_flag": True, "variant": "degraded",
          "sources": [hero_s], "degradation": {"type": "blur", "severity": 3},
          "margin": _margin({"superior": [0.2, 0.3, 0.45, 0.58, 0.41], "medial": [0.3, 0.35]}, True, mpp)}),
        (schemas.composite_case_id(hero_s, 0),
         {"role": "surgical_composite", "truth": "Invasive", "lane": "surgical", "probs": [0.05, 0.12, 0.18, 0.65],
          "set": ["InSitu", "Invasive"], "variant": "composite", "sources": [hero_s, clear_s],
          "margin": _margin({"lateral": [0.71, 0.84, 0.77, 0.62, 0.55, 0.49],
                             "superior": [0.03, 0.03, 0.03, 0.03, 0.04, 0.22, 0.58, 0.69],
                             "inferior": [0.03, 0.03, 0.03, 0.03, 0.05, 0.18, 0.61, 0.66]}, False, mpp)}),
    ]
    for case_id, scenario in surgical:
        t += mean_interarrival_s
        written.append(_write(build_case(surgical_base, scenario, case_id, t, cfg, refs), out, cfg))
    _spotcheck(cfg, out, [io.read_json(p) for p in written])
    if with_eval:
        _dev_eval(cfg, out, seed)
    return written


def _write(case: Mapping[str, Any], out: Path, cfg: Mapping[str, Any]) -> Path:
    case_id = case["case_id"]
    label = f"{case_id} ({case['lane']}{', ' + case['simulated_variant'] if case.get('simulated_variant') else ''})"
    placeholder(out / "images" / f"{case_id}.jpg", LANE_HUES[case["lane"]], label)   # lane hue: never the truth
    for i, tile in enumerate((case.get("cells") or {}).get("tiles", [])):
        overlay(schemas.case_cell_overlay(out, case_id, tile["idx"]), cfg["cells"]["overlay_colors"], seed=hash_seed(
            case_id, i))
    return io.write_json(schemas.case_json(out, case_id), case)


def hash_seed(case_id: str, i: int) -> int:
    return sum(ord(ch) for ch in case_id) * 31 + i


def _spotcheck(cfg: Mapping[str, Any], out: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    items = []
    biopsies = [c for c in cases if c["lane"] == "biopsy"]
    for concept_id in ("invasion", "pleomorphism", "duct_confined"):
        for case in biopsies[:4]:
            concept = next((c for c in case["concepts"] if c["id"] == concept_id), None)
            if concept is None:
                continue
            idx = concept["top_tiles"][0]
            name = f"{concept_id}_{case['case_id']}_{idx}.png"
            placeholder(out / "spotcheck" / name, (74, 59, 143), f"{concept_id} tile {idx}", size=(192, 192))
            items.append({"item_id": f"concept:{concept_id}:{case['case_id']}:{idx}", "kind": "concept",
                          "target": concept_id, "label": concept["label"], "case_id": case["case_id"],
                          "tile_idx": idx, "image": f"spotcheck/{name}",
                          "score": next(t["concepts"].get(concept_id) for t in case["tiles"] if t["idx"] == idx)})
    for case in cases:
        for tile in (case.get("cells") or {}).get("tiles", []):
            items.append({"item_id": f"cells:overlay:{case['case_id']}:{tile['idx']}", "kind": "cells",
                          "target": "overlay", "label": "Cell overlay", "case_id": case["case_id"],
                          "tile_idx": tile["idx"], "image": tile["overlay"], "score": None})
    io.write_json(schemas.spotcheck_items_json(out), {"created_utc": io.utc_now(), "synthetic": True, "items": items})


def _ci(rng: random.Random, est: float, half: float, n: int, groups: int) -> dict[str, Any]:
    est = round(est, 3)
    return {"estimate": est, "lo": round(est - half * (0.8 + 0.4 * rng.random()), 3),
            "hi": round(est + half * (0.8 + 0.4 * rng.random()), 3), "ci": 0.95, "n": n, "n_groups": groups}


def _dev_eval(cfg: Mapping[str, Any], out: Path, seed: int) -> None:
    """Made-up eval JSONs in the documented Task 11 shapes (estimate/lo/hi/n), labeled synthetic: UI development only."""
    rng = random.Random(seed)
    env = {"synthetic": True, "created_utc": io.utc_now(), "note": "dev_cases sample: made-up numbers for UI work"}
    n, g = 132, 60
    files = {
        "metrics": {"auroc_invasive": {"model_a": _ci(rng, 0.93, 0.04, n, g), "model_b": _ci(rng, 0.90, 0.05, n, g)},
                    "production": {"macro_f1": _ci(rng, 0.78, 0.07, n, g), "balanced_accuracy": _ci(rng, 0.79, 0.07, n, g)},
                    "insitu_invasive_confusions": {"insitu_as_invasive": 6, "invasive_as_insitu": 4}},
        "calibration": {"before": {"ece": _ci(rng, 0.11, 0.03, n, g),
                                   "bins": [{"confidence": c, "accuracy": round(min(1, c * 0.82 + 0.04), 3), "n": 13}
                                            for c in (0.15, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)]},
                        "after": {"ece": _ci(rng, 0.05, 0.02, n, g),
                                  "bins": [{"confidence": c, "accuracy": round(min(1, c * 0.97 + 0.01), 3), "n": 13}
                                           for c in (0.15, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)]}},
        "conformal": {"lac": {"coverage": _ci(rng, 0.91, 0.05, n, g), "mean_set_size": _ci(rng, 1.4, 0.15, n, g)},
                      "aps": {"coverage": _ci(rng, 0.93, 0.04, n, g), "mean_set_size": _ci(rng, 1.7, 0.2, n, g)}},
        "concepts": {"auroc": {k: _ci(rng, v, 0.04, 300, 300) for k, v in
                               (("lymphocytes", 0.94), ("necrosis", 0.81), ("pleomorphism", 0.77),
                                ("benign_stroma_fat", 0.88))}},
        "reliability": {"accuracy_by_severity": {t: [{"severity": s, "accuracy": _ci(rng, 0.8 - 0.09 * s * w, 0.06, 20, 15)}
                                                     for s in (0, 1, 2, 3)]
                                                 for t, w in (("blur", 1.0), ("jpeg", 0.5), ("stain", 0.7),
                                                              ("resample", 0.9))},
                        "error_auroc": _ci(rng, 0.74, 0.08, 240, 15)},
        "retrieval": {"precision_at_5": _ci(rng, 0.71, 0.06, n, g)},
        "queue_sim": {"policies": {p: {"aggressive_first_look_min": {"mean": _ci(rng, v, v * 0.15, 1000, 1000)}}
                                   for p, v in (("fifo", 46.0), ("tiers", 14.0), ("tiers_aging", 15.5))},
                      "assumptions": ["Poisson arrivals", "one reader, fixed read time", "made-up dev numbers"]},
    }
    for name, payload in files.items():
        io.write_json(schemas.eval_json(out, name), {"name": name, **env, **payload})


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("backend.dev_cases", "Synthetic contract-valid cases for UI development and e2e tests.")
    parser.add_argument("--out", default="artifacts/dev", help="output artifacts root (default artifacts/dev)")
    parser.add_argument("--fill", type=int, default=6, help="extra routine biopsies (default 6)")
    parser.add_argument("--with-eval", action="store_true",
                        help="also write made-up eval JSONs (synthetic: true) for Metrics-page development")
    parser.add_argument("--mean-interarrival-s", type=float, default=None,
                        help="mean biopsy inter-arrival on the simulator clock (default settings sim.arrival)")
    args = parser.parse_args(argv)
    cfg = config.load_settings(args.config)
    out = config.resolve_repo_path(args.out)
    fixtures = config.fixtures_root(cfg)
    if out.resolve() == fixtures.resolve() or fixtures.resolve() in out.resolve().parents:
        parser.error("dev cases never go into fixtures/")
    if (out / "cases").is_dir() and any((out / "cases").glob("*.json")) and not args.force:
        print(f"{out}/cases already holds cases; pass --force to rebuild")
        return 0
    if args.force and (out / "cases").is_dir():
        for stale in (out / "cases").glob(f"{PREFIX}_*.json"):
            stale.unlink()
    mean = args.mean_interarrival_s if args.mean_interarrival_s is not None else \
        float(cfg["sim"]["arrival"]["mean_interarrival_s"])
    written = build_dev_cases(cfg, out, fill=args.fill, mean_interarrival_s=mean,
                              seed=int(cfg["sim"]["arrival"]["seed"]), fixtures=fixtures, with_eval=args.with_eval)
    io.write_meta(out / "meta" / "dev_cases.json", step="backend.dev_cases", cfg=cfg,
                  inputs=[schemas.fixture_case_json(fixtures, "demo_0001"), schemas.fixture_case_json(fixtures,
                                                                                                     "demo_0002")],
                  outputs=written, args={"fill": args.fill, "mean_interarrival_s": mean}, synthetic=True)
    print(f"wrote {len(written)} synthetic cases to {out}/cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
