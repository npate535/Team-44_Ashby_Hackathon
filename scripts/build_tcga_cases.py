#!/usr/bin/env python3
"""Materialize the production model's TCGA external crops as browseable specimen artifacts (images + case JSON).

    python scripts/build_tcga_cases.py [--eval PATH] [--tcga-root DIR] [--out DIR]

Reads the served eval file <artifacts>/eval/external_tcga.json (production model, eval/external_tcga_prod.py) and
the crop PNGs under its data.tcga_root (the 25-patient cohort, data/tcga_brca_prod/protocol25). Refuses an eval file
whose model is not eval/model_choice.json `production`. Writes:
  <artifacts>/tcga/images/<crop_id>.jpg   display JPEG made as 01_build_cases makes BACH's (long side
                                          tiles.display.long_side, LANCZOS, tiles.display.jpeg_quality);
  <artifacts>/tcga/cases/<crop_id>.json   case-shaped for the biopsy UI, coordinates in 2048 x 1536 source pixels,
                                          exact region contributions, nulls for what is not computed, no labels;
  <artifacts>/tcga/index.json,
and removes specimens of earlier builds that are not in this eval file.
<artifacts> is $PATHQUEUE_ARTIFACTS_ROOT when set, else <repo>/artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

from core import config, io
from pipeline import common

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
GRID = (3, 4)                     # g512 regions of a 2048 x 1536 field


def tiles(crop: dict) -> tuple[list[dict], dict | None]:
    """(tiles, prediction.attribution) as 08_build_case_json builds them for BACH reads: one tile per QC-passing g512
    region (QC-failed regions are not listed, so the UI hatches them), each with the read's exact region contribution
    (eval/external_tcga_prod.region_contributions, 4 dp). `risk` stays null: BACH's tile risk is the CONCH tile probe,
    which is not computed for TCGA, and no other quantity is shown under that label. The attribution is null when the
    read pooled regions the case does not list (no QC-passing region), exactly as for BACH."""
    qc = sorted(int(t) for t in crop.get("qc_regions") or [])
    contrib = {int(k): float(v) for k, v in (crop.get("region_contributions") or {}).items()}
    att = crop.get("attribution")
    out = []
    for idx in qc:
        row, col = divmod(idx, 4)
        out.append({"idx": idx, "row": row, "col": col, "x": col * 512, "y": row * 512, "risk": None,
                    "concepts": {}, "attention": None,
                    "contribution": round(contrib[idx], 4) if idx in contrib else None})
    if not att or not contrib or not set(contrib) <= set(qc):
        return out, None
    return out, {"method": att["method"], "target": att["target"], "constant": round(float(att["constant"]), 4),
                 "unit": att["unit"]}


def _label(cls: str, labels: dict) -> str:
    """Display name in running text, as backend.summary writes it: 'invasive carcinoma', 'in situ carcinoma'."""
    text = str(labels.get(cls, cls))
    return text[:1].lower() + text[1:] if len(text) > 1 and text[1:2].islower() else text


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _pct(p: float) -> str:
    """Probability as the case header shows it (why.js fmtPct): '>99%', '<1%', else the rounded percent."""
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{int(100 * p + 0.5)}%"


def summary_sentences(top: str, p_aggr: float, conf: list[str], alpha: float, n_qc: int, min_qc: int,
                      labels: dict, aggressive: str) -> list[dict]:
    """Template summary in the BACH template's wording, with display labels and the header's percent format.
    Sentence 1 cites the prediction (F1); sentence 2 cites the quality finding (F2) and says what is not computed."""
    level = f"{round(100 * (1 - alpha))}% confidence set"
    head = (f"Suggested category: {_label(top, labels)} (calibrated probability of {_label(aggressive, labels)} "
            f"{_pct(p_aggr)})")
    others = [c for c in conf if c != top]
    if not conf:
        first = f"{head}; the {level} is empty, so this read is uncertain."
    elif top not in conf:
        held = _join([_label(c, labels) for c in conf])
        first = f"{head}; the {level} holds {held} instead, so this read is uncertain."
    elif not others:
        first = f"{head}; the {level} holds this category only."
    else:
        verb = "is" if len(others) == 1 else "are"
        first = f"{head}; {_join([_label(c, labels) for c in others])} {verb} also in the {level}."
    if n_qc < min_qc:
        second = (f"Only {n_qc} tissue {'region' if n_qc == 1 else 'regions'} passed quality control (fewer than "
                  f"{min_qc}), so this read rests on little tissue.")
    else:
        second = (f"{n_qc} of {GRID[0] * GRID[1]} regions passed tissue quality control; the reliability score and "
                  f"input typicality are not computed for TCGA fields.")
    return [{"text": first, "findings": ["F1"]}, {"text": second, "findings": ["F2"]}]


def case_json(crop: dict, model: str, alpha: float, labels: dict | None = None, aggressive: str = "Invasive",
              min_qc: int = 3) -> dict:
    classes = CLASSES
    probs = [float(crop["probs"][c]) for c in classes]
    inv = float(crop["p_invasive"])
    top = crop["top"]
    conf = list(crop["conformal_set"])
    tier = int(crop["tier"])
    reasons = list(crop.get("tier_reasons") or [])
    prep = crop["prep"]
    prep_label = "frozen-section" if prep == "frozen" else "FFPE"
    n_qc = int(crop["qc_tiles_kept"])
    cid = crop["crop_id"]
    tile_rows, attribution = tiles(crop)
    findings = [
        {
            "id": "F1",
            "kind": "prediction",
            # same keys as a BACH read's prediction finding: p is P(top class), risk is P(aggressive class)
            "data": {"top": top, "p": round(float(crop["probs"][top]), 3), "set": conf,
                     "confidence": round(1 - alpha, 4), "risk": round(inv, 3)},
        },
        {
            "id": "F2",
            "kind": "quality",
            "data": {
                "reliability_flag": False,
                "reliability": None,
                "ood_percentile": None,
                "qc_tiles_kept": n_qc,
                "min_tiles_kept": min_qc,
            },
        },
    ]
    labels = labels or {}
    return {
        "schema_version": "1.0",
        "synthetic": False,
        "case_id": cid,
        "lane": "biopsy",
        "simulated": False,
        "simulated_variant": None,
        "degradation": None,
        "source_case_ids": [],
        "arrived_offset_s": 0.0,
        "cohort": "tcga",
        # the header already says "TCGA-BRCA external" (adapter laneLabel); this names the slide only
        "specimen_question": f"Field from {'a' if prep == 'frozen' else 'an'} {prep_label} slide",
        "tcga_meta": {
            "patient_id": crop["patient_id"],
            "prep": prep,
            "slide_file_name": crop.get("slide_file_name"),
        },
        "image": {
            "display": f"tcga/{cid}.jpg",
            "width": 2048,
            "height": 1536,
            "tile_size": 512,
            "grid": list(GRID),
        },
        # Not computed for TCGA (needs CONCH and MUSK): null, never a placeholder value.
        "quality": {
            "qc_tiles_kept": n_qc,
            "ood_percentile": None,
            "reliability": None,
            "reliability_flag": False,
        },
        "prediction": {
            "model": model,
            "classes": classes,
            "probs": probs,
            "risk_score": inv,
            "conformal_alpha": alpha,
            "conformal_set": conf,
            "attribution": attribution,
        },
        "concepts": [],
        "tiles": tile_rows,
        "similar_cases": [],
        "contrast_case": None,
        "agreement": {},
        "margin": None,
        "cells": {"status": "disabled"},
        "flag": {"tier": tier, "reasons": reasons},
        "findings": findings,
        "summary": {
            "generator": "template",
            "sentences": summary_sentences(top, inv, conf, alpha, n_qc, min_qc, labels, aggressive),
        },
        "agent": {"status": "none", "orders": [], "blocked": [], "n_orders": 0},
        # Read-only external specimens: no queue decisions (not part of the arrival replay).
        "runtime": {
            "tier": tier,
            "reasons": reasons,
            "allowed_actions": [],
            "arrived_sim_s": 0.0,
            "done": False,
        },
    }


def artifacts_root() -> Path:
    env = os.environ.get("PATHQUEUE_ARTIFACTS_ROOT")
    return Path(env).expanduser() if env else ROOT / "artifacts"


def main(argv: list[str] | None = None) -> int:
    art = artifacts_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=art / "eval" / "external_tcga.json")
    ap.add_argument("--model-choice", type=Path, default=art / "eval" / "model_choice.json")
    ap.add_argument("--tcga-root", type=Path, default=None, help="crop root (default: the eval file's data.tcga_root)")
    ap.add_argument("--out", type=Path, default=art / "tcga")
    a = ap.parse_args(argv)
    if not a.eval.is_file():
        print(f"missing {a.eval}", file=sys.stderr)
        return 1
    payload = json.loads(a.eval.read_text())
    production = json.loads(a.model_choice.read_text())["production"]
    model = str(payload.get("model") or "")
    if model != production:
        print(f"{a.eval} reports model {model!r}, not the production model {production!r}; refusing to build",
              file=sys.stderr)
        return 1
    tcga = a.tcga_root or Path((payload.get("data") or {}).get("tcga_root") or "")
    if not tcga.is_dir():
        print(f"crop root {tcga} not found (pass --tcga-root)", file=sys.stderr)
        return 1
    crops = payload.get("crops") or []
    settings = config.load_settings()
    display = settings["tiles"]["display"]
    labels = dict(settings["ui"]["class_labels"])
    aggressive = str(settings.get("aggressive_class") or "Invasive")
    min_qc = int(settings["qc"]["min_tiles_kept"])
    long_side, quality = int(display["long_side"]), int(display["jpeg_quality"])
    alpha = float((payload.get("recipe") or {}).get("alpha", 0.1))
    out_img, out_case = a.out / "images", a.out / "cases"
    out_img.mkdir(parents=True, exist_ok=True)
    out_case.mkdir(parents=True, exist_ok=True)
    index = []
    for crop in crops:
        cid = crop["crop_id"]
        src = tcga / crop["path"]
        if not src.is_file():
            print(f"skip missing image {src}", file=sys.stderr)
            continue
        with Image.open(src) as handle:              # display JPEG exactly as 01_build_cases makes BACH's
            io.save_image(out_img / f"{cid}.jpg", common.to_display(handle.convert("RGB"), long_side), quality=quality)
        case = case_json(crop, model, alpha, labels, aggressive, min_qc)
        (out_case / f"{cid}.json").write_text(json.dumps(case, indent=2) + "\n")
        index.append({
            "case_id": cid,
            "prep": crop["prep"],
            "patient_id": crop["patient_id"],
            "tier": crop["tier"],
            "top": crop["top"],
            "p_invasive": crop["p_invasive"],
            "conformal_set": list(crop.get("conformal_set") or []),
            "tier_reasons": list(crop.get("tier_reasons") or []),
            "qc_tiles_kept": int(crop["qc_tiles_kept"]),
            "image": f"tcga/{cid}.jpg",
            "probs": {c: float(crop["probs"][c]) for c in CLASSES},
            "model": model,
        })
    keep = {item["case_id"] for item in index}
    stale = [p for p in out_case.glob("*.json") if p.stem not in keep] + \
            [p for p in out_img.glob("*.jpg") if p.stem not in keep]
    for path in stale:                       # specimens of an earlier build (another model or cohort)
        path.unlink()
    (a.out / "index.json").write_text(json.dumps({
        "model": model,
        "n": len(index),
        "items": index,
        "n_patients": payload.get("n_patients"),
        "n_slides": payload.get("n_slides"),
        "n_crops": payload.get("n_crops"),
        "device": payload.get("device"),
        "quantized": payload.get("quantized"),
        "limitations": payload.get("limitations"),
    }, indent=2) + "\n")
    print(f"wrote {len(index)} TCGA specimens ({model}) -> {a.out}; removed {len(stale)} stale files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
