#!/usr/bin/env python3
"""Materialize TCGA external crops as browseable specimen artifacts (images + case-shaped JSON).

    python scripts/build_tcga_cases.py

Reads artifacts/eval/external_tcga.json and data/tcga_brca_test/<path>.png.
Writes artifacts/tcga/images/<crop_id>.jpg and artifacts/tcga/cases/<crop_id>.json
(case-shaped for the biopsy UI; not ground-truth labeled in the public payload).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval" / "external_tcga.json"
TCGA = ROOT / "data" / "tcga_brca_test"
OUT_IMG = ROOT / "artifacts" / "tcga" / "images"
OUT_CASE = ROOT / "artifacts" / "tcga" / "cases"
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]


def tiles(p_invasive: float, n_qc: int) -> list[dict]:
    """3x4 g512 grid; risk filled with the case invasive score (no per-tile probe on this path)."""
    out = []
    for idx in range(12):
        row, col = divmod(idx, 4)
        out.append({
            "idx": idx, "row": row, "col": col,
            "x": col * 512, "y": row * 512,
            "risk": float(p_invasive),
            "concepts": {},
            "attention": None,
            "contribution": None,
        })
    # Contract-style: qc_tiles_kept == len(tiles). Keep all 12; QC count is recorded in quality.
    _ = n_qc
    return out


def case_json(crop: dict, model: str, alpha: float) -> dict:
    classes = CLASSES
    probs = [float(crop["probs"][c]) for c in classes]
    inv = float(crop["p_invasive"])
    top = crop["top"]
    conf = list(crop["conformal_set"])
    tier = int(crop["tier"])
    reasons = list(crop.get("tier_reasons") or [])
    prep = crop["prep"]
    prep_label = "frozen section" if prep == "frozen" else "FFPE"
    n_qc = int(crop.get("qc_tiles_kept") or 12)
    cid = crop["crop_id"]
    findings = [
        {
            "id": "F1",
            "kind": "prediction",
            "data": {"top": top, "p": round(inv, 3), "set": conf, "alpha": alpha},
        },
        {
            "id": "F2",
            "kind": "quality",
            "data": {
                "reliability_flag": False,
                "reliability": None,
                "ood_percentile": None,
                "qc_tiles_kept": n_qc,
                "min_tiles_kept": 3,
            },
        },
    ]
    set_txt = ", ".join(conf) if conf else "empty"
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
        "specimen_question": f"TCGA-BRCA external field · {prep_label}",
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
            "grid": [3, 4],
        },
        "quality": {
            "qc_tiles_kept": 12,
            "ood_percentile": 0.5,
            "reliability": 0.5,
            "reliability_flag": False,
        },
        "prediction": {
            "model": model,
            "classes": classes,
            "probs": probs,
            "risk_score": inv,
            "conformal_alpha": alpha,
            "conformal_set": conf,
            "attribution": None,
        },
        "concepts": [],
        "tiles": tiles(inv, n_qc),
        "similar_cases": [],
        "contrast_case": None,
        "agreement": {},
        "margin": None,
        "cells": {"status": "disabled"},
        "flag": {"tier": tier, "reasons": reasons},
        "findings": findings,
        "summary": {
            "generator": "template",
            "sentences": [
                {
                    "text": (
                        f"External TCGA field ({prep_label}). Suggested category: {top} "
                        f"(calibrated P(Invasive) {inv:.2f}); 90% confidence set: {set_txt}."
                    ),
                    "findings": ["F1"],
                }
            ],
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


def main() -> int:
    if not EVAL.is_file():
        print(f"missing {EVAL}", file=sys.stderr)
        return 1
    payload = json.loads(EVAL.read_text())
    crops = payload.get("crops") or []
    model = str(payload.get("model") or "uni2h_ft")
    alpha = 0.1
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_CASE.mkdir(parents=True, exist_ok=True)
    index = []
    for crop in crops:
        cid = crop["crop_id"]
        src = TCGA / crop["path"]
        if not src.is_file():
            print(f"skip missing image {src}", file=sys.stderr)
            continue
        jpg = OUT_IMG / f"{cid}.jpg"
        if not jpg.is_file() or jpg.stat().st_mtime < src.stat().st_mtime:
            Image.open(src).convert("RGB").save(jpg, quality=92, optimize=True)
        case = case_json(crop, model, alpha)
        (OUT_CASE / f"{cid}.json").write_text(json.dumps(case, indent=2) + "\n")
        index.append({
            "case_id": cid,
            "prep": crop["prep"],
            "patient_id": crop["patient_id"],
            "tier": crop["tier"],
            "top": crop["top"],
            "p_invasive": crop["p_invasive"],
            "conformal_set": list(crop.get("conformal_set") or []),
            "tier_reasons": list(crop.get("tier_reasons") or []),
            "image": f"tcga/{cid}.jpg",
            "probs": {c: float(crop["probs"][c]) for c in CLASSES},
            "model": model,
        })
    (OUT_CASE.parent / "index.json").write_text(json.dumps({
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
    print(f"wrote {len(index)} TCGA specimens -> {OUT_CASE.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
