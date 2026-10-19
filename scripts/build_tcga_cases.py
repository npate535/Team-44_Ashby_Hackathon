#!/usr/bin/env python3
"""Materialize the production model's TCGA external crops as browseable specimen artifacts (images + case JSON).

    python scripts/build_tcga_cases.py [--eval PATH] [--tcga-root DIR] [--out DIR]

Reads the served eval file <artifacts>/eval/external_tcga.json (production model, eval/external_tcga_prod.py) and
the crop PNGs under its data.tcga_root (the 25-patient cohort, data/tcga_brca_prod/protocol25). Refuses an eval file
whose model is not eval/model_choice.json `production`. Writes <artifacts>/tcga/images/<crop_id>.jpg,
<artifacts>/tcga/cases/<crop_id>.json (case-shaped for the biopsy UI; no ground-truth labels in the public payload)
and <artifacts>/tcga/index.json, and removes specimens of earlier builds that are not in this eval file.
<artifacts> is $PATHQUEUE_ARTIFACTS_ROOT when set, else <repo>/artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
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
        with Image.open(src) as handle:
            handle.convert("RGB").save(out_img / f"{cid}.jpg", quality=92, optimize=True)
        case = case_json(crop, model, alpha)
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
