"""Write the committed demo fixtures: fixtures/demo_0001.json (biopsy), demo_0002.json (simulated surgical) and
their media. Every image is a GENERATED non-tissue placeholder (BACH-derived images are never committed, §A23).

    uv run python scripts/make_fixtures.py

The UI (team44-triage-frontend ?mock=1) and backend (PATHQUEUE_CASES_DIR=fixtures) can run on these before the
pipeline produces real artifacts. Flags come from core.tiers so fixtures never disagree with the worklist rules.
Rerun after a contract change (plan §16) together with scripts/gen_types_ts.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.dev_cases import with_contributions  # noqa: E402
from core import config, contract, tiers  # noqa: E402

OUT = REPO / "fixtures"
CFG = config.load_settings()
CLASSES = list(CFG["classes"])
W, H, TILE, ROWS, COLS = 2048, 1536, 512, 3, 4
MPP = float(CFG["dataset"]["mpp"])
LABELS = {
    "tubules": "Well-formed tubules / glands",
    "pleomorphism": "Marked nuclear pleomorphism",
    "mitoses": "Frequent mitotic figures",
    "invasion": "Tumor infiltrating stroma",
    "duct_confined": "Cells confined within ducts",
    "normal_lobules": "Normal two-layered lobules",
    "benign_stroma_fat": "Benign fibrous stroma or fat",
    "lymphocytes": "Dense lymphocytic infiltrate",
    "necrosis": "Tumor necrosis",
}
DIRECTION = {"tubules": "down", "pleomorphism": "up", "mitoses": "up", "invasion": "up", "duct_confined": "down",
             "normal_lobules": "down", "benign_stroma_fat": "down", "lymphocytes": "context", "necrosis": "up"}


def placeholder(path: Path, size: tuple[int, int], hue: tuple[int, int, int], label: str) -> None:
    """A striped, clearly synthetic image with a caption (never tissue)."""
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
    draw.rectangle([(6, 6), (min(size[0] - 6, 360), 40)], fill=(27, 26, 46))
    draw.text((12, 14), f"SYNTHETIC PLACEHOLDER - {label}", fill=(245, 242, 236))
    if path.suffix == ".png":
        img.save(path, optimize=True)
    else:
        img.save(path, quality=70, optimize=True)


def overlay(path: Path) -> None:
    """Placeholder cell overlay: generated ellipses in the CELLS_SPEC §5 type colors (not a segmentation)."""
    colors = list(CFG["cells"]["overlay_colors"].values())
    img = Image.new("RGB", (TILE, TILE), (240, 236, 230))
    draw = ImageDraw.Draw(img)
    k = 0
    for y in range(24, TILE, 48):
        for x in range(24, TILE, 48):
            color = colors[k % len(colors)]
            draw.ellipse([(x - 12, y - 9), (x + 12, y + 9)], outline=color, width=3)
            k += 1
    draw.rectangle([(4, 4), (250, 26)], fill=(27, 26, 46))
    draw.text((10, 10), "SYNTHETIC OVERLAY PLACEHOLDER", fill=(245, 242, 236))
    img.save(path, optimize=True)


def tiles_block(hot: dict[int, float], concept_bias: dict[str, dict[int, float]]) -> list[dict]:
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            idx = r * COLS + c
            concepts = {cid: round(concept_bias.get(cid, {}).get(idx, -0.3 + 0.1 * ((idx + i) % 4)), 2)
                        for i, cid in enumerate(CFG["concepts"]["ids"])}
            out.append({"idx": idx, "row": r, "col": c, "x": c * TILE, "y": r * TILE,
                        "risk": hot.get(idx, 0.12), "concepts": concepts})
    return out


def concept(cid: str, value: float, contribution: float, top: list[int], spread: float, **extra) -> dict:
    return {"id": cid, "label": LABELS[cid], "value": value, "contribution": contribution,
            "direction": DIRECTION[cid], "spread": spread, "top_tiles": top, **extra}


def biopsy() -> dict:
    case = {
        "schema_version": contract.SCHEMA_VERSION,
        "synthetic": True,
        "case_id": "demo_0001",
        "lane": "biopsy",
        "simulated": False,
        "arrived_offset_s": 340,
        "image": {"display": "images/demo_0001.jpg", "width": W, "height": H, "tile_size": TILE, "grid": [ROWS, COLS]},
        "quality": {"qc_tiles_kept": 12, "ood_percentile": 0.42, "reliability": 0.81, "reliability_flag": False},
        "prediction": {"model": "concept_bottleneck", "classes": CLASSES, "probs": [0.02, 0.05, 0.21, 0.72],
                       "risk_score": 0.72, "conformal_alpha": float(CFG["alpha"]), "conformal_set": ["InSitu", "Invasive"]},
        "concepts": [
            concept("invasion", 1.9, 0.44, [5, 6], 0.06, source="cav",
                    components={"text_z": 1.7, "cav_z": 2.1}, caption_consistent=True),
            concept("pleomorphism", 1.4, 0.21, [5, 9], 0.08),
            concept("necrosis", 0.6, 0.05, [6], 0.11),
            concept("lymphocytes", 0.9, 0.0, [10], 0.07),
            concept("mitoses", 0.3, 0.02, [5], 0.15),
            concept("duct_confined", 0.8, -0.12, [2], 0.09),
            concept("tubules", -0.7, -0.08, [0], 0.07),
            concept("normal_lobules", -1.1, -0.15, [3], 0.05),
            concept("benign_stroma_fat", -0.4, -0.06, [11], 0.06),
            {"id": "nuc_area_cv", "label": "Variation in nuclear size", "value": 0.48, "contribution": 0.21,
             "direction": "up", "top_tiles": [5, 6], "source": "cell"},
            {"id": "neo_density", "label": "Tumor cell density", "value": 3150.0, "contribution": 0.18,
             "direction": "up", "top_tiles": [5, 6], "source": "cell"},
        ],
        "tiles": tiles_block({5: 0.88, 6: 0.81, 9: 0.66, 2: 0.35, 1: 0.22},
                             {"invasion": {5: 2.3, 6: 2.0, 9: 1.2}, "pleomorphism": {5: 1.4, 9: 1.3},
                              "duct_confined": {2: 1.1}, "normal_lobules": {3: 0.9}, "lymphocytes": {10: 1.5}}),
        "similar_cases": [
            {"case_id": "demo_ref_01", "label": "Invasive", "similarity": 0.93, "image": "images/demo_ref_01.jpg"},
            {"case_id": "demo_ref_02", "label": "Invasive", "similarity": 0.91, "image": "images/demo_ref_02.jpg"},
            {"case_id": "demo_ref_03", "label": "InSitu", "similarity": 0.89, "image": "images/demo_ref_03.jpg"},
            {"case_id": "demo_ref_04", "label": "Invasive", "similarity": 0.88, "image": "images/demo_ref_04.jpg"},
            {"case_id": "demo_ref_05", "label": "Invasive", "similarity": 0.87, "image": "images/demo_ref_05.jpg"},
        ],
        "contrast_case": {"case_id": "demo_ref_03", "label": "InSitu", "similarity": 0.89,
                          "differs_most_in": ["invasion", "duct_confined"], "image": "images/demo_ref_03.jpg",
                          "detail": [   # synthetic placeholder values (pipeline 07 computes the real ones)
                              {"concept": "invasion", "label": LABELS["invasion"], "query_z": 1.62,
                               "contrast_z": -0.41, "delta": 2.03, "spotcheck": {"acceptable": 19, "reviewed": 20},
                               "reliable": True, "query_region": 5, "contrast_region": 6},
                              {"concept": "duct_confined", "label": LABELS["duct_confined"], "query_z": -0.35,
                               "contrast_z": 1.12, "delta": -1.47, "spotcheck": {"acceptable": 3, "reviewed": 20},
                               "reliable": False, "query_region": 2, "contrast_region": 1},
                          ]},
        "agreement": {"conch_musk_same_top": True, "neighbor_agreement": 0.8},
        "margin": None,
        "cells": {
            "status": "ok", "model": "hovernet_fast-pannuke", "mpp_model": float(CFG["cells"]["model_mpp"]),
            "tiles": [
                {"idx": 5, "overlay": "cells/demo_0001_5_overlay.png",
                 "counts": {"neoplastic": 212, "inflammatory": 31, "connective": 18, "dead": 4, "epithelial": 2}},
                {"idx": 6, "overlay": "cells/demo_0001_6_overlay.png",
                 "counts": {"neoplastic": 187, "inflammatory": 44, "connective": 25, "dead": 6, "epithelial": 5}},
            ],
            "features": {
                "nuc_area_cv": {"value": 0.48, "unit": "", "ref_percentile": 0.95, "contribution": 0.21},
                "neo_density": {"value": 3150.0, "unit": "per mm²", "ref_percentile": 0.91, "contribution": 0.18},
                "til_proximity": {"value": 0.22, "unit": "", "ref_percentile": 0.40, "contribution": 0.0},
            },
            "flags": {"insufficient_tumor_nuclei": False},
        },
        "findings": [
            {"id": "F1", "kind": "prediction", "data": {"top": "Invasive", "p": 0.72, "set": ["InSitu", "Invasive"]}},
            {"id": "F2", "kind": "concept", "data": {"concept": "invasion", "contribution": 0.44, "tiles": [5, 6],
                                                     "spotcheck": {"acceptable": 19, "reviewed": 20}, "reliable": True}},
            {"id": "F3", "kind": "concept", "data": {"concept": "pleomorphism", "contribution": 0.21, "tiles": [5, 9],
                                                     "spotcheck": {"acceptable": 18, "reviewed": 20}, "reliable": True}},
            {"id": "F4", "kind": "retrieval", "data": {"agree": 4, "of": 5, "label": "Invasive"}},
            {"id": "F5", "kind": "quality", "data": {"reliability_flag": False}},
            {"id": "F6", "kind": "cells", "data": {"feature": "nuc_area_cv", "value": 0.48, "ref_percentile": 0.95,
                                                    "tiles": [5, 6]}},
        ],
        "summary": {"generator": "template", "sentences": [
            {"text": "Suggested category: invasive carcinoma (calibrated probability 0.72), with in situ carcinoma "
                     "also in the confidence set.", "findings": ["F1"]},
            {"text": "Tumor cells infiltrating stroma (regions 5 and 6) is the feature most associated with the "
                     "suggestion, followed by nuclear pleomorphism.", "findings": ["F2", "F3"]},
            {"text": "Variation in nuclear size is higher than in 95% of benign reference cases (regions 5 and 6).",
             "findings": ["F6"]},
            {"text": "4 of 5 similar reference cases carry the same suggested category.", "findings": ["F4"]},
        ]},
        "agent": {"status": "drafted", "orders": [
            {"type": "stain", "items": ["p63", "calponin"], "rule_id": "breast_insitu_vs_invasive",
             "rationale": "Myoepithelial markers help distinguish in situ from invasive disease."},
            {"type": "stain", "items": ["ER", "PR", "HER2", "Ki-67"], "rule_id": "breast_invasive",
             "rationale": "Standard biomarker panel if invasion is confirmed."},
        ]},
        "eval_only": {"true_label": "Invasive", "split": "test"},
    }
    return case


def surgical() -> dict:
    hot = {(768, 0): 0.81, (1024, 0): 0.64, (1024, 1280): 0.07, (0, 512): 0.12, (1792, 768): 0.04}
    segs = [{"margin": "superior", "x": x, "y": 0, "size": 256} for x in range(0, W, 256)]
    segs += [{"margin": "inferior", "x": x, "y": H - 256, "size": 256} for x in range(0, W, 256)]
    segs += [{"margin": "medial", "x": 0, "y": y, "size": 256} for y in range(0, H, 256)]
    segs += [{"margin": "lateral", "x": W - 256, "y": y, "size": 256} for y in range(0, H, 256)]
    for s in segs:
        s["p"] = hot.get((s["x"], s["y"]), 0.03)
    per_margin = {m: max(s["p"] for s in segs if s["margin"] == m) for m in contract.MARGIN_NAMES}
    threshold = 0.5
    involved = [m for m in contract.MARGIN_NAMES if per_margin[m] > threshold]
    length = {m: round(sum(s["size"] * MPP for s in segs if s["margin"] == m and s["p"] > threshold), 2)
              for m in involved}
    case = biopsy()
    case.update(case_id="demo_0002", lane="surgical", simulated=True, arrived_offset_s=900)
    case["image"]["display"] = "images/demo_0002.jpg"
    case["cells"] = {"status": "disabled"}
    case["concepts"] = [c for c in case["concepts"] if c.get("source") != "cell"]
    case["findings"] = [f for f in case["findings"] if f["kind"] != "cells"]
    case["summary"]["sentences"] = [s for s in case["summary"]["sentences"] if "F6" not in s["findings"]]
    case["margin"] = {"suggested_call": "tumor_at_margin", "involved": involved, "per_margin": per_margin,
                      "strip_um": round(int(CFG["margin"]["strip_px"]) * MPP, 2), "threshold": threshold,
                      "segments": segs, "involved_length_um": length}
    case["findings"].append({"id": "F7", "kind": "margin", "data": {"suggested_call": "tumor_at_margin",
                                                                   "involved": involved, "per_margin": per_margin}})
    case["summary"]["sentences"].append({"text": "Suggested margin call: tumor at the superior margin "
                                                 "(simulated frozen section).", "findings": ["F7"]})
    case["agent"] = {"status": "none", "orders": [], "blocked": [
        {"action": "draft_stains", "reason": "Stains are not drafted for simulated frozen sections."}]}
    case["eval_only"] = {"true_label": "Invasive", "split": "test"}
    return case


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for case in (biopsy(), surgical()):
        pred = case["prediction"]
        case["tiles"], pred["attribution"] = with_contributions(case["tiles"], pred["probs"], pred["classes"],
                                                                CFG["aggressive_class"])
        case["flag"] = tiers.flag_for(case, CFG)
        contract.Case.model_validate(case)
        (OUT / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    placeholder(OUT / "demo_0001.jpg", (1024, 768), (74, 59, 143), "biopsy demo")
    placeholder(OUT / "demo_0002.jpg", (1024, 768), (176, 51, 92), "surgical demo")
    for k in range(1, 6):
        placeholder(OUT / f"demo_ref_{k:02d}.jpg", (256, 192), (120, 110, 160), f"reference {k}")
    for idx in (5, 6):
        overlay(OUT / f"demo_0001_{idx}_overlay.png")
    print("wrote fixtures/demo_0001.json, fixtures/demo_0002.json and placeholder media")


if __name__ == "__main__":
    main()
