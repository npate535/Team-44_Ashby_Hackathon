"""Tiny schema-exact downstream data root for testing step 08 and the eval suite before the producers merge.

    info = write_downstream(tmp_path / "data", cfg)      # cases, tiles, predictions, tile risk, contributions,
                                                         # reliability, retrieval, margins, case features,
                                                         # concepts_tile, composites, cells, display JPEGs
    bank = write_concept_bank(tmp_path / "concepts.yaml", cfg)
    cells_yaml = write_cell_features_yaml(tmp_path / "cell_features.yaml")

Everything is synthetic (SYNTHETIC.json marker; images are flat non-tissue colour fields). The layout:
  20 cases: train 8 / calibration 4 / test 8 (2 per class). Test lanes: TEST_SURGICAL surgical, the rest biopsy.
  HERO_BIOPSY (Invasive, set {InSitu, Invasive}); INSUFFICIENT keeps 2 QC tiles; surgical Invasive case has tumor at
  the superior margin; degraded variants of each surgical case are reliability-flagged; 2 composites.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from core import cli, io, margin, schemas

CLASSES = ("Normal", "Benign", "InSitu", "Invasive")
CONCEPT_LABELS = {
    "tubules": ("Well-formed tubules / glands", "down"), "pleomorphism": ("Marked nuclear pleomorphism", "up"),
    "mitoses": ("Frequent mitotic figures", "up"), "invasion": ("Tumor infiltrating stroma", "up"),
    "duct_confined": ("Cells confined within ducts", "down"), "normal_lobules": ("Normal two-layered lobules", "down"),
    "benign_stroma_fat": ("Benign fibrous stroma or fat", "down"), "lymphocytes": ("Dense lymphocytic infiltrate", "context"),
    "necrosis": ("Tumor necrosis", "up"),
}
CELL_FEATURES = {"nuc_area_cv": ("Variation in nuclear size", "", "higher"),
                 "neo_density": ("Tumor cell density", "per mm²", "higher"),
                 "til_proximity": ("Immune cells near tumor", "", "context")}

TEST = {  # case_id: (label, lane)
    "BACH_0101": ("Normal", "surgical"), "BACH_0102": ("Normal", "biopsy"),
    "BACH_0103": ("Benign", "biopsy"), "BACH_0104": ("Benign", "biopsy"),
    "BACH_0105": ("InSitu", "biopsy"), "BACH_0106": ("InSitu", "biopsy"),
    "BACH_0107": ("Invasive", "surgical"), "BACH_0108": ("Invasive", "biopsy"),
}
TEST_SURGICAL = ("BACH_0101", "BACH_0107")
HERO_BIOPSY = "BACH_0108"
INSUFFICIENT = "BACH_0103"
TRAIN = {f"BACH_00{i:02d}": CLASSES[i % 4] for i in range(1, 9)}
CAL = {f"BACH_00{i:02d}": CLASSES[i % 4] for i in range(11, 15)}
WIDTH, HEIGHT, DISPLAY = 2048, 1536, (64, 48)

# Calibrated probabilities and LAC sets per test case (Normal, Benign, InSitu, Invasive).
PRED = {
    "BACH_0101": ([0.80, 0.15, 0.03, 0.02], ["Normal"]),
    "BACH_0102": ([0.70, 0.25, 0.03, 0.02], ["Normal"]),
    "BACH_0103": ([0.10, 0.60, 0.25, 0.05], ["Benign", "InSitu"]),
    "BACH_0104": ([0.15, 0.75, 0.07, 0.03], ["Benign"]),
    "BACH_0105": ([0.02, 0.08, 0.70, 0.20], ["InSitu"]),
    "BACH_0106": ([0.02, 0.05, 0.48, 0.45], ["InSitu", "Invasive"]),
    "BACH_0107": ([0.01, 0.02, 0.17, 0.80], ["Invasive"]),
    "BACH_0108": ([0.02, 0.05, 0.21, 0.72], ["InSitu", "Invasive"]),
}


def _color(i: int) -> tuple[int, int, int]:
    return (40 + (37 * i) % 200, 60 + (53 * i) % 180, 90 + (29 * i) % 160)


def _jpg(path: Path, i: int) -> str:
    io.save_image(path, Image.new("RGB", DISPLAY, _color(i)), quality=90)
    return path.as_posix()


def write_concept_bank(path: Path, cfg: dict[str, Any]) -> Path:
    group_of = {cid: g for g, spec in cfg["concepts"]["groups"].items() for cid in spec["ids"]}
    data = {
        "version": "test",
        "templates": ["an H&E image of {}.", "a photomicrograph showing {}."],
        "concepts": [{"id": cid, "group": group_of[cid], "label": CONCEPT_LABELS[cid][0],
                      "direction": CONCEPT_LABELS[cid][1], "weak": cid == "mitoses",
                      "positives": [f"{cid} phrase {i}" for i in range(3)],
                      "contrasts": [f"{cid} contrast {i}" for i in range(2)]} for cid in cfg["concepts"]["ids"]],
        "class_prompts": {c: [f"{c} breast tissue"] for c in cfg["classes"]},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def write_cell_features_yaml(path: Path) -> Path:
    data = {"features": [{"id": f, "label": lab, "unit": unit, "direction": d} for f, (lab, unit, d) in CELL_FEATURES.items()]}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def make_ctx(cfg: dict[str, Any], root: Path, art: Path, *, limit: int | None = None, force: bool = False,
             step: str = "08_build_case_json") -> cli.StepContext:
    return cli.StepContext(step=step, cfg=cfg, data_root=root, artifacts_root=art, limit=limit, force=force,
                           synthetic=cli.is_synthetic_root(root), args=None)  # type: ignore[arg-type]


def write_downstream(root: Path, cfg: dict[str, Any], *, cells: bool = True, composites: bool = True) -> dict[str, Any]:
    rng = np.random.default_rng(44)
    root.mkdir(parents=True, exist_ok=True)
    io.write_json(root / cli.SYNTHETIC_MARKER, {"synthetic": True, "writer": "tests/downstream_fixture.py"})
    concept_ids = list(cfg["concepts"]["ids"])
    k = int(cfg["concepts"]["top_k"])
    severity = int(cfg["degrade"]["display_severity"])

    # ---- cases + displays
    rows = []
    all_cases = [(c, l, "train", "biopsy") for c, l in TRAIN.items()] + \
                [(c, l, "calibration", "biopsy") for c, l in CAL.items()] + \
                [(c, l, "test", lane) for c, (l, lane) in TEST.items()]
    for i, (cid, label, split, lane) in enumerate(all_cases):
        _jpg(schemas.display_jpg(root, cid), i)
        rows.append({"case_id": cid, "source": "synthetic", "source_file": f"Photos/{label}/{cid}.tif", "source_index": i,
                     "image_path": f"/synthetic/{cid}.tif", "label": label, "split": split, "patient_id": "",
                     "group": f"b{i // 2}", "lane": lane, "width": WIDTH, "height": HEIGHT,
                     "display_path": f"display/{cid}.jpg"})
    cases = pd.DataFrame(rows)
    io.write_csv(schemas.cases_csv(root), cases)

    variants = [schemas.variant_case_id(c, t, severity) for c in TEST_SURGICAL for t in cfg["degrade"]["types"]]
    for j, vid in enumerate(variants):
        _jpg(schemas.display_jpg(root, vid), 100 + j)
    comp_ids: list[str] = []
    comp_rows = []
    if composites:
        for kk, (carc, benign, side) in enumerate([("BACH_0107", "BACH_0101", "top"), ("BACH_0108", "BACH_0104", "left")]):
            cid = schemas.composite_case_id(carc, kk)
            comp_ids.append(cid)
            _jpg(schemas.display_jpg(root, cid), 200 + kk)
            truth = {"top": {"superior": "tumor_side", "inferior": "exact_negative", "medial": "mixed", "lateral": "mixed"},
                     "left": {"superior": "mixed", "inferior": "mixed", "medial": "tumor_side", "lateral": "exact_negative"}}[side]
            comp_rows.append({"case_id": cid, "carcinoma_case_id": carc, "benign_case_id": benign, "orientation": kk,
                              "carcinoma_side": side, "width": WIDTH, "height": HEIGHT, "path": f"composites/{cid}.png",
                              "display_path": f"display/{cid}.jpg", **{f"truth_{m}": truth[m] for m in schemas.MARGINS}})
        io.write_csv(schemas.composites_csv(root), pd.DataFrame(comp_rows))

    # ---- tiles (g512) with QC
    tile_rows = []
    qc: dict[str, list[int]] = {}
    for cid in cases["case_id"]:
        keep = list(range(12)) if cid != INSUFFICIENT else [5, 6]
        if cid == "BACH_0102":
            keep = [t for t in range(12) if t not in (0, 11)]
        qc[cid] = keep
        for t in range(12):
            r, c = divmod(t, 4)
            tile_rows.append({"case_id": cid, "grid": "g512", "tile_idx": t, "row": r, "col": c, "x": c * 512,
                              "y": r * 512, "size": 512, "tissue_frac": 0.9 if t in keep else 0.1,
                              "qc_pass": int(t in keep), "path": f"tiles/g512/{cid}/{r}_{c}.png"})
    io.write_csv(schemas.tiles_csv(root), pd.DataFrame(tile_rows))
    for vid in variants:
        qc[vid] = qc[vid.split("_FSIM_")[0]]
    for cid in comp_ids:
        qc[cid] = list(range(12))

    test_ids = list(TEST)
    scored = test_ids + variants + comp_ids          # cases with predictions-side outputs

    def probs_for(cid: str) -> tuple[list[float], list[str]]:
        if cid in PRED:
            return PRED[cid]
        if "_FSIM_" in cid:
            base, members = PRED[cid.split("_FSIM_")[0]]
            p = np.asarray(base) * 0.6 + 0.1
            return [float(x) for x in p / p.sum()], sorted(set(members) | {"Benign"}, key=CLASSES.index)
        return [0.05, 0.10, 0.25, 0.60], ["InSitu", "Invasive"]

    # ---- predictions
    pred_rows = []
    for cid in list(CAL) + scored:
        if cid in CAL:
            p = [0.1, 0.1, 0.1, 0.1]
            p[CLASSES.index(CAL[cid])] = 0.7
            members = [CAL[cid]]
        else:
            p, members = probs_for(cid)
        split = "calibration" if cid in CAL else "test"
        row = {"case_id": cid, "split": split, "model": "cb_text", "temperature": 1.3,
               "set_lac": schemas.format_set(members, CLASSES), "qhat_lac": 0.62, "set_size": len(members),
               "risk": p[3], "top": CLASSES[int(np.argmax(p))], "agree_top": int(cid != "BACH_0106")}
        for c, v in zip(CLASSES, p):
            row[schemas.cal_prob_col(c)] = v
            row[schemas.prob_col("cb_text", c)] = v
        pred_rows.append(row)
    io.write_csv(schemas.predictions_csv(root), pd.DataFrame(pred_rows))

    # ---- reliability
    rel_rows = []
    for cid in list(CAL) + scored:
        degraded = "_FSIM_" in cid
        flag = degraded
        p_error = 0.6 if degraded else 0.15
        rel_rows.append({"case_id": cid, "source_case_id": cid.split("_FSIM_")[0] if degraded else cid,
                         "variant": cid.split("_FSIM_")[1] if degraded else "clean",
                         "split": "calibration" if cid in CAL else "test", "ood_distance": 0.3 + 0.2 * degraded,
                         "ood_percentile": 0.95 if degraded else 0.4, "set_size": len(probs_for(cid)[1]) if cid not in CAL else 1,
                         "conch_musk_disagree": int(degraded), "p_error": p_error, "reliability": 1 - p_error,
                         "reliability_flag": int(flag), "threshold": 0.5})
    io.write_csv(schemas.reliability_csv(root), pd.DataFrame(rel_rows))

    # ---- concepts_tile.npz (every case, QC tiles)
    np_case, np_tile, z_rows = [], [], []
    for cid in list(cases["case_id"]) + variants + comp_ids:
        for t in qc[cid]:
            z = rng.normal(0.0, 0.5, len(concept_ids))
            if cid in (HERO_BIOPSY, "BACH_0107") and t in (5, 6):
                z[concept_ids.index("invasion")] = 2.0 + 0.1 * t
                z[concept_ids.index("pleomorphism")] = 1.4
            np_case.append(cid)
            np_tile.append(t)
            z_rows.append(z)
    z = np.asarray(z_rows, dtype=np.float32)
    z_cav = np.full_like(z, np.nan)
    z_cav[:, concept_ids.index("invasion")] = z[:, concept_ids.index("invasion")]
    io.save_npz(schemas.concepts_tile_npz(root), case_id=np.asarray(np_case), tile_idx=np.asarray(np_tile, dtype=np.int32),
                grid=np.asarray("g512"), concept_ids=np.asarray(concept_ids), z=z, z_text=z, z_cav=z_cav,
                spread=np.abs(z) * 0.05 + 0.02, encoder=np.asarray("conch"), synthetic=np.asarray(True))
    z_by = {}
    for cid, t, zz in zip(np_case, np_tile, z_rows):
        z_by.setdefault(cid, []).append(zz)

    # ---- case_features.csv (every case + variants + composites)
    cf_rows = []
    for cid in list(cases["case_id"]) + variants + comp_ids:
        zz = np.asarray(z_by[cid])
        row = {"case_id": cid, "split": str(cases.set_index("case_id")["split"].get(cid, "test"))}
        for ci, c in enumerate(concept_ids):
            top = np.sort(zz[:, ci])[::-1][:k]
            row[f"{c}__top{k}"] = float(top.mean())
            row[f"{c}__frac"] = float((zz[:, ci] > cfg["concepts"]["z_threshold"]).mean())
            row[f"{c}__spread"] = 0.05
            row[f"{c}__text_top{k}"] = float(top.mean())
            row[f"{c}__cav_top{k}"] = float(top.mean()) if c == "invasion" else np.nan
        cf_rows.append(row)
    case_features = pd.DataFrame(cf_rows)
    io.write_csv(schemas.case_features_csv(root), case_features)

    # ---- contributions.csv
    weights = {"invasion": 0.25, "pleomorphism": 0.15, "mitoses": 0.02, "necrosis": 0.03, "lymphocytes": 0.0,
               "duct_confined": -0.08, "tubules": -0.05, "normal_lobules": -0.07, "benign_stroma_fat": -0.04}
    cf = case_features.set_index("case_id")
    con_rows = []
    for cid in scored:
        for c in concept_ids:
            for feat in (f"{c}__top{k}", f"{c}__frac"):
                value = float(cf.loc[cid, feat])
                con_rows.append({"case_id": cid, "split": "test", "model": "cb_text", "feature": feat, "concept": c,
                                 "source": "cav" if c == "invasion" else "text", "value": value, "value_std": value,
                                 "contribution": weights[c] * value})
        if cells and cid in test_ids:
            for f in ("nuc_area_cv", "neo_density"):
                con_rows.append({"case_id": cid, "split": "test", "model": "cb_text_cells", "feature": f, "concept": f,
                                 "source": "cell", "value": 0.4, "value_std": 1.2 if cid == HERO_BIOPSY else 0.1,
                                 "contribution": 0.21 if cid == HERO_BIOPSY else 0.01})
    io.write_csv(schemas.contributions_csv(root), pd.DataFrame(con_rows))

    # ---- tile_risk.csv
    tr_rows = []
    for cid in scored:
        base = probs_for(cid)[0][3]
        for t in qc[cid]:
            risk = min(1.0, base + (0.15 if t in (5, 6) else -0.1 * base))
            tr_rows.append({"case_id": cid, "split": "test", "grid": "g512", "tile_idx": t, "risk": round(risk, 4),
                            "probe": "tile_probe", "fold": -1})
    io.write_csv(schemas.tile_risk_csv(root), pd.DataFrame(tr_rows))

    # ---- retrieval.json (clean test cases)
    train_ids = list(TRAIN)
    rcases = {}
    for i, cid in enumerate(test_ids):
        p, _ = PRED[cid]
        predicted = CLASSES[int(np.argmax(p))]
        nbrs = [{"case_id": train_ids[(i + j) % len(train_ids)], "label": TRAIN[train_ids[(i + j) % len(train_ids)]],
                 "similarity": round(0.9 - 0.02 * j, 3)} for j in range(5)]
        agree = sum(n["label"] == predicted for n in nbrs)
        other = next(n for n in nbrs if n["label"] != predicted)
        rcases[cid] = {"neighbors": nbrs, "predicted": predicted, "agree": agree, "of": 5,
                       "neighbor_agreement": agree / 5,
                       "contrast": {**other, "differs_most_in": ["invasion", "duct_confined"]}}
    io.write_json(schemas.retrieval_json(root), {"encoder": "conch", "head": "align", "k": 5, "cases": rcases,
                                                 "created_utc": io.utc_now(), "config_hash": "test", "synthetic": True})

    # ---- margins.json (surgical + variants + composites)
    geometry = margin.strip_segments(WIDTH, HEIGHT, int(cfg["margin"]["segment_px"]), cfg["margin"]["names"])
    mcases = {}
    for cid in list(TEST_SURGICAL) + variants + comp_ids:
        hot = cid.startswith("BACH_0107") or cid in comp_ids
        segs = []
        for s in geometry:
            p = 0.05
            if hot and s["margin"] == "superior" and s["x"] in (768, 1024):
                p = 0.81 if s["x"] == 768 else 0.64
            if cid in comp_ids[1:] and s["margin"] == "medial" and s["y"] == 512:
                p = 0.77
            segs.append({**s, "p": p})
        mcases[cid] = {"width": WIDTH, "height": HEIGHT, "segments": segs,
                       "per_margin": margin.per_margin_scores(geometry, [s["p"] for s in segs])}
    io.write_json(schemas.margins_json(root), {
        "probe": "fine_probe", "strip_px": 256, "segment_px": 256, "max_fpr": 0.05, "threshold": 0.5,
        "threshold_ci": [0.4, 0.6], "n_benign_cal_images": 2, "cases": mcases, "created_utc": io.utc_now(),
        "config_hash": "test", "synthetic": True})

    # ---- cells
    if cells:
        feat_rows, tile_rows_c = [], []
        for i, cid in enumerate(test_ids):
            if cid == INSUFFICIENT:
                continue
            row = {"case_id": cid, "split": "test", "n_tiles": 2, "n_nuclei": 300, "n_neoplastic": 150,
                   "insufficient_tumor_nuclei": 0}
            for f in CELL_FEATURES:
                row[f] = {"nuc_area_cv": 0.48, "neo_density": 3150.0, "til_proximity": 0.22}[f] if cid == HERO_BIOPSY else 0.3 + 0.01 * i
                row[f"{f}__missing"] = 0
                row[f"{f}__ref_pct"] = {"nuc_area_cv": 0.95, "neo_density": 0.91, "til_proximity": 0.4}[f] if cid == HERO_BIOPSY else 0.5
            feat_rows.append(row)
            for t in (5, 6):
                overlay = schemas.cell_overlay_png(root, cid, t)
                io.save_image(overlay, Image.new("RGB", (32, 32), (176, 51, 92)))
                tile_rows_c.append({"case_id": cid, "tile_idx": t, "n_nuclei": 150,
                                    **{f"n_{ct}": n for ct, n in zip(schemas.CELL_TYPES, (100, 20, 15, 5, 10))},
                                    "overlay": overlay.relative_to(root).as_posix(),
                                    **{f: 0.3 for f in CELL_FEATURES}})
        io.write_csv(schemas.cell_features_csv(root), pd.DataFrame(feat_rows))
        io.write_csv(schemas.cell_features_csv(root, per_tile=True), pd.DataFrame(tile_rows_c))
    return {"test": test_ids, "variants": variants, "composites": comp_ids, "train": train_ids, "cal": list(CAL)}


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys

    from core import config

    out = write_downstream(Path(sys.argv[1]), config.load_settings())
    print(json.dumps(out, indent=1))


# ---- eval fixture: a larger synthetic root with planted signal (eval suite tests) -------------------------------------
def write_eval_root(root: Path, cfg: dict[str, Any], *, n_test_per_class: int = 12, n_cal_per_class: int = 10,
                    seed: int = 7) -> dict[str, Any]:
    """cases.csv, tiles.csv, predictions.csv (Model A conch_probe_lr, Model B / production cb_text, LAC/APS/Mondrian
    sets fit on calibration with core.conformal), reliability.csv, composites.csv, wsi_windows.csv and margins.json
    for every test case, composite and WSI window. Returns ids by role."""
    from core import conformal

    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    io.write_json(root / cli.SYNTHETIC_MARKER, {"synthetic": True, "writer": "tests/downstream_fixture.py:eval"})
    classes = list(cfg["classes"])
    rows, n = [], 0
    for split, per in (("train", 2), ("calibration", n_cal_per_class), ("test", n_test_per_class)):
        for ci_, cls in enumerate(classes):
            for j in range(per):
                cid = f"EV_{n:04d}"
                lane = "surgical" if split == "test" and j % 4 == 0 else "biopsy"
                rows.append({"case_id": cid, "source": "synthetic", "source_file": f"Photos/{cls}/{cid}.tif",
                             "source_index": n, "image_path": f"/synthetic/{cid}.tif", "label": cls, "split": split,
                             "patient_id": "", "group": f"b{n // 3}", "lane": lane, "width": WIDTH, "height": HEIGHT,
                             "display_path": f"display/{cid}.jpg"})
                n += 1
    cases = pd.DataFrame(rows)
    io.write_csv(schemas.cases_csv(root), cases)
    y = cases["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    onehot = np.eye(len(classes))[y]
    neighbour = np.roll(np.eye(len(classes)), 1, axis=1)[y]          # confusable class (e.g. InSitu <-> Invasive)
    logits_a = 2.6 * onehot + 0.9 * neighbour + rng.normal(0, 0.9, onehot.shape)
    logits_b = 2.0 * onehot + 1.0 * neighbour + rng.normal(0, 1.0, onehot.shape)
    from core import calibration

    raw_logits = logits_b * 3.0                                      # B is overconfident before temperature
    p_a, p_b_raw = softmax(logits_a), softmax(raw_logits)
    cal = (cases["split"] == "calibration").to_numpy()
    temperature = calibration.fit_temperature(raw_logits[cal], y[cal], cfg["calibration"]["temperature_bounds"])
    p_cal = softmax(raw_logits / temperature)
    alpha = float(cfg["alpha"])
    qhat = conformal.fit_lac(p_cal[cal], y[cal], alpha)
    lac = conformal.lac_sets(p_cal, qhat)
    q_aps = conformal.fit_aps(p_cal[cal], y[cal], alpha, 44, True)
    aps = conformal.aps_sets(p_cal, q_aps, 45, True)
    mond = conformal.mondrian_sets(p_cal, conformal.mondrian_quantiles(p_cal[cal], y[cal], alpha))
    k = classes.index(cfg["aggressive_class"])
    pred_rows = []
    for i, row in cases.iterrows():
        if row["split"] == "train":
            continue
        r = {"case_id": row["case_id"], "split": row["split"], "model": "cb_text", "temperature": temperature,
             "set_lac": schemas.format_set(conformal.set_labels(lac[i], classes), classes), "qhat_lac": qhat,
             "set_size": int(lac[i].sum()), "risk": float(p_cal[i, k]), "top": classes[int(p_cal[i].argmax())],
             "set_aps": schemas.format_set(conformal.set_labels(aps[i], classes), classes),
             "set_mondrian": schemas.format_set(conformal.set_labels(mond[i], classes), classes),
             "agree_top": int(p_a[i].argmax() == p_cal[i].argmax())}
        for c_i, c in enumerate(classes):
            r[schemas.cal_prob_col(c)] = float(p_cal[i, c_i])
            r[schemas.prob_col("cb_text", c)] = float(p_b_raw[i, c_i])
            r[schemas.prob_col("conch_probe_lr", c)] = float(p_a[i, c_i])
        pred_rows.append(r)
    # one degraded variant row (must never enter headline metrics)
    variant = dict(pred_rows[-1], case_id=schemas.variant_case_id(pred_rows[-1]["case_id"], "blur", 3))
    pred_rows.append(variant)
    io.write_csv(schemas.predictions_csv(root), pd.DataFrame(pred_rows))

    test = cases[cases["split"] == "test"].reset_index(drop=True)
    tile_rows = []
    for _, row in test.iterrows():
        keep = 2 if row["case_id"] == test["case_id"].iloc[1] else 12
        for t in range(12):
            r_, c_ = divmod(t, 4)
            tile_rows.append({"case_id": row["case_id"], "grid": "g512", "tile_idx": t, "row": r_, "col": c_,
                              "x": c_ * 512, "y": r_ * 512, "size": 512, "tissue_frac": 0.9 if t < keep else 0.1,
                              "qc_pass": int(t < keep), "path": f"tiles/g512/{row['case_id']}/{r_}_{c_}.png"})
    io.write_csv(schemas.tiles_csv(root), pd.DataFrame(tile_rows))
    rel_rows = []
    for i, row in test.iterrows():
        flag = int(i % 10 == 3)
        rel_rows.append({"case_id": row["case_id"], "source_case_id": row["case_id"], "variant": "clean",
                         "split": "test", "ood_distance": 0.3, "ood_percentile": 0.5, "set_size": 1,
                         "conch_musk_disagree": 0, "p_error": 0.6 if flag else 0.1,
                         "reliability": 0.4 if flag else 0.9, "reliability_flag": flag, "threshold": 0.5})
    io.write_csv(schemas.reliability_csv(root), pd.DataFrame(rel_rows))

    geometry = margin.strip_segments(WIDTH, HEIGHT, int(cfg["margin"]["segment_px"]), cfg["margin"]["names"])
    carcinoma = set(cfg["carcinoma_classes"])

    def segments_for(hot_margins: Sequence[str], hot_p: float = 0.85) -> list[dict[str, Any]]:
        out = []
        for s in geometry:
            p = float(rng.uniform(0.01, 0.3))
            if s["margin"] in hot_margins and rng.uniform() < 0.5:
                p = hot_p
            out.append({**s, "p": round(p, 4)})
        for m in hot_margins:                                   # at least one hot segment per hot margin
            first = next(j for j, s in enumerate(out) if s["margin"] == m)
            out[first]["p"] = hot_p
        return out

    mcases: dict[str, Any] = {}
    names = list(schemas.MARGINS)
    for i, row in test.iterrows():
        if row["label"] in carcinoma:
            hot = [] if i % 5 == 0 else [names[i % 4]]
        else:
            hot = [names[0]] if i % 11 == 0 else []             # occasional benign false positive
        segs = segments_for(hot)
        mcases[row["case_id"]] = {"width": WIDTH, "height": HEIGHT, "segments": segs,
                                  "per_margin": margin.per_margin_scores(geometry, [s["p"] for s in segs])}
    carc_ids = test.loc[test["label"].isin(carcinoma), "case_id"].tolist()
    ben_ids = test.loc[~test["label"].isin(carcinoma), "case_id"].tolist()
    sides = {"top": "superior", "bottom": "inferior", "left": "medial", "right": "lateral"}
    opposite = {"superior": "inferior", "inferior": "superior", "medial": "lateral", "lateral": "medial"}
    comp_rows = []
    for kk in range(6):
        side = list(sides)[kk % 4]
        cid = schemas.composite_case_id(carc_ids[kk], kk)
        tumor = sides[side]
        truth = {m: "mixed" for m in names}
        truth[tumor], truth[opposite[tumor]] = "tumor_side", "exact_negative"
        comp_rows.append({"case_id": cid, "carcinoma_case_id": carc_ids[kk], "benign_case_id": ben_ids[kk],
                          "orientation": kk % 4, "carcinoma_side": side, "width": WIDTH, "height": HEIGHT,
                          "path": f"composites/{cid}.png", "display_path": f"display/{cid}.jpg",
                          **{f"truth_{m}": truth[m] for m in names}})
        hot = [tumor] + ([opposite[tumor]] if kk == 5 else [])   # one echo-like composite
        segs = segments_for(hot)
        mcases[cid] = {"width": WIDTH, "height": HEIGHT, "segments": segs,
                       "per_margin": margin.per_margin_scores(geometry, [s["p"] for s in segs])}
    io.write_csv(schemas.composites_csv(root), pd.DataFrame(comp_rows))
    win_rows = []
    for w in range(8):
        wid = f"WSI_A{w // 2:02d}_{w:02d}"
        tumor_edges = [names[w % 4]] if w % 3 != 2 else []
        win_rows.append({"window_id": wid, "wsi_id": f"A{w // 2:02d}", "patient_id": f"P{w // 2}", "fold": w % 2,
                         "x": 0, "y": 0, "width": WIDTH, "height": HEIGHT,
                         "stratum": "edge_tumor" if tumor_edges else "none", "path": f"wsi_windows/{wid}.png",
                         **{f"truth_{m}": ("tumor" if m in tumor_edges else "clear") for m in names}})
        segs = segments_for(tumor_edges if w != 7 else [])
        mcases[wid] = {"width": WIDTH, "height": HEIGHT, "segments": segs,
                       "per_margin": margin.per_margin_scores(geometry, [s["p"] for s in segs])}
    io.write_csv(schemas.wsi_windows_csv(root), pd.DataFrame(win_rows))
    io.write_json(schemas.margins_json(root), {
        "probe": "fine_probe", "strip_px": 256, "segment_px": 256, "max_fpr": 0.05, "threshold": 0.5,
        "threshold_ci": [0.4, 0.6], "n_benign_cal_images": 12, "cases": mcases, "created_utc": io.utc_now(),
        "config_hash": "test", "synthetic": True})
    return {"test": test["case_id"].tolist(), "composites": [r["case_id"] for r in comp_rows],
            "windows": [r["window_id"] for r in win_rows], "variant": variant["case_id"]}
