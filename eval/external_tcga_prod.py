"""External test of the production model on TCGA-BRCA crops (pre-registration: docs/EXTERNAL_TCGA_R2.md).

    python -m eval.external_tcga_prod --stage parity    # BACH parity gate -> <data>/tcga_brca_prod/parity.json
    python -m eval.external_tcga_prod --stage score     # both cohorts -> <artifacts>/eval/external_tcga.json (served)

The recipe is the production model's BACH calibration/test read, checked at run time against the stored artifacts:

  * model: `production` in eval/model_choice.json (uni2h512_lr), data/models/<model>.npz (riskmodels.LinearModel)
    on the mean of QC-passing UNI2-h `bach512` probe region embeddings;
  * views: the n_views the stored bach512 embeddings were made with (emb/uni2h/meta.json and the index agree: 1,
    view 0), with 02_embed's transforms (crop -> dihedral view -> encoder preprocessing) and float16 storage;
  * tissue QC: the 01q rule (display rendition, common.tissue_mask, region fraction >= qc.min_tissue_frac);
  * temperature: eval/calibration.json; q-hat: model_choice.json conformal[<model>].qhat_aps, re-derived from
    predictions.csv calibration rows; sets: randomized APS seeded conformal.seed + 1, one call over the cohort's
    crop matrix sorted by crop_id, then nonempty (exactly how 05_risk_model builds BACH test sets);
  * tier: core.tiers.assign_tier, biopsy lane, reliability_flag false (CONCH and MUSK are not on this laptop).

Slide-level analyses, bootstrap intervals and the slide table are imported from eval/external_tcga.py.
The served file keeps the top-level layout the app reads (model, primary, slides, crops, n_*, limitations, data) for
the `protocol25` cohort (data/tcga_brca_prod/protocol25: 25 patients, TCGA_script.py --source tiles) and adds the
`same8` cohort (the 132 openslide crops of the first 8 patients, data/tcga_brca_test) under `cohorts`, plus parity,
tile equivalence, BACH reference, latency and the pre-registered sensitivity analyses. Only the production model is
reported. Region embeddings are cached under data/tcga_brca_prod/embeddings/.
"""

from __future__ import annotations

import hashlib
import importlib
import io as _io
import json
import math
import platform
import subprocess
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from core import calibration, cli, conformal, io, schemas, tiers
from eval import external_tcga as X
from eval import metrics as emetrics
from pipeline import common
from pipeline import encoders as E
from pipeline import riskmodels as rm

STEP = "eval_external_tcga_prod"
NAME = "external_tcga"
PROTOCOL = "docs/EXTERNAL_TCGA_R2.md"
PROD_DIR = "tcga_brca_prod"
SAME8_DIR = "tcga_brca_test"
P25_DIR = "tcga_brca_prod/protocol25"
PARITY_CASES = ("BACH_0110", "BACH_0006", "BACH_0004", "BACH_0005", "BACH_0087")   # pre-registered
BACH_ZIP_URL = "https://zenodo.org/api/records/3632035/files/ICIAR2018_BACH_Challenge.zip/content"
BACH_ZIP_ROOT = "ICIAR2018_BACH_Challenge"
PARITY_MIN_COSINE = 0.99
PARITY_MAX_PROB_DIFF = 0.02
WARMUP = 2
EMBED_BATCH = 96
CROP_SIZE = (2048, 1536)
QC01Q = importlib.import_module("pipeline.01q_requalify_qc")


# ---- the production recipe ------------------------------------------------------------------------------------------
def stored_views(data_root: Path, art_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Model, feature source and the views its stored BACH embeddings were made with (meta.json == index)."""
    model = io.read_json(schemas.eval_json(art_root, "model_choice"))["production"]
    spec = cfg["risk_model"]["embedding_candidates"].get(model)
    if spec is None:
        raise SystemExit(f"production model {model!r} is not a mean-pooled embedding candidate; this test does not apply")
    enc, set_name, head = spec["encoder"], spec["set"], spec["head"]
    meta = io.read_json(Path(data_root) / "emb" / enc / "meta.json")
    n_meta = int(meta["sets"][set_name]["n_views"])
    index = rm.read_csv(schemas.emb_index_csv(data_root, enc, set_name, head))
    n_index = sorted({int(v) for v in index["n_views"]})
    if n_index != [n_meta]:
        raise SystemExit(f"{enc}/{set_name}: meta.json n_views {n_meta} != index n_views {n_index}")
    grids = sorted({str(g) for g in index["grid"]})
    if len(grids) != 1:
        raise SystemExit(f"{enc}/{set_name}: expected one grid, found {grids}")
    return {"model": model, "encoder": enc, "set": set_name, "head": head, "grid": grids[0],
            "n_views": n_meta, "views": list(range(n_meta))}


def view_images(tile: Image.Image, n_views: int) -> list[Image.Image]:
    """Views 0..n_views-1 of one region (02_embed keeps views < n with --max-views; view 0 is the identity)."""
    return [E.dihedral(tile, v) for v in range(int(n_views))]


def region_tensors(img: Image.Image, boxes: Sequence[Mapping[str, Any]], n_views: int, preprocess: Any) -> Any:
    """Preprocessed tensors, region-major then view (02_embed's (item_id, view) order): crop -> view -> preprocess."""
    import torch

    out = []
    for b in boxes:
        size = int(b["size"])
        tile = E.crop_box(img, int(b["x"]), int(b["y"]), size, size)
        out += [preprocess(v) for v in view_images(tile, n_views)]
    return torch.stack(out)


def embed_regions(enc: E.Encoder, img: Image.Image, boxes: Sequence[Mapping[str, Any]], n_views: int, head: str,
                  *, use_autocast: bool = True, batch: int = EMBED_BATCH) -> np.ndarray:
    """(n_regions, D) float32 holding float16 values: per-view float16 rows, view mean in float32, stored float16
    (02_embed chunks + reduce_views)."""
    x = region_tensors(img, boxes, n_views, enc.preprocess)
    parts = [enc.embed(x[i:i + batch], use_autocast=use_autocast)[head].astype(np.float16)
             for i in range(0, len(x), batch)]
    views = np.concatenate(parts).astype(np.float32).reshape(len(boxes), int(n_views), -1)
    return (views.sum(axis=1) / float(n_views)).astype(np.float16).astype(np.float32)


def qc_regions(img: Image.Image, cfg: Mapping[str, Any], grid: str) -> list[dict[str, Any]]:
    """BACH's QC rule (pipeline/01q): tissue fraction of each region on the display rendition of the image."""
    disp_cfg = cfg["tiles"]["display"]
    buf = _io.BytesIO()
    common.to_display(img, int(disp_cfg["long_side"])).save(buf, format="JPEG", quality=int(disp_cfg["jpeg_quality"]))
    buf.seek(0)
    with Image.open(buf) as handle:
        disp = handle.convert("RGB")
    boxes = common.grid_boxes(cfg, grid)
    fracs = np.round(QC01Q.tile_fracs(common.tissue_mask(disp, cfg), pd.DataFrame(boxes),
                                      disp.width / float(img.width), disp.height / float(img.height)), 4)
    min_frac = float(cfg["qc"]["min_tissue_frac"])
    return [{**b, "tissue_frac": float(f), "qc_pass": bool(f >= min_frac)} for b, f in zip(boxes, fracs)]


@dataclass
class Production:
    views: dict[str, Any]
    linear: rm.LinearModel
    temperature: float
    qhat: float
    classes: list[str]
    model_path: Path

    @property
    def name(self) -> str:
        return str(self.views["model"])


def recompute_qhat(data_root: Path, model: str, cfg: Mapping[str, Any]) -> float:
    """APS q-hat from predictions.csv calibration rows of `model` (clean cases, case_id order as 05 fits it)."""
    cases = rm.read_cases(data_root)
    pred = rm.read_csv(schemas.predictions_csv(data_root))
    label = dict(zip(cases["case_id"], cases["label"]))
    cal = pred[(pred["split"] == "calibration") & pred["case_id"].isin(set(cases["case_id"]))
               & (pred["model"] == model)].sort_values("case_id")
    classes = list(cfg["classes"])
    probs = cal[[schemas.cal_prob_col(c) for c in classes]].to_numpy(dtype=float)
    y = np.array([classes.index(label[c]) for c in cal["case_id"]])
    return float(conformal.fit_aps(probs, y, float(cfg["alpha"]), int(cfg["conformal"]["seed"]),
                                   bool(cfg["conformal"]["aps_randomized"])))


def production_qhat(data_root: Path, art_root: Path, cfg: Mapping[str, Any]) -> float:
    """The app's stored APS q-hat for the production model, after checking it re-derives from predictions.csv."""
    if conformal.tier_set_source(cfg) != "set_aps":
        raise SystemExit(f"settings conformal {cfg['conformal']['method']}/{cfg['conformal']['mode']} is not the "
                         "pre-registered marginal APS")
    mc = io.read_json(schemas.eval_json(art_root, "model_choice"))
    model = mc["production"]
    stored = float(mc["conformal"][model]["qhat_aps"])
    again = recompute_qhat(data_root, model, cfg)
    if abs(again - stored) > 1e-9:
        raise SystemExit(f"q-hat drift: model_choice.json {stored} vs predictions.csv calibration rows {again}")
    return stored


def build_sets(probs: np.ndarray, qhat: float, cfg: Mapping[str, Any], *, randomized: bool | None = None
               ) -> np.ndarray:
    """Tier-driving sets as 05_risk_model builds them: APS on the whole matrix in one call (randomized: one U per
    row from default_rng(conformal.seed + 1)), then the argmax added to empty rows (conformal.nonempty)."""
    probs = np.asarray(probs, dtype=float)
    rand = bool(cfg["conformal"]["aps_randomized"]) if randomized is None else bool(randomized)
    sets = conformal.aps_sets(probs, qhat, int(cfg["conformal"]["seed"]) + 1 if rand else None, rand)
    if bool(cfg["conformal"].get("nonempty", False)):
        sets = conformal.nonempty_sets(sets, probs)
    return sets


def app_row_order(data_root: Path, model: str) -> pd.DataFrame:
    """predictions.csv rows of `model` in the order 05_risk_model builds them before sorting (clean calibration and
    test reads by case_id, variant reads by read id, out-of-fold train rows by case_id): the U order of set_tier."""
    cases = rm.read_cases(data_root)
    pred = rm.read_csv(schemas.predictions_csv(data_root))
    real = set(cases["case_id"])
    p = pred[pred["model"] == model]
    clean = p[p["case_id"].isin(real) & p["split"].isin(["calibration", "test"])].sort_values("case_id")
    variants = p[~p["case_id"].isin(real)].sort_values("case_id")
    train = p[p["case_id"].isin(real) & (p["split"] == "train")].sort_values("case_id")
    return pd.concat([clean, variants, train]).reset_index(drop=True)


def load_production(data_root: Path, art_root: Path, cfg: Mapping[str, Any]) -> Production:
    views = stored_views(data_root, art_root, cfg)
    model = views["model"]
    path = Path(data_root) / cfg["risk_model"]["models_dir"] / f"{model}.npz"
    linear = rm.LinearModel.from_npz(path)
    if list(linear.classes) != list(cfg["classes"]):
        raise SystemExit(f"{path}: classes {linear.classes} != settings {cfg['classes']}")
    cal = io.read_json(schemas.eval_json(art_root, "calibration"))
    if cal.get("model") != model:
        raise SystemExit(f"eval/calibration.json is for {cal.get('model')!r}, production is {model!r}")
    temperature = float(cal["temperature"])
    others = {"model npz": float(io.load_npz(path)["temperature"]),
              "model_choice.json": float(io.read_json(schemas.eval_json(art_root, "model_choice"))["temperatures"][model])}
    bad = {k: v for k, v in others.items() if abs(v - temperature) > 1e-9}
    if bad:
        raise SystemExit(f"temperature mismatch: calibration.json {temperature} vs {bad}")
    return Production(views, linear, temperature, production_qhat(data_root, art_root, cfg), list(cfg["classes"]), path)


def calibrated(prod: Production, feats: np.ndarray) -> np.ndarray:
    return calibration.softmax(prod.linear.logits(np.atleast_2d(feats)), prod.temperature)


def tier_of(probs: np.ndarray, members: list[str], classes: Sequence[str], qc_kept: int,
            cfg: Mapping[str, Any]) -> tuple[int, list[str]]:
    case = {"lane": "biopsy", "arrived_offset_s": 0.0,
            "prediction": {"classes": list(classes), "probs": [float(p) for p in probs], "conformal_set": members},
            "quality": {"reliability_flag": False, "qc_tiles_kept": int(qc_kept)}}
    return tiers.assign_tier(case, cfg, now=None)


def slide_table(crops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Slide rows (eval/external_tcga.slide_rows): invasive score = max crop P(Invasive); called = any crop with top
    class Invasive."""
    return X.slide_rows(crops)


# ---- crops -> features (cached) ----------------------------------------------------------------------------------------
def file_key(crop_id: str, path: Path) -> str:
    return f"{crop_id}__{hashlib.sha1(path.read_bytes()).hexdigest()[:16]}"


class FeatureCache:
    """Region embeddings per crop file (keyed by crop id + content hash) and their measured latency."""

    def __init__(self, path: Path):
        self.path = path
        self.lat_path = path.with_suffix(".latency.json")
        self.x: dict[str, np.ndarray] = {}
        self.latency: dict[str, dict[str, float]] = {}
        if path.is_file():
            self.x = {k: v for k, v in io.load_npz(path).items()}
        if self.lat_path.is_file():
            self.latency = io.read_json(self.lat_path)
        self.dirty = 0

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        io.save_npz(self.path, **{k: v.astype(np.float16) for k, v in self.x.items()})
        io.write_json(self.lat_path, self.latency)
        self.dirty = 0


def featurize(rows: Sequence[Mapping[str, Any]], root: Path, enc: E.Encoder | None, prod: Production,
              cfg: Mapping[str, Any], cache: FeatureCache, n_views: int, *, timed: bool) -> dict[str, dict[str, Any]]:
    """Per crop: QC regions (01q rule and full-resolution), region embeddings (cached), pooled feature."""
    grid, head = prod.views["grid"], prod.views["head"]
    out: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows, 1):
        crop_id = str(row["crop_id"])
        path = root / str(row["path"])
        try:
            key = file_key(crop_id, path)
            t0 = time.perf_counter()
            with Image.open(path) as handle:
                img = handle.convert("RGB")
            if img.size != CROP_SIZE:
                raise ValueError(f"expected {CROP_SIZE}, got {img.size}")
            regions = qc_regions(img, cfg, grid)
            cached = key in cache.x
            if cached:
                x = cache.x[key].astype(np.float32)
                t1 = t2 = time.perf_counter()
            else:
                if enc is None:
                    raise RuntimeError("embedding needed but no encoder loaded")
                t1 = time.perf_counter()
                x = embed_regions(enc, img, regions, n_views, head)
                t2 = time.perf_counter()
            feat, n_qc, fallback = X.pool_qc(x, regions)
            probs = calibrated(prod, feat)[0]
            members = conformal.set_labels(build_sets(probs[None, :], prod.qhat, cfg)[0], prod.classes)
            tier_of(probs, members, prod.classes, n_qc, cfg)
            t3 = time.perf_counter()
            if not cached:
                cache.x[key] = x.astype(np.float16)
                cache.dirty += 1
                if timed:
                    cache.latency[key] = {"total_s": t3 - t0, "embed_s": t2 - t1, "order": len(cache.latency)}
                if cache.dirty >= 25:
                    cache.save()
            full = X.tile_case(img, cfg)
            out[crop_id] = {"key": key, "feat": feat, "n_qc": int(n_qc), "qc_fallback": bool(fallback),
                            "qc_fullres_disagree": int(sum(a["qc_pass"] != b["qc_pass"] for a, b in zip(regions, full)))}
            print(f"  [{i}/{len(rows)}] {crop_id}{' (cached)' if cached else f' {t3 - t0:.2f}s'}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failed crop is reported, never silently dropped
            print(f"  FAILED {crop_id}: {exc}", file=sys.stderr, flush=True)
            out[crop_id] = {"error": f"{type(exc).__name__}: {exc}"}
    cache.save()
    return out


# ---- scoring and analyses ---------------------------------------------------------------------------------------------
def score_crops(rows: Sequence[Mapping[str, Any]], feats: Mapping[str, Mapping[str, Any]], prod: Production,
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Calibrated probabilities, cohort-matrix conformal sets (randomized as production; deterministic for S3) and
    tiers for every crop that has features. Rows must already be sorted by crop_id."""
    ok = [r for r in rows if "feat" in feats.get(str(r["crop_id"]), {})]
    if not ok:
        return []
    probs = calibrated(prod, np.stack([feats[str(r["crop_id"])]["feat"] for r in ok]))
    sets = build_sets(probs, prod.qhat, cfg)
    det = build_sets(probs, prod.qhat, cfg, randomized=False)
    inv = prod.classes.index(cfg["aggressive_class"])
    out = []
    for i, row in enumerate(ok):
        f = feats[str(row["crop_id"])]
        members = conformal.set_labels(sets[i], prod.classes)
        members_det = conformal.set_labels(det[i], prod.classes)
        tier, reasons = tier_of(probs[i], members, prod.classes, f["n_qc"], cfg)
        tier_det, _ = tier_of(probs[i], members_det, prod.classes, f["n_qc"], cfg)
        out.append({
            "crop_id": str(row["crop_id"]), "path": str(row["path"]), "label": row["label"], "group": row["group"],
            "prep": row["prep"], "patient_id": row["patient_id"], "slide_file_name": row["slide_file_name"],
            "pixel_source": str(row.get("source") or "openslide"),
            "qc_fallback": f["qc_fallback"], "qc_fullres_disagree": f["qc_fullres_disagree"],
            "mpp_estimated": str(row.get("mpp_estimated", "")).lower() in ("1", "true"),
            "probs": {c: float(probs[i, j]) for j, c in enumerate(prod.classes)},
            "p_invasive": float(probs[i, inv]), "top": prod.classes[int(probs[i].argmax())],
            "conformal_set": members, "set_size": len(members), "tier": int(tier), "tier_reasons": reasons,
            "conformal_set_deterministic": members_det, "tier_deterministic_sets": int(tier_det),
            "qc_tiles_kept": f["n_qc"],
        })
    return out


def rate_block(slides: Sequence[Mapping[str, Any]], key: str, boot: Mapping[str, Any]) -> dict[str, Any]:
    if not slides:
        return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_groups": 0}
    return X.ci_rate(np.array([1.0 if s[key] else 0.0 for s in slides]),
                     np.array([s["patient_id"] for s in slides]), boot)


def analyze_cohort(crops: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], bach_flag: float | None
                   ) -> dict[str, Any]:
    """Primary 1-3, secondary 6-7 (eval/external_tcga.analyze), secondary 4-5 as not computable, S1 and S3."""
    boot = cfg["bootstrap"]
    slides = slide_table(crops)
    primary = X.analyze(slides, crops, boot, bach_flag)
    six_keys = {(s["patient_id"], s["group"]) for s in slides if s["n_crops"] == 6}
    six = [s for s in slides if (s["patient_id"], s["group"]) in six_keys]
    s1 = X.analyze(six, [c for c in crops if (c["patient_id"], c["group"]) in six_keys], boot, bach_flag)
    s1.pop("crop_p_invasive_hist", None)
    det = slide_table([{**c, "tier": c["tier_deterministic_sets"]} for c in crops])
    tumor_det = [s for s in det if s["group"] in ("ffpe_tumor", "frozen_tumor")]
    return {
        "primary": primary,
        "secondary_4_reliability_flag_rate": {
            "computable": False,
            "reason": "needs the CONCH OOD percentile and the CONCH-MUSK disagreement; CONCH and MUSK are not on "
                      "this laptop, so reliability_flag is forced false",
            "bach_test_clean_flag_rate": bach_flag, "bach_source": "eval/reliability.json flag_rate_test.clean"},
        "secondary_5_ood_percentile": {"computable": False, "reason": "needs CONCH embeddings (not on this laptop)"},
        "sensitivity_s1_six_crop_slides": {"n_slides": len(six), **s1},
        "sensitivity_s3_deterministic_sets": {
            "rule": "deterministic APS sets (U = 1), nonempty; tier 1-2 share of tumor slides",
            "tumor_tier_1_2_rate": rate_block(tumor_det, "any_tier_1_2", boot)},
        "slides": slides,
    }


def cohort_payload(name: str, root: Path, rows: Sequence[Mapping[str, Any]], feats: Mapping[str, Mapping[str, Any]],
                   crops: list[dict[str, Any]], analysis: Mapping[str, Any]) -> dict[str, Any]:
    slides = analysis["slides"]
    selection = io.read_json(root / "selection.json") if (root / "selection.json").is_file() else []
    with_crops = {s["slide_file_name"] for s in crops}
    run_info = io.read_json(root / "run_info.json") if (root / "run_info.json").is_file() else {}
    failed = [{"crop_id": k, "error": v["error"]} for k, v in feats.items() if "error" in v]
    return {
        "name": name, "root": str(root), "n_crops": len(crops), "n_crops_failed": len(failed),
        "n_slides": len(slides), "n_patients": len({s["patient_id"] for s in slides}),
        "groups": dict(sorted(Counter(s["group"] for s in slides).items())),
        "n_slides_selected": len(selection),
        "slides_without_crops": [{"file_name": s["file_name"], "group": s["group"], "patient_id": s["patient"]}
                                 for s in selection if s["file_name"] not in with_crops],
        "slides_short_crops": [{k: s[k] for k in ("patient_id", "group", "n_crops")} for s in slides
                               if s["n_crops"] < 6],
        "failed_crops": failed,
        "pixel_sources": dict(Counter(c["pixel_source"] for c in crops)),
        "qc_fullres_disagree_regions": int(sum(c["qc_fullres_disagree"] for c in crops)),
        "qc_fallback_crops": [c["crop_id"] for c in crops if c["qc_fallback"]],
        **{k: v for k, v in analysis.items() if k != "slides"},
        "slides": slides, "crops": crops,
        "run_info": {k: run_info[k] for k in ("created_utc", "args", "slides_selected", "crops_written", "groups",
                                               "pixel_source", "tile_equivalence", "network_test") if k in run_info},
    }


def tta8_block(crops: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], n_views: int,
               bach_flag: float | None) -> dict[str, Any]:
    slides = slide_table(crops)
    primary = X.analyze(slides, crops, cfg["bootstrap"], bach_flag)
    primary.pop("crop_p_invasive_hist", None)
    return {"n_views": n_views, "n_crops": len(crops), "primary": primary,
            "slides": [{k: s[k] for k in ("patient_id", "group", "n_crops", "slide_invasive_score", "called_invasive",
                                          "any_tier_1_2")} for s in slides]}


# ---- BACH reference ------------------------------------------------------------------------------------------------------
def bach_reference(data_root: Path, art_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """BACH test (single images): the slide-comparable statistics, from eval files where they exist."""
    boot = cfg["bootstrap"]
    df, _ = emetrics.load_test(cfg, data_root)
    classes = list(cfg["classes"])
    agg = cfg["aggressive_class"]
    probs = df[[schemas.cal_prob_col(c) for c in classes]].to_numpy(dtype=float)
    top = np.array(classes)[probs.argmax(axis=1)]
    p_inv = probs[:, classes.index(agg)]
    label = df["label"].to_numpy()
    group = df["group"].astype(str).to_numpy()
    in_set = np.array([agg in schemas.parse_set(v) for v in df["set_tier"]])
    inv, norm = label == agg, label == "Normal"
    metrics = io.read_json(schemas.eval_json(art_root, "metrics"))
    rel = io.read_json(schemas.eval_json(art_root, "reliability"))
    both = inv | norm
    return {
        "split": "test", "unit": "single BACH image (2048 x 1536 field), not a slide",
        "n_images": int(len(df)), "n_groups": int(len(set(group))), "model": metrics["models"]["production"],
        "invasive_sensitivity": {**metrics["production"]["invasive_sensitivity"],
                                 "source": "eval/metrics.json production.invasive_sensitivity",
                                 "definition": "share of Invasive images whose top class is Invasive"},
        "invasive_sensitivity_patient_bootstrap": {
            **X.ci_rate((top[inv] == agg).astype(float), group[inv], boot),
            "source": "data/predictions.csv test rows (computed here, same bootstrap as TCGA)"},
        "normal_specificity": {
            **X.ci_rate((top[norm] != agg).astype(float), group[norm], boot),
            "source": "data/predictions.csv test rows (computed here)",
            "definition": "share of Normal images whose top class is not Invasive"},
        "invasive_vs_normal_auroc": {
            **X.ci_auroc(inv[both].astype(int), p_inv[both], group[both], boot),
            "source": "data/predictions.csv test rows (computed here)",
            "definition": "AUROC of calibrated P(Invasive), Invasive vs Normal images"},
        "invasive_tier_1_2_rate": {
            **X.ci_rate(((top[inv] == agg) | in_set[inv]).astype(float), group[inv], boot),
            "source": "data/predictions.csv test rows set_tier (computed here)",
            "definition": "share of Invasive images with top class Invasive or Invasive in set_tier (biopsy-lane rule)"},
        "reliability_flag_rate_clean": {"estimate": (rel.get("flag_rate_test") or {}).get("clean"),
                                        "source": "eval/reliability.json flag_rate_test.clean"},
        "classes_in_bach_not_in_tcga": ["Benign", "InSitu"],
    }


# ---- latency ------------------------------------------------------------------------------------------------------------
def hardware_string(device: str, dtype: str) -> str:
    import torch

    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                             check=False).stdout.strip()
    except OSError:
        cpu = ""
    return (f"{cpu or platform.processor() or 'unknown CPU'}; macOS {platform.mac_ver()[0] or 'n/a'}; "
            f"torch {torch.__version__}; device {device}; {dtype} autocast")


def latency_summary(cache: FeatureCache, keys: set[str], hardware: str) -> dict[str, Any]:
    recs = sorted((v for k, v in cache.latency.items() if k in keys), key=lambda v: v["order"])
    recs = [r for r in recs if r["order"] >= WARMUP]

    def summ(vals: Sequence[float]) -> dict[str, Any]:
        if not vals:
            return {"median": None, "p25": None, "p75": None, "iqr": None, "mean": None, "n": 0}
        p25, med, p75 = (float(v) for v in np.percentile(vals, [25, 50, 75]))
        return {"median": med, "p25": p25, "p75": p75, "iqr": p75 - p25, "mean": float(np.mean(vals)), "n": len(vals)}

    return {"cohort": "protocol25", "warmup_excluded": WARMUP, "hardware": hardware,
            "definition": "wall time per 2048 x 1536 crop: PNG read, QC, 12 regions x 1 view preprocessed, one MPS "
                          "batch, pooling, calibrated probabilities, a one-row APS set and the tier",
            "end_to_end_s": summ([r["total_s"] for r in recs]), "embedding_only_s": summ([r["embed_s"] for r in recs])}


# ---- BACH parity gate ------------------------------------------------------------------------------------------------------
class HTTPRangeFile(_io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests (1 MiB blocks, small LRU cache)."""

    def __init__(self, url: str, block: int = 1 << 20, max_blocks: int = 48):
        import requests

        self.url, self.block, self.max_blocks = url, block, max_blocks
        self.session = requests.Session()
        self.cache: dict[int, bytes] = {}
        r = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        if r.status_code != 206:
            raise IOError(f"{url}: range requests not honored (HTTP {r.status_code})")
        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            start = i * self.block
            end = min(self.size, start + self.block) - 1
            for attempt in range(5):
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == end - start + 1:
                    break
                time.sleep(2 ** attempt)
            else:
                raise IOError(f"{self.url}: range {start}-{end} failed")
            if len(self.cache) >= self.max_blocks:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = r.content
        return self.cache[i]

    def readinto(self, b: Any) -> int:
        n = min(len(b), max(0, self.size - self.pos))
        got = 0
        while got < n:
            i, off = divmod(self.pos + got, self.block)
            chunk = self._block(i)[off: off + n - got]
            b[got: got + len(chunk)] = chunk
            got += len(chunk)
        self.pos += got
        return got


def fetch_bach_originals(data_root: Path, out_dir: Path, cases: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Original BACH TIFFs of `cases` from the public Zenodo archive (range reads of the zip; never committed)."""
    table = rm.read_cases(data_root).set_index("case_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {c: f"{BACH_ZIP_ROOT}/{table.loc[c, 'source_file']}" for c in cases}
    found: dict[str, dict[str, Any]] = {}
    missing = [c for c in cases if not (out_dir / f"{c}.tif").is_file()]
    if missing:
        with zipfile.ZipFile(_io.BufferedReader(HTTPRangeFile(BACH_ZIP_URL), buffer_size=1 << 16)) as zf:
            names = zf.namelist()
            for c in missing:
                hits = [n for n in names if n.endswith(want[c])]
                if len(hits) != 1:
                    raise SystemExit(f"{want[c]}: {len(hits)} matches in the BACH archive")
                (out_dir / f"{c}.tif").write_bytes(zf.read(hits[0]))
    for c in cases:
        data = (out_dir / f"{c}.tif").read_bytes()
        found[c] = {"archive_member": want[c], "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return found


def run_parity(ctx: cli.StepContext, device: str) -> dict[str, Any]:
    cfg, data_root = ctx.cfg, ctx.data_root
    prod = load_production(data_root, ctx.artifacts_root, cfg)
    v = prod.views
    out_dir = data_root / PROD_DIR / "bach_parity"
    try:
        sources = fetch_bach_originals(data_root, out_dir, PARITY_CASES)
        source_kind = "original BACH TIFF (Zenodo record 3632035)"
    except Exception as exc:  # noqa: BLE001 - fall back to display JPEGs, labelled
        print(f"parity: original TIFFs unavailable ({exc}); using display JPEGs", file=sys.stderr)
        sources, source_kind = {}, "display JPEG upsampled to 2048 x 1536 (informational only)"
    enc = E.load_encoder(v["encoder"], E.load_encoders_config(), device=device)
    x_store, index = rm.load_emb(data_root, v["encoder"], v["set"], v["head"])
    tiles = rm.read_tiles(data_root)
    pred = rm.read_csv(schemas.predictions_csv(data_root)).set_index("case_id")
    cases = rm.read_cases(data_root).set_index("case_id")
    rows, cos_all, dp_all = [], [], []
    for case in PARITY_CASES:
        if sources:
            with Image.open(out_dir / f"{case}.tif") as handle:
                img = handle.convert("RGB")
        else:
            with Image.open(data_root / cases.loc[case, "display_path"]) as handle:
                img = handle.convert("RGB").resize(CROP_SIZE, Image.LANCZOS)
        stored_idx = index[(index["case_id"] == case) & (index["grid"] == v["grid"])].sort_values("tile_idx")
        stored = x_store[stored_idx["emb_row"].to_numpy()]
        tq = tiles[(tiles["case_id"] == case) & (tiles["grid"] == v["grid"])].sort_values("tile_idx")
        boxes = common.grid_boxes(cfg, v["grid"])
        local = embed_regions(enc, img, boxes, v["n_views"], v["head"])
        local32 = embed_regions(enc, img, boxes, v["n_views"], v["head"], use_autocast=False)
        cos = rowwise_cosine(local, stored)
        cos32 = rowwise_cosine(local32, stored)
        regions = qc_regions(img, cfg, v["grid"])
        app_qc = [{"qc_pass": bool(p)} for p in tq["qc_pass"]]
        app = pred.loc[case, [schemas.cal_prob_col(c) for c in prod.classes]].to_numpy(dtype=float)
        p_store = calibrated(prod, X.pool_qc(stored, app_qc)[0])[0]
        p_local_appqc = calibrated(prod, X.pool_qc(local, app_qc)[0])[0]
        p_local = calibrated(prod, X.pool_qc(local, regions)[0])[0]
        dp = float(np.abs(p_local - app).max())
        cos_all += cos.tolist()
        dp_all.append(dp)
        rows.append({
            "case_id": case, "label": cases.loc[case, "label"], "source": source_kind, **sources.get(case, {}),
            "region_cosine_min": float(cos.min()), "region_cosine_mean": float(cos.mean()),
            "region_cosine_min_fp32": float(cos32.min()),
            "qc_regions_differ_from_tiles_csv": int(sum(r["qc_pass"] != bool(p) for r, p in zip(regions, tq["qc_pass"]))),
            "tissue_frac_max_abs_diff": float(np.abs(np.array([r["tissue_frac"] for r in regions])
                                                     - tq["tissue_frac"].to_numpy(dtype=float)).max()),
            "probs_app": dict(zip(prod.classes, app.tolist())),
            "probs_local": dict(zip(prod.classes, p_local.tolist())),
            "max_abs_prob_diff": dp,
            "max_abs_prob_diff_local_emb_app_qc": float(np.abs(p_local_appqc - app).max()),
            "max_abs_prob_diff_stored_emb_app_qc": float(np.abs(p_store - app).max()),
        })
        print(f"parity {case}: cos min {cos.min():.5f} (fp32 {cos32.min():.5f}), max |dP| {dp:.5f}", flush=True)
    passed = bool(sources) and min(cos_all) >= PARITY_MIN_COSINE and max(dp_all) <= PARITY_MAX_PROB_DIFF
    result = {
        "created_utc": io.utc_now(), "device": device, "hardware": hardware_string(device, enc.info["compute_dtype"]),
        "encoder_revision": enc.info.get("revision"), "model": prod.name, "n_views": v["n_views"],
        "images": list(PARITY_CASES), "image_source": source_kind,
        "criteria": {"region_cosine_min": PARITY_MIN_COSINE, "max_abs_prob_diff": PARITY_MAX_PROB_DIFF,
                     "probabilities": "local QC + local embeddings vs data/predictions.csv pcal__*"},
        "region_cosine_min": float(min(cos_all)), "region_cosine_mean": float(np.mean(cos_all)),
        "max_abs_prob_diff": float(max(dp_all)), "pass": passed, "cases": rows,
    }
    io.write_json(data_root / PROD_DIR / "parity.json", result)
    return result


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


# ---- score stage ---------------------------------------------------------------------------------------------------------
LIMITATIONS = [
    "Labels are slide-level sample types (Primary Tumor, Solid Tissue Normal); a tumor-slide crop may show only "
    "stroma or fat, and there are no tumor outlines.",
    "TCGA-BRCA has no benign or in situ labels: four-class accuracy, in situ vs invasive and the p63/calponin rule are "
    "not tested; specificity is measured only against frozen normal tissue adjacent to tumor.",
    "The reliability flag and the OOD percentile are not computed (CONCH and MUSK are not on this laptop); "
    "reliability_flag is forced false, which can only lower tiers relative to the app.",
    "The sample is small (at most 25 patients); bootstrap intervals over patients are wide.",
    "protocol25 crops come from the GDC DeepZoom tile API rather than local openslide decoding; re-extracting 23 "
    "openslide crops through the API reproduced them pixel for pixel, but the API's thumbnails can shift which "
    "tissue-rich fields are sampled (tile_equivalence).",
    "BACH reference values are per image; TCGA values are per slide (maximum over up to 6 fields), so the "
    "BACH-vs-TCGA comparison is indicative, not a matched test.",
    "Latency is measured on one laptop (Apple MPS) for single 2048 x 1536 fields, not whole slides.",
    "Research prototype; not clinically validated.",
    "Credit: TCGA Research Network (https://www.cancer.gov/tcga), via the NCI Genomic Data Commons.",
]


def manifest_rows(root: Path) -> list[dict[str, Any]]:
    return sorted(X.read_manifest(root), key=lambda r: str(r["crop_id"]))


def tile_equivalence_block(data_root: Path) -> dict[str, Any] | None:
    """TCGA_script.py --equivalence-check result, the one-slide thumbnail diagnostic and the network test record."""
    eq_dir = data_root / PROD_DIR / "equivalence"
    path = eq_dir / "tile_equivalence.json"
    if not path.is_file():
        return None
    eq = io.read_json(path)
    keep = ("created_utc", "reference", "rule", "n_slides", "n_slides_picks_identical", "n_slides_mpp_match",
            "n_crops_compared", "mad_max", "mad_mean", "mad_median", "max_abs_diff_max", "reuse_allowed")
    extra = {name: io.read_json(eq_dir / f"{name}.json") if (eq_dir / f"{name}.json").is_file() else None
             for name in ("thumbnail_diagnostic", "network_test")}
    slides = [{k: s.get(k) for k in ("file_name", "group", "picks_identical", "n_candidates", "n_picks", "n_ref_crops",
                                     "thumb_slide_level", "thumb_dz_downsample", "mpp_matches_manifest")}
              for s in eq.get("slides", [])]
    decision = ("reuse the earlier openslide crops for the first 8 patients" if eq.get("reuse_allowed") else
                "extract all 25 patients through tiles (pre-registered branch: FOV picks were not identical on every "
                "earlier openslide slide)")
    return {**{k: eq.get(k) for k in keep}, "decision": decision, "source_file": str(path.relative_to(data_root)),
            "crops": eq.get("crops"), "slides": slides, **extra}


def run_score(ctx: cli.StepContext, device: str, skip_tta8: bool) -> Path:
    cfg, data_root, art = ctx.cfg, ctx.data_root, ctx.artifacts_root
    parity_path = data_root / PROD_DIR / "parity.json"
    if not parity_path.is_file():
        raise SystemExit("run --stage parity first (pre-registered gate)")
    parity = io.read_json(parity_path)
    if not parity.get("pass"):
        raise SystemExit(f"parity gate did not pass ({parity_path}); not scoring TCGA")
    prod = load_production(data_root, art, cfg)
    v = prod.views
    enc = E.load_encoder(v["encoder"], E.load_encoders_config(), device=device)
    meta_rev = io.read_json(data_root / "emb" / v["encoder"] / "meta.json").get("revision")
    if enc.info.get("revision") != meta_rev:
        raise SystemExit(f"encoder revision {enc.info.get('revision')} != stored embeddings {meta_rev}")
    hardware = hardware_string(device, enc.info["compute_dtype"])
    rel = io.read_json(schemas.eval_json(art, "reliability"))
    bach_flag = (rel.get("flag_rate_test") or {}).get("clean")
    emb_dir = data_root / PROD_DIR / "embeddings"
    tta_views = int(cfg["encoders"]["views"][v["grid"]])
    cohorts_def = {"protocol25": data_root / P25_DIR, "same8": data_root / SAME8_DIR}   # protocol25 first: latency
    cache = FeatureCache(emb_dir / f"{v['encoder']}_{v['set']}_{device}_v{v['n_views']}.npz")
    cache8 = None if skip_tta8 else FeatureCache(emb_dir / f"{v['encoder']}_{v['set']}_{device}_v{tta_views}.npz")
    cohorts: dict[str, Any] = {}
    p25_keys: set[str] = set()
    for name, root in cohorts_def.items():
        rows = manifest_rows(root)
        print(f"{name}: {len(rows)} crops from {root}", flush=True)
        feats = featurize(rows, root, enc, prod, cfg, cache, v["n_views"], timed=True)
        if name == "protocol25":
            p25_keys = {f["key"] for f in feats.values() if "key" in f}
        crops = score_crops(rows, feats, prod, cfg)
        payload = cohort_payload(name, root, rows, feats, crops, analyze_cohort(crops, cfg, bach_flag))
        if cache8 is not None:
            print(f"{name}: S2 {tta_views}-view read", flush=True)
            feats8 = featurize(rows, root, enc, prod, cfg, cache8, tta_views, timed=False)
            payload["sensitivity_s2_tta8"] = tta8_block(score_crops(rows, feats8, prod, cfg), cfg, tta_views, bach_flag)
        cohorts[name] = payload
    p25 = cohorts["protocol25"]
    top_keys = [k for k in p25 if k not in ("name", "root", "run_info", "slides", "crops")]
    out = {
        "protocol": PROTOCOL, "model": prod.name, "encoder": v["encoder"], "head": v["head"], "device": device,
        "quantized": False, "temperature": prod.temperature, "qhat_aps": prod.qhat, "n_views": v["n_views"],
        "views": v["views"], "cohort": "protocol25",
        **{k: p25[k] for k in top_keys},
        "recipe": {
            "encoder": v["encoder"], "encoder_revision": enc.info.get("revision"), "set": v["set"], "head": v["head"],
            "grid": v["grid"], "views_source": f"data/emb/{v['encoder']}/meta.json sets.{v['set']}.n_views and the "
                                                f"{v['set']}_{v['head']}_index.csv n_views column",
            "compute_dtype": enc.info["compute_dtype"], "device": device, "stored_precision": "float16",
            "qc": "pipeline/01q rule on the display rendition; region passes at tissue fraction >= "
                  f"{cfg['qc']['min_tissue_frac']}; insufficient tissue below {cfg['qc']['min_tiles_kept']} regions",
            "model_file": str(prod.model_path.relative_to(data_root)), "temperature": prod.temperature,
            "temperature_source": "eval/calibration.json (train out-of-fold fit)",
            "qhat_aps": prod.qhat, "qhat_source": f"eval/model_choice.json conformal.{prod.name}.qhat_aps, re-derived "
                                                  "from data/predictions.csv calibration rows",
            "alpha": float(cfg["alpha"]),
            "sets": f"randomized APS, U from default_rng({int(cfg['conformal']['seed']) + 1}), one call per cohort "
                    "matrix sorted by crop_id, argmax added to empty sets",
            "tier": "core.tiers.assign_tier, biopsy lane, no aging, reliability_flag false",
            "reliability": "not computed (CONCH/MUSK not local); forced false",
        },
        "parity": parity,
        "tile_equivalence": tile_equivalence_block(data_root),
        "cohorts": {"same8": cohorts["same8"]},
        "bach_reference": bach_reference(data_root, art, cfg),
        "latency_s_per_image": latency_summary(cache, p25_keys, hardware),
        "tcga_labels_note": "TCGA-BRCA provides no benign or in situ labels; only Invasive (primary tumor) and Normal "
                            "(solid tissue normal) slide labels are tested.",
        "limitations": LIMITATIONS,
        "slides": p25["slides"],
        "crops": p25["crops"],
        "data": {"tcga_root": p25["root"], "run_info": p25["run_info"], "embedding_cache": str(cache.path),
                 "same8_root": cohorts["same8"]["root"]},
    }
    return emetrics.write_eval(art, NAME, out, cfg, synthetic=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.external_tcga_prod", "TCGA-BRCA external test of the production model.")
    parser.add_argument("--stage", choices=("parity", "score"), required=True)
    parser.add_argument("--device", default=None, help="mps | cuda | cpu (default: best available)")
    parser.add_argument("--skip-tta8", action="store_true", help="skip sensitivity analysis S2 (8-view read)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    device = X.resolve_device(args.device)
    if args.stage == "parity":
        res = run_parity(ctx, device)
        print(f"parity: pass={res['pass']} cosine min {res['region_cosine_min']:.5f}, max |dP| "
              f"{res['max_abs_prob_diff']:.5f} ({res['image_source']})")
        return 0 if res["pass"] else 3
    out = run_score(ctx, device, bool(args.skip_tta8))
    print(f"eval.external_tcga_prod: wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
