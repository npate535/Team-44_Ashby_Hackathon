"""Archived round 1: Delta-trained UNI2-h LoRA (uni2h_ft) on TCGA-BRCA crops (TCGA_TODO.md). Not used in Sepia.

    python -m eval.external_tcga --tcga-root data/tcga_brca_test [--device mps|cpu] [--limit N]

uni2h_ft is not the production model. This script writes only the archive file
artifacts/archive/eval/external_tcga_round1.json, which the app does not serve or package. The served
artifacts/eval/external_tcga.json is the production-model test (eval/external_tcga_prod.py, docs/EXTERNAL_TCGA_R2.md).
Its helpers (tile_case, pool_qc, slide_rows, analyze, ci_rate, ci_auroc, read_manifest) are shared with that test.

This is the GPU fine-tune from pipeline/05c_finetune.py (not the frozen uni2h512_lr probe):
  * encoder uni2h_ft = UNI2-h + LoRA at ~/.cache/pathqueue_ft/uni2h_ft_lora.pt  (full precision, never quantized)
  * head data/models/uni2h_ft_head.npz
  * temperature from artifacts/eval/finetune.json temperatures_from_oof.uni2h_ft
  * APS q-hat fit on BACH calibration rows of p__uni2h_ft__* (same method as settings; thresholds from this model)

Slide-level primary analyses per TCGA_TODO.md.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from core import calibration, cli, conformal, io, stats, tiers
from eval import metrics as emetrics
from pipeline import common, encoders, riskmodels as rm

STEP = "eval_external_tcga"
MODEL = "uni2h_ft"
ENCODER = "uni2h_ft"
HEAD = "probe"
GRID = "g512"
LORA_PATH = Path.home() / ".cache" / "pathqueue_ft" / "uni2h_ft_lora.pt"
ARCHIVE = Path("archive") / "eval" / "external_tcga_round1.json"   # under the artifacts root; never artifacts/eval/


def resolve_device(requested: str | None) -> str:
    import torch

    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(f"--device {requested} requested but CUDA is unavailable")
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise SystemExit("--device mps requested but MPS is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_ft_artifacts(data: Path) -> Path:
    head = data / "models" / "uni2h_ft_head.npz"
    missing = []
    if not LORA_PATH.is_file():
        missing.append(f"LoRA weights missing: {LORA_PATH}")
    if not head.is_file():
        missing.append(f"classification head missing: {head}")
    if missing:
        raise SystemExit(
            "Delta-trained uni2h_ft artifacts are not on this laptop:\n  - "
            + "\n  - ".join(missing)
            + "\n\nFrom an authenticated Delta session, copy them (paths are private — do not put under /work/nvme/bfqi shared):\n"
            "  mkdir -p ~/.cache/pathqueue_ft && chmod 700 ~/.cache/pathqueue_ft\n"
            "  ./scripts/delta.sh 'ls -la ~/.cache/pathqueue_ft /work/nvme/bfqi/$USER/pathqueue/data/models/uni2h_ft_head.npz 2>/dev/null; "
            "ls -la ~/pathqueue/data/models/uni2h_ft_head.npz 2>/dev/null'\n"
            "  # then scp the .pt and uni2h_ft_head.npz into the paths above\n"
            "Also need a Hugging Face token with MahmoodLab/UNI2-h access (base encoder, full precision)."
        )
    return head


def load_frozen(art: Path, data: Path) -> tuple[rm.LinearModel, float, float, list[str], dict[str, Any]]:
    head_path = require_ft_artifacts(data)
    model = rm.LinearModel.from_npz(head_path)
    classes = list(model.classes)
    ft = io.read_json(art / "eval" / "finetune.json")
    temps = ft.get("temperatures_from_oof") or {}
    if MODEL not in temps:
        raise SystemExit(f"finetune.json has no temperatures_from_oof[{MODEL!r}]")
    temperature = float(temps[MODEL])
    qhat = fit_aps_qhat(data, classes, temperature)
    meta = {
        "finetune": {
            "model": ft.get("model"),
            "adopted_as_production": ft.get("adopted_as_production"),
            "rule_choice": ft.get("rule_choice"),
            "recipe": ft.get("recipe"),
        },
        "lora_path": str(LORA_PATH),
        "head_path": str(head_path),
    }
    return model, temperature, qhat, classes, meta


def fit_aps_qhat(data: Path, classes: Sequence[str], temperature: float) -> float:
    """APS q-hat on BACH calibration using ingested uni2h_ft probabilities (settings conformal)."""
    cfg_path_preds = data / "predictions.csv"
    cases = rm.read_cases(data)
    cal_ids = set(cases.loc[cases["split"] == "calibration", "case_id"].astype(str))
    pred = pd.read_csv(cfg_path_preds, low_memory=False)
    pred["case_id"] = pred["case_id"].astype(str)
    cal = pred[pred["case_id"].isin(cal_ids)].copy()
    # Prefer clean calibration rows only (no _FSIM_ / COMP_).
    cal = cal[~cal["case_id"].str.contains("_FSIM_|_COMP_|COMP_", regex=True)]
    cols = [f"p__{MODEL}__{c}" for c in classes]
    missing = [c for c in cols if c not in cal.columns]
    if missing:
        raise SystemExit(f"predictions.csv missing {missing}; run pipeline.05c_finetune ingest first")
    # Join labels
    label_map = dict(zip(cases["case_id"].astype(str), cases["label"].astype(str)))
    cal["label"] = cal["case_id"].map(label_map)
    cal = cal.dropna(subset=["label"])
    probs = cal[cols].to_numpy(dtype=float)
    # Ingested p__ columns are already the model's probabilities (uncalibrated softmax); apply temperature
    # via logits recovery when needed. Here we treat them as uncalibrated probs and re-temperature:
    # softmax(log(p)/T) is equivalent when p > 0.
    probs = np.clip(probs, 1e-12, 1.0)
    logits = np.log(probs)
    probs_t = calibration.softmax(logits, temperature)
    y = np.array([list(classes).index(lbl) for lbl in cal["label"]])
    from core import config as cfgmod

    cfg = cfgmod.load_settings()
    alpha = float(cfg["alpha"])
    seed = int(cfg["conformal"]["seed"])
    randomized = bool(cfg["conformal"]["aps_randomized"])
    return float(conformal.fit_aps(probs_t, y, alpha, seed, randomized))


def read_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "manifest.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run: python TCGA_script.py --out {root}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def tile_case(img: Image.Image, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    common.check_grid_fits(cfg, img.width, img.height)
    mask = common.tissue_mask(img, cfg)
    min_frac = float(cfg["qc"]["min_tissue_frac"])
    out = []
    for box in common.grid_boxes(cfg, GRID):
        frac = common.tissue_fraction(mask, box["x"], box["y"], box["size"], box["size"])
        out.append({**box, "tissue_frac": frac, "qc_pass": frac >= min_frac})
    return out


def embed_tiles(enc: encoders.Encoder, img: Image.Image, tiles: Sequence[Mapping[str, Any]],
                batch: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import torch

    crops, kept = [], []
    for t in tiles:
        crop = img.crop((t["x"], t["y"], t["x"] + t["size"], t["y"] + t["size"]))
        crops.append(enc.preprocess(crop))
        kept.append(dict(t))
    if not crops:
        return np.zeros((0, int(enc.spec["heads"][HEAD]["dim"])), np.float32), []
    feats = []
    for i in range(0, len(crops), batch):
        batch_t = torch.stack(crops[i:i + batch])
        out = enc.embed(batch_t, use_autocast=True)
        feats.append(out[HEAD])
    x = np.concatenate(feats, axis=0).astype(np.float32)
    for j, row in enumerate(kept):
        row["emb_row"] = j
    return x, kept


def pool_qc(x: np.ndarray, tiles: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, int, bool]:
    pass_idx = [i for i, t in enumerate(tiles) if t["qc_pass"]]
    fallback = not pass_idx
    use = pass_idx or list(range(len(tiles)))
    return x[use].mean(axis=0), len(pass_idx), fallback


def score_case(feat: np.ndarray, model: rm.LinearModel, temperature: float, qhat: float,
               classes: Sequence[str], cfg: Mapping[str, Any], qc_tiles: int
               ) -> dict[str, Any]:
    logits = model.logits(feat[None, :])[0]
    probs = calibration.softmax(logits[None, :], temperature)[0]
    aps = conformal.aps_sets(probs[None, :], qhat, seed=None, randomized=False)
    if bool(cfg["conformal"].get("nonempty", False)):
        aps = conformal.nonempty_sets(aps, probs[None, :])
    set_labels = conformal.set_labels(aps[0], classes)
    inv_i = list(classes).index("Invasive")
    top_i = int(probs.argmax())
    case_for_tier = {
        "lane": "biopsy",
        "arrived_offset_s": 0.0,
        "prediction": {"classes": list(classes), "probs": probs.tolist(), "conformal_set": set_labels},
        "quality": {"reliability_flag": False, "qc_tiles_kept": int(qc_tiles)},
    }
    tier, reasons = tiers.assign_tier(case_for_tier, cfg, now=None)
    return {
        "probs": {c: float(probs[i]) for i, c in enumerate(classes)},
        "p_invasive": float(probs[inv_i]),
        "top": classes[top_i],
        "conformal_set": set_labels,
        "set_size": len(set_labels),
        "tier": int(tier),
        "tier_reasons": reasons,
        "qc_tiles_kept": int(qc_tiles),
    }


def ci_rate(y: np.ndarray, groups: np.ndarray, boot: Mapping[str, Any]) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_groups": 0}

    def mean(a: np.ndarray) -> float:
        return float(a.mean()) if a.size else math.nan

    return stats.bootstrap_ci(mean, [y], groups, None, int(boot["n"]), float(boot["ci"]), int(boot["seed"]))


def ci_auroc(y: np.ndarray, scores: np.ndarray, groups: np.ndarray, boot: Mapping[str, Any]) -> dict[str, Any]:
    return stats.bootstrap_ci(stats.auroc, [y, scores], groups, None,
                              int(boot["n"]), float(boot["ci"]), int(boot["seed"]))


def slide_rows(crop_preds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_slide: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in crop_preds:
        by_slide[(row["patient_id"], row["group"])].append(row)
    out = []
    for (patient, group), crops in sorted(by_slide.items()):
        inv_scores = [c["p_invasive"] for c in crops]
        tops = [c["top"] for c in crops]
        tiers_ = [c["tier"] for c in crops]
        out.append({
            "patient_id": patient,
            "group": group,
            "label": crops[0]["label"],
            "prep": crops[0]["prep"],
            "n_crops": len(crops),
            "slide_invasive_score": float(max(inv_scores)),
            "called_invasive": any(t == "Invasive" for t in tops),
            "any_tier_1_2": any(t in (1, 2) for t in tiers_),
            "min_tier": int(min(tiers_)),
            "crop_ids": [c["crop_id"] for c in crops],
        })
    return out


def analyze(slides: Sequence[Mapping[str, Any]], crops: Sequence[Mapping[str, Any]],
            boot: Mapping[str, Any], bach_flag_clean: float | None) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for s in slides:
        by_group[s["group"]].append(s)

    def group_sens(group: str) -> dict[str, Any]:
        rows = by_group.get(group, [])
        if not rows:
            return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_groups": 0}
        y = np.array([1.0 if r["called_invasive"] else 0.0 for r in rows])
        g = np.array([r["patient_id"] for r in rows])
        return ci_rate(y, g, boot)

    ffpe = group_sens("ffpe_tumor")
    frozen_tumor = group_sens("frozen_tumor")
    ft = by_group.get("frozen_tumor", [])
    fn = by_group.get("frozen_normal", [])
    if ft and fn:
        paired = ft + fn
        y = np.array([1 if r["label"] == "Invasive" else 0 for r in paired])
        scores = np.array([r["slide_invasive_score"] for r in paired], dtype=float)
        groups = np.array([r["patient_id"] for r in paired])
        frozen_disc = {
            "auroc": ci_auroc(y, scores, groups, boot),
            "sensitivity": ci_rate(np.array([1.0 if r["called_invasive"] else 0.0 for r in ft]),
                                   np.array([r["patient_id"] for r in ft]), boot),
            "specificity": ci_rate(np.array([1.0 if not r["called_invasive"] else 0.0 for r in fn]),
                                   np.array([r["patient_id"] for r in fn]), boot),
        }
    else:
        frozen_disc = {"auroc": None, "sensitivity": None, "specificity": None}

    tumor_slides = [s for s in slides if s["group"] in ("ffpe_tumor", "frozen_tumor")]
    tier_rate = None
    if tumor_slides:
        tier_rate = ci_rate(np.array([1.0 if s["any_tier_1_2"] else 0.0 for s in tumor_slides]),
                            np.array([s["patient_id"] for s in tumor_slides]), boot)

    hist: dict[str, list[float]] = defaultdict(list)
    for c in crops:
        hist[c["group"]].append(float(c["p_invasive"]))
    edges = [i / 10 for i in range(11)]
    hist_out = {}
    for group, vals in sorted(hist.items()):
        counts, _ = np.histogram(vals, bins=edges)
        hist_out[group] = {
            "n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "bins": edges, "counts": counts.astype(int).tolist(),
        }

    return {
        "ffpe_tumor_sensitivity": ffpe,
        "frozen_tumor_sensitivity": frozen_tumor,
        "ffpe_vs_frozen_tumor_sensitivity_gap": {
            "estimate": (None if ffpe.get("estimate") is None or frozen_tumor.get("estimate") is None
                         else float(ffpe["estimate"]) - float(frozen_tumor["estimate"])),
            "ffpe": ffpe.get("estimate"),
            "frozen": frozen_tumor.get("estimate"),
        },
        "frozen_tumor_vs_normal": frozen_disc,
        "tumor_tier_1_2_rate": tier_rate,
        "crop_p_invasive_hist": hist_out,
        "bach_reliability_flag_rate_clean": bach_flag_clean,
        "reliability_note": (
            "Full BACH reliability (CONCH OOD + MUSK disagree) not recomputed on TCGA; "
            "tier uses conformal set size + top-class only (reliability_flag=false)."
        ),
    }


def run(tcga_root: Path, data_root: Path, art_root: Path, cfg: Mapping[str, Any],
        device: str, batch: int, limit: int | None, force: bool) -> Path:
    boot = cfg["bootstrap"]
    model, temperature, qhat, classes, meta = load_frozen(art_root, data_root)
    rows = read_manifest(tcga_root)
    if limit is not None:
        rows = rows[: int(limit)]

    cache_path = tcga_root / "embeddings" / f"{ENCODER}_{device}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ecfg = encoders.load_encoders_config()
    enc = encoders.load_encoder(ENCODER, ecfg, device=device)

    cached: dict[str, np.ndarray] = {}
    if cache_path.is_file() and not force:
        z = np.load(cache_path, allow_pickle=False)
        for k in z.files:
            cached[k] = z[k]

    crop_preds: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for i, row in enumerate(rows, 1):
        crop_id = row["crop_id"]
        img_path = tcga_root / row["path"]
        print(f"[{i}/{len(rows)}] {crop_id}", flush=True)
        try:
            if not img_path.is_file():
                raise FileNotFoundError(img_path)
            img = Image.open(img_path).convert("RGB")
            if img.size != (2048, 1536):
                raise ValueError(f"expected 2048x1536, got {img.size}")
            tiles = tile_case(img, cfg)
            if crop_id in cached and not force:
                feat = np.asarray(cached[crop_id], dtype=np.float32)
                n_qc = sum(1 for t in tiles if t["qc_pass"])
                qc_fallback = n_qc == 0
            else:
                x, tile_rows = embed_tiles(enc, img, tiles, batch)
                feat, n_qc, qc_fallback = pool_qc(x, tile_rows)
                cached[crop_id] = feat.astype(np.float32)
            scored = score_case(feat, model, temperature, qhat, classes, cfg, n_qc)
            crop_preds.append({
                "crop_id": crop_id, "path": row["path"], "label": row["label"], "group": row["group"],
                "prep": row["prep"], "patient_id": row["patient_id"],
                "slide_file_name": row["slide_file_name"], "qc_fallback": qc_fallback,
                "mpp_estimated": str(row.get("mpp_estimated", "")).lower() in ("1", "true"),
                **scored,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {crop_id}: {exc}", file=sys.stderr, flush=True)
            failed.append({"crop_id": crop_id, "error": f"{type(exc).__name__}: {exc}"})

    if cached:
        np.savez_compressed(cache_path, **{k: np.asarray(v, dtype=np.float32) for k, v in cached.items()})

    slides = slide_rows(crop_preds)
    short = [s for s in slides if s["n_crops"] < 6]
    bach_flag = None
    rel_path = art_root / "eval" / "reliability.json"
    if rel_path.is_file():
        bach_flag = io.read_json(rel_path).get("flag_rate_test", {}).get("clean")

    primary = analyze(slides, crop_preds, boot, bach_flag)
    run_info = io.read_json(tcga_root / "run_info.json") if (tcga_root / "run_info.json").is_file() else {}

    payload = {
        "protocol": "TCGA_TODO.md",
        "model": MODEL,
        "encoder": ENCODER,
        "head": HEAD,
        "device": device,
        "quantized": False,
        "temperature": temperature,
        "qhat_aps": qhat,
        "ft_meta": meta,
        "n_crops": len(crop_preds),
        "n_crops_failed": len(failed),
        "n_slides": len(slides),
        "n_patients": len({s["patient_id"] for s in slides}),
        "groups": {g: sum(1 for s in slides if s["group"] == g) for g in sorted({s["group"] for s in slides})},
        "slides_short_crops": short,
        "failed_crops": failed,
        "primary": primary,
        "slides": slides,
        "crops": crop_preds,
        "limitations": [
            "Model: Delta-trained UNI2-h LoRA (uni2h_ft), not the frozen uni2h512_lr production probe.",
            "Slide-level labels only; a tumor-slide crop may show stroma or fat.",
            "No benign or in-situ labels in TCGA-BRCA; specificity is vs frozen normal tissue only.",
            "Normal slides are adjacent to tumor and may not be strictly normal.",
            "Sample size is small; bootstrap intervals are wide.",
            "Reliability combiner not applied; reliability_flag forced false.",
            "Credit: TCGA Research Network (https://www.cancer.gov/tcga).",
        ],
        "data": {"tcga_root": str(tcga_root), "run_info": run_info, "embedding_cache": str(cache_path)},
    }
    return write_archive(art_root, payload, cfg)


def write_archive(art_root: Path, payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> Path:
    """Write the round-1 result to the archive (eval envelope included), never to the served artifacts/eval/."""
    obj = {**emetrics.envelope("external_tcga_round1", cfg, synthetic=False), **payload}
    return io.write_json(Path(art_root) / ARCHIVE, emetrics._finite(obj))


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser("eval.external_tcga", "Archived round 1: TCGA-BRCA test of the uni2h_ft LoRA (not served).")
    parser.add_argument("--tcga-root", type=Path, default=Path("data/tcga_brca_test"))
    parser.add_argument("--device", default=None, help="mps | cuda | cpu (default: best available)")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    device = resolve_device(args.device)
    print(f"model={MODEL} device={device} quantized=false", flush=True)
    try:
        out = run(Path(args.tcga_root), ctx.data_root, ctx.artifacts_root, ctx.cfg,
                  device, int(args.batch), ctx.limit, ctx.force)
    except encoders.EncoderUnavailable as exc:
        print(
            f"eval.external_tcga: {exc}\n"
            "Need base UNI2-h (gated) + LoRA on top. Set HF_TOKEN with MahmoodLab/UNI2-h access, then download:\n"
            "  HF_TOKEN=... uv run python -c \"from huggingface_hub import hf_hub_download as d; "
            "d('MahmoodLab/UNI2-h','pytorch_model.bin'); d('MahmoodLab/UNI2-h','config.json')\"",
            file=sys.stderr,
        )
        return 2
    print(f"eval.external_tcga: wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
