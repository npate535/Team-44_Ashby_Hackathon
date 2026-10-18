"""Step 05c (GPU, Delta): LoRA fine-tuning of UNI2-h on the BACH train split (accuracy round 2).

    python -m pipeline.05c_finetune cv    [--folds 0,1,2,3,4] [--max-minutes 50]
    python -m pipeline.05c_finetune final [--max-minutes 20]
    python -m pipeline.05c_finetune smoke [--out DIR]          # CPU plumbing test: tiny random ViT, synthetic images
    python -m pipeline.05c_finetune [ingest]                   # laptop, after 05: GPU outputs -> predictions + eval

The recipe is fixed in docs/ACCURACY_ROUND2.md before any fine-tuning result exists (RECIPE below; not tuned).
Training sees train-split images only: random 512-px crops (scale jitter +-20%, >= 50% tissue by the 01 tissue mask),
a random dihedral view and mild colour jitter, resized to 224 and normalized per the UNI2-h card; rank-8 LoRA on every
block's attention qkv and proj (pipeline.encoders.add_lora) plus a linear head; AdamW, 5% warm-up then cosine, label
smoothing 0.1, bf16 autocast. An image's logits are the head applied to the mean CLS feature over its QC-passing g512
tiles x 8 dihedral views: the same pooling as the 05 probes, so the head is a LinearModel on pooled features.

cv     one model per fold of round 1's CV repeat 0 (riskmodels.cv_splits), trained on the other folds. Writes the held-out
       logits per fold to <data-root>/finetune/ (resumable: finished folds are skipped) and cv_oof_r0.npz once every
       fold exists, for the same fold-calibrated log-loss as the probes (repeat 0).
final  one model on all train images. LoRA weights go to the PRIVATE path in encoders.yaml uni2h_ft.lora.weights (a
       CC-BY-NC-ND derivative: never an event-readable folder); the head to <data-root>/models/uni2h_ft_head.npz. Then
       `02_embed --encoders uni2h_ft` re-extracts features for every set the pipeline reads (finetune.sbatch).

ingest (laptop; the Makefile runs it after 05) reads <data-root>/finetune/cv_oof.csv and final_logits.csv (from the
       Delta run), re-runs the round-2 rule on CV repeat 0 against the UNI2-h probes on the same folds, adds
       p__uni2h_ft__* (uncalibrated, like every evaluated model) to predictions.csv for train (out-of-fold), calibration
       and test, and writes <artifacts>/eval/finetune.json. It never changes the production model: if the rule picks the
       fine-tune, finetune.json says so and adoption is a separate, recorded step.

--max-minutes bounds the training time: a short timing pass (no optimizer step) measures seconds per batch before
training, and when the recipe's epochs would overrun, the epoch count is reduced (never below 3) and recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from core import cli, io, schemas
from pipeline import common
from pipeline import encoders as E
from pipeline import riskmodels as rm

STEP = "05c_finetune"
ENCODER = "uni2h_ft"
RECIPE: dict[str, Any] = {
    "crop": 512, "scale_jitter": 0.2, "min_tissue": 0.5, "epochs": 10, "crops_per_image": 24,
    "batch": 32, "accum": 2, "lr_lora": 2e-4, "lr_head": 1e-3, "wd_head": 1e-4, "warmup": 0.05,
    "label_smoothing": 0.1, "grad_clip": 1.0, "lora_r": 8, "lora_alpha": 16,
    "jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02},
    "eval_grid": "g512", "eval_views": 8, "seed": 44, "min_epochs": 3, "timing_batches": 6,
}
MASK_DOWN = 8                    # tissue masks at 1/8 resolution (crop acceptance only)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- data ---------------------------------------------------------------------------------------------------------------
def load_images(paths: Sequence[str], cfg: Mapping[str, Any], workers: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    def one(path: str) -> tuple[np.ndarray, np.ndarray]:
        img = common.read_rgb(path)
        small = img.resize((max(1, img.width // MASK_DOWN), max(1, img.height // MASK_DOWN)), Image.BILINEAR)
        return np.asarray(img), common.tissue_mask(small, cfg)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pairs = list(pool.map(one, paths))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def color_jitter(img: Image.Image, rng: np.random.Generator, j: Mapping[str, float]) -> Image.Image:
    for enhancer, key in ((ImageEnhance.Brightness, "brightness"), (ImageEnhance.Contrast, "contrast"),
                          (ImageEnhance.Color, "saturation")):
        img = enhancer(img).enhance(float(rng.uniform(1 - j[key], 1 + j[key])))
    shift = int(round(float(rng.uniform(-j["hue"], j["hue"])) * 255))
    if shift:
        hsv = np.asarray(img.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
        img = Image.fromarray(hsv, "HSV").convert("RGB")
    return img


class CropSet:
    """Training crops: len = images x crops_per_image; randomness from (seed, epoch, index), so runs repeat."""

    def __init__(self, images: list[np.ndarray], masks: list[np.ndarray], labels: np.ndarray, pre: E.Preprocess,
                 recipe: Mapping[str, Any], seed: int) -> None:
        self.images, self.masks, self.labels, self.pre, self.r, self.seed = images, masks, labels, pre, recipe, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.images) * int(self.r["crops_per_image"])

    def __getitem__(self, i: int) -> tuple[Any, int]:
        k = i % len(self.images)
        rng = np.random.default_rng([self.seed, self.epoch, i])
        img, mask = self.images[k], self.masks[k]
        h, w = img.shape[:2]
        side = int(round(self.r["crop"] * rng.uniform(1 - self.r["scale_jitter"], 1 + self.r["scale_jitter"])))
        side = min(side, h, w)
        best = (-1.0, 0, 0)
        for _ in range(30):
            x, y = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
            win = mask[y // MASK_DOWN:(y + side) // MASK_DOWN, x // MASK_DOWN:(x + side) // MASK_DOWN]
            frac = float(win.mean()) if win.size else 0.0
            if frac > best[0]:
                best = (frac, x, y)
            if frac >= self.r["min_tissue"]:
                break
        _, x, y = best
        crop = Image.fromarray(img[y:y + side, x:x + side]).resize((self.pre.crop, self.pre.crop), Image.BILINEAR)
        crop = color_jitter(E.dihedral(crop, int(rng.integers(0, 8))), rng, self.r["jitter"])
        return self.pre(crop), int(self.labels[k])


class TileSet:
    """Inference tiles: every (image, QC tile box, dihedral view) -> (tensor, image position)."""

    def __init__(self, images: list[np.ndarray], boxes: list[list[tuple[int, int, int]]], views: int,
                 pre: E.Preprocess) -> None:
        self.images, self.pre = images, pre
        self.items = [(k, x, y, s, v) for k, bs in enumerate(boxes) for (x, y, s) in bs for v in range(views)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        k, x, y, s, v = self.items[i]
        tile = Image.fromarray(self.images[k][y:y + s, x:x + s])
        return self.pre(E.dihedral(tile, v)), k


def qc_boxes(case_ids: Sequence[str], tiles: pd.DataFrame | None, cfg: Mapping[str, Any], grid: str,
             shape: tuple[int, int]) -> list[list[tuple[int, int, int]]]:
    """QC-passing tile boxes per case (all boxes when none pass or tiles.csv is absent, as riskmodels.pool_tiles)."""
    spec = cfg["tiles"]["grids"][grid]
    rows, cols, size = int(spec["rows"]), int(spec["cols"]), int(spec["size"])
    if shape[0] < rows * size or shape[1] < cols * size:              # smoke images: one centered box
        s = min(shape)
        return [[(0, 0, s)] for _ in case_ids]
    every = [(c * size, r * size, size) for r in range(rows) for c in range(cols)]
    if tiles is None:
        return [list(every) for _ in case_ids]
    t = tiles[tiles["grid"] == grid]
    passing = t[t["qc_pass"].astype(int) == 1].groupby("case_id")["tile_idx"].apply(lambda s: set(s.astype(int)))
    out = []
    for cid in case_ids:
        keep = passing.get(cid, set())
        out.append([every[i] for i in sorted(keep)] or list(every))
    return out


# ---- model --------------------------------------------------------------------------------------------------------------
def build(spec: Mapping[str, Any], n_classes: int, device: str, smoke: bool) -> tuple[Any, list[Any]]:
    import timm
    import torch

    if smoke:
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    else:
        model = timm.create_model(f"hf-hub:{spec['repo']}", pretrained=True, **E._timm_kwargs(spec))
    for p in model.parameters():
        p.requires_grad_(False)
    lora = E.add_lora(model, int(RECIPE["lora_r"]), float(RECIPE["lora_alpha"]))
    model.head_ft = torch.nn.Linear(model.num_features, n_classes)
    return model.to(device), lora


def reset_trainable(model: Any, seed: int) -> None:
    """Fresh LoRA adapters and head (the frozen base never changes, so one loaded model serves every fold)."""
    import torch

    torch.manual_seed(seed)
    for module in model.modules():
        if hasattr(module, "lora_a"):
            torch.nn.init.kaiming_uniform_(module.lora_a, a=math.sqrt(5))
            torch.nn.init.zeros_(module.lora_b)
    model.head_ft.reset_parameters()


def autocast(device: str, spec: Mapping[str, Any]) -> Any:
    import contextlib

    import torch

    if device.startswith("cuda") and spec.get("autocast") == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def loader(ds: Any, batch: int, workers: int, shuffle: bool, seed: int) -> Any:
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, generator=g,
                                       pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                                       persistent_workers=False)


def fit_lora(model: Any, lora: list[Any], ds: CropSet, spec: Mapping[str, Any], device: str, workers: int,
          budget_s: float | None, epochs: int | None, seed: int) -> dict[str, Any]:
    """Train LoRA + head. epochs=None: the recipe's epochs, reduced to fit budget_s after a timing pass."""
    import torch

    r = RECIPE
    reset_trainable(model, seed)
    head = list(model.head_ft.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": r["lr_lora"], "weight_decay": 0.0},
                             {"params": head, "lr": r["lr_head"], "weight_decay": r["wd_head"]}])
    ce = torch.nn.CrossEntropyLoss(label_smoothing=float(r["label_smoothing"]))
    model.train()

    def step_batch(xb: Any, yb: Any) -> Any:
        with autocast(device, spec):
            logits = model.head_ft(model(xb.to(device, non_blocking=True)).float())
        loss = ce(logits.float(), yb.to(device, non_blocking=True))
        (loss / r["accum"]).backward()
        return loss

    info: dict[str, Any] = {}
    if epochs is None:
        n_t, t0 = 0, None
        ds.epoch = 10_000                               # timing crops: an epoch index training never uses
        for i, (xb, yb) in enumerate(loader(ds, r["batch"], workers, True, seed + 10_000)):
            if i == 1:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.time()
            step_batch(xb, yb)
            if i >= 1:
                n_t += 1
            if n_t >= r["timing_batches"]:
                break
        opt.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        sec_per_batch = (time.time() - t0) / max(1, n_t) if t0 else 0.0
        batches_per_epoch = len(ds) // r["batch"]
        need = sec_per_batch * batches_per_epoch * r["epochs"]
        epochs = int(r["epochs"])
        if budget_s and need > budget_s:
            epochs = max(int(r["min_epochs"]), int(r["epochs"] * budget_s / need))
        info.update(sec_per_batch=round(sec_per_batch, 4), projected_s=round(need, 1), epochs_reduced=epochs < r["epochs"])
        log(f"timing: {sec_per_batch:.3f} s/batch of {r['batch']}; {batches_per_epoch} batches/epoch; "
            f"recipe {r['epochs']} epochs ~{need / 60:.1f} min; using {epochs} epochs")
    info["epochs"] = int(epochs)
    steps_per_epoch = (len(ds) // r["batch"]) // r["accum"]
    total = max(1, steps_per_epoch * epochs)
    warm = max(1, int(total * r["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    step, checked, t_start = 0, False, time.time()
    for ep in range(epochs):
        ds.epoch = ep
        losses = []
        for i, (xb, yb) in enumerate(loader(ds, r["batch"], workers, True, seed + ep)):
            losses.append(step_batch(xb, yb).item())
            if (i + 1) % r["accum"]:
                continue
            if not checked:                             # LoRA must receive gradients (no silent frozen run)
                missing = [p for p in lora if p.grad is None]
                if missing or all(float(p.grad.abs().sum()) == 0.0 for p in lora[1::2]):
                    raise RuntimeError("LoRA parameters received no gradient")
                checked = True
            torch.nn.utils.clip_grad_norm_(lora + head, r["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        log(f"epoch {ep + 1}/{epochs}: loss {np.mean(losses):.4f} ({(time.time() - t_start) / 60:.1f} min)")
    model.eval()
    info["train_s"] = round(time.time() - t_start, 1)
    info["steps"] = step
    return info


def image_logits(model: Any, images: list[np.ndarray], boxes: list[list[tuple[int, int, int]]],
                 pre: E.Preprocess, spec: Mapping[str, Any], device: str, workers: int) -> np.ndarray:
    """Head applied to the mean CLS feature over each image's tiles x views (= mean-pooled features, as in 05)."""
    import torch

    ds = TileSet(images, boxes, int(RECIPE["eval_views"]), pre)
    sums = np.zeros((len(images), model.num_features))
    counts = np.zeros(len(images))
    with torch.no_grad():
        for xb, kb in loader(ds, 128, workers, False, 0):
            with autocast(device, spec):
                f = model(xb.to(device, non_blocking=True)).float().cpu().numpy()
            k = kb.numpy()
            np.add.at(sums, k, f)
            np.add.at(counts, k, 1)
    feats = torch.from_numpy(sums / counts[:, None]).float().to(device)
    with torch.no_grad():
        return model.head_ft(feats).cpu().numpy().astype(np.float64)


# ---- modes --------------------------------------------------------------------------------------------------------------
def save_head(model: Any, classes: Sequence[str], path: Path, **meta: Any) -> None:
    w = model.head_ft.weight.detach().cpu().numpy().astype(np.float64)
    b = model.head_ft.bias.detach().cpu().numpy().astype(np.float64)
    m = rm.LinearModel(classes, float("nan"), 1, None)
    m.median = m.mean = np.zeros(w.shape[1])
    m.scale = np.ones(w.shape[1])
    m.coef, m.intercept = w, b
    m.names = [f"{ENCODER}_probe_{j}" for j in range(w.shape[1])]
    m.to_npz(path, model=ENCODER, family="black_box", **meta)


def run(mode: str, cfg: Mapping[str, Any], root: Path, folds_arg: str | None, budget_min: float | None,
        workers: int, smoke_out: Path | None) -> dict[str, Any]:
    import torch

    smoke = mode == "smoke"
    device = "cuda" if torch.cuda.is_available() and not smoke else "cpu"
    ecfg = E.load_encoders_config()
    spec = E.encoder_spec(ecfg, ENCODER)
    classes = list(cfg["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    if smoke:
        rng = np.random.default_rng(0)
        n = 20
        train = pd.DataFrame({"case_id": [f"S{i:03d}" for i in range(n)], "label": [classes[i % 4] for i in range(n)],
                              "group": [f"g{i // 2}" for i in range(n)], "split": "train"})
        images = [np.clip(rng.normal(150 + 20 * (i % 4), 40, (288, 384, 3)), 0, 255).astype(np.uint8) for i in range(n)]
        masks = [np.ones((288 // MASK_DOWN, 384 // MASK_DOWN), bool) for _ in range(n)]
        tiles = None
        RECIPE.update(crop=160, epochs=1, crops_per_image=2, batch=8, accum=1, timing_batches=2)
        cvcfg = {**cfg, "split": {**cfg["split"], "cv": {**cfg["split"]["cv"], "folds": 2, "repeats": 1}}}
        out_dir = smoke_out or Path("finetune_smoke")
    else:
        cases = rm.read_cases(root)
        train = cases[cases["split"] == "train"].sort_values("case_id").reset_index(drop=True)
        tiles = rm.read_tiles(root)
        log(f"loading {len(train)} train images")
        images, masks = load_images(train["image_path"].tolist(), cfg, workers)
        cvcfg = cfg
        out_dir = root / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = train.sort_values("case_id").reset_index(drop=True)
    y = train["label"].map(cidx).to_numpy()
    pre = E.Preprocess.from_spec(spec)
    model, lora = build(spec, len(classes), device, smoke)
    log(f"model ready on {device}: {sum(p.numel() for p in lora) / 1e6:.2f}M LoRA params, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M total")
    budget = budget_min * 60 if budget_min else None
    summary: dict[str, Any] = {"mode": mode, "recipe": dict(RECIPE), "device": device, "n_train": len(train)}

    if mode in ("cv", "smoke"):
        splits = rm.cv_splits(train, cvcfg)[0]                       # repeat 0 of round 1's folds
        wanted = [int(f) for f in folds_arg.split(",")] if folds_arg else list(range(len(splits)))
        epochs = None
        per_fold_budget = budget / len(wanted) if budget else None
        for k in wanted:
            path = out_dir / f"cv_fold{k}.npz"
            if path.is_file():
                log(f"fold {k}: done already")
                continue
            held = splits[k]
            fit = np.setdiff1d(np.arange(len(train)), held)
            ds = CropSet([images[i] for i in fit], [masks[i] for i in fit], y[fit], pre, RECIPE, RECIPE["seed"] + k)
            info = fit_lora(model, lora, ds, spec, device, workers, per_fold_budget, epochs, RECIPE["seed"] + k)
            epochs = info["epochs"]                                     # every fold trains the same epochs
            ids = train["case_id"].to_numpy()[held]
            boxes = qc_boxes(ids, tiles, cfg, RECIPE["eval_grid"], images[0].shape[:2])
            logits = image_logits(model, [images[i] for i in held], boxes, pre, spec, device, workers)
            acc = float((logits.argmax(1) == y[held]).mean())
            np.savez(path, held=held, case_ids=ids, logits=logits, epochs=info["epochs"], acc=acc)
            log(f"fold {k}: held-out accuracy {acc:.3f} on {len(held)} images; {info}")
            summary[f"fold{k}"] = {**info, "held_acc": round(acc, 4)}
        files = [out_dir / f"cv_fold{k}.npz" for k in range(len(splits))]
        if all(f.is_file() for f in files):
            oof = np.full((len(train), len(classes)), np.nan)
            for f in files:
                z = np.load(f, allow_pickle=False)
                oof[z["held"]] = z["logits"]
            np.savez(out_dir / "cv_oof_r0.npz", case_ids=train["case_id"].to_numpy(), logits=oof, y=y)
            summary["oof_accuracy"] = round(float((oof.argmax(1) == y).mean()), 4)
            log(f"cv complete: OOF accuracy {summary['oof_accuracy']}")
    if mode in ("final", "smoke"):
        ds = CropSet(images, masks, y, pre, RECIPE, RECIPE["seed"] + 100)
        info = fit_lora(model, lora, ds, spec, device, workers, budget, None, RECIPE["seed"] + 100)
        weights = (out_dir / "uni2h_ft_lora.pt") if smoke else E.lora_weights_path(spec)
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not smoke:
            os.chmod(weights.parent, 0o700)
        torch.save({"lora": E.lora_state(model), "recipe": dict(RECIPE), "info": info}, weights)
        os.chmod(weights, 0o600)
        head_path = (out_dir / "uni2h_ft_head.npz") if smoke else root / cfg["risk_model"]["models_dir"] / "uni2h_ft_head.npz"
        save_head(model, classes, head_path, epochs=info["epochs"])
        summary["final"] = info
        log(f"final: LoRA weights -> {weights}; head -> {head_path}")
    io.write_json(out_dir / f"{mode}_summary.json", summary)
    return summary


def ingest(ctx: cli.StepContext) -> int:
    import importlib

    from core import calibration

    cfg, root = ctx.cfg, ctx.data_root
    d = root / "finetune"
    oof_path, fin_path = d / "cv_oof.csv", d / "final_logits.csv"
    if not (oof_path.is_file() and fin_path.is_file()):
        print(f"{STEP}: no fine-tune outputs in {d}; nothing to ingest")
        return 0
    step05 = importlib.import_module("pipeline.05_risk_model")
    classes = list(cfg["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    cases = rm.read_cases(root)
    labels = {r.case_id: cidx[r.label] for r in cases.itertuples()}
    train = cases[cases["split"] == "train"].sort_values("case_id").reset_index(drop=True)
    ids = train["case_id"].tolist()
    y = np.array([labels[i] for i in ids])
    split0 = [rm.cv_splits(train, cfg)[0]]
    oof = pd.read_csv(oof_path).set_index("case_id").loc[ids]
    fold = np.full(len(ids), -1)
    for k, held in enumerate(split0[0]):
        fold[held] = k
    if not (fold == oof["fold"].to_numpy()).all():
        raise SystemExit(f"{STEP}: cv_oof.csv folds differ from round 1's CV repeat 0")
    logits = oof[classes].to_numpy(np.float64)
    bounds = cfg["calibration"]["temperature_bounds"]
    ft = rm.CVResult(ENCODER, "black_box", {"C": 0.0}, logits[None],
                     rm.fold_calibrated_losses(logits[None], y, split0, bounds), len(cfg["risk_model"]["complexity"]),
                     int(E.encoder_spec(E.load_encoders_config(), ENCODER)["heads"]["probe"]["dim"]))
    tiles = rm.read_tiles(root)
    rank = {part: i for i, n in enumerate(cfg["risk_model"]["complexity"]) for part in str(n).split("|")}
    results = {ENCODER: ft}
    specs = cfg["risk_model"]["embedding_candidates"]
    for name in ("uni2h_lr", "uni2h512_lr"):
        fs = rm.embedding_features(root, name, specs[name], tiles, use_views=bool(cfg["risk_model"]["use_views"]))
        if fs is None:
            continue
        cand = step05.LRCandidate(name, "black_box", fs, cfg, rank[name])
        cand.run_cv(ids, labels, y, split0)
        results[name], _ = rm.one_se_choice(cand.results, len(split0[0]))
    chosen, info = rm.one_se_choice(list(results.values()), len(split0[0]))
    temperature = calibration.fit_temperature(logits, y, bounds)
    fin = pd.read_csv(fin_path).set_index("case_id")
    raw = pd.concat([oof[classes], fin[classes]])
    probs = pd.DataFrame(calibration.softmax(raw.to_numpy(np.float64), 1.0), index=raw.index, columns=classes)
    pred = io.read_csv(schemas.predictions_csv(root), keep_default_na=False, na_values=[""])
    for c in classes:
        pred[schemas.prob_col(ENCODER, c)] = pred["case_id"].map(probs[c])
    schemas.validate_frame(schemas.PREDICTIONS, pred)
    io.write_csv(schemas.predictions_csv(root), pred)

    def row(r: rm.CVResult) -> dict[str, Any]:
        top = r.oof_logits[0].argmax(1)
        recall = [float(np.mean(top[y == k] == k)) for k in range(len(classes))]
        return {"cv_log_loss_repeat0": r.mean_loss, "se": r.se(len(split0[0])), "config": r.config,
                "complexity": r.complexity, "oof_accuracy": float(np.mean(top == y)),
                "oof_balanced_accuracy": float(np.mean(recall))}

    summary = io.read_json(d / "cv_summary.json") if (d / "cv_summary.json").is_file() else {}
    out = {**step05.envelope("finetune", cfg, ctx.synthetic),
           "model": ENCODER, "recipe": dict(RECIPE), "protocol": "docs/ACCURACY_ROUND2.md: fold-wise LoRA on round 1's "
           "CV repeat 0; same fold-calibrated log-loss and one-SE rule as the probes, compared on the same folds",
           "cv_repeat0": {n: row(r) for n, r in results.items()}, "one_se": info, "rule_choice": chosen.name,
           "adopted_as_production": False, "temperature_from_oof": temperature, "n_train": len(ids),
           "n_final_reads": int(len(fin)), "gpu_summary": summary,
           "note": "uni2h_ft test rows are reported in metrics.json per_model; production changes only through the rule"}
    io.write_json(schemas.eval_json(ctx.artifacts_root, "finetune"), out)
    print(f"{STEP}: ingested {len(ids)} out-of-fold + {len(fin)} final reads; repeat-0 CV log-loss "
          + ", ".join(f"{n} {r.mean_loss:.3f}" for n, r in results.items()) + f"; rule picks {chosen.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "LoRA fine-tuning of UNI2-h (accuracy round 2).")
    parser.add_argument("mode", nargs="?", default="ingest", choices=("cv", "final", "smoke", "ingest"))
    parser.add_argument("--folds", default=None, help="cv: comma list of fold indices (default: all)")
    parser.add_argument("--max-minutes", type=float, default=None, help="training-time budget (cv: all folds)")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: CPUs - 2)")
    parser.add_argument("--out", default=None, help="smoke: output folder")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    if args.mode == "ingest":
        return ingest(ctx)
    workers = args.workers if args.workers is not None else max(0, (os.cpu_count() or 2) - 2)
    if args.mode == "smoke":
        workers = 0 if args.workers is None else args.workers
    summary = run(args.mode, ctx.cfg, ctx.data_root, args.folds, args.max_minutes, workers,
                  Path(args.out) if args.out else None)
    print(json.dumps({k: v for k, v in summary.items() if k != "recipe"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
