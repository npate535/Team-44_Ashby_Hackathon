"""Encode the concept prompt bank + class prompts + organizer NCT prompts -> text/<model>.npz (Task 3, plan §6.3).

    python -m pipeline.03_text [--encoders conch,musk,pathgenclip] [--device cpu] [--force]

Every core.schemas.expand_prompts prompt of config/concepts.yaml (concepts x phrasings x templates, then class
prompts) plus every resources/official/prompts_nct.json sentence (untemplated, ID schemas.nct_prompt_id) is encoded
by each text-aligned encoder (settings encoders.text_aligned) with the organizers' adapter encode_texts — the same
adapter whose image side 02_embed checks against — and written per core.schemas TEXT_NPZ:
ids [P], texts [P], emb float32 [P, D], model, synthetic.

Checks (exit 2): text dim == the encoder's align dim, finite, unit L2 norms (the adapters normalize; asserted, never
silently fixed), and — when emb/<model>/nct_align.npy exists — NCT zero-shot accuracy of the stored image
embeddings against these text embeddings above the encoders.yaml floor, which proves image and text share a space.
An existing npz whose ids and texts equal the current prompt list is up to date and skipped, so prompt edits
re-encode automatically. Runs as a CPU job (text_cpu.sbatch) so prompt edits cost no GPU time.

Delta-portable: Python >= 3.10, pytorch-conda/2.8 + organizer installs.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core import cli, config, io, schemas
from pipeline import encoders as E

STEP = "pipeline.03_text"
META_STEP = "03_text"
embed = importlib.import_module("pipeline.02_embed")


def prompt_table(bank: schemas.ConceptBank) -> tuple[list[str], list[str]]:
    """(ids, texts): expand_prompts order, then the NCT prompts in prompts_nct.json order."""
    prompts = schemas.expand_prompts(bank)
    ids, texts = [p.id for p in prompts], [p.text for p in prompts]
    for cls, text in embed.nct_prompts().items():
        ids.append(schemas.nct_prompt_id(cls))
        texts.append(text)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt ids")
    return ids, texts


def up_to_date(path: Path, ids: Sequence[str], texts: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    data = io.load_npz(path)
    return data["ids"].tolist() == list(ids) and data["texts"].tolist() == list(texts)


def check_text(name: str, spec: Mapping[str, Any], emb: np.ndarray, n: int, tol: float) -> None:
    dim = int(spec["heads"]["align"]["dim"])
    if emb.shape != (n, dim):
        raise embed.CheckFailed(f"{name}: text embeddings {emb.shape}, expected ({n}, {dim}) (align dim)")
    if not np.isfinite(emb).all():
        raise embed.CheckFailed(f"{name}: NaN or Inf in the text embeddings")
    dev = np.abs(np.linalg.norm(emb.astype(np.float64), axis=1) - 1.0).max()
    if dev > tol:
        raise embed.CheckFailed(f"{name}: text embeddings are not L2-normalized (max |norm-1| = {dev:.4g})")


def same_space_check(root: Path, name: str, spec: Mapping[str, Any], ids: Sequence[str], emb: np.ndarray,
                     ecfg: Mapping[str, Any]) -> dict[str, Any]:
    """NCT zero-shot of the stored image align embeddings with these text embeddings."""
    npy, idx_path = schemas.emb_npy(root, name, "nct", "align"), schemas.emb_index_csv(root, name, "nct", "align")
    if not io.outputs_exist(npy, idx_path):
        return {"status": "skipped", "reason": "no emb/<model>/nct_align.npy yet"}
    align = np.load(npy).astype(np.float32)
    labels = pd.read_csv(idx_path, keep_default_na=False, dtype={"label": str})["label"].to_numpy()
    classes = list(embed.nct_prompts())
    pos = {pid: i for i, pid in enumerate(ids)}
    text = emb[[pos[schemas.nct_prompt_id(c)] for c in classes]]
    keep = np.isin(labels, classes)
    floor = float(spec["nct_floor"])
    out: dict[str, Any] = {"n": int(keep.sum()), "floor": floor}
    if keep.sum() < int(ecfg["checks"]["nct_min_items"]):
        return {**out, "status": "skipped", "reason": f"only {int(keep.sum())} labeled NCT items"}
    if align.shape[1] != text.shape[1]:
        raise embed.CheckFailed(f"{name}: image align dim {align.shape[1]} != text dim {text.shape[1]}")
    acc = embed.zero_shot_accuracy(align[keep], labels[keep], text, classes)
    out.update(accuracy=round(acc, 4), status="pass" if acc >= floor else "fail")
    if acc < floor:
        raise embed.CheckFailed(f"{name}: NCT zero-shot with stored image embeddings {acc:.3f} < floor {floor}: "
                                f"text and image spaces disagree")
    return out


def main(argv: Sequence[str] | None = None, log: Any = print) -> int:
    parser = cli.step_parser(STEP, "Encode concept, class and NCT prompts per text-aligned encoder (plan §6.3).")
    parser.add_argument("--encoders", default=None, help="comma list (default settings encoders.text_aligned)")
    parser.add_argument("--device", default=None, help="cpu | cuda (default: cuda when available)")
    parser.add_argument("--concepts", default=None, help="concepts.yaml path (default settings concepts.file)")
    parser.add_argument("--encoders-config", default=None)
    parser.add_argument("--skip-unavailable", action="store_true", help="skip encoders whose weights are not cached")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    ecfg = E.load_encoders_config(args.encoders_config)
    bank = config.load_concept_bank(args.concepts, ctx.cfg)
    ids, texts = prompt_table(bank)
    names = embed.parse_list(args.encoders) or list(ctx.cfg["encoders"]["text_aligned"])
    for name in names:
        if not E.encoder_spec(ecfg, name)["text_aligned"]:
            log(f"ABORT: {name} is not text-aligned (encoders.yaml text_aligned: false)")
            return 2
    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict[str, Any] = {}
    outputs: list[Path] = []
    failed = False
    for name in names:
        spec = E.encoder_spec(ecfg, name)
        out = schemas.text_npz(ctx.data_root, name)
        if not ctx.force and up_to_date(out, ids, texts):
            log(f"[{name}] {out} is up to date ({len(ids)} prompts)")
            report[name] = {"status": "up to date", "n_prompts": len(ids)}
            continue
        try:
            enc = E.load_encoder(name, ecfg, device)
        except E.EncoderUnavailable as exc:
            if not args.skip_unavailable:
                log(f"ABORT: {exc}")
                return 2
            log(f"SKIP {name}: {exc}")
            report[name] = {"status": f"SKIPPED (unavailable): {exc}"}
            continue
        try:
            emb = enc.encode_texts(texts, batch=int(ecfg["engine"]["text_batch_size"])).astype(np.float32)
            check_text(name, spec, emb, len(ids), float(ecfg["checks"]["text_norm_tol"]))
            space = same_space_check(ctx.data_root, name, spec, ids, emb, ecfg)
        except embed.CheckFailed as exc:
            log(f"ABORT: check failed: {exc}")
            report[name] = {"status": f"FAILED: {exc}"}
            failed = True
            continue
        io.save_npz(out, ids=np.asarray(ids, dtype=str), texts=np.asarray(texts, dtype=str), emb=emb,
                    model=np.asarray(name), synthetic=np.asarray(bool(ctx.synthetic)))
        outputs.append(out)
        report[name] = {"status": "encoded", "n_prompts": len(ids), "dim": int(emb.shape[1]),
                        "revision": enc.info["revision"], "nct_same_space": space}
        log(f"[{name}] encoded {len(ids)} prompts -> {out}; same-space check {space}")
    log("SUMMARY " + json.dumps(report, default=str))
    io.write_meta(schemas.step_meta_json(ctx.data_root, META_STEP), step=META_STEP, cfg=ctx.cfg,
                  inputs=[str(config.resolve_repo_path(args.concepts or ctx.cfg["concepts"]["file"])),
                          str(E.PROMPTS_NCT)],
                  outputs=outputs, synthetic=ctx.synthetic,
                  args={"encoders": names, "device": device, "force": ctx.force}, extra={"report": report})
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
