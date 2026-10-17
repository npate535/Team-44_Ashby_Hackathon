"""Our test-split macro-F1 and balanced accuracy beside the published THUNDER BACH probes (BUILD_PLAN §A4).

    python -m eval.benchmark_thunder

Our test split IS the THUNDER test split (patient-safe fixed ranges, §A2), so the numbers sit on the same images, but
the protocol differs (published: whole-image resize linear probes; ours: tiles + TTA + the selected model), which the
output says verbatim ("same split, different protocol"). No comparison is made to the 87% ICIAR challenge result,
which used a different, hidden test set.

Inputs: <artifacts>/eval/metrics.json (eval.metrics; run it first), optional <artifacts>/eval/zeroshot.json (Agent 2).
Reference numbers come from settings eval.thunder_reference (macro-F1, percent). Writes <artifacts>/eval/thunder.json.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

from core import cli, io, schemas
from eval.metrics import write_eval

STEP = "eval_benchmark_thunder"


def _pct(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not block:
        return None
    scale = lambda v: None if v is None else round(100.0 * float(v), 1)      # noqa: E731
    return {"estimate": scale(block.get("estimate")), "lo": scale(block.get("lo")), "hi": scale(block.get("hi")),
            "n": block.get("n"), "n_groups": block.get("n_groups")}


def payload(metrics: Mapping[str, Any], zeroshot: Mapping[str, Any] | None, cfg: Mapping[str, Any]) -> dict[str, Any]:
    ref = cfg["eval"]["thunder_reference"]
    prod = metrics["production"]
    ours = [{"model": metrics["models"]["production"], "role": "production (calibrated)",
             "macro_f1": _pct(prod["macro_f1"]), "balanced_accuracy": _pct(prod["balanced_accuracy"])}]
    for name, block in (metrics.get("per_model") or {}).items():
        ours.append({"model": name, "role": "evaluated model", "macro_f1": _pct(block["macro_f1"]),
                     "balanced_accuracy": _pct(block["balanced_accuracy"])})
    ours_zero = []
    if zeroshot:
        for name, block in (zeroshot.get("encoders") or zeroshot.get("models") or {}).items():
            if isinstance(block, Mapping) and "macro_f1" in block:
                ours_zero.append({"model": name, "macro_f1": _pct(block["macro_f1"]) if isinstance(block["macro_f1"], Mapping)
                                  else {"estimate": round(100 * float(block["macro_f1"]), 1)}})
    return {
        "split": "test", "n": metrics["n"], "n_groups": metrics["n_groups"],
        "label": ref["split_note"], "metric": ref["metric"], "source": ref["source"],
        "published": {"linear_probe": dict(ref["linear_probe"]), "zero_shot": dict(ref["zero_shot"])},
        "ours": ours, "ours_zero_shot": ours_zero,
        "notes": ["Same split, different protocol: published THUNDER probes resize the whole image; we embed tiles "
                  "with test-time augmentation and select models by grouped CV inside train.",
                  "Not compared with the 87% ICIAR 2018 challenge result, which used a different, hidden test set.",
                  "Published numbers carry no CI here; ours are cluster-bootstrap 95% CIs in percent."],
    }


def run(ctx: cli.StepContext) -> Any:
    art = ctx.artifacts_root
    metrics_path = schemas.eval_json(art, "metrics")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"run eval.metrics first: {metrics_path} is missing")
    metrics = io.read_json(metrics_path)
    zpath = schemas.eval_json(art, "zeroshot")
    zeroshot = io.read_json(zpath) if zpath.is_file() else None
    synthetic = ctx.synthetic or bool(metrics.get("synthetic"))
    out = write_eval(art, "thunder", payload(metrics, zeroshot, ctx.cfg), ctx.cfg, synthetic)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg,
                  inputs=[metrics_path] + ([zpath] if zeroshot else []), outputs=[out], synthetic=synthetic)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ctx = cli.context(cli.step_parser("eval.benchmark_thunder", "Comparison with published THUNDER BACH probes.").parse_args(argv))
    try:
        out = run(ctx)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"eval.benchmark_thunder: {exc}", file=sys.stderr)
        return 1
    print(f"eval.benchmark_thunder: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
