"""Human reader baseline next to Sepia, on the same four BACH classes.

    python -m eval.human_baseline

Pathologist numbers are parsed from the verified figures in config/literature.yaml (Aresta et al. 2019, Table 4:
four pathologists reading the 100 hidden BACH test images; Elmore et al. 2015, B-Path category concordance), so no
human number is typed here. Sepia numbers are per-class recall and accuracy on the held-out THUNDER test split, with
95% patient-grouped bootstrap intervals. The images differ (hidden test set vs THUNDER test split), so this is a
reference, not a head-to-head comparison; docs/HUMAN_BASELINE.md lists every difference.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core import cli, io, schemas, stats
from eval.metrics import bootstrap_settings, write_eval

STEP = "eval_human_baseline"
LITERATURE = Path(__file__).resolve().parents[1] / "config" / "literature.yaml"
BACH_KEY = "aresta2019_bach"
BPATH_KEY = "elmore2015_bpath"
SE_NAMES = {"Normal": "Normal", "Benign": "Benign", "In situ": "InSitu", "Invasive": "Invasive"}


def _figures(lit: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    for c in lit["citations"]:
        if c["key"] == key:
            return c.get("figures") or []
    raise KeyError(f"{key} not in config/literature.yaml")


def parse_bach_readers(figures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-class sensitivity and accuracy of the four BACH pathologists (E1-E4) from the Table 4 figures."""
    se: dict[str, list[float]] = {}
    acc: list[float] | None = None
    for f in figures:
        value = str(f["value"])
        for name, cls in SE_NAMES.items():
            m = re.search(rf"{name} Se ([0-9.]+(?:/[0-9.]+){{3}})", value)
            if m:
                se[cls] = [float(x) for x in m.group(1).split("/")]
        m = re.match(r"Accuracy (\d+)% \(E1[^)]*\), (\d+)%, (\d+)%, (\d+)%", value)
        if m:
            acc = [int(x) / 100 for x in m.groups()]
    if set(se) != set(SE_NAMES.values()) or acc is None:
        raise ValueError(f"could not parse all BACH reader figures (classes {sorted(se)}, accuracy {acc})")
    return {"sensitivity": se, "accuracy": acc, "readers": ["E1", "E2", "E3", "E4"],
            "note": "E1 co-labelled BACH and re-read after one month; E2-E4 are external. Readers saw only the field "
                    "image. Hidden test set: 100 images, 25 per class."}


def parse_bpath(figures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """B-Path concordance with the expert consensus, by reference category."""
    out = {}
    for f in figures:
        m = re.match(r"(\d+)% \((\d+)-(\d+)%\) concordant", str(f["value"]))
        if not m:
            continue
        meaning = str(f["meaning"]).lower()
        for cat in ("invasive", "dcis", "atypia", "benign"):
            if meaning.startswith(cat):
                out[cat] = {"concordance": int(m.group(1)) / 100, "lo": int(m.group(2)) / 100,
                            "hi": int(m.group(3)) / 100}
    if set(out) != {"invasive", "dcis", "atypia", "benign"}:
        raise ValueError(f"could not parse all B-Path categories: {sorted(out)}")
    return out


def sepia_rates(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, classes: Sequence[str],
                boot: Mapping[str, Any]) -> dict[str, Any]:
    def recall(k: int):
        return lambda t, p: float(np.mean(p[t == k] == k)) if np.any(t == k) else float("nan")

    out = {"recall": {}}
    for k, c in enumerate(classes):
        block = stats.bootstrap_ci(recall(k), [y, pred], groups, None, boot["n_boot"], boot["ci"], boot["seed"])
        block["correct"], block["n_class"] = int(np.sum((y == k) & (pred == k))), int(np.sum(y == k))
        out["recall"][c] = block
    return out


def run(ctx: cli.StepContext) -> Path:
    cfg = ctx.cfg
    classes = list(cfg["classes"])
    lit = yaml.safe_load(LITERATURE.read_text())
    readers = parse_bach_readers(_figures(lit, BACH_KEY))
    bpath = parse_bpath(_figures(lit, BPATH_KEY))
    cases = pd.read_csv(schemas.cases_csv(ctx.data_root))
    preds = pd.read_csv(schemas.predictions_csv(ctx.data_root), low_memory=False)
    test = cases[cases["split"] == "test"].set_index("case_id")
    rows = preds.set_index("case_id").loc[test.index]
    y = np.array([classes.index(v) for v in test["label"]])
    pred = np.array([classes.index(v) for v in rows["top"]])
    sepia = sepia_rates(y, pred, test["group"].astype(str).to_numpy(), classes, bootstrap_settings(cfg))
    sepia["accuracy"] = io.read_json(ctx.artifacts_root / "eval" / "metrics.json")["production"]["accuracy"]
    prod = io.read_json(ctx.artifacts_root / "eval" / "model_choice.json")["production"]
    payload = {
        "production": prod, "n_test": int(len(y)), "n_groups": int(test["group"].nunique()),
        "bach_readers": readers, "bpath": bpath, "sepia": sepia,
        "citations": {"bach_readers": "Aresta G, et al. Med Image Anal. 2019;56:122-139 (Table 4)",
                      "bpath": "Elmore JG, et al. JAMA. 2015;313(11):1122-1132 (Table 3)"},
        "not_comparable": [
            "Different images: the BACH readers read the 100 hidden test images; Sepia is scored on the 132-image "
            "THUNDER test split of the public images.",
            "BACH labels are curated (Normal vs Benign disagreements discarded; doubtful cases confirmed by "
            "immunohistochemistry), so the hardest cases are under-represented.",
            "B-Path readers read whole glass slides of real biopsies, including atypia, which BACH does not contain.",
            "Pathologist ranges are the spread of four readers, not confidence intervals.",
        ],
    }
    return write_eval(ctx.artifacts_root, "human_baseline", payload, cfg, ctx.synthetic)


def main(argv: Sequence[str] | None = None) -> int:
    args = cli.step_parser("eval.human_baseline", "Human reader baseline next to Sepia.").parse_args(argv)
    out = run(cli.context(args))
    print(f"eval.human_baseline: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
