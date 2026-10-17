"""Shared helpers for Agent 2's embedding tests: tiny generated (clearly non-tissue) image trees and schema-exact
embed manifests, plus a test copy of config/encoders.yaml with small chunks and no DataLoader workers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from core import schemas
from pipeline import encoders as E

embed = importlib.import_module("pipeline.02_embed")

TILE = 32  # source tile edge of the generated images


def noise_image(path: Path, w: int, h: int, seed: int) -> Path:
    """Random colour blocks: deterministic, obviously not tissue."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    arr = np.kron(blocks, np.ones((8, 8, 1), dtype=np.uint8))[:h, :w]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def manifest_row(set_name: str, case_id: str, split: str, tile: int | None, path: str, box: tuple[int, int, int, int],
                 view: int, *, grid: str = "g512", label: str = "", dtype: str = "", severity: int = 0,
                 group: str = "") -> dict[str, Any]:
    item = schemas.make_item_id(set_name, case_id, tile, dtype or None, severity or None)
    return {"row_id": f"{item}|v{view}", "item_id": item, "set": set_name, "case_id": case_id, "split": split,
            "grid": grid if tile is not None else "", "tile_idx": -1 if tile is None else tile, "path": path,
            "x": box[0], "y": box[1], "w": box[2], "h": box[3], "view": view, "degrade_type": dtype,
            "degrade_severity": severity, "degrade_seed": 7 if dtype else 0, "label": label, "group": group}


def build_tree(root: Path, *, train_views: int = 3, test_views: int = 2) -> pd.DataFrame:
    """2 train cases + 1 calibration + 1 test case (2x2 tiles each, crops of one 64x64 image), 6 NCT whole tiles
    (labels from prompts_nct.json classes), and 2 degraded tiles. Paths are data-root relative."""
    rows = []
    cases = [("BACH_0001", "train"), ("BACH_0002", "train"), ("BACH_0003", "calibration"), ("BACH_0004", "test")]
    for n, (case_id, split) in enumerate(cases):
        rel = f"src/{case_id}.png"
        noise_image(root / rel, 2 * TILE, 2 * TILE, seed=n)
        views = train_views if split == "train" else test_views
        for r in range(2):
            for c in range(2):
                for v in range(views):
                    rows.append(manifest_row("bach512", case_id, split, r * 2 + c, rel, (c * TILE, r * TILE, TILE, TILE), v))
    for k, cls in enumerate(["ADI", "LYM", "TUM", "STR", "DEB", "NORM"]):
        rel = f"nct/{cls}/{cls}-T{k}.png"
        noise_image(root / rel, TILE, TILE, seed=100 + k)
        rows.append(manifest_row("nct", f"{cls}-T{k}", "", None, rel, (0, 0, TILE, TILE), 0, label=cls))
    for t, dtype in enumerate(["blur", "jpeg"]):
        rows.append(manifest_row("deg512", "BACH_0004", "test", t, "src/BACH_0004.png", (t * TILE, 0, TILE, TILE), 0,
                                 dtype=dtype, severity=2))
    df = pd.DataFrame(rows, columns=[c.name for c in schemas.EMBED_MANIFEST.columns])
    schemas.validate_frame(schemas.EMBED_MANIFEST, df)
    df.to_csv(schemas.embed_manifest_csv(root), index=False)
    return df


def make_encoders_yaml(tmp: Path, *, chunk_rows: int = 4, keep_chunks: bool = False, **test_overrides: Any) -> Path:
    data = yaml.safe_load(E.ENCODERS_YAML.read_text())
    data["engine"].update(chunk_rows=chunk_rows, num_workers=0, keep_chunks=keep_chunks)
    data["checks"]["nct_min_items"] = test_overrides.pop("nct_min_items", data["checks"]["nct_min_items"])
    data["encoders"]["test"].update(test_overrides)
    path = tmp / "encoders_test.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def run_embed(root: Path, ecfg_path: Path, *extra: str) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = embed.main(["--data-root", str(root), "--artifacts-root", str(root / "artifacts"), "--encoders", "test",
                       "--device", "cpu", "--encoders-config", str(ecfg_path), *extra], log=lines.append)
    return code, lines


def expected_test_embedding(root: Path, row: pd.Series, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """What the official test adapter gives for one manifest row (crop -> view -> 8x8 resize -> channel means)."""
    with Image.open(schemas.resolve(root, row["path"])) as handle:
        image = handle.convert("RGB")
    tile = E.crop_box(image, row["x"], row["y"], row["w"], row["h"])
    if row["degrade_type"]:
        from pipeline import degrade

        tile = degrade.apply(tile, row["degrade_type"], row["degrade_severity"], row["degrade_seed"], cfg)
    tile = E.dihedral(tile, int(row["view"]))
    arr = np.asarray(tile.resize((8, 8), Image.BILINEAR)).astype("float32") / 255.0
    return arr.mean(axis=(0, 1))
