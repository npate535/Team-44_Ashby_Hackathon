"""Synthetic data root in exactly the core.schemas formats (BUILD_PLAN Task 3 `make_synthetic`; Agent 3 CP2).

    python -m pipeline.make_synthetic --out data_synthetic [--seed N] [--with-cells] [--small] [--force]

Writes a complete fake data root that every laptop step and the app can integrate against before real Delta data
exists. Everything is stamped synthetic: the core.cli.SYNTHETIC_MARKER file, `synthetic: true` in every JSON/meta,
`source = synthetic` in cases.csv. Images are generated, clearly non-tissue pictures (colored blocks + "SYNTHETIC").

Planted structure (so downstream metrics are non-trivial and testable). One latent tissue model drives everything:
  * Each 256-px tile has a latent vector = 9 concept activations (settings concepts.ids) + 6 nuisance dims (stain /
    scanner). Lesional tiles follow their case class profile, others a benign-stroma profile, in spatially coherent
    blobs; background tiles (low tissue_frac, QC fail) carry no concept signal. 512-px tiles average their 4
    sub-tiles, so both grids, composites and margin segments are geometrically consistent.
  * Case and patient-group shifts make tiles of a case (and cases of a patient) correlated; ~15% of Invasive cases
    have InSitu-like lesions, so InSitu<->Invasive stays genuinely confusable. `mitoses` is a deliberately weak
    concept (small class differences, extra noise).
  * Every encoder maps the latent linearly to its own space (its own seeded matrices, noise level and dims, never
    comparable across encoders). MUSK shares the latent with CONCH but is noisier and weaker on invasion vs
    duct-confined, so the two agree often but not always.
  * Text embeddings (text-aligned encoders) are the encoder's own image directions of each concept: a concept's
    positive prompts point along its latent column, contrast prompts along the generic tissue profile, so concept
    scores are highest in their expected class. Class and NCT prompts follow their class profiles.
  * Degradations shrink the concept signal and push embeddings off-manifold along a per-type direction, growing with
    severity. Composite fields take the carcinoma case's tiles in their carcinoma half and the benign case's tiles in
    the other half (carcinoma signal only in that half's tiles, so a localizer beats an echo).
  * WSI tiles are labeled with their region class (patient groups, scanner shift); WSI margin windows plant tumor at
    a named edge, centrally, or not at all, with per-margin truth from the 256-px strips. Quilt/PathGen images sit
    along concept directions (pos) or away from them (neg), grouped by video/slide. NCT classes follow their concepts
    plus a cross-organ shift. Cell features are computed from generated nuclei whose type mix and nuclear size vary
    by class in the CELLS_SPEC expected directions.

Choices a consumer should know (also listed in agent3.md):
  * embed_manifest.csv and emb/* list EVERY tile (QC pass and fail); consumers filter with tiles.csv qc_pass.
  * deg512/deg256 items keep the SOURCE case_id plus degrade_type/severity; the variant ID is
    core.schemas.variant_case_id(case_id, type, severity). comp*/wsi_margin256 items use the composite / window ID.
  * Ruling M9: degraded sets are embedded at view 0 only (degrade.views when that key exists, else 1); bach512/bach256
    `_views` files cover every split (view-0 clean rows are the like-for-like baseline for degraded reads).
  * Case IDs are "SYN_<n:04d>" (seeded permutation), so synthetic IDs can never collide with real BACH IDs.
  * image_path points at the display JPEG (no full-size source images are generated). Every manifest `path` exists
    and its crop box fits the file: tile PNGs are full-size flat squares, composites and WSI windows are full-size
    (2048x1536) block pictures cropped per tile, whole-image sets (NCT, Quilt, PathGen, WSI tiles) are small PNGs
    under aux/<set>/ with (0, 0, width, height).

Laptop only (numpy, pandas, Pillow, PyYAML via core.config).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import zlib
from io import BytesIO
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

from core import cli, config, io, schemas

STEP = "make_synthetic"
ID_PREFIX = "SYN"
N_NUISANCE = 6

# ---- generator parameters (synthetic only; class/concept names are checked against settings) --------------------
# Mean concept activation of lesional tiles per class (keys: settings classes x concepts.ids).
CLASS_PROFILE: dict[str, dict[str, float]] = {
    "Normal":   {"tubules": 1.6, "pleomorphism": 0.0, "mitoses": 0.0, "invasion": 0.0, "duct_confined": 0.2,
                 "normal_lobules": 2.0, "benign_stroma_fat": 0.9, "lymphocytes": 0.3, "necrosis": 0.0},
    "Benign":   {"tubules": 1.2, "pleomorphism": 0.3, "mitoses": 0.0, "invasion": 0.1, "duct_confined": 0.5,
                 "normal_lobules": 0.8, "benign_stroma_fat": 2.0, "lymphocytes": 0.4, "necrosis": 0.0},
    "InSitu":   {"tubules": 0.3, "pleomorphism": 1.5, "mitoses": 0.2, "invasion": 0.6, "duct_confined": 2.0,
                 "normal_lobules": 0.2, "benign_stroma_fat": 0.5, "lymphocytes": 0.6, "necrosis": 0.3},
    "Invasive": {"tubules": 0.0, "pleomorphism": 1.8, "mitoses": 0.35, "invasion": 2.0, "duct_confined": 0.6,
                 "normal_lobules": 0.0, "benign_stroma_fat": 0.4, "lymphocytes": 0.8, "necrosis": 0.8},
}
STROMA_PROFILE = {"tubules": 0.2, "pleomorphism": 0.0, "mitoses": 0.0, "invasion": 0.0, "duct_confined": 0.0,
                  "normal_lobules": 0.1, "benign_stroma_fat": 1.6, "lymphocytes": 0.3, "necrosis": 0.0}
CONCEPT_NOISE = {"mitoses": 1.2}          # extra-noisy (weak) concepts; others use TILE_NOISE
TILE_NOISE = 0.7
CASE_SHIFT, GROUP_SHIFT = 0.45, 0.3
LESIONAL_FRACTION = {"Normal": 0.85, "Benign": 0.85, "InSitu": 0.6, "Invasive": 0.65}
INSITU_LIKE_INVASIVE = 0.15
BACKGROUND_FRACTION = 0.1
NCT_PROFILE: dict[str, dict[str, float]] = {
    "ADI": {"benign_stroma_fat": 2.2}, "BACK": {}, "DEB": {"necrosis": 2.4}, "LYM": {"lymphocytes": 2.6},
    "MUC": {"tubules": 0.4}, "MUS": {"benign_stroma_fat": 1.0}, "NORM": {"normal_lobules": 1.0, "tubules": 1.3},
    "STR": {"benign_stroma_fat": 1.9}, "TUM": {"pleomorphism": 2.2, "invasion": 1.0},
}
# Encoder: noise sd, (align dim or None, probe dim), weakening factor on invasion/duct_confined.
ENCODER_SPEC: dict[str, tuple[float, tuple[int | None, int], float]] = {
    "conch": (0.35, (32, 48), 1.0), "musk": (0.85, (24, 40), 0.5), "uni2h": (0.30, (None, 44), 1.0),
    "hoptimus0": (0.32, (None, 52), 1.0), "virchow2": (0.38, (None, 40), 1.0), "phikonv2": (0.45, (None, 36), 1.0),
    "pathgenclip": (0.60, (32, 36), 0.8),
}
VIEW_NOISE = 0.12
DEGRADED_SETS = ("deg512", "deg256")
VIEWS_FILE_SETS = ("bach512", "bach256")
CELL_FEATURES_BUILTIN = ("neo_fraction", "neo_density", "nuc_area_mean", "nuc_area_cv", "nuc_shape_cv",
                         "infl_fraction", "til_proximity", "dead_fraction", "epi_fraction")
CELL_MIX = {  # type probabilities (neoplastic, inflammatory, connective, dead, epithelial) and neo area mean/cv
    "Invasive": ((0.55, 0.15, 0.15, 0.07, 0.08), 58.0, 0.45),
    "InSitu": ((0.50, 0.10, 0.20, 0.03, 0.17), 52.0, 0.35),
    "Benign": ((0.08, 0.12, 0.40, 0.01, 0.39), 36.0, 0.20),
    "Normal": ((0.03, 0.10, 0.40, 0.01, 0.46), 34.0, 0.18),
    "stroma": ((0.05, 0.20, 0.65, 0.02, 0.08), 34.0, 0.18),
}


# ---- small helpers ------------------------------------------------------------------------------------------------------
def _rng(seed: int, *keys: str | int) -> np.random.Generator:
    parts = [int(seed)] + [zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng(parts)


def _vec(profile: Mapping[str, float], concept_ids: Sequence[str]) -> np.ndarray:
    return np.array([float(profile.get(c, 0.0)) for c in concept_ids])


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def _write_frame(root: Path, table: schemas.Table, df: pd.DataFrame, path: Path | None = None) -> Path:
    schemas.validate_frame(table, df)
    return io.write_csv(path or (root / table.filename), df)


class World:
    """All planted latents for one seed; images/CSVs/embeddings are rendered from it."""

    def __init__(self, cfg: Mapping[str, Any], seed: int, small: bool):
        self.cfg, self.seed, self.small = cfg, int(seed), bool(small)
        self.classes = list(cfg["classes"])
        self.concepts = list(cfg["concepts"]["ids"])
        missing = [c for c in self.classes if c not in CLASS_PROFILE]
        unknown = [c for c in self.concepts if c not in STROMA_PROFILE]
        if missing or unknown:
            raise ValueError(f"synthetic profiles do not cover classes {missing} / concepts {unknown}")
        self.n_lat = len(self.concepts) + N_NUISANCE
        self.g512 = cfg["tiles"]["grids"]["g512"]
        self.g256 = cfg["tiles"]["grids"]["g256"]
        if (self.g256["rows"], self.g256["cols"]) != (2 * self.g512["rows"], 2 * self.g512["cols"]):
            raise ValueError("synthetic generator expects g256 to be exactly a 2x refinement of g512")
        self.width, self.height = (int(v) for v in cfg["dataset"]["image_size"])
        self.mean_profile = np.mean([_vec(CLASS_PROFILE[c], self.concepts) for c in self.classes], axis=0)
        self.noise = np.array([CONCEPT_NOISE.get(c, TILE_NOISE) for c in self.concepts])
        self._build_cases()
        self._build_tiles()

    # -- cases ----------------------------------------------------------------------------------------------------------
    def _build_cases(self) -> None:
        cfg = self.cfg
        n = int(cfg["dataset"]["n_images"])
        order = list(cfg["split"]["class_order"])
        per_class = n // len(order)
        if per_class * len(order) != n:
            raise ValueError("dataset.n_images must split evenly over split.class_order")
        splits = config.split_assignments(cfg)
        perm = _rng(self.seed, "ids").permutation(n) + 1
        rows = []
        for i in range(n):
            label = order[i // per_class]
            cid = f"{ID_PREFIX}_{perm[i]:04d}"
            rows.append({"case_id": cid, "source": "synthetic",
                         "source_file": f"synthetic/{label}/{label[0].lower()}{i % per_class + 1:03d}.tif",
                         "source_index": i, "image_path": f"display/{cid}.jpg", "label": label, "split": splits[i],
                         "patient_id": "", "group": "", "lane": "biopsy", "width": self.width,
                         "height": self.height, "display_path": f"display/{cid}.jpg"})
        cases = pd.DataFrame(rows)
        self._assign_patients(cases)
        self._assign_lanes(cases)
        rng = _rng(self.seed, "insitu_like")
        inv = cases.index[cases.label == cfg["aggressive_class"]]
        self.insitu_like = set(cases.loc[inv[rng.uniform(size=len(inv)) < INSITU_LIKE_INVASIVE], "case_id"])
        # three insufficient-tissue cases: one train, one calibration, one test-biopsy
        rng = _rng(self.seed, "insufficient")
        picks = []
        for split, lane in (("train", "biopsy"), ("calibration", "biopsy"), ("test", "biopsy")):
            pool = cases[(cases.split == split) & (cases.lane == lane)].case_id.to_numpy()
            picks.append(str(rng.choice(np.sort(pool))))
        self.insufficient = set(picks)
        self.cases = cases.sort_values("case_id").reset_index(drop=True)

    def _assign_patients(self, cases: pd.DataFrame) -> None:
        """~36% of images get patient IDs (like the BACH xlsx: 145 of 400); patients stay inside one split but may
        span classes. Others are grouped by contiguous source_index blocks (split.cv.block_size)."""
        rng = _rng(self.seed, "patients")
        block = int(self.cfg["split"]["cv"]["block_size"])
        cases["group"] = [f"b{i // block}" for i in cases.source_index]
        pid = 1
        for split in schemas.SPLITS:
            idx = cases.index[cases.split == split].to_numpy()
            known = np.sort(rng.choice(idx, size=int(round(len(idx) * 145 / 400)), replace=False))
            rng.shuffle(known)
            pos = 0
            while pos < len(known):
                size = int(min(len(known) - pos, rng.choice([1, 1, 2, 2, 3, 4, 6, 8])))
                for j in known[pos:pos + size]:
                    cases.at[j, "patient_id"] = str(pid)
                    cases.at[j, "group"] = f"p{pid}"
                pid += 1
                pos += size

    def _assign_lanes(self, cases: pd.DataFrame) -> None:
        lanes = self.cfg["lanes"]
        rng = _rng(int(lanes["seed"]) + self.seed, "lanes")
        test = cases[cases.split == lanes["surgical_source_split"]]
        n_surg = int(round(len(test) * float(lanes["surgical_fraction"])))
        chosen: list[int] = []
        for group in lanes["surgical_must_cover"]:
            pool = np.sort(test.index[test.label.isin(group)].difference(chosen).to_numpy())
            chosen.append(int(rng.choice(pool)))
        rest = np.sort(test.index.difference(chosen).to_numpy())
        chosen += [int(i) for i in rng.choice(rest, size=max(0, n_surg - len(chosen)), replace=False)]
        cases.loc[chosen, "lane"] = "surgical"

    # -- tile latents ---------------------------------------------------------------------------------------------------
    def _lesion_mask(self, rng: np.random.Generator, frac: float) -> np.ndarray:
        r, c = self.g256["rows"], self.g256["cols"]
        field = rng.normal(size=(r + 2, c + 2))
        smooth = sum(field[i:i + r, j:j + c] for i in range(3) for j in range(3)) / 9.0
        return smooth >= np.quantile(smooth, 1.0 - frac)

    def _build_tiles(self) -> None:
        r, c, k = self.g256["rows"], self.g256["cols"], len(self.concepts)
        group_shift = {g: _rng(self.seed, "group", g).normal(scale=GROUP_SHIFT, size=k)
                       for g in sorted(self.cases.group.unique())}
        lat = np.zeros((len(self.cases), r, c, self.n_lat))
        tissue = np.zeros((len(self.cases), r, c))
        tile_class = np.empty((len(self.cases), r, c), dtype=object)
        stroma = _vec(STROMA_PROFILE, self.concepts)
        for n, row in self.cases.iterrows():
            rng = _rng(self.seed, "case", row.case_id)
            label = row.label
            profile = _vec(CLASS_PROFILE[label], self.concepts)
            if row.case_id in self.insitu_like:
                profile = 0.5 * profile + 0.5 * _vec(CLASS_PROFILE["InSitu"], self.concepts)
            lesional = self._lesion_mask(rng, LESIONAL_FRACTION[label])
            n_bg = int(round(BACKGROUND_FRACTION * r * c))
            if row.case_id in self.insufficient:
                n_bg = r * c - 8                     # leaves 2 QC-passing 512 tiles (< qc.min_tiles_kept)
            background = np.zeros(r * c, dtype=bool)
            if row.case_id in self.insufficient:
                keep = [(a, b) for a in (0, 1) for b in range(4)]          # 512 tiles 0 and 1 (top-left)
                background[:] = True
                for rr, cc in keep:
                    background[rr * c + cc] = False
            else:
                border = [i for i in range(r * c) if i // c in (0, r - 1) or i % c in (0, c - 1)]
                background[rng.choice(border, size=n_bg, replace=False)] = True
            background = background.reshape(r, c)
            case_shift = rng.normal(scale=CASE_SHIFT, size=k) + group_shift[row.group]
            stain = rng.normal(size=N_NUISANCE)
            for i in range(r):
                for j in range(c):
                    if background[i, j]:
                        concepts = rng.normal(scale=0.2, size=k)
                        tile_class[n, i, j] = "background"
                        tissue[n, i, j] = rng.beta(1.5, 8.0)
                    else:
                        base = profile if lesional[i, j] else stroma
                        concepts = base + case_shift + rng.normal(size=k) * self.noise
                        tile_class[n, i, j] = label if lesional[i, j] else "stroma"
                        tissue[n, i, j] = rng.beta(9.0, 2.0)
                    lat[n, i, j] = np.concatenate([concepts, stain + rng.normal(scale=0.3, size=N_NUISANCE)])
        self.lat256 = lat
        self.tissue256 = tissue
        self.class256 = tile_class
        self.case_row = {cid: i for i, cid in enumerate(self.cases.case_id)}

    @staticmethod
    def pool512(grid: np.ndarray) -> np.ndarray:
        """(..., 2R, 2C, D) -> (..., R, C, D) by averaging 2x2 sub-tiles."""
        return (grid[..., 0::2, 0::2, :] + grid[..., 1::2, 0::2, :] + grid[..., 0::2, 1::2, :]
                + grid[..., 1::2, 1::2, :]) / 4.0

    def lat512(self, n: int) -> np.ndarray:
        return self.pool512(self.lat256[n])

    def tissue512(self, n: int) -> np.ndarray:
        return self.pool512(self.tissue256[n][..., None])[..., 0]

    def degrade_latent(self, lat: np.ndarray, severity: int) -> np.ndarray:
        s = 0.15 * severity
        out = lat.copy()
        out[..., : len(self.concepts)] = (1 - s) * lat[..., : len(self.concepts)] + s * self.mean_profile
        return out


class Encoder:
    """Seeded linear map latent -> embedding for one encoder and head."""

    def __init__(self, world: World, name: str, head: str):
        noise, dims, weak = ENCODER_SPEC[name]
        dim = dims[0] if head == "align" else dims[1]
        if dim is None:
            raise ValueError(f"{name} has no {head} head")
        self.name, self.head, self.dim, self.noise = name, head, int(dim), float(noise)
        rng = _rng(world.seed, "encoder", name, head)
        self.w = rng.normal(size=(world.n_lat, self.dim)) / math.sqrt(world.n_lat)
        self.w[len(world.concepts):] *= 0.6                     # nuisance dims matter less than tissue
        scale = np.ones(world.n_lat)
        for c in ("invasion", "duct_confined"):
            if c in world.concepts:
                scale[world.concepts.index(c)] = weak
        self.w *= scale[:, None]
        self.bias = rng.normal(scale=0.3, size=self.dim)
        self.deg_dirs = {t: _normalize(rng.normal(size=self.dim)) for t in schemas.DEGRADE_TYPES}
        self.world = world

    def concept_direction(self, concept_vec: np.ndarray) -> np.ndarray:
        lat = np.concatenate([concept_vec, np.zeros(N_NUISANCE)])
        return lat @ self.w

    def embed(self, lat: np.ndarray, rng: np.random.Generator, n_views: int, degrade: tuple[str, int] | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """lat (N, n_lat) -> (per-item TTA mean (N, D), views (N, V, D)). Align heads are unit-norm per view and
        re-normalized after averaging (core.schemas.EMB_NPY)."""
        base = lat @ self.w + self.bias + rng.normal(scale=self.noise, size=(len(lat), self.dim))
        if degrade is not None:
            dtype, sev = degrade
            radius = float(np.linalg.norm(base.std(axis=0))) if len(base) > 1 else 1.0   # typical spread of rows
            base = base + 0.5 * sev * radius * self.deg_dirs[dtype] + rng.normal(scale=0.15 * sev, size=base.shape)
        views = base[:, None, :] + rng.normal(scale=VIEW_NOISE, size=(len(lat), n_views, self.dim))
        if self.head == "align":
            views = _normalize(views)
            return _normalize(views.mean(axis=1)), views
        return views.mean(axis=1), views


# ---- concept bank ---------------------------------------------------------------------------------------------------------
def builtin_bank(cfg: Mapping[str, Any]) -> schemas.ConceptBank:
    """A schema-valid synthetic concept bank (used only when config/concepts.yaml does not exist yet)."""
    group_of = {cid: name for name, spec in cfg["concepts"]["groups"].items() for cid in spec["ids"]}
    n_pos, n_con = int(cfg["concepts"]["positives"][0]), int(cfg["concepts"]["contrasts"][0])
    data = {
        "templates": ["a synthetic image of {}.", "synthetic tissue showing {}."],
        "concepts": [{"id": cid, "group": group_of[cid], "label": f"Synthetic {cid.replace('_', ' ')}",
                      "direction": "context" if cid == "lymphocytes" else "up", "weak": cid == "mitoses",
                      "positives": [f"synthetic {cid} pattern {i}" for i in range(n_pos)],
                      "contrasts": [f"synthetic contrast to {cid} {i}" for i in range(n_con)]}
                     for cid in cfg["concepts"]["ids"]],
        "class_prompts": {cls: [f"synthetic {cls} tissue"] for cls in cfg["classes"]},
        "version": "synthetic",
    }
    return schemas.validate_concept_bank(data, cfg)


def concept_bank(cfg: Mapping[str, Any]) -> tuple[schemas.ConceptBank, str]:
    path = config.resolve_repo_path(cfg["concepts"]["file"])
    if path.is_file():
        return config.load_concept_bank(path, cfg), str(cfg["concepts"]["file"])
    return builtin_bank(cfg), "builtin_synthetic_bank"


def nct_prompts() -> dict[str, str]:
    path = io.REPO_ROOT / "resources" / "official" / "prompts_nct.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for cls, value in data.items():
        out[cls] = value if isinstance(value, str) else (value[0] if value else cls)
    return out


# ---- the generator ----------------------------------------------------------------------------------------------------------
class Generator:
    def __init__(self, cfg: Mapping[str, Any], out: Path, seed: int, small: bool, with_cells: bool,
                 build: Mapping[str, Any] | None = None):
        self.cfg, self.out, self.seed, self.small, self.with_cells = cfg, out, int(seed), small, with_cells
        self.build = dict(build or build_key(cfg, seed, small, with_cells))
        self.world = World(cfg, seed, small)
        self.mpp = float(cfg["dataset"]["mpp"])
        self.views = {g: int(v) for g, v in cfg["encoders"]["views"].items()}
        self.outputs: list[Path] = []
        self.n_nct = 20 if small else int(cfg["nct"]["per_class"])
        self.n_comp = 8 if small else int(cfg["composites"]["n"])
        self.n_quilt = 6 if small else 30
        self.n_windows = 8 if small else 24
        self.n_wsi = 10
        self.wsi_tiles = 8 if small else 40
        self.display_long = 384 if small else int(cfg["tiles"]["display"]["long_side"])
        self.aux_px = {"nct": 224, "quilt": 256, "pathgen": 256, "wsi256": 256, "wsi512": 512}
        self._png_cache: dict[tuple[Any, ...], bytes] = {}

    # -- driver ----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        marker = self.out / cli.SYNTHETIC_MARKER
        self._emit(io.write_json(marker, {"synthetic": True, "generator": "pipeline.make_synthetic", "seed": self.seed,
                                          "small": self.small, "with_cells": self.with_cells,
                                          "build": self.build, "note": "Synthetic data root: generated non-tissue images and planted "
                                                  "embeddings. Never report its numbers as results."}))
        self.write_inventory()
        self.write_cases_tiles()
        self.write_images()
        self.write_composites()
        self.write_wsi()
        self.write_aux_items()
        self.write_manifest()
        self.write_embeddings()
        self.write_text()
        if self.with_cells:
            self.write_cells()
        io.write_meta(schemas.step_meta_json(self.out, STEP), step=STEP, cfg=self.cfg, inputs=[],
                      outputs=[p.relative_to(self.out).as_posix() for p in self.outputs[:200]],
                      args={"seed": self.seed, "small": self.small, "with_cells": self.with_cells}, synthetic=True,
                      extra={"n_outputs": len(self.outputs)})

    def _emit(self, path: Path) -> Path:
        self.outputs.append(Path(path))
        return Path(path)

    # -- inventory / cases / tiles ------------------------------------------------------------------------------------------
    def write_inventory(self) -> None:
        cases = self.world.cases
        self._emit(io.write_json(schemas.inventory_json(self.out), {
            "created_utc": io.utc_now(), "synthetic": True, schemas.INVENTORY_MPP_KEY: self.mpp,
            "mpp_source": "synthetic (settings dataset.mpp)", "label_source": "synthetic generator",
            "n_images": int(len(cases)), "per_class_counts": {c: int((cases.label == c).sum()) for c in self.world.classes},
            "image_size": [self.world.width, self.world.height], "bach_root": "synthetic",
            "patient_file": "synthetic", "wsi": {"n_svs": self.n_wsi, "n_xml": self.n_wsi},
            "pathmmu_bach": {"note": "synthetic root: no comparison"}, "gate_passed": True,
        }))

    def tile_rows(self) -> pd.DataFrame:
        rows = []
        qc_min = float(self.cfg["qc"]["min_tissue_frac"])
        for n, row in self.world.cases.iterrows():
            for grid, spec in (("g512", self.world.g512), ("g256", self.world.g256)):
                tissue = self.world.tissue512(n) if grid == "g512" else self.world.tissue256[n]
                for r in range(spec["rows"]):
                    for c in range(spec["cols"]):
                        frac = float(round(tissue[r, c], 4))
                        rows.append({"case_id": row.case_id, "grid": grid, "tile_idx": r * spec["cols"] + c,
                                     "row": r, "col": c, "x": c * spec["size"], "y": r * spec["size"],
                                     "size": spec["size"], "tissue_frac": frac, "qc_pass": int(frac >= qc_min),
                                     "path": schemas.tile_png(self.out, grid, row.case_id, r, c)
                                     .relative_to(self.out).as_posix()})
        return pd.DataFrame(rows)

    def write_cases_tiles(self) -> None:
        self._emit(_write_frame(self.out, schemas.CASES, self.world.cases))
        self.tiles = self.tile_rows()
        self._emit(_write_frame(self.out, schemas.TILES, self.tiles))

    # -- images --------------------------------------------------------------------------------------------------------------
    def _picture(self, lat256: np.ndarray, classes: np.ndarray | None, label: str, width: int | None = None
                 ) -> Image.Image:
        """Non-tissue picture: one flat color block per 256-px tile (hue from the latent), grid lines, a banner."""
        r, c = lat256.shape[:2]
        k = len(self.world.concepts)
        act = lat256[..., :k]
        hue = (np.argmax(act, axis=-1) / max(1, k - 1))
        val = np.clip(0.55 + 0.1 * np.tanh(act.max(axis=-1)), 0.3, 0.9)
        rgb = np.stack([0.3 + 0.6 * hue, 0.8 - 0.5 * hue, 0.4 + 0.2 * np.sin(6 * hue)], axis=-1) * val[..., None]
        if classes is not None:
            rgb[classes == "background"] = 0.96
        small = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        w = width or self.display_long
        h = int(round(w * self.world.height / self.world.width))
        img = small.resize((w, h), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        for j in range(1, c):
            draw.line([(j * w // c, 0), (j * w // c, h)], fill=(30, 30, 30), width=1)
        for i in range(1, r):
            draw.line([(0, i * h // r), (w, i * h // r)], fill=(30, 30, 30), width=1)
        draw.rectangle([0, 0, w, 14], fill=(20, 20, 20))
        draw.text((4, 2), f"SYNTHETIC - not tissue - {label}", fill=(255, 255, 255))
        return img

    def write_images(self) -> None:
        w = self.world
        for n, row in w.cases.iterrows():
            img = self._picture(w.lat256[n], w.class256[n], row.case_id)
            self._emit(io.save_image(schemas.display_jpg(self.out, row.case_id), img, quality=85))
        # tile PNGs (both grids): full-size flat squares; PNG bytes cached per (size, shade) -> thousands of writes
        made: set[Path] = set()
        for t in self.tiles.itertuples(index=False):
            shade = (int(8 * t.tissue_frac), w.case_row[t.case_id] % 3)
            self.flat_png(self.out / t.path, int(t.size), int(t.size), shade, made)
        self._emit(self.out / "tiles")
        # degraded displays for test-surgical cases at degrade.display_severity
        sev = int(self.cfg["degrade"]["display_severity"])
        for row in w.cases[(w.cases.split == "test") & (w.cases.lane == "surgical")].itertuples():
            n = w.case_row[row.case_id]
            for dtype in self.cfg["degrade"]["types"]:
                vid = schemas.variant_case_id(row.case_id, dtype, sev)
                img = self._picture(w.degrade_latent(w.lat256[n], sev), w.class256[n], f"{vid}")
                self._emit(io.save_image(schemas.display_jpg(self.out, vid), img, quality=85))

    def flat_png(self, path: Path, width: int, height: int, shade: tuple[int, int], made: set[Path]) -> None:
        """Write a flat-color PNG (cached encoded bytes per size and shade); plain write, the root is rebuilt whole."""
        key = (width, height, shade)
        if key not in self._png_cache:
            gray = 80 + 18 * shade[0]
            buf = BytesIO()
            Image.new("RGB", (width, height), (gray, (gray + 40 * shade[1]) % 256, 200)).save(buf, "PNG")
            self._png_cache[key] = buf.getvalue()
        if path.parent not in made:
            path.parent.mkdir(parents=True, exist_ok=True)
            made.add(path.parent)
        path.write_bytes(self._png_cache[key])

    # -- composites ---------------------------------------------------------------------------------------------------------------
    def composite_layout(self) -> list[dict[str, Any]]:
        w, cfg = self.world, self.cfg
        rng = _rng(self.seed, "composites", int(cfg["composites"]["seed"]))
        test = w.cases[w.cases.split == cfg["composites"]["source_split"]]
        carc = np.sort(test[test.label.isin(cfg["carcinoma_classes"])].case_id.to_numpy())
        benign = np.sort(test[test.label.isin(cfg["benign_classes"])].case_id.to_numpy())
        carc = carc[[cid not in w.insufficient for cid in carc]]
        benign = benign[[cid not in w.insufficient for cid in benign]]
        sides = ("left", "right", "top", "bottom")
        n_orient = int(cfg["composites"]["orientations"])
        out = []
        for k in range(self.n_comp):
            orientation = k % n_orient
            out.append({"k": k, "carcinoma_case_id": str(rng.choice(carc)), "benign_case_id": str(rng.choice(benign)),
                        "orientation": orientation, "carcinoma_side": sides[orientation % len(sides)]})
        return out

    def composite_masks(self, side: str) -> np.ndarray:
        """Boolean (R256, C256): True where the composite takes the carcinoma image."""
        r, c = self.world.g256["rows"], self.world.g256["cols"]
        m = np.zeros((r, c), dtype=bool)
        if side == "left":
            m[:, : c // 2] = True
        elif side == "right":
            m[:, c // 2:] = True
        elif side == "top":
            m[: r // 2, :] = True
        else:
            m[r // 2:, :] = True
        return m

    @staticmethod
    def composite_truth(side: str, names: Mapping[str, str]) -> dict[str, str]:
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        truth = {}
        for edge, margin in names.items():
            truth[margin] = "tumor_side" if edge == side else "exact_negative" if edge == opposite[side] else "mixed"
        return truth

    def write_composites(self) -> None:
        w = self.world
        names = self.cfg["margin"]["names"]
        rows, self.comp = [], {}
        for spec in self.composite_layout():
            cid = schemas.composite_case_id(spec["carcinoma_case_id"], spec["k"])
            mask = self.composite_masks(spec["carcinoma_side"])
            a, b = w.case_row[spec["carcinoma_case_id"]], w.case_row[spec["benign_case_id"]]
            lat = np.where(mask[..., None], w.lat256[a], w.lat256[b])
            cls = np.where(mask, w.class256[a], w.class256[b])
            self.comp[cid] = {"lat256": lat, "class256": cls, "mask": mask, **spec}
            truth = self.composite_truth(spec["carcinoma_side"], names)
            rows.append({"case_id": cid, "carcinoma_case_id": spec["carcinoma_case_id"],
                         "benign_case_id": spec["benign_case_id"], "orientation": spec["orientation"],
                         "carcinoma_side": spec["carcinoma_side"], "width": w.width, "height": w.height,
                         "path": f"composites/{cid}.png", "display_path": f"display/{cid}.jpg",
                         **{f"truth_{m}": truth[m] for m in schemas.MARGINS}})
            img = self._picture(lat, cls, cid)
            self._emit(io.save_image(schemas.display_jpg(self.out, cid), img, quality=85))
            self._emit(io.save_image(self.out / f"composites/{cid}.png", self._picture(lat, cls, cid, w.width)))
        self.composites = pd.DataFrame(rows)
        self._emit(_write_frame(self.out, schemas.COMPOSITES, self.composites))

    # -- WSI ------------------------------------------------------------------------------------------------------------------------
    def write_wsi(self) -> None:
        w, cfg = self.world, self.cfg
        rng = _rng(self.seed, "wsi")
        k = len(w.concepts)
        self.wsi_patients = {f"A{i + 1:02d}": f"WP{(i // 2) + 1:02d}" for i in range(self.n_wsi)}  # 2 slides/patient
        scanner = rng.normal(scale=0.8, size=N_NUISANCE)
        stroma = _vec(STROMA_PROFILE, w.concepts)

        def lat_for(cls: str) -> np.ndarray:
            prof = stroma if cls in ("stroma",) else _vec(CLASS_PROFILE[cls], w.concepts)
            return np.concatenate([prof + rng.normal(size=k) * w.noise, scanner + rng.normal(scale=0.3, size=N_NUISANCE)])

        # labeled tiles (wsi256 / wsi512)
        self.wsi_items: dict[str, list[dict[str, Any]]] = {"wsi256": [], "wsi512": []}
        for set_name, grid in (("wsi256", "g256"), ("wsi512", "g512")):
            for wsi_id, patient in self.wsi_patients.items():
                for t in range(self.wsi_tiles if set_name == "wsi256" else max(4, self.wsi_tiles // 4)):
                    cls = w.classes[(t + int(wsi_id[1:])) % len(w.classes)]
                    self.wsi_items[set_name].append({"case_id": f"WSI_{wsi_id}_{grid}_{t:04d}", "label": cls,
                                                     "group": patient, "lat": lat_for(cls), "grid": grid})
        # margin windows
        r, c = w.g256["rows"], w.g256["cols"]
        n_folds = int(cfg["wsi"]["folds"])
        patients = sorted(set(self.wsi_patients.values()))
        fold_of = {p: i % n_folds for i, p in enumerate(patients)}
        names = cfg["margin"]["names"]
        rows, self.windows = [], {}
        strata = ("edge_tumor", "central_only", "none")
        edges = ("top", "bottom", "left", "right")
        for i in range(self.n_windows):
            wsi_id = sorted(self.wsi_patients)[i % self.n_wsi]
            stratum = strata[i % 3]
            tumor = np.zeros((r, c), dtype=bool)
            if stratum == "edge_tumor":
                edge = edges[(i // 3) % 4]
                if edge == "top":
                    tumor[:2, 2:5] = True
                elif edge == "bottom":
                    tumor[-2:, 3:6] = True
                elif edge == "left":
                    tumor[2:4, :2] = True
                else:
                    tumor[2:5, -2:] = True
            elif stratum == "central_only":
                tumor[2:4, 3:5] = True
            carc = w.classes[-1] if i % 2 == 0 else cfg["carcinoma_classes"][0]
            cls = np.where(tumor, carc, "stroma").astype(object)
            lat = np.stack([[lat_for(cls[a, b]) for b in range(c)] for a in range(r)])
            wid = f"WSI_{wsi_id}_{i:02d}"
            strips = {"top": tumor[0, :], "bottom": tumor[-1, :], "left": tumor[:, 0], "right": tumor[:, -1]}
            truth = {names[e]: ("tumor" if strips[e].any() else "clear") for e in edges}
            self.windows[wid] = {"lat256": lat, "class256": cls}
            rows.append({"window_id": wid, "wsi_id": wsi_id, "patient_id": self.wsi_patients[wsi_id],
                         "fold": fold_of[self.wsi_patients[wsi_id]], "x": 2048 * i, "y": 1536 * (i % 3),
                         "width": w.width, "height": w.height, "stratum": stratum, "path": f"wsi_windows/{wid}.png",
                         **{f"truth_{m}": truth[m] for m in schemas.MARGINS}})
            self._emit(io.save_image(self.out / f"wsi_windows/{wid}.png", self._picture(lat, cls, wid, w.width)))
        self.wsi_windows = pd.DataFrame(rows)
        self._emit(_write_frame(self.out, schemas.WSI_WINDOWS, self.wsi_windows))

    # -- auxiliary whole-image sets ------------------------------------------------------------------------------------------------
    def write_aux_items(self) -> None:
        w, cfg = self.world, self.cfg
        k = len(w.concepts)
        rng = _rng(self.seed, "nct")
        organ = rng.normal(scale=0.8, size=N_NUISANCE)
        self.nct_items = []
        for cls in cfg["nct"]["classes"]:
            prof = _vec(NCT_PROFILE.get(cls, {}), w.concepts)
            for t in range(self.n_nct):
                lat = np.concatenate([prof + rng.normal(size=k) * w.noise * 0.8,
                                      organ + rng.normal(scale=0.3, size=N_NUISANCE)])
                self.nct_items.append({"case_id": f"NCT_{cls}_{t:04d}", "label": cls, "group": "", "lat": lat})
        self.concept_items: dict[str, list[dict[str, Any]]] = {"quilt": [], "pathgen": []}
        for set_name, group_word in (("quilt", "video"), ("pathgen", "slide")):
            rng = _rng(self.seed, set_name)
            n = 0
            for cid in w.concepts:
                j = w.concepts.index(cid)
                for polarity in ("pos", "neg"):
                    for t in range(self.n_quilt):
                        base_cls = w.classes[int(rng.integers(len(w.classes)))]
                        prof = _vec(CLASS_PROFILE[base_cls], w.concepts) + rng.normal(size=k) * w.noise
                        prof[j] = prof[j] + 2.0 if polarity == "pos" else -0.5 + 0.2 * rng.normal()
                        lat = np.concatenate([prof, rng.normal(scale=1.0, size=N_NUISANCE)])
                        self.concept_items[set_name].append({
                            "case_id": f"{set_name.upper()}_{n:05d}", "label": f"{cid}:{polarity}",
                            "group": f"{group_word}_{(n // 4):04d}", "lat": lat})
                        n += 1

    # -- items per set -------------------------------------------------------------------------------------------------------------
    def items(self, set_name: str) -> pd.DataFrame:
        """One row per manifest item: identity columns + the latent (column 'lat') + degrade spec."""
        if hasattr(self, "_items_cache") and set_name in self._items_cache:
            return self._items_cache[set_name]
        w = self.world
        rows: list[dict[str, Any]] = []

        def tiled(case_id: str, split: str, grid: str, lat_grid: np.ndarray, label: str = "", group: str = "",
                  degrade: tuple[str, int] | None = None, path_case: str | None = None, image: str = "") -> None:
            """Tile rows; path = the tile PNG of path_case (crop = whole PNG) or a full-size field image (crop box)."""
            cols = lat_grid.shape[1]
            size = int(self.world.g512["size"] if grid == "g512" else self.world.g256["size"])
            for r in range(lat_grid.shape[0]):
                for c in range(cols):
                    idx = r * cols + c
                    if path_case:
                        path, x, y = schemas.tile_png(self.out, grid, path_case, r, c).relative_to(self.out).as_posix(), 0, 0
                    else:
                        path, x, y = image, c * size, r * size
                    rows.append({
                        "item_id": schemas.make_item_id(set_name, case_id, idx, *(degrade or (None, None))),
                        "set": set_name, "case_id": case_id, "split": split, "grid": grid, "tile_idx": idx,
                        "degrade_type": degrade[0] if degrade else "", "degrade_severity": degrade[1] if degrade else 0,
                        "label": label, "group": group, "lat": lat_grid[r, c], "path": path,
                        "x": x, "y": y, "w": size, "h": size})

        if set_name in ("bach512", "bach256"):
            grid = "g512" if set_name == "bach512" else "g256"
            for n, row in w.cases.iterrows():
                lat = w.lat512(n) if grid == "g512" else w.lat256[n]
                tiled(row.case_id, row.split, grid, lat, path_case=row.case_id)
        elif set_name in ("deg512", "deg256"):
            grid = "g512" if set_name == "deg512" else "g256"
            if set_name == "deg512":
                sel = w.cases[w.cases.split.isin(["calibration", "test"])]
                sevs = [int(s) for s in self.cfg["degrade"]["severities"]]
            else:
                sel = w.cases[(w.cases.split == "test") & (w.cases.lane == "surgical")]
                sevs = [int(self.cfg["degrade"]["display_severity"])]
            for row in sel.itertuples():
                n = w.case_row[row.case_id]
                for dtype in self.cfg["degrade"]["types"]:
                    for sev in sevs:
                        lat = w.degrade_latent(w.lat512(n) if grid == "g512" else w.lat256[n], sev)
                        tiled(row.case_id, row.split, grid, lat, degrade=(dtype, sev), path_case=row.case_id)
        elif set_name in ("comp512", "comp256"):
            for cid, comp in self.comp.items():
                lat = w.pool512(comp["lat256"]) if set_name == "comp512" else comp["lat256"]
                tiled(cid, "test", "g512" if set_name == "comp512" else "g256", lat, image=f"composites/{cid}.png")
        elif set_name == "wsi_margin256":
            for wid, win in self.windows.items():
                before = len(rows)
                tiled(wid, "", "g256", win["lat256"], image=f"wsi_windows/{wid}.png")
                for j, cls in enumerate(win["class256"].ravel()):       # row-major, same order as tiled()
                    rows[before + j]["label"] = str(cls)
                    rows[before + j]["group"] = self.wsi_patients[wid.split("_")[1]]
        elif set_name in ("wsi256", "wsi512"):
            for it in self.wsi_items[set_name]:
                rows.append({"item_id": schemas.make_item_id(set_name, it["case_id"]), "set": set_name,
                             "case_id": it["case_id"], "split": "", "grid": "", "tile_idx": -1, "degrade_type": "",
                             "degrade_severity": 0, "label": it["label"], "group": it["group"], "lat": it["lat"],
                             "path": f"aux/{set_name}/{it['case_id']}.png", "x": 0, "y": 0,
                             "w": self.aux_px[set_name], "h": self.aux_px[set_name]})
        elif set_name == "nct" or set_name in ("quilt", "pathgen"):
            source = self.nct_items if set_name == "nct" else self.concept_items[set_name]
            for it in source:
                rows.append({"item_id": schemas.make_item_id(set_name, it["case_id"]), "set": set_name,
                             "case_id": it["case_id"], "split": "", "grid": "", "tile_idx": -1, "degrade_type": "",
                             "degrade_severity": 0, "label": it["label"], "group": it["group"], "lat": it["lat"],
                             "path": f"aux/{set_name}/{it['case_id']}.png", "x": 0, "y": 0,
                             "w": self.aux_px[set_name], "h": self.aux_px[set_name]})
        else:
            raise ValueError(f"unknown embedding set {set_name!r}")
        df = pd.DataFrame(rows)
        if not hasattr(self, "_items_cache"):
            self._items_cache = {}
        self._items_cache[set_name] = df
        return df

    def n_views(self, set_name: str, grid: str) -> int:
        if set_name in DEGRADED_SETS:
            return int(self.cfg["degrade"].get("views", 1))      # ruling M9: degraded sets at view 0 only
        return self.views[grid] if grid in self.views else 1

    # -- manifest --------------------------------------------------------------------------------------------------------------
    def write_manifest(self) -> None:
        frames = []
        for set_name in schemas.EMBED_SETS:
            items = self.items(set_name)
            if items.empty:
                continue
            grid = items["grid"].iloc[0]
            nv = self.n_views(set_name, grid)
            base = items.drop(columns=["lat"]).copy()
            if set_name in self.aux_px:                           # whole-image sets: write the small image files
                made: set[Path] = set()
                for p, lab in zip(base["path"], base["label"]):
                    self.flat_png(self.out / p, self.aux_px[set_name], self.aux_px[set_name],
                                  (zlib.crc32(str(lab).encode()) % 8, 0), made)
                self._emit(self.out / "aux" / set_name)
            base["degrade_seed"] = [int(self.cfg["degrade"]["seed"]) + (zlib.crc32(i.encode()) % 100000)
                                    if t else 0 for i, t in zip(base["item_id"], base["degrade_type"])]
            for v in range(nv):
                part = base.copy()
                part["view"] = v
                part["row_id"] = part["item_id"] + f"|v{v}"
                frames.append(part)
        manifest = pd.concat(frames, ignore_index=True)
        cols = [c.name for c in schemas.EMBED_MANIFEST.columns]
        self._emit(_write_frame(self.out, schemas.EMBED_MANIFEST, manifest[cols]))

    # -- embeddings --------------------------------------------------------------------------------------------------------------
    def encoder_sets(self, encoder: str) -> list[str]:
        grid = self.cfg["encoders"]["native_grid"][encoder]
        text = encoder in self.cfg["encoders"]["text_aligned"]
        if encoder == self.cfg["encoders"]["primary"]:
            return ["bach512", "deg512", "comp512", "wsi512", "nct", "quilt", "pathgen"]
        if encoder == self.cfg["encoders"]["agreement"]:
            return ["bach512", "deg512", "comp512", "nct"]
        if grid == "g256" and not text:
            return ["bach256", "deg256", "comp256", "wsi256", "wsi_margin256"]
        return ["bach256", "nct"] if grid == "g256" else ["bach512", "nct"]

    def encoder_heads(self, encoder: str) -> list[str]:
        return [h for h in self.cfg["encoders"]["heads"]
                if h != "align" or encoder in self.cfg["encoders"]["text_aligned"]]

    def write_embeddings(self) -> None:
        index_cols = [c.name for c in schemas.EMB_INDEX.columns]
        for encoder in self.cfg["encoders"]["priority"]:
            heads = self.encoder_heads(encoder)
            meta_sets, meta_heads = {}, {}
            for head in heads:
                enc = Encoder(self.world, encoder, head)
                meta_heads[head] = {"dim": enc.dim, "normalized": head == "align"}
                for set_name in self.encoder_sets(encoder):
                    items = self.items(set_name)
                    if items.empty:
                        continue
                    nv = self.n_views(set_name, items["grid"].iloc[0])
                    rng = _rng(self.seed, "emb", encoder, head, set_name)
                    lat = np.stack(items["lat"].to_numpy())
                    mean = np.zeros((len(items), enc.dim))
                    views = np.zeros((len(items), nv, enc.dim))
                    groups = items.groupby(["degrade_type", "degrade_severity"], sort=True).indices
                    for (dtype, sev), rows in groups.items():
                        spec = (dtype, int(sev)) if dtype else None
                        m, v = enc.embed(lat[rows], rng, nv, spec)
                        mean[rows], views[rows] = m, v
                    index = items.drop(columns=["lat", "path", "x", "y", "w", "h"]).copy()
                    index.insert(0, "emb_row", np.arange(len(index)))
                    index["view"] = -1
                    index["n_views"] = nv
                    self._emit(io.save_npy(schemas.emb_npy(self.out, encoder, set_name, head), mean.astype(np.float16)))
                    self._emit(_write_frame(self.out, schemas.EMB_INDEX, index[index_cols],
                                            schemas.emb_index_csv(self.out, encoder, set_name, head)))
                    # ruling M9: _views files for every split of the clean multi-view sets (bach512/bach256)
                    in_views = np.full(len(index), set_name in VIEWS_FILE_SETS)
                    has_views = bool(in_views.any() and nv > 1)
                    if has_views:
                        vt = views[in_views]
                        vidx = index[in_views].loc[index.index[in_views].repeat(nv)].copy()
                        vidx["view"] = np.tile(np.arange(nv), int(in_views.sum()))
                        vidx["n_views"] = 1
                        vidx["emb_row"] = np.arange(len(vidx))
                        self._emit(io.save_npy(schemas.emb_npy(self.out, encoder, set_name, head, views=True),
                                               vt.reshape(-1, enc.dim).astype(np.float16)))
                        self._emit(_write_frame(self.out, schemas.EMB_INDEX, vidx[index_cols].reset_index(drop=True),
                                                schemas.emb_index_csv(self.out, encoder, set_name, head, views=True)))
                    meta_sets[set_name] = {"n_items": int(len(index)), "n_views": int(nv), "views_file": has_views}
            prov = io.provenance(self.cfg, synthetic=True)
            self._emit(io.write_json(schemas.emb_meta_json(self.out, encoder), {
                "encoder": encoder, "model_card": f"synthetic://{encoder}", "revision": "synthetic", "heads": meta_heads,
                "sets": meta_sets, "preprocess": {"resize": None, "crop": None, "mean": None, "std": None,
                                                  "input_px": None, "mpp": self.mpp, "note": "synthetic embeddings"},
                "dtype": "float16", "created_utc": prov["created_utc"], "git_sha": prov["git_sha"],
                "config_hash": prov["config_hash"], "versions": prov["versions"], "synthetic": True,
            }))

    # -- text ---------------------------------------------------------------------------------------------------------------------
    def write_text(self) -> None:
        w = self.world
        bank, source = concept_bank(self.cfg)
        self.bank_source = source
        prompts = schemas.expand_prompts(bank)
        nct = nct_prompts()
        generic = w.mean_profile
        for encoder in self.cfg["encoders"]["text_aligned"]:
            enc = Encoder(w, encoder, "align")
            rng = _rng(self.seed, "text", encoder)
            ids, texts, embs = [], [], []
            for p in prompts:
                if p.kind == "concept":
                    vec = np.zeros(len(w.concepts))
                    if p.polarity == "pos":
                        vec[w.concepts.index(p.target)] = 2.0
                    else:
                        vec = generic.copy()
                else:
                    vec = _vec(CLASS_PROFILE[p.target], w.concepts)
                d = enc.concept_direction(vec) + enc.bias
                ids.append(p.id)
                texts.append(p.text)
                embs.append(d / np.linalg.norm(d) + rng.normal(scale=0.03, size=enc.dim))
            for cls in self.cfg["nct"]["classes"]:
                d = enc.concept_direction(_vec(NCT_PROFILE.get(cls, {}), w.concepts) + 0.1) + enc.bias
                ids.append(schemas.nct_prompt_id(cls))
                texts.append(nct.get(cls, f"synthetic NCT prompt for {cls}"))
                embs.append(d / np.linalg.norm(d) + rng.normal(scale=0.03, size=enc.dim))
            emb = _normalize(np.asarray(embs)).astype(np.float32)
            self._emit(io.save_npz(schemas.text_npz(self.out, encoder), ids=np.asarray(ids), texts=np.asarray(texts),
                                   emb=emb, model=np.asarray(encoder), synthetic=np.asarray(True)))

    # -- cells ----------------------------------------------------------------------------------------------------------------------
    def cell_feature_ids(self) -> list[str]:
        path = io.REPO_ROOT / "config" / "cell_features.yaml"
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            feats = data.get("features") if isinstance(data, Mapping) else None
            if isinstance(feats, list) and all(isinstance(f, Mapping) and "id" in f for f in feats):
                return [str(f["id"]) for f in feats]
            if isinstance(feats, Mapping):
                return [str(k) for k in feats]
        return list(CELL_FEATURES_BUILTIN)

    def write_cells(self) -> None:
        w, cfg = self.world, self.cfg
        ccfg = cfg["cells"]
        types = list(ccfg["types"])
        feature_ids = self.cell_feature_ids()
        tile_um = self.world.g512["size"] * self.mpp
        g = self.world.g512
        qc = self.tiles[(self.tiles.grid == "g512")].set_index(["case_id", "tile_idx"])["qc_pass"]
        inv_j = w.concepts.index("invasion") if "invasion" in w.concepts else 0
        pl_j = w.concepts.index("pleomorphism") if "pleomorphism" in w.concepts else 0
        sel_rows, tile_rows, case_rows = [], [], []
        for n, row in w.cases.iterrows():
            lat = w.lat512(n)
            cls512 = w.class256[n][0::2, 0::2]                 # class of each 512 tile's top-left 256 sub-tile
            proxy = lat[..., inv_j] + lat[..., pl_j]
            cand = [(float(proxy[r, c]), r * g["cols"] + c) for r in range(g["rows"]) for c in range(g["cols"])
                    if int(qc[(row.case_id, r * g["cols"] + c)]) == 1]
            cand.sort(key=lambda t: (-t[0], t[1]))
            chosen = cand[: int(ccfg["tiles_per_case"])]
            pooled: list[dict[str, Any]] = []
            rng = _rng(self.seed, "cells", row.case_id)
            for rank, (score, idx) in enumerate(chosen, start=1):
                r, c = divmod(idx, g["cols"])
                kind = cls512[r, c] if cls512[r, c] in CELL_MIX else "stroma"
                nuclei = self.make_nuclei(rng, kind, tile_um, types)
                pooled += nuclei
                sel_rows.append({"case_id": row.case_id, "tile_idx": idx, "split": row.split, "rank": rank,
                                 "tile_risk": float(1 / (1 + math.exp(-score))),
                                 "path": schemas.tile_png(self.out, "g512", row.case_id, r, c).relative_to(self.out)
                                 .as_posix(), "n_tiles": len(chosen)})
                self._emit(io.write_json(schemas.cell_json(self.out, row.case_id, idx), {
                    "case_id": row.case_id, "tile_idx": idx, "backend": ccfg["backend"], "model": ccfg["model"],
                    "mpp_model": float(ccfg["model_mpp"]), "mpp_source": self.mpp,
                    "scale": self.mpp / float(ccfg["model_mpp"]), "n_nuclei": len(nuclei), "n_untyped": 0,
                    "nuclei": [{**{k: v for k, v in nu.items() if not k.startswith("_")},
                                "contour_px": self.contour(nu)} for nu in nuclei],
                    "synthetic": True}, indent=None))
                overlay = schemas.cell_overlay_png(self.out, row.case_id, idx)
                self._emit(io.save_image(overlay, self.overlay(nuclei, tile_um, ccfg)))
                feats = self.cell_features(nuclei, tile_um ** 2 / 1e6, ccfg, feature_ids)
                tile_rows.append({"case_id": row.case_id, "tile_idx": idx, "n_nuclei": len(nuclei),
                                  **{f"n_{t}": sum(1 for nu in nuclei if nu["type"] == t) for t in schemas.CELL_TYPES},
                                  "overlay": overlay.relative_to(self.out).as_posix(), **feats})
            area = len(chosen) * tile_um ** 2 / 1e6
            feats = self.cell_features(pooled, area, ccfg, feature_ids) if chosen else {f: math.nan for f in feature_ids}
            n_neo = sum(1 for nu in pooled if nu["type"] == "neoplastic")
            case_rows.append({"case_id": row.case_id, "split": row.split, "n_tiles": len(chosen),
                              "n_nuclei": len(pooled), "n_neoplastic": n_neo,
                              "insufficient_tumor_nuclei": int(n_neo < int(ccfg["min_neoplastic_nuclei"])), **feats})
        cf = pd.DataFrame(case_rows)
        ref_labels = set(ccfg["reference_classes"])
        ref_ids = set(w.cases[(w.cases.split == "train") & w.cases.label.isin(ref_labels)].case_id)
        for f in feature_ids:
            cf[f"{f}__missing"] = cf[f].isna().astype(int)
            ref = np.sort(cf.loc[cf.case_id.isin(ref_ids), f].dropna().to_numpy())
            cf[f"{f}__ref_pct"] = [float(np.searchsorted(ref, v, side="right") / len(ref)) if (len(ref) and
                                   not math.isnan(v)) else math.nan for v in cf[f]]
        self._emit(_write_frame(self.out, schemas.CELL_TILES, pd.DataFrame(sel_rows), schemas.cell_tiles_csv(self.out)))
        self._emit(_write_frame(self.out, schemas.CELL_FEATURES, cf, schemas.cell_features_csv(self.out)))
        self._emit(_write_frame(self.out, schemas.CELL_FEATURES_TILE, pd.DataFrame(tile_rows),
                                schemas.cell_features_csv(self.out, per_tile=True)))

    @staticmethod
    def make_nuclei(rng: np.random.Generator, kind: str, tile_um: float, types: Sequence[str]) -> list[dict[str, Any]]:
        mix, area_mean, area_cv = CELL_MIX[kind]
        n = int(rng.poisson(45 if kind in ("Normal", "stroma") else 80))
        kinds = rng.choice(len(types), size=n, p=np.asarray(mix) / sum(mix))
        centers = rng.uniform(0, tile_um, size=(n, 2))
        neo = centers[kinds == types.index("neoplastic")] if "neoplastic" in types else np.zeros((0, 2))
        out = []
        for i in range(n):
            t = types[kinds[i]]
            if t == "inflammatory" and len(neo) and kind in ("Invasive", "InSitu") and rng.uniform() < 0.6:
                centers[i] = np.clip(neo[rng.integers(len(neo))] + rng.normal(scale=12, size=2), 0, tile_um)
            if t == "neoplastic":
                sigma = math.sqrt(math.log(1 + area_cv ** 2))
                area = float(rng.lognormal(math.log(area_mean) - sigma ** 2 / 2, sigma))
                ecc = float(np.clip(rng.beta(4, 3) if kind in ("Invasive", "InSitu") else rng.beta(5, 6), 0, 0.99))
            else:
                area = float(rng.lognormal(math.log(28), 0.2))
                ecc = float(np.clip(rng.beta(3, 5), 0, 0.99))
            out.append({"centroid_um": [float(centers[i, 0]), float(centers[i, 1])], "area_um2": area,
                        "perimeter_um": float(2 * math.sqrt(math.pi * area) * (1 + 0.3 * ecc)), "eccentricity": ecc,
                        "solidity": float(np.clip(rng.normal(0.93, 0.03), 0.6, 1.0)), "type": t,
                        "type_prob": float(np.clip(rng.normal(0.85, 0.08), 0.34, 1.0)),
                        "edge": bool(min(centers[i].min(), tile_um - centers[i].max()) < 3.0)})
        return out

    def contour(self, nucleus: Mapping[str, Any]) -> list[list[int]]:
        r_px = math.sqrt(nucleus["area_um2"] / math.pi) / self.mpp
        cx, cy = (v / self.mpp for v in nucleus["centroid_um"])
        return [[int(round(cx + r_px * math.cos(a))), int(round(cy + r_px * math.sin(a)))]
                for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]

    @staticmethod
    def overlay(nuclei: Sequence[Mapping[str, Any]], tile_um: float, ccfg: Mapping[str, Any]) -> Image.Image:
        size = 64
        img = Image.new("RGB", (size, size), (245, 242, 236))
        draw = ImageDraw.Draw(img)
        colors = ccfg["overlay_colors"]
        for nu in nuclei:
            x, y = (v / tile_um * size for v in nu["centroid_um"])
            draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=colors.get(nu["type"], "#000000"))
        draw.text((2, 2), "SYN", fill=(0, 0, 0))
        return img

    @staticmethod
    def cell_features(nuclei: Sequence[Mapping[str, Any]], area_mm2: float, ccfg: Mapping[str, Any],
                      feature_ids: Sequence[str]) -> dict[str, float]:
        """CELLS_SPEC §6 features from nuclei; nuc_* need >= min_neoplastic_nuclei, ratios >= min_total_nuclei."""
        total = len(nuclei)
        by = {t: [nu for nu in nuclei if nu["type"] == t] for t in schemas.CELL_TYPES}
        neo, infl = by["neoplastic"], by["inflammatory"]
        enough_neo = len(neo) >= int(ccfg["min_neoplastic_nuclei"])
        enough = total >= int(ccfg["min_total_nuclei"])

        def cv(values: Sequence[float]) -> float:
            arr = np.asarray(values, dtype=float)
            return float(arr.std(ddof=1) / arr.mean()) if len(arr) > 1 and arr.mean() > 0 else math.nan

        radius = float(ccfg["til_radius_um"])
        if infl and neo:
            npos = np.asarray([nu["centroid_um"] for nu in neo])
            near = [np.min(np.linalg.norm(npos - np.asarray(nu["centroid_um"]), axis=1)) <= radius for nu in infl]
            til = float(np.mean(near))
        else:
            til = 0.0
        values = {
            "neo_fraction": len(neo) / total if enough else math.nan,
            "neo_density": len(neo) / area_mm2 if enough and area_mm2 > 0 else math.nan,
            "nuc_area_mean": float(np.mean([nu["area_um2"] for nu in neo])) if enough_neo else math.nan,
            "nuc_area_cv": cv([nu["area_um2"] for nu in neo]) if enough_neo else math.nan,
            "nuc_shape_cv": cv([nu["eccentricity"] for nu in neo]) if enough_neo else math.nan,
            "infl_fraction": len(infl) / total if enough else math.nan,
            "til_proximity": til if enough else math.nan,
            "dead_fraction": len(by["dead"]) / total if enough else math.nan,
            "epi_fraction": len(by["epithelial"]) / total if enough else math.nan,
        }
        return {f: float(values.get(f, math.nan)) for f in feature_ids}


def _file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None


def build_key(cfg: Mapping[str, Any], seed: int, small: bool, with_cells: bool) -> dict[str, Any]:
    """Everything the output depends on: a root is reused only when this matches its marker exactly."""
    return {"seed": int(seed), "small": bool(small), "with_cells": bool(with_cells),
            "config_hash": config.config_hash(cfg),
            "concepts_yaml": _file_hash(config.resolve_repo_path(cfg["concepts"]["file"])),
            "cell_features_yaml": _file_hash(io.REPO_ROOT / "config" / "cell_features.yaml"),
            "generator": _file_hash(Path(__file__))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.make_synthetic", description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output data root (e.g. data_synthetic)")
    parser.add_argument("--seed", type=int, default=None, help="generator seed (default: settings.seed)")
    parser.add_argument("--with-cells", action="store_true", help="also write the cells module outputs")
    parser.add_argument("--small", action="store_true", help="smaller aux sets and images (tests)")
    parser.add_argument("--force", action="store_true", help="rebuild even if a matching synthetic root exists")
    parser.add_argument("--config", default=None, help="settings.yaml path (default: config/settings.yaml)")
    args = parser.parse_args(argv)
    cfg = config.load_settings(args.config)
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    out = Path(args.out)
    marker = out / cli.SYNTHETIC_MARKER
    meta = schemas.step_meta_json(out, STEP)
    if out.exists() and any(out.iterdir()) and not marker.is_file():
        print(f"refusing to write into {out}: it exists, is not empty and is not a synthetic root", file=sys.stderr)
        return 2
    build = build_key(cfg, seed, args.small, args.with_cells)
    if marker.is_file() and not args.force:
        prev = io.read_json(marker).get("build")
        if prev == build and meta.is_file():
            print(f"{out}: synthetic root already built (seed {seed}); use --force to rebuild")
            return 0
        print(f"{out}: an existing synthetic root was built with different arguments, settings or concept bank; "
              f"rerun with --force to delete and rebuild it (this also deletes downstream outputs under it)",
              file=sys.stderr)
        return 3
    if marker.is_file():
        shutil.rmtree(out)                                     # only ever deletes a root we generated (marker present)
    Generator(cfg, out, seed, args.small, args.with_cells, build).run()
    print(f"wrote synthetic data root {out} (seed {seed}{', small' if args.small else ''}"
          f"{', with cells' if args.with_cells else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
