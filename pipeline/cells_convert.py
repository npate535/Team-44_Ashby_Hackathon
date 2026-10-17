"""Backend-agnostic conversion of nucleus segmentations to the CELLS_SPEC §5 per-nucleus JSON + overlay PNGs.

Inputs from any backend (TIAToolbox zarr, vqdang/hover_net JSON, ...): per nucleus a contour (x, y) in MODEL
pixels (the tile resized to cells.model_mpp), a centroid, a PanNuke type id and a type vote fraction.
Outputs (core.schemas.CELL_JSON): centroid_um, area_um2, perimeter_um, eccentricity, solidity, type, type_prob,
edge, contour_px (simplified, ORIGINAL tile pixels); µm use mpp_model (CELLS_SPEC §3), so features are independent
of the rescale. scale = model px / original px = mpp_source / mpp_model.

Geometry (pure numpy + Pillow so it runs in every environment and is testable without cv2/scikit-image):
  area        pixels of the contour polygon filled with Pillow (boundary pixels included, like cv2.fillPoly)
  perimeter   length of the closed simplified contour (a raw pixel staircase overestimates by ~5-10%)
  eccentricity sqrt(1 - l2 / l1) of the filled region's coordinate covariance (as skimage.regionprops)
  solidity    area / pixels of the filled convex hull of the contour (same rasterizer, so <= 1 up to rounding)
  edge        bounding box within cells_features rules.edge_px of the model-tile border (truncated nucleus)
Type 0 (background) instances are counted in n_untyped and not emitted. Contours padded with int32 min
(TIAToolbox zarr) are stripped.

Portable (Delta-side): Python >= 3.10, numpy + Pillow + PyYAML + core.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from core import config, io, schemas

PAD = np.iinfo(np.int32).min  # TIAToolbox pad_contours() fill value


# ---- config ------------------------------------------------------------------------------------------------------
def load_feature_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Parsed config/cell_features.yaml, checked against settings cells.types (schema CELL_TYPES)."""
    with open(config.resolve_repo_path(cfg["cells"]["features_file"]), encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    keys = [data["types"][k]["key"] for k in sorted(data["types"])]
    if keys != list(cfg["cells"]["types"]) or tuple(keys) != schemas.CELL_TYPES:
        raise ValueError(f"cell_features.yaml types {keys} != settings cells.types {cfg['cells']['types']}")
    ids = [f["id"] for f in data["features"]]
    if len(set(ids)) != len(ids):
        raise ValueError("cell_features.yaml feature ids must be unique")
    for f in data["features"]:
        if f["direction"] not in schemas.CONCEPT_DIRECTIONS:
            raise ValueError(f"feature {f['id']}: direction must be one of {schemas.CONCEPT_DIRECTIONS}")
        if f["min_count"] not in data["rules"]["min_count_rules"]:
            raise ValueError(f"feature {f['id']}: unknown min_count rule {f['min_count']!r}")
    return data


def type_names(fcfg: Mapping[str, Any]) -> dict[int, str]:
    return {int(k): v["key"] for k, v in fcfg["types"].items()}


# ---- geometry ----------------------------------------------------------------------------------------------------
def strip_pad(contour: Any) -> np.ndarray:
    c = np.asarray(contour, dtype=np.int64).reshape(-1, 2)
    return c[(c != PAD).all(axis=1)]


def fill_polygon(points: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Filled boolean mask of a polygon (pixel coordinates) on its bounding box; returns (mask, x0, y0)."""
    pts = np.round(np.asarray(points, dtype=float)).astype(np.int64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    img = Image.new("1", (int(x1 - x0 + 1), int(y1 - y0 + 1)), 0)
    ImageDraw.Draw(img).polygon([(int(x - x0), int(y - y0)) for x, y in pts], fill=1, outline=1)
    return np.asarray(img, dtype=bool), int(x0), int(y0)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; returns hull vertices counter-clockwise (>= 1 point)."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points)})
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def perimeter(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.sqrt((np.diff(np.vstack([p, p[:1]]), axis=0) ** 2).sum(axis=1)).sum())


def eccentricity(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0
    cov = np.cov(np.vstack([xs, ys]).astype(float), bias=True)
    l2, l1 = np.linalg.eigvalsh(cov)  # ascending
    return float(np.sqrt(max(0.0, 1.0 - l2 / l1))) if l1 > 0 else 0.0


def simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed contour (iterative; keeps the first point and the farthest point from it)."""
    p = np.asarray(points, dtype=float)
    if len(p) <= 3 or tol <= 0:
        return p
    far = int(np.argmax(((p - p[0]) ** 2).sum(axis=1)))
    keep = np.zeros(len(p), dtype=bool)
    keep[[0, far]] = True
    stack = [(0, far), (far, len(p))]
    while stack:
        a, b = stack.pop()
        end = p[b % len(p)]
        if b - a < 2:
            continue
        seg = p[a + 1:b]
        d = end - p[a]
        norm = np.hypot(*d)
        dist = (np.abs(d[0] * (p[a, 1] - seg[:, 1]) - d[1] * (p[a, 0] - seg[:, 0])) / norm if norm > 0
                else np.hypot(*(seg - p[a]).T))
        i = int(np.argmax(dist))
        if dist[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    out = p[keep]
    return out if len(out) >= 3 else p


# ---- records -----------------------------------------------------------------------------------------------------
def nuclei_records(contours: Sequence[Any], centroids: Sequence[Any], types: Sequence[Any], probs: Sequence[Any], *,
                   mpp_model: float, scale: float, hw_model: Sequence[int], names: Mapping[int, str],
                   simplify_px: float, edge_px: int) -> tuple[list[dict[str, Any]], int]:
    """Model-pixel instances -> CELL_JSON nuclei records (µm units, original-pixel contours); (records, n_untyped)."""
    h, w = int(hw_model[0]), int(hw_model[1])
    recs: list[dict[str, Any]] = []
    n_untyped = 0
    for contour, cen, t, p in zip(contours, centroids, types, probs):
        c = strip_pad(contour)
        if len(c) < 3:
            continue
        if int(t) not in names:
            n_untyped += 1
            continue
        mask, _, _ = fill_polygon(c)
        hull = convex_hull(c)
        hull_mask = fill_polygon(hull)[0] if len(hull) >= 3 else mask
        area = int(mask.sum())
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        simp = simplify(c, simplify_px)
        recs.append({
            "centroid_um": [round(float(cen[0]) * mpp_model, 2), round(float(cen[1]) * mpp_model, 2)],
            "area_um2": round(area * mpp_model ** 2, 2),
            "perimeter_um": round(perimeter(simp) * mpp_model, 2),
            "eccentricity": round(eccentricity(mask), 4),
            "solidity": round(min(1.0, area / max(1, int(hull_mask.sum()))), 4),
            "type": names[int(t)],
            "type_prob": round(float(p), 4),
            "edge": bool(x0 <= edge_px or y0 <= edge_px or x1 >= w - 1 - edge_px or y1 >= h - 1 - edge_px),
            "contour_px": (simp / float(scale)).round(1).tolist(),
        })
    return recs, n_untyped


def tile_document(meta: Mapping[str, Any], recs: list[dict[str, Any]], n_untyped: int) -> dict[str, Any]:
    doc = {**meta, "n_nuclei": len(recs), "n_untyped": int(n_untyped), "nuclei": recs}
    problems = schemas.CELL_JSON.problems(doc)
    if problems:
        raise ValueError("; ".join(problems))
    return doc


def write_tile_json(path: str | Path, meta: Mapping[str, Any], recs: list[dict[str, Any]], n_untyped: int) -> Path:
    """Atomic CELL_JSON write; written LAST for a tile, so its existence marks the tile done (resume)."""
    return io.write_json(path, tile_document(meta, recs, n_untyped), indent=None)


# ---- images ------------------------------------------------------------------------------------------------------
def resize_to_model(src: str | Path, dst: str | Path, mpp_src: float, mpp_model: float) -> dict[str, Any]:
    """Bicubic resize of an original tile to the model resolution (alpha dropped); returns the geometry record."""
    with Image.open(src) as handle:
        im = handle.convert("RGB")
    s = float(mpp_src) / float(mpp_model)
    size = (max(1, round(im.width * s)), max(1, round(im.height * s)))
    io.save_image(dst, im.resize(size, Image.BICUBIC))
    return {"scale": size[0] / im.width, "mpp_model": float(mpp_src) * im.width / size[0],
            "hw_model": [size[1], size[0]], "tile_px": [im.width, im.height]}


def render_overlay(tile: Image.Image, recs: Sequence[Mapping[str, Any]], colors: Mapping[str, str],
                   width: int = 1) -> Image.Image:
    """Contours drawn on the original-resolution tile, colored by type (settings cells.overlay_colors)."""
    out = tile.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for r in recs:
        pts = [tuple(map(float, xy)) for xy in r["contour_px"]]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=colors[r["type"]], width=width)
    return out
