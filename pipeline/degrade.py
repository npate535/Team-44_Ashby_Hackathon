"""Simulated frozen-section artifacts: controlled image degradations (plan §6.7, BUILD_PLAN Task 2).

FROZEN INTERFACE (Task 1 stub; Task 2 implements; tests/test_contract.py pins every signature).
Callers: pipeline/01_build_cases.py (degraded display JPEGs), pipeline/02_embed.py (deg512 / deg256 sets, applied
on the fly), tests.

Types and severities come from settings.degrade (types blur | jpeg | stain | resample; severities 1..3; list
index i of each params list = severity i + 1). Every function is pure and deterministic for a given seed:
input image unchanged, output a new RGB PIL image of the same size.
Delta-side portable: Python >= 3.10, numpy + Pillow only (tests/test_config.py enforces this).

Notes:
  * Gaussian blur uses Pillow's GaussianBlur, whose radius is the kernel standard deviation in pixels.
  * The stain shift works in optical density OD = -log((I + 1) / 256), unmixed into hematoxylin / eosin / DAB
    concentrations with the Ruifrok-Johnston matrix (as in skimage.color.rgb2hed), jittered per channel, remixed
    and converted back. With alpha = beta = 0 it is the identity up to 8-bit rounding.
  * Severity parameters are only looked up here; they live in settings.degrade.params.
"""

from __future__ import annotations

import io
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter

# Ruifrok & Johnston (2001) stain vectors, rows = hematoxylin, eosin, DAB in RGB optical density.
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]], dtype=np.float64)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def _rgb(image: Image.Image) -> Image.Image:
    """An RGB copy (never the caller's object), so the input stays untouched."""
    return image.convert("RGB") if image.mode != "RGB" else image.copy()


def _check_type_severity(cfg: Mapping[str, Any], degrade_type: str, severity: int) -> int:
    types = list(cfg["degrade"]["types"])
    if degrade_type not in types:
        raise ValueError(f"unknown degrade type {degrade_type!r}; settings degrade.types are {types}")
    severities = list(cfg["degrade"]["severities"])
    if isinstance(severity, bool) or not isinstance(severity, (int, np.integer)) or int(severity) not in severities:
        raise ValueError(f"unknown severity {severity!r}; settings degrade.severities are {severities}")
    return severities.index(int(severity))


def severity_params(cfg: Mapping[str, Any], degrade_type: str, severity: int) -> dict[str, Any]:
    """Parameters of one (type, severity) from cfg["degrade"]["params"], e.g. {"sigma_px": 2.0} for blur 2."""
    i = _check_type_severity(cfg, degrade_type, severity)
    params = cfg["degrade"]["params"][degrade_type]
    return {name: values[i] for name, values in params.items()}


def gaussian_blur(image: Image.Image, sigma_px: float) -> Image.Image:
    """Gaussian blur with standard deviation sigma_px (source pixels)."""
    if sigma_px < 0:
        raise ValueError(f"sigma_px must be >= 0, got {sigma_px!r}")
    return _rgb(image).filter(ImageFilter.GaussianBlur(radius=float(sigma_px)))


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through in-memory JPEG at the given quality."""
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"JPEG quality must be in 1..100, got {quality!r}")
    buffer = io.BytesIO()
    _rgb(image).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")  # convert() forces the decode while the buffer is open


def stain_shift(image: Image.Image, alpha: float, beta: float, seed: int) -> Image.Image:
    """H&E optical-density jitter (Ruifrok-Johnston HED matrix): per channel c' = c * (1 + a) + b with
    a ~ U(-alpha, alpha), b ~ U(-beta, beta) drawn from np.random.default_rng(seed); back to RGB, clipped."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-alpha, alpha, size=3)
    b = rng.uniform(-beta, beta, size=3)
    rgb = np.asarray(_rgb(image), dtype=np.float64)
    od = -np.log((rgb + 1.0) / 256.0)
    conc = od.reshape(-1, 3) @ HED_FROM_RGB
    conc = conc * (1.0 + a) + b
    od_new = conc @ RGB_FROM_HED
    out = 256.0 * np.exp(-od_new) - 1.0
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(rgb.shape)
    return Image.fromarray(out, "RGB")


def down_up_sample(image: Image.Image, factor: int) -> Image.Image:
    """Downsample by an integer factor, then upsample back to the original size (bicubic both ways)."""
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    img = _rgb(image)
    w, h = img.size
    small = img.resize((max(1, w // int(factor)), max(1, h // int(factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def apply(image: Image.Image, degrade_type: str, severity: int, seed: int, cfg: Mapping[str, Any]) -> Image.Image:
    """Dispatch: apply degrade_type at severity using severity_params(cfg, ...); seed is used by stain_shift."""
    p = severity_params(cfg, degrade_type, severity)
    if degrade_type == "blur":
        return gaussian_blur(image, p["sigma_px"])
    if degrade_type == "jpeg":
        return jpeg_compress(image, p["quality"])
    if degrade_type == "stain":
        return stain_shift(image, p["alpha"], p["beta"], seed)
    if degrade_type == "resample":
        return down_up_sample(image, p["factor"])
    raise ValueError(f"no implementation for degrade type {degrade_type!r}")
