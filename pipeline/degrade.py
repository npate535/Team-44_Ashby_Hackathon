"""Simulated frozen-section artifacts: controlled image degradations (plan §6.7, BUILD_PLAN Task 2).

FROZEN INTERFACE (Task 1 stub; Task 2 implements; tests/test_contract.py pins every signature).
Callers: pipeline/01_build_cases.py (degraded display JPEGs), pipeline/02_embed.py (deg512 / deg256 sets, applied
on the fly), tests.

Types and severities come from settings.degrade (types blur | jpeg | stain | resample; severities 1..3; list
index i of each params list = severity i + 1). Every function is pure and deterministic for a given seed:
input image unchanged, output a new RGB PIL image of the same size.
Delta-side portable: Python >= 3.10, numpy + Pillow only (tests/test_config.py enforces this).
"""

from __future__ import annotations

from typing import Any, Mapping

from PIL import Image


def severity_params(cfg: Mapping[str, Any], degrade_type: str, severity: int) -> dict[str, Any]:
    """Parameters of one (type, severity) from cfg["degrade"]["params"], e.g. {"sigma_px": 2.0} for blur 2."""
    raise NotImplementedError("Task 2 implements pipeline.degrade.severity_params")


def gaussian_blur(image: Image.Image, sigma_px: float) -> Image.Image:
    """Gaussian blur with standard deviation sigma_px (source pixels)."""
    raise NotImplementedError("Task 2 implements pipeline.degrade.gaussian_blur")


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through in-memory JPEG at the given quality."""
    raise NotImplementedError("Task 2 implements pipeline.degrade.jpeg_compress")


def stain_shift(image: Image.Image, alpha: float, beta: float, seed: int) -> Image.Image:
    """H&E optical-density jitter (Ruifrok-Johnston HED matrix): per channel c' = c * (1 + a) + b with
    a ~ U(-alpha, alpha), b ~ U(-beta, beta) drawn from np.random.default_rng(seed); back to RGB, clipped."""
    raise NotImplementedError("Task 2 implements pipeline.degrade.stain_shift")


def down_up_sample(image: Image.Image, factor: int) -> Image.Image:
    """Downsample by an integer factor, then upsample back to the original size (bicubic both ways)."""
    raise NotImplementedError("Task 2 implements pipeline.degrade.down_up_sample")


def apply(image: Image.Image, degrade_type: str, severity: int, seed: int, cfg: Mapping[str, Any]) -> Image.Image:
    """Dispatch: apply degrade_type at severity using severity_params(cfg, ...); seed is used by stain_shift."""
    raise NotImplementedError("Task 2 implements pipeline.degrade.apply")
