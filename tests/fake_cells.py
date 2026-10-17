"""Fake nucleus-segmentation backend for Agent 1's tests (no TIAToolbox on the laptop): generated round nuclei."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def circle(cx: float, cy: float, r: float, n: int = 24) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1).round().astype(np.int32)


class FakeBackend:
    """Per tile a 3x3 lattice of nuclei with types 1,2,3,4,5,1,1,2,0 (one untyped); radius grows with the case
    number so features differ between cases. fail_on: substring of a png stem that makes the chunk raise."""

    name = "tiatoolbox"
    fail_on: str | None = None
    calls: list[int] = []

    def __init__(self, cfg):
        self.cfg = cfg

    def version(self) -> str:
        return "fake==0"

    def segment(self, pngs, scratch):
        FakeBackend.calls.append(len(pngs))
        out = {}
        for png in pngs:
            if FakeBackend.fail_on and FakeBackend.fail_on in Path(png).stem:
                raise RuntimeError("fake backend failure")
            with Image.open(png) as im:
                w, h = im.size
            num = int("".join(ch for ch in Path(png).stem.split("_")[1] if ch.isdigit()) or 1)
            r = max(1.5, min(w, h) / 12 * (0.6 + 0.1 * (num % 5)))
            cs, cents, types = [], [], []
            for k, t in enumerate([1, 2, 3, 4, 5, 1, 1, 2, 0]):
                cx, cy = w * (0.2 + 0.3 * (k % 3)), h * (0.2 + 0.3 * (k // 3))
                c = circle(cx, cy, r * (1 + 0.15 * (k % 2)))
                cs.append(c)
                cents.append(c.mean(axis=0))
                types.append(t)
            out[Path(png)] = type("R", (), {"contours": cs, "centroids": cents, "types": types,
                                            "probs": [0.9] * len(types)})()
        return out
