#!/usr/bin/env python3
"""
Build a small external test set from TCGA-BRCA for Sepia.

What it does
  1. Lists open-access TCGA-BRCA slide images from the NCI Genomic Data Commons (GDC).
  2. Reads each slide's TCGA barcode to get its sample type and preparation:
       sample code 01 = Primary Tumor (invasive carcinoma), 11 = Solid Tissue Normal
       slide code  DX = diagnostic FFPE (permanent section); TS/BS/MS = frozen section
  3. Picks N patients (seeded) that have BOTH a frozen tumor slide and a frozen normal slide,
     plus their FFPE tumor slide when one exists. Using the same patients across groups
     avoids confounding patient with label.
  4. Downloads those slides, finds tissue on a thumbnail, samples K tissue-rich fields of view
     at random (NOT chosen by any model), crops each at BACH resolution, and resizes it to
     2048 x 1536 px.
  5. Writes PNG crops plus manifest.csv and run_info.json.

Labels are SLIDE-LEVEL. A crop from a tumor slide may show stroma, fat or normal ducts.
Evaluate per slide (see TCGA_EXTERNAL_TEST.md), not per crop.

TCGA-BRCA has no benign or in-situ labels, so those classes are not produced.

Requirements:  pip install requests numpy pillow openslide-python openslide-bin
               (with uv:  uv add requests numpy pillow openslide-python openslide-bin)

Pixel sources (--source):
  download (default)  download each SVS, crop with openslide (steps 4-5 above).
  tiles               no slide download: the same selection, thumbnail tissue mask, FOV sampling, read level and
                      LANCZOS resize, with pixels from the GDC DeepZoom tile API (full-resolution level) and µm/px
                      from the SVS ImageDescription read with HTTP Range requests. Per-slide results are kept in
                      <out>/_done/ so an interrupted run resumes. Full-resolution pixels match openslide's exactly;
                      the thumbnail does not always: GDC renders a DeepZoom level from the next finer slide level when
                      the slide's own level downsample is slightly above the power of two (e.g. 16.004), so a FOV whose
                      tissue fraction sits near --min-tissue can enter or leave the candidate list and change the picks.
  --equivalence-check REF   re-run REF's FOV selection through tiles and re-extract REF's crops from its manifest
                      coordinates; writes <out>/tile_equivalence.json (picks identical? mean absolute pixel difference).
  --reuse-from REF --equivalence-json J   reuse REF's crops for slides REF already extracted, only when J (from the
                      check above) says reuse_allowed (docs/EXTERNAL_TCGA_R2.md).

Examples:
  python TCGA_script.py --out data/tcga_brca_test --dry-run          # list selection + total size
  python TCGA_script.py --out data/tcga_brca_test --patients 8 --fovs 6
  python TCGA_script.py --out data/tcga_brca_prod/equivalence --equivalence-check data/tcga_brca_test
  python TCGA_script.py --out data/tcga_brca_prod/protocol25 --patients 25 --source tiles \
      --reuse-from data/tcga_brca_test --equivalence-json data/tcga_brca_prod/equivalence/tile_equivalence.json
  # (tcga_extract.py is a symlink to this file)
On Delta: run downloads on the login node, cropping in a CPU job (it decodes large slides).
Laptop tip: start with --patients 5–8 (slides are deleted after cropping unless --keep-slides).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data/"

# TCGA-A1-A0SB-01Z-00-DX1.<uuid>.svs
BARCODE = re.compile(
    r"^(?P<patient>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(?P<sample>\d{2})(?P<vial>[A-Z])-"
    r"(?P<portion>\d{2})-(?P<slide>DX|TS|BS|MS)(?P<slide_n>[A-Z0-9]*)\."
)
FROZEN = {"TS", "BS", "MS"}
SAMPLE_LABEL = {"01": "Invasive", "11": "Normal"}  # Primary Tumor / Solid Tissue Normal

TARGET_W, TARGET_H = 2048, 1536


@dataclass
class SlideRec:
    file_id: str
    file_name: str
    file_size: int
    patient: str
    sample_code: str
    slide_code: str

    @property
    def prep(self) -> str:
        return "frozen" if self.slide_code in FROZEN else "ffpe"

    @property
    def label(self) -> str:
        return SAMPLE_LABEL[self.sample_code]

    @property
    def group(self) -> str:
        return f"{self.prep}_{'tumor' if self.sample_code == '01' else 'normal'}"


# ----------------------------------------------------------------------------- GDC listing
def list_brca_slides(session: requests.Session) -> list[SlideRec]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    payload = {"filters": filters, "fields": "file_id,file_name,file_size", "format": "JSON",
               "size": 20000}
    r = session.post(GDC_FILES, json=payload, timeout=120)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    out = []
    for h in hits:
        m = BARCODE.match(h["file_name"])
        if not m or m["sample"] not in SAMPLE_LABEL:
            continue
        out.append(SlideRec(h["file_id"], h["file_name"], int(h["file_size"]),
                            m["patient"], m["sample"], m["slide"]))
    return out


def select(slides: list[SlideRec], n_patients: int, max_bytes: int, seed: int) -> list[SlideRec]:
    """Patients with both a frozen tumor and a frozen normal slide; add FFPE tumor if present."""
    by_patient: dict[str, dict[str, list[SlideRec]]] = {}
    for s in slides:
        if s.file_size > max_bytes:
            continue
        by_patient.setdefault(s.patient, {}).setdefault(s.group, []).append(s)
    eligible = sorted(p for p, g in by_patient.items() if "frozen_tumor" in g and "frozen_normal" in g)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = []
    for p in eligible[:n_patients]:
        g = by_patient[p]
        for grp in ("frozen_tumor", "frozen_normal", "ffpe_tumor", "ffpe_normal"):
            if grp in g:
                chosen.append(sorted(g[grp], key=lambda s: s.file_name)[0])  # deterministic pick
    return chosen


# ----------------------------------------------------------------------------- download
def download(session: requests.Session, rec: SlideRec, dest: Path, retries: int = 6) -> Path:
    """Stream a slide to disk with resume (Range) and per-chunk progress. Deletes .part on final success."""
    path = dest / rec.file_name
    if path.exists() and path.stat().st_size == rec.file_size:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = tmp.stat().st_size if tmp.exists() else 0
            if have > rec.file_size:
                tmp.unlink(missing_ok=True)
                have = 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with session.get(GDC_DATA + rec.file_id, stream=True, timeout=(30, 120), headers=headers) as r:
                if have and r.status_code == 200:
                    # Server ignored Range — restart from scratch.
                    tmp.unlink(missing_ok=True)
                    have = 0
                    r.close()
                    continue
                if have and r.status_code not in (206, 200):
                    r.raise_for_status()
                elif not have:
                    r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                wrote = have
                t0 = time.time()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        wrote += len(chunk)
                        if time.time() - t0 > 15:
                            pct = 100.0 * wrote / max(rec.file_size, 1)
                            print(f"    … {wrote / 1e6:.0f}/{rec.file_size / 1e6:.0f} MB ({pct:.0f}%)",
                                  flush=True)
                            t0 = time.time()
            if tmp.stat().st_size != rec.file_size:
                raise IOError(f"size mismatch {tmp.stat().st_size} != {rec.file_size}")
            tmp.replace(path)
            return path
        except Exception as e:  # noqa: BLE001
            print(f"  download failed ({attempt}/{retries}) {rec.file_name}: {e}", file=sys.stderr, flush=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"could not download {rec.file_name}")


# ----------------------------------------------------------------------------- cropping
def slide_mpp(slide) -> tuple[float, bool]:
    """Microns per pixel at level 0, and whether it had to be estimated."""
    props = slide.properties
    for key in ("openslide.mpp-x", "aperio.MPP"):
        if key in props:
            try:
                v = float(props[key])
                if v > 0:
                    return v, False
            except ValueError:
                pass
    power = props.get("openslide.objective-power") or props.get("aperio.AppMag")
    if power:
        return 10.0 / float(power), True  # 40x ~ 0.25, 20x ~ 0.5
    raise ValueError("no resolution metadata")


def tissue_mask(thumb: Image.Image, sat_min: int = 20, val_max: int = 235, val_min: int = 40) -> np.ndarray:
    """True where the thumbnail looks like stained tissue (not glass, not black marker)."""
    hsv = np.asarray(thumb.convert("RGB").convert("HSV"))
    s, v = hsv[..., 1], hsv[..., 2]
    return (s >= sat_min) & (v <= val_max) & (v >= val_min)


def candidate_fovs(dims0: tuple[int, int], fov0: tuple[int, int], mask: np.ndarray,
                   min_tissue: float) -> list[tuple[int, int, float]]:
    """Non-overlapping FOV grid in level-0 coords with tissue fraction >= min_tissue."""
    W, H = dims0
    fw, fh = fov0
    mh, mw = mask.shape
    sx, sy = mw / W, mh / H
    out = []
    for y in range(0, H - fh + 1, fh):
        for x in range(0, W - fw + 1, fw):
            x0, y0 = int(x * sx), int(y * sy)
            x1, y1 = max(x0 + 1, int((x + fw) * sx)), max(y0 + 1, int((y + fh) * sy))
            frac = float(mask[y0:y1, x0:x1].mean())
            if frac >= min_tissue:
                out.append((x, y, frac))
    return out


def fov_size0(target_mpp: float, mpp0: float) -> tuple[int, int]:
    """Level-0 size of one field of view that becomes TARGET_W x TARGET_H px at target_mpp."""
    return (math.ceil(TARGET_W * target_mpp / mpp0), math.ceil(TARGET_H * target_mpp / mpp0))


def pick_fovs(dims0: tuple[int, int], fov0: tuple[int, int], thumb: Image.Image, min_tissue: float, k: int,
              seed: int, file_id: str) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Tissue-rich candidate FOVs on the thumbnail, and k of them sampled at random (seeded per slide; no model)."""
    cands = candidate_fovs(dims0, fov0, tissue_mask(thumb), min_tissue)
    rng = random.Random(f"{seed}:{file_id}")
    return cands, rng.sample(cands, min(k, len(cands)))


def read_level(downsamples: Sequence[float], mpp0: float, target_mpp: float) -> int:
    """Highest-resolution level that is still at least as fine as the target."""
    level = 0
    for i, d in enumerate(downsamples):
        if mpp0 * d <= target_mpp:
            level = i
    return level


def to_target(rgba: Image.Image) -> Image.Image:
    """Composite a read region on white and resize it to TARGET_W x TARGET_H (LANCZOS)."""
    rgb = Image.new("RGB", rgba.size, (255, 255, 255))
    rgb.paste(rgba, mask=rgba.split()[3])
    return rgb.resize((TARGET_W, TARGET_H), Image.LANCZOS)


def save_crop(rgb: Image.Image, rec: SlideRec, j: int, out_dir: Path) -> tuple[str, Path]:
    crop_id = f"{rec.patient}_{rec.group}_{j:02d}"
    rel = Path(rec.group) / f"{crop_id}.png"
    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_dir / rel)
    return crop_id, rel


def crop_row(crop_id: str, rel: Path, rec: SlideRec, x: int, y: int, fov0: tuple[int, int], level: int,
             mpp0: float, estimated: bool, target_mpp: float, frac: float) -> dict:
    return {
        "crop_id": crop_id, "path": str(rel), "label": rec.label, "label_level": "slide",
        "group": rec.group, "prep": rec.prep, "patient_id": rec.patient,
        "slide_file_name": rec.file_name, "slide_file_id": rec.file_id,
        "x0": x, "y0": y, "fov_w0": fov0[0], "fov_h0": fov0[1], "level": level,
        "mpp_level0": round(mpp0, 4), "mpp_estimated": estimated,
        "target_mpp": target_mpp, "tissue_fraction": round(frac, 3),
    }


def crop_slide(path: Path, rec: SlideRec, out_dir: Path, target_mpp: float, k: int,
               min_tissue: float, seed: int) -> list[dict]:
    import openslide  # imported here so --dry-run works without it

    slide = openslide.OpenSlide(str(path))
    mpp0, estimated = slide_mpp(slide)
    W, H = slide.dimensions
    fov0 = fov_size0(target_mpp, mpp0)
    thumb = slide.get_thumbnail((2048, 2048))
    cands, picks = pick_fovs((W, H), fov0, thumb, min_tissue, k, seed, rec.file_id)

    level = read_level(slide.level_downsamples, mpp0, target_mpp)
    d = slide.level_downsamples[level]
    size_l = (math.ceil(fov0[0] / d), math.ceil(fov0[1] / d))

    rows = []
    for j, (x, y, frac) in enumerate(picks):
        crop_id, rel = save_crop(to_target(slide.read_region((x, y), level, size_l)), rec, j, out_dir)
        rows.append(crop_row(crop_id, rel, rec, x, y, fov0, level, mpp0, estimated, target_mpp, frac))
    if len(picks) < k:
        print(f"  note: {rec.file_name} had only {len(cands)} tissue-rich fields", file=sys.stderr)
    slide.close()
    return rows


# ----------------------------------------------------------------------------- GDC tile source (--source tiles)
# The GDC tile service serves each slide as a DeepZoom pyramid (openslide DeepZoomGenerator conventions): level
# max = ceil(log2(max(W, H))) is full resolution, each lower level halves (rounding up), tiles are TileSize px plus
# Overlap px on every interior edge, PNG (lossless). Pixels at the full-resolution level are openslide level-0 pixels
# as decoded by the GDC server. µm/px is not in the tile metadata; it comes from the SVS ImageDescription, read with
# HTTP Range requests on the data endpoint (a few KB per slide; the slide itself is never downloaded).
GDC_TILE = "https://api.gdc.cancer.gov/tile/"
TILE_WORKERS = 8
RETRY_STATUS = {429, 500, 502, 503, 504}
THUMB_SIZE = (2048, 2048)


def http_get(session: requests.Session, url: str, *, params: dict | None = None, headers: dict | None = None,
             stream: bool = False, retries: int = 6, timeout: tuple[int, int] = (20, 120)) -> requests.Response:
    """GET with exponential backoff on network errors and 429/5xx; any other status is returned to the caller."""
    err = ""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, headers=headers, stream=stream, timeout=timeout)
            if r.status_code not in RETRY_STATUS:
                return r
            err = f"HTTP {r.status_code}"
            r.close()
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(min(60, 2 ** attempt))
    raise IOError(f"GET {url} failed after {retries} attempts ({err})")


def range_fetcher(session: requests.Session, file_id: str) -> Callable[[int, int], bytes]:
    """fetch(start, length) -> bytes of the slide file via one HTTP Range request (refuses a full-file reply)."""
    url = GDC_DATA + file_id

    def fetch(start: int, length: int) -> bytes:
        r = http_get(session, url, headers={"Range": f"bytes={start}-{start + length - 1}"}, stream=True)
        try:
            if r.status_code != 206:
                raise IOError(f"range request not honored for {file_id}: HTTP {r.status_code}")
            return r.content
        finally:
            r.close()

    return fetch


TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}
TIFF_TAGS = {256: "width", 257: "height", 270: "description", 322: "tile_width", 323: "tile_height"}


def tiff_ifds(fetch: Callable[[int, int], bytes], max_ifds: int = 32, chunk: int = 8192) -> list[dict[str, Any]]:
    """Width, height, tiling and (first directory only) ImageDescription of every TIFF/BigTIFF directory."""
    head = fetch(0, 16)
    order = {b"II": "<", b"MM": ">"}.get(head[:2])
    if order is None:
        raise ValueError("not a TIFF file")
    magic = struct.unpack(order + "H", head[2:4])[0]
    if magic == 42:
        big, off = False, struct.unpack(order + "I", head[4:8])[0]
    elif magic == 43:
        big, off = True, struct.unpack(order + "Q", head[8:16])[0]
    else:
        raise ValueError(f"not a TIFF file (magic {magic})")
    n_fmt, n_size, e_size, v_size, o_fmt = ("Q", 8, 20, 8, "Q") if big else ("H", 2, 12, 4, "I")
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    while off and off not in seen and len(out) < max_ifds:
        seen.add(off)
        block = fetch(off, chunk)
        n = struct.unpack(order + n_fmt, block[:n_size])[0]
        need = n_size + n * e_size + v_size
        if len(block) < need:
            block = fetch(off, need)
        ifd: dict[str, Any] = {}
        for i in range(n):
            e = block[n_size + i * e_size: n_size + (i + 1) * e_size]
            if big:
                tag, typ, count = struct.unpack(order + "HHQ", e[:12])
                raw = e[12:20]
            else:
                tag, typ, count = struct.unpack(order + "HHI", e[:8])
                raw = e[8:12]
            if tag not in TIFF_TAGS or (tag == 270 and out):
                continue
            size = TIFF_TYPE_SIZE.get(typ, 1) * count
            data = raw[:size] if size <= v_size else fetch(struct.unpack(order + o_fmt, raw)[0], size)
            if typ == 2:
                ifd[TIFF_TAGS[tag]] = data.split(b"\0", 1)[0].decode("latin-1")
            elif typ in (3, 4, 16):
                ifd[TIFF_TAGS[tag]] = struct.unpack(order + {3: "H", 4: "I", 16: "Q"}[typ],
                                                    data[:TIFF_TYPE_SIZE[typ]])[0]
        out.append(ifd)
        off = struct.unpack(order + o_fmt, block[n_size + n * e_size: need])[0]
    return out


def aperio_props(description: str) -> dict[str, str]:
    """'Aperio ...|AppMag = 40|MPP = 0.2485|...' -> {'AppMag': '40', 'MPP': '0.2485', ...}."""
    props = {}
    for part in description.split("|")[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            props[key.strip()] = value.strip()
    return props


def mpp_from_description(description: str) -> tuple[float, bool]:
    """Same rule as slide_mpp: aperio MPP (openslide.mpp-x), else 10 / AppMag flagged as estimated."""
    props = aperio_props(description)
    try:
        v = float(props.get("MPP", ""))
        if v > 0:
            return v, False
    except ValueError:
        pass
    power = props.get("AppMag")
    if power:
        return 10.0 / float(power), True
    raise ValueError("no resolution metadata")


def svs_header(fetch: Callable[[int, int], bytes]) -> dict[str, Any]:
    """Level sizes (tiled directories, as openslide's Aperio reader), downsamples and µm/px from the SVS header."""
    ifds = tiff_ifds(fetch)
    levels = [(int(d["width"]), int(d["height"])) for d in ifds if "tile_width" in d]
    if not levels:
        raise ValueError("no tiled TIFF directories")
    w0, h0 = levels[0]
    desc = ifds[0].get("description", "")
    mpp, estimated = mpp_from_description(desc)
    appmag = aperio_props(desc).get("AppMag")
    return {"width": w0, "height": h0, "levels": levels,
            "downsamples": [(w0 / w + h0 / h) / 2.0 for w, h in levels],   # openslide's level downsample
            "mpp": mpp, "mpp_estimated": estimated, "appmag": float(appmag) if appmag else None,
            "description_head": desc.split("|")[0].strip()[:160]}


def best_level_for_downsample(downsamples: Sequence[float], downsample: float) -> int:
    """openslide_get_best_level_for_downsample: the last level whose downsample is <= the requested one."""
    if downsample < downsamples[0]:
        return 0
    for i in range(1, len(downsamples)):
        if downsample < downsamples[i]:
            return i - 1
    return len(downsamples) - 1


@dataclass
class DeepZoom:
    """A DeepZoom pyramid (openslide DeepZoomGenerator layout) read through fetch_tile(level, col, row)."""

    width: int
    height: int
    tile: int
    overlap: int
    fetch_tile: Callable[[int, int, int], Image.Image]
    workers: int = TILE_WORKERS
    fmt: str = "png"

    def __post_init__(self) -> None:
        size = (int(self.width), int(self.height))
        dims = [size]
        while size[0] > 1 or size[1] > 1:
            size = (max(1, math.ceil(size[0] / 2)), max(1, math.ceil(size[1] / 2)))
            dims.append(size)
        self.dims: list[tuple[int, int]] = list(reversed(dims))

    @property
    def level_count(self) -> int:
        return len(self.dims)

    @property
    def max_level(self) -> int:
        return len(self.dims) - 1

    def level_for_power(self, k: int) -> int:
        """DeepZoom level with downsample 2**k."""
        if not 0 <= k <= self.max_level:
            raise ValueError(f"no DeepZoom level with downsample 2**{k}")
        return self.max_level - k

    def read(self, level: int, x: int, y: int, w: int, h: int) -> Image.Image:
        """RGBA region (x, y, w, h) in level pixels; outside the level is transparent (as openslide.read_region)."""
        lw, lh = self.dims[level]
        size, ov = self.tile, self.overlap
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        x0, y0, x1, y1 = max(0, x), max(0, y), min(lw, x + w), min(lh, y + h)
        if x1 <= x0 or y1 <= y0:
            return canvas
        addrs = [(c, r) for r in range(y0 // size, (y1 - 1) // size + 1)
                 for c in range(x0 // size, (x1 - 1) // size + 1)]

        def core(addr: tuple[int, int]) -> tuple[tuple[int, int], Image.Image]:
            c, r = addr
            img = self.fetch_tile(level, c, r)
            tl = (ov if c > 0 else 0, ov if r > 0 else 0)
            body = (min(size, lw - size * c), min(size, lh - size * r))
            expect = (tl[0] + body[0] + (ov if size * (c + 1) < lw else 0),
                      tl[1] + body[1] + (ov if size * (r + 1) < lh else 0))
            if img.size != expect:
                raise ValueError(f"DeepZoom tile {level}/{c}_{r} is {img.size}, expected {expect}")
            return addr, img.convert("RGBA").crop((tl[0], tl[1], tl[0] + body[0], tl[1] + body[1]))

        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            for (c, r), part in pool.map(core, addrs):
                canvas.paste(part, (size * c - x, size * r - y))
        return canvas


def gdc_deepzoom(session: requests.Session, file_id: str, workers: int = TILE_WORKERS) -> DeepZoom:
    """DeepZoom reader over the GDC tile API (one requests.Session per worker thread)."""
    r = http_get(session, GDC_TILE + "metadata/" + file_id)
    r.raise_for_status()
    meta = r.json()
    local = threading.local()

    def fetch_tile(level: int, col: int, row: int) -> Image.Image:
        s = getattr(local, "session", None)
        if s is None:
            s = local.session = requests.Session()
        resp = http_get(s, GDC_TILE + file_id, params={"level": level, "x": col, "y": row})
        if resp.status_code != 200:
            raise IOError(f"tile {level}/{col}_{row} of {file_id}: HTTP {resp.status_code} {resp.text[:120]}")
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        return img.convert("RGBA")

    return DeepZoom(int(meta["Width"]), int(meta["Height"]), int(meta["TileSize"]), int(meta["Overlap"]),
                    fetch_tile, workers, str(meta.get("Format", "")).lower())


def power_of_two(d: float) -> int | None:
    """k when d is 2**k within 1%, else None."""
    if d <= 0:
        return None
    k = int(round(math.log2(d)))
    return k if k >= 0 and abs(2 ** k / d - 1.0) < 0.01 else None


def tiles_thumbnail(dz: DeepZoom, header: Mapping[str, Any], size: tuple[int, int] = THUMB_SIZE
                    ) -> tuple[Image.Image, dict[str, Any]]:
    """openslide get_thumbnail semantics from DeepZoom tiles: read the level openslide would use (the DeepZoom level
    with the same power-of-two downsample), composite on white, then PIL thumbnail(size, LANCZOS)."""
    downsample = max(dim / t for dim, t in zip((dz.width, dz.height), size))
    slide_level = best_level_for_downsample(header["downsamples"], downsample)
    k = power_of_two(header["downsamples"][slide_level])
    matched = k is not None
    if k is None:                                   # no matching DeepZoom level: finest level coarser than needed
        k = max(0, int(math.floor(math.log2(downsample))))
    level = dz.level_for_power(k)
    lw, lh = dz.dims[level]
    rgba = dz.read(level, 0, 0, lw, lh)
    thumb = Image.new("RGB", rgba.size, (255, 255, 255))
    thumb.paste(rgba, None, rgba)
    thumb.thumbnail(size, Image.LANCZOS)
    return thumb, {"thumb_slide_level": slide_level, "thumb_dz_level": level, "thumb_dz_downsample": 2 ** k,
                   "thumb_level_matched": matched, "thumb_size": list(thumb.size)}


def plan_slide_tiles(session: requests.Session, rec: SlideRec, target_mpp: float, k: int, min_tissue: float,
                     seed: int, workers: int = TILE_WORKERS) -> dict[str, Any]:
    """Header, pyramid, thumbnail and FOV picks of one slide through the tile API (no crops read yet)."""
    header = svs_header(range_fetcher(session, rec.file_id))
    dz = gdc_deepzoom(session, rec.file_id, workers)
    if (dz.width, dz.height) != (header["width"], header["height"]):
        raise ValueError(f"DeepZoom size {(dz.width, dz.height)} != SVS level 0 {(header['width'], header['height'])}")
    fov0 = fov_size0(target_mpp, header["mpp"])
    thumb, tinfo = tiles_thumbnail(dz, header)
    cands, picks = pick_fovs((dz.width, dz.height), fov0, thumb, min_tissue, k, seed, rec.file_id)
    level = read_level(header["downsamples"], header["mpp"], target_mpp)
    kk = 0 if level == 0 else power_of_two(header["downsamples"][level])
    if kk is None:
        raise ValueError(f"read level {level} (downsample {header['downsamples'][level]:.3f}) has no DeepZoom match")
    info = {"file_id": rec.file_id, "file_name": rec.file_name, "group": rec.group, "patient_id": rec.patient,
            "dims0": [dz.width, dz.height], "dz_format": dz.fmt, "mpp_level0": round(header["mpp"], 4),
            "mpp_estimated": header["mpp_estimated"], "appmag": header["appmag"],
            "level_downsamples": [round(d, 4) for d in header["downsamples"]], "read_level": level,
            "fov0": list(fov0), "n_candidates": len(cands), "n_picks": len(picks), **tinfo}
    return {"header": header, "dz": dz, "fov0": fov0, "cands": cands, "picks": picks, "level": level,
            "dz_level": dz.level_for_power(kk), "downsample": header["downsamples"][level], "info": info}


def read_fov_tiles(plan: Mapping[str, Any], x: int, y: int) -> Image.Image:
    """The FOV at level-0 (x, y) through tiles: same read level and size as crop_slide, then to_target."""
    d = plan["downsample"]
    fov0 = plan["fov0"]
    size_l = (math.ceil(fov0[0] / d), math.ceil(fov0[1] / d))
    return to_target(plan["dz"].read(plan["dz_level"], int(x / d), int(y / d), *size_l))


def crop_slide_tiles(session: requests.Session, rec: SlideRec, out_dir: Path, target_mpp: float, k: int,
                     min_tissue: float, seed: int, workers: int = TILE_WORKERS) -> tuple[list[dict], dict]:
    plan = plan_slide_tiles(session, rec, target_mpp, k, min_tissue, seed, workers)
    h = plan["header"]
    rows = []
    for j, (x, y, frac) in enumerate(plan["picks"]):
        crop_id, rel = save_crop(read_fov_tiles(plan, x, y), rec, j, out_dir)
        rows.append(crop_row(crop_id, rel, rec, x, y, plan["fov0"], plan["level"], h["mpp"], h["mpp_estimated"],
                             target_mpp, frac) | {"source": "gdc_tiles"})
    if len(plan["picks"]) < k:
        print(f"  note: {rec.file_name} had only {len(plan['cands'])} tissue-rich fields", file=sys.stderr)
    return rows, plan["info"]


def tile_crop_from_row(session: requests.Session, row: Mapping[str, Any], workers: int = TILE_WORKERS) -> Image.Image:
    """Re-extract one manifest crop (level-0 coordinates and FOV size from the row) through the tile API."""
    if int(row["level"]) != 0:
        raise NotImplementedError("only level-0 manifest crops can be re-extracted from the row alone")
    dz = gdc_deepzoom(session, str(row["slide_file_id"]), workers)
    return to_target(dz.read(dz.max_level, int(row["x0"]), int(row["y0"]), int(row["fov_w0"]), int(row["fov_h0"])))


def pixel_diff(a: Image.Image, b: Image.Image) -> dict[str, float]:
    """Mean / 99th percentile / max absolute difference in 0-255 units over every pixel and channel."""
    d = np.abs(np.asarray(a.convert("RGB"), np.int16) - np.asarray(b.convert("RGB"), np.int16))
    return {"mad": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": int(d.max()),
            "share_gt_10": float((d > 10).mean())}


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def equivalence_check(session: requests.Session, ref: Path, out: Path, crops_per_slide: int = 1,
                      workers: int = TILE_WORKERS) -> dict[str, Any]:
    """Tile path vs a finished openslide extraction (ref = its --out): FOV picks per slide, and pixels of the first
    `crops_per_slide` crops of every slide re-extracted from the manifest coordinates. Writes tile_equivalence.json."""
    info = json.loads((ref / "run_info.json").read_text())
    args = info["args"]
    target, k, min_tissue, seed = float(args["target_mpp"]), int(args["fovs"]), float(args["min_tissue"]), int(args["seed"])
    selection = [SlideRec(**{f: s[f] for f in SlideRec.__dataclass_fields__}) for s in
                 json.loads((ref / "selection.json").read_text())]
    manifest = read_csv_rows(ref / "manifest.csv")
    eq_dir = out / "equivalence_crops"
    slides, crops = [], []
    for i, rec in enumerate(selection, 1):
        mine = [r for r in manifest if r["slide_file_id"] == rec.file_id]
        mine.sort(key=lambda r: r["crop_id"])
        print(f"[equivalence {i}/{len(selection)}] {rec.group:14s} {rec.file_name}", flush=True)
        plan = plan_slide_tiles(session, rec, target, k, min_tissue, seed, workers)
        tile_xy = [[int(x), int(y)] for x, y, _ in plan["picks"]]
        ref_xy = [[int(r["x0"]), int(r["y0"])] for r in mine]
        mpp_ref = {float(r["mpp_level0"]) for r in mine}
        slides.append({**plan["info"], "n_ref_crops": len(mine), "picks_identical": tile_xy == ref_xy,
                       "mpp_matches_manifest": (not mpp_ref) or mpp_ref == {round(plan["header"]["mpp"], 4)},
                       "tile_picks": tile_xy, "manifest_picks": ref_xy})
        for r in mine[:crops_per_slide]:
            got = read_fov_tiles(plan, int(r["x0"]), int(r["y0"]))
            eq_dir.mkdir(parents=True, exist_ok=True)
            got.save(eq_dir / f"{r['crop_id']}.png")
            with Image.open(ref / r["path"]) as handle:
                stats = pixel_diff(got, handle)
            crops.append({"crop_id": r["crop_id"], "group": r["group"], **stats})
            print(f"    {r['crop_id']}: MAD {stats['mad']:.3f}, p99 {stats['p99']:.0f}, max {stats['max']}", flush=True)
    mads = [c["mad"] for c in crops]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "reference": str(ref),
        "rule": "reuse reference crops only if every slide's tile-path FOV picks equal the manifest's and the maximum "
                "per-crop mean absolute difference is < 3 (0-255 units)  [docs/EXTERNAL_TCGA_R2.md]",
        "n_slides": len(slides), "n_slides_picks_identical": sum(s["picks_identical"] for s in slides),
        "n_slides_mpp_match": sum(s["mpp_matches_manifest"] for s in slides),
        "n_crops_compared": len(crops), "mad_max": max(mads) if mads else None,
        "mad_mean": float(np.mean(mads)) if mads else None, "mad_median": float(np.median(mads)) if mads else None,
        "max_abs_diff_max": max((c["max"] for c in crops), default=None),
        "slides": slides, "crops": crops,
    }
    result["reuse_allowed"] = bool(slides and crops and result["n_slides_picks_identical"] == len(slides)
                                   and result["mad_max"] < 3.0)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tile_equivalence.json").write_text(json.dumps(result, indent=2))
    return result


def reuse_rows(ref: Path, out: Path, rec: SlideRec) -> list[dict]:
    """Manifest rows (and linked PNGs) of `rec` from a reference extraction whose equivalence check allowed reuse."""
    rows = []
    for r in read_csv_rows(ref / "manifest.csv"):
        if r["slide_file_id"] != rec.file_id:
            continue
        dst = out / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.link(ref / r["path"], dst)
            except OSError:
                shutil.copy2(ref / r["path"], dst)
        rows.append(dict(r) | {"source": "reused_openslide"})
    return rows


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patients", type=int, default=25)
    ap.add_argument("--fovs", type=int, default=6, help="fields of view per slide")
    ap.add_argument("--target-mpp", type=float, default=0.42,
                    help="BACH microns/pixel; use the value measured in Sepia's inventory")
    ap.add_argument("--min-tissue", type=float, default=0.75)
    ap.add_argument("--max-slide-gb", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--keep-slides", action="store_true", help="keep .svs files after cropping")
    ap.add_argument("--dry-run", action="store_true", help="list the selection only")
    ap.add_argument("--source", choices=("download", "tiles"), default="download",
                    help="download: whole SVS + openslide (default); tiles: GDC DeepZoom tile API, no slide download")
    ap.add_argument("--workers", type=int, default=TILE_WORKERS, help="concurrent tile requests (tiles source)")
    ap.add_argument("--equivalence-check", type=Path, default=None, metavar="REF",
                    help="compare the tile path with a finished extraction REF, write --out/tile_equivalence.json, exit")
    ap.add_argument("--equivalence-crops", type=int, default=1, help="crops per slide re-extracted by the check")
    ap.add_argument("--reuse-from", type=Path, default=None, metavar="REF",
                    help="reuse REF's crops for slides REF already extracted; needs REF's passing equivalence JSON")
    ap.add_argument("--equivalence-json", type=Path, default=None,
                    help="tile_equivalence.json that allows --reuse-from (reuse_allowed must be true)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if a.equivalence_check:
        res = equivalence_check(session, a.equivalence_check, a.out, a.equivalence_crops, a.workers)
        print(f"Equivalence: picks identical on {res['n_slides_picks_identical']}/{res['n_slides']} slides; "
              f"crop MAD max {res['mad_max']}, mean {res['mad_mean']} over {res['n_crops_compared']} crops; "
              f"reuse_allowed={res['reuse_allowed']}")
        return
    equivalence = None
    if a.reuse_from:
        if a.equivalence_json is None:
            raise SystemExit("--reuse-from needs --equivalence-json (a passing tile_equivalence.json)")
        equivalence = json.loads(a.equivalence_json.read_text())
        if not equivalence.get("reuse_allowed") or Path(equivalence["reference"]).resolve() != a.reuse_from.resolve():
            raise SystemExit(f"{a.equivalence_json} does not allow reusing {a.reuse_from}")

    slides = list_brca_slides(session)
    chosen = select(slides, a.patients, int(a.max_slide_gb * 1e9), a.seed)
    total_gb = sum(s.file_size for s in chosen) / 1e9
    counts: dict[str, int] = {}
    for s in chosen:
        counts[s.group] = counts.get(s.group, 0) + 1
    print(f"Selected {len(chosen)} slides from {len({s.patient for s in chosen})} patients; "
          f"{total_gb:.1f} GB {'of slides (read through tiles, not downloaded)' if a.source == 'tiles' else 'to download'}."
          f" Groups: {counts}", flush=True)

    (a.out / "selection.json").write_text(json.dumps([asdict(s) | {"group": s.group} for s in chosen], indent=2))
    if a.dry_run:
        return

    reuse_ids: set[str] = set()
    if a.reuse_from:
        reuse_ids = {s["file_id"] for s in json.loads((a.reuse_from / "selection.json").read_text())}
    slide_dir = a.out / "_slides"
    done_dir = a.out / "_done"                      # tiles source: per-slide results, so a rerun resumes
    rows: list[dict] = []
    slide_info: list[dict] = []
    skipped: list[dict] = []
    for i, rec in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {rec.group:14s} {rec.file_name}", flush=True)
        try:
            if rec.file_id in reuse_ids:
                got = reuse_rows(a.reuse_from, a.out, rec)
                rows += got
                slide_info.append({"file_id": rec.file_id, "file_name": rec.file_name, "group": rec.group,
                                   "patient_id": rec.patient, "reused_from": str(a.reuse_from), "n_picks": len(got)})
                print(f"  reused {len(got)} crops from {a.reuse_from}", flush=True)
                continue
            if a.source == "tiles":
                done = done_dir / f"{rec.file_id}.json"
                if done.is_file():
                    saved = json.loads(done.read_text())
                else:
                    got, info = crop_slide_tiles(session, rec, a.out, a.target_mpp, a.fovs, a.min_tissue, a.seed,
                                                 a.workers)
                    saved = {"rows": got, "info": info}
                    done_dir.mkdir(exist_ok=True)
                    done.write_text(json.dumps(saved))
                rows += saved["rows"]
                slide_info.append(saved["info"])
                continue
            slide_dir.mkdir(exist_ok=True)
            path = download(session, rec, slide_dir)
            rows += crop_slide(path, rec, a.out, a.target_mpp, a.fovs, a.min_tissue, a.seed)
            if not a.keep_slides:
                path.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIPPED {rec.file_name}: {e}", file=sys.stderr, flush=True)
            skipped.append({"file_name": rec.file_name, "group": rec.group, "error": f"{type(e).__name__}: {e}"})

    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NCI GDC, TCGA-BRCA open-access slide images",
        "args": {k: str(v) for k, v in vars(a).items()},
        "slides_selected": len(chosen), "crops_written": len(rows), "groups": counts,
        "caveats": [
            "Labels are slide-level (sample type), not crop-level annotations.",
            "No benign or in-situ labels exist in TCGA-BRCA.",
            "Normal slides are frozen sections; compare frozen tumor vs frozen normal for a fair test.",
            "Fields of view were sampled at random among tissue-rich regions, not chosen by any model.",
        ],
    }
    if a.source == "tiles":
        info["pixel_source"] = ("GDC DeepZoom tile API (full-resolution level, PNG); thumbnail from the DeepZoom level "
                                "matching openslide's thumbnail level; µm/px from the SVS ImageDescription via HTTP "
                                "Range requests")
        info["slides"] = slide_info
        info["slides_skipped"] = skipped
        if equivalence is not None:
            info["tile_equivalence"] = {k: equivalence[k] for k in (
                "reference", "rule", "n_slides", "n_slides_picks_identical", "n_crops_compared", "mad_max",
                "mad_mean", "mad_median", "max_abs_diff_max", "reuse_allowed")}
    (a.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Done: {len(rows)} crops in {a.out}")


if __name__ == "__main__":
    main()
