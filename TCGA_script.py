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

Examples:
  python TCGA_script.py --out data/tcga_brca_test --dry-run          # list selection + total size
  python TCGA_script.py --out data/tcga_brca_test --patients 8 --fovs 6
  # (tcga_extract.py is a symlink to this file)
On Delta: run downloads on the login node, cropping in a CPU job (it decodes large slides).
Laptop tip: start with --patients 5–8 (slides are deleted after cropping unless --keep-slides).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

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


def crop_slide(path: Path, rec: SlideRec, out_dir: Path, target_mpp: float, k: int,
               min_tissue: float, seed: int) -> list[dict]:
    import openslide  # imported here so --dry-run works without it

    slide = openslide.OpenSlide(str(path))
    mpp0, estimated = slide_mpp(slide)
    W, H = slide.dimensions
    fov0 = (math.ceil(TARGET_W * target_mpp / mpp0), math.ceil(TARGET_H * target_mpp / mpp0))
    thumb = slide.get_thumbnail((2048, 2048))
    cands = candidate_fovs((W, H), fov0, tissue_mask(thumb), min_tissue)
    rng = random.Random(f"{seed}:{rec.file_id}")
    picks = rng.sample(cands, min(k, len(cands)))

    # Highest-resolution level that is still at least as fine as the target.
    level = 0
    for i, d in enumerate(slide.level_downsamples):
        if mpp0 * d <= target_mpp:
            level = i
    d = slide.level_downsamples[level]
    size_l = (math.ceil(fov0[0] / d), math.ceil(fov0[1] / d))

    rows = []
    for j, (x, y, frac) in enumerate(picks):
        rgba = slide.read_region((x, y), level, size_l)
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.split()[3])
        rgb = rgb.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        crop_id = f"{rec.patient}_{rec.group}_{j:02d}"
        rel = Path(rec.group) / f"{crop_id}.png"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        rgb.save(out_dir / rel)
        rows.append({
            "crop_id": crop_id, "path": str(rel), "label": rec.label, "label_level": "slide",
            "group": rec.group, "prep": rec.prep, "patient_id": rec.patient,
            "slide_file_name": rec.file_name, "slide_file_id": rec.file_id,
            "x0": x, "y0": y, "fov_w0": fov0[0], "fov_h0": fov0[1], "level": level,
            "mpp_level0": round(mpp0, 4), "mpp_estimated": estimated,
            "target_mpp": target_mpp, "tissue_fraction": round(frac, 3),
        })
    if len(picks) < k:
        print(f"  note: {rec.file_name} had only {len(cands)} tissue-rich fields", file=sys.stderr)
    slide.close()
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
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    slides = list_brca_slides(session)
    chosen = select(slides, a.patients, int(a.max_slide_gb * 1e9), a.seed)
    total_gb = sum(s.file_size for s in chosen) / 1e9
    counts: dict[str, int] = {}
    for s in chosen:
        counts[s.group] = counts.get(s.group, 0) + 1
    print(f"Selected {len(chosen)} slides from {len({s.patient for s in chosen})} patients; "
          f"{total_gb:.1f} GB to download. Groups: {counts}", flush=True)

    (a.out / "selection.json").write_text(json.dumps([asdict(s) | {"group": s.group} for s in chosen], indent=2))
    if a.dry_run:
        return

    slide_dir = a.out / "_slides"
    slide_dir.mkdir(exist_ok=True)
    rows: list[dict] = []
    for i, rec in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {rec.group:14s} {rec.file_name}", flush=True)
        try:
            path = download(session, rec, slide_dir)
            rows += crop_slide(path, rec, a.out, a.target_mpp, a.fovs, a.min_tissue, a.seed)
            if not a.keep_slides:
                path.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIPPED {rec.file_name}: {e}", file=sys.stderr, flush=True)

    if rows:
        with open(a.out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    (a.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Done: {len(rows)} crops in {a.out}")


if __name__ == "__main__":
    main()
