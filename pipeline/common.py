"""Shared helpers for Agent 1's data-preparation steps (00_inventory, 01_build_cases, composites, wsi_extract,
mine_captions, 04a-04c). Other agents do not depend on this module.

Contents:
    source roots       --bach-root/--nct-root/... flags whose defaults come from settings paths.delta
    BACH listing       class-folder-sorted, file-name-sorted photo list (THUNDER index order)
    image IO + QC      read_rgb (16-bit TIFF safe), HSV tissue mask, tile-grid boxes, display sizes
    xlsx               stdlib reader for the BACH patient workbooks (openpyxl is not guaranteed on Delta)
    manifest           replace_manifest_sets: rewrite embed_manifest.csv atomically, touching only given sets
    misc               md5, stable seeds, worker counts

Portable: Python >= 3.10; stdlib + numpy + pandas + Pillow + core.{config,io,schemas} (runs inside Delta jobs).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core import io, schemas

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# The 16-bit BACH TIFFs exceed Pillow's default decompression-bomb guard only for much larger images; keep the
# guard but make sure the 2048x1536 microscopy images and WSI windows are allowed.
Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, 2048 * 1536 * 4)


# ---- source roots -----------------------------------------------------------------------------------------------
SOURCE_ROOTS = {
    # flag dest -> settings paths.delta lookup (tuple of keys)
    "bach_root": ("bach_official",),
    "nct_root": ("shared", "nct_crc"),
    "pathmmu_bach_root": ("shared", "pathmmu_bach"),
    "quilt_root": ("shared", "quilt"),
    "pathgen_root": ("shared", "pathgen"),
}


def add_source_args(parser: argparse.ArgumentParser, which: Sequence[str] = tuple(SOURCE_ROOTS)) -> None:
    """Add --bach-root / --nct-root / --pathmmu-bach-root / --quilt-root / --pathgen-root (defaults: settings)."""
    for dest in which:
        flag = "--" + dest.replace("_", "-")
        parser.add_argument(flag, dest=dest, default=None,
                            help=f"override settings paths.delta.{'.'.join(SOURCE_ROOTS[dest])}")


def source_root(cfg: Mapping[str, Any], args: argparse.Namespace, dest: str) -> Path:
    value = getattr(args, dest, None)
    if value:
        return Path(os.path.expanduser(value))
    node: Any = cfg["paths"]["delta"]
    for key in SOURCE_ROOTS[dest]:
        node = node[key]
    return Path(node)


# ---- BACH listing -----------------------------------------------------------------------------------------------
def bach_photos_dir(bach_root: str | Path, cfg: Mapping[str, Any]) -> Path:
    return Path(bach_root) / cfg["dataset"]["official"]["photos_subdir"]


def bach_dataset_dir(bach_root: str | Path, cfg: Mapping[str, Any]) -> Path:
    """Folder that `cases.csv` source_file paths are relative to (the parent of Photos/)."""
    return bach_photos_dir(bach_root, cfg).parent


def bach_wsi_dir(bach_root: str | Path, cfg: Mapping[str, Any]) -> Path:
    return Path(bach_root) / cfg["dataset"]["official"]["wsi_subdir"]


def patient_file(bach_root: str | Path, cfg: Mapping[str, Any], kind: str = "micro") -> Path:
    """Patient workbooks are fetched next to the unpacked zip: <bach_root>/<file> (they are not in the zip)."""
    return Path(bach_root) / cfg["dataset"]["patients"][f"{kind}_file"]


def list_bach_photos(photos_dir: str | Path, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Photos in THUNDER order: classes in split.class_order, files sorted, '.tif' only (junk files ignored).

    Returns one dict per image with source_index, label, file, source_file ('Photos/<folder>/<file>') and the
    absolute image_path. Raises FileNotFoundError when a class folder is missing.
    """
    photos = Path(photos_dir)
    folders = cfg["dataset"]["official"]["class_folders"]
    out: list[dict[str, Any]] = []
    for label in cfg["split"]["class_order"]:
        folder = photos / folders[label]
        if not folder.is_dir():
            raise FileNotFoundError(f"BACH class folder missing: {folder}")
        names = sorted(n for n in os.listdir(folder) if n.endswith(".tif") and not n.startswith("."))
        for name in names:
            out.append({
                "source_index": len(out),
                "label": label,
                "file": name,
                "source_file": f"{photos.name}/{folders[label]}/{name}",
                "image_path": str((folder / name).resolve()),
            })
    return out


# ---- image IO and QC --------------------------------------------------------------------------------------------
def read_rgb(path: str | Path, expected_size: Sequence[int] | None = None) -> Image.Image:
    """Decode an image to 8-bit RGB (handles BACH's 16-bit and 8-bit TIFFs) and check its size."""
    with Image.open(path) as handle:
        img = handle.convert("RGB")
    if img.mode != "RGB":
        raise ValueError(f"{path}: decoded mode {img.mode}, expected RGB")
    if expected_size is not None and tuple(img.size) != tuple(expected_size):
        raise ValueError(f"{path}: size {img.size}, expected {tuple(expected_size)}")
    return img


def tissue_mask(img: Image.Image, cfg: Mapping[str, Any]) -> np.ndarray:
    """Boolean HxW mask: HSV saturation (0-1) > qc.saturation_min and, when qc.brightness_max < 1, value (0-1) <
    qc.brightness_max. brightness_max >= 1 disables the value test: HSV value is the max RGB channel, so pale eosin
    and hematoxylin pixels sit at V ~ 0.95-1.0 and a value cut rejects real tissue (see 01q_requalify_qc)."""
    hsv = np.asarray(img.convert("RGB").convert("HSV"), dtype=np.float32) / 255.0
    mask = hsv[..., 1] > float(cfg["qc"]["saturation_min"])
    brightness_max = float(cfg["qc"]["brightness_max"])
    if brightness_max < 1.0:
        mask &= hsv[..., 2] < brightness_max
    return mask


def tissue_fraction(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    window = mask[y:y + h, x:x + w]
    return float(window.mean()) if window.size else 0.0


def grid_boxes(cfg: Mapping[str, Any], grid: str) -> list[dict[str, int]]:
    """Tiles of one grid in tile_idx order: tile_idx = row * cols + col, x = col * size, y = row * size."""
    spec = cfg["tiles"]["grids"][grid]
    size, rows, cols = int(spec["size"]), int(spec["rows"]), int(spec["cols"])
    return [{"tile_idx": r * cols + c, "row": r, "col": c, "x": c * size, "y": r * size, "size": size}
            for r in range(rows) for c in range(cols)]


def check_grid_fits(cfg: Mapping[str, Any], width: int, height: int) -> None:
    for grid, spec in cfg["tiles"]["grids"].items():
        if spec["cols"] * spec["size"] > width or spec["rows"] * spec["size"] > height:
            raise ValueError(f"grid {grid} ({spec['rows']}x{spec['cols']} of {spec['size']} px) does not fit "
                             f"a {width}x{height} image")


def display_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Size with the long side scaled to `long_side` (never upscaled), aspect ratio kept."""
    scale = min(1.0, float(long_side) / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def to_display(img: Image.Image, long_side: int) -> Image.Image:
    size = display_size(img.width, img.height, long_side)
    return img if size == img.size else img.resize(size, Image.LANCZOS)


def image_files(folder: str | Path, exts: Sequence[str] = IMAGE_EXTS) -> list[Path]:
    """Image files directly inside `folder`, sorted by name (hidden files skipped)."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts)


# ---- xlsx (stdlib) ------------------------------------------------------------------------------------------------
_REF = re.compile(r"^([A-Z]+)(\d+)$")


def col_index(letters: str) -> int:
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26."""
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def col_letters(index: int) -> str:
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out


def xlsx_cells(path: str | Path, sheet: str = "xl/worksheets/sheet1.xml") -> dict[str, str]:
    """{cell ref: text} of the first worksheet; shared, inline and numeric cells; blank cells are absent."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{XLSX_NS}t")) for si in root.findall(f"{XLSX_NS}si")]
        root = ET.fromstring(z.read(sheet))
    cells: dict[str, str] = {}
    for c in root.iter(f"{XLSX_NS}c"):
        ref, kind = c.get("r"), c.get("t")
        if kind == "inlineStr":
            text = "".join(t.text or "" for t in c.iter(f"{XLSX_NS}t"))
        else:
            v = c.find(f"{XLSX_NS}v")
            if v is None or v.text is None:
                continue
            text = shared[int(v.text)] if kind == "s" else v.text
        if ref and text.strip() != "":
            cells[ref] = text.strip()
    return cells


def _split_ref(ref: str) -> tuple[int, int]:
    m = _REF.match(ref)
    if not m:
        raise ValueError(f"bad cell reference {ref!r}")
    return col_index(m.group(1)), int(m.group(2))


def load_micro_patients(path: str | Path) -> dict[str, int]:
    """{histology file name: patient id} from ICIAR2018_BACH_dataset_micro_patient.xlsx.

    The sheet holds column groups 'Histology | Label | Patient' side by side (one group per class). Each header
    cell 'Patient' in row 1 marks a group whose Histology column is two to its left. Blank patient cells are
    unidentified images and are left out (the organizers: unidentified images come from patients other than the
    identified ones).
    """
    cells = xlsx_cells(path)
    patient_cols = [col for ref, text in cells.items() for col, row in [_split_ref(ref)]
                    if row == 1 and text.strip().lower() == "patient"]
    if not patient_cols:
        raise ValueError(f"{path}: no 'Patient' header in row 1")
    max_row = max(_split_ref(ref)[1] for ref in cells)
    out: dict[str, int] = {}
    for pcol in sorted(patient_cols):
        hcol = pcol - 2
        header = cells.get(f"{col_letters(hcol)}1", "").lower()
        if header != "histology":
            raise ValueError(f"{path}: expected 'Histology' two columns left of 'Patient' at {col_letters(pcol)}1")
        for row in range(2, max_row + 1):
            patient = cells.get(f"{col_letters(pcol)}{row}")
            name = cells.get(f"{col_letters(hcol)}{row}")
            if patient is None or name is None:
                continue
            if name in out:
                raise ValueError(f"{path}: {name} listed twice")
            out[name] = int(float(patient))
    return out


def load_wsi_patients(path: str | Path) -> dict[str, dict[str, Any]]:
    """{slide stem (e.g. 'A03', '07'): {type, patient}} from ICIAR2018_BACH_dataset_WSI_patient.xlsx
    (columns 'WSI filenames | Type | Patient', header in row 1)."""
    cells = xlsx_cells(path)
    header = {text.strip().lower(): col for ref, text in cells.items() for col, row in [_split_ref(ref)] if row == 1}
    name_col = next((c for h, c in header.items() if h.startswith("wsi")), None)
    if name_col is None or "patient" not in header:
        raise ValueError(f"{path}: expected headers 'WSI filenames' and 'Patient' in row 1")
    type_col = header.get("type")
    max_row = max(_split_ref(ref)[1] for ref in cells)
    out: dict[str, dict[str, Any]] = {}
    for row in range(2, max_row + 1):
        name = cells.get(f"{col_letters(name_col)}{row}")
        patient = cells.get(f"{col_letters(header['patient'])}{row}")
        if not name or patient is None:
            continue
        stem = Path(name).stem
        out[stem] = {"type": cells.get(f"{col_letters(type_col)}{row}", "") if type_col is not None else "",
                     "patient": int(float(patient))}
    return out


# ---- manifest ---------------------------------------------------------------------------------------------------
MANIFEST_COLUMNS = [c.name for c in schemas.EMBED_MANIFEST.columns]


def manifest_row(*, set_name: str, case_id: str, split: str, grid: str, tile_idx: int | None, path: str,
                 x: int, y: int, w: int, h: int, view: int, degrade_type: str = "", degrade_severity: int = 0,
                 degrade_seed: int = 0, label: str = "", group: str = "") -> dict[str, Any]:
    item_id = schemas.make_item_id(set_name, case_id, tile_idx, degrade_type or None,
                                   degrade_severity if degrade_type else None)
    return {
        "row_id": f"{item_id}|v{view}", "item_id": item_id, "set": set_name, "case_id": case_id, "split": split,
        "grid": grid, "tile_idx": -1 if tile_idx is None else int(tile_idx), "path": path, "x": int(x), "y": int(y),
        "w": int(w), "h": int(h), "view": int(view), "degrade_type": degrade_type,
        "degrade_severity": int(degrade_severity), "degrade_seed": int(degrade_seed), "label": label, "group": group,
    }


def replace_manifest_sets(root: str | Path, sets: Iterable[str], rows: Any) -> Any:
    """Rewrite embed_manifest.csv atomically: drop existing rows of `sets`, append `rows` (DataFrame or list of
    dicts), keep every other set untouched, validate against EMBED_MANIFEST. Returns the full frame."""
    import pandas as pd

    sets = set(sets)
    unknown = sets - set(schemas.EMBED_SETS)
    if unknown:
        raise ValueError(f"unknown embedding sets {sorted(unknown)}")
    new = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows), columns=MANIFEST_COLUMNS)
    if len(new) and not set(new["set"]).issubset(sets):
        raise ValueError(f"rows contain sets {sorted(set(new['set']) - sets)} outside {sorted(sets)}")
    path = schemas.embed_manifest_csv(root)
    if path.exists():
        old = io.read_csv(path, dtype=str, keep_default_na=False)
        old = old[~old["set"].isin(sets)]
        frame = pd.concat([old.astype(str), new[MANIFEST_COLUMNS].astype(str)], ignore_index=True)
    else:
        frame = new[MANIFEST_COLUMNS].astype(str)
    schemas.validate_frame(schemas.EMBED_MANIFEST, frame)
    io.write_csv(path, frame, columns=MANIFEST_COLUMNS)
    return frame


# ---- misc -------------------------------------------------------------------------------------------------------
def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from any parts (crc32 of their string form); independent of PYTHONHASHSEED."""
    return zlib.crc32(":".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


def degrade_seed(cfg: Mapping[str, Any], case_id: str, degrade_type: str, severity: int) -> int:
    """Seed for pipeline.degrade.apply: one per (case, type, severity), shared by every tile and view of the case,
    so a case gets one stain jitter everywhere (tiles, display JPEG)."""
    return stable_seed(cfg["degrade"]["seed"], case_id, degrade_type, severity)


def worker_count(requested: int | None) -> int:
    """--workers if given, else $SLURM_CPUS_PER_TASK, else 1."""
    if requested:
        return max(1, int(requested))
    env = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    return max(1, int(env)) if env.isdigit() else 1
