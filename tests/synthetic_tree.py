"""Fake dataset trees for Agent 1's tests: BACH (official layout, tiny generated non-tissue TIFFs incl. 16-bit ones),
patient workbooks, WSI stubs, NCT, Quilt-1M, PathGen and pathmmu/BACH. Everything is generated; nothing derives
from BACH or any other real image.

`small_settings(tmp)` writes a scaled-down copy of config/settings.yaml (64x48 images; grids g512 = 3x4 of 16 px,
g256 = 6x8 of 8 px) so the full 400-image pipeline runs in seconds. Grid names stay g512/g256 (schema vocabulary).
"""

from __future__ import annotations

import csv
import json
import struct
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from core import config

REPO = Path(__file__).resolve().parents[1]
SMALL_SIZE = (64, 48)
PREFIX = {"Benign": "b", "InSitu": "is", "Invasive": "iv", "Normal": "n"}
# Real BACH: these 19 files are 8-bit, the rest 16-bit (data_formats.md §1.2). We make a few 16-bit ones.
SIXTEEN_BIT = ("b001.tif", "is002.tif", "iv003.tif", "n004.tif")


# ---- settings ---------------------------------------------------------------------------------------------------
def small_settings(tmp: Path, **overrides: Any) -> tuple[Path, dict[str, Any]]:
    """Write a scaled settings.yaml under tmp; `overrides` are dotted keys, e.g. {"nct.per_class": 2}."""
    raw = yaml.safe_load((REPO / "config" / "settings.yaml").read_text())
    raw["dataset"]["image_size"] = list(SMALL_SIZE)
    raw["tiles"]["grids"] = {"g512": {"size": 16, "rows": 3, "cols": 4}, "g256": {"size": 8, "rows": 6, "cols": 8}}
    raw["tiles"]["display"]["long_side"] = 48
    raw["nct"]["per_class"] = 3
    for dotted, value in overrides.items():
        node = raw
        *parents, last = dotted.split(".")
        for key in parents:
            node = node[key]
        node[last] = value
    path = Path(tmp) / "settings_small.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path, config.load_settings(path)


# ---- images -----------------------------------------------------------------------------------------------------
def tissue_image(size: tuple[int, int] = SMALL_SIZE, seed: int = 0, blank_box: tuple[int, int, int, int] | None = None,
                 color: tuple[int, int, int] = (200, 110, 170)) -> np.ndarray:
    """uint8 HxWx3 'tissue-like' pink noise (passes QC); blank_box (x, y, w, h) is painted white (fails QC)."""
    w, h = size
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(color, dtype=np.int16) + rng.integers(-25, 26, size=(h, w, 3)), 0, 255).astype(np.uint8)
    if blank_box is not None:
        x, y, bw, bh = blank_box
        arr[y:y + bh, x:x + bw] = 250
    return arr


def write_tiff16(path: Path, arr8: np.ndarray) -> None:
    """Minimal uncompressed little-endian RGB TIFF with 16 bits/sample (v16 = v8 * 257), like most BACH photos."""
    h, w, _ = arr8.shape
    data = (arr8.astype(np.uint16) * 257).astype("<u2").tobytes()
    entries = []  # (tag, type, count, value)
    n_tags = 10
    ifd_offset = 8
    ifd_size = 2 + n_tags * 12 + 4
    bps_offset = ifd_offset + ifd_size
    data_offset = bps_offset + 6
    entries = [
        (256, 4, 1, w), (257, 4, 1, h), (258, 3, 3, bps_offset), (259, 3, 1, 1), (262, 3, 1, 2),
        (273, 4, 1, data_offset), (277, 3, 1, 3), (278, 4, 1, h), (279, 4, 1, len(data)), (284, 3, 1, 1),
    ]
    out = bytearray(b"II*\x00" + struct.pack("<I", ifd_offset))
    out += struct.pack("<H", n_tags)
    for tag, typ, count, value in entries:
        if typ == 3 and count == 1:
            out += struct.pack("<HHIHH", tag, typ, count, value, 0)
        else:
            out += struct.pack("<HHII", tag, typ, count, value)
    out += struct.pack("<I", 0)
    out += struct.pack("<HHH", 16, 16, 16)
    out += data
    path.write_bytes(bytes(out))


# ---- BACH -------------------------------------------------------------------------------------------------------
def bach_names(label: str, n: int = 100) -> list[str]:
    return [f"{PREFIX[label]}{i:03d}.tif" for i in range(1, n + 1)]


def default_patients(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Patient map that respects the THUNDER split: identified images only in train (b032-b041) and test (the
    real identified test ranges), with disjoint patient IDs per split."""
    order = cfg["split"]["class_order"]
    names = [n for label in order for n in bach_names(label)]
    split_of = config.split_assignments(cfg)
    pmap: dict[str, int] = {}
    for idx, name in enumerate(names):
        num = int("".join(ch for ch in name if ch.isdigit()))
        identified = ((name.startswith("b") and 32 <= num <= 90) or (name.startswith("is") and 40 <= num <= 69)
                      or (name.startswith("iv") and 41 <= num <= 73) or (name.startswith("n") and 46 <= num <= 68))
        if not identified:
            continue
        split = split_of[idx]
        base = {"train": 11, "calibration": 60, "test": 20}[split]
        pmap[name] = base + (num % 5)
    return pmap


def write_micro_patient_xlsx(path: Path, pmap: Mapping[str, int], cfg: Mapping[str, Any]) -> None:
    """Wide layout as the real file: per class a 'Histology | Label | Patient' group (A-C, E-G, I-K, M-O),
    rows 2-101 = images 001-100; blank Patient = unidentified. Written with openpyxl (shared strings)."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for g, label in enumerate(["Benign", "InSitu", "Invasive", "Normal"]):
        c0 = 1 + 4 * g
        ws.cell(row=1, column=c0, value="Histology")
        ws.cell(row=1, column=c0 + 1, value="Label")
        ws.cell(row=1, column=c0 + 2, value="Patient")
        for i, name in enumerate(bach_names(label)):
            ws.cell(row=2 + i, column=c0, value=name)
            ws.cell(row=2 + i, column=c0 + 1, value=label)
            if name in pmap:
                ws.cell(row=2 + i, column=c0 + 2, value=int(pmap[name]))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_wsi_patient_xlsx(path: Path, n_labeled: int = 10) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["WSI filenames", "Type", "Patient"])
    for k in range(1, 2 * n_labeled + 1):
        ws.append([f"{k:02d}.svs", "unlabeled", (k + 1) // 2])
    for k in range(1, n_labeled + 1):
        ws.append([f"A{k:02d}.svs", "labeled", k])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def make_bach_tree(root: Path, cfg: Mapping[str, Any], *, n_per_class: int = 100, patients: Mapping[str, int] | None
                   = None, with_patients: bool = True, with_wsi: bool = True, junk: bool = True,
                   size: tuple[int, int] | None = None) -> Path:
    """Official-layout BACH tree under root (= settings paths.delta.bach_official). Returns root.

    Image k of class c: pink noise seeded by its global index; the top-left contract tile area is white (fails QC).
    """
    size = tuple(size or cfg["dataset"]["image_size"])
    photos = Path(root) / cfg["dataset"]["official"]["photos_subdir"]
    g = cfg["tiles"]["grids"][cfg["tiles"]["contract_grid"]]
    blank = (0, 0, int(g["size"]), int(g["size"]))
    gt_rows = []
    idx = 0
    for label in cfg["split"]["class_order"]:
        folder = photos / cfg["dataset"]["official"]["class_folders"][label]
        folder.mkdir(parents=True, exist_ok=True)
        for name in bach_names(label, n_per_class):
            arr = tissue_image(size, seed=idx, blank_box=blank)
            if name in SIXTEEN_BIT:
                write_tiff16(folder / name, arr)
            else:
                Image.fromarray(arr, "RGB").save(folder / name, format="TIFF")
            gt_rows.append((name, label))
            idx += 1
        if junk:
            (folder / "Thumbs.db").write_bytes(b"junk")
            (folder / ".DS_Store").write_bytes(b"junk")
    with open(photos / "microscopy_ground_truth.csv", "w", newline="") as handle:
        csv.writer(handle, lineterminator="\r\n").writerows(gt_rows)
    if with_patients:
        write_micro_patient_xlsx(Path(root) / cfg["dataset"]["patients"]["micro_file"],
                                 default_patients(cfg) if patients is None else patients, cfg)
    if with_wsi:
        wsi = Path(root) / cfg["dataset"]["official"]["wsi_subdir"]
        wsi.mkdir(parents=True, exist_ok=True)
        for k in range(1, 3):
            (wsi / f"{k:02d}.svs").write_bytes(b"fake")
        for k in range(1, 3):
            (wsi / f"A{k:02d}.svs").write_bytes(b"fake")
            (wsi / f"A{k:02d}.xml").write_text('<Annotations MicronsPerPixel="0.466667"/>')
        write_wsi_patient_xlsx(Path(root) / cfg["dataset"]["patients"]["wsi_file"])
    return Path(root)


# ---- other datasets ---------------------------------------------------------------------------------------------
def make_nct_tree(root: Path, classes: Iterable[str], per_class: int = 5, size: int = 12) -> Path:
    for c, cls in enumerate(classes):
        folder = Path(root) / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = tissue_image((size, size), seed=1000 + c * 100 + i)
            Image.fromarray(arr, "RGB").save(folder / f"{cls}-{i:04d}ABCD.tif", format="TIFF")
    return Path(root)


QUILT_COLUMNS = ["", "caption", "image_path", "subset", "split", "pathology", "roi_text", "noisy_text",
                 "corrected_text", "med_umls_ids", "magnification", "height", "width", "not_histology", "single_wsi"]


def make_quilt_tree(root: Path, rows: Iterable[Mapping[str, Any]], csv_name: str = "quilt_1M_lookup-002.csv",
                    part: int = 1, write_images: bool = True) -> Path:
    """Quilt lookup CSV (unnamed index column first) + images under images_part_<part>/quilt_1m/."""
    root = Path(root)
    img_dir = root / f"images_part_{part}" / "quilt_1m"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(root / csv_name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(QUILT_COLUMNS)
        for i, row in enumerate(rows):
            full = {"caption": "", "image_path": f"img{i}.jpg", "subset": "quilt", "split": "train",
                    "pathology": "['Breast']", "roi_text": "[]", "noisy_text": "", "corrected_text": "",
                    "med_umls_ids": "[]", "magnification": "1.0", "height": "100", "width": "100",
                    "not_histology": "0", "single_wsi": "1", **row}
            writer.writerow([i] + [full[c] for c in QUILT_COLUMNS[1:]])
            if write_images and not (img_dir / full["image_path"]).exists():
                Image.fromarray(tissue_image((16, 16), seed=i), "RGB").save(img_dir / full["image_path"])
    return root


def make_pathgen_tree(root: Path, entries: Iterable[Mapping[str, Any]], *, with_patches: bool) -> Path:
    """PathGen JSON list ({wsi_id, position, caption}); with_patches adds README-named crops under patches/."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    entries = list(entries)
    (root / "PathGen-1.6M.json").write_text(json.dumps(entries))
    if with_patches:
        patch_dir = root / "patches"
        patch_dir.mkdir(exist_ok=True)
        for i, e in enumerate(entries):
            x, y = e["position"]
            Image.fromarray(tissue_image((16, 16), seed=500 + i), "RGB").save(patch_dir / f"{e['wsi_id']}_{x}_{y}.png")
    else:
        slides = root / "slides"
        slides.mkdir(exist_ok=True)
        for e in entries[:2]:
            (slides / f"{e['wsi_id']}.svs").write_bytes(b"fake")
    return root


def make_pathmmu_bach(root: Path) -> Path:
    """pathmmu/BACH stand-in that, like the real one, is NOT ICIAR 2018 BACH: a BACH.zip holding a 2015
    Bioimaging-style layout (Training_data/<Class>/t<k>.tif + Test_data/test<k>.tif)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "BACH.zip", "w") as z:
        for cls, n in (("Normal", 3), ("Benign", 3), ("InSitu", 2), ("Invasive", 2)):
            for k in range(n):
                z.writestr(f"BACH/Training_data/{cls}/t{k}.tif", b"fake")
        for k in range(2):
            z.writestr(f"BACH/Test_data/test{k}.tif", b"fake")
    (root / "README.txt").write_text("fake")
    return root


# ---- portability ------------------------------------------------------------------------------------------------
DELTA_ALLOWED = {"numpy", "pandas", "yaml", "PIL", "core", "pipeline", "sklearn", "scipy"}
FORBIDDEN_CORE = {"core.contract", "core.tiers"}


def delta_import_problems(relpath: str, allowed: set[str] = DELTA_ALLOWED) -> list[str]:
    """Problems of a Delta-side module: must parse with the Python 3.10 grammar, import only stdlib + `allowed`
    top-level packages, never pydantic/fastapi, core.contract or core.tiers (BUILD_PLAN §C.2)."""
    import ast
    import sys

    tree = ast.parse((REPO / relpath).read_text(), feature_version=(3, 10))
    imported: set[str] = set()
    full: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
                full.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
            full.update(f"{node.module}.{a.name}" for a in node.names)
            full.add(node.module)
    problems = sorted(imported - set(sys.stdlib_module_names) - allowed - {"__future__"})
    problems += sorted(full & FORBIDDEN_CORE)
    return problems


def write_cases_csv(data_root: Path, cfg: Mapping[str, Any], specs: Iterable[tuple[str, str, str, str]],
                    img_dir: Path | None = None) -> "Any":
    """Hand-made cases.csv for steps downstream of 01: specs = (case_id, label, split, lane). Each case gets a
    generated solid-color-plus-noise image (color derived from its position) at settings dataset.image_size."""
    import pandas as pd

    data_root = Path(data_root)
    img_dir = Path(img_dir or data_root / "_src")
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (case_id, label, split, lane) in enumerate(specs):
        path = img_dir / f"{case_id}.tif"
        color = (40 + (37 * i) % 200, 60 + (91 * i) % 180, 80 + (53 * i) % 160)
        Image.fromarray(tissue_image(tuple(cfg["dataset"]["image_size"]), seed=i, color=color), "RGB").save(path)
        rows.append({"case_id": case_id, "source": "synthetic", "source_file": f"Photos/{label}/{case_id}.tif",
                     "source_index": i, "image_path": str(path), "label": label, "split": split, "patient_id": "",
                     "group": f"b{i}", "lane": lane, "width": cfg["dataset"]["image_size"][0],
                     "height": cfg["dataset"]["image_size"][1], "display_path": f"display/{case_id}.jpg"})
    df = pd.DataFrame(rows)
    data_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_root / "cases.csv", index=False)
    return df


# ---- WSI stand-ins ------------------------------------------------------------------------------------------------
class FakeSlide:
    """openslide.OpenSlide stand-in over a generated level-0 array (levels = integer downsamples)."""

    def __init__(self, level0: np.ndarray, downsamples: tuple[int, ...] = (1, 2, 8)):
        base = Image.fromarray(level0, "RGB")
        self.levels = [base if d == 1 else base.resize((max(1, base.width // d), max(1, base.height // d)),
                                                       Image.BILINEAR) for d in downsamples]
        self.level_count = len(self.levels)
        self.level_dimensions = tuple(im.size for im in self.levels)
        self.dimensions = self.level_dimensions[0]
        self.level_downsamples = tuple(self.dimensions[0] / im.width for im in self.levels)

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        d = self.level_downsamples[level]
        x, y = int(location[0] / d), int(location[1] / d)
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))  # outside the slide = transparent, like openslide
        src = self.levels[level]
        x1, y1 = min(src.width, x + size[0]), min(src.height, y + size[1])
        if x1 > x and y1 > y:
            canvas.paste(src.crop((x, y, x1, y1)).convert("RGBA"), (0, 0))
        return canvas


def write_bach_xml(path: Path, regions: Iterable[tuple[str, list[tuple[int, int]]]], mpp: float = 0.466667,
                   use_attribute: bool = False) -> None:
    """Aperio ImageScope XML like BACH's: one Annotation; Regions with Text (or A01-style Attribute@Value)."""
    parts = [f'<Annotations MicronsPerPixel="{mpp}"><Annotation Id="1" Type="4"><Attributes/><Regions>'
             '<RegionAttributeHeaders/>']
    for i, (label, xy) in enumerate(regions, start=1):
        text = "" if use_attribute else label
        attrs = f'<Attributes><Attribute Name="1" Id="0" Value="{label}"/></Attributes>' if use_attribute else "<Attributes/>"
        verts = "".join(f'<Vertex X="{x}" Y="{y}" Z="0"/>' for x, y in xy)
        parts.append(f'<Region Id="{i}" Type="0" Text="{text}" NegativeROA="0">{attrs}<Vertices>{verts}</Vertices></Region>')
    parts.append("</Regions></Annotation></Annotations>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(parts))


def rect(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
