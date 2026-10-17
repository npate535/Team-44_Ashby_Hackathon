"""Step 00: inventory the datasets on disk and gate the pipeline (plan §5.2; BUILD_PLAN §A1, Task 2).

Login-node safe: directory listings, image headers of a small sample, zip central directories, file sizes and a
`nrows`-bounded peek of the Quilt CSV. It never decodes all 400 TIFFs and never loads the PathGen JSON.

Inventories:
  * official BACH (dataset of record, Zenodo 3632035): Photos/<Class>/*.tif counts (".tif" only; Thumbs.db and
    .DS_Store ignored), microscopy_ground_truth.csv cross-check against folders, image size / mode / bit depth /
    dpi from a sample, bit depth of all 400 from file sizes, WSI/*.svs + *.xml, patient workbooks (presence, md5,
    identified images);
  * pathmmu/BACH: structure, formats, label source, counts; whether it is ICIAR 2018 BACH at all (on Delta it is
    the 2015 Bioimaging set, so the official release stays the dataset of record);
  * NCT-CRC-HE-100K class folders and counts; Quilt-1M CSV columns + a breast-row estimate from a bounded peek;
    the PathGen layout (patch images, or captions + slides only?).

mpp: BACH photos are 0.42 µm/px by the challenge specification (THUNDER uses the same); the TIFF resolution tag
(96 dpi) is meaningless, so measured_mpp records the published value and mpp_source says so.

Writes <data_root>/inventory.json (core.schemas.INVENTORY) and meta/00_inventory.json; prints a short summary.
Exit status: 0 when the gate passes (>= dataset.min_labeled_per_class labeled photos per class), 1 otherwise.
Recomputed on every run (cheap, and it must describe the disk as it is now); --limit and --force are accepted
for interface uniformity and have no effect.

Run: python -m pipeline.00_inventory [--bach-root ...] [--nct-root ...] [--pathmmu-bach-root ...] [--quilt-root ...]
     [--pathgen-root ...]
"""

from __future__ import annotations

import ast
import collections
import csv
import json
import os
import re
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from core import cli, io, schemas
from pipeline import common

STEP = "00_inventory"
ICIAR_NAME = re.compile(r"^(b|is|iv|n)\d{3}\.tif$")
TCGA_SVS = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}.*\.svs$")
PATCH_NAME = re.compile(r"_\d+_\d+\.(png|jpe?g)$", re.IGNORECASE)


# ---- helpers ------------------------------------------------------------------------------------------------------
def bounded_walk(root: Path, max_entries: int) -> dict[str, Any]:
    """Walk at most max_entries directory entries: extension counts, per-folder file counts, example paths."""
    ext_counts: collections.Counter[str] = collections.Counter()
    folder_counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    files: list[Path] = []
    seen = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        names = sorted(n for n in filenames if not n.startswith("."))
        folder_counts[rel] = len(names)
        examples[rel] = names[:3]
        for name in names:
            ext_counts[Path(name).suffix.lower() or "<none>"] += 1
            files.append(Path(dirpath) / name)
        seen += len(filenames) + len(dirnames)
        if seen >= max_entries:
            truncated = True
            break
    return {"ext_counts": dict(ext_counts), "folder_counts": folder_counts, "examples": examples,
            "files": files, "truncated": truncated, "entries_scanned": seen}


def _image_header(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        bits = None
        tags = getattr(im, "tag_v2", None)
        if tags is not None and 258 in tags:
            value = tags[258]
            bits = list(value) if isinstance(value, tuple) else [value]
        dpi = im.info.get("dpi")
        return {"path": str(path), "size": list(im.size), "mode": im.mode, "bits_per_sample": bits,
                "dpi": [float(d) for d in dpi] if dpi else None}


def _file_info(path: Path, expected_md5: str | None) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    md5 = common.md5_file(path)
    return {"path": str(path), "present": True, "bytes": path.stat().st_size, "md5": md5,
            "md5_ok": (md5 == expected_md5) if expected_md5 else None}


# ---- official BACH -------------------------------------------------------------------------------------------------
def inventory_bach(bach_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    inv_cfg = cfg["inventory"]
    photos = common.bach_photos_dir(bach_root, cfg)
    classes = list(cfg["classes"])
    out: dict[str, Any] = {"bach_root": str(bach_root), "photos_dir": str(photos), "photos_present": photos.is_dir()}
    per_class = {c: 0 for c in classes}
    listing: list[dict[str, Any]] = []
    junk: collections.Counter[str] = collections.Counter()
    if photos.is_dir():
        for label in classes:
            folder = photos / cfg["dataset"]["official"]["class_folders"][label]
            if not folder.is_dir():
                continue
            names = sorted(os.listdir(folder))
            tifs = [n for n in names if n.endswith(".tif") and not n.startswith(".")]
            junk.update(n for n in names if n not in tifs)
            per_class[label] = len(tifs)
            listing += [{"label": label, "file": n, "path": folder / n} for n in tifs]
    out["per_class_counts"] = per_class
    out["n_images"] = sum(per_class.values())
    out["ignored_files"] = dict(junk)
    out["iciar_names"] = sum(bool(ICIAR_NAME.match(r["file"])) for r in listing)

    # Ground-truth CSV cross-check: folder = label; the CSV must agree.
    gt_path = Path(bach_root) / cfg["dataset"]["official"]["ground_truth_csv"]
    gt: dict[str, Any] = {"path": str(gt_path), "present": gt_path.is_file()}
    if gt_path.is_file():
        with open(gt_path, newline="", encoding="utf-8-sig") as handle:
            rows = [r for r in csv.reader(handle) if r]
        csv_label = {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2}
        folder_label = {r["file"]: r["label"] for r in listing}
        disagree = sorted(f for f in folder_label if f in csv_label and csv_label[f] != folder_label[f])
        gt.update({"rows": len(rows), "in_folders_not_csv": sorted(set(folder_label) - set(csv_label))[:10],
                   "in_csv_not_folders": sorted(set(csv_label) - set(folder_label))[:10],
                   "label_disagreements": disagree[:10], "n_label_disagreements": len(disagree),
                   "agrees": not disagree and set(folder_label) == set(csv_label)})
    out["ground_truth_csv"] = gt

    # Header sample + file-size bit depth for all photos (stat only).
    sample: list[dict[str, Any]] = []
    for label in classes:
        for r in [r for r in listing if r["label"] == label][: int(inv_cfg["sample_per_class"])]:
            try:
                sample.append({"label": label, **_image_header(r["path"])})
            except OSError as exc:  # unreadable header is recorded, not hidden
                sample.append({"label": label, "path": str(r["path"]), "error": str(exc)})
    out["header_sample"] = sample
    sizes = collections.Counter(tuple(s["size"]) for s in sample if "size" in s)
    out["image_size"] = list(sizes.most_common(1)[0][0]) if sizes else None
    w, h = (out["image_size"] or cfg["dataset"]["image_size"])
    depth = collections.Counter()
    for r in listing:
        nbytes = r["path"].stat().st_size
        depth["16" if nbytes > w * h * 3 * 1.5 else "8"] += 1
    out["bit_depth_by_file_size"] = dict(depth)
    out["dpi_in_sample"] = sorted({tuple(s["dpi"]) for s in sample if s.get("dpi")})

    # WSIs
    wsi_dir = common.bach_wsi_dir(bach_root, cfg)
    svs = sorted(p.name for p in wsi_dir.glob("*.svs")) if wsi_dir.is_dir() else []
    xml = sorted(p.name for p in wsi_dir.glob("*.xml")) if wsi_dir.is_dir() else []
    xml_mpp = None
    if xml:
        head = (wsi_dir / xml[0]).read_text(encoding="utf-8", errors="replace")[:2000]
        m = re.search(r'MicronsPerPixel="([0-9.]+)"', head)
        xml_mpp = float(m.group(1)) if m else None
    out["wsi"] = {"dir": str(wsi_dir), "n_svs": len(svs), "n_xml": len(xml),
                  "annotated": sorted(Path(x).stem for x in xml if Path(x).stem + ".svs" in svs),
                  "xml_microns_per_pixel": xml_mpp}

    # Patient workbooks (not in the zip; prep_a fetches them)
    pcfg = cfg["dataset"]["patients"]
    micro = _file_info(common.patient_file(bach_root, cfg, "micro"), pcfg["micro_md5"])
    if micro["present"]:
        try:
            pmap = common.load_micro_patients(micro["path"])
            micro.update({"n_identified": len(pmap), "n_patients": len(set(pmap.values()))})
        except (ValueError, KeyError, zipfile.BadZipFile) as exc:
            micro["parse_error"] = str(exc)
    out["patient_file"] = micro
    wsi_p = _file_info(common.patient_file(bach_root, cfg, "wsi"), pcfg["wsi_md5"])
    if wsi_p["present"]:
        try:
            wmap = common.load_wsi_patients(wsi_p["path"])
            wsi_p.update({"n_slides": len(wmap), "n_patients": len({v["patient"] for v in wmap.values()})})
        except (ValueError, KeyError, zipfile.BadZipFile) as exc:
            wsi_p["parse_error"] = str(exc)
    out["wsi_patient_file"] = wsi_p
    return out


# ---- pathmmu/BACH --------------------------------------------------------------------------------------------------
def inventory_pathmmu_bach(root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": str(root), "present": False}
    walk = bounded_walk(root, int(cfg["inventory"]["max_walk_entries"]))
    zips: dict[str, Any] = {}
    names_all: list[str] = []
    for path in [f for f in walk["files"] if f.suffix.lower() == ".zip"]:
        try:
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
        except zipfile.BadZipFile as exc:
            zips[path.name] = {"error": str(exc)}
            continue
        names_all += names
        dirs = collections.Counter(str(Path(n).parent) for n in names)
        zips[path.name] = {"n_files": len(names),
                           "ext_counts": dict(collections.Counter(Path(n).suffix.lower() for n in names)),
                           "per_folder": dict(sorted(dirs.items())[:40]),
                           "examples": names[:int(cfg["inventory"]["examples_per_folder"])]}
    loose = [f.name for f in walk["files"]]
    basenames = [Path(n).name for n in names_all] + loose
    iciar = sum(bool(ICIAR_NAME.match(b)) for b in basenames)
    class_hits = {c: sum(f"/{c}/" in f"/{n}" for n in names_all) for c in cfg["classes"]}
    is_iciar = iciar >= int(cfg["dataset"]["n_images"])
    label_source = ("class folders inside zip" if any(class_hits.values())
                    else "VQA question files" if any(f.suffix.lower() == ".json" for f in walk["files"])
                    else "unknown")
    return {
        "root": str(root), "present": True, "ext_counts": walk["ext_counts"], "folder_counts": walk["folder_counts"],
        "examples": walk["examples"], "truncated": walk["truncated"], "zips": zips, "label_source": label_source,
        "class_folder_file_counts": class_hits, "iciar2018_named_files": iciar, "is_iciar2018_bach": is_iciar,
        "note": ("matches the ICIAR 2018 BACH microscopy naming" if is_iciar else
                 "not the ICIAR 2018 BACH microscopy set (file names/counts differ; on Delta this is the 2015 "
                 "Bioimaging breast dataset); the official Zenodo release is the dataset of record"),
    }


# ---- NCT / Quilt / PathGen -----------------------------------------------------------------------------------------
def inventory_nct(root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": str(root), "present": False}
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    size = None
    for cls in cfg["nct"]["classes"]:
        folder = root / cls
        if not folder.is_dir():
            counts[cls] = 0
            continue
        files = common.image_files(folder)
        counts[cls] = len(files)
        examples[cls] = [f.name for f in files[: int(cfg["inventory"]["examples_per_folder"])]]
        if files and size is None:
            size = _image_header(files[0])["size"]
    extra = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in counts)
    return {"root": str(root), "present": True, "per_class_counts": counts, "examples": examples,
            "tile_size": size, "other_folders": extra,
            "enough_for_sample": all(n >= int(cfg["nct"]["per_class"]) for n in counts.values())}


def _is_breast(pathology: str) -> bool:
    try:
        values = ast.literal_eval(pathology) if pathology.strip().startswith("[") else [pathology]
    except (ValueError, SyntaxError):
        values = [pathology]
    return any(str(v).strip().lower().startswith("breast") for v in values)


def inventory_quilt(root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd

    if not root.is_dir():
        return {"root": str(root), "present": False}
    csvs = sorted(root.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    parts = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("images_part"))
    out: dict[str, Any] = {"root": str(root), "present": True, "csv_files": [p.name for p in csvs],
                           "image_parts": parts}
    if not csvs:
        return out
    path = next((p for p in csvs if "lookup" in p.name), csvs[0])
    nrows = int(cfg["inventory"]["quilt_peek_rows"])
    peek = pd.read_csv(path, nrows=nrows, dtype=str, keep_default_na=False)
    out.update({"csv": path.name, "csv_bytes": path.stat().st_size, "columns": list(peek.columns),
                "peek_rows": len(peek)})
    if "pathology" in peek.columns and len(peek):
        breast = peek["pathology"].map(_is_breast)
        avg_row_bytes = max(1.0, len(peek.to_csv(index=False).encode("utf-8")) / len(peek))
        est_rows = int(path.stat().st_size / avg_row_bytes)
        out.update({"peek_breast_fraction": round(float(breast.mean()), 4), "estimated_total_rows": est_rows,
                    "estimated_breast_rows": int(est_rows * float(breast.mean())),
                    "estimate_note": f"bounded peek of the first {len(peek)} rows; may not be representative"})
        if "subset" in peek.columns:
            out["peek_subsets"] = dict(collections.Counter(peek["subset"]))
        if "not_histology" in peek.columns:
            out["peek_not_histology"] = int((peek["not_histology"] == "1").sum())
    return out


def inventory_pathgen(root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": str(root), "present": False}
    walk = bounded_walk(root, int(cfg["inventory"]["max_walk_entries"]))
    files = walk["files"]
    images = [f for f in files if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    jsons = [f for f in files if f.suffix.lower() == ".json"]
    json_info = []
    for f in jsons[:5]:
        with open(f, encoding="utf-8", errors="replace") as handle:
            head = handle.read(65536)
        keys = sorted(set(re.findall(r'"([A-Za-z_]+)"\s*:', head)))
        json_info.append({"name": str(f.relative_to(root)), "bytes": f.stat().st_size, "keys_in_head": keys[:20]})
    patch_named = sum(bool(PATCH_NAME.search(f.name)) for f in images)
    return {
        "root": str(root), "present": True, "ext_counts": walk["ext_counts"], "folder_counts": walk["folder_counts"],
        "examples": walk["examples"], "truncated": walk["truncated"], "json_files": json_info,
        "n_images_scanned": len(images), "n_patch_named_images": patch_named,
        "n_svs": sum(f.suffix.lower() == ".svs" for f in files),
        "n_tcga_svs": sum(bool(TCGA_SVS.match(f.name)) for f in files),
        "has_patch_images": len(images) > 0,
    }


# ---- main ---------------------------------------------------------------------------------------------------------
def build_inventory(cfg: Mapping[str, Any], roots: Mapping[str, Path], *, synthetic: bool) -> dict[str, Any]:
    bach = inventory_bach(roots["bach_root"], cfg)
    minimum = int(cfg["dataset"]["min_labeled_per_class"])
    gate = all(bach["per_class_counts"].get(c, 0) >= minimum for c in cfg["classes"])
    inv = {
        "created_utc": io.utc_now(),
        "synthetic": bool(synthetic),
        schemas.INVENTORY_MPP_KEY: float(cfg["dataset"]["mpp"]),
        "mpp_source": cfg["inventory"]["mpp_source"],
        "label_source": "class folder (Photos/<Class>/); microscopy_ground_truth.csv cross-check",
        "n_images": bach["n_images"],
        "per_class_counts": bach["per_class_counts"],
        "image_size": bach["image_size"],
        "bach_root": str(roots["bach_root"]),
        "patient_file": bach["patient_file"],
        "wsi": bach["wsi"],
        "pathmmu_bach": inventory_pathmmu_bach(roots["pathmmu_bach_root"], cfg),
        "gate_passed": bool(gate),
        "gate_rule": f">= {minimum} labeled photos per class in the official BACH release",
        "bach": bach,
        "nct": inventory_nct(roots["nct_root"], cfg),
        "quilt": inventory_quilt(roots["quilt_root"], cfg),
        "pathgen": inventory_pathgen(roots["pathgen_root"], cfg),
    }
    problems = schemas.INVENTORY.problems(inv)
    if problems:
        raise ValueError("; ".join(problems))
    return inv


def summary_lines(inv: Mapping[str, Any]) -> list[str]:
    bach = inv["bach"]
    lines = [
        f"BACH photos: {inv['n_images']} {inv['per_class_counts']} size {inv['image_size']} "
        f"bit depth by file size {bach['bit_depth_by_file_size']}",
        f"labels: {inv['label_source']}; ground-truth CSV agrees: {bach['ground_truth_csv'].get('agrees')}",
        f"mpp {inv[schemas.INVENTORY_MPP_KEY]} ({inv['mpp_source']})",
        f"WSI: {inv['wsi']['n_svs']} svs, {inv['wsi']['n_xml']} xml",
        f"patient xlsx: present={inv['patient_file'].get('present')} md5_ok={inv['patient_file'].get('md5_ok')} "
        f"identified={inv['patient_file'].get('n_identified')}",
        f"pathmmu/BACH: {inv['pathmmu_bach'].get('note', 'absent')}",
        f"NCT: {inv['nct'].get('per_class_counts', 'absent')}",
        f"Quilt: csv={inv['quilt'].get('csv')} est. breast rows={inv['quilt'].get('estimated_breast_rows')}",
        f"PathGen: patch images={inv['pathgen'].get('has_patch_images')} json={inv['pathgen'].get('json_files')}",
        f"GATE {'PASSED' if inv['gate_passed'] else 'FAILED'}: {inv['gate_rule']}",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Inventory datasets on disk and gate the pipeline (plan §5.2).")
    common.add_source_args(parser)
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    roots = {dest: common.source_root(ctx.cfg, args, dest) for dest in common.SOURCE_ROOTS}
    inv = build_inventory(ctx.cfg, roots, synthetic=ctx.synthetic)
    out = schemas.inventory_json(ctx.data_root)
    io.write_json(out, inv)
    io.write_meta(schemas.step_meta_json(ctx.data_root, STEP), step=STEP, cfg=ctx.cfg,
                  inputs=[str(p) for p in roots.values()], outputs=[out],
                  args={"limit": ctx.limit, "force": ctx.force, **{k: str(v) for k, v in roots.items()}},
                  synthetic=ctx.synthetic, extra={"gate_passed": inv["gate_passed"]})
    for line in summary_lines(inv):
        print(line)
    print(json.dumps({"inventory": str(out), "gate_passed": inv["gate_passed"]}))
    return 0 if inv["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
