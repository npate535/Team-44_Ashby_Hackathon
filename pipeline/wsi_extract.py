"""Labeled tiles and margin windows from BACH's annotated WSIs (BUILD_PLAN §A9, Task 2; CPU job prep_b).

Runs in the `wsi` venv (openslide-python + openslide-bin; recipe pipeline/jobs/wsi_env.sh). openslide is imported
lazily inside open_slide(), so everything else is Delta-portable and testable with a fake slide reader.

Sources: <bach_root>/<dataset.official.wsi_subdir>/A01..A10.svs + .xml (Aperio ImageScope XML; data_formats.md
§1.4-1.5). Unannotated slides 01..20 are not used (no labels).
  * XML: every <Region> is a freehand polygon in level-0 pixels (X = column, Y = row). Its label is the
    Attribute@Value (A01) or the Text attribute, normalized by case-insensitive substring (wsi.label_patterns).
    MicronsPerPixel comes from each XML (0.466667); the SVS header's "MPP = 0,500000" is not used (conflict noted
    in meta; if 0.5 were right, physical sizes would be ~7% off).
  * Label mask: rasterized at the level closest to wsi.mask_downsample with codes 1..3 in wsi.paint_order (later
    classes overwrite earlier, so nested regions keep the more severe label; overlap pixel counts are logged).
    0 = unannotated.
  * Resampling 0.467 -> 0.42 µm/px: a tile of S px at wsi.target_mpp is read as round(S * target_mpp / mpp0)
    level-0 px and resized to S (Lanczos). Same for windows (2048x1536 <- ~1843x1382 level-0 px).
  * Labeled tiles (sets wsi512 / wsi256, grids wsi.tile_grids): non-overlapping level-0 lattice; the majority mask
    code must be annotated with purity >= wsi.min_label_purity, and tissue fraction >= wsi.tile_min_tissue_frac.
    Capped per (slide, class, grid) at wsi.max_tiles_per_class_per_wsi (seeded). label = region class, group =
    WSI patient. Written to wsi/tiles/<grid>/<wsi_id>/<x0>_<y0>.png; listed in wsi_tiles.csv (below).
  * Margin windows (wsi_windows.csv, set wsi_margin256): candidate windows on a lattice (stride
    wsi.window_stride_frac of the window) with tissue >= wsi.window_min_tissue_frac. For each of the four
    outermost margin.strip_px strips (in window pixels) the annotated-carcinoma fraction (settings
    carcinoma_classes) decides the truth: >= wsi.margin_tumor_min_frac -> tumor; exactly 0 -> clear; in between ->
    ambiguous, window skipped. Strata: edge_tumor (some margin tumor), central_only (all margins clear, carcinoma
    >= wsi.central_tumor_min_frac inside), none (no annotated carcinoma anywhere in the window). Up to
    wsi.windows_per_stratum per slide and stratum, seeded, non-overlapping. Windows are saved at the full window
    size (wsi/windows/<window_id>.png); the manifest lists their g256 tiles (crop boxes, encoders.views views).
  * Folds: patient-disjoint (ICIAR2018_BACH_dataset_WSI_patient.xlsx), seeded, wsi.folds folds; every tile and
    window carries its slide's fold. Models scored on a window must not be trained on tiles of the same fold.

CAVEAT — "unannotated != normal": BACH annotates benign, in situ and invasive regions only; the rest of each slide
is "considered normal and not relevant for evaluation" by the challenge, but it is NOT verified normal. So "clear"
and "none" mean "no annotated carcinoma", never "tumor-free"; unannotated tissue is never used as a clean label.
WSI labels also carry scanner shift vs the microscopy photos. Anything scored on the microscopy test split stays
WSI-free (§A9); this module only feeds the WSI-enhanced ablation and the WSI-window margin evaluation.

wsi_tiles.csv (produced and consumed here; consumers of the WSI ablation may read it):
  tile_id, wsi_id, patient_id, fold, grid, x0, y0, side0, label, purity, tissue_frac, path
  (x0, y0, side0 in level-0 slide pixels; path data-root relative).
Also writes wsi_folds.json {patient_id: fold} and meta/wsi_extract.json (counts, overlaps, rasterizer check against
WSI/gt_thumbnails/<id>.png, caveats).

Resumable: existing tile/window PNGs are skipped unless --force; --limit N processes the first N annotated slides.

Run (wsi venv): python -m pipeline.wsi_extract [--limit 1] [--workers 10] [--bach-root ...]
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from core import cli, io, schemas
from pipeline import common

STEP = "wsi_extract"
SETS = ("wsi512", "wsi256", "wsi_margin256")
TILE_SET = {"g512": "wsi512", "g256": "wsi256"}
STRATA = ("edge_tumor", "central_only", "none")
WINDOW_COLUMNS = [c.name for c in schemas.WSI_WINDOWS.columns]
TILE_COLUMNS = ["tile_id", "wsi_id", "patient_id", "fold", "grid", "x0", "y0", "side0", "label", "purity",
                "tissue_frac", "path"]
GT_COLORS = {"Benign": 0, "InSitu": 1, "Invasive": 2}  # gt_thumbnails: red / green / blue channel


# ---- slides -----------------------------------------------------------------------------------------------------
def open_slide(path: str | Path) -> Any:
    """openslide.OpenSlide (imported lazily: only the `wsi` venv has openslide)."""
    import openslide

    return openslide.OpenSlide(str(path))


SLIDE_OPENER: Callable[[str | Path], Any] = open_slide  # tests substitute a fake reader


def best_level(slide: Any, downsample: float) -> int:
    downs = [float(d) for d in slide.level_downsamples]
    return int(np.argmin([abs(d - downsample) for d in downs]))


def read_rgb(slide: Any, xy0: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
    """read_region (RGBA, transparent = no data) composited on white -> RGB."""
    tile = slide.read_region((int(xy0[0]), int(xy0[1])), level, (int(size[0]), int(size[1]))).convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", tile.size, (255, 255, 255, 255)), tile).convert("RGB")


# ---- XML + masks ------------------------------------------------------------------------------------------------
def norm_label(raw: str, cfg: Mapping[str, Any]) -> str:
    text = (raw or "").lower()
    for name in cfg["wsi"]["paint_order"]:
        if cfg["wsi"]["label_patterns"][name] in text:
            return name
    raise ValueError(f"unknown BACH WSI region label {raw!r}")


def parse_bach_xml(path: str | Path, cfg: Mapping[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """(microns per pixel, regions[{id, raw, label, xy}]) from an Aperio ImageScope XML."""
    root = ET.parse(path).getroot()
    mpp_attr = root.get("MicronsPerPixel")
    mpp = float(mpp_attr) if mpp_attr else float(cfg["wsi"]["mpp"])
    regions = []
    for reg in root.iter("Region"):
        attr = reg.find("./Attributes/Attribute")
        raw = (attr.get("Value") if attr is not None else "") or reg.get("Text") or ""
        xy = [(float(v.get("X")), float(v.get("Y"))) for v in reg.findall("./Vertices/Vertex")]
        if len(xy) >= 3:
            regions.append({"id": reg.get("Id"), "raw": raw, "label": norm_label(raw, cfg), "xy": xy})
    return mpp, regions


def label_mask(dims0: tuple[int, int], regions: Sequence[Mapping[str, Any]], size: tuple[int, int],
               cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    """uint8 mask of `size` (w, h) over a level-0 extent `dims0`; codes 1.. in paint_order; overlap pixel counts."""
    w0, h0 = dims0
    w, h = size
    sx, sy = w / w0, h / h0
    order = list(cfg["wsi"]["paint_order"])
    mask = np.zeros((h, w), dtype=np.uint8)
    overlaps: dict[str, int] = {}
    for code, name in enumerate(order, start=1):
        layer = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(layer)
        for r in regions:
            if r["label"] == name:
                draw.polygon([(x * sx, y * sy) for x, y in r["xy"]], fill=1)
        paint = np.asarray(layer, dtype=bool)
        over = int((paint & (mask > 0)).sum())
        if over:
            overlaps[name] = over
        mask[paint] = code
    return mask, overlaps


def tissue_mask(slide: Any, level: int, cfg: Mapping[str, Any]) -> np.ndarray:
    img = read_rgb(slide, (0, 0), level, slide.level_dimensions[level])
    return common.tissue_mask(img, cfg)


def side0(px: int, target_mpp: float, mpp0: float) -> int:
    """Level-0 pixels covering px pixels at target_mpp."""
    return max(1, int(round(px * float(target_mpp) / float(mpp0))))


def _slice(mask: np.ndarray, scale: tuple[float, float], x0: int, y0: int, w0: int, h0: int) -> np.ndarray:
    sx, sy = scale
    return mask[int(y0 * sy):max(int(y0 * sy) + 1, int(round((y0 + h0) * sy))),
                int(x0 * sx):max(int(x0 * sx) + 1, int(round((x0 + w0) * sx)))]


# ---- tiles ------------------------------------------------------------------------------------------------------
def tile_candidates(mask: np.ndarray, mscale: tuple[float, float], tissue: np.ndarray, tscale: tuple[float, float],
                    dims0: tuple[int, int], side: int, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Non-overlapping level-0 lattice tiles whose majority code is annotated with enough purity and tissue."""
    w0, h0 = dims0
    order = list(cfg["wsi"]["paint_order"])
    purity_min = float(cfg["wsi"]["min_label_purity"])
    tissue_min = float(cfg["wsi"]["tile_min_tissue_frac"])
    out = []
    for y0 in range(0, h0 - side + 1, side):
        for x0 in range(0, w0 - side + 1, side):
            win = _slice(mask, mscale, x0, y0, side, side)
            if not win.size:
                continue
            counts = np.bincount(win.ravel(), minlength=len(order) + 1)
            code = int(counts.argmax())
            purity = counts[code] / win.size
            if code == 0 or purity < purity_min:
                continue
            tfrac = float(_slice(tissue, tscale, x0, y0, side, side).mean())
            if tfrac < tissue_min:
                continue
            out.append({"x0": x0, "y0": y0, "side0": side, "label": order[code - 1], "purity": round(float(purity), 4),
                        "tissue_frac": round(tfrac, 4)})
    return out


def cap_per_class(cands: list[dict[str, Any]], cap: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for label in sorted({c["label"] for c in cands}):
        group = [c for c in cands if c["label"] == label]
        if len(group) > cap:
            keep = sorted(rng.choice(len(group), size=cap, replace=False))
            group = [group[i] for i in keep]
        out += group
    return sorted(out, key=lambda c: (c["y0"], c["x0"]))


# ---- windows ----------------------------------------------------------------------------------------------------
def window_truth(win: np.ndarray, carcinoma_codes: Sequence[int], strip_frac: tuple[float, float],
                 cfg: Mapping[str, Any]) -> tuple[str | None, dict[str, str], dict[str, float]]:
    """(stratum or None if ambiguous, {margin: tumor|clear}, {margin|interior: carcinoma fraction}) of a window's
    label-mask slice. strip_frac = (strip width / window width, strip height / window height)."""
    h, w = win.shape
    carc = np.isin(win, list(carcinoma_codes))
    sw = max(1, int(round(w * strip_frac[0])))
    sh = max(1, int(round(h * strip_frac[1])))
    names = cfg["margin"]["names"]
    strips = {names["top"]: carc[:sh, :], names["bottom"]: carc[h - sh:, :],
              names["left"]: carc[:, :sw], names["right"]: carc[:, w - sw:]}
    fracs = {m: float(s.mean()) for m, s in strips.items()}
    interior = carc[sh:h - sh, sw:w - sw]
    fracs["interior"] = float(interior.mean()) if interior.size else 0.0
    thr = float(cfg["wsi"]["margin_tumor_min_frac"])
    truth: dict[str, str] = {}
    for m in schemas.MARGINS:
        if fracs[m] >= thr:
            truth[m] = "tumor"
        elif fracs[m] == 0.0:
            truth[m] = "clear"
        else:
            return None, {}, fracs
    if any(t == "tumor" for t in truth.values()):
        return "edge_tumor", truth, fracs
    if fracs["interior"] >= float(cfg["wsi"]["central_tumor_min_frac"]):
        return "central_only", truth, fracs
    if not carc.any():
        return "none", truth, fracs
    return None, {}, fracs


def select_windows(cands: list[dict[str, Any]], per_stratum: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Seeded, non-overlapping, up to per_stratum per stratum (rarer strata first)."""
    chosen: list[dict[str, Any]] = []

    def overlaps(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
        return not (a["x0"] + a["w0"] <= b["x0"] or b["x0"] + b["w0"] <= a["x0"]
                    or a["y0"] + a["h0"] <= b["y0"] or b["y0"] + b["h0"] <= a["y0"])

    for stratum in STRATA:
        group = [c for c in cands if c["stratum"] == stratum]
        taken = 0
        for i in rng.permutation(len(group)):
            if taken >= per_stratum:
                break
            if not any(overlaps(group[i], c) for c in chosen):
                chosen.append(group[i])
                taken += 1
    return chosen


# ---- folds ------------------------------------------------------------------------------------------------------
def assign_folds(patients: Sequence[str], cfg: Mapping[str, Any]) -> dict[str, int]:
    uniq = sorted(set(patients))
    perm = np.random.default_rng(int(cfg["wsi"]["seed"])).permutation(len(uniq))
    k = int(cfg["wsi"]["folds"])
    return {uniq[j]: rank % k for rank, j in enumerate(perm)}


# ---- rasterizer check -------------------------------------------------------------------------------------------
def gt_thumbnail_agreement(dims0: tuple[int, int], regions: Sequence[Mapping[str, Any]], thumb: Path,
                           cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """Pixel agreement of our rasterization with the official gt_thumbnails PNG over pixels annotated in either."""
    if not thumb.is_file():
        return None
    gt = np.asarray(Image.open(thumb).convert("RGB"), dtype=np.int16)
    h, w = gt.shape[:2]
    ours, _ = label_mask(dims0, regions, (w, h), cfg)
    order = list(cfg["wsi"]["paint_order"])
    lit = gt.max(axis=2) > 127
    gt_code = np.zeros((h, w), dtype=np.uint8)
    for name, channel in GT_COLORS.items():
        if name in order:
            gt_code[lit & (gt.argmax(axis=2) == channel)] = order.index(name) + 1
    either = (gt_code > 0) | (ours > 0)
    n = int(either.sum())
    return {"pixel_agreement": round(float((gt_code[either] == ours[either]).mean()), 4) if n else None,
            "n_annotated_px": n, "thumb_size": [w, h]}


# ---- per-slide work ---------------------------------------------------------------------------------------------
def extract_slide(task: Mapping[str, Any]) -> dict[str, Any]:
    cfg, root, force = task["cfg"], Path(task["data_root"]), task["force"]
    wsi_id, patient, fold, group = task["wsi_id"], task["patient_id"], task["fold"], task["group"]
    wcfg = cfg["wsi"]
    rng = np.random.default_rng(common.stable_seed(wcfg["seed"], wsi_id))
    mpp0, regions = parse_bach_xml(task["xml"], cfg)
    slide = SLIDE_OPENER(task["svs"])
    dims0 = tuple(int(v) for v in slide.dimensions)
    mlevel = best_level(slide, float(wcfg["mask_downsample"]))
    mdims = tuple(int(v) for v in slide.level_dimensions[mlevel])
    mask, overlaps = label_mask(dims0, regions, mdims, cfg)
    mscale = (mdims[0] / dims0[0], mdims[1] / dims0[1])
    tlevel = best_level(slide, float(wcfg["tissue_downsample"]))
    tdims = tuple(int(v) for v in slide.level_dimensions[tlevel])
    tissue = tissue_mask(slide, tlevel, cfg)
    tscale = (tdims[0] / dims0[0], tdims[1] / dims0[1])
    target = float(wcfg["target_mpp"])
    png_level = int(cfg["tiles"]["png_compress_level"])
    written = 0

    tiles = []
    for grid in wcfg["tile_grids"]:
        size = int(cfg["tiles"]["grids"][grid]["size"])
        s0 = side0(size, target, mpp0)
        cands = tile_candidates(mask, mscale, tissue, tscale, dims0, s0, cfg)
        for c in cap_per_class(cands, int(wcfg["max_tiles_per_class_per_wsi"][grid]), rng):
            rel = f"wsi/tiles/{grid}/{wsi_id}/{c['x0']}_{c['y0']}.png"
            if force or not (root / rel).exists():
                img = read_rgb(slide, (c["x0"], c["y0"]), 0, (s0, s0)).resize((size, size), Image.LANCZOS)
                io.save_image(root / rel, img, compress_level=png_level)
                written += 1
            tiles.append({"tile_id": f"WSI_{wsi_id}_{grid}_{c['x0']}_{c['y0']}", "wsi_id": wsi_id,
                          "patient_id": patient, "fold": fold, "grid": grid, **c, "path": rel})

    win_w, win_h = (int(v) for v in wcfg["window"])
    w0, h0 = side0(win_w, target, mpp0), side0(win_h, target, mpp0)
    strip_px = int(cfg["margin"]["strip_px"])
    strip_frac = (strip_px / win_w, strip_px / win_h)
    carc_codes = [list(wcfg["paint_order"]).index(c) + 1 for c in cfg["carcinoma_classes"] if c in wcfg["paint_order"]]
    sx, sy = max(1, int(round(w0 * float(wcfg["window_stride_frac"])))), max(1, int(round(h0 * float(wcfg["window_stride_frac"]))))
    cands = []
    n_ambiguous = 0
    for y in range(0, dims0[1] - h0 + 1, sy):
        for x in range(0, dims0[0] - w0 + 1, sx):
            if float(_slice(tissue, tscale, x, y, w0, h0).mean()) < float(wcfg["window_min_tissue_frac"]):
                continue
            stratum, truth, fracs = window_truth(_slice(mask, mscale, x, y, w0, h0), carc_codes, strip_frac, cfg)
            if stratum is None:
                n_ambiguous += 1
                continue
            cands.append({"x0": x, "y0": y, "w0": w0, "h0": h0, "stratum": stratum, "truth": truth, "fracs": fracs})
    chosen = sorted(select_windows(cands, int(wcfg["windows_per_stratum"]), rng), key=lambda c: (c["y0"], c["x0"]))
    windows = []
    for k, c in enumerate(chosen):
        window_id = schemas.check_case_id(f"WSI_{wsi_id}_{k:02d}")
        rel = f"wsi/windows/{window_id}.png"
        if force or not (root / rel).exists():
            img = read_rgb(slide, (c["x0"], c["y0"]), 0, (w0, h0)).resize((win_w, win_h), Image.LANCZOS)
            io.save_image(root / rel, img, compress_level=png_level)
            written += 1
        windows.append({"window_id": window_id, "wsi_id": wsi_id, "patient_id": patient, "fold": fold,
                        "x": int(round(c["x0"] * mpp0 / target)), "y": int(round(c["y0"] * mpp0 / target)),
                        "width": win_w, "height": win_h, "stratum": c["stratum"], "path": rel,
                        **{f"truth_{m}": c["truth"][m] for m in schemas.MARGINS}, "group": group})
    thumb = Path(task["xml"]).parent / "gt_thumbnails" / f"{wsi_id}.png"
    stats = {"wsi_id": wsi_id, "mpp_xml": mpp0, "dims0": list(dims0), "mask_level": mlevel, "n_regions": len(regions),
             "regions_per_label": {n: sum(r["label"] == n for r in regions) for n in wcfg["paint_order"]},
             "overlap_px": overlaps, "tiles_per_grid_label": {}, "window_candidates": {s: sum(c["stratum"] == s for c in cands) for s in STRATA},
             "windows_ambiguous": n_ambiguous, "windows_per_stratum": {s: sum(w["stratum"] == s for w in windows) for s in STRATA},
             "gt_thumbnail_check": gt_thumbnail_agreement(dims0, regions, thumb, cfg), "files_written": written}
    for t in tiles:
        key = f"{t['grid']}:{t['label']}"
        stats["tiles_per_grid_label"][key] = stats["tiles_per_grid_label"].get(key, 0) + 1
    return {"tiles": tiles, "windows": windows, "stats": stats, "group": group}


# ---- manifest ---------------------------------------------------------------------------------------------------
def manifest_rows(tiles: pd.DataFrame, windows: pd.DataFrame, groups: Mapping[str, str],
                  cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in tiles.itertuples(index=False):
        size = int(cfg["tiles"]["grids"][t.grid]["size"])
        rows.append(common.manifest_row(set_name=TILE_SET[t.grid], case_id=schemas.check_case_id(t.tile_id), split="",
                                        grid=t.grid, tile_idx=None, path=t.path, x=0, y=0, w=size, h=size, view=0,
                                        label=t.label, group=groups[t.wsi_id]))
    n_views = int(cfg["encoders"]["views"]["g256"])
    for w in windows.itertuples(index=False):
        for box in common.grid_boxes(cfg, "g256"):
            first = common.manifest_row(set_name="wsi_margin256", case_id=w.window_id, split="", grid="g256",
                                        tile_idx=box["tile_idx"], path=w.path, x=box["x"], y=box["y"], w=box["size"],
                                        h=box["size"], view=0, group=groups[w.wsi_id])
            rows.append(first)
            rows += [{**first, "row_id": f"{first['item_id']}|v{v}", "view": v} for v in range(1, n_views)]
    return rows


def run_slides(tasks: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(tasks) <= 1:
        return [extract_slide(t) for t in tasks]
    with multiprocessing.get_context().Pool(min(workers, len(tasks))) as pool:
        return list(pool.imap_unordered(extract_slide, tasks))


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Labeled WSI tiles + margin windows from annotated BACH WSIs (§A9).")
    common.add_source_args(parser, ("bach_root",))
    parser.add_argument("--workers", type=int, default=None, help="processes (default $SLURM_CPUS_PER_TASK or 1)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    try:
        bach_root = common.source_root(cfg, args, "bach_root")
        wsi_dir = common.bach_wsi_dir(bach_root, cfg)
        xmls = sorted(p for p in wsi_dir.glob("*.xml") if (wsi_dir / f"{p.stem}.svs").is_file())
        if not xmls:
            raise FileNotFoundError(f"no annotated slides (*.xml with a matching .svs) under {wsi_dir}")
        if ctx.limit is not None:
            xmls = xmls[: ctx.limit]
        pfile = common.patient_file(bach_root, cfg, "wsi")
        if pfile.is_file():
            pmap = {stem: str(v["patient"]) for stem, v in common.load_wsi_patients(pfile).items()}
            missing = [p.stem for p in xmls if p.stem not in pmap]
            if missing:
                raise ValueError(f"WSI patient workbook lacks slides {missing}")
            patient_source = str(pfile)
        else:
            pmap = {p.stem: "" for p in xmls}
            patient_source = ("fallback: WSI patient workbook missing; one group per annotated slide (each annotated "
                              "slide is a distinct patient per data_formats.md §1.3)")
        group_of = {p.stem: (f"wsi_p{pmap[p.stem]}" if pmap[p.stem] else f"wsi_{p.stem}") for p in xmls}
        folds = assign_folds([group_of[p.stem] for p in xmls], cfg)
        tasks = [{"cfg": cfg, "data_root": str(root), "force": ctx.force, "wsi_id": p.stem, "xml": str(p),
                  "svs": str(wsi_dir / f"{p.stem}.svs"), "patient_id": pmap[p.stem], "group": group_of[p.stem],
                  "fold": folds[group_of[p.stem]]} for p in xmls]
        results = sorted(run_slides(tasks, common.worker_count(args.workers)), key=lambda r: r["stats"]["wsi_id"])
        tiles = pd.DataFrame([t for r in results for t in r["tiles"]], columns=TILE_COLUMNS)
        windows = pd.DataFrame([w for r in results for w in r["windows"]], columns=WINDOW_COLUMNS + ["group"])
        schemas.validate_frame(schemas.WSI_WINDOWS, windows[WINDOW_COLUMNS])
        rows = manifest_rows(tiles, windows, {r["stats"]["wsi_id"]: r["group"] for r in results}, cfg)
    except (ValueError, FileNotFoundError, KeyError, ET.ParseError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2
    io.write_csv(root / "wsi_tiles.csv", tiles, columns=TILE_COLUMNS)
    io.write_csv(schemas.wsi_windows_csv(root), windows[WINDOW_COLUMNS], columns=WINDOW_COLUMNS)
    io.write_json(root / "wsi_folds.json", folds)
    common.replace_manifest_sets(root, SETS, rows)
    summary = {
        "n_slides": len(results), "folds": folds, "patient_source": patient_source,
        "tiles_per_grid_label": tiles.groupby(["grid", "label"]).size().rename("n").reset_index().to_dict("records"),
        "windows_per_stratum": windows["stratum"].value_counts().to_dict(),
        "manifest_rows": {s: sum(r["set"] == s for r in rows) for s in SETS},
        "per_slide": [r["stats"] for r in results],
    }
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg, inputs=[str(wsi_dir), patient_source],
                  outputs=[root / "wsi_tiles.csv", schemas.wsi_windows_csv(root), root / "wsi_folds.json", root / "wsi",
                           schemas.embed_manifest_csv(root)],
                  args={"limit": ctx.limit, "force": ctx.force, "bach_root": str(bach_root)}, synthetic=ctx.synthetic,
                  extra={"summary": summary, "caveats": [
                      "unannotated != normal: 'clear'/'none' mean no annotated carcinoma, never verified tumor-free",
                      "mpp from each XML (0.466667); the SVS header says 0.5 (comma decimal) and is not used; "
                      "if 0.5 were right, physical sizes would be ~7% off",
                      "WSI labels carry scanner shift vs the microscopy photos; microscopy test metrics stay WSI-free (§A9)"]})
    print(json.dumps({k: v for k, v in summary.items() if k != "per_slide"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
