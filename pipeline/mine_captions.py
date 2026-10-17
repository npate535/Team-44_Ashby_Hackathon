"""Mine Quilt-1M (and PathGen, when patch images exist) captions for concept positives/negatives
(BUILD_PLAN §A11, Task 2; CPU job prep_b). Draft synonym lists; caption co-mention != tissue truth.

Rules live in config/caption_mining.yaml (settings caption_mining.file); the concept ids must equal settings
concepts.ids. Quilt: the lookup CSV (2.2 GB) is read in chunks with `usecols`, never whole. Rows are kept when
not_histology == 0, the image is breast (Quilt `pathology` list or caption regex), and then matched per concept
(positive regex not negated and not excluded; negative regex = captions affirmatively describing contrasting
tissue). Aggregation is per image: positive if any caption is; negative if any caption is negative and none is
positive; any caption hitting the stain/modality filter drops the image. Images are located by trying
images_part_<k>/quilt_1m/<image_path> (only for selected candidates, no full directory scan); missing files are
counted and skipped. Caps per concept and polarity, and per leakage group (Quilt video = 11-char YouTube id prefix;
Twitter thread = tweet id; PubMed/LAION = the image itself), are applied with a seeded shuffle.

PathGen: used only when <data_root>/inventory.json says pathgen.has_patch_images; patches are matched to
{wsi_id, position, caption} entries by the README crop name; group = TCGA slide.

Manifest rows (sets quilt / pathgen): one row per (image, concept, polarity), label '<concept_id>:pos|neg',
group = leakage group (so held-out AUROC splits by video/slide), absolute image path, view 0, whole image.
item_id = '<set>:<image_id>:<concept_id>_<pos|neg>' (opaque), so an image that is, say, an invasion negative and a
duct_confined positive appears in two rows. Report: meta/mine_captions.json (counts per concept and polarity,
groups, filter counts, example captions).

Images from Quilt/PathGen are used only to fit concept vectors; they never appear in the deck (§A11).
--limit N caps each concept/polarity at N images (smoke runs). Recomputed on every run (CSV scan; no per-item
outputs to resume).

Run: python -m pipeline.mine_captions [--limit 5] [--quilt-root ...] [--pathgen-root ...]
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from core import cli, config, io, schemas
from pipeline import common

STEP = "mine_captions"
SETS = ("quilt", "pathgen")
VIDEO = re.compile(r"^([A-Za-z0-9_-]{11})_image_")
SAFE = re.compile(r"[^A-Za-z0-9_\-]")


# ---- rules ------------------------------------------------------------------------------------------------------
class Rules:
    """Compiled caption_mining.yaml."""

    def __init__(self, data: Mapping[str, Any], cfg: Mapping[str, Any]):
        ids = list(cfg["concepts"]["ids"])
        if sorted(data["concepts"]) != sorted(ids):
            raise ValueError(f"caption_mining.yaml concepts {sorted(data['concepts'])} != settings concepts.ids {ids}")
        flags = re.IGNORECASE
        self.data = data
        self.ids = ids
        self.organ_prefixes = [p.lower() for p in data["organ_filter"]["pathology_prefixes"]]
        self.organ = re.compile(data["organ_filter"]["caption_regex"], flags)
        self.stain = re.compile(data["stain_filter"]["exclude_regex"], flags)
        self.negation = re.compile(data["negation"]["cues"], flags)
        self.window = int(data["negation"]["window_words"])
        self.pos = {c: [re.compile(p, flags) for p in data["concepts"][c]["positive"]] for c in ids}
        self.exc = {c: [re.compile(p, flags) for p in data["concepts"][c].get("exclude", [])] for c in ids}
        self.neg = {c: [re.compile(p, flags) for p in data["concepts"][c]["negative"]] for c in ids}
        self.caps = data["caps"]

    def negated(self, text: str, start: int) -> bool:
        before = text[:start].split()[-self.window:]
        return bool(self.negation.search(" ".join(before)))

    def classify(self, caption: str) -> dict[str, str]:
        """{concept: 'pos' | 'neg'} for one caption (concepts with neither are absent)."""
        out: dict[str, str] = {}
        for c in self.ids:
            pos = any(not self.negated(caption, m.start()) for rx in self.pos[c] for m in rx.finditer(caption))
            if pos and not any(rx.search(caption) for rx in self.exc[c]):
                out[c] = "pos"
            elif not pos and any(rx.search(caption) for rx in self.neg[c]):
                out[c] = "neg"
        return out

    def is_breast(self, pathology: str, caption: str) -> bool:
        return pathology_is_breast(pathology, self.organ_prefixes) or bool(self.organ.search(caption))


def load_rules(cfg: Mapping[str, Any], path: str | Path | None = None) -> Rules:
    path = Path(path) if path else config.resolve_repo_path(cfg["caption_mining"]["file"])
    with open(path, encoding="utf-8") as handle:
        return Rules(yaml.safe_load(handle), cfg)


def pathology_is_breast(pathology: str, prefixes: Iterable[str]) -> bool:
    text = (pathology or "").strip()
    if not text or not any(p in text.lower() for p in prefixes):  # cheap pre-check (~1M rows)
        return False
    try:
        values = ast.literal_eval(text) if text.startswith("[") else [text]
    except (ValueError, SyntaxError):
        values = [text]
    return any(str(v).strip().lower().startswith(p) for v in values for p in prefixes)


def quilt_group(subset: str, image_path: str) -> str:
    m = VIDEO.match(image_path)
    if subset == "quilt" and m:
        return "yt:" + m.group(1)
    if subset == "openpath":
        return "tw:" + image_path.split("_")[0]
    return f"{subset or 'unknown'}:{image_path}"


def safe_image_id(stem: str) -> str:
    """A schema-valid case_id from an image file stem (characters outside [A-Za-z0-9_-] become '-')."""
    cid = SAFE.sub("-", stem).lstrip("-_") or "img"
    if len(cid) > 120:
        cid = f"{cid[:100]}-{common.stable_seed(stem):010d}"
    return schemas.check_case_id(cid)


# ---- aggregation ------------------------------------------------------------------------------------------------
class ImageVotes:
    """Per-image state across chunks: dropped by the stain filter, concept votes, example captions."""

    def __init__(self) -> None:
        self.images: dict[str, dict[str, Any]] = {}

    def add(self, key: str, group: str, caption: str, rules: Rules) -> None:
        state = self.images.setdefault(key, {"group": group, "dropped": False, "votes": {}, "captions": []})
        if rules.stain.search(caption):
            state["dropped"] = True
            return
        for concept, pol in rules.classify(caption).items():
            state["votes"].setdefault(concept, set()).add(pol)
        if len(state["captions"]) < 2:
            state["captions"].append(caption[:200])

    def labels(self) -> dict[str, dict[str, str]]:
        """{image: {concept: pos|neg}} after image-level aggregation (positive wins; dropped images removed)."""
        out = {}
        for key, state in self.images.items():
            if state["dropped"]:
                continue
            lab = {c: ("pos" if "pos" in v else "neg") for c, v in state["votes"].items()}
            if lab:
                out[key] = lab
        return out


def select(candidates: list[tuple[str, str]], cap: int, per_group: int, rng: np.random.Generator) -> list[str]:
    """candidates = [(image key, group)] sorted; seeded shuffle; take up to cap with at most per_group per group."""
    taken, per = [], {}
    for i in rng.permutation(len(candidates)):
        key, group = candidates[int(i)]
        if per.get(group, 0) >= per_group:
            continue
        taken.append(key)
        per[group] = per.get(group, 0) + 1
        if len(taken) >= cap:
            break
    return sorted(taken)


# ---- Quilt ------------------------------------------------------------------------------------------------------
def find_quilt_csv(root: Path, rules: Rules) -> Path:
    want = rules.data["quilt"]["csv_name_contains"]
    csvs = sorted((p for p in root.glob("*.csv") if want in p.name), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"no *.csv containing {want!r} in {root}")
    return csvs[0]


def scan_quilt(csv_path: Path, rules: Rules, stats: dict[str, int]) -> ImageVotes:
    q = rules.data["quilt"]
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    missing = [c for c in ("caption", "image_path") if c not in header]
    if missing:
        raise ValueError(f"{csv_path.name} lacks columns {missing}; header {header}")
    usecols = [c for c in q["usecols"] if c in header]
    votes = ImageVotes()
    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=str, keep_default_na=False, chunksize=int(q["chunksize"])):
        stats["rows_scanned"] += len(chunk)
        if q["require_not_histology_zero"] and "not_histology" in chunk.columns:
            chunk = chunk[chunk["not_histology"].isin(["0", "0.0"])]
        stats["rows_histology"] += len(chunk)
        pathology = chunk["pathology"] if "pathology" in chunk.columns else pd.Series("", index=chunk.index)
        roi = chunk["roi_text"] if "roi_text" in chunk.columns else pd.Series("", index=chunk.index)
        subset = chunk["subset"] if "subset" in chunk.columns else pd.Series("", index=chunk.index)
        for cap, img, path_txt, roi_txt, sub in zip(chunk["caption"], chunk["image_path"], pathology, roi, subset):
            text = f"{cap} {roi_txt if roi_txt not in ('', '[]') else ''}".strip()
            if not img or not rules.is_breast(path_txt, text):
                continue
            stats["rows_breast"] += 1
            votes.add(img, quilt_group(sub, img), text, rules)
    return votes


def locate_quilt(root: Path, image_path: str, rules: Rules) -> Path | None:
    for part in rules.data["quilt"]["parts"]:
        for pattern in rules.data["quilt"]["image_dirs"]:
            p = root / pattern.format(part=part) / image_path
            if p.is_file():
                return p
    return None


# ---- PathGen ----------------------------------------------------------------------------------------------------
def scan_pathgen(root: Path, rules: Rules, stats: dict[str, int]) -> tuple[ImageVotes, dict[str, Path]]:
    """Captions of entries whose README-named patch exists anywhere under root."""
    patches = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in common.IMAGE_EXTS and p.is_file()}
    with open(root / rules.data["pathgen"]["json_name"], encoding="utf-8") as handle:
        entries = json.load(handle)
    votes = ImageVotes()
    for e in entries:
        wsi_id = e.get("wsi_id") or e.get("WSI_id") or ""
        pos = e.get("position") or ["", ""]
        stem = rules.data["pathgen"]["patch_name"].format(wsi_id=wsi_id, x=pos[0], y=pos[1])
        stats["pathgen_entries"] += 1
        if stem not in patches:
            continue
        caption = e.get("caption") or ""
        if not rules.is_breast("", caption):
            continue
        stats["pathgen_breast_with_patch"] += 1
        votes.add(stem, f"tcga:{wsi_id}", caption, rules)
    return votes, patches


# ---- rows -------------------------------------------------------------------------------------------------------
def build_rows(set_name: str, votes: ImageVotes, locate: Any, rules: Rules, cfg: Mapping[str, Any],
               limit: int | None, stats: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = votes.labels()
    rng = np.random.default_rng(int(rules.data["seed"]))
    caps = rules.caps
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    located: dict[str, Path | None] = {}
    for concept in rules.ids:
        report[concept] = {}
        for pol, cap_key in (("pos", "positives_per_concept"), ("neg", "negatives_per_concept")):
            cap = int(caps[cap_key]) if limit is None else min(int(caps[cap_key]), limit)
            cands = sorted((k, votes.images[k]["group"]) for k, lab in labels.items() if lab.get(concept) == pol)
            # locate lazily, in shuffled order, until the cap is met (missing files are skipped)
            chosen, per_group = [], {}
            for i in rng.permutation(len(cands)):
                key, group = cands[int(i)]
                if per_group.get(group, 0) >= int(caps["max_images_per_group"]):
                    continue
                if key not in located:
                    located[key] = locate(key)
                    if located[key] is None:
                        stats[f"{set_name}_missing_files"] += 1
                if located[key] is None:
                    continue
                chosen.append(key)
                per_group[group] = per_group.get(group, 0) + 1
                if len(chosen) >= cap:
                    break
            for key in sorted(chosen):
                path = located[key]
                with Image.open(path) as im:
                    w, h = im.size
                row = common.manifest_row(set_name=set_name, case_id=safe_image_id(Path(key).stem), split="", grid="",
                                          tile_idx=None, path=str(Path(path).resolve()), x=0, y=0, w=w, h=h, view=0,
                                          label=f"{concept}:{pol}", group=votes.images[key]["group"])
                row["item_id"] = f"{row['item_id']}:{concept}_{pol}"
                row["row_id"] = f"{row['item_id']}|v0"
                rows.append(row)
            report[concept][pol] = {"candidates": len(cands), "selected": len(chosen),
                                    "groups": len({votes.images[k]["group"] for k in chosen}),
                                    "examples": [votes.images[k]["captions"][0] for k in sorted(chosen)[:3]
                                                 if votes.images[k]["captions"]]}
    return rows, report


def main(argv: list[str] | None = None) -> int:
    parser = cli.step_parser(f"pipeline.{STEP}", "Mine Quilt-1M / PathGen captions for concept images (§A11).")
    common.add_source_args(parser, ("quilt_root", "pathgen_root"))
    parser.add_argument("--rules", default=None, help="caption_mining.yaml (default settings caption_mining.file)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    cfg, root = ctx.cfg, ctx.data_root
    stats = {k: 0 for k in ("rows_scanned", "rows_histology", "rows_breast", "quilt_missing_files", "pathgen_entries",
                            "pathgen_breast_with_patch", "pathgen_missing_files")}
    try:
        rules = load_rules(cfg, args.rules)
        quilt_root = common.source_root(cfg, args, "quilt_root")
        csv_path = find_quilt_csv(quilt_root, rules)
        votes = scan_quilt(csv_path, rules, stats)
        q_rows, q_report = build_rows("quilt", votes, lambda k: locate_quilt(quilt_root, k, rules), rules, cfg,
                                      ctx.limit, stats)
        inv_path = schemas.inventory_json(root)
        inv = io.read_json(inv_path) if inv_path.is_file() else {}
        use_pathgen = bool(inv.get("pathgen", {}).get("has_patch_images"))
        p_rows, p_report = [], {"status": "skipped: inventory.json reports no PathGen patch images"}
        if use_pathgen:
            pathgen_root = common.source_root(cfg, args, "pathgen_root")
            pvotes, patches = scan_pathgen(pathgen_root, rules, stats)
            p_rows, p_report = build_rows("pathgen", pvotes, lambda k: patches.get(k), rules, cfg, ctx.limit, stats)
    except (ValueError, FileNotFoundError, KeyError, yaml.YAMLError) as exc:
        print(f"ERROR {STEP}: {exc}", file=sys.stderr)
        return 2
    common.replace_manifest_sets(root, SETS, q_rows + p_rows)
    summary = {"stats": stats, "quilt_csv": str(csv_path), "pathgen_used": use_pathgen,
               "manifest_rows": {"quilt": len(q_rows), "pathgen": len(p_rows)},
               "per_concept": {c: {p: q_report[c][p]["selected"] for p in ("pos", "neg")} for c in rules.ids}}
    io.write_meta(schemas.step_meta_json(root, STEP), step=STEP, cfg=cfg,
                  inputs=[str(csv_path), str(config.resolve_repo_path(args.rules or cfg["caption_mining"]["file"]))],
                  outputs=[schemas.embed_manifest_csv(root)], args={"limit": ctx.limit, "force": ctx.force},
                  synthetic=ctx.synthetic,
                  extra={"summary": summary, "quilt_report": q_report, "pathgen_report": p_report,
                         "note": "draft synonym lists; caption co-mention != tissue truth (caption-consistent, not validated)"})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
