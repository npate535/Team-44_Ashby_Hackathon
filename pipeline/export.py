"""Pack the Delta data root into one archive for the laptop, and unpack it with hash verification (Task 3; Agent 2).

    python -m pipeline.export [--out <file.tgz>] [--no-views] [--dry-run]         # on Delta (login node is fine)
    python -m pipeline.export --unpack <file.tgz> --dest data                      # on the laptop

Contents (paths relative to the data root, kept as is):
    emb/<encoder>/<set>_<head>.npy + _index.csv + meta.json   per-item (TTA-reduced) arrays, plus the *_views files
                                                              (05_risk_model train augmentation and the ruling-M9
                                                              clean view-0 baselines of 06_reliability need them;
                                                              --no-views drops them); float16 chunk folders never
    text/*.npz                                                prompt embeddings
    top-level *.csv / *.json                                  cases, tiles, composites, wsi_windows, inventory,
                                                              cell_tiles, cell_features*, tile_risk_oof, SYNTHETIC
                                                              marker, ... (embed_manifest.csv only with --with-manifest)
    meta/, display/, cells/                                   step metadata, display JPEGs, cell outputs
Never: tiles/, composites/ images, smoke/, logs/, exports/, chunk folders. BACH-derived images (display JPEGs, cell
overlays) travel only to the team's private laptop data root; they are git-ignored and never published.

The archive carries EXPORT_MANIFEST.json ({created_utc, git_sha, source_root, with_views, files: [{path, bytes,
sha256}]}) and SHA256SUMS. --unpack refuses unsafe member paths (absolute, '..', links), extracts into a temporary
folder inside --dest, verifies every size and sha256 against the manifest, then moves the files into place; any
mismatch exits 2 and leaves --dest unchanged.

Portable: Python >= 3.10, stdlib + core (runs on a Delta login node: file I/O only).
"""

from __future__ import annotations

import hashlib
import io as _io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from core import cli, io, schemas

STEP = "pipeline.export"
MANIFEST_NAME = "EXPORT_MANIFEST.json"
SUMS_NAME = "SHA256SUMS"
TREE_DIRS = ("meta", "display", "cells")
EXCLUDED_TOP = ("embed_manifest.csv",)


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(bufsize), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_under(folder: Path) -> Iterable[Path]:
    if folder.is_dir():
        yield from sorted(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()
                          and not any(part.startswith(".") for part in p.relative_to(folder).parts))


def collect(root: Path, *, with_views: bool = False, with_manifest: bool = False,
            encoders: Sequence[str] | None = None) -> list[Path]:
    """Relative paths of every file to export, sorted."""
    root = Path(root)
    out: list[Path] = []
    for path in sorted(root.glob("*")):
        if path.is_file() and path.suffix in (".csv", ".json") and not path.name.startswith("."):
            if path.name in EXCLUDED_TOP and not with_manifest:
                continue
            out.append(path)
    emb = root / "emb"
    for enc_dir in sorted(p for p in emb.glob("*") if p.is_dir()) if emb.is_dir() else []:
        if encoders and enc_dir.name not in encoders:
            continue
        for path in sorted(enc_dir.glob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            is_views = path.name.endswith(("_views.npy", "_views_index.csv"))
            if path.name == "meta.json" or (path.suffix in (".npy", ".csv") and (with_views or not is_views)):
                out.append(path)
    out += [p for p in sorted((root / "text").glob("*.npz")) if p.is_file()] if (root / "text").is_dir() else []
    for name in TREE_DIRS:
        out += list(_files_under(root / name))
    return [p.relative_to(root) for p in out]


def build_manifest(root: Path, rel_paths: Sequence[Path], *, with_views: bool) -> dict[str, Any]:
    files = [{"path": p.as_posix(), "bytes": (root / p).stat().st_size, "sha256": sha256_file(root / p)}
             for p in rel_paths]
    return {"created_utc": io.utc_now(), "git_sha": io.git_sha(), "source_root": str(root), "with_views": with_views,
            "n_files": len(files), "total_bytes": sum(f["bytes"] for f in files), "files": files}


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, _io.BytesIO(data))


def pack(root: Path, out: Path, rel_paths: Sequence[Path], *, with_views: bool) -> dict[str, Any]:
    manifest = build_manifest(root, rel_paths, with_views=with_views)
    sums = "".join(f"{f['sha256']}  {f['path']}\n" for f in manifest["files"]).encode("utf-8")
    with io.atomic_path(out) as tmp:
        with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
            _add_bytes(tar, MANIFEST_NAME, json.dumps(manifest, indent=1).encode("utf-8"))
            _add_bytes(tar, SUMS_NAME, sums)
            for rel in rel_paths:
                tar.add(root / rel, arcname=rel.as_posix(), recursive=False)
    return manifest


def _safe_member(member: tarfile.TarInfo) -> str:
    name = member.name
    pure = PurePosixPath(name)
    if not (member.isfile() or member.isdir()) or pure.is_absolute() or ".." in pure.parts or name.startswith("/"):
        raise ValueError(f"unsafe archive member {name!r}")
    return name


def unpack(archive: Path, dest: Path, *, log: Any = print) -> dict[str, Any]:
    """Extract, verify against EXPORT_MANIFEST.json, then move into dest. Raises ValueError on any mismatch."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest, prefix=".unpack_") as tmp_name:
        tmp = Path(tmp_name)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                _safe_member(member)
            for member in members:
                if member.isfile():
                    target = tmp / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, open(target, "wb") as dst:  # type: ignore[union-attr]
                        shutil.copyfileobj(src, dst)
        manifest_path = tmp / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError(f"{archive}: no {MANIFEST_NAME}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = {f["path"] for f in manifest["files"]}
        present = {p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file()} - {MANIFEST_NAME, SUMS_NAME}
        if present != listed:
            raise ValueError(f"archive files differ from the manifest: extra {sorted(present - listed)[:5]}, "
                             f"missing {sorted(listed - present)[:5]}")
        bad = [f["path"] for f in manifest["files"]
               if (tmp / f["path"]).stat().st_size != f["bytes"] or sha256_file(tmp / f["path"]) != f["sha256"]]
        if bad:
            raise ValueError(f"sha256/size mismatch for {len(bad)} file(s), e.g. {bad[:5]}")
        for f in manifest["files"]:
            target = dest / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp / f["path"], target)
        io.write_json(dest / "meta" / "export_unpack.json",
                      {k: v for k, v in manifest.items() if k != "files"} | {"archive": str(archive),
                                                                             "unpacked_utc": io.utc_now()})
    log(f"unpacked {manifest['n_files']} files ({manifest['total_bytes'] / 1e6:.1f} MB), all sha256 verified -> {dest}")
    return manifest


def main(argv: Sequence[str] | None = None, log: Any = print) -> int:
    parser = cli.step_parser(STEP, "Pack the data root for the laptop / unpack with sha256 verification.")
    parser.add_argument("--out", default=None, help="archive path (default <data-root>/exports/pathqueue_<utc>.tgz)")
    parser.add_argument("--no-views", action="store_true",
                        help="skip the *_views files (05/06 then lose train augmentation and M9 view-0 baselines)")
    parser.add_argument("--with-manifest", action="store_true", help="also pack embed_manifest.csv")
    parser.add_argument("--encoders", default=None, help="comma list of encoders to pack (default all)")
    parser.add_argument("--dry-run", action="store_true", help="print the file count and size, write nothing")
    parser.add_argument("--unpack", default=None, help="archive to unpack (laptop)")
    parser.add_argument("--dest", default=None, help="--unpack destination data root (default the data root)")
    args = parser.parse_args(argv)
    ctx = cli.context(args)
    if args.unpack:
        try:
            unpack(Path(args.unpack), Path(args.dest) if args.dest else ctx.data_root, log=log)
        except (ValueError, tarfile.TarError) as exc:
            log(f"ABORT: {exc}")
            return 2
        return 0
    root = ctx.data_root
    with_views = not args.no_views
    encoders = [e for e in (args.encoders or "").split(",") if e] or None
    rel = collect(root, with_views=with_views, with_manifest=args.with_manifest, encoders=encoders)
    total = sum((root / p).stat().st_size for p in rel)
    by_top: dict[str, int] = {}
    for p in rel:
        key = p.parts[0] if len(p.parts) > 1 else "(top-level)"
        by_top[key] = by_top.get(key, 0) + (root / p).stat().st_size
    log(f"export: {len(rel)} files, {total / 1e6:.1f} MB uncompressed: "
        + ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in sorted(by_top.items())))
    if not rel:
        log("ABORT: nothing to export")
        return 2
    if args.dry_run:
        return 0
    out = Path(args.out) if args.out else root / "exports" / f"pathqueue_{io.utc_now().replace(':', '')}.tgz"
    manifest = pack(root, out, rel, with_views=with_views)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB compressed, {manifest['n_files']} files, sha256 listed)")
    io.write_meta(schemas.step_meta_json(root, "export"), step="export", cfg=ctx.cfg, inputs=[str(root)],
                  outputs=[out], synthetic=ctx.synthetic,
                  args={"with_views": with_views, "encoders": encoders, "out": str(out)},
                  extra={"n_files": manifest["n_files"], "total_bytes": manifest["total_bytes"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
