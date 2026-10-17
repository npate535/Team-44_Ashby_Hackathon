"""Atomic file writes and provenance for every pipeline step (BUILD_PLAN §C.2: idempotent, resumable,
atomic writes, and a meta.json with inputs, config hash, git SHA and versions).

Every writer creates parent directories, writes to a hidden temporary file in the destination directory
and renames it into place with os.replace, so a crash never leaves a partial output that a resumed run
would mistake for a finished one. Temporary names start with "." and end in ".tmp". Files get the normal
umask permissions, so teammates can read outputs in the shared team folder (no keys there, ever).

    write_json / read_json        JSON (NaN and Infinity rejected; numpy scalars/arrays converted)
    write_csv / read_csv          pandas CSV without the index column
    save_npy / save_npz / load_npz numpy arrays (allow_pickle=False), exact file names kept
    save_image                    PIL image, format from the destination suffix
    atomic_path                   context manager for any other writer
    provenance / write_meta       meta.json content; outputs_exist for skip-if-done checks

Portable: Python >= 3.10; stdlib + numpy (pandas and Pillow are imported only by their helpers).
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

VERSION_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "pillow", "pyyaml", "torch", "timm", "transformers",
    "open_clip_torch", "pydantic",
)
IMAGE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- atomic primitives ------------------------------------------------------------------------------------
@contextlib.contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path next to `path`; on a clean exit rename it onto `path`, on error delete it.

    The caller must create the temporary file inside the block (raises FileNotFoundError otherwise).
    """
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"nothing was written for {final}")
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)
    return Path(path)


def atomic_write_text(path: str | Path, text: str) -> Path:
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return Path(path)


# ---- JSON ----------------------------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | Path, obj: Any, *, indent: int | None = 2, sort_keys: bool = False) -> Path:
    """Atomically write strict JSON. NaN/Infinity raise ValueError before anything touches the disk."""
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys, allow_nan=False, ensure_ascii=False, default=_json_default)
    return atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---- tables and arrays ---------------------------------------------------------------------------------------
def write_csv(path: str | Path, df: Any, *, columns: Sequence[str] | None = None) -> Path:
    """Atomically write a pandas DataFrame as plain CSV without the index column."""
    with atomic_path(path) as tmp:
        df.to_csv(tmp, index=False, columns=list(columns) if columns is not None else None)
    return Path(path)


def read_csv(path: str | Path, **kwargs: Any) -> Any:
    import pandas as pd

    return pd.read_csv(path, **kwargs)


def save_npy(path: str | Path, array: Any) -> Path:
    """Atomically np.save to exactly `path` (no extra .npy suffix appended)."""
    with atomic_path(path) as tmp, open(tmp, "wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    return Path(path)


def save_npz(path: str | Path, *, compressed: bool = False, **arrays: Any) -> Path:
    """Atomically write named arrays to exactly `path`. String arrays must be fixed-width unicode (no objects)."""
    writer = np.savez_compressed if compressed else np.savez
    with atomic_path(path) as tmp, open(tmp, "wb") as handle:
        writer(handle, **{name: np.asarray(value) for name, value in arrays.items()})
    return Path(path)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_image(path: str | Path, image: Any, **save_kwargs: Any) -> Path:
    """Atomically save a PIL image; the format comes from the destination suffix (.png/.jpg/.jpeg/.tif)."""
    final = Path(path)
    fmt = IMAGE_FORMATS.get(final.suffix.lower())
    if fmt is None:
        raise ValueError(f"unsupported image suffix {final.suffix!r}; expected one of {sorted(IMAGE_FORMATS)}")
    with atomic_path(final) as tmp:
        image.save(tmp, format=fmt, **save_kwargs)
    return final


def outputs_exist(*paths: str | Path) -> bool:
    """True when at least one path is given and every path exists (skip-if-done check for resumable steps)."""
    return bool(paths) and all(Path(p).exists() for p in paths)


# ---- provenance -----------------------------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(args: list[str], repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_sha(repo: str | Path | None = None) -> str | None:
    """HEAD commit of the repo, or None outside a git checkout (e.g. a tarball on Delta)."""
    return _git(["rev-parse", "HEAD"], Path(repo) if repo else REPO_ROOT) or None


def git_dirty(repo: str | Path | None = None) -> bool | None:
    status = _git(["status", "--porcelain", "--untracked-files=no"], Path(repo) if repo else REPO_ROOT)
    return None if status is None else bool(status)


def package_versions(names: Sequence[str] = VERSION_PACKAGES) -> dict[str, str | None]:
    """Installed version per distribution name (None when not installed), plus the Python version."""
    versions: dict[str, str | None] = {"python": ".".join(map(str, sys.version_info[:3]))}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def provenance(cfg: Mapping[str, Any], *, synthetic: bool = False) -> dict[str, Any]:
    """Common provenance fields for meta.json and eval JSON envelopes."""
    from core.config import config_hash  # local import keeps core.io free of import cycles

    return {
        "created_utc": utc_now(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(cfg),
        "versions": package_versions(),
        "platform": platform.platform(),
        "synthetic": bool(synthetic),
    }


def write_meta(
    path: str | Path,
    *,
    step: str,
    cfg: Mapping[str, Any],
    inputs: Sequence[str | Path],
    outputs: Sequence[str | Path],
    args: Mapping[str, Any] | None = None,
    synthetic: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write meta/<step>.json (core.schemas.STEP_META): step, inputs, outputs, args + provenance (+ extra)."""
    meta = {
        "step": step,
        "inputs": [Path(p).as_posix() if isinstance(p, Path) else str(p) for p in inputs],
        "outputs": [Path(p).as_posix() if isinstance(p, Path) else str(p) for p in outputs],
        "args": dict(args or {}),
        **provenance(cfg, synthetic=synthetic),
        **dict(extra or {}),
    }
    return write_json(path, meta)
