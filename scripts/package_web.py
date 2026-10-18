"""Package only the web application's source. Credentials, datasets, databases and browser traces are excluded."""
from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("backend", "core", "agent", "config", "deploy", "fixtures", "team44-triage-frontend/app")
SOURCE_FILES = ("docs/RESULTS.md", "docs/API.md", "README.md")
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".js", ".css", ".html", ".svg", ".jpg", ".jpeg", ".png", ".md", ".txt", ".sh", ".service"}


def files() -> list[Path]:
    selected = []
    for directory in SOURCE_DIRS:
        for item in (ROOT / directory).rglob("*"):
            if not item.is_file() or item.is_symlink() or any(p.startswith(".") or p == "__pycache__" for p in item.relative_to(ROOT).parts):
                continue
            if item.suffix in SUFFIXES or item.name == "Caddyfile":
                selected.append(item)
    selected.extend(ROOT / name for name in SOURCE_FILES if (ROOT / name).is_file())
    return sorted(set(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected = files()
    with tarfile.open(args.out, "w:gz") as archive:
        for item in selected:
            archive.add(item, arcname=item.relative_to(ROOT), recursive=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(f"Packaged {len(selected)} source files ({args.out.stat().st_size:,} bytes); checkout {revision}")


if __name__ == "__main__":
    main()
