"""Build a private Vercel upload, with immutable media served from the CDN.

Usage: uv run python scripts/package_vercel.py --out /tmp/sepia-vercel
The upload directory contains case data and must never be committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit('Use a private staging directory outside the checkout')
    out.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store', '.env*', '*.sqlite*')
    for name in ('backend', 'core', 'agent', 'config', 'fixtures'):
        shutil.copytree(ROOT / name, out / name, dirs_exist_ok=True, ignore=ignore)
    (out / 'docs').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'docs/RESULTS.md', out / 'docs/RESULTS.md')
    shutil.copytree(ROOT / 'team44-triage-frontend/app', out / 'public', dirs_exist_ok=True, ignore=ignore)
    manifest = {}
    for section in ('cases', 'eval', 'spotcheck', 'interpretations'):
        source = ROOT / 'artifacts' / section
        for path in source.rglob('*'):
            if not path.is_file() or path.name.startswith('.'):
                continue
            relative = path.relative_to(ROOT / 'artifacts')
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.svg'):
                dest = out / 'public' / relative
                manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif path.suffix == '.json':
                dest = out / 'artifacts' / relative
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    # TCGA external specimens: case JSON stays private under artifacts/ (served by /api/tcga), display JPEGs go to the
    # CDN at /tcga/<name>, the URL the case JSON's image.display ("tcga/<name>") resolves to (backend media kind 'tcga').
    tcga = ROOT / 'artifacts' / 'tcga'
    for path in [tcga / 'index.json', *sorted((tcga / 'cases').glob('*.json'))]:
        if path.is_file():
            dest = out / 'artifacts' / path.relative_to(ROOT / 'artifacts')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    for path in sorted((tcga / 'images').glob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        relative = Path('tcga') / path.name
        dest = out / 'public' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for section in ('images', 'cells'):
        for path in (ROOT / 'artifacts' / section).rglob('*'):
            if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            relative = path.relative_to(ROOT / 'artifacts')
            dest = out / 'public' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'media-manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    shutil.copy2(ROOT / 'deploy/requirements.txt', out / 'requirements.txt')
    shutil.copy2(ROOT / 'deploy/build_vercel.sh', out / 'build.sh')
    (out / '.python-version').write_text('3.13\n')
    dependencies = [line.strip() for line in (out / 'requirements.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
    (out / 'pyproject.toml').write_text('[project]\nname = "sepia-web"\nversion = "0.1.0"\nrequires-python = ">=3.13,<3.14"\ndependencies = ' + json.dumps(dependencies) + '\n\n[tool.vercel]\nentrypoint = "backend.vercel:app"\n')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    settings = {'framework': 'fastapi', 'regions': ['iad1'], 'buildCommand': 'sh build.sh',
                'functions': {'backend/vercel.py': {'maxDuration': 400, 'excludeFiles': 'public/**'}},
                'env': {'PATHQUEUE_RELEASE_SHA': revision, 'PATHQUEUE_SITE': 'local'},
                'headers': [{'source': '/(.*)', 'headers': [{'key': 'Cache-Control', 'value': 'no-store'}]}]}
    (out / 'vercel.json').write_text(json.dumps(settings, indent=2) + '\n')
    # Review state belongs to the shared Redis workspace. Never upload a running
    # laptop demo database, which may describe an older set of case evidence.
    for database in (out / 'artifacts').glob('*.sqlite*'):
        database.unlink()
    print(f'Packaged private Vercel upload: {out}; {len(manifest)} verified CDN assets; source {revision}')


if __name__ == '__main__':
    main()
