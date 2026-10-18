"""Pre-generate "Evidence in plain language" for every case (and the results page) so the demo makes no model calls.

    uv run python scripts/interpret_all.py [--workers 16] [--limit N] [--no-results]
    uv run python scripts/interpret_all.py --repair [--workers 8]   # check + fix claims in already-cached texts

Uses the same evidence, prompt and cache key as the app (backend/interpret.py; cache under <artifacts>/interpretations/),
so cached cases are skipped and a rerun only fills gaps. The model comes from PATHQUEUE_INTERPRET_MODEL in .env.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.interpret import InterpretError, Interpreter  # noqa: E402
from core import config, contract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-results", action="store_true")
    ap.add_argument("--repair", action="store_true", help="check categorical claims of cached texts and fix failures")
    args = ap.parse_args()

    art = config.artifacts_root(config.load_settings())
    interp = Interpreter(art / "interpretations", dotenv=config.REPO_ROOT / ".env", eval_dir=art / "eval")
    files = sorted((art / "cases").glob("*.json"))[: args.limit or None]
    cases = [contract.CasePublic.model_validate(contract.to_public(json.loads(f.read_text()))).model_dump(mode="json")
             for f in files]
    if args.repair:
        return repair(interp, cases, args.workers)
    todo = [c for c in cases if interp.case(c).get("status") != "ok"]
    print(f"{len(cases)} cases, {len(cases) - len(todo)} cached, {len(todo)} to generate", flush=True)

    jobs: list[tuple[str, callable]] = [(c["case_id"], lambda c=c: interp.case(c, generate=True)) for c in todo]
    results_md = config.REPO_ROOT / "docs" / "RESULTS.md"
    if not args.no_results and results_md.is_file():
        jobs.append(("RESULTS", lambda: interp.results(results_md.read_text(), generate=True)))

    started, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for fut in as_completed(futures):
            name = futures[fut]
            done += 1
            try:
                r = fut.result()
                v = r.get("verification") or {}
                note = f"{v.get('checked', 0) - len(v.get('unsupported', []))}/{v.get('checked', 0)} traced"
                print(f"[{done}/{len(jobs)}] {name} ok {r.get('elapsed_s')}s {note} key={r.get('key', 'cached')}",
                      flush=True)
            except InterpretError as exc:
                failed.append(name)
                print(f"[{done}/{len(jobs)}] {name} FAILED: {exc}", flush=True)
    print(f"finished in {(time.time() - started) / 60:.1f} min; {len(failed)} failed: {' '.join(failed)}", flush=True)
    return 1 if failed else 0


def repair(interp: Interpreter, cases: list[dict], workers: int) -> int:
    cached = [c for c in cases if interp.case(c).get("status") == "ok"]
    print(f"{len(cached)} cached cases to check", flush=True)
    fixed = clean = left = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(interp.repair, c): c["case_id"] for c in cached}
        for i, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                claims = fut.result().get("claims") or {}
            except InterpretError as exc:
                failed.append(name)
                print(f"[{i}/{len(cached)}] {name} REPAIR FAILED: {exc}", flush=True)
                continue
            found, remaining = claims.get("found") or [], claims.get("remaining") or []
            clean += not found
            fixed += bool(found) and not remaining
            left += bool(remaining)
            if found:
                print(f"[{i}/{len(cached)}] {name} {len(found)} found, {len(remaining)} remaining", flush=True)
    print(f"repair finished: {clean} clean, {fixed} fixed, {left} still with issues, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
