"""Shared command-line conventions for every pipeline and eval step (BUILD_PLAN §C.2).

Every step is a module run from the repo root, ``python -m pipeline.<step>`` or ``python -m eval.<script>``, and
accepts the same base flags:

    --config PATH          settings.yaml (default config/settings.yaml)
    --data-root PATH       pipeline files root (default: $PATHQUEUE_DATA_ROOT, else the site default; core.config)
    --artifacts-root PATH  app-facing files root (default: $PATHQUEUE_ARTIFACTS_ROOT, else the site default)
    --limit N              only the first N cases by sorted case_id (smoke runs: always try --limit 5 first)
    --force                recompute outputs that already exist (default: skip finished work; steps are resumable)

Usage:

    from core import cli
    parser = cli.step_parser("pipeline.04_concepts", "Concept scores per tile and case (plan §6.5).")
    parser.add_argument("--encoder", default="conch")          # step-specific flags go after the base ones
    ctx = cli.context(parser.parse_args())
    ids = cli.limit_case_ids(cases["case_id"], ctx.limit)

A data root holding the SYNTHETIC_MARKER file (written by pipeline.make_synthetic) is synthetic: steps pass
ctx.synthetic to core.io.write_meta and stamp `synthetic: true` on case JSONs built from it.

Portable: Python >= 3.10, stdlib + core.config (runs inside Delta jobs).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import config

SYNTHETIC_MARKER = "SYNTHETIC.json"


@dataclass(frozen=True)
class StepContext:
    step: str
    cfg: dict[str, Any]
    data_root: Path
    artifacts_root: Path
    limit: int | None
    force: bool
    synthetic: bool
    args: argparse.Namespace


def step_parser(step: str, description: str) -> argparse.ArgumentParser:
    """ArgumentParser with the base flags every step shares; add step-specific flags after calling this."""
    parser = argparse.ArgumentParser(prog=f"python -m {step}", description=description)
    parser.set_defaults(_step=step)
    parser.add_argument("--config", default=None, help="settings.yaml path (default: config/settings.yaml)")
    parser.add_argument("--data-root", default=None, help="pipeline files root (default: $PATHQUEUE_DATA_ROOT or site default)")
    parser.add_argument("--artifacts-root", default=None,
                        help="app-facing files root (default: $PATHQUEUE_ARTIFACTS_ROOT or site default)")
    parser.add_argument("--limit", type=_positive_int, default=None,
                        help="process only the first N cases by sorted case_id (smoke runs)")
    parser.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    return parser


def context(args: argparse.Namespace, env: dict[str, str] | None = None) -> StepContext:
    """Load settings and resolve both roots, honoring --data-root / --artifacts-root over the environment."""
    cfg = config.load_settings(args.config)
    merged = dict(os.environ if env is None else env)
    if args.data_root:
        merged[config.ENV_DATA_ROOT] = args.data_root
    if args.artifacts_root:
        merged[config.ENV_ARTIFACTS_ROOT] = args.artifacts_root
    data_root = config.data_root(cfg, merged)
    return StepContext(
        step=getattr(args, "_step", "step"),
        cfg=cfg,
        data_root=data_root,
        artifacts_root=config.artifacts_root(cfg, merged),
        limit=args.limit,
        force=bool(args.force),
        synthetic=is_synthetic_root(data_root),
        args=args,
    )


def is_synthetic_root(root: str | Path) -> bool:
    """True when the data root was written by pipeline.make_synthetic (holds SYNTHETIC_MARKER)."""
    return (Path(root) / SYNTHETIC_MARKER).is_file()


def limit_case_ids(case_ids: Iterable[str], limit: int | None) -> list[str]:
    """Unique case IDs in sorted order, truncated to the first `limit` (all when limit is None)."""
    ids = sorted({str(c) for c in case_ids})
    return ids if limit is None else ids[:limit]


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"--limit must be a positive integer, got {text}")
    return value
