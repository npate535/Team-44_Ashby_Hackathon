"""Spot-check counts per concept from <artifacts>/eval/spotcheck.json (summary.concept {n, acceptable}).

The reviews are AI or human (see that file's reviewer_kind); the reliability rule that uses these counts is
core.contract.spotcheck_passed (SPOTCHECK_RELIABLE_MIN). Read by step 07 (contrast detail) and step 08 (concept
findings), so both use the same counts.
"""

from __future__ import annotations

from pathlib import Path

from core import io, schemas


def counts(artifacts_root: str | Path) -> dict[str, dict[str, int]]:
    """Concept id -> {acceptable, reviewed}; concepts with nothing reviewed are left out. {} when the file is absent."""
    path = schemas.eval_json(artifacts_root, "spotcheck")
    if not path.is_file():
        return {}
    summary = (io.read_json(path).get("summary") or {}).get("concept") or {}
    out: dict[str, dict[str, int]] = {}
    for cid, entry in summary.items():
        n, ok = int(entry.get("n") or 0), int(entry.get("acceptable") or 0)
        if n > 0:
            if not 0 <= ok <= n:
                raise ValueError(f"eval/spotcheck.json: {cid} has {ok} acceptable of {n} reviewed")
            out[str(cid)] = {"acceptable": ok, "reviewed": n}
    return out
