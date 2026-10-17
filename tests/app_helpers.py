"""Shared helpers for the backend tests (Agent 5): a fake clock, case-folder builders and a TestClient factory."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend import dev_cases
from backend.app import create_app
from backend.simulator import Clock
from core import config, io, schemas, tiers

CFG = config.load_settings()
FIXTURES = config.fixtures_root(CFG)
LEAK_KEYS = ("eval_only", "true_label")


@dataclass
class FakeClock:
    wall_s: float = 1_800_000_000.0
    mono_s: float = 0.0

    def advance(self, seconds: float) -> None:
        self.wall_s += seconds
        self.mono_s += seconds

    def clock(self) -> Clock:
        return Clock(wall=lambda: self.wall_s, mono=lambda: self.mono_s)


def fixture_case(name: str) -> dict[str, Any]:
    return io.read_json(schemas.fixture_case_json(FIXTURES, name))


def variant(base: Mapping[str, Any], case_id: str, *, probs: list[float] | None = None,
            conformal_set: list[str] | None = None, offset: float | None = None,
            reliability_flag: bool | None = None, agent: Mapping[str, Any] | None = None,
            margin: Mapping[str, Any] | None | str = "keep", **extra: Any) -> dict[str, Any]:
    """A copy of a fixture case with a new id and edited prediction/quality, flags recomputed by core.tiers."""
    case = copy.deepcopy(dict(base))
    case["case_id"] = case_id
    if probs is not None:
        classes = case["prediction"]["classes"]
        case["prediction"]["probs"] = probs
        case["prediction"]["risk_score"] = probs[classes.index(CFG["aggressive_class"])]
    if conformal_set is not None:
        case["prediction"]["conformal_set"] = conformal_set
    if offset is not None:
        case["arrived_offset_s"] = offset
    if reliability_flag is not None:
        case["quality"]["reliability_flag"] = reliability_flag
    if agent is not None:
        case["agent"] = dict(agent)
    if margin != "keep":
        case["margin"] = margin
    case.update(extra)
    case["flag"] = tiers.flag_for(case, CFG)
    return case


def write_cases(folder: Path, cases: Iterable[Mapping[str, Any]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for case in cases:
        io.write_json(folder / f"{case['case_id']}.json", case)
    return folder


def dev_root(tmp_path: Path, *, fill: int = 2, mean_interarrival_s: float = 60.0) -> Path:
    out = tmp_path / "dev"
    dev_cases.build_dev_cases(CFG, out, fill=fill, mean_interarrival_s=mean_interarrival_s, seed=44,
                              fixtures=FIXTURES)
    return out


def dev_roles(root: Path) -> dict[str, str]:
    """role -> case_id, read from the (test-only) eval_only.role of the dev case files."""
    roles = {}
    for path in sorted((root / "cases").glob("*.json")):
        case = json.loads(path.read_text())
        roles[case["eval_only"]["role"]] = case["case_id"]
    return roles


def make_client(tmp_path: Path, *, cases_dir: Path | None = None, artifacts_root: Path | None = None,
                clock: FakeClock | None = None, cfg: Mapping[str, Any] | None = None,
                frontend: bool = False) -> tuple[TestClient, FakeClock]:
    clock = clock or FakeClock()
    art = artifacts_root or tmp_path / "art"
    app = create_app(cfg=cfg or CFG, cases_dir=cases_dir if cases_dir is not None else art / "cases",
                     artifacts_root=art, db_path=tmp_path / "test.sqlite", fixtures_dir=FIXTURES,
                     clock=clock.clock(), frontend_dir=None if not frontend else
                     config.REPO_ROOT / "team44-triage-frontend" / "app", env={})
    return TestClient(app), clock


def arrive_all_biopsies(client: TestClient, clock: FakeClock) -> dict[str, Any]:
    """Start the simulator and advance until every biopsy has arrived; returns the worklist."""
    client.post("/api/sim/start")
    for _ in range(200):
        status = client.get("/api/health").json()["sim"]
        if status["pending"]["biopsy"] == 0:
            break
        clock.advance(10)
    client.post("/api/sim/pause")
    return client.get("/api/worklist").json()


def find_keys(obj: Any, keys: Iterable[str] = LEAK_KEYS, path: str = "$") -> list[str]:
    """Paths of any forbidden key anywhere in a JSON value."""
    keys = tuple(keys)
    hits: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k in keys:
                hits.append(f"{path}.{k}")
            hits += find_keys(v, keys, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_keys(v, keys, f"{path}[{i}]")
    return hits


def row_for(worklist: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    return next(item for item in worklist["items"] if item["case_id"] == case_id)


def ids(worklist: Mapping[str, Any], pred: Callable[[Mapping[str, Any]], bool] = lambda _: True) -> list[str]:
    return [item["case_id"] for item in worklist["items"] if pred(item)]
