"""PathQueue demo backend: FastAPI over the SQLite store and the arrival simulator (plan §10; BUILD_PLAN Task 6).

    uvicorn backend.app:app                     (make demo; settings backend.host / backend.port)
    PATHQUEUE_CASES_DIR=fixtures  make demo     cases from the committed fixtures (flat media)
    PATHQUEUE_ARTIFACTS_ROOT=<dir> make demo    cases from <dir>/cases, media from <dir>/images, <dir>/cells ...

Endpoints are documented with example JSON in docs/API.md. Every case payload is built from the stored CasePublic
form (core.contract.to_public at load time), so `eval_only` never leaves the process; eval files served by
/api/metrics are scrubbed of ground-truth keys as a second guard. Tiers come only from core.tiers (aging guard on
the simulator clock); surgical timers run on the wall clock (amber/red from settings margin.timer).

Media: when the cases folder is the flat fixtures folder, images/<file> and cells/<file> resolve to fixtures/<file>
(core.schemas.fixture_media); otherwise they resolve under the artifacts root. The teammate frontend
(team44-triage-frontend/app) is served at /. Research prototype — not for clinical use.
"""

from __future__ import annotations

import datetime
import json
import logging
import mimetypes
import os
import re
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Body, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import yaml
from pydantic import BaseModel, ConfigDict, Field

from backend import db
from backend.simulator import VARIANTS, Clock, Simulator
from core import config, contract, io, schemas, tiers

log = logging.getLogger("pathqueue.backend")

ENV_CASES_DIR = "PATHQUEUE_CASES_DIR"
ENV_DEV_CORS = "PATHQUEUE_DEV_CORS"
FRONTEND_DIR = config.REPO_ROOT / "team44-triage-frontend" / "app"
GROUND_TRUTH_KEYS = frozenset({"eval_only", "true_label", "true_labels", "y_true", "label_true"})
_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
FIGURE_SUFFIXES = (".png", ".svg", ".jpg", ".jpeg")
# Static routes serve media only: case JSONs and eval JSONs go through the API (which strips ground truth).
MEDIA_SUFFIXES = {"images": (".jpg", ".jpeg", ".png"), "cells": (".png", ".jpg", ".jpeg"), "eval": FIGURE_SUFFIXES,
                  "spotcheck": (".png", ".jpg", ".jpeg"), "fixtures": (".jpg", ".jpeg", ".png", ".json")}


def cell_feature_labels(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Display names / units / directions of measured cell features and cell types (settings cells.features_file)."""
    data = yaml.safe_load(config.resolve_repo_path(cfg["cells"]["features_file"]).read_text(encoding="utf-8"))
    features = {f["id"]: {"label": f["display_name"], "unit": f["unit"], "direction": f["direction"]}
                for f in data["features"]}
    types = {t["key"]: t["display_name"] for t in data["types"].values()}
    return {"features": features, "types": types}


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    stamp = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def scrub_ground_truth(obj: Any) -> tuple[Any, int]:
    """Drop ground-truth keys anywhere in a JSON value; returns (clean value, number of keys dropped)."""
    if isinstance(obj, Mapping):
        out, dropped = {}, 0
        for key, value in obj.items():
            if key in GROUND_TRUTH_KEYS:
                dropped += 1
                continue
            out[key], n = scrub_ground_truth(value)
            dropped += n
        return out, dropped
    if isinstance(obj, list):
        items = [scrub_ground_truth(v) for v in obj]
        return [v for v, _ in items], sum(n for _, n in items)
    return obj, 0


# ---- request bodies -----------------------------------------------------------------------------------------------------
class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["approve_orders", "edit_orders", "closer_look", "dismiss_flag", "sign_out", "margin_call"]
    payload: dict[str, Any] = {}


class InjectIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: Literal["normal", "degraded", "composite"] = "normal"


class SpotReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1, max_length=256)
    verdict: Literal["acceptable", "not_acceptable", "unsure"]
    note: str = Field(default="", max_length=500)


class SpotcheckIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=1, max_length=80)
    reviewer_kind: Literal["human", "ai"]
    reviews: list[SpotReview] = Field(min_length=1)


# ---- the backend ------------------------------------------------------------------------------------------------------------
class Backend:
    """Resolved paths + store + simulator + response builders (one per app)."""

    def __init__(self, cfg: Mapping[str, Any], *, cases_dir: Path, artifacts_root: Path, db_path: Path,
                 fixtures_dir: Path, clock: Clock | None = None):
        self.cfg = cfg
        self.cases_dir = cases_dir
        self.artifacts_root = artifacts_root
        self.fixtures_dir = fixtures_dir
        self.flat_media = cases_dir.resolve() == fixtures_dir.resolve()
        self.store = db.Store(db_path)
        try:
            self.sim = Simulator(self.store, cfg, lambda: db.load_case_dir(cases_dir), clock)
        except Exception:
            self.store.close()
            raise
        timer = cfg["margin"]["timer"]
        self.amber_s = float(timer["amber_min"]) * 60.0
        self.red_s = float(timer["red_min"]) * 60.0
        self.or_rooms = int(cfg["sim"]["or_rooms"])
        self.targets = cfg["aging"]["turnaround_target_min"]
        self.cell_labels = cell_feature_labels(cfg)         # read once; a bad file fails at startup

    # ---- paths ----------------------------------------------------------------------------------------------------
    def media_path(self, kind: str, name: str) -> Path:
        if not _FILE_RE.fullmatch(name) or ".." in name or kind not in MEDIA_SUFFIXES or \
                not name.lower().endswith(MEDIA_SUFFIXES[kind]):
            raise db.NotFoundError(f"{kind}/{name} not found")
        if kind in ("images", "cells"):
            if self.flat_media:
                return schemas.fixture_media(self.fixtures_dir, f"{kind}/{name}")
            return self.artifacts_root / kind / name
        if kind == "eval":
            return schemas.eval_asset(self.artifacts_root, name)
        if kind == "spotcheck":
            return schemas.spotcheck_items_json(self.artifacts_root).parent / name
        if kind == "fixtures":
            return self.fixtures_dir / name
        raise db.NotFoundError(f"unknown media kind {kind!r}")

    # ---- runtime ----------------------------------------------------------------------------------------------------
    @property
    def synthetic(self) -> bool:
        return any(c.get("synthetic", False) for c in self.sim.loaded.cases)

    def _effective(self, row: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
        """The case as core.tiers sees it: arrival = when it actually arrived on the simulator clock."""
        return {**case, "arrived_offset_s": float(row["arrived_sim_s"])}

    def _timer(self, row: Mapping[str, Any], tier: int, wall: float, sim: float) -> dict[str, Any]:
        if row["lane"] == "surgical":
            started, stopped = row["timer_started_wall"], row["timer_stopped_wall"]
            elapsed = max(0.0, (stopped if stopped is not None else wall) - started)
            if stopped is not None:
                state = "stopped"
            elif elapsed >= self.red_s:
                state = "red"
            elif elapsed >= self.amber_s:
                state = "amber"
            else:
                state = "ok"
            return {"kind": "surgical", "clock": "wall", "started_at": iso(started), "stopped_at": iso(stopped),
                    "elapsed_s": elapsed, "state": state, "amber_s": self.amber_s, "red_s": self.red_s}
        end = row["done_sim_s"] if row["done_sim_s"] is not None else sim     # frozen once signed out
        elapsed = max(0.0, end - float(row["arrived_sim_s"]))
        target = float(self.targets[tier]) * 60.0   # aging.turnaround_target_min of the tier before any promotion
        return {"kind": "biopsy", "clock": "sim", "elapsed_s": elapsed, "target_s": target,
                "state": "stopped" if row["done_sim_s"] is not None else "none"}

    def runtime(self, row: Mapping[str, Any], case: Mapping[str, Any], wall: float, sim: float) -> dict[str, Any]:
        tier, reasons = tiers.assign_tier(self._effective(row, case), self.cfg, now=sim)
        call = json.loads(row["margin_call_json"]) if row["margin_call_json"] else None
        return {
            "tier": tier,
            "reasons": reasons,
            "aging_promoted": tiers.AGING_PROMOTED in reasons,
            "arrived_sim_s": float(row["arrived_sim_s"]),
            "arrived_at": iso(row["arrived_wall"]),
            "injected": bool(row["injected"]),
            "timer": self._timer(row, tier + int(tiers.AGING_PROMOTED in reasons), wall, sim),
            "decision": {
                "signed_out_at": iso(row["signed_out_wall"]),
                "margin_call": None if call is None else {**call, "at": iso(row["margin_called_wall"])},
            },
            "done": db.is_done(row),
            "allowed_actions": db.allowed_actions(row),
            "or_room": self.or_room(row["case_id"]) if row["lane"] == "surgical" else None,
        }

    def or_room(self, case_id: str) -> str:
        return f"OR-{zlib.crc32(case_id.encode('utf-8')) % self.or_rooms + 1}"

    def sort_rows(self, rows: list[Any], order: str, sim: float) -> list[Any]:
        def key(row: Any) -> tuple:
            effective = self._effective(row, json.loads(row["public_json"]))
            done = db.is_done(row)
            if order == "arrival":
                return done, float(row["arrived_sim_s"]), row["case_id"]
            return (done, *tiers.sort_key(effective, self.cfg, now=sim))
        return sorted(rows, key=key)

    # ---- payloads -------------------------------------------------------------------------------------------------
    def worklist(self, lane: str, order: str) -> dict[str, Any]:
        with self.store.lock:
            self.sim.tick()
            sim, wall = self.sim.sim_now(), self.sim.wall_now()
            rows = self.store.rows(lane=None if lane == "all" else lane)
            items = []
            for row in self.sort_rows(rows, order, sim):
                case = json.loads(row["public_json"])
                orders = [o for o in self.store.orders(row["case_id"]) if o["status"] != "removed"]
                margin = case.get("margin")
                items.append({
                    "case_id": case["case_id"],
                    "lane": case["lane"],
                    "simulated": case["simulated"],
                    "simulated_variant": case.get("simulated_variant"),
                    "synthetic": case.get("synthetic", False),
                    "arrived_offset_s": case["arrived_offset_s"],
                    "prediction": case["prediction"],
                    "quality": case["quality"],
                    "margin": None if margin is None else {k: margin[k] for k in
                                                           ("suggested_call", "involved", "per_margin")},
                    "flag": case["flag"],
                    "agent": {"status": row["agent_status"], "n_orders": len(orders),
                              "orders_drafted": row["agent_status"] == "drafted",
                              "blocked": len(case["agent"].get("blocked") or [])},
                    "runtime": self.runtime(row, case, wall, sim),
                })
            return {"server_time": wall, "server_time_iso": iso(wall), "sim": self.sim.status(), "order": order,
                    "lane": lane, "synthetic": self.synthetic, "items": items}

    def case(self, case_id: str) -> dict[str, Any]:
        with self.store.lock:
            self.sim.tick()
            sim, wall = self.sim.sim_now(), self.sim.wall_now()
            row = self.store.arrived_row(case_id)
            case = json.loads(row["public_json"])
            order_rows = self.store.orders(case_id)
            case["agent"] = {
                **case["agent"],
                "status": row["agent_status"],
                "orders": [{"type": o["type"], "items": o["items"], "rule_id": o["rule_id"],
                            "rationale": o["rationale"]} for o in order_rows if o["status"] != "removed"],
            }
            public = contract.CasePublic.model_validate(case).model_dump(mode="json")
            runtime = self.runtime(row, case, wall, sim)
            runtime["orders"] = [{"seq": o["seq"], "rule_id": o["rule_id"], "type": o["type"], "status": o["status"],
                                  "drafted_items": o["drafted_items"], "items": o["items"]} for o in order_rows]
            return {**public, "runtime": runtime, "server_time": wall, "server_time_iso": iso(wall)}

    def events(self, case_id: str) -> list[dict[str, Any]]:
        with self.store.lock:
            self.store.arrived_row(case_id)
            return [{**{k: v for k, v in e.items() if k != "at_wall"}, "at": iso(e["at_wall"])}
                    for e in self.store.events(case_id)]

    def routed(self, to_role: str) -> list[dict[str, Any]]:
        with self.store.lock:
            self.sim.tick()
            out = []
            for item in self.store.routed(to_role):
                at = item.pop("at_wall")
                item["at"] = iso(at)
                if to_role == "surgeon":
                    item["or_room"] = self.or_room(item["case_id"])
                out.append(item)
            return out

    def decide(self, case_id: str, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.store.lock:
            self.sim.tick()
            result = self.store.decide(case_id, action, payload, wall=self.sim.wall_now(), sim=self.sim.sim_now())
            return {"result": result, "case": self.case(case_id)}

    def config_payload(self) -> dict[str, Any]:
        cfg = self.cfg
        ui = cfg["ui"]
        return {
            "organ": cfg["organ"],
            "classes": list(cfg["classes"]),
            "aggressive_class": cfg["aggressive_class"],
            "alpha": cfg["alpha"],
            "banner": ui["banner"],
            "palette": dict(ui["palette"]),
            "poll_ms": int(ui["poll_ms"]),
            "labels": {
                "tiers": {str(k): v for k, v in ui["tier_labels"].items()},
                "reasons": dict(ui["reason_labels"]),
                "classes": dict(ui["class_labels"]),
                "concept_groups": {k: {"label": g["label"], "ids": list(g["ids"])}
                                   for k, g in cfg["concepts"]["groups"].items()},
                "cell_features": {k: v["label"] for k, v in self.cell_labels["features"].items()},
                "cell_types": dict(self.cell_labels["types"]),
            },
            "cell_features": self.cell_labels["features"],
            "margin": {"names": dict(cfg["margin"]["names"]), "note": cfg["margin"]["note"],
                       "timer": dict(cfg["margin"]["timer"]), "order": list(contract.MARGIN_NAMES)},
            "timer": dict(cfg["margin"]["timer"]),
            "aging": {"enabled": bool(cfg["aging"]["enabled"]),
                      "promote_at_fraction": cfg["aging"]["promote_at_fraction"],
                      "turnaround_target_min": {str(k): v for k, v in cfg["aging"]["turnaround_target_min"].items()}},
            "cells": {"enabled": bool(cfg["cells"]["enabled"]), "types": list(cfg["cells"]["types"]),
                      "overlay_colors": dict(cfg["cells"]["overlay_colors"])},
            "sim": {"speed": cfg["sim"]["speed"], "hold_surgical_for_injection": self.sim.hold_surgical,
                    "variants": list(VARIANTS)},
            "flag_reasons": list(contract.FLAG_REASONS),
            "synthetic": self.synthetic,
        }

    def metrics(self) -> dict[str, Any]:
        eval_dir = self.artifacts_root / "eval"
        files: dict[str, Any] = {}
        figures: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        if eval_dir.is_dir():
            for path in sorted(eval_dir.iterdir()):
                if path.name.startswith(".") or not _FILE_RE.fullmatch(path.name):
                    continue
                if path.suffix == ".json":
                    try:
                        data = io.read_json(path)
                    except (OSError, ValueError) as exc:
                        log.warning("unreadable eval file %s: %s", path.name, exc)
                        errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    clean, dropped = scrub_ground_truth(data)
                    if dropped:
                        log.warning("dropped %d ground-truth key(s) from eval/%s before serving", dropped, path.name)
                    files[path.stem] = clean
                elif path.suffix.lower() in FIGURE_SUFFIXES:
                    figures.append({"file": path.name, "url": f"/eval/{path.name}"})
        synthetic = any(isinstance(v, Mapping) and v.get("synthetic") for v in files.values())
        return {"eval": files, "figures": figures, "errors": errors, "expected": dict(schemas.EVAL_FILES),
                "synthetic": synthetic or self.synthetic}

    def spotcheck_items(self) -> dict[str, Any]:
        path = schemas.spotcheck_items_json(self.artifacts_root)
        if not path.is_file():
            return {"available": False, "items": [], "synthetic": self.synthetic}
        data, _ = scrub_ground_truth(io.read_json(path))
        missing = [k for k in schemas.SPOTCHECK_ITEMS.required_keys if k not in data]
        if missing:
            raise db.PayloadError(f"spotcheck/items.json is missing {missing}")
        return {"available": True, **data}

    def record_spotcheck(self, body: SpotcheckIn) -> dict[str, Any]:
        with self.store.lock:                    # read-merge-write of eval/spotcheck.json: one writer at a time
            return self._record_spotcheck(body)

    def _record_spotcheck(self, body: SpotcheckIn) -> dict[str, Any]:
        items = self.spotcheck_items()
        if not items["available"]:
            raise db.TransitionError("no spot-check items are available (spotcheck/items.json is missing)")
        by_id = {item["item_id"]: item for item in items["items"]}
        unknown = [r.item_id for r in body.reviews if r.item_id not in by_id]
        if unknown:
            raise db.PayloadError(f"unknown spot-check item ids {unknown[:5]}")
        out_path = schemas.eval_json(self.artifacts_root, "spotcheck")
        previous: list[dict[str, Any]] = []
        if out_path.is_file():
            existing = io.read_json(out_path)
            if "reviews" not in existing:
                raise db.ServerDataError(f"{out_path} exists but has no 'reviews'; refusing to overwrite it")
            previous = existing["reviews"]
        now = iso(self.sim.wall_now())
        new = {(body.reviewer, r.item_id): {"item_id": r.item_id, "reviewer": body.reviewer,
                                           "reviewer_kind": body.reviewer_kind, "verdict": r.verdict,
                                           "note": r.note, "at": now} for r in body.reviews}
        kept = [r for r in previous if (r.get("reviewer"), r.get("item_id")) not in new]
        reviews = kept + list(new.values())
        kinds = sorted({r["reviewer_kind"] for r in reviews})
        summary: dict[str, Any] = {"concept": {}, "cells": {"n": 0, "acceptable": 0}, "orphaned": 0}
        for review in reviews:
            item = by_id.get(review["item_id"])
            if item is None:                     # item removed from a regenerated items.json: kept, not counted
                summary["orphaned"] += 1
                continue
            bucket = summary["cells"] if item["kind"] == "cells" else \
                summary["concept"].setdefault(item["target"], {"n": 0, "acceptable": 0})
            bucket["n"] += 1
            bucket["acceptable"] += int(review["verdict"] == "acceptable")
        if summary["orphaned"]:
            log.warning("spotcheck: %d earlier review(s) refer to items no longer in items.json",
                        summary["orphaned"])
        payload = {
            "name": "spotcheck",
            **io.provenance(self.cfg, synthetic=bool(items.get("synthetic")) or self.synthetic),
            "reviewer_kind": kinds[0] if len(kinds) == 1 else "mixed",
            "reviews": reviews,
            "summary": summary,
        }
        io.write_json(out_path, payload)
        self.store.log_event(None, "spotcheck", wall=self.sim.wall_now(), sim=self.sim.sim_now(),
                             payload={"reviewer_kind": body.reviewer_kind, "n": len(body.reviews)})
        return {"recorded": len(body.reviews), "total": len(reviews), "reviewer_kind": payload["reviewer_kind"],
                "summary": summary}

    def artifact_status(self) -> dict[str, Any]:
        """Check the media actually referenced by loaded cases; never substitute fixture images."""
        assets: set[str] = set()
        models: set[str] = set()
        for case in self.sim.loaded.cases:
            assets.add(case["image"]["display"])
            models.add(case["prediction"]["model"])
            for ref in case.get("similar_cases", []):
                assets.add(ref["image"])
            contrast = case.get("contrast_case")
            if contrast:
                assets.add(contrast.get("image") or f"images/{contrast['case_id']}.jpg")
            cells = case.get("cells") or {}
            for tile in cells.get("tiles", []):
                if tile.get("overlay"):
                    assets.add(tile["overlay"])
        missing = []
        for asset in sorted(assets):
            kind, _, name = asset.partition("/")
            try:
                exists = self.media_path(kind, name).is_file()
            except db.NotFoundError:
                exists = False
            if not exists:
                missing.append(asset)
        return {"referenced": len(assets), "available": len(assets) - len(missing),
                "missing": missing, "models": sorted(models)}

    def health(self) -> dict[str, Any]:
        wall = self.sim.wall_now()
        loaded = len(self.sim.loaded.cases)
        assets = self.artifact_status()
        return {
            "status": "ok" if loaded and not self.sim.loaded.skipped and not assets["missing"] else "degraded",
            "assets": assets,
            "cases_dir": str(self.cases_dir),
            "artifacts_root": str(self.artifacts_root),
            "media": "fixtures" if self.flat_media else "artifacts",
            "loaded": loaded,
            "open_cases": sum(not db.is_done(row) for row in self.store.rows()),
            "skipped": self.sim.loaded.skipped,
            "synthetic": self.synthetic,
            "sim": self.sim.status(),
            "server_time": wall,
            "server_time_iso": iso(wall),
        }


# ---- app factory --------------------------------------------------------------------------------------------------------------
def resolve_paths(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> dict[str, Path]:
    """cases_dir: $PATHQUEUE_CASES_DIR (repo-relative allowed) else <artifacts>/cases; db: <artifacts>/backend.db_file."""
    source = os.environ if env is None else env
    art = config.artifacts_root(cfg, env)
    override = (source.get(ENV_CASES_DIR) or "").strip()
    cases_dir = config.resolve_repo_path(override) if override else art / "cases"
    return {"artifacts_root": art, "cases_dir": cases_dir, "db_path": art / cfg["backend"]["db_file"],
            "fixtures_dir": config.fixtures_root(cfg)}


def create_app(*, cfg: Mapping[str, Any] | None = None, cases_dir: str | Path | None = None,
               artifacts_root: str | Path | None = None, db_path: str | Path | None = None,
               fixtures_dir: str | Path | None = None, frontend_dir: str | Path | None = FRONTEND_DIR,
               clock: Clock | None = None, env: Mapping[str, str] | None = None) -> FastAPI:
    cfg = cfg if cfg is not None else config.load_settings()
    source = os.environ if env is None else env
    paths = resolve_paths(cfg, source)
    art = Path(artifacts_root) if artifacts_root is not None else paths["artifacts_root"]
    if cases_dir is not None:
        cases = Path(cases_dir)
    elif (source.get(ENV_CASES_DIR) or "").strip():
        cases = paths["cases_dir"]
    else:
        cases = art / "cases"
    if not cases.is_dir():
        raise FileNotFoundError(
            f"cases folder {cases} does not exist: run `make analyze` to build artifacts, or start the fixture demo "
            f"with `make demo CASES=fixtures` (or set {ENV_CASES_DIR})")
    backend = Backend(
        cfg, cases_dir=cases, artifacts_root=art,
        db_path=Path(db_path) if db_path is not None else art / cfg["backend"]["db_file"],
        fixtures_dir=Path(fixtures_dir) if fixtures_dir is not None else paths["fixtures_dir"], clock=clock)
    log.info("PathQueue backend: %d cases from %s (%d skipped)", len(backend.sim.loaded.cases), cases,
             len(backend.sim.loaded.skipped))
    if not backend.sim.loaded.cases:
        log.warning("no valid case JSONs in %s: the worklist will stay empty", cases)

    app = FastAPI(title="PathQueue demo API", version=contract.SCHEMA_VERSION,
                  description="Research prototype — not for clinical use. See docs/API.md.")
    app.state.backend = backend

    if (source.get(ENV_DEV_CORS) or "").strip() == "1":
        app.add_middleware(CORSMiddleware, allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def no_store(request: Request, call_next):   # stale ES modules break the demo; everything is local
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(db.StoreError)
    async def store_error(_request: Request, exc: db.StoreError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return backend.health()

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return backend.config_payload()

    @app.get("/api/worklist")
    def worklist(lane: Literal["all", "surgical", "biopsy"] = "all",
                 order: Literal["ai", "arrival"] = "ai") -> dict[str, Any]:
        return backend.worklist(lane, order)

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        return backend.case(case_id)

    @app.get("/api/cases/{case_id}/events")
    def get_events(case_id: str) -> list[dict[str, Any]]:
        return backend.events(case_id)

    @app.post("/api/cases/{case_id}/decision")
    def decide(case_id: str, body: DecisionIn) -> dict[str, Any]:
        return backend.decide(case_id, body.action, body.payload)

    @app.get("/api/routed")
    def routed(to: Literal["surgeon", "oncologist"]) -> list[dict[str, Any]]:
        return backend.routed(to)

    @app.post("/api/sim/{command}")
    def sim_control(command: Literal["start", "pause", "reset", "inject_surgical"],
                    body: Optional[InjectIn] = Body(default=None)) -> dict[str, Any]:
        injected = None
        if command == "start":
            backend.sim.start()
        elif command == "pause":
            backend.sim.pause()
        elif command == "reset":
            backend.sim.reset()
        else:
            injected = backend.sim.inject_surgical((body or InjectIn()).variant)
        return {"command": command, "injected": injected, "sim": backend.sim.status()}

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        return backend.metrics()

    @app.get("/api/spotcheck/items")
    def spotcheck_items() -> dict[str, Any]:
        return backend.spotcheck_items()

    @app.post("/api/spotcheck")
    def spotcheck(body: SpotcheckIn) -> dict[str, Any]:
        return backend.record_spotcheck(body)

    def media(kind: str):
        def serve(name: str) -> Any:
            path = backend.media_path(kind, name)
            if not path.is_file():
                raise db.NotFoundError(f"{kind}/{name} not found")
            if path.suffix == ".json":          # fixture cases for ?mock=1: public form only (no eval_only)
                try:
                    return contract.to_public(io.read_json(path))
                except (ValueError, OSError) as exc:
                    raise db.ServerDataError(f"fixtures/{name} is not a valid case: {exc}") from exc
            return FileResponse(path)
        serve.__name__ = f"media_{kind}"
        return serve

    for kind in ("images", "cells", "eval", "spotcheck", "fixtures"):
        app.get(f"/{kind}/{{name}}", include_in_schema=False)(media(kind))

    if frontend_dir is not None:
        if Path(frontend_dir).is_dir():
            mimetypes.add_type("text/javascript", ".js")
            app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        else:
            log.warning("frontend folder %s is missing: only the API is served", frontend_dir)
    return app


class _LazyApp:
    """`uvicorn backend.app:app` target: builds the real app on first use, so importing this module has no side effects."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    async def __call__(self, scope, receive, send) -> None:
        if self._app is None:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
            try:
                self._app = create_app()
            except Exception as exc:
                if scope["type"] != "lifespan":
                    raise
                await receive()                  # lifespan.startup: report the failure so uvicorn exits at once
                await send({"type": "lifespan.startup.failed", "message": f"{type(exc).__name__}: {exc}"})
                return
        await self._app(scope, receive, send)


app = _LazyApp()
