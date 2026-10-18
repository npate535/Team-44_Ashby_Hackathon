"""Arrival simulator: replays cases on a simulator clock (plan §10; BUILD_PLAN Task 6).

    SimClock   seconds on the simulator clock: start / pause / reset; runs at settings `sim.speed` x wall time
    Simulator  arrivals and controls over a backend.db.Store

Arrivals: a case arrives when the simulator clock passes its `arrived_offset_s` (checked lazily by tick(), which the
API calls before every read). Only arrived cases appear anywhere in the API. When `sim.hold_surgical_for_injection`
is true, surgical cases never arrive on their own: `inject_surgical {variant}` brings them in one at a time, which is
how the demo script (plan §14 steps 3-4) introduces simulated frozen sections.

inject_surgical(variant) picks the next pending surgical case of that variant (normal = no simulated_variant,
degraded, composite): first the best match of the settings `demo.heroes` entry for that variant (surgical,
surgical_degraded, surgical_composite), then the earliest `arrived_offset_s`, then case_id. The case arrives now on
the simulator clock and its surgical timer starts on the wall clock.

Clocks: surgical timers and event stamps use the wall clock (Clock.wall, epoch seconds); arrivals and aging use the
simulator clock driven by Clock.mono. Tests pass a fake Clock.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from backend import db

log = logging.getLogger("pathqueue.backend")

VARIANTS = ("normal", "degraded", "composite")
_HERO_KEY = {"normal": "surgical", "degraded": "surgical_degraded", "composite": "surgical_composite"}


@dataclass
class Clock:
    """Wall clock (epoch seconds) and a monotonic clock; replaced by a fake in tests."""

    wall: Callable[[], float] = time.time
    mono: Callable[[], float] = time.monotonic


@dataclass
class SimClock:
    speed: float
    running: bool = False
    _base: float = 0.0                       # simulator seconds accumulated before the current run
    _started: float | None = field(default=None, repr=False)   # mono time of the current start

    def now(self, mono: float) -> float:
        if self.running and self._started is not None:
            return self._base + (mono - self._started) * self.speed
        return self._base

    def start(self, mono: float) -> None:
        if not self.running:
            self.running, self._started = True, mono

    def pause(self, mono: float) -> None:
        if self.running:
            self._base = self.now(mono)
            self.running, self._started = False, None

    def reset(self) -> None:
        self.running, self._base, self._started = False, 0.0, None


def variant_of(case: Mapping[str, Any]) -> str:
    return case.get("simulated_variant") or "normal"


def _hero_score(case: Mapping[str, Any], spec: Mapping[str, Any]) -> int:
    """Number of demo.heroes criteria a case meets. Unknown criteria raise (config is read strictly)."""
    margin = case.get("margin") or {}
    readers: dict[str, Callable[[], Any]] = {
        "suggested_call": lambda: margin.get("suggested_call"),
        "involved": lambda: sorted(margin.get("involved") or []),
        "reliability_flag": lambda: case["quality"]["reliability_flag"],
        "simulated_variant": lambda: case.get("simulated_variant"),
    }
    score = 0
    for key, want in spec.items():
        if key not in readers:
            raise KeyError(f"demo.heroes criterion {key!r} is not understood by the simulator")
        got = readers[key]()
        score += int(got == (sorted(want) if key == "involved" else want))
    return score


class Simulator:
    """Owns the simulator clock and drives arrivals into the store."""

    def __init__(self, store: db.Store, cfg: Mapping[str, Any], loader: Callable[[], db.LoadResult],
                 clock: Clock | None = None):
        self.store = store
        self.cfg = cfg
        self.loader = loader
        self.clock = clock or Clock()
        sim_cfg = cfg["sim"]
        self.base_speed = float(sim_cfg["speed"])
        self.max_replay_wall_s = sim_cfg["max_replay_wall_s"]
        self.hold_surgical = bool(sim_cfg["hold_surgical_for_injection"])
        self.clock_sim = SimClock(speed=self.base_speed)
        self.heroes: Mapping[str, Any] = cfg["demo"]["heroes"]
        missing = [k for k in _HERO_KEY.values() if k not in self.heroes]
        if missing:
            raise KeyError(f"settings demo.heroes is missing {missing} (needed by inject_surgical)")
        self.loaded: db.LoadResult = loader()
        self.clock_sim.speed = self.replay_speed(self.loaded)
        restored = self.store.restore(self.loaded)
        if restored is None:
            self.store.seed(self.loaded, wall=self.clock.wall(), sim=0.0)
        else:
            # Resume explicitly: server downtime must not silently accelerate the arrival replay.
            self.clock_sim._base = restored
            log.info("restored workspace at simulator +%.1fs (paused); decisions preserved", restored)

    def replay_speed(self, loaded: db.LoadResult) -> float:
        """sim.speed, raised so the last self-arriving case arrives within sim.max_replay_wall_s of Start."""
        if self.max_replay_wall_s is None:
            return self.base_speed
        offsets = [float(c["arrived_offset_s"]) for c in loaded.cases
                   if not (self.hold_surgical and c["lane"] == "surgical")]
        needed = max(offsets, default=0.0) / float(self.max_replay_wall_s)
        if needed > self.base_speed:
            log.info("simulator speed raised from %gx to %.1fx so every arrival lands within %gs of Start",
                     self.base_speed, needed, self.max_replay_wall_s)
        return max(self.base_speed, needed)

    # ---- clocks -------------------------------------------------------------------------------------------------------
    def sim_now(self) -> float:
        return self.clock_sim.now(self.clock.mono())

    def wall_now(self) -> float:
        return self.clock.wall()

    def status(self) -> dict[str, Any]:
        with self.store.lock:
            return self._status()

    def _status(self) -> dict[str, Any]:
        pending = self.store.pending_rows()
        counts = self.store.counts()
        surgical = {v: 0 for v in VARIANTS}
        biopsy = 0
        for row in pending:
            if row["lane"] == "surgical":
                surgical[row["variant"] or "normal"] += 1
            else:
                biopsy += 1
        return {
            "running": self.clock_sim.running,
            "speed": self.clock_sim.speed,
            "base_speed": self.base_speed,
            "sim_time_s": self.sim_now(),
            "hold_surgical_for_injection": self.hold_surgical,
            "cases": counts["cases"],
            "arrived": counts["arrived"],
            "pending": {"biopsy": biopsy, "surgical": surgical},
        }

    # ---- arrivals -------------------------------------------------------------------------------------------------------
    def tick(self) -> list[str]:
        """Mark every case whose offset the simulator clock has passed as arrived; returns the new case ids."""
        with self.store.lock:
            sim, wall = self.sim_now(), self.wall_now()
            speed = self.clock_sim.speed
            arrived: list[str] = []
            for row in self.store.pending_rows():
                if row["arrived_offset_s"] > sim:
                    break   # pending_rows is ordered by offset
                if self.hold_surgical and row["lane"] == "surgical":
                    continue
                # wall time at which the simulator clock crossed the offset (exact while running, else now)
                lag = (sim - row["arrived_offset_s"]) / speed if self.clock_sim.running else 0.0
                self.store.mark_arrived(row["case_id"], arrived_sim=float(row["arrived_offset_s"]),
                                        arrived_wall=wall - lag, injected=False, wall=wall, sim=sim)
                arrived.append(row["case_id"])
            self.store.checkpoint_clock(sim)
            return arrived

    def inject_surgical(self, variant: str) -> str:
        """Make the next matching pending surgical case arrive now. Raises TransitionError if none is left."""
        if variant not in VARIANTS:
            raise db.PayloadError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
        with self.store.lock:
            candidates = [r for r in self.store.pending_rows()
                          if r["lane"] == "surgical" and (r["variant"] or "normal") == variant]
            if not candidates:
                raise db.TransitionError(f"no pending simulated frozen section of variant {variant!r} is left;"
                                         " reset the simulator to replay")
            spec = self.heroes[_HERO_KEY[variant]]

            def rank(row: Any) -> tuple[int, float, str]:
                case = json.loads(row["public_json"])
                return -_hero_score(case, spec), float(row["arrived_offset_s"]), row["case_id"]

            chosen = min(candidates, key=rank)
            score = _hero_score(json.loads(chosen["public_json"]), spec)
            if score < len(spec):                # settings demo.fallback: closest_match, logged
                log.info("inject_surgical(%s): no exact demo.heroes.%s match; closest is %s (%d/%d criteria)",
                         variant, _HERO_KEY[variant], chosen["case_id"], score, len(spec))
            sim, wall = self.sim_now(), self.wall_now()
            self.store.mark_arrived(chosen["case_id"], arrived_sim=sim, arrived_wall=wall, injected=True, wall=wall,
                                    sim=sim)
            return chosen["case_id"]

    # ---- controls ----------------------------------------------------------------------------------------------------------
    def start(self) -> None:
        with self.store.lock:
            if not self.clock_sim.running:
                self.clock_sim.start(self.clock.mono())
                self.store.log_event(None, "sim_start", wall=self.wall_now(), sim=self.sim_now())
            self.tick()

    def pause(self) -> None:
        with self.store.lock:
            self.tick()
            if self.clock_sim.running:
                self.clock_sim.pause(self.clock.mono())
                self.store.log_event(None, "sim_pause", wall=self.wall_now(), sim=self.sim_now())

    def reset(self) -> None:
        """Re-read the case JSONs, re-seed every table (drafted orders restored) and rewind the paused clock to 0."""
        with self.store.lock:
            self.loaded = self.loader()
            self.clock_sim.reset()
            self.clock_sim.speed = self.replay_speed(self.loaded)
            self.store.seed(self.loaded, wall=self.wall_now(), sim=0.0)
