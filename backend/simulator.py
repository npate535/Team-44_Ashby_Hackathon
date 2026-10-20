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

After the replay: once every self-arriving case is in, the clock runs at `sim.after_arrivals_speed` and holds at
(last scheduled arrival + `sim.after_arrivals_max_s`), so an unattended demo does not age every case past its target
overnight. Held, the clock still accepts Start / Pause and frozen-section injection (their timers run on the wall
clock); status() reports `held`. Reset clears the hold.
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
    limit: float | None = None               # simulator time the clock holds at (None = no hold)

    def _unheld(self, mono: float) -> float:
        if self.running and self._started is not None:
            return self._base + (mono - self._started) * self.speed
        return self._base

    def now(self, mono: float) -> float:
        t = self._unheld(mono)
        return t if self.limit is None else min(t, self.limit)

    def held(self, mono: float) -> bool:
        """True once the clock has reached its hold limit (it no longer advances)."""
        return self.limit is not None and self._unheld(mono) >= self.limit

    def start(self, mono: float) -> None:
        if not self.running:
            self.running, self._started = True, mono

    def pause(self, mono: float) -> None:
        if self.running:
            self._base = self.now(mono)
            self.running, self._started = False, None

    def reset(self) -> None:
        self.running, self._base, self._started, self.limit = False, 0.0, None, None


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
        after = sim_cfg.get("after_arrivals_speed")
        self.after_arrivals_speed = None if after is None else float(after)
        cap = sim_cfg.get("after_arrivals_max_s")
        self.after_arrivals_max_s = None if cap is None else float(cap)
        if self.after_arrivals_max_s is not None and not self.after_arrivals_max_s >= 0:
            raise ValueError(f"sim.after_arrivals_max_s must be >= 0 or null, got {cap!r}")
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
            self._resume_after_arrivals(restored)
            log.info("restored workspace at simulator +%.1fs (paused, %gx); decisions preserved", restored,
                     self.clock_sim.speed)

    def _self_arriving(self, lane: str) -> bool:
        return not (self.hold_surgical and lane == "surgical")

    def replay_speed(self, loaded: db.LoadResult) -> float:
        """sim.speed, raised so the last self-arriving case arrives within sim.max_replay_wall_s of Start."""
        if self.max_replay_wall_s is None:
            return self.base_speed
        needed = self.last_arrival_s(loaded) / float(self.max_replay_wall_s)
        if needed > self.base_speed:
            log.info("simulator speed raised from %gx to %.1fx so every arrival lands within %gs of Start",
                     self.base_speed, needed, self.max_replay_wall_s)
        return max(self.base_speed, needed)

    def last_arrival_s(self, loaded: db.LoadResult | None = None) -> float:
        """Simulator time of the last scheduled (self-arriving) arrival: when the replay is complete."""
        cases = (loaded or self.loaded).cases
        return max((float(c["arrived_offset_s"]) for c in cases if self._self_arriving(c["lane"])), default=0.0)

    def _scheduled_pending(self) -> bool:
        return any(self._self_arriving(r["lane"]) for r in self.store.pending_rows())

    def _hold(self, floor: float) -> float | None:
        """Every scheduled arrival is in: hold the clock at the last one + sim.after_arrivals_max_s. Never below `floor`
        (the simulator time already reached in this run or saved by an older build): the clock does not run backwards.
        Returns the hold limit (None = no cap configured)."""
        clock = self.clock_sim
        if self.after_arrivals_max_s is not None and clock.limit is None:
            clock.limit = max(self.last_arrival_s() + self.after_arrivals_max_s, floor)
            log.info("replay complete: the simulator clock holds at +%.1fs", clock.limit)
        return clock.limit

    def _resume_after_arrivals(self, sim: float) -> None:
        """A restored clock with nothing left to arrive: after-arrivals speed (not the replay speed) and the hold."""
        if self._scheduled_pending():
            return
        clock = self.clock_sim
        if self.after_arrivals_speed is not None and clock.speed > self.after_arrivals_speed:
            if clock.running:
                return                           # the next tick() dates the switch to the last arrival, then holds
            clock.speed = self.after_arrivals_speed
        self._hold(floor=sim)

    def resume(self, *, running: bool, speed: float, sim_time_s: float, wall_gap_s: float) -> None:
        """Restore a saved clock over an already restored store (backend/vercel.py, once per request).

        A running clock resumes as if it had kept running for `wall_gap_s`: the next tick re-dates the after-arrivals
        switch to the last arrival (from the saved simulator time, not from the extrapolated one) and holds at the cap,
        so a cold start after a long gap neither runs the replay speed through the gap nor drifts past the hold."""
        with self.store.lock:
            clock = self.clock_sim
            clock.running, clock.speed, clock._base, clock.limit = bool(running), float(speed), float(sim_time_s), None
            clock._started = self.clock.mono() - max(0.0, float(wall_gap_s)) if clock.running else None
            self._resume_after_arrivals(float(sim_time_s))

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
        mono = self.clock.mono()
        return {
            "running": self.clock_sim.running,
            "speed": self.clock_sim.speed,
            "base_speed": self.base_speed,
            "sim_time_s": self.clock_sim.now(mono),
            "held": self.clock_sim.held(mono),          # replay complete and the clock reached its hold limit
            "hold_at_s": self.clock_sim.limit,          # where it holds (null until every scheduled case is in)
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
            speed, running = self.clock_sim.speed, self.clock_sim.running
            due: list[Any] = []
            waiting = False                      # a self-arriving case is still ahead on the simulator clock
            for row in self.store.pending_rows():
                if not self._self_arriving(row["lane"]):
                    continue
                if row["arrived_offset_s"] > sim:
                    waiting = True
                    break   # pending_rows is ordered by offset
                due.append(row)
            stamp = sim
            if not waiting:
                started_at = self.clock_sim._base            # simulator time at this Start (or restore)
                stamp = self._settle_after_arrivals(sim, max((float(r["arrived_offset_s"]) for r in due), default=None))
                limit = self._hold(floor=started_at)
                if limit is not None:
                    stamp = min(stamp, limit)
            arrived: list[str] = []
            for row in due:
                # wall time at which the simulator clock crossed the offset (exact while running, else now)
                lag = (sim - row["arrived_offset_s"]) / speed if running else 0.0
                self.store.mark_arrived(row["case_id"], arrived_sim=float(row["arrived_offset_s"]),
                                        arrived_wall=wall - lag, injected=False, wall=wall, sim=stamp)
                arrived.append(row["case_id"])
            self.store.checkpoint_clock(stamp)
            return arrived

    def _settle_after_arrivals(self, sim: float, last_due: float | None) -> float:
        """Once every self-arriving case is in, run the clock at sim.after_arrivals_speed (1 = wall time) so biopsy waits
        keep their meaning: at the boosted replay speed every case would pass its target within minutes and the aging
        guard would promote the whole queue. Reset restores the replay speed.

        Arrivals are checked lazily (every API call ticks), so the switch is dated to when the last scheduled arrival
        came due in this run, not to this tick: otherwise the clock keeps the replay speed for as long as nobody polls
        (a backgrounded tab polls about once a minute: ~2.7 simulated hours at 160x) and the hero opens far past its
        target. Every arrival stays at or before the corrected time. Returns the simulator time now, before the hold
        (tick() then caps it at sim.after_arrivals_max_s past the last arrival)."""
        target = self.after_arrivals_speed
        clock = self.clock_sim
        if target is None or not clock.running or clock.speed <= target:
            return sim
        knee = clock._base if last_due is None else max(clock._base, last_due)   # _base: simulator time at this Start
        settled = knee + max(0.0, sim - knee) * target / clock.speed
        mono = self.clock.mono()
        clock.pause(mono)
        clock._base, clock.speed = settled, target
        clock.start(mono)
        log.info("every scheduled arrival is in: simulator speed now %gx (from +%.1fs)", target, knee)
        return settled

    def inject_surgical(self, variant: str) -> str:
        """Make the next matching pending surgical case arrive now. Raises TransitionError if none is left."""
        if variant not in VARIANTS:
            raise db.PayloadError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
        with self.store.lock:
            self.tick()                          # cases already due arrive first (event order, settled clock)
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

    def pause_for_shutdown(self) -> None:
        """Process shutdown: deliver due arrivals and checkpoint the clock, without logging a user pause."""
        with self.store.lock:
            self.tick()

    def reset(self) -> None:
        """Re-read the case JSONs, re-seed every table (drafted orders restored) and rewind the paused clock to 0."""
        with self.store.lock:
            self.loaded = self.loader()
            self.clock_sim.reset()
            self.clock_sim.speed = self.replay_speed(self.loaded)
            self.store.seed(self.loaded, wall=self.wall_now(), sim=0.0)
