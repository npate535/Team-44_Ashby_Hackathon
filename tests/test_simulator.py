"""backend.simulator: the simulator clock, arrivals, surgical holding and hero-first injection."""

from __future__ import annotations

import pytest

from backend import db
from backend.simulator import SimClock, Simulator, _hero_score, variant_of
from tests.app_helpers import CFG, FakeClock, fixture_case, variant, write_cases


def test_sim_clock_start_pause_reset():
    clock = SimClock(speed=30)
    assert clock.now(100) == 0 and not clock.running
    clock.start(100)
    assert clock.now(101) == 30
    clock.start(101)                       # starting twice keeps the original start
    assert clock.now(102) == 60
    clock.pause(102)
    assert clock.now(500) == 60 and not clock.running
    clock.pause(600)
    assert clock.now(700) == 60
    clock.start(700)
    assert clock.now(702) == 120
    clock.reset()
    assert clock.now(800) == 0 and not clock.running


def _sim(tmp_path, cases, cfg=CFG):
    folder = write_cases(tmp_path / "c", cases)
    fake = FakeClock()
    store = db.Store(tmp_path / "s.sqlite")
    return Simulator(store, cfg, lambda: db.load_case_dir(folder), fake.clock()), fake, store


def test_tick_marks_arrivals_in_offset_order(tmp_path):
    base = fixture_case("demo_0001")
    cases = [variant(base, f"c{i}", offset=off) for i, off in enumerate((300, 30, 90))]
    sim, fake, store = _sim(tmp_path, cases)
    assert sim.tick() == []
    sim.start()
    fake.advance(2)                        # 60 sim seconds
    assert sim.tick() == ["c1"]
    fake.advance(1)
    assert sim.tick() == ["c2"]
    fake.advance(10)
    assert sim.tick() == ["c0"]
    row = store.row("c0")
    assert row["arrived_sim_s"] == 300
    # wall time of the crossing, not of the tick: 300 sim s at 30x = 10 s after start
    assert row["arrived_wall"] == pytest.approx(1_800_000_000.0 + 10)
    assert sim.status()["arrived"] == 3


def test_surgical_cases_are_held_for_injection(tmp_path):
    sim, fake, store = _sim(tmp_path, [variant(fixture_case("demo_0002"), "s1", offset=1)])
    sim.start()
    fake.advance(100)
    assert sim.tick() == [] and sim.status()["pending"]["surgical"]["normal"] == 1
    assert sim.inject_surgical("normal") == "s1"
    assert store.row("s1")["arrived_sim_s"] == pytest.approx(3000)
    with pytest.raises(db.TransitionError):
        sim.inject_surgical("normal")
    with pytest.raises(db.PayloadError):
        sim.inject_surgical("frozen")


def test_surgical_arrive_on_their_own_when_not_held(tmp_path):
    cfg = {**CFG, "sim": {**CFG["sim"], "hold_surgical_for_injection": False}}
    sim, fake, store = _sim(tmp_path, [variant(fixture_case("demo_0002"), "s1", offset=30)], cfg=cfg)
    sim.start()
    fake.advance(2)
    assert sim.tick() == ["s1"]
    assert store.row("s1")["timer_started_wall"] == pytest.approx(1_800_000_000.0 + 1)


def test_missing_sim_setting_raises(tmp_path):
    cfg = {**CFG, "sim": {k: v for k, v in CFG["sim"].items() if k != "hold_surgical_for_injection"}}
    with pytest.raises(KeyError):
        _sim(tmp_path, [fixture_case("demo_0001")], cfg=cfg)


def test_hero_score_and_unknown_criteria():
    hero = fixture_case("demo_0002")
    assert _hero_score(hero, CFG["demo"]["heroes"]["surgical"]) == 3
    assert _hero_score(hero, {"involved": ["lateral"]}) == 0
    with pytest.raises(KeyError):
        _hero_score(hero, {"color": "red"})
    assert variant_of(hero) == "normal" and variant_of({"simulated_variant": "degraded"}) == "degraded"


def test_reset_rewinds_and_reseeds(tmp_path):
    sim, fake, store = _sim(tmp_path, [variant(fixture_case("demo_0001"), "b1", offset=10)])
    sim.start()
    fake.advance(5)
    sim.tick()
    store.decide("b1", "approve_orders", {}, wall=fake.wall_s, sim=sim.sim_now())
    sim.reset()
    assert sim.sim_now() == 0 and not sim.clock_sim.running
    assert store.row("b1")["agent_status"] == "drafted" and not store.row("b1")["arrived"]
    assert [e["type"] for e in store.events()] == ["reset"]
