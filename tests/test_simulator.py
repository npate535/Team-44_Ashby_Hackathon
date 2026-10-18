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


NO_SETTLE = {**CFG, "sim": {**CFG["sim"], "after_arrivals_speed": None}}   # keep the replay speed after arrivals


def test_surgical_cases_are_held_for_injection(tmp_path):
    sim, fake, store = _sim(tmp_path, [variant(fixture_case("demo_0002"), "s1", offset=1)], cfg=NO_SETTLE)
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


def test_missing_demo_hero_key_raises(tmp_path):
    heroes = {k: v for k, v in CFG["demo"]["heroes"].items() if k != "surgical_degraded"}
    cfg = {**CFG, "demo": {**CFG["demo"], "heroes": heroes}}
    with pytest.raises(KeyError, match="surgical_degraded"):
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


def test_replay_speed_brings_the_last_arrival_within_the_window(tmp_path):
    base = fixture_case("demo_0001")
    window = CFG["sim"]["max_replay_wall_s"]
    late = window * CFG["sim"]["speed"] * 4           # would take 4x the window at sim.speed
    cases = [variant(base, "early", offset=10), variant(base, "hero_last", offset=late),
             variant(fixture_case("demo_0002"), "held", offset=late * 10)]   # held surgical: ignored
    sim, fake, _ = _sim(tmp_path, cases)
    assert sim.clock_sim.speed == pytest.approx(late / window)
    assert sim.status()["base_speed"] == CFG["sim"]["speed"]
    sim.start()
    fake.advance(window)
    assert "hero_last" in sim.tick()
    sim.reset()
    assert sim.clock_sim.speed == pytest.approx(late / window)


def test_replay_speed_never_slows_below_sim_speed(tmp_path):
    sim, _, _ = _sim(tmp_path, [variant(fixture_case("demo_0001"), "b", offset=5)])
    assert sim.clock_sim.speed == CFG["sim"]["speed"]
    cfg = {**CFG, "sim": {**CFG["sim"], "max_replay_wall_s": None}}
    sim2, _, _ = _sim(tmp_path / "x", [variant(fixture_case("demo_0001"), "b", offset=1e9)], cfg=cfg)
    assert sim2.clock_sim.speed == CFG["sim"]["speed"]


def test_clock_drops_to_after_arrivals_speed_once_every_biopsy_is_in(tmp_path):
    base = fixture_case("demo_0001")
    cases = [variant(base, "b1", offset=30), variant(base, "b2", offset=60),
             variant(fixture_case("demo_0002"), "held", offset=90)]          # held surgical never blocks the settle
    sim, fake, _ = _sim(tmp_path, cases)
    replay = sim.clock_sim.speed
    sim.start()
    fake.advance(1.5)
    assert sim.tick() == ["b1"] and sim.clock_sim.speed == replay
    fake.advance(1.0)
    assert sim.tick() == ["b2"]
    assert sim.clock_sim.speed == CFG["sim"]["after_arrivals_speed"]
    t = sim.sim_now()
    fake.advance(10)
    assert sim.sim_now() == pytest.approx(t + 10 * CFG["sim"]["after_arrivals_speed"])   # continuous, no jump
    sim.reset()
    assert sim.clock_sim.speed == replay
