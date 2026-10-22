"""backend.vercel: the Redis snapshot round trip honours the simulator hold (no Redis: snapshot/restore only)."""

from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest

from backend import vercel
from backend.app import create_app
from tests.app_helpers import CFG, FIXTURES, FakeClock, dev_root

CAP_S = 600.0
CAPPED = {**CFG, "sim": {**CFG["sim"], "after_arrivals_max_s": CAP_S}}


class _Time:
    """Stands in for the time module inside backend.vercel (snapshot/restore read the wall clock directly)."""

    def __init__(self, fake: FakeClock):
        self.fake = fake

    def time(self) -> float:
        return self.fake.wall_s

    def monotonic(self) -> float:
        return self.fake.mono_s


def _workspace(root, db, fake):
    app = create_app(cfg=CAPPED, cases_dir=root / "cases", artifacts_root=root, db_path=db, fixtures_dir=FIXTURES,
                     clock=fake.clock(), frontend_dir=None, env={})
    ws = vercel.Workspace()
    ws.inner, ws.backend = app, app.state.backend
    ws.case_payloads = [(row["public_json"], row["case_id"])
                        for row in ws.backend.store._conn.execute("SELECT case_id, public_json FROM cases")]
    ws.evidence_hash = hashlib.sha256(json.dumps(sorted(ws.case_payloads)).encode()).hexdigest()
    ws.original_files = {}
    return ws


def test_cold_start_after_a_long_gap_stays_at_the_hold(tmp_path, monkeypatch):
    root = dev_root(tmp_path)
    fake = FakeClock()
    monkeypatch.setattr(vercel, "WORK", root)
    monkeypatch.setattr(vercel, "time", _Time(fake))
    warm = _workspace(root, tmp_path / "warm.sqlite", fake)
    sim = warm.backend.sim
    sim.start()
    for _ in range(200):
        fake.advance(5)
        sim.tick()
        if sim.status()["pending"]["biopsy"] == 0:
            break
    status = sim.status()
    assert status["pending"]["biopsy"] == 0 and status["hold_at_s"] == pytest.approx(sim.last_arrival_s() + CAP_S)
    saved = warm.snapshot()                                   # running at 1x, below the hold
    fake.advance(12 * 3600)                                   # overnight: no request, no function instance
    cold = _workspace(root, tmp_path / "cold.sqlite", fake)  # a fresh instance seeded at 0, then the saved state
    cold.restore(saved)
    status = cold.backend.sim.status()
    assert status["running"] is True and status["held"] is True
    assert status["sim_time_s"] == pytest.approx(sim.last_arrival_s() + CAP_S)
    assert status["speed"] == CFG["sim"]["after_arrivals_speed"] and status["pending"]["biopsy"] == 0
    # the next snapshot saves the held time, so later restores stay there too
    fake.advance(3600)
    cold.restore(cold.snapshot())
    assert cold.backend.sim.status()["sim_time_s"] == pytest.approx(sim.last_arrival_s() + CAP_S)


def test_restore_mid_replay_redates_instead_of_running_the_gap_at_the_replay_speed(tmp_path, monkeypatch):
    root = dev_root(tmp_path)
    fake = FakeClock()
    monkeypatch.setattr(vercel, "WORK", root)
    monkeypatch.setattr(vercel, "time", _Time(fake))
    warm = _workspace(root, tmp_path / "warm.sqlite", fake)
    sim = warm.backend.sim
    sim.start()
    fake.advance(1)
    sim.tick()
    assert sim.status()["pending"]["biopsy"] > 0
    saved = warm.snapshot()                                   # mid-replay at the boosted speed
    fake.advance(3 * 3600)
    cold = _workspace(root, tmp_path / "cold.sqlite", fake)
    cold.restore(saved)
    cold.backend.sim.tick()
    status = cold.backend.sim.status()
    assert status["pending"]["biopsy"] == 0 and status["held"] is True
    assert status["sim_time_s"] == pytest.approx(sim.last_arrival_s() + CAP_S)   # not +3 h x the replay speed


def test_storage_quota_error_is_identifiable_without_logging_command_data():
    async def run():
        ws = vercel.Workspace()
        async with httpx.AsyncClient(base_url="https://storage.invalid", transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": "ERR monthly request limit exceeded"})
        )) as ws.redis:
            with pytest.raises(RuntimeError, match=r"quota exceeded \(HTTP 400\)") as exc:
                await ws.command("SET", "private-key", "private-review")
            assert "private" not in str(exc.value)
    asyncio.run(run())


def test_empty_integration_credentials_fail_before_sending_a_request(monkeypatch):
    for key in ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KV_REST_API_URL", "")
    monkeypatch.setenv("KV_REST_API_TOKEN", "")
    with pytest.raises(RuntimeError, match="credentials are missing or invalid"):
        asyncio.run(vercel.Workspace().command("PING"))


def test_storage_failure_and_failed_release_still_return_one_503(tmp_path, monkeypatch):
    root = dev_root(tmp_path)
    fake = FakeClock()
    monkeypatch.setattr(vercel, "WORK", root)
    ws = _workspace(root, tmp_path / "workspace.sqlite", fake)
    ws.seed_snapshot = ws.snapshot()

    async def command(*args):
        if args[0] == "SET":
            return "OK"
        if args[0] == "GET":
            return ws.seed_snapshot
        raise RuntimeError("Storage unavailable during save and release")

    ws.command = command

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=ws), base_url="http://test") as client:
            response = await client.get("/api/health")
            assert response.status_code == 503
            assert response.json() == {"detail": "Workspace storage is unavailable; please retry."}
    asyncio.run(run())
