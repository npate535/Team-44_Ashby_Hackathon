"""eval/queue_sim.py: policy mechanics on hand-made pools, paired comparison, honest cost to non-flagged cases."""

from __future__ import annotations

import copy

import pytest

from core import config, io, schemas
from eval import queue_sim as Q
from tests import downstream_fixture as D

CFG = copy.deepcopy(config.load_settings())
CLASSES = list(CFG["classes"])


def item(cid, top, conformal_set, lane="biopsy", truth=None, flag=False, qc=12):
    probs = [0.1] * 4
    probs[CLASSES.index(top)] = 0.7
    case = {"case_id": cid, "lane": lane, "arrived_offset_s": 0.0,
            "prediction": {"classes": CLASSES, "probs": probs, "conformal_set": list(conformal_set),
                           "risk_score": probs[3]},
            "quality": {"reliability_flag": flag, "qc_tiles_kept": qc}}
    from core import tiers
    tier, reasons = tiers.assign_tier(case, CFG)
    return {"case": case, "truth": truth or top, "tier": tier, "reasons": reasons}


POOL = [item("n", "Normal", ["Normal"]), item("i", "Invasive", ["Invasive"]),
        item("s", "Normal", ["Normal"], lane="surgical"), item("u", "Benign", ["Benign", "InSitu"])]


def order(records):
    return [POOL[r["pool"]]["case"]["case_id"] for r in records]


def test_fifo_reads_in_arrival_order_and_tiers_by_priority():
    arrivals = [(0.0, 0), (1.0, 0), (2.0, 3), (3.0, 1), (4.0, 2)]     # n, n, u, i, s
    assert order(Q.simulate_day(POOL, arrivals, "fifo", CFG)) == ["n", "n", "u", "i", "s"]
    # reader busy with the first n (0-9 min); at 9 min the queue holds n, u, i, s -> surgical, invasive, uncertain, n
    assert order(Q.simulate_day(POOL, arrivals, "tiers", CFG)) == ["n", "s", "i", "u", "n"]


def test_waits_and_overtime_are_computed():
    read = CFG["queue_sim"]["read_time_min"]
    recs = Q.simulate_day(POOL, [(0.0, 0), (0.0, 1)], "fifo", CFG)
    assert [r["wait"] for r in recs] == [0.0, read]
    late = Q.simulate_day(POOL, [(CFG["queue_sim"]["day_hours"] * 60.0 - 1.0, 0)], "fifo", CFG)
    assert Q.day_metrics(late, POOL, CFG)["overtime"] == pytest.approx(read - 1.0)


def test_aging_guard_promotes_a_long_waiting_case():
    cfg = copy.deepcopy(CFG)
    cfg["aging"]["turnaround_target_min"] = {1: 20, 2: 600, 3: 600, 4: 20}   # tier 4 promotes after 15 min
    arrivals = [(0.0, 3), (0.0, 0), (5.0, 3), (10.0, 3)]                     # u, n, u, u
    plain = order(Q.simulate_day(POOL, arrivals, "tiers", cfg))
    aged = order(Q.simulate_day(POOL, arrivals, "tiers_aging", cfg))
    assert plain == ["u", "u", "u", "n"] and aged.index("n") < plain.index("n")


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="policy"):
        Q.simulate_day(POOL, [(0.0, 0)], "random", CFG)


def test_simulate_is_paired_and_deterministic():
    a = Q.simulate(POOL, CFG, reps=20)
    b = Q.simulate(POOL, CFG, reps=20)
    assert a == b
    assert a["tiers"]["aggressive_mean"]["mean"] <= a["fifo"]["aggressive_mean"]["mean"]
    assert a["tiers"]["vs_fifo"]["non_flagged_mean"]["mean"] >= 0     # the cost to non-flagged cases is reported
    assert set(a) == set(CFG["queue_sim"]["policies"]) | {f"tiers_aging_{n}" for n in CFG["queue_sim"]["aging_scenarios"]}


def test_run_on_the_eval_fixture(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    out = Q.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art", step="eval_queue_sim"), reps=30)
    q = io.read_json(out)
    assert q["name"] == "queue_sim" and q["synthetic"] is True and q["reps"] == 30
    assert q["n_cases"] == 48 and sum(q["test_case_tiers"].values()) == 48
    assert {"arrivals", "reader", "utilization", "tiers", "aging", "interval"} <= set(q["assumptions"])
    fifo, tiered = q["results"]["fifo"], q["results"]["tiers"]
    assert tiered["aggressive_mean"]["mean"] < fifo["aggressive_mean"]["mean"]
    for block in (fifo["aggressive_p90"], tiered["vs_fifo"]["non_flagged_mean"]):
        assert block["lo"] <= block["mean"] <= block["hi"]
    assert schemas.eval_json(tmp_path / "art", "queue_sim") == out


def test_configured_aging_never_fires_and_the_sensitivity_run_is_labelled(tmp_path):
    D.write_eval_root(tmp_path / "data", CFG)
    q = io.read_json(Q.run(D.make_ctx(CFG, tmp_path / "data", tmp_path / "art"), reps=40))
    res = q["results"]
    assert res["tiers_aging"]["promoted_fraction"]["mean"] == 0
    assert res["tiers_aging"]["aggressive_mean"] == res["tiers"]["aggressive_mean"]
    assert any("NO measured effect" in n for n in q["notes"])
    assert "tiers_aging_day_scale" in res and "Sensitivity" in q["policies"]["tiers_aging_day_scale"]
    assert res["tiers_aging_day_scale"]["promoted_fraction"]["mean"] > 0
    assert any("sensitivity run" in n for n in q["notes"])
