"""backend.db: case loading, the allowed-action table, edit validation, seeding/reset semantics."""

from __future__ import annotations

import json

import pytest

from backend import db
from core import io
from tests.app_helpers import FIXTURES, fixture_case, variant, write_cases

WALL = 1_800_000_000.0


def test_load_case_dir_validates_and_strips_eval_only(tmp_path):
    folder = write_cases(tmp_path / "c", [fixture_case("demo_0001"), fixture_case("demo_0002")])
    dup = fixture_case("demo_0001")
    io.write_json(folder / "zz_duplicate.json", dup)
    (folder / "notes.txt").write_text("ignored")
    loaded = db.load_case_dir(folder)
    assert [c["case_id"] for c in loaded.cases] == ["demo_0001", "demo_0002"]
    assert all("eval_only" not in c for c in loaded.cases)
    assert loaded.skipped == [{"file": "zz_duplicate.json", "error": "duplicate case_id demo_0001"}]
    assert loaded.sources == {"demo_0001": "demo_0001.json", "demo_0002": "demo_0002.json"}


def test_load_case_dir_skips_invalid_with_reason(tmp_path):
    bad = fixture_case("demo_0001")
    bad["lane"] = "colon"
    folder = write_cases(tmp_path / "c", [bad])
    (folder / "empty.json").write_text("")
    loaded = db.load_case_dir(folder)
    assert loaded.cases == []
    reasons = {s["file"]: s["error"] for s in loaded.skipped}
    assert reasons["demo_0001.json"].startswith("lane:")
    assert "JSONDecodeError" in reasons["empty.json"]


def test_one_rule_may_draft_several_orders(tmp_path):
    case = fixture_case("demo_0001")
    case["agent"]["orders"][1]["rule_id"] = case["agent"]["orders"][0]["rule_id"]
    store = _store(tmp_path, [case])
    store.mark_arrived("demo_0001", arrived_sim=1, arrived_wall=WALL, injected=False, wall=WALL, sim=1)
    store.decide("demo_0001", "edit_orders", {"orders": [{"seq": 1, "items": ["HER2"]}]}, wall=WALL, sim=2)
    assert [(o["seq"], o["status"], o["items"]) for o in store.orders("demo_0001")] == [
        (0, "removed", []), (1, "edited", ["HER2"])]


def test_load_case_dir_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        db.load_case_dir(tmp_path / "missing")


def test_fixtures_load_cleanly():
    loaded = db.load_case_dir(FIXTURES)
    assert len(loaded.cases) == 2 and loaded.skipped == []


def _store(tmp_path, cases):
    store = db.Store(tmp_path / "s.sqlite")
    store.seed(db.load_case_dir(write_cases(tmp_path / "c", cases)), wall=WALL)
    return store


def test_seed_creates_every_table_and_drafted_orders(tmp_path):
    store = _store(tmp_path, [fixture_case("demo_0001"), fixture_case("demo_0002")])
    tables = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(db.TABLES) <= tables
    orders = store.orders("demo_0001")
    assert [(o["rule_id"], o["status"]) for o in orders] == [("breast_insitu_vs_invasive", "drafted"),
                                                            ("breast_invasive", "drafted")]
    assert store.orders("demo_0002") == []
    stored = json.loads(store.row("demo_0001")["public_json"])
    assert "eval_only" not in stored
    assert [e["type"] for e in store.events()] == ["reset"]


def test_reseed_restores_state(tmp_path):
    store = _store(tmp_path, [fixture_case("demo_0001")])
    store.mark_arrived("demo_0001", arrived_sim=340, arrived_wall=WALL, injected=False, wall=WALL, sim=340)
    store.decide("demo_0001", "edit_orders", {"orders": [{"seq": 1, "items": ["ER"]}]},
                 wall=WALL, sim=340)
    store.decide("demo_0001", "sign_out", {}, wall=WALL, sim=341)
    assert store.routed("oncologist")
    store.seed(db.load_case_dir(tmp_path / "c"), wall=WALL)
    row = store.row("demo_0001")
    assert row["agent_status"] == "drafted" and not row["arrived"] and row["signed_out_wall"] is None
    assert [o["items"] for o in store.orders("demo_0001")] == [["p63", "calponin"], ["ER", "PR", "HER2", "Ki-67"]]
    assert store.routed("oncologist") == [] and store.decisions("demo_0001") == []


def test_decisions_table_records_each_action(tmp_path):
    store = _store(tmp_path, [fixture_case("demo_0001")])
    store.mark_arrived("demo_0001", arrived_sim=340, arrived_wall=WALL, injected=False, wall=WALL, sim=340)
    store.decide("demo_0001", "closer_look", {}, wall=WALL + 1, sim=341)
    store.decide("demo_0001", "dismiss_flag", {}, wall=WALL + 2, sim=342)
    assert [(d["action"], d["from"], d["to"]) for d in store.decisions("demo_0001")] == [
        ("closer_look", "drafted", "deferred_closer_look"), ("dismiss_flag", "deferred_closer_look", "dismissed")]


def test_decide_rejects_unarrived_and_unknown(tmp_path):
    store = _store(tmp_path, [fixture_case("demo_0001")])
    with pytest.raises(db.NotFoundError):
        store.decide("demo_0001", "approve_orders", {}, wall=WALL, sim=0)
    with pytest.raises(db.NotFoundError):
        store.decide("nope", "approve_orders", {}, wall=WALL, sim=0)
    with pytest.raises(db.PayloadError):
        store.decide("demo_0001", "launch", {}, wall=WALL, sim=0)


def test_mark_arrived_twice_is_refused(tmp_path):
    store = _store(tmp_path, [fixture_case("demo_0002")])
    store.mark_arrived("demo_0002", arrived_sim=0, arrived_wall=WALL, injected=True, wall=WALL, sim=0)
    assert store.row("demo_0002")["timer_started_wall"] == WALL
    with pytest.raises(db.TransitionError):
        store.mark_arrived("demo_0002", arrived_sim=0, arrived_wall=WALL, injected=True, wall=WALL, sim=0)


def _row(**kw):
    base = {"arrived": 1, "lane": "biopsy", "agent_status": "drafted", "signed_out_wall": None,
            "margin_call_json": None}
    return {**base, **kw}


@pytest.mark.parametrize("status, allowed", [
    ("drafted", ["approve_orders", "edit_orders", "closer_look", "dismiss_flag"]),
    ("deferred_closer_look", ["approve_orders", "edit_orders", "dismiss_flag"]),
    ("none", ["sign_out"]), ("approved", ["sign_out"]), ("edited_and_approved", ["sign_out"]),
    ("dismissed", ["sign_out"]),
])
def test_allowed_actions_biopsy(status, allowed):
    assert db.allowed_actions(_row(agent_status=status)) == allowed


def test_allowed_actions_terminal_and_surgical():
    assert db.allowed_actions(_row(arrived=0)) == []
    assert db.allowed_actions(_row(agent_status="approved", signed_out_wall=WALL)) == []
    assert db.allowed_actions(_row(lane="surgical", agent_status="none")) == ["margin_call"]
    assert db.allowed_actions(_row(lane="surgical", agent_status="none", margin_call_json="{}")) == []


def test_edit_keeps_drafted_order_of_items(tmp_path):
    store = _store(tmp_path, [fixture_case("demo_0001")])
    rows = store._conn.execute("SELECT * FROM orders WHERE case_id='demo_0001' ORDER BY seq").fetchall()
    kept = db._validate_edit({"orders": [{"seq": 1, "rule_id": "breast_invasive", "items": ["Ki-67", "PR"]},
                                         {"seq": 0, "items": ["p63"]}]}, rows)
    assert kept == {1: ["PR", "Ki-67"], 0: ["p63"]}
    for bad in ({"orders": [{"seq": 1, "items": ["ER", "ER"]}]},
                {"orders": [{"seq": 1, "items": ["ER"]}, {"seq": 1, "items": ["PR"]}]},
                {"orders": [{"seq": 1, "items": ["ER"], "note": "x"}]},
                {"orders": [{"seq": 1, "rule_id": "breast_insitu_vs_invasive", "items": ["ER"]}]},
                {"orders": [{"seq": 7, "items": ["ER"]}]}, {"orders": [{"seq": True, "items": ["ER"]}]},
                {"orders": [{"rule_id": "breast_invasive", "items": ["ER"]}]},
                {"orders": []}, {}, {"orders": [], "extra": 1}):
        with pytest.raises(db.PayloadError):
            db._validate_edit(bad, rows)


def test_margin_call_validation():
    assert db._validate_margin_call({"call": "tumor_at_margin", "margins": ["lateral", "superior"]}) == (
        "tumor_at_margin", ["superior", "lateral"])
    assert db._validate_margin_call({"call": "defer"}) == ("defer", [])
    for bad in ({}, {"margins": []}, {"call": "clear", "margins": ["medial"]},
                {"call": "tumor_at_margin", "margins": ["medial", "medial"]}, {"call": "clear", "x": 1}):
        with pytest.raises(db.PayloadError):
            db._validate_margin_call(bad)


def test_sign_out_payload_never_includes_ground_truth(tmp_path):
    store = _store(tmp_path, [variant(fixture_case("demo_0001"), "case_a", agent={"status": "none", "orders": []})])
    store.mark_arrived("case_a", arrived_sim=1, arrived_wall=WALL, injected=False, wall=WALL, sim=1)
    store.decide("case_a", "sign_out", {}, wall=WALL, sim=2)
    text = json.dumps(store.routed("oncologist"))
    assert "eval_only" not in text and "true_label" not in text


def test_allowed_actions_surgical_with_drafted_recut():
    drafted = _row(lane="surgical", agent_status="drafted")
    assert db.allowed_actions(drafted) == ["margin_call", "approve_orders", "edit_orders", "closer_look", "dismiss_flag"]
    called = _row(lane="surgical", agent_status="drafted", margin_call_json="{}")
    assert db.allowed_actions(called) == ["approve_orders", "edit_orders", "closer_look", "dismiss_flag"]
    assert "sign_out" not in db.allowed_actions(_row(lane="surgical", agent_status="approved"))


def test_surgical_stain_orders_are_rejected_at_load(tmp_path):
    case = fixture_case("demo_0002")
    case["agent"] = {"status": "drafted", "orders": [{"type": "stain", "items": ["ER"], "rule_id": "breast_invasive",
                                                       "rationale": "x"}]}
    loaded = db.load_case_dir(write_cases(tmp_path / "c", [case]))
    assert loaded.cases == [] and "never allowed in the surgical lane" in loaded.skipped[0]["error"]
