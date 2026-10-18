"""HTTP API tests (BUILD_PLAN Task 6): sorting, decisions, routing, timers, simulator, media, metrics, no leaks."""

from __future__ import annotations

import json

import pytest

from core import contract, io, schemas
from tests.app_helpers import (CFG, FIXTURES, arrive_all_biopsies, dev_roles, dev_root, find_keys, fixture_case, ids,
                               make_client, row_for, variant, write_cases)

AMBER_S = CFG["margin"]["timer"]["amber_min"] * 60
RED_S = CFG["margin"]["timer"]["red_min"] * 60
SPEED = CFG["sim"]["speed"]


@pytest.fixture()
def dev(tmp_path):
    root = dev_root(tmp_path)
    client, clock = make_client(tmp_path, artifacts_root=root)
    return client, clock, dev_roles(root), root


def decide(client, case_id, action, payload=None):
    return client.post(f"/api/cases/{case_id}/decision", json={"action": action, "payload": payload or {}})


# ---- health, config, loading ---------------------------------------------------------------------------------------------
def test_health_reports_loaded_and_skipped_files(tmp_path):
    cases = write_cases(tmp_path / "cases", [fixture_case("demo_0001")])
    (cases / "broken.json").write_text("{not json")
    bad = fixture_case("demo_0001")
    bad["case_id"] = "bad_probs"
    bad["prediction"]["probs"] = [0.9, 0.9, 0.1, 0.1]
    io.write_json(cases / "bad_probs.json", bad)
    client, _ = make_client(tmp_path, cases_dir=cases)
    health = client.get("/api/health").json()
    assert health["loaded"] == 1 and health["synthetic"] is True
    assert sorted(s["file"] for s in health["skipped"]) == ["bad_probs.json", "broken.json"]
    assert client.get("/api/health").status_code == 200


def test_missing_cases_dir_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="make demo CASES=fixtures"):
        make_client(tmp_path, cases_dir=tmp_path / "nope")


def test_config_exposes_ui_settings(tmp_path):
    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    cfg = client.get("/api/config").json()
    assert cfg["banner"] == CFG["ui"]["banner"]
    assert cfg["poll_ms"] == CFG["ui"]["poll_ms"]
    assert cfg["timer"] == CFG["margin"]["timer"]
    assert cfg["margin"]["note"] == CFG["margin"]["note"]
    assert cfg["labels"]["tiers"]["2"] == CFG["ui"]["tier_labels"][2]
    assert set(cfg["labels"]["reasons"]) == set(contract.FLAG_REASONS)
    assert cfg["aging"]["turnaround_target_min"]["4"] == CFG["aging"]["turnaround_target_min"][4]
    assert cfg["cells"]["enabled"] == CFG["cells"]["enabled"]
    assert cfg["cells"]["overlay_colors"] == CFG["cells"]["overlay_colors"]
    assert cfg["classes"] == CFG["classes"] and cfg["aggressive_class"] == "Invasive"


# ---- arrivals and sorting -------------------------------------------------------------------------------------------------
def test_only_arrived_cases_appear(dev):
    client, clock, roles, _ = dev
    assert client.get("/api/worklist").json()["items"] == []
    assert client.get(f"/api/cases/{roles['hero_biopsy']}").status_code == 404
    client.post("/api/sim/start")
    clock.advance(1)
    first = client.get("/api/worklist").json()
    sim = first["sim"]["sim_time_s"]
    assert sim == pytest.approx(SPEED)
    for item in first["items"]:
        assert item["arrived_offset_s"] <= sim and item["lane"] == "biopsy"
    wl = arrive_all_biopsies(client, clock)
    assert all(item["lane"] == "biopsy" for item in wl["items"])   # surgical cases wait for injection
    assert wl["sim"]["pending"]["surgical"] == {"normal": 2, "degraded": 1, "composite": 1}


def test_ai_order_follows_tiers_risk_arrival(dev):
    client, clock, roles, _ = dev
    wl = arrive_all_biopsies(client, clock)
    tiers_seen = [item["runtime"]["tier"] for item in wl["items"]]
    assert tiers_seen == sorted(tiers_seen)
    for a, b in zip(wl["items"], wl["items"][1:]):
        if a["runtime"]["tier"] == b["runtime"]["tier"]:
            ra, rb = a["prediction"]["risk_score"], b["prediction"]["risk_score"]
            assert ra > rb or (ra == rb and a["runtime"]["arrived_sim_s"] <= b["runtime"]["arrived_sim_s"])
    tier2 = ids(wl, lambda i: i["runtime"]["tier"] == 2)
    assert tier2 == [roles["invasive_confident"], roles["hero_biopsy"]]
    hero = row_for(wl, roles["hero_biopsy"])
    assert "aggressive_in_conformal_set" in hero["runtime"]["reasons"]
    assert hero["agent"]["orders_drafted"] is True and hero["agent"]["n_orders"] == 2
    # the hero arrived last but sits above every earlier tier-3/4 arrival
    arrival = client.get("/api/worklist?order=arrival").json()
    assert ids(arrival)[-1] == roles["hero_biopsy"]
    assert ids(wl).index(roles["hero_biopsy"]) < ids(wl).index(roles["routine_normal"])


def test_arrival_order_and_lane_filter(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    client.post("/api/sim/inject_surgical", json={"variant": "normal"})
    arrival = client.get("/api/worklist?order=arrival").json()
    times = [i["runtime"]["arrived_sim_s"] for i in arrival["items"]]
    assert times == sorted(times)
    assert ids(arrival)[-1] == roles["hero_surgical"]          # injected now = latest arrival
    ai = client.get("/api/worklist").json()
    assert ids(ai)[0] == roles["hero_surgical"]                # but tier 1 on AI priority
    surgical = client.get("/api/worklist?lane=surgical").json()
    assert ids(surgical) == [roles["hero_surgical"]]
    biopsy = client.get("/api/worklist?lane=biopsy").json()
    assert roles["hero_surgical"] not in ids(biopsy) and len(biopsy["items"]) == len(ai["items"]) - 1
    assert client.get("/api/worklist?lane=colon").status_code == 422
    assert client.get("/api/worklist?order=random").status_code == 422


def test_conformal_safety_rule_puts_invasive_in_set_in_tier_2(tmp_path):
    base = fixture_case("demo_0001")
    benign_top = variant(base, "safety_case", probs=[0.10, 0.55, 0.05, 0.30], conformal_set=["Benign", "Invasive"],
                         offset=10)
    empty_invasive = variant(base, "empty_inv", probs=[0.05, 0.05, 0.10, 0.80], conformal_set=[], offset=20)
    plain = variant(base, "plain_benign", probs=[0.10, 0.85, 0.04, 0.01], conformal_set=["Benign"], offset=5,
                    agent={"status": "none", "orders": []})
    client, clock = make_client(tmp_path, cases_dir=write_cases(tmp_path / "c", [benign_top, empty_invasive, plain]))
    wl = arrive_all_biopsies(client, clock)
    assert row_for(wl, "safety_case")["runtime"]["tier"] == 2
    assert row_for(wl, "safety_case")["runtime"]["reasons"] == ["aggressive_in_conformal_set", "set_size_gt_1"]
    assert row_for(wl, "empty_inv")["runtime"]["tier"] == 2
    assert "empty_set" in row_for(wl, "empty_inv")["runtime"]["reasons"]
    assert row_for(wl, "plain_benign")["runtime"]["tier"] == 4
    assert ids(wl) == ["empty_inv", "safety_case", "plain_benign"]


def test_aging_guard_promotes_one_tier_on_the_sim_clock(tmp_path):
    base = fixture_case("demo_0001")
    routine = variant(base, "routine", probs=[0.10, 0.85, 0.04, 0.01], conformal_set=["Benign"], offset=0,
                      agent={"status": "none", "orders": []})
    unsure = variant(base, "unsure", probs=[0.10, 0.50, 0.39, 0.01], conformal_set=["Benign", "InSitu"], offset=0)
    no_settle = {**CFG, "sim": {**CFG["sim"], "after_arrivals_speed": None}}   # aging runs on the replay clock
    client, clock = make_client(tmp_path, cases_dir=write_cases(tmp_path / "c", [routine, unsure]), cfg=no_settle)
    client.post("/api/sim/start")
    clock.advance(1)
    wl = client.get("/api/worklist").json()
    assert {i["case_id"]: i["runtime"]["tier"] for i in wl["items"]} == {"routine": 4, "unsure": 3}
    aging = CFG["aging"]
    t3 = aging["promote_at_fraction"] * aging["turnaround_target_min"][3] * 60
    clock.advance(t3 / SPEED)            # sim clock now just past the tier-3 threshold
    wl = client.get("/api/worklist").json()
    assert row_for(wl, "unsure")["runtime"]["tier"] == 2
    assert row_for(wl, "unsure")["runtime"]["aging_promoted"] is True
    assert "aging_promoted" in row_for(wl, "unsure")["runtime"]["reasons"]
    t4 = aging["promote_at_fraction"] * aging["turnaround_target_min"][4] * 60
    clock.advance(t4 / SPEED)
    wl = client.get("/api/worklist").json()
    assert row_for(wl, "routine")["runtime"]["tier"] == 3            # one tier only
    assert row_for(wl, "unsure")["runtime"]["tier"] == 2             # never above tier 2
    timer = row_for(wl, "routine")["runtime"]["timer"]
    # the progress target stays the case's own tier target (no jump on promotion)
    assert timer["kind"] == "biopsy" and timer["clock"] == "sim" and timer["target_s"] == \
        aging["turnaround_target_min"][4] * 60


def test_done_cases_sort_after_open_ones(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    first = client.post("/api/sim/inject_surgical", json={"variant": "normal"}).json()["injected"]
    decide(client, first, "margin_call", {"call": "clear"})
    second = client.post("/api/sim/inject_surgical", json={"variant": "degraded"}).json()["injected"]
    wl = client.get("/api/worklist").json()
    assert ids(wl)[0] == second
    assert ids(wl)[-1] == first and row_for(wl, first)["runtime"]["done"] is True


# ---- case detail ------------------------------------------------------------------------------------------------------------
def test_case_detail_is_public_with_runtime(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    body = client.get(f"/api/cases/{roles['hero_biopsy']}").json()
    runtime = body.pop("runtime")
    body.pop("server_time"), body.pop("server_time_iso")
    contract.CasePublic.model_validate(body)
    assert "eval_only" not in body
    assert runtime["allowed_actions"] == ["approve_orders", "edit_orders", "closer_look", "dismiss_flag"]
    assert [o["rule_id"] for o in runtime["orders"]] == ["breast_insitu_vs_invasive", "breast_invasive"]
    assert client.get("/api/cases/NOPE_0001").status_code == 404


# ---- decisions ---------------------------------------------------------------------------------------------------------------
def test_approve_then_sign_out_routes_to_oncologist(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    hero = roles["hero_biopsy"]
    assert decide(client, hero, "sign_out").status_code == 409            # act on drafted orders first
    res = decide(client, hero, "approve_orders")
    assert res.status_code == 200
    assert res.json()["result"] == {"case_id": hero, "action": "approve_orders", "from": "drafted", "to": "approved"}
    assert res.json()["case"]["agent"]["status"] == "approved"
    assert decide(client, hero, "approve_orders").status_code == 409
    assert decide(client, hero, "sign_out").status_code == 200
    inbox = client.get("/api/routed?to=oncologist").json()
    assert [r["case_id"] for r in inbox] == [hero]
    assert inbox[0]["kind"] == "sign_out" and inbox[0]["agent_status"] == "approved"
    assert [o["items"] for o in inbox[0]["orders"]] == [["p63", "calponin"], ["ER", "PR", "HER2", "Ki-67"]]
    assert inbox[0]["summary"]["sentences"]
    assert client.get("/api/routed?to=surgeon").json() == []
    for action in ("sign_out", "approve_orders", "dismiss_flag", "closer_look"):
        assert decide(client, hero, action).status_code == 409
    wl = client.get("/api/worklist").json()
    assert row_for(wl, hero)["runtime"]["done"] and row_for(wl, hero)["runtime"]["allowed_actions"] == []


def test_edit_orders_only_removes_drafted_items(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    hero = roles["hero_biopsy"]
    invented = {"orders": [{"seq": 1, "items": ["ER", "PD-L1"]}]}
    assert decide(client, hero, "edit_orders", invented).status_code == 422
    unknown_rule = {"orders": [{"seq": 1, "rule_id": "made_up", "items": ["ER"]}]}
    assert decide(client, hero, "edit_orders", unknown_rule).status_code == 422
    nothing = {"orders": [{"seq": 1, "items": []}]}
    assert decide(client, hero, "edit_orders", nothing).status_code == 422
    assert decide(client, hero, "edit_orders", {"orders": "ER"}).status_code == 422
    ok = {"orders": [{"seq": 1, "items": ["Ki-67", "ER"]}]}
    res = decide(client, hero, "edit_orders", ok)
    assert res.status_code == 200 and res.json()["result"]["to"] == "edited_and_approved"
    case = res.json()["case"]
    assert case["agent"]["orders"] == [{"type": "stain", "items": ["ER", "Ki-67"], "rule_id": "breast_invasive",
                                        "rationale": "Standard biomarker panel if invasion is confirmed."}]
    statuses = {o["rule_id"]: o["status"] for o in case["runtime"]["orders"]}
    assert statuses == {"breast_insitu_vs_invasive": "removed", "breast_invasive": "edited"}
    assert decide(client, hero, "sign_out").status_code == 200
    routed = client.get("/api/routed?to=oncologist").json()[0]
    assert routed["agent_status"] == "edited_and_approved" and routed["orders"][0]["items"] == ["ER", "Ki-67"]


def test_closer_look_then_dismiss(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    case_id = roles["benign_vs_insitu"]
    assert decide(client, case_id, "closer_look").json()["result"]["to"] == "deferred_closer_look"
    assert decide(client, case_id, "closer_look").status_code == 409
    assert decide(client, case_id, "sign_out").status_code == 409
    assert decide(client, case_id, "approve_orders", {"extra": 1}).status_code == 422
    res = decide(client, case_id, "dismiss_flag")
    assert res.json()["result"]["to"] == "dismissed"
    assert all(o["status"] == "dismissed" for o in res.json()["case"]["runtime"]["orders"])
    assert decide(client, case_id, "sign_out").status_code == 200
    assert client.get("/api/routed?to=oncologist").json()[0]["orders"] == []


def test_closer_look_then_approve(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    case_id = roles["low_reliability"]
    decide(client, case_id, "closer_look")
    assert decide(client, case_id, "approve_orders").json()["result"]["to"] == "approved"


def test_no_orders_case_signs_out_directly(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    case_id = roles["routine_normal"]
    for action in ("approve_orders", "edit_orders", "closer_look", "dismiss_flag"):
        assert decide(client, case_id, action).status_code == 409
    assert decide(client, case_id, "sign_out").status_code == 200


def test_unknown_action_and_wrong_lane(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    surgical = client.post("/api/sim/inject_surgical", json={"variant": "normal"}).json()["injected"]
    hero = roles["hero_biopsy"]
    assert decide(client, hero, "order_everything").status_code == 422
    assert decide(client, hero, "margin_call", {"call": "clear"}).status_code == 409
    for action in ("approve_orders", "edit_orders", "closer_look", "dismiss_flag", "sign_out"):
        res = decide(client, surgical, action)
        assert res.status_code == 409, action
    assert "never placed" in decide(client, surgical, "approve_orders").json()["detail"]
    assert client.post(f"/api/cases/{hero}/decision", json={"action": "sign_out", "extra": 1}).status_code == 422


# ---- margin calls and timers ---------------------------------------------------------------------------------------------------
def test_margin_call_routes_to_surgeon_and_stops_timer(dev):
    client, clock, roles, _ = dev
    client.post("/api/sim/start")
    surgical = client.post("/api/sim/inject_surgical", json={"variant": "normal"}).json()["injected"]
    assert surgical == roles["hero_surgical"]
    timer = lambda: row_for(client.get("/api/worklist").json(), surgical)["runtime"]["timer"]  # noqa: E731
    assert timer()["state"] == "ok" and timer()["clock"] == "wall"
    clock.advance(AMBER_S - 1)
    assert timer()["state"] == "ok"
    clock.advance(1)
    assert timer()["state"] == "amber"
    clock.advance(RED_S - AMBER_S)
    assert timer()["state"] == "red" and timer()["elapsed_s"] == pytest.approx(RED_S)
    assert decide(client, surgical, "margin_call", {"call": "tumor_at_margin"}).status_code == 422
    assert decide(client, surgical, "margin_call", {"call": "clear", "margins": ["superior"]}).status_code == 422
    assert decide(client, surgical, "margin_call", {"call": "tumor_at_margin",
                                                    "margins": ["anterior"]}).status_code == 422
    assert decide(client, surgical, "margin_call", {"call": "maybe"}).status_code == 422
    res = decide(client, surgical, "margin_call", {"call": "tumor_at_margin", "margins": ["superior"]})
    assert res.status_code == 200
    clock.advance(300)
    stopped = timer()
    assert stopped["state"] == "stopped" and stopped["elapsed_s"] == pytest.approx(RED_S)
    board = client.get("/api/routed?to=surgeon").json()
    assert board[0]["case_id"] == surgical and board[0]["call"] == "tumor_at_margin"
    assert board[0]["margins"] == ["superior"] and board[0]["timer_elapsed_s"] == pytest.approx(RED_S)
    assert board[0]["or_room"].startswith("OR-") and board[0]["simulated"] is True
    assert decide(client, surgical, "margin_call", {"call": "defer"}).status_code == 409
    detail = client.get(f"/api/cases/{surgical}").json()
    assert detail["runtime"]["decision"]["margin_call"]["margins"] == ["superior"]
    types = [e["type"] for e in client.get(f"/api/cases/{surgical}/events").json()]
    assert types == ["injected", "timer_started", "routed", "margin_call", "timer_stopped"]


def test_events_are_timestamped(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    hero = roles["hero_biopsy"]
    decide(client, hero, "closer_look")
    clock.advance(5)
    decide(client, hero, "approve_orders")
    events = client.get(f"/api/cases/{hero}/events").json()
    transitions = [(e["from"], e["to"], e["action"]) for e in events if e["type"] == "agent_transition"]
    assert transitions == [("drafted", "deferred_closer_look", "closer_look"),
                           ("deferred_closer_look", "approved", "approve_orders")]
    assert all(e["at"].endswith("Z") and e["sim_time_s"] >= 0 for e in events)
    assert client.get("/api/cases/NOPE/events").status_code == 404


# ---- simulator controls --------------------------------------------------------------------------------------------------------
def test_pause_start_reset(dev):
    client, clock, roles, _ = dev
    client.post("/api/sim/start")
    clock.advance(2)
    client.post("/api/sim/pause")
    before = client.get("/api/health").json()["sim"]
    clock.advance(1000)
    after = client.get("/api/health").json()["sim"]
    assert after["sim_time_s"] == before["sim_time_s"] and after["running"] is False
    client.post("/api/sim/start")
    client.post("/api/sim/start")                      # idempotent
    clock.advance(1)
    assert client.get("/api/health").json()["sim"]["sim_time_s"] == pytest.approx(before["sim_time_s"] + SPEED)
    wl = arrive_all_biopsies(client, clock)
    hero = roles["hero_biopsy"]
    decide(client, hero, "approve_orders")
    decide(client, hero, "sign_out")
    client.post("/api/sim/inject_surgical", json={"variant": "normal"})
    res = client.post("/api/sim/reset").json()
    assert res["sim"]["sim_time_s"] == 0 and res["sim"]["running"] is False and res["sim"]["arrived"] == 0
    assert client.get("/api/worklist").json()["items"] == []
    assert client.get("/api/routed?to=oncologist").json() == []
    assert client.get("/api/routed?to=surgeon").json() == []
    wl = arrive_all_biopsies(client, clock)
    assert row_for(wl, hero)["agent"]["status"] == "drafted"                       # drafted orders restored
    assert [e["type"] for e in client.get(f"/api/cases/{hero}/events").json()] == ["arrived"]
    assert client.post("/api/sim/rewind").status_code == 422


def test_reset_rereads_case_files(tmp_path):
    folder = write_cases(tmp_path / "c", [fixture_case("demo_0001")])
    client, clock = make_client(tmp_path, cases_dir=folder)
    assert client.get("/api/health").json()["loaded"] == 1
    write_cases(folder, [variant(fixture_case("demo_0001"), "second_case", offset=1)])
    client.post("/api/sim/reset")
    assert client.get("/api/health").json()["loaded"] == 2


def test_inject_variants_prefer_demo_heroes(tmp_path):
    hero = fixture_case("demo_0002")
    segments_clear = [{**s, "p": 0.03} for s in hero["margin"]["segments"]]
    clear_margin = {**hero["margin"], "suggested_call": "clear", "involved": [], "segments": segments_clear,
                    "per_margin": {m: 0.03 for m in contract.MARGIN_NAMES}, "involved_length_um": None}
    clear = variant(hero, "early_clear", offset=1, margin=clear_margin, findings=[
        f for f in hero["findings"] if f["kind"] != "margin"], summary={
        "generator": "template", "sentences": hero["summary"]["sentences"][:3]})
    degraded = variant(hero, "deg_case", offset=2, reliability_flag=True, simulated_variant="degraded",
                       degradation={"type": "blur", "severity": 3}, source_case_ids=["demo_0002"],
                       margin={**hero["margin"], "suggested_call": "defer", "involved": [],
                               "involved_length_um": None})
    client, clock = make_client(tmp_path, cases_dir=write_cases(tmp_path / "c", [hero, clear, degraded]))
    first = client.post("/api/sim/inject_surgical", json={"variant": "normal"}).json()
    assert first["injected"] == "demo_0002"                 # demo hero beats the earlier clear case
    assert client.post("/api/sim/inject_surgical").json()["injected"] == "early_clear"   # default variant normal
    res = client.post("/api/sim/inject_surgical", json={"variant": "normal"})
    assert res.status_code == 409 and "no pending" in res.json()["detail"]
    assert client.post("/api/sim/inject_surgical", json={"variant": "degraded"}).json()["injected"] == "deg_case"
    assert client.post("/api/sim/inject_surgical", json={"variant": "composite"}).status_code == 409
    assert client.post("/api/sim/inject_surgical", json={"variant": "weird"}).status_code == 422
    wl = client.get("/api/worklist").json()
    assert all(i["runtime"]["tier"] == 1 and i["runtime"]["injected"] for i in wl["items"])
    assert row_for(wl, "deg_case")["simulated_variant"] == "degraded"


def test_dev_cases_inject_each_variant(dev):
    client, _, roles, _ = dev
    for variant_name, role in (("normal", "hero_surgical"), ("degraded", "surgical_degraded"),
                               ("composite", "surgical_composite"), ("normal", "surgical_clear")):
        injected = client.post("/api/sim/inject_surgical", json={"variant": variant_name}).json()["injected"]
        assert injected == roles[role]
    detail = client.get(f"/api/cases/{roles['surgical_degraded']}").json()
    assert detail["quality"]["reliability_flag"] is True and detail["margin"]["suggested_call"] == "defer"


# ---- media, frontend, metrics, spot-check ---------------------------------------------------------------------------------------
def test_fixture_media_is_served_flat(tmp_path):
    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    assert client.get("/api/health").json()["media"] == "fixtures"
    img = client.get("/images/demo_0001.jpg")
    assert img.status_code == 200 and img.content == (FIXTURES / "demo_0001.jpg").read_bytes()
    assert client.get("/cells/demo_0001_5_overlay.png").status_code == 200
    assert client.get("/images/missing.jpg").status_code == 404
    assert client.get("/images/..%2Fpyproject.toml").status_code == 404
    fixture_json = client.get("/fixtures/demo_0001.json")        # ?mock=1 reads it: public form only
    assert fixture_json.status_code == 200 and find_keys(fixture_json.json()) == []
    contract.CasePublic.model_validate(fixture_json.json())
    for url in ("/images/demo_0001.json", "/cells/demo_0002.json", "/fixtures/pyproject.toml"):
        assert client.get(url).status_code == 404, url


def test_image_thumbnails_are_downscaled_jpegs(tmp_path):
    from io import BytesIO

    from PIL import Image

    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    res = client.get("/images/demo_0001.jpg?w=128")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    with Image.open(BytesIO(res.content)) as im:
        assert im.width == 128
    assert client.get("/images/demo_0001.jpg?w=128").content == res.content          # cached, deterministic
    assert client.get("/images/demo_0001.jpg?w=5000").status_code == 422              # bounded size
    assert client.get("/images/missing.jpg?w=128").status_code == 404
    assert client.get("/images/demo_0001.jpg").content == (FIXTURES / "demo_0001.jpg").read_bytes()


def test_thumbnail_cache_is_bounded_by_bytes(tmp_path, monkeypatch):
    from backend import app as app_module

    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    one = len(client.get("/images/demo_0001.jpg?w=300").content)
    monkeypatch.setattr(app_module, "THUMB_CACHE_BYTES", 3 * one)
    monkeypatch.setattr(app_module, "_THUMBS", {})
    for w in range(300, 340):                           # any width in range is a new entry
        assert client.get(f"/images/demo_0001.jpg?w={w}").status_code == 200
        assert sum(map(len, app_module._THUMBS.values())) <= 3 * one


def test_static_routes_never_serve_json(tmp_path):
    root = dev_root(tmp_path)
    io.write_json(root / "eval" / "metrics.json", {"name": "metrics", "true_label": ["Invasive"]})
    client, _ = make_client(tmp_path, artifacts_root=root)
    for url in ("/eval/metrics.json", "/spotcheck/items.json", "/images/x.json", "/cells/y.json"):
        assert client.get(url).status_code == 404, url
    assert client.get("/fixtures/demo_0001.json").status_code == 200       # public form, artifacts mode too
    assert "true_label" not in client.get("/fixtures/demo_0001.json").text


def test_fixture_json_suffix_case_never_serves_the_raw_case(tmp_path):
    # case-insensitive filesystems (macOS/APFS) open demo_0001.json for demo_0001.JSON: still the public form only
    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    for url in ("/fixtures/demo_0001.JSON", "/fixtures/demo_0001.Json"):
        res = client.get(url)
        assert res.status_code in (200, 404), url
        assert "eval_only" not in res.text and "true_label" not in res.text, url
        if res.status_code == 200:
            assert find_keys(res.json()) == []


def test_non_string_payload_values_are_422(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    hero = roles["hero_biopsy"]
    assert decide(client, hero, "edit_orders", {"orders": [{"seq": ["x"], "items": ["ER"]}]}).status_code == 422
    assert decide(client, hero, "edit_orders", {"orders": [{"seq": 1,
                                                            "items": [{"a": 1}]}]}).status_code == 422
    surgical = client.post("/api/sim/inject_surgical", json={"variant": "normal"}).json()["injected"]
    assert decide(client, surgical, "margin_call", {"call": "tumor_at_margin",
                                                    "margins": [{"m": 1}]}).status_code == 422
    assert decide(client, surgical, "margin_call", {"call": ["clear"]}).status_code == 422


def test_biopsy_timer_freezes_at_sign_out(dev):
    client, clock, roles, _ = dev
    arrive_all_biopsies(client, clock)
    case_id = roles["routine_normal"]
    client.post("/api/sim/start")
    decide(client, case_id, "sign_out")
    before = row_for(client.get("/api/worklist").json(), case_id)["runtime"]["timer"]
    clock.advance(100)
    after = row_for(client.get("/api/worklist").json(), case_id)["runtime"]["timer"]
    assert after["state"] == "stopped" and after["elapsed_s"] == before["elapsed_s"]


def test_health_is_degraded_without_valid_cases(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    client, _ = make_client(tmp_path, cases_dir=folder)
    assert client.get("/api/health").json()["status"] == "degraded"


def test_artifact_media_and_frontend(tmp_path):
    root = dev_root(tmp_path)
    client, _ = make_client(tmp_path, artifacts_root=root, frontend=True)
    assert client.get("/api/health").json()["media"] == "artifacts"
    assert client.get("/images/DEV_0001.jpg").status_code == 200
    assert client.get("/images/DEVREF_01.jpg").status_code == 200
    index = client.get("/")
    assert index.status_code == 200 and "<html" in index.text.lower()
    assert index.headers["cache-control"] == "no-store"
    js = client.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def test_metrics_serves_eval_files_without_ground_truth(tmp_path):
    root = dev_root(tmp_path)
    eval_dir = root / "eval"
    io.write_json(eval_dir / "calibration.json", {"name": "calibration", "synthetic": True, "ece": {"value": 0.04},
                                                  "per_case": [{"case_id": "x", "true_label": "Invasive", "p": 0.7}]})
    io.write_json(eval_dir / "queue_sim.json", {"name": "queue_sim", "synthetic": True, "eval_only": {"a": 1}})
    (eval_dir / "broken.json").write_text("{")
    (eval_dir / "calibration.png").write_bytes(b"\x89PNG fake")
    client, _ = make_client(tmp_path, artifacts_root=root)
    body = client.get("/api/metrics").json()
    assert set(body["eval"]) == {"calibration", "queue_sim"}
    assert body["eval"]["calibration"]["per_case"] == [{"case_id": "x", "p": 0.7}]
    assert find_keys(body) == []
    assert body["figures"] == [{"file": "calibration.png", "url": "/eval/calibration.png"}]
    assert body["errors"][0]["file"] == "broken.json"
    assert body["synthetic"] is True and "queue_sim" in body["expected"]
    assert client.get("/eval/calibration.png").content == b"\x89PNG fake"


def test_metrics_empty_when_no_eval_dir(tmp_path):
    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    body = client.get("/api/metrics").json()
    assert body["eval"] == {} and body["figures"] == []


def test_spotcheck_items_and_reviews(tmp_path):
    root = dev_root(tmp_path)
    client, _ = make_client(tmp_path, artifacts_root=root)
    items = client.get("/api/spotcheck/items").json()
    assert items["available"] is True and items["synthetic"] is True
    concept_item = next(i for i in items["items"] if i["kind"] == "concept")
    cell_item = next(i for i in items["items"] if i["kind"] == "cells")
    assert client.get("/" + concept_item["image"]).status_code == 200
    assert client.get("/" + cell_item["image"]).status_code == 200
    body = {"reviewer": "Tester", "reviewer_kind": "ai",
            "reviews": [{"item_id": concept_item["item_id"], "verdict": "acceptable"},
                        {"item_id": cell_item["item_id"], "verdict": "unsure", "note": "edge"}]}
    res = client.post("/api/spotcheck", json=body)
    assert res.status_code == 200 and res.json()["recorded"] == 2
    saved = io.read_json(schemas.eval_json(root, "spotcheck"))
    assert saved["reviewer_kind"] == "ai" and saved["synthetic"] is True and saved["name"] == "spotcheck"
    assert saved["summary"]["concept"][concept_item["target"]] == {"n": 1, "acceptable": 1}
    assert saved["summary"]["cells"] == {"n": 1, "acceptable": 0}
    human = {"reviewer": "Pat", "reviewer_kind": "human",
             "reviews": [{"item_id": concept_item["item_id"], "verdict": "not_acceptable"}]}
    client.post("/api/spotcheck", json=human)
    saved = io.read_json(schemas.eval_json(root, "spotcheck"))
    assert saved["reviewer_kind"] == "mixed" and len(saved["reviews"]) == 3
    client.post("/api/spotcheck", json={**human, "reviews": [{"item_id": concept_item["item_id"],
                                                               "verdict": "acceptable"}]})
    assert len(io.read_json(schemas.eval_json(root, "spotcheck"))["reviews"]) == 3   # same reviewer: replaced
    assert client.post("/api/spotcheck", json={**body, "reviews": [{"item_id": "nope", "verdict": "acceptable"}]}
                       ).status_code == 422
    assert client.post("/api/spotcheck", json={**body, "reviewer_kind": "robot"}).status_code == 422
    assert "spotcheck" in client.get("/api/metrics").json()["eval"]


def test_spotcheck_without_items(tmp_path):
    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    assert client.get("/api/spotcheck/items").json() == {"available": False, "items": [], "synthetic": True}
    res = client.post("/api/spotcheck", json={"reviewer": "A", "reviewer_kind": "human",
                                              "reviews": [{"item_id": "x", "verdict": "acceptable"}]})
    assert res.status_code == 409


# ---- no ground truth anywhere ------------------------------------------------------------------------------------------------
def test_no_response_contains_ground_truth(dev):
    client, clock, roles, root = dev
    seen = []

    def get(url):
        res = client.get(url)
        assert res.status_code == 200, url
        seen.append((url, res))
        return res.json()

    get("/api/health"), get("/api/config"), get("/api/metrics"), get("/api/spotcheck/items")
    arrive_all_biopsies(client, clock)
    for variant_name in ("normal", "degraded", "composite", "normal"):
        client.post("/api/sim/inject_surgical", json={"variant": variant_name})
    decide(client, roles["hero_biopsy"], "approve_orders")
    seen.append(("decision", decide(client, roles["hero_biopsy"], "sign_out")))
    seen.append(("margin", decide(client, roles["hero_surgical"], "margin_call",
                                  {"call": "tumor_at_margin", "margins": ["superior"]})))
    for order in ("ai", "arrival"):
        for lane in ("surgical", "biopsy", "all"):
            wl = get(f"/api/worklist?lane={lane}&order={order}")
    for item in wl["items"]:
        get(f"/api/cases/{item['case_id']}")
        get(f"/api/cases/{item['case_id']}/events")
    get("/api/routed?to=surgeon"), get("/api/routed?to=oncologist")
    assert len(wl["items"]) == len(list((root / "cases").glob("*.json")))
    for url, res in seen:
        assert find_keys(res.json()) == [], url
        assert "eval_only" not in res.text and "true_label" not in res.text, url
    # the files on disk do carry it (so the test is meaningful)
    assert "true_label" in (root / "cases" / f"{roles['hero_biopsy']}.json").read_text()
    assert json.loads((root / "cases" / f"{roles['hero_biopsy']}.json").read_text())["eval_only"]


def test_lazy_app_reports_startup_failure_to_uvicorn(monkeypatch, tmp_path):
    import asyncio

    from backend import app as app_module

    def boom():
        raise FileNotFoundError("cases folder missing")

    monkeypatch.setattr(app_module, "create_app", boom)
    lazy = app_module._LazyApp()
    sent = []

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        sent.append(message)

    asyncio.run(lazy({"type": "lifespan"}, receive, send))
    assert sent == [{"type": "lifespan.startup.failed", "message": "FileNotFoundError: cases folder missing"}]
    with pytest.raises(FileNotFoundError):
        asyncio.run(lazy({"type": "http"}, receive, send))


def test_surgical_recut_drafts_can_be_approved_alongside_the_margin_call(tmp_path):
    recut = {"status": "drafted", "orders": [{"type": "recut", "items": ["recut H&E section"], "rule_id": "low_quality",
                                              "rationale": "Reliability flag."}],
             "blocked": [{"action": "draft_stains", "reason": "Stains are not drafted for simulated frozen sections."}]}
    case = variant(fixture_case("demo_0002"), "surg_recut", agent=recut, reliability_flag=True,
                   margin={**fixture_case("demo_0002")["margin"], "suggested_call": "defer", "involved": [],
                           "involved_length_um": None})
    client, _ = make_client(tmp_path, cases_dir=write_cases(tmp_path / "c", [case]))
    client.post("/api/sim/inject_surgical")
    detail = client.get("/api/cases/surg_recut").json()
    assert detail["runtime"]["allowed_actions"][0] == "margin_call" and "approve_orders" in detail["runtime"]["allowed_actions"]
    assert decide(client, "surg_recut", "margin_call", {"call": "defer"}).status_code == 200
    assert decide(client, "surg_recut", "approve_orders").json()["result"]["to"] == "approved"
    assert decide(client, "surg_recut", "sign_out").status_code == 409
    wl = client.get("/api/worklist").json()
    assert row_for(wl, "surg_recut")["runtime"]["done"] is True


def test_config_carries_cell_feature_names_from_the_features_file(tmp_path):
    import yaml
    client, _ = make_client(tmp_path, cases_dir=FIXTURES)
    body = client.get("/api/config").json()
    spec = yaml.safe_load((config_root() / CFG["cells"]["features_file"]).read_text())
    assert body["labels"]["cell_features"] == {f["id"]: f["display_name"] for f in spec["features"]}
    assert set(body["labels"]["cell_types"]) == set(CFG["cells"]["types"])
    assert body["cell_features"]["neo_density"]["unit"] == "per mm²"


def config_root():
    from core import config
    return config.REPO_ROOT


def test_health_reports_missing_referenced_media(dev):
    client, _, _, root = dev
    health = client.get('/api/health').json()
    assert health['assets']['missing'] == []
    assert health['assets']['available'] == health['assets']['referenced']
    assert health['assets']['models']
    image = next((root / 'images').glob('*.jpg'))
    image.unlink()
    health = client.get('/api/health').json()
    assert health['status'] == 'degraded'
    assert f'images/{image.name}' in health['assets']['missing']
    assert health['assets']['available'] == health['assets']['referenced'] - 1


def test_restart_preserves_decisions_routes_and_replay_clock(dev, tmp_path):
    client, clock, roles, root = dev
    arrive_all_biopsies(client, clock)
    hero = roles['hero_biopsy']
    assert client.post(f'/api/cases/{hero}/decision', json={'action': 'approve_orders'}).status_code == 200
    assert client.post(f'/api/cases/{hero}/decision', json={'action': 'sign_out'}).status_code == 200
    before = client.get('/api/health').json()['sim']
    routed = client.get('/api/routed?to=oncologist').json()
    restarted, _ = make_client(tmp_path, cases_dir=root / 'cases', artifacts_root=root, clock=clock)
    after = restarted.get('/api/health').json()['sim']
    assert after['sim_time_s'] == pytest.approx(before['sim_time_s'])
    assert after['running'] is False
    assert restarted.get('/api/routed?to=oncologist').json() == routed
    case = restarted.get(f'/api/cases/{hero}').json()
    assert case['runtime']['done'] is True
    assert case['agent']['status'] == 'approved'


def test_changed_artifacts_cannot_silently_erase_saved_workspace(dev, tmp_path):
    from backend.db import ServerDataError
    client, clock, roles, root = dev
    arrive_all_biopsies(client, clock)
    before = client.get('/api/worklist').json()['items']
    path = root / 'cases' / f"{roles['hero_biopsy']}.json"
    payload = json.loads(path.read_text())
    payload['prediction']['model'] = 'a_new_model'
    path.write_text(json.dumps(payload))
    with pytest.raises(ServerDataError, match='case artifacts differ'):
        make_client(tmp_path, cases_dir=root / 'cases', artifacts_root=root, clock=clock)
    assert client.get('/api/worklist').json()['items'] == before


def test_shutdown_checkpoints_the_running_replay_clock(tmp_path):
    root = dev_root(tmp_path)
    client, clock = make_client(tmp_path, artifacts_root=root)
    with client:                                         # lifespan: the shutdown hook runs on exit
        client.post("/api/sim/start")
        clock.advance(3)                                 # no request after this: only the shutdown hook sees it
    from backend import db as store_db
    restored = store_db.Store(tmp_path / "test.sqlite").restore(store_db.load_case_dir(root / "cases"))
    assert restored > 0 and restored == pytest.approx(client.app.state.backend.sim.sim_now())   # not the Start time
    restarted, _ = make_client(tmp_path, artifacts_root=root, clock=clock)
    sim = restarted.get("/api/health").json()["sim"]
    assert sim["running"] is False and sim["sim_time_s"] == pytest.approx(restored)


def test_health_sim_time_never_runs_ahead_of_the_next_read(tmp_path):
    # health ticks like every other read: it must not report the unticked boosted-speed time that the
    # after-arrivals slowdown then moves back from
    root = dev_root(tmp_path)
    client, clock = make_client(tmp_path, artifacts_root=root)
    client.post("/api/sim/start")
    clock.advance(CFG["sim"]["max_replay_wall_s"] + 30)      # every scheduled biopsy came due; nobody polled
    health = client.get("/api/health").json()["sim"]
    assert health["pending"]["biopsy"] == 0
    later = client.get("/api/worklist").json()["sim"]
    assert later["sim_time_s"] >= health["sim_time_s"] - 1e-9
