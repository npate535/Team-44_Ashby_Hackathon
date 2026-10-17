"""agent/agent.py: default zero LLM calls; tool calling -> JSON prompt -> rules; LLM may only choose rule outputs."""

from __future__ import annotations

import json

import httpx
import pytest

from agent import agent as A
from agent import lumen
from agent import rules as R
from core import config, contract

CFG = config.load_settings()
RULES = R.load_rules(CFG)


def case(name="demo_0001"):
    return json.loads((config.fixtures_root(CFG) / f"{name}.json").read_text(encoding="utf-8"))


def tool_call(name, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Server:
    """Scripted chat replies: dict(content=..., tool_calls=[...]) or an HTTP status int."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "ornith-1.0-35b"}]})
        self.bodies.append(json.loads(request.content))
        item = self.replies.pop(0)
        if isinstance(item, int):
            return httpx.Response(item, text="error")
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", **item}}], "usage": {}})


def client_for(server, tmp_path, max_calls=10, key="k"):
    return lumen.LumenClient(base_url="https://lumen.example/v1", api_key=key, model_preference=["ornith-1.0-35b"],
                             cache_dir=tmp_path / "cache", max_calls=max_calls, retries=0, timeout_s=5,
                             temperature=0.2, max_tokens=100, transport=httpx.MockTransport(server),
                             sleep=lambda s: None)


GOOD_P63 = {"option_id": "breast_insitu_vs_invasive:0", "items": ["p63", "calponin"],
            "rationale": "Myoepithelial markers can separate in situ from invasive growth in regions 5 and 6."}
GOOD_ER = {"option_id": "breast_invasive:0", "items": ["ER", "PR", "HER2", "Ki-67"],
           "rationale": "Biomarker panel to draft if invasion is accepted by the pathologist."}


def test_default_is_rules_with_zero_llm_calls():
    agent, info = A.draft(case(), CFG, rules=RULES)
    assert info == {"path": "rules", "llm_calls": 0, "cached": False, "fallbacks": []}
    assert agent == R.draft_orders(case(), CFG, RULES)
    contract.Agent.model_validate(agent)


def test_tool_calling_path_uses_llm_rationales(tmp_path):
    server = Server([
        {"content": None, "tool_calls": [tool_call("lookup_rules", {}, "a")]},
        {"content": None, "tool_calls": [tool_call("draft_order", GOOD_P63, "b"), tool_call("draft_order", GOOD_ER, "c")]},
    ])
    agent, info = A.draft(case(), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert info["path"] == "tools" and info["llm_calls"] == 2
    assert [o["rule_id"] for o in agent["orders"]] == ["breast_insitu_vs_invasive", "breast_invasive"]
    assert agent["orders"][0]["rationale"] == GOOD_P63["rationale"]
    contract.Agent.model_validate(agent)
    # the tool result for lookup_rules carried only rule options
    tool_msg = [m for m in server.bodies[1]["messages"] if m["role"] == "tool"][0]
    assert {o["option_id"] for o in json.loads(tool_msg["content"])["options"]} == {
        "breast_insitu_vs_invasive:0", "breast_invasive:0"}


def test_llm_may_choose_a_subset_of_items_in_rule_order(tmp_path):
    choice = {**GOOD_ER, "items": ["HER2", "ER"]}
    server = Server([{"content": None, "tool_calls": [tool_call("draft_order", choice)]}, {"content": "done"}])
    agent, info = A.draft(case(), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert agent["orders"] == [{"type": "stain", "items": ["ER", "HER2"], "rule_id": "breast_invasive",
                                "rationale": GOOD_ER["rationale"]}]
    assert info["dropped_options"] == ["breast_insitu_vs_invasive:0"] and info["llm_calls"] == 2


def test_one_draft_order_per_turn_keeps_every_order(tmp_path):
    server = Server([{"content": None, "tool_calls": [tool_call("draft_order", GOOD_P63, "a")]},
                     {"content": None, "tool_calls": [tool_call("draft_order", GOOD_ER, "b")]}])
    agent, info = A.draft(case(), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert [o["rule_id"] for o in agent["orders"]] == ["breast_insitu_vs_invasive", "breast_invasive"]
    assert info["path"] == "tools" and info["llm_calls"] == 2 and info["dropped_options"] == []


def test_running_out_of_turns_is_a_fallback(tmp_path):
    lookups = [{"content": None, "tool_calls": [tool_call("lookup_rules", {}, f"l{i}")]}
               for i in range(CFG["llm"]["agent_max_turns"] - 1)]
    server = Server(lookups + [{"content": None, "tool_calls": [tool_call("draft_order", GOOD_P63)]}, 400])
    agent, info = A.draft(case(), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert info["path"] == "rules" and "did not finish" in info["fallbacks"][0]


@pytest.mark.parametrize("bad,why", [
    ({**GOOD_ER, "items": ["ER", "BRCA1"]}, "not rule outputs"),
    ({**GOOD_ER, "option_id": "made_up:0"}, "unknown option_id"),
    ({**GOOD_ER, "rationale": "This confirms the diagnosis."}, "banned"),
    ({**GOOD_ER, "rationale": "Probability 0.99 says so."}, "not in the findings"),
    ({**GOOD_ER, "rationale": ""}, "rationale"),
    ({**GOOD_ER, "items": []}, "items"),
])
def test_violations_fall_back_to_json_then_rules(tmp_path, bad, why):
    server = Server([
        {"content": None, "tool_calls": [tool_call("draft_order", bad)]},
        {"content": "done"},
        {"content": json.dumps({"orders": [bad]})},
    ])
    agent, info = A.draft(case(), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert info["path"] == "rules" and len(info["fallbacks"]) == 2 and why in info["fallbacks"][0]
    assert agent == R.draft_orders(case(), CFG, RULES)


def test_tool_failure_falls_back_to_json_prompt(tmp_path):
    server = Server([400, {"content": json.dumps({"orders": [GOOD_P63, GOOD_ER]})}])
    agent, info = A.draft(case(), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert info["path"] == "json" and info["llm_calls"] == 2 and "tools" in info["fallbacks"][0]
    assert agent["orders"][1]["rationale"] == GOOD_ER["rationale"]


def test_dropping_every_order_is_rejected(tmp_path):
    server = Server([{"content": "done", "tool_calls": []}, {"content": json.dumps({"orders": []})}])
    agent, info = A.draft(case(), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert info["path"] == "rules" and "no draft_order" in info["fallbacks"][0]
    assert "none were chosen" in info["fallbacks"][1]
    assert agent["status"] == "drafted"


def test_missing_key_and_spent_budget_fall_back_without_calls(tmp_path):
    for client in (client_for(Server([]), tmp_path, key=None), client_for(Server([]), tmp_path / "b", max_calls=0)):
        client._model = "ornith-1.0-35b"
        agent, info = A.draft(case(), CFG, rules=RULES, client=client)
        assert info["path"] == "rules" and info["llm_calls"] == 0 and agent == R.draft_orders(case(), CFG, RULES)


def test_surgical_blocked_actions_never_reach_the_llm(tmp_path):
    server = Server([])
    agent, info = A.draft(case("demo_0002"), CFG, rules=RULES, client=client_for(server, tmp_path))
    assert server.bodies == [] and info["path"] == "rules"
    assert agent["blocked"] and agent["status"] == "none"


def test_json_mode_flag_is_sent(tmp_path):
    cfg = {**CFG, "llm": {**CFG["llm"], "tool_calling": False}}
    server = Server([{"content": json.dumps({"orders": [GOOD_P63]})}])
    agent, info = A.draft(case(), cfg, rules=RULES, client=client_for(server, tmp_path))
    assert info["path"] == "json" and server.bodies[0]["response_format"] == {"type": "json_object"}
    assert "tools" not in server.bodies[0]
    assert [o["rule_id"] for o in agent["orders"]] == ["breast_insitu_vs_invasive"]


def test_replay_hits_the_cache(tmp_path):
    replies = [{"content": None, "tool_calls": [tool_call("draft_order", GOOD_P63), tool_call("draft_order", GOOD_ER, "d")]}]
    first, _ = A.draft(case(), CFG, rules=RULES, client=client_for(Server(list(replies)), tmp_path))
    again, info = A.draft(case(), CFG, rules=RULES, client=client_for(Server([]), tmp_path, max_calls=0))
    assert again == first and info["cached"] and info["llm_calls"] == 0
