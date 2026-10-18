"""agent/lumen.py with every HTTP call mocked (no network): key lookup, model pick, cache, retries, budget."""

from __future__ import annotations

import json

import httpx
import pytest

from agent import lumen
from core import config

CFG = config.load_settings()
SECRET = "sk-test-SECRET-should-never-appear"


class FakeServer:
    """httpx.MockTransport handler: GET /models and POST /chat/completions with scripted responses."""

    def __init__(self, models=("gemma-4-31b-it", "ornith-1.0-35b"), replies=None):
        self.models = list(models)
        self.replies = list(replies or [])
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        item = self.replies.pop(0) if self.replies else {"content": "ok"}
        if isinstance(item, int):
            return httpx.Response(item, text="busy")
        if isinstance(item, Exception):
            raise item
        message = {"role": "assistant", "content": item.get("content"), "tool_calls": item.get("tool_calls")}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    @property
    def chat_requests(self):
        return [r for r in self.requests if r.url.path.endswith("/chat/completions")]


def make_client(server, tmp_path, max_calls=5, key=SECRET, **kw):
    sleeps = []
    client = lumen.LumenClient(base_url="https://lumen.example/v1", api_key=key,
                               model_preference=CFG["llm"]["model_preference"], cache_dir=tmp_path / "cache",
                               max_calls=max_calls, retries=CFG["llm"]["retries"], timeout_s=5, temperature=0.2,
                               max_tokens=100, transport=httpx.MockTransport(server), sleep=sleeps.append, **kw)
    return client, sleeps


def test_key_from_environment_wins_over_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "lumen.env"
    env_file.write_text('export LUMEN_API_KEY="from-file"\nLUMEN_BASE_URL=https://other/v1/\n', encoding="utf-8")
    cfg = {**CFG, "llm": {**CFG["llm"], "env_file": str(env_file)}}
    assert lumen.api_key(cfg, env={}) == "from-file"
    assert lumen.base_url(cfg, env={}) == "https://other/v1"
    assert lumen.api_key(cfg, env={"LUMEN_API_KEY": "from-env"}) == "from-env"
    missing = {**CFG, "llm": {**CFG["llm"], "env_file": str(tmp_path / "absent.env")}}
    assert lumen.api_key(missing, env={}) is None
    assert lumen.base_url(missing, env={}) == CFG["llm"]["base_url"]


def test_model_is_first_preference_present_in_live_listing(tmp_path):
    server = FakeServer(models=["ornith-1.0-35b", "gemma-4-31b-it", "glm-5.3"])
    client, _ = make_client(server, tmp_path)
    assert client.model == "gemma-4-31b-it"          # settings llm.model_preference order wins, not listing order
    server2 = FakeServer(models=["ornith-1.0-35b"])
    assert make_client(server2, tmp_path)[0].model == "ornith-1.0-35b"
    with pytest.raises(lumen.LumenError, match="model_preference"):
        make_client(FakeServer(models=["glm-5.3"]), tmp_path)[0].pick_model()


def test_key_is_sent_only_in_the_authorization_header(tmp_path):
    server = FakeServer(replies=[{"content": "hi"}])
    client, _ = make_client(server, tmp_path)
    client.chat([{"role": "user", "content": "x"}])
    request = server.chat_requests[0]
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert SECRET not in request.content.decode()
    for path in (tmp_path / "cache").rglob("*.json"):
        assert SECRET not in path.read_text(encoding="utf-8")
    assert SECRET not in json.dumps(client.log) and SECRET not in repr(client.__dict__.get("log"))


def test_cache_hit_costs_no_call_and_changed_prompt_misses(tmp_path):
    server = FakeServer(replies=[{"content": "first"}, {"content": "second"}])
    client, _ = make_client(server, tmp_path)
    a = client.chat([{"role": "user", "content": "same"}])
    b = client.chat([{"role": "user", "content": "same"}])
    assert (a["content"], a["cached"]) == ("first", False)
    assert (b["content"], b["cached"]) == ("first", True)
    assert client.calls == 1 and client.cache_hits == 1
    c = client.chat([{"role": "user", "content": "different"}])
    assert c["content"] == "second" and client.calls == 2
    # A fresh client reuses the disk cache.
    client2, _ = make_client(FakeServer(), tmp_path)
    assert client2.chat([{"role": "user", "content": "same"}])["cached"] is True and client2.calls == 0


def test_budget_is_hard_and_checked_before_sending(tmp_path):
    server = FakeServer(replies=[{"content": "a"}, {"content": "b"}])
    client, _ = make_client(server, tmp_path, max_calls=1)
    client.chat([{"role": "user", "content": "1"}])
    with pytest.raises(lumen.BudgetExceeded):
        client.chat([{"role": "user", "content": "2"}])
    assert len(server.chat_requests) == 1
    zero, _ = make_client(FakeServer(), tmp_path / "z", max_calls=0)
    with pytest.raises(lumen.BudgetExceeded):
        zero.chat([{"role": "user", "content": "x"}])


def test_retries_with_backoff_on_429_and_transport_errors(tmp_path):
    server = FakeServer(replies=[429, httpx.ConnectTimeout("slow"), {"content": "ok"}])
    client, sleeps = make_client(server, tmp_path)
    assert client.chat([{"role": "user", "content": "x"}])["content"] == "ok"
    assert sleeps == [1.0, 2.0] and client.calls == 3


def test_retries_give_up_and_count_against_budget(tmp_path):
    server = FakeServer(replies=[503] * 10)
    client, _ = make_client(server, tmp_path, max_calls=10)
    with pytest.raises(lumen.LumenError, match="attempts"):
        client.chat([{"role": "user", "content": "x"}])
    assert client.calls == CFG["llm"]["retries"] + 1


def test_client_error_is_not_retried_and_hides_the_key(tmp_path):
    server = FakeServer(replies=[400])
    client, sleeps = make_client(server, tmp_path)
    with pytest.raises(lumen.LumenError) as info:
        client.chat([{"role": "user", "content": "x"}])
    assert "400" in str(info.value) and SECRET not in str(info.value) and sleeps == []


def test_missing_key_raises_no_api_key_without_network(tmp_path):
    server = FakeServer()
    client, _ = make_client(server, tmp_path, key=None, model="ornith-1.0-35b")
    assert not client.has_key
    with pytest.raises(lumen.NoApiKey):
        client.chat([{"role": "user", "content": "x"}])
    assert server.requests == []


def test_request_body_carries_json_mode_and_tools(tmp_path):
    client, _ = make_client(FakeServer(), tmp_path, model="ornith-1.0-35b")
    body = client.request_body([{"role": "user", "content": "x"}], json_mode=True, tools=[lumen.PROBE_TOOL],
                               tool_choice="auto", max_tokens=50)
    assert body["response_format"] == {"type": "json_object"} and body["tools"][0]["function"]["name"] == "draft_order"
    assert body["max_tokens"] == 50 and body["model"] == "ornith-1.0-35b"


@pytest.mark.parametrize("text,expected", [
    ('[{"text": "a", "findings": ["F1"]}]', [{"text": "a", "findings": ["F1"]}]),
    ('```json\n{"ok": true}\n```', {"ok": True}),
    ('Here you go: {"ok": true} thanks', {"ok": True}),
])
def test_parse_json_content_tolerates_fences_and_prose(text, expected):
    assert lumen.parse_json_content(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", None])
def test_parse_json_content_rejects_non_json(text):
    with pytest.raises(ValueError):
        lumen.parse_json_content(text)


def test_check_features_reports_json_and_tool_support(tmp_path):
    tool_call = [{"id": "c1", "type": "function", "function": {"name": "draft_order", "arguments": '{"option_id": "a"}'}}]
    server = FakeServer(replies=[{"content": '{"ok": true}'}, {"content": None, "tool_calls": tool_call}])
    client, _ = make_client(server, tmp_path)
    out = lumen.check_features(client)
    assert out["json_mode"] is True and out["tool_calling"] is True and client.calls == 2


def test_probe_cost_is_uncached(tmp_path):
    server = FakeServer(replies=[{"content": "ok"}, {"content": "ok"}])
    client, _ = make_client(server, tmp_path)
    lumen.probe_cost(client)
    lumen.probe_cost(client)
    assert client.calls == 2 and not list((tmp_path / "cache").rglob("*.json"))


def test_auth_errors_and_echoed_tokens_never_reach_the_message(tmp_path):
    class Echo(FakeServer):
        def __call__(self, request):
            if request.url.path.endswith("/chat/completions"):
                status = self.replies.pop(0)
                return httpx.Response(status, text=f"bad key {SECRET} token abcdefghijklmnopqrstuvwxyz123456")
            return super().__call__(request)

    for status in (401, 400):
        client, _ = make_client(Echo(replies=[status]), tmp_path)
        with pytest.raises(lumen.LumenError) as info:
            client.chat([{"role": "user", "content": "x"}])
        assert SECRET not in str(info.value) and "abcdefghijklmnopqrstuvwxyz123456" not in str(info.value)


def test_truncated_or_empty_replies_are_not_cached(tmp_path):
    class Trunc(FakeServer):
        def __call__(self, request):
            if request.url.path.endswith("/chat/completions"):
                self.requests.append(request)
                return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
            return super().__call__(request)

    server = Trunc()
    client, _ = make_client(server, tmp_path)
    client.chat([{"role": "user", "content": "x"}])
    client.chat([{"role": "user", "content": "x"}])
    assert client.calls == 2 and client.cache_hits == 0 and not list((tmp_path / "cache").rglob("*.json"))


def test_cache_only_client_reuses_cached_replies_with_zero_network_calls(tmp_path):
    cache = tmp_path / "cache"
    live = lumen.LumenClient.from_settings(CFG, cache_dir=cache, max_calls=1, env={"LUMEN_API_KEY": SECRET},
                                           transport=httpx.MockTransport(FakeServer(replies=[{"content": "kept"}])))
    live.chat([{"role": "user", "content": "same"}])
    server = FakeServer()
    client = lumen.LumenClient.cache_only(CFG, cache_dir=cache, transport=httpx.MockTransport(server))
    assert not client.has_key and client.model == live.model
    hit = client.chat([{"role": "user", "content": "same"}])
    assert (hit["content"], hit["cached"]) == ("kept", True)
    with pytest.raises(lumen.NoApiKey):
        client.chat([{"role": "user", "content": "not cached"}])
    assert server.requests == [] and client.calls == 0 and client.cache_hits == 1
