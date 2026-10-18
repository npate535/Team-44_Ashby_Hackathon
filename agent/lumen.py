"""OpenAI-compatible client for NCSA Lumen with a disk cache, retries and a hard call budget (plan §2, BUILD_PLAN §A17).

    client = LumenClient.from_settings(cfg, cache_dir=data_root / cfg["llm"]["cache_dir"], max_calls=40)
    reply = client.chat([{"role": "user", "content": "..."}], json_mode=True)   # {"content", "tool_calls", ...}

Key handling: the key comes from $LUMEN_API_KEY (settings llm.api_key_env) or the private env file
(settings llm.env_file, default ~/.config/ashby-2026/lumen.env, mode 0600). It is sent only in the Authorization
header and is never printed, logged, cached or put in an exception message. Base URL: $LUMEN_BASE_URL
(llm.base_url_env) if set, else llm.base_url.

Model: the first of settings llm.model_preference present in the live GET /models listing (listing is free).

Cache: <cache_dir>/<model>/<sha256>.json keyed by (model, the exact request: messages, tools, tool_choice,
json_mode, max_tokens, temperature). A cache hit costs no call. Truncated (finish_reason "length") or empty replies are never cached, so a rerun can recover. Because prompts embed the case findings, any change to
the findings is a cache miss: with the budget spent, callers fall back to templates instead of stale text.

Reasoning models (ornith-1.0-35b) spend part of max_tokens on hidden reasoning (~150 tokens even for a trivial
reply, measured 2026-10-17), so settings llm.max_output_tokens leaves room for it.

Budget: every request sent to /chat/completions (retries included) counts toward `max_calls`; at the cap chat()
raises BudgetExceeded before sending anything. Retries: settings llm.retries on 429/5xx/timeouts/transport errors
with exponential backoff (1, 2, 4 s ...). Other HTTP errors raise LumenError at once.

CLI (run by the master; each prints models, usage and timing, never the key):
    python -m agent.lumen models                  # GET /models (no coins)
    python -m agent.lumen probe                   # 1 tiny call: compare the Lumen coin balance before/after
    python -m agent.lumen check                   # 2 tiny calls: JSON mode + tool calling support
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from core import config as core_config
from core import io

log = logging.getLogger(__name__)

RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)
_TOKENISH = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|[A-Za-z0-9_\-]{24,})")


class LumenError(RuntimeError):
    """A Lumen request failed (never carries the key)."""


class NoApiKey(LumenError):
    """No key in the environment or the env file."""


class BudgetExceeded(LumenError):
    """The hard --max-calls budget is spent."""


# ---- key and endpoint ----------------------------------------------------------------------------------------------
def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (optional `export`, quotes, # comments). Missing file => {}."""
    path = Path(os.path.expanduser(str(path)))
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def _lookup(cfg: Mapping[str, Any], env_key: str, env: Mapping[str, str] | None) -> str:
    source = os.environ if env is None else env
    value = (source.get(cfg["llm"][env_key]) or "").strip()
    if value:
        return value
    return (read_env_file(cfg["llm"]["env_file"]).get(cfg["llm"][env_key]) or "").strip()


def api_key(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> str | None:
    """The Lumen key from the environment, else the private env file; None when neither has it."""
    return _lookup(cfg, "api_key_env", env) or None


def base_url(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> str:
    return (_lookup(cfg, "base_url_env", env) or cfg["llm"]["base_url"]).rstrip("/")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]", "_", model)


# ---- client -----------------------------------------------------------------------------------------------------------
class LumenClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model_preference: Sequence[str],
        cache_dir: str | Path | None,
        max_calls: int,
        retries: int,
        timeout_s: float,
        temperature: float,
        max_tokens: int,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        self._key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_preference = list(model_preference)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_calls = int(max_calls)
        self.retries = int(retries)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._model = model
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        self.calls = 0          # requests sent to /chat/completions (retries included)
        self.cache_hits = 0
        self.log: list[dict[str, Any]] = []   # one entry per chat(): model, prompt hash, cached, usage (never the key)

    @classmethod
    def from_settings(
        cls,
        cfg: Mapping[str, Any],
        *,
        cache_dir: str | Path | None,
        max_calls: int,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "LumenClient":
        llm = cfg["llm"]
        return cls(base_url=base_url(cfg, env), api_key=api_key(cfg, env), model_preference=llm["model_preference"],
                   cache_dir=cache_dir, max_calls=max_calls, retries=llm["retries"], timeout_s=llm["timeout_s"],
                   temperature=llm["temperature"], max_tokens=llm["max_output_tokens"], **kwargs)

    @classmethod
    def cache_only(cls, cfg: Mapping[str, Any], *, cache_dir: str | Path, **kwargs: Any) -> "LumenClient":
        """A client that can only replay cached replies: no key, no budget, no /models lookup. Its model is the first
        llm.model_preference entry with a cache directory; a cache miss raises NoApiKey before any request."""
        llm = cfg["llm"]
        cached = [m for m in llm["model_preference"] if (Path(cache_dir) / _safe_name(m)).is_dir()]
        return cls(base_url=llm["base_url"], api_key=None, model_preference=llm["model_preference"],
                   cache_dir=cache_dir, max_calls=0, retries=llm["retries"], timeout_s=llm["timeout_s"],
                   temperature=llm["temperature"], max_tokens=llm["max_output_tokens"],
                   model=(cached or llm["model_preference"])[0], **kwargs)

    @property
    def has_key(self) -> bool:
        return bool(self._key)

    @property
    def budget_left(self) -> int:
        return max(0, self.max_calls - self.calls)

    def _headers(self) -> dict[str, str]:
        if not self._key:
            raise NoApiKey("no Lumen API key: set the environment variable or the private env file (see settings llm)")
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    # -- models ----------------------------------------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Model ids from GET /models (no coins)."""
        try:
            response = self._http.get(f"{self.base_url}/models", headers=self._headers())
        except httpx.HTTPError as exc:
            raise LumenError(f"GET /models failed: {type(exc).__name__}") from None
        if response.status_code != 200:
            raise LumenError(f"GET /models returned HTTP {response.status_code}")
        return [str(m["id"]) for m in response.json().get("data", []) if isinstance(m, Mapping) and "id" in m]

    def pick_model(self, available: Sequence[str] | None = None) -> str:
        available = self.list_models() if available is None else list(available)
        for model in self.model_preference:
            if model in available:
                return model
        raise LumenError(f"none of llm.model_preference {self.model_preference} is in /models")

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.pick_model()
        return self._model

    # -- chat --------------------------------------------------------------------------------------------------------
    def request_body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "max_tokens": int(max_tokens if max_tokens is not None else self.max_tokens),
            "temperature": float(temperature if temperature is not None else self.temperature),
        }
        if tools:
            body["tools"] = [dict(t) for t in tools]
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _cache_path(self, model: str, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / _safe_name(model) / f"{key}.json"

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """One chat completion (cached). Returns {"content": str|None, "tool_calls": [...], "usage": {...},
        "model", "prompt_hash", "cached": bool}. Raises BudgetExceeded, NoApiKey or LumenError."""
        body = self.request_body(messages, **kwargs)
        key = sha256(body)
        path = self._cache_path(body["model"], key)
        if path is not None and path.is_file():
            cached = io.read_json(path)
            self.cache_hits += 1
            reply = {**cached["reply"], "model": body["model"], "prompt_hash": key, "cached": True}
            self.log.append({"model": body["model"], "prompt_hash": key, "cached": True, "usage": reply.get("usage")})
            return reply
        headers = self._headers()
        data = self._post_with_retries(body, headers)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LumenError("malformed /chat/completions response (no choices[0].message)") from None
        reply = {"content": message.get("content"), "tool_calls": list(message.get("tool_calls") or []),
                 "usage": data.get("usage") or {}, "finish_reason": data["choices"][0].get("finish_reason")}
        cacheable = reply["finish_reason"] != "length" and (bool(reply["content"]) or bool(reply["tool_calls"]))
        if path is not None and cacheable:
            io.write_json(path, {"model": body["model"], "prompt_hash": key, "created_utc": io.utc_now(),
                                 "reply": reply})
        self.log.append({"model": body["model"], "prompt_hash": key, "cached": False, "usage": reply["usage"]})
        return {**reply, "model": body["model"], "prompt_hash": key, "cached": False}

    def _post_with_retries(self, body: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        last = "no attempt"
        for attempt in range(self.retries + 1):
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"LLM call budget of {self.max_calls} is spent")
            self.calls += 1
            try:
                response = self._http.post(f"{self.base_url}/chat/completions", headers=dict(headers), json=body)
            except httpx.HTTPError as exc:
                last = type(exc).__name__
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError:
                        raise LumenError("non-JSON /chat/completions response") from None
                if response.status_code in (401, 403):
                    raise LumenError(f"/chat/completions returned HTTP {response.status_code} (authentication)")
                if response.status_code not in RETRY_STATUS:
                    raise LumenError(f"/chat/completions returned HTTP {response.status_code}: "
                                     f"{self._redact(_short(response.text))}")
                last = f"HTTP {response.status_code}"
            if attempt < self.retries:
                self._sleep(2.0 ** attempt)
        raise LumenError(f"/chat/completions failed after {self.retries + 1} attempts ({last})")

    def _redact(self, text: str) -> str:
        """Server text with the key and anything token-shaped removed (error messages must never carry the key)."""
        if self._key:
            text = text.replace(self._key, "[redacted]")
        return _TOKENISH.sub("[redacted]", text)

    def close(self) -> None:
        self._http.close()


def _short(text: str, n: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def parse_json_content(content: str | None) -> Any:
    """Parse model text as JSON, tolerating ``` fences and leading/trailing prose. Raises ValueError."""
    if not content:
        raise ValueError("empty model reply")
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in model reply")
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    if end <= start:
        raise ValueError("no JSON found in model reply")
    return json.loads(text[start:end + 1])


# ---- master-run probes --------------------------------------------------------------------------------------------------
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "draft_order",
        "description": "Choose one option by id.",
        "parameters": {"type": "object", "properties": {"option_id": {"type": "string"}}, "required": ["option_id"]},
    },
}


def probe_cost(client: LumenClient) -> dict[str, Any]:
    """One tiny uncached call: returns model, usage and seconds. Compare the Lumen coin balance before/after."""
    started = time.monotonic()
    cache_dir, client.cache_dir = client.cache_dir, None
    try:
        reply = client.chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=400)
    finally:
        client.cache_dir = cache_dir
    return {"model": reply["model"], "usage": reply["usage"], "seconds": round(time.monotonic() - started, 2),
            "content": _short(reply["content"] or "", 40)}


def check_features(client: LumenClient) -> dict[str, Any]:
    """Two tiny calls: does JSON mode return parseable JSON, and does the model emit a tool call?"""
    out: dict[str, Any] = {"model": client.model}
    try:
        reply = client.chat([{"role": "user", "content": 'Return the JSON object {"ok": true} and nothing else.'}],
                            json_mode=True, max_tokens=400)
        out["json_usage"], out["json_finish_reason"] = reply["usage"], reply.get("finish_reason")
        out["json_mode"] = parse_json_content(reply["content"]) == {"ok": True}
    except (LumenError, ValueError) as exc:
        out["json_mode"] = False
        out["json_error"] = str(exc)
    try:
        reply = client.chat([{"role": "user", "content": "Call draft_order with option_id 'a'."}],
                            tools=[PROBE_TOOL], tool_choice="auto", max_tokens=400)
        calls = reply["tool_calls"]
        out["tool_calling"] = bool(calls) and calls[0].get("function", {}).get("name") == "draft_order"
        out["tool_usage"] = reply["usage"]
    except LumenError as exc:
        out["tool_calling"] = False
        out["tool_error"] = str(exc)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.lumen", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("models", "probe", "check"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-calls", type=int, default=2, help="hard budget for this command (default 2)")
    args = parser.parse_args(argv)
    cfg = core_config.load_settings(args.config)
    client = LumenClient.from_settings(cfg, cache_dir=None, max_calls=args.max_calls)
    if not client.has_key:
        print("no Lumen key found (environment or settings llm.env_file)", file=sys.stderr)
        return 2
    try:
        if args.command == "models":
            models = client.list_models()
            result: dict[str, Any] = {"models": models, "picked": client.pick_model(models)}
        elif args.command == "probe":
            result = probe_cost(client)
        else:
            result = check_features(client)
    except LumenError as exc:
        print(f"lumen error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    result["calls"] = client.calls
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
