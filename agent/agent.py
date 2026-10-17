"""Agent that drafts (never places) follow-up orders: rules first, an optional Lumen layer on top (plan §9).

    agent, info = draft(case, cfg)                    # rules + template rationales, zero LLM calls (default)
    agent, info = draft(case, cfg, client=client)     # tool calling -> JSON prompt -> rules + template

The deterministic engine (agent.rules) computes the draftable options for the case: deduped rule orders that passed
the lane guard, each with an option id "<rule_id>:<order index>". The LLM may only CHOOSE among those options and write
the rationale. Every LLM draft is validated:
  * each chosen option id exists, appears once, and its items are a non-empty subset of that option's items;
  * when the rules drafted anything, at least one option is chosen (the LLM cannot silently drop every order);
  * rationales are non-empty, at most RATIONALE_MAX_CHARS, use no settings summary.banned_patterns wording, and
    quote only numbers that appear in the case findings or the option's items.
Any violation, error, missing key or spent budget falls back to the next layer; the rules layer always succeeds.
`blocked` actions always come from the rules (blocked options are never shown to the LLM).

Tools offered to the model (OpenAI function calling):
    get_case_findings()                      -> the case findings (structured; no image, no ground truth)
    lookup_rules()                           -> the draftable options with rule labels, notes and citations
    draft_order(option_id, items, rationale) -> records one chosen order
The JSON-prompt fallback asks for {"orders": [{"option_id", "items", "rationale"}]} in one call.

info = {"path": "tools" | "json" | "rules", "llm_calls": int, "cached": bool, "fallbacks": [reasons]}.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from agent import rules as R
from agent.lumen import LumenError, parse_json_content

RATIONALE_MAX_CHARS = 300

SYSTEM_PROMPT = (
    "You assist a pathologist with a research-prototype breast pathology worklist (simulated cases; not for clinical "
    "use). Deterministic rules have already proposed follow-up orders as numbered options. You may only choose among "
    "those options (you may drop items from an option, never add any) and write a one-sentence rationale per chosen "
    "order. Orders are drafts: nothing is ordered until the pathologist approves. Never state a diagnosis as fact; say "
    "'suggested' or 'associated with'. Quote only numbers present in the findings."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "get_case_findings", "description": "Structured findings of the case (no image).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "lookup_rules", "description": "The draftable order options proposed by the rules.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "draft_order", "description": "Draft one order from an option (subset of its items allowed).",
        "parameters": {"type": "object", "properties": {
            "option_id": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"}},
            "required": ["option_id", "items", "rationale"]}}},
]


class DraftInvalid(ValueError):
    """An LLM draft broke a rule (unknown option, invented item, bad rationale)."""


def _options_payload(options: Sequence[R.Option], rules: R.RuleSet) -> list[dict[str, Any]]:
    by_id = rules.by_id()
    return [{"option_id": o.option_id, "type": o.type, "items": list(o.items), "rule": by_id[o.rule_id].label,
             "note": by_id[o.rule_id].note, "citations": list(by_id[o.rule_id].citations),
             "template_rationale": o.rationale} for o in options]


def validate_draft(chosen: Any, options: Sequence[R.Option], findings: Sequence[Mapping[str, Any]],
                   cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Contract orders from an LLM choice list [{"option_id", "items", "rationale"}]; raises DraftInvalid."""
    from backend.summary import finding_numbers, number_supported, numbers_in_text

    if not isinstance(chosen, list):
        raise DraftInvalid("orders must be a list")
    by_id = {o.option_id: o for o in options}
    banned = [re.compile(p, re.I) for p in cfg["summary"]["banned_patterns"]]
    allowed_numbers = [x for f in findings for x in finding_numbers(f.get("data"))]
    orders: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in chosen:
        if not isinstance(entry, Mapping):
            raise DraftInvalid("each order must be an object")
        option = by_id.get(str(entry.get("option_id")))
        if option is None:
            raise DraftInvalid(f"unknown option_id {entry.get('option_id')!r}")
        if option.option_id in seen:
            raise DraftInvalid(f"option {option.option_id} chosen twice")
        seen.add(option.option_id)
        items = entry.get("items")
        if items is None:
            items = list(option.items)
        if not isinstance(items, list) or not items or not all(isinstance(i, str) for i in items):
            raise DraftInvalid(f"option {option.option_id}: items must be a non-empty list of strings")
        invented = [i for i in items if i not in option.items]
        if invented:
            raise DraftInvalid(f"option {option.option_id}: items {invented} are not rule outputs")
        rationale = entry.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > RATIONALE_MAX_CHARS:
            raise DraftInvalid(f"option {option.option_id}: rationale must be 1-{RATIONALE_MAX_CHARS} characters")
        rationale = " ".join(rationale.split())
        if any(p.search(rationale) for p in banned):
            raise DraftInvalid(f"option {option.option_id}: rationale uses banned wording")
        item_numbers = [float(n) for it in option.items for n in re.findall(r"\d+(?:\.\d+)?", it)]
        for value, decimals, percent in numbers_in_text(rationale):
            if not number_supported(value, decimals, percent, allowed_numbers + item_numbers):
                raise DraftInvalid(f"option {option.option_id}: rationale number {value} is not in the findings")
        ordered_items = [i for i in option.items if i in items]   # keep rule order, drop duplicates
        orders.append({"type": option.type, "items": ordered_items, "rule_id": option.rule_id, "rationale": rationale})
    if options and not orders:
        raise DraftInvalid("the rules drafted orders but none were chosen")
    return orders


def _tool_result(name: str, findings: Sequence[Mapping[str, Any]], payload: list[dict[str, Any]]) -> Any:
    if name == "get_case_findings":
        return {"findings": list(findings)}
    if name == "lookup_rules":
        return {"options": payload}
    return {"error": f"unknown tool {name}"}


def _draft_with_tools(client: Any, findings: Sequence[Mapping[str, Any]], options: Sequence[R.Option],
                      rules: R.RuleSet, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _options_payload(options, rules)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Review the case findings and the rule options, then call draft_order once per "
                                    "order to draft. Reply 'done' when finished."},
    ]
    chosen: list[dict[str, Any]] = []
    for _ in range(int(cfg["llm"]["agent_max_turns"])):
        reply = client.chat(messages, tools=TOOLS, tool_choice="auto")
        calls = reply.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except ValueError:
                raise DraftInvalid(f"tool {name}: arguments are not JSON") from None
            if name == "draft_order":
                chosen.append(args)
                result: Any = {"recorded": args.get("option_id")}
            else:
                result = _tool_result(str(name), findings, payload)
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result)})
        if chosen and all((c.get("function") or {}).get("name") == "draft_order" for c in calls):
            break
    if not chosen:
        raise DraftInvalid("no draft_order tool call")
    return validate_draft(chosen, options, findings, cfg)


def _draft_with_json(client: Any, findings: Sequence[Mapping[str, Any]], options: Sequence[R.Option],
                     rules: R.RuleSet, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = {"findings": list(findings), "options": _options_payload(options, rules)}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Answer with JSON only: {\"orders\": [{\"option_id\": ..., \"items\": [...], "
                                    "\"rationale\": ...}]}\n" + json.dumps(payload, sort_keys=True, default=str)},
    ]
    reply = client.chat(messages, json_mode=bool(cfg["llm"]["json_mode"]))
    data = parse_json_content(reply.get("content"))
    if isinstance(data, Mapping):
        data = data.get("orders")
    return validate_draft(data, options, findings, cfg)


def draft(case: Mapping[str, Any], cfg: Mapping[str, Any], *, rules: R.RuleSet | None = None, client: Any = None
          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Contract `Agent` dict + info. case needs lane, prediction, quality and (for the LLM) findings."""
    rules = R.load_rules(cfg) if rules is None else rules
    options, blocked = R.options_for(case, cfg, rules)
    info: dict[str, Any] = {"path": "rules", "llm_calls": 0, "cached": False, "fallbacks": []}
    rules_orders = [R.option_order(o) for o in options]
    if client is None or not options:
        if client is not None:
            info["fallbacks"].append("no draftable options")
        return R.agent_block(rules_orders, blocked), info
    findings = list(case.get("findings") or [])
    layers = (("tools", _draft_with_tools),) if cfg["llm"]["tool_calling"] else ()
    layers += (("json", _draft_with_json),)
    calls_before, hits_before = client.calls, client.cache_hits
    for name, layer in layers:
        try:
            orders = layer(client, findings, options, rules, cfg)
        except (LumenError, ValueError) as exc:
            info["fallbacks"].append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        info.update(path=name, llm_calls=client.calls - calls_before, cached=client.cache_hits > hits_before)
        return R.agent_block(orders, blocked), info
    info.update(llm_calls=client.calls - calls_before)
    return R.agent_block(rules_orders, blocked), info
