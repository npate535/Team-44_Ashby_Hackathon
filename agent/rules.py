"""Deterministic follow-up order engine over config/stain_rules.yaml (plan §9, BUILD_PLAN §A14).

The rules are data. Each rule has `id`, `label`, `when` (conditions, all must hold), `orders` (one or more
{type, items}), a template `rationale` (str.format fields below), an optional `note` and `citations`. The engine
never invents an item: every drafted item comes verbatim from a rule that fired.

    rules = load_rules(cfg)                      # validated RuleSet (settings agent.stain_rules)
    hits = fired_rules(case, cfg, rules)         # rules whose conditions hold, in file order
    agent = draft_orders(case, cfg, rules)       # contract `Agent` dict: {status, orders, blocked}

Conditions (stain_rules.yaml header): top_class, in_set_all, reliability_flag, insufficient_tissue
(quality.qc_tiles_kept < settings qc.min_tiles_kept), lanes. The case is a Case-JSON-shaped mapping; only `lane`,
`prediction.{classes, probs, conformal_set}` and `quality.{qc_tiles_kept, reliability_flag}` are read, so step 08 can
call this before the rest of the case exists.

Rationale fields: {top_class}, {set} (conformal set joined by ", "), {qc_tiles_kept}, {min_tiles_kept}.

Lane guard: order types in `lane_guard.blocked_types` are dropped for lanes in settings agent.blocked_lanes, and a
single visible `blocked` action ({action, reason}) is emitted instead when at least one such order fired.

Dedupe: an item already drafted by an earlier order of the same type is removed from later orders; an order left
without items is dropped. Status is "drafted" when at least one order remains, else "none".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core import config as core_config

ORDER_TYPES = ("stain", "recut", "deeper_levels", "new_sample")
CONDITION_KEYS = ("top_class", "in_set_all", "reliability_flag", "insufficient_tissue", "lanes")
RULE_KEYS = ("id", "label", "when", "orders", "rationale", "note", "citations")
RULE_REQUIRED = ("id", "label", "when", "orders", "rationale")
RATIONALE_FIELDS = ("top_class", "set", "qc_tiles_kept", "min_tiles_kept")
LANES = ("biopsy", "surgical")


@dataclass(frozen=True)
class OrderSpec:
    type: str
    items: tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    id: str
    label: str
    when: Mapping[str, Any]
    orders: tuple[OrderSpec, ...]
    rationale: str
    note: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class LaneGuard:
    blocked_types: tuple[str, ...]
    action: str
    reason: str


@dataclass(frozen=True)
class RuleSet:
    version: Any
    status: str
    lane_guard: LaneGuard
    rules: tuple[Rule, ...]

    def by_id(self) -> dict[str, Rule]:
        return {r.id: r for r in self.rules}


@dataclass(frozen=True)
class RuleHit:
    """A rule whose conditions hold for a case, with its rendered rationale."""

    rule: Rule
    rationale: str


# ---- loading ---------------------------------------------------------------------------------------------------
def rules_path(cfg: Mapping[str, Any]) -> Path:
    return core_config.resolve_repo_path(cfg["agent"]["stain_rules"])


def load_rules(cfg: Mapping[str, Any], path: str | Path | None = None) -> RuleSet:
    """Read and validate the stain rules (default: settings agent.stain_rules). Raises ValueError on any problem."""
    path = Path(path) if path is not None else rules_path(cfg)
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return parse_rules(data, cfg)


def parse_rules(data: Any, cfg: Mapping[str, Any]) -> RuleSet:
    problems: list[str] = []
    if not isinstance(data, Mapping):
        raise ValueError("stain_rules.yaml: top level must be a mapping")
    classes = list(cfg["classes"])
    for key in ("status", "lane_guard", "rules"):
        if key not in data:
            problems.append(f"missing top-level key {key!r}")
    status = data.get("status")
    if not isinstance(status, str) or "clinical review" not in status.lower():
        problems.append("status must state the rules are illustrative and require clinical review")
    guard_raw = data.get("lane_guard") or {}
    blocked_types = tuple(guard_raw.get("blocked_types") or ())
    bad_types = [t for t in blocked_types if t not in ORDER_TYPES]
    if bad_types:
        problems.append(f"lane_guard.blocked_types has unknown order types {bad_types}")
    for key in ("action", "reason"):
        if not isinstance(guard_raw.get(key), str) or not guard_raw.get(key):
            problems.append(f"lane_guard.{key} must be a non-empty string")
    guard = LaneGuard(blocked_types, str(guard_raw.get("action", "")), str(guard_raw.get("reason", "")))

    rules: list[Rule] = []
    seen: set[str] = set()
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        problems.append("rules must be a non-empty list")
        raw_rules = []
    for i, raw in enumerate(raw_rules):
        where = f"rules[{i}] ({raw.get('id', '?') if isinstance(raw, Mapping) else '?'})"
        if not isinstance(raw, Mapping):
            problems.append(f"{where} must be a mapping")
            continue
        missing = [k for k in RULE_REQUIRED if k not in raw]
        extra = [k for k in raw if k not in RULE_KEYS]
        if missing:
            problems.append(f"{where}: missing keys {missing}")
            continue
        if extra:
            problems.append(f"{where}: unexpected keys {extra}")
        rid = raw["id"]
        if not isinstance(rid, str) or not rid or rid in seen:
            problems.append(f"{where}: id must be a unique non-empty string")
            continue
        seen.add(rid)
        when = raw["when"]
        if not isinstance(when, Mapping) or not when:
            problems.append(f"{where}: when must be a non-empty mapping")
            continue
        for key, value in when.items():
            if key not in CONDITION_KEYS:
                problems.append(f"{where}: unknown condition {key!r}; allowed {list(CONDITION_KEYS)}")
            elif key == "top_class" and value not in classes:
                problems.append(f"{where}: top_class {value!r} is not a class {classes}")
            elif key == "in_set_all" and (not isinstance(value, list) or not value
                                          or any(v not in classes for v in value)):
                problems.append(f"{where}: in_set_all must be a non-empty list of classes {classes}")
            elif key in ("reliability_flag", "insufficient_tissue") and not isinstance(value, bool):
                problems.append(f"{where}: {key} must be true or false")
            elif key == "lanes" and (not isinstance(value, list) or any(v not in LANES for v in value)):
                problems.append(f"{where}: lanes must be a list drawn from {list(LANES)}")
        orders: list[OrderSpec] = []
        if not isinstance(raw["orders"], list) or not raw["orders"]:
            problems.append(f"{where}: orders must be a non-empty list")
        else:
            for j, order in enumerate(raw["orders"]):
                otype = order.get("type") if isinstance(order, Mapping) else None
                items = order.get("items") if isinstance(order, Mapping) else None
                if otype not in ORDER_TYPES:
                    problems.append(f"{where}: orders[{j}].type must be one of {list(ORDER_TYPES)}")
                    continue
                if (not isinstance(items, list) or not items or len(set(items)) != len(items)
                        or not all(isinstance(it, str) and it.strip() for it in items)):
                    problems.append(f"{where}: orders[{j}].items must be unique non-empty strings")
                    continue
                orders.append(OrderSpec(otype, tuple(items)))
        rationale = raw["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            problems.append(f"{where}: rationale must be a non-empty string")
        else:
            try:
                rationale.format(**{f: "x" for f in RATIONALE_FIELDS})
            except (KeyError, IndexError, ValueError) as exc:
                problems.append(f"{where}: rationale uses unknown fields ({exc}); allowed {list(RATIONALE_FIELDS)}")
        citations = raw.get("citations") or []
        if not isinstance(citations, list):
            problems.append(f"{where}: citations must be a list")
            citations = []
        rules.append(Rule(rid, str(raw["label"]), dict(when), tuple(orders), str(rationale),
                          str(raw.get("note") or ""), tuple(str(c) for c in citations)))
    if problems:
        raise ValueError("stain_rules.yaml is invalid:\n- " + "\n- ".join(problems))
    return RuleSet(data.get("version"), str(status), guard, tuple(rules))


# ---- evaluation --------------------------------------------------------------------------------------------------
def top_class(prediction: Mapping[str, Any], aggressive: str | None = None) -> str:
    """Argmax class of the calibrated probabilities. Ties resolve toward `aggressive` (as core.tiers does), else
    toward the first class in order."""
    classes = list(prediction["classes"])
    probs = [float(p) for p in prediction["probs"]]
    if len(classes) != len(probs) or not classes:
        raise ValueError(f"prediction has {len(probs)} probs for {len(classes)} classes")
    if aggressive in classes and probs[classes.index(aggressive)] >= max(probs):
        return aggressive
    return classes[max(range(len(probs)), key=lambda i: (probs[i], -i))]


def _context(case: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    lane = case["lane"]
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}")
    prediction = case["prediction"]
    conformal_set = list(prediction["conformal_set"])
    unknown = [c for c in conformal_set if c not in prediction["classes"]]
    if unknown:
        raise ValueError(f"conformal_set has classes {unknown} outside prediction.classes")
    quality = case["quality"]
    qc_kept = int(quality["qc_tiles_kept"])
    min_kept = int(cfg["qc"]["min_tiles_kept"])
    return {
        "lane": lane,
        "top_class": top_class(prediction, cfg["aggressive_class"]),
        "conformal_set": conformal_set,
        "reliability_flag": bool(quality["reliability_flag"]),
        "insufficient_tissue": qc_kept < min_kept,
        "qc_tiles_kept": qc_kept,
        "min_tiles_kept": min_kept,
    }


def _holds(when: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    for key, value in when.items():
        if key == "top_class" and ctx["top_class"] != value:
            return False
        if key == "in_set_all" and not all(c in ctx["conformal_set"] for c in value):
            return False
        if key == "reliability_flag" and ctx["reliability_flag"] != value:
            return False
        if key == "insufficient_tissue" and ctx["insufficient_tissue"] != value:
            return False
        if key == "lanes" and ctx["lane"] not in value:
            return False
    return True


def fired_rules(case: Mapping[str, Any], cfg: Mapping[str, Any], rules: RuleSet) -> list[RuleHit]:
    """Rules whose conditions hold, in file order, with rendered template rationales."""
    ctx = _context(case, cfg)
    fields = {"top_class": ctx["top_class"], "set": ", ".join(ctx["conformal_set"]) or "empty",
              "qc_tiles_kept": ctx["qc_tiles_kept"], "min_tiles_kept": ctx["min_tiles_kept"]}
    return [RuleHit(rule, rule.rationale.format(**fields)) for rule in rules.rules if _holds(rule.when, ctx)]


def is_blocked(order_type: str, lane: str, cfg: Mapping[str, Any], rules: RuleSet) -> bool:
    return lane in cfg["agent"]["blocked_lanes"] and order_type in rules.lane_guard.blocked_types


@dataclass(frozen=True)
class Option:
    """One draftable order (a rule order that survived the lane guard and dedupe): what an LLM may choose from."""

    option_id: str            # "<rule_id>:<order index>"
    rule_id: str
    type: str
    items: tuple[str, ...]
    rationale: str


def options_for(case: Mapping[str, Any], cfg: Mapping[str, Any], rules: RuleSet) -> tuple[list[Option], list[dict]]:
    """Draftable options (deduped, lane-guarded) and the contract `blocked` actions for a case."""
    lane = case["lane"]
    options: list[Option] = []
    seen: dict[str, set[str]] = {}
    blocked_hit = False
    for hit in fired_rules(case, cfg, rules):
        for j, spec in enumerate(hit.rule.orders):
            if is_blocked(spec.type, lane, cfg, rules):
                blocked_hit = True
                continue
            taken = seen.setdefault(spec.type, set())
            items = tuple(it for it in spec.items if it not in taken)
            if not items:
                continue
            taken.update(items)
            options.append(Option(f"{hit.rule.id}:{j}", hit.rule.id, spec.type, items, hit.rationale))
    blocked = [{"action": rules.lane_guard.action, "reason": rules.lane_guard.reason}] if blocked_hit else []
    return options, blocked


def agent_block(orders: Sequence[Mapping[str, Any]], blocked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"status": "drafted" if orders else "none", "orders": [dict(o) for o in orders],
            "blocked": [dict(b) for b in blocked]}


def option_order(option: Option, rationale: str | None = None) -> dict[str, Any]:
    return {"type": option.type, "items": list(option.items), "rule_id": option.rule_id,
            "rationale": rationale if rationale is not None else option.rationale}


def draft_orders(case: Mapping[str, Any], cfg: Mapping[str, Any], rules: RuleSet | None = None) -> dict[str, Any]:
    """Contract `Agent` dict drafted by the rules alone (template rationales; zero LLM calls)."""
    rules = load_rules(cfg) if rules is None else rules
    options, blocked = options_for(case, cfg, rules)
    return agent_block([option_order(o) for o in options], blocked)


def rule_metadata(rules: RuleSet) -> list[dict[str, Any]]:
    """JSON-ready rule descriptions (for the UI's 'why was this drafted' panel and the LLM's lookup_rules tool)."""
    return [{"rule_id": r.id, "label": r.label, "when": dict(r.when),
             "orders": [{"type": o.type, "items": list(o.items)} for o in r.orders],
             "rationale": r.rationale, "note": r.note, "citations": list(r.citations)} for r in rules.rules]
