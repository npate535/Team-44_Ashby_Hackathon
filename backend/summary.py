"""Grounded case summaries: findings-only LLM prompt, citation/number validator, deterministic template (plan §8.2).

The LLM never sees an image: it gets the structured `findings` array only and must answer JSON
{"sentences": [{"text": ..., "findings": ["F1", ...]}]} (a bare list is accepted too).

    summary, info = generate(findings, cfg)                  # template (zero LLM calls)
    summary, info = generate(findings, cfg, client=client)   # Lumen; validated; template fallback
    kept, dropped = validate(sentences, findings, cfg)
    sentences = template(findings, cfg)

Validator (a sentence is dropped when any rule fails; reasons are returned):
  * it cites no finding, or a finding ID that does not exist;
  * it uses a spelled-out number word ("ninety", "percent", "one" ...) or a letter-digit token that is not a finding ID
    or one of settings summary.allowed_codes (e.g. "Ki-67", "CK5/6"; "pT3", "Grade-3", "95th" are rejected);
  * a number in its text is not supported by the data of the findings it cites:
      - a decimal ("0.72", ".7") must be a rounding of a cited value at the same number of decimals;
      - a whole number ("4", "3150") must equal a cited value exactly, or round to one that is >= 10;
      - a percentage ("72%") must be a rounding of 100 x a cited value (or equal a cited value >= 1);
      - counts and indices (keys tiles, agree, of, qc_tiles_kept, min_tiles_kept) support only whole numbers, never
        decimals or percentages; signs are ignored; "3,150" is 3150;
  * it matches a settings summary.banned_patterns regex (no diagnosis stated as fact, no causal claims);
  * it is not a {"text": str, "findings": [str]} object.
At most summary.max_sentences survivors are kept; fewer than summary.min_valid_sentences => the template.

Template: one sentence per finding kind in a fixed narrative order (prediction, concepts, cells, retrieval, quality,
margin), chosen by clinical priority when more than max_sentences are possible. It only quotes numbers present in the
findings, so its own output always passes the validator (tested).

Cache: the Lumen client caches by (model, exact prompt); the prompt embeds the canonical findings and
summary.prompt_version, so changed findings never reuse stale text (a miss with no budget falls back to the template).
info["findings_hash"] records which findings a summary was written for.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

MARGIN_ORDER = ("superior", "inferior", "medial", "lateral")
NARRATIVE = ("prediction", "concept_up", "concept_down", "cells", "retrieval", "quality", "margin")

_ID_TOKEN = re.compile(r"\bF[1-9][0-9]*\b")
_HYBRID = re.compile(r"[A-Za-z][-/]?\d|\d[A-Za-z]")
_NUMBER_WORD = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"percent|percentage|half|quarter|dozen|once|twice)\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])(?:(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?|\.(\d+))(\s*%)?")
COUNT_KEYS = frozenset({"tiles", "agree", "of", "qc_tiles_kept", "min_tiles_kept"})


# ---- hashing ------------------------------------------------------------------------------------------------------
def findings_hash(findings: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(list(findings), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- numbers --------------------------------------------------------------------------------------------------------
def _strip_codes(text: str, codes: Sequence[str]) -> str:
    text = _ID_TOKEN.sub(" ", text)
    for code in sorted(codes, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(code)}(?![\w])", " ", text)
    return text


def unknown_tokens(text: str, codes: Sequence[str] = ()) -> list[str]:
    """Number words and letter-digit tokens that are neither finding IDs nor allowed codes."""
    cleaned = _strip_codes(text, codes)
    bad = [m.group(0) for m in _NUMBER_WORD.finditer(cleaned)]
    bad += [w for w in re.split(r"[\s,;:()\[\]]+", cleaned) if w and _HYBRID.search(w)]
    return bad


def numbers_in_text(text: str, codes: Sequence[str] = ()) -> list[tuple[float, int, bool]]:
    """(value, decimals, is_percent) for every number in `text` (finding IDs and allowed codes excluded)."""
    out = []
    for match in _NUMBER.finditer(_strip_codes(text, codes)):
        whole, frac, lead, pct = match.group(1), match.group(2), match.group(3), match.group(4)
        if lead is not None:
            whole, frac = "0", lead
        whole = whole.replace(",", "")
        value = float(f"{whole}.{frac}") if frac else float(whole)
        out.append((value, len(frac) if frac else 0, bool(pct)))
    return out


def finding_numbers(obj: Any, key: str | None = None) -> list[tuple[float, bool]]:
    """(absolute value, is_count) for every number in a finding's data (recursively; booleans excluded).
    Values under COUNT_KEYS are counts/indices."""
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [(abs(float(obj)), key in COUNT_KEYS)]
    if isinstance(obj, Mapping):
        return [x for k, v in obj.items() for x in finding_numbers(v, str(k))]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in finding_numbers(v, key)]
    return []


def number_supported(value: float, decimals: int, percent: bool, allowed: Sequence[tuple[float, bool]]) -> bool:
    """True when a text number is supported by an allowed (value, is_count) pair (rules in the module docstring)."""
    tol = 0.5 * 10.0 ** (-decimals) + 1e-9
    for x, is_count in allowed:
        if percent:
            if is_count:
                continue
            if abs(100.0 * x - value) <= tol or (x >= 1.0 and abs(x - value) <= tol):
                return True
        elif decimals == 0:
            if abs(x - value) <= 1e-9 or (not is_count and x >= 10.0 and abs(x - value) <= tol):
                return True
        elif not is_count and abs(x - value) <= tol:
            return True
    return False


# ---- validation ---------------------------------------------------------------------------------------------------
def _banned(cfg: Mapping[str, Any]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.I) for p in cfg["summary"]["banned_patterns"]]


def check_sentence(sentence: Any, by_id: Mapping[str, Mapping[str, Any]], banned: Sequence[re.Pattern[str]],
                   codes: Sequence[str] = ()) -> str | None:
    """Reason the sentence is invalid, or None when it passes."""
    if not isinstance(sentence, Mapping) or not isinstance(sentence.get("text"), str) or not sentence["text"].strip():
        return "not a {text, findings} object"
    cited = sentence.get("findings")
    if not isinstance(cited, list) or not all(isinstance(f, str) for f in cited):
        return "findings must be a list of finding IDs"
    if not cited:
        return "cites no finding"
    unknown = [f for f in cited if f not in by_id]
    if unknown:
        return f"cites unknown findings {unknown}"
    text = sentence["text"]
    for pattern in banned:
        if pattern.search(text):
            return f"banned wording /{pattern.pattern}/"
    odd = unknown_tokens(text, codes)
    if odd:
        return f"unrecognized number word or code {odd[0]!r}"
    allowed = [x for f in cited for x in finding_numbers(by_id[f].get("data"))]
    for value, decimals, percent in numbers_in_text(text, codes):
        if not number_supported(value, decimals, percent, allowed):
            shown = f"{value:.{decimals}f}{'%' if percent else ''}"
            return f"number {shown} is not in the cited findings"
    return None


def validate(sentences: Any, findings: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, dropped): kept sentences as {"text", "findings"} (at most summary.max_sentences), dropped with reasons."""
    by_id = {f["id"]: f for f in findings}
    banned = _banned(cfg)
    codes = list(cfg["summary"]["allowed_codes"])
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for sentence in sentences if isinstance(sentences, list) else []:
        reason = check_sentence(sentence, by_id, banned, codes)
        if reason is not None:
            dropped.append({"text": str(sentence.get("text", "")) if isinstance(sentence, Mapping) else str(sentence),
                            "reason": reason})
        elif len(kept) >= int(cfg["summary"]["max_sentences"]):
            dropped.append({"text": sentence["text"], "reason": "beyond summary.max_sentences"})
        else:
            kept.append({"text": " ".join(sentence["text"].split()),
                         "findings": list(dict.fromkeys(sentence["findings"]))})
    if not isinstance(sentences, list):
        dropped.append({"text": str(sentences)[:200], "reason": "reply is not a list of sentences"})
    return kept, dropped


# ---- template --------------------------------------------------------------------------------------------------------
def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if len(text) > 1 and text[1:2].islower() else text


def _class_label(cfg: Mapping[str, Any], cls: str) -> str:
    return _lower_first(str(cfg["ui"]["class_labels"].get(cls, cls)))


def _join(words: Sequence[str]) -> str:
    words = list(words)
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _regions(tiles: Any) -> str:
    tiles = [int(t) for t in tiles or []]
    if not tiles:
        return ""
    return ("region " if len(tiles) == 1 else "regions ") + _join([str(t) for t in tiles])


def _humanize(feature: str) -> str:
    return feature.replace("__", " ").replace("_", " ")


def _fmt_p(p: float) -> str:
    return f"{float(p):.2f}"


def _prediction_sentence(f: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    d = f["data"]
    top, members = d["top"], list(d["set"])
    level = d.get("confidence")
    set_name = f"{round(100 * float(level))}% confidence set" if level is not None else "confidence set"
    head = f"Suggested category: {_class_label(cfg, top)} (calibrated probability {_fmt_p(d['p'])})"
    others = [m for m in members if m != top]
    if not members:
        return f"{head}; the {set_name} is empty, so this read is uncertain."
    if top not in members:
        return f"{head}; the {set_name} holds {_join([_class_label(cfg, m) for m in members])} instead, so this read is uncertain."
    if not others:
        return f"{head}; the {set_name} holds this category only."
    verb = "is" if len(others) == 1 else "are"
    return f"{head}, and {_join([_class_label(cfg, m) for m in others])} {verb} also in the {set_name}."


def _concept_label(f: Mapping[str, Any]) -> str:
    d = f["data"]
    return _lower_first(str(d.get("label") or _humanize(d["concept"])))


def _concept_up_sentence(fs: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> str:
    aggressive = _class_label(cfg, cfg["aggressive_class"])
    first = fs[0]
    where = _regions(first["data"]["tiles"])
    label = _concept_label(first)
    label = label[:1].upper() + label[1:]
    text = f"{label}{f' ({where})' if where else ''} is the feature most associated with {aggressive} in this read"
    if len(fs) > 1:
        text += ", followed by " + _join([_concept_label(f) for f in fs[1:]])
    return text + "."


def _concept_down_sentence(fs: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> str:
    aggressive = _class_label(cfg, cfg["aggressive_class"])
    labels = [_concept_label(f) for f in fs]
    joined = _join(labels)
    joined = joined[:1].upper() + joined[1:]
    verb = "is a feature" if len(fs) == 1 else "are features"
    where = _regions(fs[0]["data"]["tiles"]) if len(fs) == 1 else ""
    return f"{joined}{f' ({where})' if where else ''} {verb} associated with a lower likelihood of {aggressive}."


def _cells_sentence(f: Mapping[str, Any]) -> str:
    d = f["data"]
    label = str(d.get("label") or _humanize(d["feature"]))
    where = _regions(d.get("tiles"))
    suffix = f" ({where})" if where else ""
    pct = d.get("ref_percentile")
    if pct is None:
        return f"{label[:1].upper() + label[1:]} is a measured feature associated with this read{suffix}."
    pct = float(pct)
    if pct >= 0.5:
        return f"{label[:1].upper() + label[1:]} is higher than in {round(100 * pct)}% of benign reference cases{suffix}."
    return f"Only {round(100 * pct)}% of benign reference cases have a lower {_lower_first(label)}{suffix}."


def _retrieval_sentence(f: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    d = f["data"]
    return (f"{int(d['agree'])} of {int(d['of'])} most similar reference cases share the suggested category "
            f"({_class_label(cfg, d['label'])}).")


def _quality_sentence(f: Mapping[str, Any]) -> tuple[str, bool]:
    """(sentence, is_concern)."""
    d = f["data"]
    kept, minimum = d.get("qc_tiles_kept"), d.get("min_tiles_kept")
    if kept is not None and minimum is not None and int(kept) < int(minimum):
        noun = "region" if int(kept) == 1 else "regions"
        return (f"Only {int(kept)} tissue {noun} passed quality control (fewer than {int(minimum)}), so this read "
                f"rests on little tissue.", True)
    if d["reliability_flag"]:
        return ("The reliability check flagged this read (image quality or an unusual appearance), so its confidence "
                "is reduced.", True)
    return "The reliability check raised no image-quality concern.", False


def _margin_sentence(f: Mapping[str, Any]) -> str:
    d = f["data"]
    call = d["suggested_call"]
    if call == "tumor_at_margin":
        involved = [m for m in MARGIN_ORDER if m in d["involved"]]
        noun = "margin" if len(involved) == 1 else "margins"
        return f"Suggested margin call (simulated frozen section): tumor at the {_join(involved)} {noun}."
    if call == "clear":
        return "Suggested margin call (simulated frozen section): margins clear."
    return ("Suggested margin call (simulated frozen section): defer to permanent sections, because the reliability "
            "check flagged this read.")


def template(findings: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deterministic grounded sentences from the findings alone (<= summary.max_sentences)."""
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for f in findings:
        by_kind.setdefault(f["kind"], []).append(f)
    candidates: list[tuple[int, str, dict[str, Any]]] = []   # (priority, narrative slot, sentence)

    def add(priority: int, slot: str, text: str, cited: Sequence[Mapping[str, Any]]) -> None:
        candidates.append((priority, slot, {"text": text, "findings": [c["id"] for c in cited]}))

    for f in by_kind.get("prediction", [])[:1]:
        add(0, "prediction", _prediction_sentence(f, cfg), [f])
    for f in by_kind.get("margin", [])[:1]:
        add(1, "margin", _margin_sentence(f), [f])
    concepts = by_kind.get("concept", [])
    ups = [f for f in concepts if float(f["data"]["contribution"]) > 0]
    downs = [f for f in concepts if float(f["data"]["contribution"]) < 0]
    if ups:
        add(2, "concept_up", _concept_up_sentence(ups, cfg), ups)
    quality = by_kind.get("quality", [])[:1]
    concern = False
    if quality:
        text, concern = _quality_sentence(quality[0])
        add(3 if concern else 8, "quality", text, quality)
    for i, f in enumerate(by_kind.get("cells", [])):
        add(4 + i * 10, "cells", _cells_sentence(f), [f])
    for f in by_kind.get("retrieval", [])[:1]:
        add(5, "retrieval", _retrieval_sentence(f, cfg), [f])
    if downs:
        add(6, "concept_down", _concept_down_sentence(downs, cfg), downs)

    chosen = sorted(candidates, key=lambda c: c[0])[: int(cfg["summary"]["max_sentences"])]
    chosen.sort(key=lambda c: (NARRATIVE.index(c[1]), c[0]))
    return [c[2] for c in chosen]


def template_summary(findings: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {"generator": cfg["summary"]["template_generator"], "sentences": template(findings, cfg)}


# ---- LLM path -------------------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You write short, grounded summaries of a research-prototype pathology worklist read (breast H&E, BACH dataset). "
    "You never see the image: you only get structured findings. Rules:\n"
    "1. Write at most {max_sentences} sentences in plain clinical English.\n"
    "2. Every sentence cites the IDs of the findings it uses, and only those findings.\n"
    "3. Use only numbers that appear in the cited findings (probabilities may be written as 0.72 or 72%).\n"
    "4. Never state a diagnosis as fact: say 'suggested', 'associated with', 'cannot be excluded'. Never claim a "
    "feature causes anything. Concepts are 'features associated with invasive carcinoma', never a Nottingham grade.\n"
    "5. Surgical cases are simulated frozen sections; margins are simulated.\n"
    'Answer with JSON only: {{"sentences": [{{"text": "...", "findings": ["F1"]}}]}}'
)


GROUND_TRUTH_KEYS = frozenset({"true_label", "eval_only", "split", "group", "patient_id"})


def build_messages(findings: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    """Findings-only prompt. Raises ValueError if a finding carries a ground-truth-like key (defensive)."""
    leaked = sorted({k for f in findings for k in (f.get("data") or {}) if k in GROUND_TRUTH_KEYS})
    if leaked:
        raise ValueError(f"findings must not carry ground truth keys {leaked}")
    payload = {
        "prompt_version": cfg["summary"]["prompt_version"],
        "classes": {c: cfg["ui"]["class_labels"].get(c, c) for c in cfg["classes"]},
        "aggressive_class": cfg["aggressive_class"],
        "findings": list(findings),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_sentences=cfg["summary"]["max_sentences"])},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)},
    ]


def parse_reply(content: str | None) -> Any:
    """Sentences list from the model reply ({"sentences": [...]} or a bare list). Raises ValueError."""
    from agent.lumen import parse_json_content

    data = parse_json_content(content)
    if isinstance(data, Mapping):
        data = data.get("sentences")
    if not isinstance(data, list):
        raise ValueError("reply has no sentences list")
    return data


def generate(findings: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], *, client: Any = None
             ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Contract `summary` block plus info {generator, path, reason, llm_calls, cached, dropped, findings_hash}.

    client=None (default) => template, zero LLM calls. With a agent.lumen.LumenClient: one findings-only call (or a
    cache hit); fewer than summary.min_valid_sentences valid sentences, a missing key, a spent budget or any error
    => the template, with the reason recorded.
    """
    info: dict[str, Any] = {"findings_hash": findings_hash(findings), "llm_calls": 0, "cached": False, "dropped": []}
    if client is None:
        return template_summary(findings, cfg), {**info, "generator": "template", "path": "template",
                                                 "reason": "llm disabled"}
    from agent.lumen import LumenError

    messages = build_messages(findings, cfg)
    calls_before = client.calls
    try:
        reply = client.chat(messages, json_mode=bool(cfg["llm"]["json_mode"]))
        sentences = parse_reply(reply["content"])
    except (LumenError, ValueError) as exc:
        info["llm_calls"] = client.calls - calls_before
        return template_summary(findings, cfg), {**info, "generator": "template", "path": "template",
                                                 "reason": f"{type(exc).__name__}: {exc}"}
    info["llm_calls"] = client.calls - calls_before
    info["cached"] = bool(reply.get("cached"))
    kept, dropped = validate(sentences, findings, cfg)
    info["dropped"] = dropped
    if len(kept) < int(cfg["summary"]["min_valid_sentences"]):
        return template_summary(findings, cfg), {**info, "generator": "template", "path": "template",
                                                 "reason": f"only {len(kept)} valid sentences"}
    generator = f"lumen:{reply['model']}"
    return {"generator": generator, "sentences": kept}, {**info, "generator": generator, "path": "llm", "reason": ""}
