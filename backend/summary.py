"""Grounded case summaries: findings-only LLM prompt, citation/number validator, deterministic template (plan §8.2).

The LLM never sees an image: it gets the structured `findings` array only and must answer JSON
{"sentences": [{"text": ..., "findings": ["F1", ...]}]} (a bare list is accepted too).

    summary, info = generate(findings, cfg)                  # template (zero LLM calls)
    summary, info = generate(findings, cfg, client=client)   # Lumen; validated; template fallback
    kept, dropped = validate(sentences, findings, cfg)
    sentences = template(findings, cfg)

Validator (a sentence is dropped when any rule fails; reasons are returned):
  * it cites no finding, or a finding ID that does not exist;
  * it uses a spelled-out number word ("ninety", "percent", "one" ...) or a letter-digit token that is not a finding ID,
    part of a cited finding's own `label` (e.g. "Normal two-layered lobules"), or one of settings summary.allowed_codes (e.g. "Ki-67", "CK5/6"; "pT3", "Grade-3", "95th" are rejected);
  * a number in its text is not supported by the data of the findings it cites:
      - a decimal ("0.72", ".7") must be a rounding of a cited value at the same number of decimals;
      - a whole number ("4", "3150") must equal a cited value exactly, or round to one that is >= 10;
      - a percentage ("72%") must be a rounding of 100 x a cited value (or equal a cited value >= 1);
      - counts and indices (keys tiles, agree, of, qc_tiles_kept, min_tiles_kept) support only whole numbers, never
        decimals or percentages; signs are ignored; "3,150" is 3150;
  * a number is not rounded for reading (more than two decimals, or decimals on a value >= 100), or reads "only 0%";
  * it matches a settings summary.banned_patterns regex (no diagnosis stated as fact, no causal claims);
  * it contradicts a cited finding's categorical facts (semantic_problem, checked last): the suggested category and
    the exact confidence set; the margin call (hedged, exact margins); the reliability flag; similar-case agreement
    (quoted, never inflated); cell wording against the benign-reference percentile; concept level wording; and the
    concept direction, which always refers to invasive carcinoma (never "associated with this result" for another
    category);
  * it is not a {"text": str, "findings": [str]} object.
At most summary.max_sentences survivors are kept; fewer than summary.min_valid_sentences => the template.
Required facts: the prediction, the margin call and any quality concern must be cited; when the LLM omits one, its
template sentence is added (ensure_required: the prediction first). When no kept sentence cites the strongest concept,
its template sentence follows the prediction (replacing a similar-cases or cells sentence if there is no room). Any
template sentence makes the generator read "lumen:<model>+template".

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
    for code in sorted({c for c in codes if c}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(code)}(?![\w])", " ", text, flags=re.I)
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


# ---- semantic consistency (safety-critical facts must not be contradicted) ------------------------------------------
_CLEAR = re.compile(r"\b(clear|negative|free of tumou?r|uninvolved)\b", re.I)
_DEFER = re.compile(r"\b(defer\w*|permanent sections?|cannot be (determined|assessed)|unreliable|not reliable)\b", re.I)
_INVOLVED = re.compile(r"\b(tumou?r (is )?(suggested )?at|involv\w*|positive)\b", re.I)
_RELIABLE = re.compile(r"\b(high(ly)? reliab\w*|reliable|no (image[- ]quality )?concern\w*|good quality)\b", re.I)
_UNRELIABLE = re.compile(r"\b(flag\w*|unreliable|not reliable|reduced (confidence|reliability)|low reliability)\b", re.I)
_PRESENT = re.compile(r"\b(prominent|present|presence|frequent|abundant|dense|marked|extensive|increased|high|"
                      r"numerous|conspicuous|notable|evident|observed|seen|shows?|showing)\b", re.I)
_ABSENT = re.compile(r"\b(scarce|absent|absence|few|fewer|sparse|reduced|low|lack\w*|minimal|less|below|"
                     r"decreased|limited|rare)\b", re.I)


def _without(text: str, phrases: Sequence[str]) -> str:
    for phrase in sorted({p for p in phrases if p}, key=len, reverse=True):
        text = re.sub(re.escape(phrase), " ", text, flags=re.I)
    return text


_SUPPORT = re.compile(r"\b(associated with|support\w*|favou?r\w*|consistent with|point\w* (to|toward)|drive\w*|"
                      r"contribut\w* to|in keeping with)\b", re.I)
_CONTRAST = re.compile(r"\b(away|against|argu\w*|counter\w*|despite|although|but|less likely|lower\w*|while|whereas|"
                       r"oppos\w*|reduc\w*|weigh\w* against|non-?invasive|benign|in situ)\b", re.I)
# A margin call is hedged only by naming it a suggestion or a simulated call ("the read is ..." is not a hedge).
_HEDGE = re.compile(r"\b(suggest\w*|simulated|call\w*)\b", re.I)
# Concept contributions only measure invasive carcinoma: "associated with this result" (or "the strongest feature")
# silently re-targets them at the suggested category, which is wrong whenever that category is not invasive.
_DEICTIC = re.compile(
    r"\b(associated with|support\w*|favou?r\w*|consistent with|in keeping with|point\w* (to|toward)|explain\w*)\s+"
    r"(this|the)\s+(finding|findings|result|results|category|read|suggestion|prediction|class|classification|call|"
    r"impression|assessment|diagnosis)\b"
    r"|\b(strongest|main|key|leading|primary|top|principal)\s+(associated\s+|supporting\s+|contributing\s+)?features?\b"
    r"|\b(associated|supporting|contributing)\s+features?\b", re.I)
# Stated direction relative to invasive carcinoma; it must match the sign of every cited concept's contribution.
_TOWARD_INV = re.compile(
    r"\b(toward\w*|favou?r\w*|support\w*|suggest\w*|point\w* to|in keeping with|consistent with|associated with|"
    r"more likely|higher likelihood of|increas\w* the likelihood of|rais\w* the likelihood of)\s+(an?\s+)?invasive\b"
    r"|\bpush\w*\s+(the read\s+|this read\s+|it\s+)?(up|toward\w*)\b", re.I)
_AWAY_INV = re.compile(
    r"\b(away from|against|lower\w* (the )?likelihood of|less likely|reduc\w* the likelihood of|decreas\w* the "
    r"likelihood of|argu\w* against|weigh\w* against)\s+(an?\s+)?invasive\b|\blower likelihood\b|\bnon-?invasive\b"
    r"|\bpush\w*\s+(the read\s+|this read\s+|it\s+)?(down|away)\b", re.I)
_DIRECTION_CLAUSE = re.compile(r";|,?\s*\b(while|whereas|but|although|though)\b", re.I)
# How the concepts of the bank are written in a sentence (label words, synonyms); others fall back to the label.
CONCEPT_MENTIONS = {
    "invasion": r"tumou?r[- ]infiltrat\w*|infiltrating stroma|infiltrative|\binvasion\b",
    "normal_lobules": r"lobul\w*|two-layered",
    "lymphocytes": r"lymphocyt\w*|\bTILs?\b|immune infiltrat\w*",
    "tubules": r"tubul\w*|\bgland\w*",
    "mitoses": r"mito\w*",
    "benign_stroma_fat": r"benign (fibrous )?stroma|fibrous stroma|\bfat\b|adipose",
    "duct_confined": r"confined|intraductal|within (the )?ducts?",
    "pleomorphism": r"pleomorph\w*",
    "necrosis": r"necros\w*|necrotic",
}


def _mentions(clause: str, f: Mapping[str, Any]) -> bool:
    d = f["data"]
    rx = CONCEPT_MENTIONS.get(str(d.get("concept")))
    if rx is None:
        words = re.findall(r"[A-Za-z]{4,}", str(d.get("label") or d.get("concept") or ""))
        rx = re.escape(words[0]) if words else None
    return bool(rx and re.search(rx, clause, re.I))


def _direction_problem(text: str, concepts: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> str | None:
    """A stated direction (toward / away from invasive carcinoma) must follow the sign of each cited concept's
    contribution: per clause for the features that clause names, else for the whole sentence."""
    def wrong_in(segment: str, fs: Sequence[Mapping[str, Any]]) -> list[str]:
        rest = _without(segment, labels)
        toward, away = bool(_TOWARD_INV.search(rest)), bool(_AWAY_INV.search(rest))
        pushes = [float(f["data"].get("contribution", 0)) for f in fs]
        if toward and away:
            return [str(f["data"].get("concept")) for f in fs] if fs and (all(p > 0 for p in pushes)
                                                                         or all(p < 0 for p in pushes)) else []
        return [str(f["data"].get("concept")) for f, p in zip(fs, pushes) if (toward and p < 0) or (away and p > 0)]

    starts = [0] + [m.start() for m in _DIRECTION_CLAUSE.finditer(text)] + [len(text)]
    clauses = [text[a:b] for a, b in zip(starts, starts[1:]) if text[a:b].strip()]
    wrong: list[str] = []
    if len(clauses) > 1:
        for clause in clauses:
            named = [f for f in concepts if _mentions(clause, f)]
            if named:
                wrong += wrong_in(clause, named)
    wrong = wrong or wrong_in(text, concepts)
    if wrong:
        return (f"{', '.join(dict.fromkeys(wrong))}: the stated direction (toward / away from invasive carcinoma) must "
                f"follow the sign of the contribution")
    return None


_VAGUE_FINDING = re.compile(r"\b(this|the) findings?\b", re.I)    # "associated with this finding": which finding?
_CONTRADICTORY = re.compile(r"\b(scarce|few|fewer|sparse|rare|minimal|absent|reduced|low|little)\s+"
                            r"(frequent|dense|marked|numerous|abundant|extensive)\b", re.I)
_CELL_SUPPORT = re.compile(r"\b(support\w*|favou?r\w*|associated with|consistent with|in keeping with|point\w* (to|toward))\b",
                           re.I)
_CELL_HIGH = re.compile(r"\b(high|higher|highest|increased|elevated|above|large|larger|more|abundant|marked)\b", re.I)
_CELL_LOW = re.compile(r"\b(low|lower|lowest|reduced|decreased|below|small|smaller|less|fewer|few|scarce|no)\b", re.I)
_CLAUSE = re.compile(r"[;,]|\b(but|while|whereas|although|though)\b", re.I)
# Class mentions (BACH ids). Feature names and reference-case phrases that share the words are removed first.
CLASS_PATTERNS = {
    "Normal": r"\bnormal\b",
    "Benign": r"\bbenign\b",
    "InSitu": r"\bin[- ]situ\b|\bDCIS\b|\bLCIS\b",
    "Invasive": r"\binvasive\b",
}
_NOT_A_CLASS = re.compile(r"\bnormal (breast )?lobul\w*|\bbenign (fibrous )?stroma\w*( or fat)?|\bbenign epitheli\w*( \w+)?|"
                          r"\b(normal|benign|in[- ]situ|invasive)( carcinoma)?( \w+)? (reference |retrieved |similar )?"
                          r"cases?\b|\bnon-?invasive\b|\b(benign|normal) reference\b", re.I)
_EXCLUDE = re.compile(r"\b(cannot be (excluded|ruled out)|not (be )?excluded|also (in|within) the\b|remains? possible|"
                      r"also possible|(is|are) also considered)", re.I)
_PRED_HEDGE = re.compile(r"\b(suggest\w*|favou?r\w*|likely|probability|uncertain)\b", re.I)   # hedges the top class


def _classes_in(text: str, codes: Sequence[str] = ()) -> set[str]:
    """BACH class ids named in `text` (feature names and reference-case phrases excluded)."""
    cleaned = _NOT_A_CLASS.sub(" ", _without(text, codes))
    return {c for c, rx in CLASS_PATTERNS.items() if re.search(rx, cleaned, re.I)}


def _prediction_problem(text: str, d: Mapping[str, Any], labels: Sequence[str]) -> str | None:
    """The suggested category is named and hedged; every other confidence-set class is named as not excluded, the
    suggested class never is, and no class outside the set is offered as possible."""
    top, members = d.get("top"), list(d.get("set") or [])
    named = _classes_in(text, labels)
    if top in CLASS_PATTERNS and top not in named:
        return f"the sentence cites the prediction but does not name the suggested category ({top})"
    if not _PRED_HEDGE.search(text):
        return "the suggested category must be hedged ('suggested'), never stated as a diagnosis"
    if not members:
        return None if re.search(r"\b(empty|uncertain)\b", text, re.I) else "the confidence set is empty: say the read is uncertain"
    others = {m for m in members if m != top and m in CLASS_PATTERNS}
    if top not in members:                     # "the set holds X instead, so this read is uncertain"
        missing = sorted(others - named)
        if missing or not re.search(r"\b(uncertain|instead)\b", text, re.I):
            return f"the confidence set holds {sorted(others)} without the suggested category: say so and that the read is uncertain"
        return None
    match = _EXCLUDE.search(text)
    if match:
        starts = [m.end() for m in _CLAUSE.finditer(text, 0, match.start())]
        excluded = _classes_in(text[starts[-1] if starts else 0:match.start()], labels)
        top_rx = CLASS_PATTERNS.get(str(top))
        if top_rx and len(re.findall(top_rx, _NOT_A_CLASS.sub(" ", _without(text, labels)), re.I)) > 1:
            return f"the suggested category ({top}) is listed as 'cannot be excluded'"
        outside = sorted(c for c in excluded if c not in members and c != top)
        if outside:
            return f"{outside} are not in the confidence set: the sentence must not offer them as possible"
    missing = sorted(others - named)
    if missing:
        return f"the confidence set also holds {missing}: the sentence must name them as not excluded"
    if others and not match:
        return "the other confidence-set classes must be named as 'cannot be excluded'"
    return None


def _retrieval_problem(text: str, d: Mapping[str, Any]) -> str | None:
    """Agreement among the most similar reference cases is quoted, never inflated."""
    agree, of = int(d.get("agree", 0)), int(d.get("of", 0))
    count = re.search(rf"\b{agree}\s*(of|/|out of)\s*(the\s+)?{of}\b", text, re.I) or \
        (agree == 0 and re.search(r"\b(none|no)\b", text, re.I))
    if agree < of and re.search(rf"\b{of}\b", text) and not count:
        return f"only {agree} of {of} reference cases share the suggested category: do not quote {of} as agreeing"
    if 2 * agree < of:
        names_category = re.search(r"\b(share|shares|agree\w*|same|match\w*|concordant|suggested|categor\w*|normal|"
                                   r"benign|in[- ]situ|invasive)\b", text, re.I)
        if not count or not names_category:
            return (f"only {agree} of {of} most similar reference cases share the suggested category: quote "
                    f"'{agree} of {of} ... share the suggested category'")
        if re.search(r"\bmost\b(?! similar)|\bmajority\b|\ball\b|\bfavou?r\w*|\bconsistent with|\bsupport\w*", text, re.I):
            return f"only {agree} of {of} reference cases agree: the sentence must not present them as support"
    return None


def _cells_problem(text: str, cells: Sequence[Mapping[str, Any]], aggressive_top: bool | None,
                   with_concepts: bool = False) -> str | None:
    """Cell measurements are read against benign reference cases (ref_percentile): a high word needs a high
    percentile and vice versa, a bare value needs that context, and cells are never support for a non-invasive
    category (cell findings carry no contribution)."""
    if cells and not with_concepts and aggressive_top is False and _CELL_SUPPORT.search(text):
        return "cell findings carry no contribution: the sentence must not present them as support for the category"
    starts = [0] + [m.end() for m in _CLAUSE.finditer(text)] + [len(text)]
    clauses = [text[a:b] for a, b in zip(starts, starts[1:])]
    for f in cells:
        d = f["data"]
        label = str(d.get("label") or _humanize(str(d.get("feature", ""))))
        pct = d.get("ref_percentile")
        longer = [str(g["data"].get("label")) for g in cells if g is not f and label.lower() in str(g["data"].get("label",
                  "")).lower()] + [r for r in ("variation in nuclear size", "variation in nuclear shape")
                                   if label.lower() in r and label.lower() != r]
        for clause in clauses:
            clause = _without(clause, longer)      # "nuclear size" inside "variation in nuclear size" is another feature
            if not re.search(re.escape(label), clause, re.I):
                continue
            rest = _without(clause, [label])
            high, low = bool(_CELL_HIGH.search(rest)), bool(_CELL_LOW.search(rest))
            if pct is not None and high and not low and float(pct) <= 0.1:
                return f"{label} is low against benign reference cases: the sentence must not call it high"
            if pct is not None and low and not high and float(pct) >= 0.9:
                return f"{label} is high against benign reference cases: the sentence must not call it low"
            at = clause.lower().find(label.lower())
            for m in _NUMBER.finditer(clause, at + len(label)):
                n_dec = len(m.group(2) or m.group(3) or "")
                value = float((m.group(1) or "0").replace(",", "") + "." + (m.group(2) or m.group(3) or "0"))
                if d.get("value") is None or m.group(4) or \
                        not number_supported(value, n_dec, False, [(abs(float(d["value"])), False)]):
                    continue
                lead = clause[max(0, at - 30):m.start()]             # "high nuclear size of 80" / "nuclear size of 80"
                if not _CELL_HIGH.search(_without(lead, [label])) and not _CELL_LOW.search(_without(lead, [label])):
                    return f"{label}: a bare value means nothing without its benign-reference percentile"
    return None


def semantic_problem(text: str, cited: Sequence[Mapping[str, Any]], aggressive_top: bool | None = None) -> str | None:
    """Reason the sentence contradicts a cited finding's categorical facts, or None.

    prediction: the suggested category is named and hedged; the other confidence-set classes are named as not
    excluded, the suggested class and classes outside the set never are. margin: the stated call must be the suggested
    call (defer never reads as clear; clear never as involved; tumor at margin names every involved margin;
    tumor-at-margin and clear are hedged as suggestions). quality: a flagged read is never called reliable and vice
    versa. retrieval: minority agreement is quoted as 'N of M ... share the suggested category', never as support,
    and the total is never quoted as agreeing. cells: high/low words follow ref_percentile, bare values need it, and
    cells are never support for a non-invasive category. concept: with every cited feature label removed, a feature
    is called present only when its `level` is above_typical, scarce only when below_typical, and neither when typical
    unless the sentence says typical; no contradictory wording ('scarce frequent mitotic figures'); features that all
    push away from invasive carcinoma are never presented as support for it; and concept support is never re-targeted
    at the suggested category ('associated with this result', 'the strongest feature') unless that category is
    invasive carcinoma and every cited feature pushes toward it ('this finding' is always too vague)."""
    labels = [x for f in cited if isinstance(f.get("data"), Mapping)
              for x in (str(f["data"].get("label", "")), str(f["data"].get("concept", "")).replace("_", " "))]
    rest = _without(text, labels)              # feature names ("Dense lymphocytic infiltrate") are not claims
    for f in cited:
        d = f.get("data") if isinstance(f.get("data"), Mapping) else {}
        kind = f.get("kind")
        if kind == "prediction":
            problem = _prediction_problem(text, d, [x for x in labels if x])
            if problem:
                return problem
        elif kind == "margin":
            call = d.get("suggested_call")
            if call == "defer" and (_CLEAR.search(text) or not _DEFER.search(text)):
                return "margin call is defer: the sentence must defer, never call margins clear"
            if call == "clear" and (_INVOLVED.search(text) or not _CLEAR.search(text)):
                return "margin call is clear: the sentence must say clear, never involved"
            if call in ("tumor_at_margin", "clear") and not _HEDGE.search(text):
                return "margin calls are simulated suggestions: say 'suggested', never state them as fact"
            if call == "tumor_at_margin":
                involved = list(d.get("involved") or [])
                every = set(involved) >= set(MARGIN_ORDER) and re.search(r"\ball (four |4 )?margins\b", text, re.I)
                missing = [] if every else [m for m in involved if not re.search(rf"\b{m}\b", text, re.I)]
                extra = [m for m in MARGIN_ORDER if m not in involved and re.search(rf"\b{m}\b", text, re.I)]
                if missing or extra or _DEFER.search(text):
                    return (f"margin call is tumor at margin: the sentence must name exactly "
                            f"{[m for m in MARGIN_ORDER if m in involved]}")
        elif kind == "quality":
            flagged = bool(d.get("reliability_flag"))
            if flagged and _RELIABLE.search(_UNRELIABLE.sub(" ", text)):
                return "reliability flag is raised: the sentence must not call the read reliable"
            if not flagged and _UNRELIABLE.search(text):
                return "reliability flag is not raised: the sentence must not call the read unreliable"
        elif kind == "retrieval":
            problem = _retrieval_problem(text, d)
            if problem:
                return problem
    cells = [f for f in cited if f.get("kind") == "cells" and isinstance(f.get("data"), Mapping)]
    if cells:
        problem = _cells_problem(text, cells, aggressive_top,
                                 with_concepts=any(f.get("kind") == "concept" for f in cited))
        if problem:
            return problem
    concepts = [f for f in cited if f.get("kind") == "concept" and isinstance(f.get("data"), Mapping)]
    if not concepts:
        return None
    if _CONTRADICTORY.search(text):
        return f"contradictory wording {_CONTRADICTORY.search(text).group(0)!r}: drop the label's quantity word"
    for f in concepts:                         # "Frequent mitotic figures are scarce": the label's own word says present
        label = str(f["data"].get("label", ""))
        if f["data"].get("level") == "below_typical" and _QUANTITY_WORD.match(label) \
                and re.search(re.escape(label), text, re.I):
            return f"{label!r} is scarce here: drop the label's quantity word ('scarce {_QUANTITY_WORD.sub('', label).lower()}')"
    pushes = [float(f["data"].get("contribution", 0)) for f in concepts]
    if all(p < 0 for p in pushes) and (aggressive_top or re.search(r"\binvasive\b", rest, re.I)) \
            and _SUPPORT.search(rest) and not _CONTRAST.search(rest):
        return "the cited features push away from invasive carcinoma: the sentence must not present them as support"
    direction = _direction_problem(text, concepts, labels)
    if direction:
        return direction
    deictic = _DEICTIC.search(rest)
    if deictic and (_VAGUE_FINDING.search(deictic.group(0)) or not aggressive_top or any(p < 0 for p in pushes)):
        return (f"{deictic.group(0)!r}: concept contributions measure invasive carcinoma only; name the direction "
                f"('toward invasive carcinoma', 'a lower likelihood of invasive carcinoma')")
    levels = {f["data"].get("level") for f in concepts if f["data"].get("level")}
    names = ", ".join(str(f["data"].get("concept")) for f in concepts)
    if levels:                                 # a presence word needs a cited concept above typical, and vice versa
        present, absent = bool(_PRESENT.search(rest)), bool(_ABSENT.search(rest))
        if present and not absent and levels == {"below_typical"}:
            return f"{names} below typical here: the sentence must not call it present"
        if absent and not present and levels == {"above_typical"}:
            return f"{names} above typical here: the sentence must not call it scarce"
        if (present or absent) and levels == {"typical"} and not re.search(r"\btypical\b", rest, re.I):
            return f"{names} typical here: the sentence must not call it {'present' if present else 'scarce'}"
    return None


def required_findings(findings: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Findings every summary must cite: the prediction (suggested category and confidence set), the margin call, and
    a quality concern (reliability flag or too few tiles)."""
    out = [f for f in findings if f["kind"] == "prediction"][:1]
    out += [f for f in findings if f["kind"] == "margin"][:1]
    for f in findings:
        if f["kind"] == "quality" and _quality_sentence(f)[1]:
            out.append(f)
            break
    return out


def _concept_backfill(findings: Sequence[Mapping[str, Any]], cited: set[str], cfg: Mapping[str, Any]
                      ) -> dict[str, Any] | None:
    """Template sentence for the strongest concept's direction when no kept sentence cites that concept (the features
    of that direction no kept sentence cites), or None."""
    concepts = [f for f in findings if f["kind"] == "concept" and float(f["data"]["contribution"]) != 0]
    if not concepts:
        return None
    strongest = max(concepts, key=lambda f: abs(float(f["data"]["contribution"])))
    if strongest["id"] in cited:
        return None
    up = float(strongest["data"]["contribution"]) > 0
    fs = [f for f in concepts if (float(f["data"]["contribution"]) > 0) == up and f["id"] not in cited]
    text = _concept_up_sentence(fs, cfg) if up else _concept_down_sentence(fs, cfg)
    return {"text": text, "findings": [f["id"] for f in fs]}


def ensure_required(kept: list[dict[str, Any]], findings: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
                    ) -> tuple[list[dict[str, Any]], int]:
    """Add the template sentence for each required finding no kept sentence cites (the prediction first, the margin
    and quality concern last), dropping trailing sentences that cite no required finding to stay within
    summary.max_sentences; then, when no kept sentence cites the strongest concept, insert the template concept sentence
    for its direction after the prediction (replacing the last similar-cases or cells sentence when there is no room).
    Returns (sentences, n_template_sentences)."""
    out = list(kept)
    need = required_findings(findings)
    req_ids = {f["id"] for f in need}
    added = 0
    for f in need:
        if any(f["id"] in s["findings"] for s in out):
            continue
        if f["kind"] == "prediction":
            text = _prediction_sentence(f, cfg)
        elif f["kind"] == "margin":
            text = _margin_sentence(f)
        else:
            text = _quality_sentence(f)[0]
        while len(out) >= int(cfg["summary"]["max_sentences"]):
            drop = next((i for i in range(len(out) - 1, -1, -1) if not req_ids & set(out[i]["findings"])), None)
            if drop is None:
                break
            out.pop(drop)
        sentence = {"text": text, "findings": [f["id"]]}
        if f["kind"] == "prediction":
            out.insert(0, sentence)
        else:
            out.append(sentence)
        added += 1
    backfill = _concept_backfill(findings, {i for s in out for i in s["findings"]}, cfg)
    if backfill is not None and len(out) >= int(cfg["summary"]["max_sentences"]):
        kinds = {f["id"]: f["kind"] for f in findings}           # a concept outranks similar cases and cells
        low = next((i for i in range(len(out) - 1, -1, -1)
                    if all(kinds.get(x) in ("retrieval", "cells") for x in out[i]["findings"])), None)
        if low is None:
            backfill = None
        else:
            out.pop(low)
    if backfill is not None:
        pred_ids = {f["id"] for f in findings if f["kind"] == "prediction"}
        at = next((i + 1 for i in range(len(out) - 1, -1, -1) if pred_ids & set(out[i]["findings"])), 0)
        out.insert(at, backfill)
        added += 1
    return out, added


def check_sentence(sentence: Any, by_id: Mapping[str, Mapping[str, Any]], banned: Sequence[re.Pattern[str]],
                   codes: Sequence[str] = (), aggressive_top: bool | None = None) -> str | None:
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
    codes = list(codes) + [str(by_id[f]["data"]["label"]) for f in cited
                           if isinstance(by_id[f].get("data"), Mapping) and by_id[f]["data"].get("label")]
    odd = unknown_tokens(text, codes)
    if odd:
        return f"unrecognized number word or code {odd[0]!r}"
    allowed = [x for f in cited for x in finding_numbers(by_id[f].get("data"))]
    for value, decimals, percent in numbers_in_text(text, codes):
        shown = f"{value:.{decimals}f}{'%' if percent else ''}"
        if not number_supported(value, decimals, percent, allowed):
            return f"number {shown} is not in the cited findings"
        if decimals > 2 or (decimals and value >= 100):
            return f"number {shown} is not rounded for reading (at most two decimals; whole numbers from 100)"
    if re.search(r"\bonly 0(\.0+)?\s*%", text, re.I):
        return "'only 0%' reads as a slip: say 'no benign reference case ...'"
    return semantic_problem(text, [by_id[f] for f in cited], aggressive_top)


def validate(sentences: Any, findings: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, dropped): kept sentences as {"text", "findings"} (at most summary.max_sentences), dropped with reasons."""
    by_id = {f["id"]: f for f in findings}
    banned = _banned(cfg)
    codes = list(cfg["summary"]["allowed_codes"])
    top = next((f["data"].get("top") for f in findings if f.get("kind") == "prediction"), None)
    aggressive_top = top == cfg["aggressive_class"] if top is not None else None
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for sentence in sentences if isinstance(sentences, list) else []:
        reason = check_sentence(sentence, by_id, banned, codes, aggressive_top)
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


_QUANTITY_WORD = re.compile(r"^(frequent|dense|marked|numerous|extensive|abundant)\s+", re.I)


def _concept_phrase(f: Mapping[str, Any]) -> str:
    """Feature name with its level against training cases: 'prominent tumor infiltrating stroma', 'frequent mitotic
    figures' (the label's own quantity word), 'scarce normal lobules ...', 'lymphocytic infiltrate (typical level)'."""
    label = _concept_label(f)
    level = f["data"].get("level")
    if level == "above_typical":
        return label if _QUANTITY_WORD.match(label) else f"prominent {label}"
    if level == "below_typical":
        return "scarce " + _QUANTITY_WORD.sub("", label)
    if level == "typical":
        return f"{_QUANTITY_WORD.sub('', label)} (typical level)"
    return label


def _with_regions(phrase: str, where: str) -> str:
    """'tumor infiltrating stroma (regions 1 and 2)'; 'dense lymphocytic infiltrate (TILs; region 7)'."""
    if not where:
        return phrase
    return f"{phrase[:-1]}; {where})" if phrase.endswith(")") else f"{phrase} ({where})"


def _concept_up_sentence(fs: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> str:
    aggressive = _class_label(cfg, cfg["aggressive_class"])
    where = _regions(fs[0]["data"]["tiles"])
    text = f"Strongest push toward {aggressive} in this read: {_with_regions(_concept_phrase(fs[0]), where)}"
    if len(fs) > 1:
        text += ", then " + _join([_concept_phrase(f) for f in fs[1:]])
    return text + "."


def _concept_down_sentence(fs: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> str:
    aggressive = _class_label(cfg, cfg["aggressive_class"])
    where = _regions(fs[0]["data"]["tiles"]) if len(fs) == 1 else ""
    noun = "Feature" if len(fs) == 1 else "Features"
    return (f"{noun} lowering the likelihood of {aggressive} in this read: "
            f"{_with_regions(_join([_concept_phrase(f) for f in fs]), where)}.")


def _cells_sentence(f: Mapping[str, Any]) -> str:
    d = f["data"]
    label = str(d.get("label") or _humanize(d["feature"]))
    where = _regions(d.get("tiles"))
    suffix = f" ({where})" if where else ""
    pct = d.get("ref_percentile")
    if pct is None:
        return f"{label[:1].upper() + label[1:]} was measured in this read{suffix}."
    pct = float(pct)
    if pct >= 0.5:
        return f"{label[:1].upper() + label[1:]} is higher than in {round(100 * pct)}% of benign reference cases{suffix}."
    if round(100 * pct) == 0:
        return f"No benign reference case has a lower {_lower_first(label)}{suffix}."
    return f"Only {round(100 * pct)}% of benign reference cases have a lower {_lower_first(label)}{suffix}."


def _retrieval_sentence(f: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    d = f["data"]
    verb = "shares" if int(d["agree"]) == 1 else "share"
    return (f"{int(d['agree'])} of {int(d['of'])} most similar reference cases {verb} the suggested category "
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
    "1. Write at most {max_sentences} short sentences in plain clinical English for a pathologist. No filler: every "
    "sentence states a specific fact from the findings it cites.\n"
    "2. Every sentence lists the IDs of the findings it uses in its \"findings\" array, and only those findings. "
    "Never write finding IDs (F1, F2 ...) inside the text itself.\n"
    "3. Use only numbers that appear in the cited findings, written as digits and rounded for reading: probabilities "
    "as a whole percent (72%) or two decimals (0.72), other measurements to at most two decimals, densities and "
    "counts as whole numbers. Describe concept values and contributions in words rather than quoting them; never "
    "quote reliability or ood_percentile.\n"
    "4. Order: first the suggested category (the prediction's top class, e.g. 'In situ carcinoma is suggested' or 'A "
    "benign lesion is suggested') and each OTHER class in the confidence set as 'cannot be excluded' (never the "
    "suggested class itself, never a class outside the set; an empty set, or a set without the top class, means the "
    "read is uncertain); then the concept features that move the read most; then context (a quality concern, "
    "similar reference cases, cells, the margin).\n"
    "5. Never state a diagnosis as fact: say 'suggested', 'cannot be excluded'. Never claim a feature causes "
    "anything. Never mention a grade.\n"
    "6. Concept findings measure the likelihood of invasive carcinoma only, never support for the suggested category. "
    "'level' compares this case with training cases: above_typical = prominent, typical = typical, below_typical = "
    "scarce. 'contribution' > 0 pushes the read toward invasive carcinoma, < 0 away from it. Name that direction, "
    "e.g. 'Prominent tumor infiltrating stroma pushes the read toward invasive carcinoma.', 'Scarce normal lobules "
    "push the read toward invasive carcinoma.', 'Prominent gland / tubule formation is associated with a lower "
    "likelihood of invasive carcinoma.' Never write 'associated with this finding', 'this result', 'this category', "
    "'supports the result' or 'the strongest feature'. When a label already holds a quantity word (Frequent mitotic "
    "figures, Dense lymphocytic infiltrate, Marked nuclear pleomorphism), drop that word: 'frequent mitotic figures' "
    "when prominent, 'scarce mitotic figures' when scarce, never 'scarce frequent mitotic figures'.\n"
    "7. Similar reference cases: quote both numbers, e.g. '1 of 5 most similar reference cases shares the suggested "
    "category'. Never say the case resembles reference cases of a class unless most of them share it.\n"
    "8. Cell findings: compare with benign reference cases using ref_percentile, e.g. 'Tumor cell fraction is higher "
    "than in 97% of benign reference cases' (ref_percentile >= 0.5) or 'Only 4% of benign reference cases have a "
    "lower nuclear size' (ref_percentile < 0.5; when that rounds to 0%, say 'no benign reference case has a lower "
    "nuclear size'). Never present a cell measurement as support for the suggested category.\n"
    "9. Surgical cases are simulated frozen sections. Write the margin finding as 'Suggested margin call (simulated "
    "frozen section): ...' with its suggested_call: tumor_at_margin names every involved margin ('tumor at the "
    "superior and medial margins'); clear says 'margins clear'; defer says 'defer to permanent sections', because "
    "the reliability check flagged the read.\n"
    "10. If a quality finding has reliability_flag true, say the reliability check flagged the read, so its "
    "confidence is reduced; if qc_tiles_kept < min_tiles_kept, say little tissue passed quality control.\n"
    'Answer with JSON only: {{"sentences": [{{"text": "...", "findings": ["F1"]}}]}}'
)


GROUND_TRUTH_KEYS = frozenset({"true_label", "eval_only", "split", "group", "patient_id"})


_INLINE_IDS = re.compile(r"\s*[\[(]\s*F[1-9][0-9]*(?:\s*[,;]\s*F[1-9][0-9]*)*\s*[\])]")


def strip_inline_ids(text: str) -> str:
    """Remove inline citations like ' [F1]' or ' (F2, F4)': the finding IDs live in the sentence's findings list and
    the UI renders them as chips."""
    out = _INLINE_IDS.sub("", text)
    return re.sub(r"\s+([.,;:])", r"\1", out).strip()


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
    kept = [{**k, "text": strip_inline_ids(k["text"])} for k in kept]
    kept, added = ensure_required(kept, findings, cfg)
    info["template_sentences"] = added
    generator = f"lumen:{reply['model']}" + ("+template" if added else "")
    return {"generator": generator, "sentences": kept}, {**info, "generator": generator, "path": "llm", "reason": ""}
