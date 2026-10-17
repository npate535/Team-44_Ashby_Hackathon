"""SQLite store for the demo backend: case loading, the agent/decision state machine, routing and the event log.

Tables (plan §10):
    cases           one row per loaded case: the CasePublic JSON (eval_only already dropped) + live state
                    (arrival, agent status, surgical timer, sign-out, margin call)
    orders          the drafted orders of each case (drafted items kept, so edits can only remove items)
    decisions       every accepted pathologist action with its payload
    routed_results  results sent to the surgeon (margin calls) and the oncologist (sign-outs)
    events          timestamped log of every transition (arrivals, injections, decisions, routing, resets)

Loading: every *.json in the cases folder is validated with core.contract.Case; invalid files are logged and skipped
(never served). Only the public form is stored, so ground truth never enters the database.

State machine (plan §9, BUILD_PLAN Task 6). Agent status: none -> drafted -> approved | edited_and_approved |
deferred_closer_look | dismissed (deferred_closer_look may still go to approved / edited_and_approved / dismissed).

    biopsy   drafted               approve_orders, edit_orders, closer_look, dismiss_flag
             deferred_closer_look  approve_orders, edit_orders, dismiss_flag
             none | approved | edited_and_approved | dismissed   sign_out (routes the summary to the oncologist)
             signed out            nothing
    surgical not yet called        margin_call (routes to the surgeon, stops the timer); stains are never ordered
             called                nothing

Anything else is a TransitionError (HTTP 409); a malformed payload is a PayloadError (HTTP 422). edit_orders may only
keep items the rules drafted for that rule_id (no free-text stains). Times: `wall` is epoch seconds (surgical timers,
event stamps); `sim` is seconds on the simulator clock (arrivals, aging).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from core import contract

log = logging.getLogger("pathqueue.backend")

ACTIONS = ("approve_orders", "edit_orders", "closer_look", "dismiss_flag", "sign_out", "margin_call")
AGENT_ACTIONS = ("approve_orders", "edit_orders", "closer_look", "dismiss_flag")
# agent status -> actions allowed from it (biopsy lane, not signed out)
_BIOPSY_ALLOWED: dict[str, tuple[str, ...]] = {
    "drafted": ("approve_orders", "edit_orders", "closer_look", "dismiss_flag"),
    "deferred_closer_look": ("approve_orders", "edit_orders", "dismiss_flag"),
    "none": ("sign_out",),
    "approved": ("sign_out",),
    "edited_and_approved": ("sign_out",),
    "dismissed": ("sign_out",),
}
_AGENT_TARGET = {
    "approve_orders": "approved",
    "edit_orders": "edited_and_approved",
    "closer_look": "deferred_closer_look",
    "dismiss_flag": "dismissed",
}
# order row status after each agent action
_ORDER_STATUS = {"approve_orders": "approved", "dismiss_flag": "dismissed"}

SCHEMA = """
CREATE TABLE cases (
    case_id            TEXT PRIMARY KEY,
    lane               TEXT NOT NULL,
    variant            TEXT,
    arrived_offset_s   REAL NOT NULL,
    public_json        TEXT NOT NULL,
    source_file        TEXT NOT NULL,
    agent_status       TEXT NOT NULL,
    arrived            INTEGER NOT NULL DEFAULT 0,
    injected           INTEGER NOT NULL DEFAULT 0,
    arrived_sim_s      REAL,
    arrived_wall       REAL,
    timer_started_wall REAL,
    timer_stopped_wall REAL,
    signed_out_wall    REAL,
    margin_call_json   TEXT,
    margin_called_wall REAL,
    done_sim_s         REAL
);
CREATE TABLE orders (
    case_id       TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    type          TEXT NOT NULL,
    rule_id       TEXT NOT NULL,
    rationale     TEXT NOT NULL,
    drafted_items TEXT NOT NULL,
    items         TEXT NOT NULL,
    status        TEXT NOT NULL,
    PRIMARY KEY (case_id, seq)
);
CREATE TABLE decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status   TEXT NOT NULL,
    at_wall     REAL NOT NULL,
    sim_time_s  REAL NOT NULL
);
CREATE TABLE routed_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    to_role    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    at_wall    REAL NOT NULL,
    sim_time_s REAL NOT NULL
);
CREATE TABLE events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT,
    type       TEXT NOT NULL,
    from_state TEXT,
    to_state   TEXT,
    action     TEXT,
    payload    TEXT NOT NULL,
    at_wall    REAL NOT NULL,
    sim_time_s REAL NOT NULL
);
CREATE INDEX events_case ON events (case_id, id);
CREATE INDEX routed_role ON routed_results (to_role, id);
"""
TABLES = ("cases", "orders", "decisions", "routed_results", "events")
ROUTE_ROLES = ("surgeon", "oncologist")


class StoreError(Exception):
    status = 400


class NotFoundError(StoreError):
    status = 404


class TransitionError(StoreError):
    status = 409


class PayloadError(StoreError):
    status = 422


class ServerDataError(StoreError):
    """A file on the server side (artifacts) is malformed: not the client's fault."""
    status = 500


@dataclass(frozen=True)
class LoadResult:
    cases: list[dict[str, Any]]           # CasePublic dicts (no eval_only), sorted by case_id
    sources: dict[str, str]               # case_id -> file name
    skipped: list[dict[str, str]]         # [{file, error}] invalid or duplicate files


def load_case_dir(cases_dir: str | Path) -> LoadResult:
    """Validate every *.json in cases_dir with core.contract.Case; return the public forms + what was skipped."""
    folder = Path(cases_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"cases folder {folder} does not exist")
    cases: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    for path in sorted(folder.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            public = contract.to_public(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            reason = _short_error(exc)
            log.warning("skipping invalid case file %s: %s", path.name, reason)
            skipped.append({"file": path.name, "error": reason})
            continue
        case_id = public["case_id"]
        stains = [o["rule_id"] for o in public["agent"]["orders"] if o["type"] == "stain"]
        if public["lane"] == "surgical" and stains:
            reason = f"stain orders {stains} drafted for a simulated frozen section (never allowed in the surgical lane)"
            log.warning("skipping %s: %s", path.name, reason)
            skipped.append({"file": path.name, "error": reason})
            continue
        if case_id in cases:
            log.warning("skipping %s: duplicate case_id %s (already loaded from %s)", path.name, case_id,
                        sources[case_id])
            skipped.append({"file": path.name, "error": f"duplicate case_id {case_id}"})
            continue
        cases[case_id] = public
        sources[case_id] = path.name
    ordered = [cases[k] for k in sorted(cases)]
    return LoadResult(cases=ordered, sources=sources, skipped=skipped)


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        where = ".".join(str(p) for p in first.get("loc", ())) or "case"
        more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
        return f"{where}: {first.get('msg', str(exc))}{more}"
    return f"{type(exc).__name__}: {exc}"


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class Store:
    """Thread-safe SQLite store. Every public method takes the lock and runs in one transaction."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- seeding ------------------------------------------------------------------------------------------------
    def seed(self, loaded: LoadResult, *, wall: float, sim: float = 0.0) -> None:
        """Drop and recreate every table, then insert the cases and their drafted orders (reset semantics)."""
        with self._lock, self._tx() as cur:
            for table in TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            # executescript() would commit mid-transaction, so run the statements one by one
            for statement in filter(None, (s.strip() for s in SCHEMA.split(";"))):
                cur.execute(statement)
            for case in loaded.cases:
                agent = case["agent"]
                cur.execute(
                    "INSERT INTO cases (case_id, lane, variant, arrived_offset_s, public_json, source_file, agent_status)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (case["case_id"], case["lane"], case.get("simulated_variant"), float(case["arrived_offset_s"]),
                     _dumps(case), loaded.sources[case["case_id"]], agent["status"]),
                )
                for seq, order in enumerate(agent.get("orders") or []):
                    items = list(order["items"])
                    cur.execute(
                        "INSERT INTO orders (case_id, seq, type, rule_id, rationale, drafted_items, items, status)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (case["case_id"], seq, order["type"], order["rule_id"], order["rationale"], _dumps(items),
                         _dumps(items), _initial_order_status(agent["status"])),
                    )
            self._event(cur, None, "reset", payload={"cases": len(loaded.cases), "skipped": len(loaded.skipped)},
                        wall=wall, sim=sim)

    # ---- transactions -------------------------------------------------------------------------------------------
    class _Tx:
        def __init__(self, conn: sqlite3.Connection):
            self.conn = conn

        def __enter__(self) -> sqlite3.Cursor:
            self.conn.execute("BEGIN IMMEDIATE")
            self.cur = self.conn.cursor()
            return self.cur

        def __exit__(self, exc_type, exc, tb) -> None:
            if exc_type is None:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute("ROLLBACK")

    def _tx(self) -> "Store._Tx":
        return Store._Tx(self._conn)

    def _event(self, cur: sqlite3.Cursor, case_id: str | None, type_: str, *, wall: float, sim: float,
               from_state: str | None = None, to_state: str | None = None, action: str | None = None,
               payload: Mapping[str, Any] | None = None) -> None:
        cur.execute(
            "INSERT INTO events (case_id, type, from_state, to_state, action, payload, at_wall, sim_time_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (case_id, type_, from_state, to_state, action, _dumps(dict(payload or {})), wall, sim),
        )

    # ---- reads ----------------------------------------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(arrived), 0) AS arrived FROM cases").fetchone()
            return {"cases": int(row["n"]), "arrived": int(row["arrived"])}

    def rows(self, *, arrived_only: bool = True, lane: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM cases"
        clauses, params = [], []
        if arrived_only:
            clauses.append("arrived = 1")
        if lane is not None:
            clauses.append("lane = ?")
            params.append(lane)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            return list(self._conn.execute(sql + " ORDER BY case_id", params).fetchall())

    def row(self, case_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()

    def arrived_row(self, case_id: str) -> sqlite3.Row:
        row = self.row(case_id)
        if row is None or not row["arrived"]:
            raise NotFoundError(f"case {case_id} not found (unknown, or not arrived yet)")
        return row

    def pending_rows(self) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(
                "SELECT * FROM cases WHERE arrived = 0 ORDER BY arrived_offset_s, case_id").fetchall())

    def orders(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM orders WHERE case_id = ? ORDER BY seq", (case_id,)).fetchall()
        return [
            {"seq": r["seq"], "type": r["type"], "rule_id": r["rule_id"], "rationale": r["rationale"],
             "drafted_items": json.loads(r["drafted_items"]), "items": json.loads(r["items"]), "status": r["status"]}
            for r in rows
        ]

    def events(self, case_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events" + (" WHERE case_id = ?" if case_id is not None else "") + " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, (case_id,) if case_id is not None else ()).fetchall()
        return [
            {"id": r["id"], "case_id": r["case_id"], "type": r["type"], "from": r["from_state"], "to": r["to_state"],
             "action": r["action"], "payload": json.loads(r["payload"]), "at_wall": r["at_wall"],
             "sim_time_s": r["sim_time_s"]}
            for r in rows
        ]

    def decisions(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM decisions WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
        return [
            {"id": r["id"], "action": r["action"], "payload": json.loads(r["payload"]), "from": r["from_status"],
             "to": r["to_status"], "at_wall": r["at_wall"], "sim_time_s": r["sim_time_s"]}
            for r in rows
        ]

    def routed(self, to_role: str) -> list[dict[str, Any]]:
        if to_role not in ROUTE_ROLES:
            raise PayloadError(f"unknown recipient {to_role!r}; expected one of {list(ROUTE_ROLES)}")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM routed_results WHERE to_role = ? ORDER BY id DESC", (to_role,)).fetchall()
        return [
            {"id": r["id"], "case_id": r["case_id"], "to": r["to_role"], "kind": r["kind"], "at_wall": r["at_wall"],
             "sim_time_s": r["sim_time_s"], **json.loads(r["payload"])}
            for r in rows
        ]

    # ---- arrivals (used by backend.simulator) ------------------------------------------------------------------------
    def mark_arrived(self, case_id: str, *, arrived_sim: float, arrived_wall: float, injected: bool, wall: float,
                     sim: float) -> None:
        with self._lock, self._tx() as cur:
            row = cur.execute("SELECT lane, arrived FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"case {case_id} not found")
            if row["arrived"]:
                raise TransitionError(f"case {case_id} has already arrived")
            timer = arrived_wall if row["lane"] == "surgical" else None
            cur.execute(
                "UPDATE cases SET arrived = 1, injected = ?, arrived_sim_s = ?, arrived_wall = ?, timer_started_wall = ?"
                " WHERE case_id = ?",
                (int(injected), arrived_sim, arrived_wall, timer, case_id),
            )
            self._event(cur, case_id, "injected" if injected else "arrived", wall=wall, sim=sim,
                        payload={"arrived_sim_s": arrived_sim, "lane": row["lane"]})
            if timer is not None:
                self._event(cur, case_id, "timer_started", wall=wall, sim=sim, payload={"started_wall": timer})

    def log_event(self, case_id: str | None, type_: str, *, wall: float, sim: float,
                  payload: Mapping[str, Any] | None = None) -> None:
        with self._lock, self._tx() as cur:
            self._event(cur, case_id, type_, wall=wall, sim=sim, payload=payload)

    # ---- decisions ----------------------------------------------------------------------------------------------------
    def decide(self, case_id: str, action: str, payload: Mapping[str, Any] | None, *, wall: float,
               sim: float) -> dict[str, Any]:
        """Apply one pathologist action. Returns {case_id, action, from, to}. Raises StoreError subclasses."""
        if action not in ACTIONS:
            raise PayloadError(f"unknown action {action!r}; expected one of {list(ACTIONS)}")
        payload = dict(payload or {})
        with self._lock, self._tx() as cur:
            row = cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None or not row["arrived"]:
                raise NotFoundError(f"case {case_id} not found (unknown, or not arrived yet)")
            allowed = allowed_actions(row)
            if action not in allowed:
                raise TransitionError(_refusal(row, action))
            status = row["agent_status"]
            if action == "margin_call":
                result = self._margin_call(cur, row, payload, wall=wall, sim=sim)
            elif action == "sign_out":
                result = self._sign_out(cur, row, payload, wall=wall, sim=sim)
            else:
                result = self._agent_action(cur, row, action, payload, wall=wall, sim=sim)
            cur.execute(
                "INSERT INTO decisions (case_id, action, payload, from_status, to_status, at_wall, sim_time_s)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (case_id, action, _dumps(result["payload"]), status, result["to"], wall, sim),
            )
            return {"case_id": case_id, "action": action, "from": status, "to": result["to"]}

    def _agent_action(self, cur: sqlite3.Cursor, row: sqlite3.Row, action: str, payload: dict[str, Any], *,
                      wall: float, sim: float) -> dict[str, Any]:
        case_id, status = row["case_id"], row["agent_status"]
        target = _AGENT_TARGET[action]
        orders = cur.execute("SELECT * FROM orders WHERE case_id = ? ORDER BY seq", (case_id,)).fetchall()
        if action == "edit_orders":
            kept = _validate_edit(payload, orders)
            for order in orders:
                items = kept.get(order["seq"], [])
                cur.execute("UPDATE orders SET items = ?, status = ? WHERE case_id = ? AND seq = ?",
                            (_dumps(items), "edited" if items else "removed", case_id, order["seq"]))
            by_seq = {o["seq"]: o["rule_id"] for o in orders}
            payload = {"orders": [{"seq": s, "rule_id": by_seq[s], "items": i} for s, i in kept.items()]}
        else:
            if payload:
                raise PayloadError(f"{action} takes no payload")
            if action in _ORDER_STATUS:
                cur.execute("UPDATE orders SET status = ? WHERE case_id = ?", (_ORDER_STATUS[action], case_id))
        cur.execute("UPDATE cases SET agent_status = ? WHERE case_id = ?", (target, case_id))
        self._event(cur, case_id, "agent_transition", from_state=status, to_state=target, action=action,
                    payload=payload, wall=wall, sim=sim)
        return {"to": target, "payload": payload}

    def _sign_out(self, cur: sqlite3.Cursor, row: sqlite3.Row, payload: dict[str, Any], *, wall: float,
                  sim: float) -> dict[str, Any]:
        if payload:
            raise PayloadError("sign_out takes no payload")
        case_id, status = row["case_id"], row["agent_status"]
        case = json.loads(row["public_json"])
        orders = cur.execute("SELECT * FROM orders WHERE case_id = ? ORDER BY seq", (case_id,)).fetchall()
        placed = [
            {"type": o["type"], "rule_id": o["rule_id"], "items": json.loads(o["items"])}
            for o in orders if o["status"] in ("approved", "edited")
        ]
        routed = {
            "agent_status": status,
            "orders": placed,
            "summary": case["summary"],
            "suggestion": {
                "classes": case["prediction"]["classes"],
                "probs": case["prediction"]["probs"],
                "risk_score": case["prediction"]["risk_score"],
                "conformal_set": case["prediction"]["conformal_set"],
                "conformal_alpha": case["prediction"]["conformal_alpha"],
            },
            "synthetic": case.get("synthetic", False),
        }
        cur.execute("UPDATE cases SET signed_out_wall = ?, done_sim_s = ? WHERE case_id = ?", (wall, sim, case_id))
        self._route(cur, case_id, "oncologist", "sign_out", routed, wall=wall, sim=sim)
        self._event(cur, case_id, "signed_out", from_state=status, to_state=status, action="sign_out",
                    payload={"orders": placed}, wall=wall, sim=sim)
        return {"to": status, "payload": {}}

    def _margin_call(self, cur: sqlite3.Cursor, row: sqlite3.Row, payload: dict[str, Any], *, wall: float,
                     sim: float) -> dict[str, Any]:
        case_id, status = row["case_id"], row["agent_status"]
        call, margins = _validate_margin_call(payload)
        started = row["timer_started_wall"]
        elapsed = None if started is None else max(0.0, wall - started)
        record = {"call": call, "margins": margins}
        cur.execute(
            "UPDATE cases SET margin_call_json = ?, margin_called_wall = ?, timer_stopped_wall = ?, done_sim_s = ?"
            " WHERE case_id = ?",
            (_dumps(record), wall, wall if started is not None else None, sim, case_id),
        )
        case = json.loads(row["public_json"])
        routed = {**record, "timer_elapsed_s": elapsed, "simulated": True,
                  "suggested_call": case["margin"]["suggested_call"],   # surgical cases always carry a margin
                  "simulated_variant": case.get("simulated_variant"), "synthetic": case.get("synthetic", False)}
        self._route(cur, case_id, "surgeon", "margin_call", routed, wall=wall, sim=sim)
        self._event(cur, case_id, "margin_call", action="margin_call", payload=record, wall=wall, sim=sim)
        if started is not None:
            self._event(cur, case_id, "timer_stopped", wall=wall, sim=sim, payload={"elapsed_s": elapsed})
        return {"to": status, "payload": record}

    def _route(self, cur: sqlite3.Cursor, case_id: str, to_role: str, kind: str, payload: Mapping[str, Any], *,
               wall: float, sim: float) -> None:
        cur.execute(
            "INSERT INTO routed_results (case_id, to_role, kind, payload, at_wall, sim_time_s) VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, to_role, kind, _dumps(dict(payload)), wall, sim),
        )
        self._event(cur, case_id, "routed", action=kind, payload={"to": to_role}, wall=wall, sim=sim)


def _initial_order_status(agent_status: str) -> str:
    return {"approved": "approved", "edited_and_approved": "edited", "dismissed": "dismissed"}.get(agent_status,
                                                                                                    "drafted")


def is_done(row: Mapping[str, Any]) -> bool:
    """A case leaves the open worklist once it is signed out (biopsy) or its margin call is sent (surgical)."""
    return row["signed_out_wall"] is not None or row["margin_call_json"] is not None


def allowed_actions(row: Mapping[str, Any]) -> list[str]:
    """The single source of truth for what a pathologist may do next (the API sends it as runtime.allowed_actions).

    Surgical lane: the margin call until it is sent, plus decisions on drafted non-stain orders (e.g. a recut for a
    reliability-flagged read); stains never reach this lane (load_case_dir rejects them) and there is no sign-out.
    """
    if not row["arrived"]:
        return []
    if row["lane"] == "surgical":
        agent = [a for a in _BIOPSY_ALLOWED[row["agent_status"]] if a in AGENT_ACTIONS]
        return ([] if row["margin_call_json"] is not None else ["margin_call"]) + agent
    if is_done(row):
        return []
    return list(_BIOPSY_ALLOWED[row["agent_status"]])


def _refusal(row: Mapping[str, Any], action: str) -> str:
    status = row["agent_status"]
    if row["lane"] == "surgical":
        if action == "margin_call":
            return "the margin call for this simulated frozen section was already sent"
        if action in AGENT_ACTIONS:
            return (f"{action} is not allowed in agent status {status}; stains are never placed in the simulated "
                    "frozen-section lane")
        return f"{action} does not apply to the surgical lane; make a margin call"
    if row["signed_out_wall"] is not None:
        return "case already signed out"
    if action == "margin_call":
        return "margin calls apply only to simulated frozen sections (surgical lane)"
    if action == "sign_out":
        return f"act on the drafted orders first (agent status {status})"
    return f"{action} is not allowed in agent status {status}"


def _validate_edit(payload: Mapping[str, Any], orders: Sequence[sqlite3.Row]) -> dict[int, list[str]]:
    """edit_orders payload {orders: [{seq, rule_id?, items}]} -> {seq: kept items}. Items must be drafted ones.

    Orders are addressed by `seq` (their position in the drafted list: one rule may draft several orders); an optional
    rule_id must match that order's rule. Orders left out are removed.
    """
    if set(payload) != {"orders"} or not isinstance(payload["orders"], list):
        raise PayloadError("edit_orders needs payload {orders: [{seq, items: [...]}]}")
    drafted = {o["seq"]: (o["rule_id"], json.loads(o["drafted_items"])) for o in orders}
    kept: dict[int, list[str]] = {}
    for entry in payload["orders"]:
        if not isinstance(entry, Mapping) or not {"seq", "items"} <= set(entry) <= {"seq", "rule_id", "items"}:
            raise PayloadError("each edited order is {seq, items} (rule_id optional)")
        seq, items = entry["seq"], entry["items"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq not in drafted:
            raise PayloadError(f"order {seq!r} was not drafted for this case")
        rule_id, drafted_items = drafted[seq]
        if "rule_id" in entry and entry["rule_id"] != rule_id:
            raise PayloadError(f"order {seq} was drafted by rule {rule_id!r}, not {entry['rule_id']!r}")
        if seq in kept:
            raise PayloadError(f"order {seq} is listed twice")
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items) or len(set(items)) != len(items):
            raise PayloadError(f"items for order {seq} must be a list of unique strings")
        invented = [i for i in items if i not in drafted_items]
        if invented:
            raise PayloadError(f"items {invented} were not drafted by rule {rule_id!r}; edits may only remove items")
        if items:
            kept[seq] = [i for i in drafted_items if i in items]   # keep the drafted order
    if not kept:
        raise PayloadError("an edit must keep at least one drafted item; use dismiss_flag to drop all orders")
    return kept


def _validate_margin_call(payload: Mapping[str, Any]) -> tuple[str, list[str]]:
    if not set(payload) <= {"call", "margins"} or "call" not in payload:
        raise PayloadError("margin_call needs payload {call: clear | tumor_at_margin | defer, margins: [...]}")
    call = payload["call"]
    margins = payload.get("margins", [])
    if call not in get_args(contract.MarginCall):
        raise PayloadError(f"unknown margin call {call!r}")
    if not isinstance(call, str) or not isinstance(margins, list) or not all(isinstance(m, str) for m in margins) \
            or len(set(margins)) != len(margins) or \
            any(m not in contract.MARGIN_NAMES for m in margins):
        raise PayloadError(f"margins must be unique names from {list(contract.MARGIN_NAMES)}")
    if call == "tumor_at_margin" and not margins:
        raise PayloadError("tumor_at_margin needs at least one margin")
    if call != "tumor_at_margin" and margins:
        raise PayloadError(f"call {call!r} takes no margins")
    ordered = [m for m in contract.MARGIN_NAMES if m in margins]
    return call, ordered
