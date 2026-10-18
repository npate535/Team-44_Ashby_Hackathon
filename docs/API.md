# PathQueue demo API

FastAPI backend in `backend/` (BUILD_PLAN Task 6), run locally from precomputed artifacts with zero LLM calls.
Research prototype — not for clinical use. All JSON; errors are `{"detail": "<message>"}`.

```bash
make demo CASES=fixtures                                   # the two committed fixtures (flat media)
uv run python -m backend.dev_cases --out artifacts/dev     # 18 synthetic cases covering the demo path
make demo ART=artifacts/dev                                # backend + UI on them
make demo                                                  # real/synthetic pipeline artifacts in artifacts/
```

| Setting / env | Meaning |
|---|---|
| `PATHQUEUE_ARTIFACTS_ROOT` | artifacts root (default `artifacts/`): `cases/`, `images/`, `cells/`, `eval/`, `spotcheck/` |
| `PATHQUEUE_CASES_DIR` | cases folder override (repo-relative allowed). When it is the flat `fixtures/` folder, `images/<f>` and `cells/<f>` resolve to `fixtures/<f>` |
| `PATHQUEUE_DEV_CORS=1` | allow `http://localhost:*` origins (frontend served elsewhere) |
| `backend.db_file` | SQLite file under the artifacts root; restored on restart and re-seeded only by explicit reset |
| `sim.speed` | simulator clock speed (× wall time) |
| `sim.max_replay_wall_s` | the last self-arriving case (the biopsy hero) arrives at most this many wall seconds after Start: the clock runs faster than `sim.speed` when the offsets need it (`sim.speed` reports the effective speed, `sim.base_speed` the setting); `null` = always `sim.speed` |
| `sim.hold_surgical_for_injection` | `true`: surgical cases arrive only through `inject_surgical` |
| `sim.after_arrivals_speed` | once every self-arriving case is in, the clock runs at this × wall time (1 = real time) so waits stay meaningful; `null` keeps the replay speed; reset restores it |
| `sim.or_rooms` | simulated OR rooms on the surgeon board (`OR-1…OR-n`, stable per case_id; also in `GET /api/config` `sim.or_rooms`) |
| `margin.timer` | surgical timer amber/red thresholds (minutes, wall clock) |
| `aging.*` | aging guard (core.tiers) on the simulator clock |

**Clocks.** Arrivals and aging use the **simulator clock** (`sim_time_s`, seconds, starts paused at 0). Surgical
timers and every timestamp (`at`, `*_at`) use the **wall clock** (ISO 8601 UTC). `server_time` is epoch seconds, for
client-side interpolation.

**Ground truth never leaves.** Case files are validated with `core.contract.Case` at load; only the `CasePublic` form
(no `eval_only`) is stored and served. Eval JSONs are additionally scrubbed of `eval_only`, `true_label`,
`true_labels`, `y_true`, `label_true` keys. `tests/test_api.py` scans every response.

**Only arrived cases exist.** A case that has not arrived yet returns 404 everywhere.

---

## GET /api/health

```json
{"status": "ok", "cases_dir": ".../artifacts/dev/cases", "artifacts_root": ".../artifacts/dev",
 "media": "artifacts", "loaded": 18, "skipped": [{"file": "broken.json", "error": "prediction.probs: ..."}],
 "synthetic": true,
 "sim": {"running": false, "speed": 30.0, "sim_time_s": 0.0, "hold_surgical_for_injection": true,
         "cases": 18, "arrived": 0, "pending": {"biopsy": 14, "surgical": {"normal": 2, "degraded": 1, "composite": 1}}},
 "server_time": 1792270000.0, "server_time_iso": "2026-10-17T20:46:40.000Z"}
```
`open_cases` counts all arrived cases awaiting a decision, across lanes; `open_surgical` counts the simulated frozen sections among them. `status` is `ok`, or `degraded` when no valid case loaded, any file was skipped, or referenced media is missing. `media` is `fixtures` (flat
fixtures folder) or `artifacts`. `skipped` lists invalid or duplicate case files, and surgical cases carrying stain
orders (never served).

## GET /api/config

UI-relevant settings (read strictly from `config/settings.yaml`).
```json
{"organ": "breast", "classes": ["Normal", "Benign", "InSitu", "Invasive"], "aggressive_class": "Invasive",
 "alpha": 0.1, "banner": "Research prototype — not for clinical use",
 "palette": {"ink": "#1B1A2E", "paper": "#F5F2EC", "hematoxylin": "#4A3B8F", "eosin": "#B0335C"},
 "poll_ms": 2000,
 "labels": {"tiers": {"1": "Surgical frozen section (simulated)", "2": "Likely invasive", "3": "Low confidence",
                      "4": "Unflagged"},
            "reasons": {"aggressive_top": "Invasive carcinoma suggested", "...": "..."},
            "classes": {"Normal": "Normal breast tissue", "...": "..."},
            "concept_groups": {"nottingham": {"label": "Nottingham-style", "ids": ["tubules", "pleomorphism", "mitoses"]}}},
 "margin": {"names": {"top": "superior", "bottom": "inferior", "left": "medial", "right": "lateral"},
            "note": "Anterior and posterior margins cannot be represented in a 2D field.",
            "timer": {"amber_min": 15, "red_min": 20}, "order": ["superior", "inferior", "medial", "lateral"]},
 "timer": {"amber_min": 15, "red_min": 20},
 "aging": {"enabled": true, "promote_at_fraction": 0.75, "turnaround_target_min": {"1": 20, "2": 1440, "3": 1440, "4": 2880}},
 "cells": {"enabled": true, "types": ["neoplastic", "inflammatory", "connective", "dead", "epithelial"],
           "overlay_colors": {"neoplastic": "#B0335C", "inflammatory": "#4A3B8F", "connective": "#C9A227",
                              "dead": "#5A5A5A", "epithelial": "#2E8BC0"}},
 "sim": {"speed": 30, "hold_surgical_for_injection": true, "variants": ["normal", "degraded", "composite"]},
 "flag_reasons": ["aggressive_top", "aggressive_in_conformal_set", "set_size_gt_1", "empty_set", "low_reliability",
                  "insufficient_tissue", "aging_promoted", "surgical_frozen_section"],
 "synthetic": true}
```

## GET /api/worklist?lane=all|surgical|biopsy&order=ai|arrival

Arrived cases only. `order=ai` (default): `core.tiers.sort_key` with the simulator clock (tier ↑ incl. the conformal
safety rule and the aging guard, risk ↓, arrival ↑, case_id). `order=arrival`: arrival on the simulator clock ↑.
In both orders, **done** cases (signed out, or margin call sent) follow all open cases, in the same order.
Invalid `lane`/`order` → 422.

```json
{"server_time": 1792270040.0, "server_time_iso": "2026-10-17T20:47:20.000Z",
 "sim": {"running": true, "speed": 30.0, "sim_time_s": 1500.0, "...": "as in /api/health"},
 "order": "ai", "lane": "all", "synthetic": true,
 "items": [
  {"case_id": "DEV_0015", "lane": "surgical", "simulated": true, "simulated_variant": null, "synthetic": true,
   "arrived_offset_s": 1386.0,
   "prediction": {"model": "concept_bottleneck", "classes": ["Normal", "Benign", "InSitu", "Invasive"],
                  "probs": [0.02, 0.05, 0.21, 0.72], "risk_score": 0.72, "conformal_alpha": 0.1,
                  "conformal_set": ["InSitu", "Invasive"]},
   "quality": {"qc_tiles_kept": 12, "ood_percentile": 0.42, "reliability": 0.81, "reliability_flag": false},
   "margin": {"suggested_call": "tumor_at_margin", "involved": ["superior"],
              "per_margin": {"superior": 0.81, "inferior": 0.03, "medial": 0.03, "lateral": 0.03}},
   "flag": {"flagged": false, "tier": 1, "reasons": ["aggressive_top", "aggressive_in_conformal_set", "set_size_gt_1",
                                                     "surgical_frozen_section"]},
   "agent": {"status": "none", "n_orders": 0, "orders_drafted": false, "blocked": 1},
   "runtime": {
     "tier": 1, "reasons": ["aggressive_top", "aggressive_in_conformal_set", "set_size_gt_1", "surgical_frozen_section"],
     "aging_promoted": false, "arrived_sim_s": 1500.0, "arrived_at": "2026-10-17T20:47:10.000Z", "injected": true,
     "timer": {"kind": "surgical", "clock": "wall", "started_at": "2026-10-17T20:47:10.000Z", "stopped_at": null,
               "elapsed_s": 10.0, "state": "ok", "amber_s": 900.0, "red_s": 1200.0},
     "decision": {"signed_out_at": null, "margin_call": null},
     "done": false, "allowed_actions": ["margin_call"], "or_room": "OR-3"}},
  {"case_id": "DEV_0014", "lane": "biopsy", "...": "...",
   "agent": {"status": "drafted", "n_orders": 2, "orders_drafted": true, "blocked": 0},
   "runtime": {"tier": 2, "reasons": ["aggressive_top", "aggressive_in_conformal_set", "set_size_gt_1"],
               "timer": {"kind": "biopsy", "clock": "sim", "elapsed_s": 234.0, "target_s": 86400.0, "state": "none"},
               "...": "..."}}]}
```
- `flag` is the static flag from step 08; `runtime.tier` / `runtime.reasons` are live (aging on the sim clock).
- `runtime.timer`: surgical → wall clock, `state` `ok | amber | red | stopped` (amber at `margin.timer.amber_min`, red
  at `red_min`, stopped once the margin call is sent; `elapsed_s` then stays frozen). Biopsy → sim-clock waiting time
  and the turnaround target of the case's tier before any aging promotion (`aging.turnaround_target_min`), `state`
  `none`, or `stopped` (frozen) once signed out.
- Items carry the case file's `arrived_offset_s`; the actual arrival on the simulator clock (differs for injected
  cases) is `runtime.arrived_sim_s`.
- `runtime.or_room` is set for surgical cases only (simulated).

## GET /api/cases/{case_id}

The full `CasePublic` (`core.contract`, mirrored in `frontend/src/types.ts`) with the live agent status and orders
merged in, plus `runtime` (as in the worklist, with `orders`) and `server_time`/`server_time_iso`. Never `eval_only`.
404 for unknown or not-yet-arrived cases.

```json
{"schema_version": "1.0", "synthetic": true, "case_id": "DEV_0014", "lane": "biopsy", "...": "CasePublic fields",
 "agent": {"status": "drafted", "orders": [
   {"type": "stain", "items": ["p63", "calponin"], "rule_id": "breast_insitu_vs_invasive",
    "rationale": "Myoepithelial markers help distinguish in situ from invasive disease."},
   {"type": "stain", "items": ["ER", "PR", "HER2", "Ki-67"], "rule_id": "breast_invasive",
    "rationale": "Standard biomarker panel if invasion is confirmed."}], "blocked": []},
 "runtime": {"tier": 2, "...": "...",
   "allowed_actions": ["approve_orders", "edit_orders", "closer_look", "dismiss_flag"],
   "orders": [{"seq": 0, "rule_id": "breast_insitu_vs_invasive", "type": "stain", "status": "drafted",
               "drafted_items": ["p63", "calponin"], "items": ["p63", "calponin"]},
              {"seq": 1, "rule_id": "breast_invasive", "type": "stain", "status": "drafted",
               "drafted_items": ["ER", "PR", "HER2", "Ki-67"], "items": ["ER", "PR", "HER2", "Ki-67"]}]},
 "server_time": 1792270040.0, "server_time_iso": "2026-10-17T20:47:20.000Z"}
```
`agent.orders` lists the active orders (edited items, removed orders dropped). `runtime.orders[].status` is
`drafted | approved | edited | removed | dismissed`.

## POST /api/cases/{case_id}/decision

Body `{"action": <action>, "payload": {...}}` (`payload` defaults to `{}`; unknown body keys → 422).

| Action | Allowed when | Payload | Effect |
|---|---|---|---|
| `approve_orders` | biopsy, agent `drafted` or `deferred_closer_look` | none | agent → `approved`; orders approved ("Order tests") |
| `edit_orders` | same | `{"orders": [{"seq", "rule_id"?, "items": [...]}]}` | agent → `edited_and_approved`. Orders are addressed by `seq` (position in `runtime.orders`; one rule may draft several orders); an optional `rule_id` must match. Items must be ones that order drafted (removal only, no free-text stains); orders left out are removed; at least one item must remain |
| `closer_look` | biopsy, agent `drafted` | none | agent → `deferred_closer_look` |
| `dismiss_flag` | biopsy, agent `drafted` or `deferred_closer_look` | none | agent → `dismissed`; orders dismissed |
| `sign_out` | biopsy, agent `none`, `approved`, `edited_and_approved` or `dismissed`; not yet signed out | none | routes the summary + placed orders to the oncologist; case done |
| (agent actions) | surgical too, on drafted **non-stain** orders (e.g. a recut for a reliability-flagged read) | as above | as above; independent of the margin call; no sign-out in this lane |
| `margin_call` | surgical, no call sent yet | `{"call": "clear" \| "tumor_at_margin" \| "defer", "margins": [...]}` | routes to the surgeon, stops the timer; case done. `tumor_at_margin` needs ≥1 margin of `superior, inferior, medial, lateral`; the other calls take none |

- **409** for a transition the state does not allow (e.g. `sign_out` while orders are drafted; anything after sign-out;
  `sign_out` on a surgical case, or an agent action there when nothing is drafted — stains are never ordered in the
  simulated frozen-section lane (a surgical case file carrying stain orders is rejected at load);
  `margin_call` on a biopsy; a second margin call). `runtime.allowed_actions` always lists what is allowed.
- **422** for an unknown action or a malformed payload (invented stain, unknown rule, unknown margin, extra keys).
- **404** for an unknown or not-yet-arrived case.

```json
{"result": {"case_id": "DEV_0014", "action": "approve_orders", "from": "drafted", "to": "approved"},
 "case": {"...": "the updated GET /api/cases/{id} body"}}
```
```json
{"detail": "act on the drafted orders first (agent status drafted)"}
```
Every accepted action is written to `decisions` and `events` with wall and sim timestamps.

## GET /api/cases/{case_id}/events

Timestamped transitions for one case, oldest first.
```json
[{"id": 10, "case_id": "DEV_0014", "type": "arrived", "from": null, "to": null, "action": null,
  "payload": {"arrived_sim_s": 1266.0, "lane": "biopsy"}, "sim_time_s": 1270.5, "at": "2026-10-17T20:47:02.150Z"},
 {"id": 14, "case_id": "DEV_0014", "type": "agent_transition", "from": "drafted", "to": "approved",
  "action": "approve_orders", "payload": {}, "sim_time_s": 1302.0, "at": "2026-10-17T20:47:03.200Z"},
 {"id": 15, "case_id": "DEV_0014", "type": "routed", "action": "sign_out", "payload": {"to": "oncologist"}, "...": "..."},
 {"id": 16, "case_id": "DEV_0014", "type": "signed_out", "from": "approved", "to": "approved", "action": "sign_out",
  "payload": {"orders": [{"type": "stain", "rule_id": "breast_insitu_vs_invasive", "items": ["p63", "calponin"]}]},
  "...": "..."}]
```
Event types: `arrived`, `injected`, `timer_started`, `agent_transition`, `signed_out`, `margin_call`,
`timer_stopped`, `routed` (case events); `reset`, `sim_start`, `sim_pause`, `spotcheck` (global, not listed here).

## GET /api/routed?to=surgeon|oncologist

Results sent after a margin call (surgeon) or a sign-out (oncologist), newest first. Invalid `to` → 422.
```json
[{"id": 2, "case_id": "DEV_0015", "to": "surgeon", "kind": "margin_call", "sim_time_s": 1530.0,
  "call": "tumor_at_margin", "margins": ["superior"], "timer_elapsed_s": 312.4, "simulated": true,
  "suggested_call": "tumor_at_margin", "simulated_variant": null, "synthetic": true,
  "at": "2026-10-17T20:52:22.400Z", "or_room": "OR-3"}]
```
```json
[{"id": 1, "case_id": "DEV_0014", "to": "oncologist", "kind": "sign_out", "sim_time_s": 1302.0,
  "agent_status": "approved",
  "orders": [{"type": "stain", "rule_id": "breast_insitu_vs_invasive", "items": ["p63", "calponin"]},
             {"type": "stain", "rule_id": "breast_invasive", "items": ["ER", "PR", "HER2", "Ki-67"]}],
  "summary": {"generator": "template", "sentences": [{"text": "Suggested category: invasive carcinoma ...",
                                                      "findings": ["F1"]}]},
  "suggestion": {"classes": ["Normal", "Benign", "InSitu", "Invasive"], "probs": [0.02, 0.05, 0.21, 0.72],
                 "risk_score": 0.72, "conformal_set": ["InSitu", "Invasive"], "conformal_alpha": 0.1},
  "synthetic": true, "at": "2026-10-17T20:47:03.300Z"}]
```
Oncologist `orders` are only approved/edited orders (dismissed → `[]`).

Every routed item also carries a read-only `report`, built at read time from the stored public case (so results routed
before it existed gain it too; never ground truth): `lane`, `top_class`, static `tier` and flag `reasons`, `quality`
(`reliability`, `reliability_flag`, `qc_tiles_kept`), `simulated`, `arrived_at`; for the oncologist the three
strongest associated `features` (`id`, `label`, `contribution`, `direction`, `source`); for the surgeon the simulated
`margin` read (`suggested_call`, `involved`, `per_margin`, `threshold`), `simulated_variant` and `degradation`.

## POST /api/sim/{start|pause|reset|inject_surgical}

```json
{"command": "inject_surgical", "injected": "DEV_0015", "sim": {"running": true, "...": "as in /api/health"}}
```
- `start` / `pause`: idempotent; the clock keeps its time across pauses.
- `reset`: re-reads the case JSONs, re-seeds every table (drafted orders restored, routing and events cleared) and
  rewinds the paused clock to 0.
- `inject_surgical`, body `{"variant": "normal" | "degraded" | "composite"}` (default `normal`): the next pending
  surgical case of that variant arrives now (sim clock) and its timer starts (wall clock). It prefers the case best
  matching the `demo.heroes` entry for that variant (`surgical`, `surgical_degraded`, `surgical_composite`), then the
  earliest `arrived_offset_s`. **409** when none of that variant is left; unknown variant/command → 422.

## GET /api/metrics

Every `<artifacts>/eval/*.json` keyed by file stem, the figure files in that folder, unreadable files, and the
registry of expected eval files (`core.schemas.EVAL_FILES`) so the UI can show "pending" for missing ones. Tonight's
additions: `analytics` (chart-ready test-split aggregates from `eval/analytics.py`) and `qc` (the tile-QC fix).
```json
{"eval": {"calibration": {"name": "calibration", "synthetic": true, "...": "payload as written by eval/"}},
 "figures": [{"file": "calibration.png", "url": "/eval/calibration.png"}],
 "errors": [{"file": "broken.json", "error": "JSONDecodeError: ..."}],
 "expected": {"metrics": "Task 11 · AUROC A vs B, ...", "queue_sim": "Task 11 · FIFO vs tiers vs tiers+aging ..."},
 "synthetic": true}
```

## GET /api/spotcheck/items · POST /api/spotcheck

`GET` returns `<artifacts>/spotcheck/items.json` (`core.schemas.SPOTCHECK_ITEMS`) with `available: true`, or
`{"available": false, "items": [], "synthetic": ...}` when it is missing. Items never carry class labels.
```json
{"available": true, "created_utc": "2026-10-17T20:40:00Z", "synthetic": true,
 "items": [{"item_id": "concept:invasion:DEV_0001:5", "kind": "concept", "target": "invasion",
            "label": "Tumor infiltrating stroma", "case_id": "DEV_0001", "tile_idx": 5,
            "image": "spotcheck/invasion_DEV_0001_5.png", "score": 2.3}]}
```
`POST` body:
```json
{"reviewer": "Rishabh", "reviewer_kind": "human",
 "reviews": [{"item_id": "concept:invasion:DEV_0001:5", "verdict": "acceptable", "note": ""}]}
```
`verdict` ∈ `acceptable | not_acceptable | unsure`; `reviewer_kind` ∈ `human | ai` (AI-reviewed checks stay labeled as
such). Reviews merge into `<artifacts>/eval/spotcheck.json` (a later review by the same reviewer of the same item
replaces the earlier one); top-level `reviewer_kind` is `human`, `ai` or `mixed`; `summary` counts acceptable per
concept and for cell overlays. Unknown item ids → 422; no items file → 409.
```json
{"recorded": 1, "total": 3, "reviewer_kind": "mixed",
 "summary": {"concept": {"invasion": {"n": 2, "acceptable": 1}}, "cells": {"n": 1, "acceptable": 0}}}
```

## Static routes

| Path | Source |
|---|---|
| `/images/<file>`, `/cells/<file>` | `<artifacts>/images`, `<artifacts>/cells`; `fixtures/<file>` when the cases folder is `fixtures/`. `?w=64…1024` returns a downscaled JPEG thumbnail (also for `/spotcheck/`), cached in memory per file and mtime |
| `/eval/<file>` | `<artifacts>/eval/<file>` (figures) |
| `/spotcheck/<file>` | `<artifacts>/spotcheck/<file>` (crops) |
| `/fixtures/<file>` | `fixtures/` media, and fixture case JSONs **in their public form** (no `eval_only`) for the frontend's `?mock=1` mode |
| `/` | `team44-triage-frontend/app` (index.html; `.js` served as `text/javascript`) |

File names must match `[A-Za-z0-9][A-Za-z0-9_.-]*` (no paths) and carry a media suffix (`.jpg/.jpeg/.png`, plus
`.svg` for `/eval`): JSON is only ever served through the API, so it is always ground-truth free. Anything else is
404. Every response carries `Cache-Control: no-store` (local demo; avoids stale ES modules). A malformed server-side
file (e.g. an `eval/spotcheck.json` without `reviews`) returns 500 with a `detail` naming it.

### Workspace integrity and restart behavior

`GET /api/health` also returns `assets: {referenced, available, missing, models}`. `missing` lists referenced images/overlays that are unavailable, including references and contrast cases. `models` lists the prediction model IDs in the loaded cases. Health is `degraded` for empty case sets, rejected JSONs, or missing referenced media. These checks establish artifact availability, not accuracy or clinical validity.

The SQLite workspace is restored on service restart, with the arrival replay paused at its last checkpoint. Decisions, order edits, arrivals, surgical timers, routed results, and the event history remain intact. A changed set of case payloads refuses startup against the saved database, preserving prior review evidence. Use a separate artifact root or intentionally archive the old database before starting with different model outputs. Explicit simulator reset remains the only normal route that reseeds the workspace; the UI asks the reviewer to confirm that destructive reset.
