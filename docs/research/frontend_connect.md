# Frontend connect: design for Task 7 (connect + extend `team44-triage-frontend/`)

Sources read in full: `team44-triage-frontend/**` (README, index.html, app.js 142 lines, styles.css 74 lines,
data/cases.json 9 mock colon cases), `plan.md` §1/§7/§8/§10/§11/§14, `CELLS_SPEC.md` §7.2–7.3, `docs/BUILD_PLAN.md`
§A12–A13, §A21, Task 6, Task 7, ledger rulings R2/R6. Ruling R6 (user directive) supersedes the older §A21 line
"stays untouched": we **extend** the teammate's app; we do not rewrite it. Task 6's exact response shapes are not
frozen yet (R2), so every contract and server field is read in **one adapter** (`toView`), which tolerates both
the shapes proposed below and a bare `CasePublic`.

---

## 1. Inventory of the existing app

| Piece | What it does today | Decision |
|---|---|---|
| `ST={tab,mode,sel,pause,cancel,dec,hide}` | Single global state object | **Keep the pattern.** Extend to `{route, caseId, order, lane, mock, base, cfg, sim, items, cases:{}, events:{}, routed:{surgeon,oncologist}, metrics, seen:Set, fail, editing, hide, toast}`. Drop `cancel` (nothing is ever auto-ordered now). `dec` moves to server state |
| `draw()` + `innerHTML` string templates | Full re-render into `#app` | **Keep** string templates and the single render root. Add `esc()` on every interpolated value, because summaries are LLM text. Add a fingerprint check and a timer-only tick (§3.4) so 2-second polling doesn't clobber focus or form state |
| Event delegation on `#app` (`data-m`, `data-c`, `data-d`, `data-r`, `#bk`, `tr.row`) + Enter/Space on rows | One click handler and one keydown handler | **Keep verbatim in style.** Add `data-act` (decision), `data-call` (margin), `data-hm` (heatmap select), `data-why` (bar), `data-f` (finding ref), `data-tab`, `data-sim` |
| `TIERS`/`LAB` (urgent/unsure/quality/routine) | Tier order + labels | **Replace** with numeric tiers 1–4 (§2.2). The old `t-*` classes stay as aliases |
| `sorted()` priority rule (frozen by minutes left → aged → tier → share used) | Client-side sort | **Live mode: never re-sort.** The server's `core.tiers` is the single tier implementation (plan §10), and it already includes the aging guard (§A21). Move an adapted `sorted()` into `mock.js` as the mock-mode order (tier ↑, risk ↓, arrival ↑). Credit the rule in the README |
| `used/aged` (75% of target) | Aging badge "Aged: promoted at 75% of target" | **Keep the badge text.** It now renders when `reasons` includes `aging_promoted` (server-computed, `aging.enabled`) |
| `fmt(m)`, `rem(c)` | Minute/hour/day formatting, "Overdue …" | **Keep verbatim** (move to `adapter.js`). `rem` takes `(elapsedMin, targetMin)` |
| `rng`, `mosaic(c)` | Fake 16×8 tumor mosaic from `seed`/`tf` | **Remove** and replace with the real tile heatmap over the display image (`heatmap.js`) |
| `acts(c)` (CRC-1/2/3, GEN-1/2, OR-1, BLOCKED) | Hard-coded colon agent actions + one blocked action | **Replace** with `agent.orders` from the contract. **Keep the blocked-action UI** (`.act.blk`): it renders `agent.blocked[]` if present, plus a static guard row on every surgical case: "Guard: no stain orders in the simulated frozen-section lane — margin call only" |
| `row()` / `worklist()` | Table: Case, Lane, AI tier, Timer, Agent; arrival vs AI toggle + note | **Adapt.** New columns: Case, Tier, Conformal set, Timer, Agent. Group headers per tier in AI order. Keep the toggle (now `?order=ai\|arrival` on the server) |
| `tasks()` | "Tasks" card from frozen timers, pre-orders, re-scans | **Keep the idea.** Tasks: unread surgical cases ("Read simulated frozen <id>, 6 min left"), `agent.status==='drafted'` ("Review drafted orders, <id>"), `low_reliability` ("Reliability flag: consider recut/defer, <id>") |
| `detail(c)` | Two-column case grid; mosaic, concept chips, look-alikes, AI read kv, agent actions, approval buttons, "hide AI opinion until first impression" (`ST.hide`) | **Adapt** into `case_biopsy.js` / `case_surgical.js` using the same `.grid/.card/.kv/.chips/.look/.act/.btns/.verdict` classes. **Keep `ST.hide`** (blind first read, anti-anchoring) behind `?blind=1`; it is not part of the demo path |
| `orboard()` | OR cards: room, countdown, 4-step bar, static "OR-2 Idle" | **Adapt** into the surgeon screen (§4.6). Steps become Received → Being read → Result called. Drop the hard-coded idle card for an empty state |
| `setInterval` 1 s: `el += 1` (1 simulated min/s) | Local fake clock | **Replace** with server clocks: surgical timers use wall clock; arrivals use the sim clock. The 1 s tick only interpolates timer text (§3.4) |
| `#pp` Pause clock | Local pause | **Wire** to `POST /api/sim/pause` / `start`. Add Start, Reset, and inject buttons (normal/degraded/composite) |
| fetch `data/cases.json` + "serve over HTTP" error card | Data load | **Replace** with `api.js`. Keep the error card's wording for the `file://` case |
| CSS tokens `--bg --sf --ink --mut --ln --hema --eos --eosl --amb --ambl --grn --grnl --bl`; dark theme via `prefers-color-scheme` + `:root[data-theme]`; IBM Plex / Iowan type; `.seg`, `.tier`, `.timer.hot`, `.pbar`, `.aged`, `.card`, `.grid`, `.chips`, `.look`, `.kv`, `.act(.off/.blk)`, `.btn(.p/.d)`, `.verdict`, `.or/.steps/.big`, reduced motion | Visual language | **Keep all verbatim.** Append a `/* Sepia additions */` section only (new tokens below). Don't recolor existing tokens |
| `.mock` amber notice | "Mock data" banner | **Keep the class.** It shows only in `?mock=1` or when any case has `synthetic:true`. Add a persistent `.banner`: "Research prototype — not for clinical use" |

New CSS tokens, appended:
- `--red: var(--eos)` (timer red);
- `--c-neo #B0335C`, `--c-inf #4A3B8F`, `--c-con #C9A227`, `--c-dead #5A5A5A`, `--c-epi #2E8BC0`. These are the CELLS_SPEC §5 overlay colors, fixed in both themes because they must match the PNG overlays;
- classes `.t-1` (solid `--eos` with white text), `.t-2` (= old `.t-urgent`), `.t-3` (= `.t-unsure`), `.t-4` (= `.t-routine`). The reason badge for quality reasons reuses `.t-quality`;
- `.timer.amber{color:var(--amb)}`, `.timer.red{color:var(--eos)}`, `.timer.stopped{color:var(--grn)}`;
- `.row.new` keyframe fade/slide, disabled under reduced motion.

---

## 2. Field mapping and the adapter

### 2.1 Old mock field → contract / server field

| Old | New source | Notes |
|---|---|---|
| `id` | `case_id` | |
| `lane: 'frozen'` | `lane: 'surgical'` (+ `simulated: true`) | Label "Simulated frozen section". Also shows `simulated_variant` (`degraded` → "artifact: <type>", `composite` → "composite field") |
| `lane: 'biopsy'` / `'resection'` | `lane: 'biopsy'` / — | There is no resection lane; drop it |
| `site: 'Colon'` | `cfg.organ` ("breast") | Breast/BACH only. Remove the colon strings from all defaults |
| `q` (clinical question) | *(no field)*, derived, neutral | surgical: "Margin status (simulated)"; biopsy: "Breast microscopy field (BACH)". Never invent clinical history |
| `el` (elapsed min) | surgical: `runtime.timer.elapsed_s` (wall clock) interpolated; biopsy: `(sim.sim_time_s − arrived_offset_s)` | §3.4 |
| `tg` (target min) | surgical: `cfg.timer.red_min` (20); biopsy: `cfg.aging.target_s_by_tier[tier]` if present | The old "hot at ≤5 min left" becomes amber at 15 min of 20 (§A12) |
| `tier` urgent/unsure/quality/routine | `runtime.tier ?? flag.tier` (1–4) + `runtime.reasons ?? flag.reasons` | The old "quality" tier is folded into tier 3 with reason `low_reliability` / `insufficient_tissue` |
| `risk` 0–100 / null | `round(prediction.risk_score*100)` | Labeled "calibrated P(Invasive)"; null if no prediction |
| `tf` (tumor fraction) | `tiles[].risk` | Drives the real heatmap |
| `rd` "3/3" readers agree | `agreement.conch_musk_same_top`, `agreement.neighbor_agreement` (+ finding `retrieval.agree/of`) | "CONCH and MUSK agree: yes"; "4 of 5 references share the predicted class" |
| `room` "OR-3" | `runtime.or_room ?? 'OR-' + (hash(case_id)%4+1)` | Labeled simulated |
| `step` 0–3 | derived: routed surgeon result → "Result called"; else "Being read" | 3 steps; "Cutting" has no data |
| `seed` | — | removed |
| `ood` text | `quality.ood_percentile`, `quality.reliability`, `quality.reliability_flag` | "Input check: typical (OOD pct 42)" / "Reliability flag raised" |
| `set` "{benign, malignant}" | `prediction.conformal_set` + `prediction.conformal_alpha` | Chips; heading "90% conformal set" |
| `tissue` / `scarce` | `quality.qc_tiles_kept` + reason `insufficient_tissue` | The rule `insufficient` drafts "request new sample" |
| *(new)* | `margin`, `concepts`, `tiles`, `similar_cases`, `contrast_case`, `findings`, `summary`, `agent`, `cells`, `image`, `schema_version`, `synthetic` | New panels (§4) |
| *(never)* | `eval_only` | The adapter `delete`s it and `console.warn`s if present. It is never rendered |

### 2.2 Tier and reason labels (plan §1; §A13 enum)

```js
export const TIER = {
  1: {label: 'Surgical frozen', cls: 't-1', note: 'Simulated frozen section, timer running'},
  2: {label: 'Likely invasive', cls: 't-2', note: 'Invasive is the top class or is in the conformal set'},
  3: {label: 'Low confidence',  cls: 't-3', note: 'More than one class possible, or reliability flag'},
  4: {label: 'Routine',         cls: 't-4', note: 'Read in turn. Routine does not mean cleared.'}}; // teammate wording kept
export const REASON = {
  aggressive_top: 'Top class: Invasive', aggressive_in_conformal_set: 'Invasive in conformal set',
  set_size_gt_1: 'More than one class possible', empty_set: 'Empty conformal set',
  low_reliability: 'Reliability flag', insufficient_tissue: 'Insufficient tissue',
  aging_promoted: 'Aged: promoted at 75% of target', surgical_frozen_section: 'Simulated frozen section'};
```
Labels may be overridden by `cfg.labels.tiers` / `cfg.labels.reasons` from `/api/config`. Unknown reason codes
render as the raw code, so nothing is hidden.

### 2.3 `toView(caseJson, nowMs, config)` (pure, no DOM; in `js/adapter.js`)

- `caseJson` is a worklist item or a full `CasePublic`, optionally with `runtime` (§3.2). The API layer stamps it
  with `_fetched_ms`.
- `config` = `/api/config` merged with defaults and the latest `sim` status.

The function returns:

```js
{ id, lane, simulated, variant, synthetic, laneLabel, question, room,
  tier, tierLabel, tierCls, reasons:[{code,label}], aged,
  risk /*0–100|null*/, probs:[{cls,p}], top, set:[...], setSize, alphaPct /*90*/,
  reliability:{score, flag, oodPct, qcTiles}, agreement:{conchMusk, neighbor, neighborText},
  timer:{kind:'surgical'|'biopsy'|'none', elapsedS, targetS, usedFrac,
         state:'ok'|'amber'|'red'|'stopped'|'none', text /* via fmt/rem */},
  agent:{status, orders, blocked, drafted:status==='drafted', allowed:[...] },
  decision:{signedOut, marginCall:{call, margins, at}|null},
  image:{src, w, h, tile}, tiles, concepts /*+ label from cfg*/, why /*rows, §4.2*/,
  similar, contrast, findings /*Map id→finding*/, summary:{sentences, generator, isLLM},
  cells /*{status,...} normalized; status 'disabled' if absent*/, margin /*normalized or null*/ }
```

Rules:
- Every field is null-safe. A missing block yields `null`, and the view shows the state text.
- Surgical timer:
  - `elapsedS = runtime.timer.elapsed_s + (stopped ? 0 : (nowMs − _fetched_ms)/1000)`;
  - if the server sends only `started_at`, use `nowMs + serverOffset − Date.parse(started_at)`, with
    `serverOffset` measured from `server_time` per fetch;
  - state: `stopped` if `stopped_at`, else red if ≥ `red_min·60`, else amber if ≥ `amber_min·60`, else ok.
- Biopsy timer: `elapsedS = (sim.sim_time_s + (sim.running ? (nowMs − sim._fetched_ms)/1000·sim.speed : 0)) −
  arrived_offset_s`, with no amber/red. Show the `.pbar` only if a tier target exists.
- `agent.allowed`: `runtime.allowed_actions` if sent, else the client table in §4.5. The server stays the
  authority, and a 409 triggers a refetch.

---

## 3. API client (`js/api.js`) and mock mode (`js/mock.js`)

### 3.1 Live client
`createApi({base})`: `base` is `''` (same origin, the default) or `?api=http://host:port`. All calls go through
`req(method, path, body)`:
- `fetch(base+path)` with an `AbortController` 4 s timeout, `Accept: application/json`, and JSON body;
- non-2xx throws `ApiError{status, detail}`, read from FastAPI's `{detail}`.

| Method | HTTP |
|---|---|
| `health()` | `GET /api/health` |
| `config()` | `GET /api/config` |
| `worklist({lane='all', order='ai'})` | `GET /api/worklist?lane=&order=` |
| `getCase(id)` | `GET /api/cases/{id}` |
| `events(id)` | `GET /api/cases/{id}/events` |
| `decide(id, action, payload)` | `POST /api/cases/{id}/decision` `{action, payload}`; action ∈ `approve_orders, edit_orders, closer_look, dismiss_flag, sign_out, margin_call` |
| `routed(to)` | `GET /api/routed?to=surgeon\|oncologist` |
| `sim(cmd, body)` | `POST /api/sim/{start\|pause\|reset\|inject_surgical}`; inject body `{variant:'normal'\|'degraded'\|'composite'}` |
| `metrics()` | `GET /api/metrics` |
| `spotcheck(body)` | `POST /api/spotcheck` |
| `assetUrl(p)` | `base + '/' + p.replace(/^\/+/, '')`. Contract paths `images/…`, `cells/…` map to `/images/…`, `/cells/…` |

### 3.2 Shapes the frontend expects (propose to Task 6 via controller; the adapter tolerates absence)
- **`/api/worklist`** returns
  `{server_time, sim:{running, speed, sim_time_s}, order, items:[…]}`, or a bare array.
  - Each item is a CasePublic subset: `case_id, lane, simulated, simulated_variant, synthetic, arrived_offset_s,
    prediction{risk_score, conformal_set, conformal_alpha}, quality{reliability_flag}, flag, agent{status}`;
  - plus `runtime{tier, reasons, timer{elapsed_s, started_at, stopped_at}, decision, or_room}`.
- **`/api/cases/{id}`** returns `CasePublic` (no `eval_only`) with live `agent.status` / edited orders and the same
  `runtime` block, plus `runtime.allowed_actions`.
- **`/api/routed`** returns
  `[{case_id, to, kind:'margin_call'|'sign_out', at, call?, margins?, summary?, orders?, agent_status?}]`.
- **`/api/config`** returns
  `{organ, classes, aggressive_class, alpha, timer:{amber_min, red_min}, aging:{enabled, threshold,
  target_s_by_tier}, cells:{enabled}, sim:{speed}, labels:{tiers, reasons, concepts, cell_features, cell_types}}`.
- **`/api/metrics`** returns `{<eval file stem>: <json>}`.
- **`/api/events`** returns `[{at, type, from?, to?, action?, payload?}]`.
- If `sim` is missing from the worklist, read it from `/api/health`.

### 3.3 Polling and failure states
- **One poll loop, every 2 s**, by route:
  - worklist: `worklist`;
  - case: `events(id)`, plus `getCase(id)` only on open, after a decision, or if the events count changed;
  - surgeon: `worklist({lane:'surgical'})` + `routed('surgeon')`;
  - oncologist: `routed('oncologist')`;
  - metrics: once, plus a manual refresh button.
- Polling stops while `document.hidden`. The loop uses `setTimeout` chaining, never overlapping requests.
- **Failures:**
  - After 2 consecutive failures, show `[data-testid=state-offline]`: "Backend unreachable at <base or this
    server>. Retrying… Start it with `make demo`, or open with `?mock=1`." The last data stays on screen,
    marked "stale since hh:mm:ss";
  - after 5 failures, back off to 5 s; on success, clear and return to 2 s;
  - a 404 on a case shows a "Case not found" card with Back;
  - a 409 on a decision shows the toast "Not allowed in state <status>" (`aria-live=polite`), then a refetch.
- **States** (each has a testid):
  - `state-loading`: initial;
  - `state-empty`: worklist empty before the sim starts ("No cases yet. Start the simulator.");
  - `cells-disabled`, `margin-null`, `metrics-missing` (per section).

### 3.4 Render discipline
- **Structural render** only when the route changes or the data fingerprint changes. The fingerprint is
  `JSON.stringify` of the items' `case_id`, `tier`, `reasons`, `agent.status`, and `decision`.
- Skip the structural render while `ST.editing` is true (the edit-orders form or margin picker is open) or while
  focus is on an `input/select/textarea` inside `#app`. Restore `scrollY` after re-render.
- **1 s tick**: for each `[data-timer]` element, recompute through the adapter's timer function and update only
  `textContent` and `class` (`ok/amber/red/stopped`). Also update the sim clock `#sim` ("sim +12:30, ×30").
- **Arrivals**: ids not yet in `ST.seen` get `class="row new"`. The first load marks everything as seen without
  animation.

### 3.5 `?mock=1` (no backend)
- `mock.js` exports `createMockApi()` with the **same method surface**. It loads
  `new URL('../mock/manifest.json', import.meta.url)`.
- `app/mock/manifest.json` (Task 7 owns `app/mock/`; `fixtures/` belongs to Task 1):
  ```json
  {"asset_base": "../../fixtures/",
   "cases": [
     {"src": "../../fixtures/demo_0001.json"},
     {"src": "../../fixtures/demo_0002.json"},
     {"src": "../../fixtures/demo_0001.json", "id": "MOCK_B02", "patch": {"arrived_offset_s": 30, "flag": {"tier": 4, "reasons": []}, "prediction": {"risk_score": 0.08, "conformal_set": ["Benign"]}, "agent": {"status": "none", "orders": []}}},
     {"src": "../../fixtures/demo_0002.json", "id": "MOCK_S02_DEG", "inject": "degraded", "patch": {"simulated_variant": "degraded", "quality": {"reliability_flag": true}, "flag": {"tier": 1, "reasons": ["surgical_frozen_section", "low_reliability"]}, "margin": {"suggested_call": "defer"}}}
   ]}
  ```
  Patches are deep-merged. Every mock case is forced to `synthetic:true`.
- Fixture URLs are document-relative. `../../fixtures/` resolves to `/fixtures/` both when the backend serves the
  app at `/` and when `python3 -m http.server` runs at the repo root and serves `/team44-triage-frontend/app/`.
- In-memory behaviour:
  - **strip `eval_only`** from every case;
  - arrivals are revealed when the sim clock passes `arrived_offset_s` (speed from config default 30);
  - entries with `inject` are held back until `sim('inject_surgical', {variant})`;
  - decision state machine (§4.5) with a 409 on invalid transitions; routing to the surgeon/oncologist arrays;
  - wall-clock timers that stop on `margin_call`;
  - sorting with the adapted teammate rule: tier, then risk desc, then arrival. It does not recompute tiers;
  - `metrics()` returns `{}`, so the page shows `metrics-missing` states. Optionally it reads
    `app/mock/metrics.json`, which must be labeled synthetic;
  - `spotcheck()` logs and resolves.
- `assetUrl(p)` = `manifest.asset_map?.[p] ?? asset_base + basename(p)`.

---

## 4. Views

Routing is by hash: `#/worklist` (default), `#/case/<id>`, `#/surgeon`, `#/oncologist`, `#/metrics`,
`#/spotcheck`. The existing `nav` tablist gets tabs `nav-worklist`, `nav-surgeon` ("Surgeon / OR board"),
`nav-oncologist`, `nav-metrics`, `nav-spotcheck`. The header keeps the title, the "Team 44 · Ashby Prize Hackathon
2026" line and `#sim`, plus a `.seg` group with these buttons:
- `sim-start`, `sim-pause` (the old `#pp`), `sim-reset`;
- `sim-inject-surgical`, `sim-inject-degraded`, `sim-inject-composite`.

Banners: `banner-research` is always on. `banner-mock` appears in mock mode or when any case is synthetic.

### 4.1 Worklist (`views/worklist.js`; adapts `row/worklist/tasks`)
- Bar: `.seg` with `order-ai` / `order-arrival` (aria-pressed); optional lane seg `lane-all|surgical|biopsy`.
  The note text:
  - AI: "Surgical frozen sections first, then likely invasive, low confidence, routine; within a tier by risk,
    then arrival. Aged cases move up one tier (never above tier 2)."
  - Arrival: "Read roughly first come, first served." (kept)
- `table[data-testid=worklist]`:
  - in AI order, `tr.grp` header rows: "Tier 2 · Likely invasive (n)";
  - rows: `tr.row[data-testid=worklist-row][data-case-id][data-tier][data-lane][tabindex=0]`.
- Cells:
  1. Case: **id**, sub = laneLabel + question; reason badges `span.reason[data-testid=reason-<code>]`;
     `.aged` when `aging_promoted`.
  2. Tier: `span.tier.t-N[data-testid=tier-badge]` + sub "72% P(Invasive), calibrated", or "no score".
  3. Set: `span.chip[data-testid=set-chip]` per class; `empty_set` shows the chip "∅ empty".
  4. Timer: `span.timer[data-timer][data-state]` + `.pbar` (surgical: share of red threshold; biopsy: share of
     tier target if known).
  5. Agent: "● Orders drafted" `[data-testid=orders-drafted]` / "Approved" / "Dismissed" / "Closer look" /
     "Signed out" / "Margin call sent" / "None".
- Below the table: the teammate's **Tasks** card (`[data-testid=tasks]`).
- ≤600 px: CSS turns rows into cards (`table, tbody, tr, td {display:block}`, thead hidden, `min-width` unset).
  The page must not scroll horizontally.

### 4.2 Biopsy case (`views/case_biopsy.js`) — `section[data-testid=case-view][data-lane=biopsy]`
- Header card:
  - `back` button (`#bk` kept), title `case_id · Likely invasive`;
  - sub: lane, "arrived +mm:ss";
  - tab strip `tab-evidence` | `tab-cells` (`role=tablist`).
- Left column, `.grid` kept:
  - **Heatmap** `figure[data-testid=heatmap]`:
    - `div.hm` (`position:relative`) holds `<img src=assetUrl(image.display)>` plus an absolutely positioned
      `<svg viewBox="0 0 {image.width} {image.height}" preserveAspectRatio="none">`. Tile coordinates are in
      original pixels, so the downscaled JPEG scales for free;
    - one `rect.hm-tile[data-tile=idx]` per tile at `(x, y, tile.size ?? image.tile_size)`;
    - fill `var(--eos)` with `fill-opacity` = risk (0–1); concept → `clamp(z/3,0,1)` (z ≤ 0 transparent);
      attention → `a/max(a)`;
    - QC-dropped grid cells: hatched `rect.hm-drop`;
    - `select[data-testid=heatmap-select]` options: `risk`, each `concepts[].id` (label), `attention` if any
      `tiles[].attention`;
    - `input[type=range][data-testid=heatmap-opacity]`; legend reuses `.legend b`;
    - highlighted tiles get `rect.hl` (`stroke:var(--ink); stroke-width:16; fill:none`).
  - **Summary** `[data-testid=summary]`:
    - each sentence is a `p`; finding refs `button.fref[data-testid=finding-ref][data-f=F2]` → "[F2]";
    - badge `[data-testid=summary-generator]`: "LLM-written from findings · <model>" when `generator` starts with
      `lumen:`, else "Template";
    - note: "Written only from structured findings; it never saw the image."
    - Ref click by `findings[id].kind`:
      - `prediction` → scroll to and flash How-sure;
      - `concept` → set heatmap to `data.concept` and highlight `data.tiles`;
      - `retrieval` → Compared-to;
      - `quality` → reliability row;
      - `cells` → switch to the Cells tab, scroll to tile `data.tiles[0]`;
      - unknown → no-op.
- Right column:
  - **Why** `[data-testid=why]`, heading "Features associated with invasive carcinoma" (sub: "contribution toward /
    away from calibrated P(Invasive); not causes; not a Nottingham grade"):
    - rows = `concepts[]` + `cells.features` entries whose id is not already a concept id (no double count);
    - sorted by |contribution|;
    - each row `button.why-row[data-testid=why-bar-<id>][data-why=id]` holds a label, a source marker
      `span.src[data-testid=src-measured|src-estimated]` ("measured" ◆ for cell features / `source:'cell'`;
      "estimated" ○ for text/cav), and a `caption-consistent` chip when `concepts[].caption_consistent`;
    - each row also holds an inline `<svg viewBox="-1 0 2 1" preserveAspectRatio="none">`: a zero line at x=0
      and `rect` from 0 to `c/maxAbs`, positive in `--eos` ("toward Invasive"), negative in `--hema`;
    - signed value `+0.44`; `spread` as a title tooltip;
    - click → heatmap selects that concept and highlights `top_tiles`. A cell feature highlights the
      `cells.tiles` idx.
  - **How sure** `[data-testid=how-sure]`:
    - big `risk%` "calibrated P(Invasive)";
    - per-class mini bars (`prediction.classes/probs`);
    - "90% conformal set" chips `[data-testid=conformal-chip]` + "single class = confident" or "N classes
      possible";
    - `.kv` rows: CONCH–MUSK agreement, neighbor agreement, prompt spread (mean `concepts[].spread`),
      reliability score + flag `[data-testid=reliability-flag]` when raised, OOD percentile, QC tiles kept;
    - stamp `prediction.model · schema_version`.
  - **Compared to** `[data-testid=compared]`:
    - `.look` grid of `similar_cases` (thumb, reference label, similarity), labeled "train-split reference
      cases";
    - contrast card `[data-testid=contrast-case]`: label, similarity, `differs_most_in` chips (concept labels).
      Image: `contrast_case.image ?? 'images/'+id+'.jpg'`.
  - **Agent** (§4.5), then **Activity** `[data-testid=case-events]` (timestamped transitions from `/events`).
- Tier-4 note kept: "Routine means read in turn. It does not mean cleared."

### 4.3 Cells tab (`views/cells.js`, CELLS_SPEC §7.3) — `[data-testid=cells-tab]`
- If `cells.status !== 'ok'`, show `[data-testid=cells-disabled]`: "Cell measurements unavailable for this case
  (module disabled or failed)." The tab is always present.
- Otherwise:
  - toggle `input[type=checkbox][data-testid=cells-overlay-toggle]` (default on);
  - legend `[data-testid=cells-legend]`: five swatches with names from `cfg.labels.cell_types`;
  - per `cells.tiles[]`: `figure[data-testid=cells-tile][data-tile=idx]` shows the overlay PNG
    (`assetUrl(overlay)`) when on, else a CSS crop of the display image (`background-size` /
    `background-position` from tile x/y scaled to display width);
  - a counts table per tile;
  - features `[data-testid=cells-feature-<id>]`: display name (`cfg.labels.cell_features`), value + unit, a
    percentile bar with a marker and the text "higher than 95% of benign reference cases", and the contribution;
  - `flags.insufficient_tumor_nuclei` → note;
  - heading wording: "measured features associated with invasive carcinoma".

### 4.4 Surgical case (`views/case_surgical.js`) — `[data-testid=case-view][data-lane=surgical]`
- `[data-testid=sim-banner]` "Simulated frozen section — image borders stand in for inked margins
  (top = superior, bottom = inferior, left = medial, right = lateral)". The variant note covers degraded/composite.
- Big timer `.big.timer[data-testid=surgical-timer][data-timer][data-state]`: amber at `cfg.timer.amber_min`, red
  at `red_min`; "Stopped at mm:ss" after the call.
- Image + SVG overlay (same as §4.2):
  - one `rect.mseg[data-testid=margin-seg][data-margin=superior][data-p][data-involved]` per `margin.segments[]`;
  - fill `--eos` when `p ≥ margin.threshold`; `--amb` when `p ≥ threshold/2`; else a thin `--grn` stroke only;
  - edge labels "Superior 0.81" from `per_margin`; involved margins in bold;
  - "involved ≈ N µm" if the server sends a length.
- Note `[data-testid=margin-2d-note]`: "Anterior/posterior margins can't be shown in a 2D field."
- Reliability row: when the flag is raised, `[data-testid=reliability-flag]` "Reliability flag raised — suggestion:
  defer to permanent sections".
- Chip "Suggested: <call>" (`margin.suggested_call`, worded "suggested").
- `margin === null` → `[data-testid=margin-null]` "Margin read unavailable"; the call buttons still work.
- Buttons:
  - `call-clear` "Margins clear";
  - `call-tumor` "Tumor at margin…" opens checkboxes `margin-pick-<name>` (pre-checked from `margin.involved`)
    and `call-confirm`;
  - `call-defer` "Defer to permanent".
  - Each posts `margin_call {call:'clear'|'tumor_at_margin'|'defer', margins:[…]}`, then `.verdict`
    `[data-testid=verdict]`.
- The agent panel shows only the blocked guard row(s).

### 4.5 Agent panel (biopsy) — `[data-testid=agent-panel]`
- Header "Drafted follow-up orders (illustrative rules; require clinical review). Nothing is ordered until you
  approve." Status pill `[data-testid=agent-status]`.
- Each order is an `.act[data-testid=agent-order][data-rule=rule_id]` showing the type, items, `rule_id`, and
  rationale. Dismissed orders show as `.act.off`; blocked ones as `.act.blk` with the guard reason.
- Buttons:
  - `action-approve` "Order tests";
  - `action-edit` "Edit". It shows a per-item checkbox list; items can only be **removed**, never added (no
    invented stains). `action-edit-save` sends `edit_orders {orders:[{rule_id, items}]}`;
  - `action-closer` "Take a closer look first";
  - `action-dismiss` "Dismiss flag";
  - `action-signout` "Sign out".
- Client enablement table (used when the server sends no `allowed_actions`):

| `agent.status` | approve / edit / closer / dismiss | sign_out |
|---|---|---|
| `none` (no orders) | disabled | enabled |
| `drafted` | enabled | disabled ("act on drafted orders first") |
| `deferred_closer_look` | approve/edit/dismiss enabled, closer disabled | disabled |
| `approved` / `edited_and_approved` / `dismissed` | disabled | enabled |
| signed out | all disabled; `.verdict` "Signed out — routed to oncologist inbox hh:mm" | — |

- Keep the teammate's "Pathologist approval — the only place a diagnosis is made" card heading around Sign out.

### 4.6 Surgeon screen / OR board (`views/surgeon.js`) — `[data-testid=surgeon-screen]`
- `.or` grid (kept). For each active surgical case, `div.card[data-testid=or-card][data-case-id]` shows:
  - room "(simulated)", a `.big` countdown, 3-step `.steps`, and "Simulated frozen section";
  - when routed, the result line `[data-testid=surgeon-result]`, one of: "Margins clear" / "Tumor at
    <names> margin — more tissue needed" / "Cannot determine — defer to permanent sections", plus the time;
  - the timer shows stopped.
- Empty state "No specimen in progress."

### 4.7 Oncologist inbox (`views/oncologist.js`) — `[data-testid=oncologist-inbox]`
- `article.card[data-testid=oncologist-item][data-case-id]` shows: signed-out time, summary sentences (escaped;
  no image), approved/edited orders, final agent status, and the note "Research prototype; suggestions only".
- Empty state.

### 4.8 Metrics (`views/metrics.js`, `js/svg.js`) — `[data-testid=metrics]`
- Each section renders only if its JSON key exists. Otherwise it shows `metrics-missing-<key>` with a `<details>`
  raw-JSON fallback, because Task 11's exact schemas aren't frozen.
- Generic number cell: `{value, ci:[lo,hi], n}` renders as "0.91 (95% CI 0.85–0.96, n=132)". A "source:
  eval/<file>.json" caption sits under each section.
- `svg.js` helpers (inline SVG, CSS-var colors, `viewBox`-responsive, axis ticks + labels, `<title>` tooltips):
  `lineChart`, `barChart(with CI whiskers)`, `reliabilityDiagram(bins)`, `table`.
- Sections:
  - `metrics-headline`: AUROC A vs B, macro-F1, BAcc, InSitu↔Invasive confusion;
  - `metrics-calibration`: reliability diagram pre/post, ECE/Brier/NLL;
  - `metrics-conformal`: coverage + set size, marginal/per-class/APS/Mondrian;
  - `metrics-concepts`: NCT AUROC bars labeled "cross-organ sanity check" + BACH per-class means; contradictions
    surfaced;
  - `metrics-reliability`: accuracy vs severity per degradation;
  - `metrics-retrieval`: precision@5;
  - `metrics-queue-sim`: FIFO vs tiers vs tiers+aging, waits for aggressive and non-flagged cases + assumptions;
  - `metrics-margin`, `metrics-cells`, `metrics-model-choice`, `metrics-thunder` ("same split, different
    protocol");
  - `limitations`: static: simulated frozen lane, small dataset/noisy coverage, rules need clinical review,
    AI-reviewed (not pathologist-reviewed) checks.

### 4.9 Spot-check (`views/spotcheck.js`) — `[data-testid=spotcheck]`
- Built client-side from the loaded cases (no new GET):
  - per concept, the top tiles across cases (CSS crops of display images at `concepts[].top_tiles`);
  - cell overlays from `cells.tiles`;
  - optional margin segments with p ≥ threshold.
- Each `div[data-testid=spot-item][data-kind][data-case-id][data-ref]` has the radios `spot-verdict-ok|not_ok|unsure`.
- Reviewer inputs: name + kind (`human` | `ai`).
- `spot-submit` → `POST /api/spotcheck {reviewer, reviewer_kind, submitted_at, items:[{kind, case_id, ref,
  verdict}]}`. Show the result count. Label the page "qualitative check".

---

## 5. File layout (no build step, no CDN)

```
team44-triage-frontend/
  README.md            # + "Connected to Sepia API" section, run modes, credits the original author's ideas
  app/
    index.html         # shell; #app pre-filled with "Loading… if this stays, serve over HTTP (see README)";
                       # <script type="module" src="app.js">
    app.js             # ENTRY (kept filename, now an ES module): ST, router, draw(), delegation, poll loop, 1 s tick
    styles.css         # original verbatim + appended "Sepia additions"
    js/api.js  js/mock.js  js/adapter.js (toView, TIER, REASON, fmt, rem, esc)  js/svg.js  js/types.js (JSDoc
       typedefs mirroring core/contract.py; header says "keep in sync with core/contract.py + frontend/src/types.ts")
    js/views/{worklist,case_biopsy,case_surgical,heatmap,why,cells,agent,surgeon,oncologist,metrics,spotcheck}.js
    mock/manifest.json  (+ optional mock/metrics.json, synthetic-labeled)
    data/legacy_mock.json   # `git mv data/cases.json` — content untouched, no code reads it
```

Each view exports `render(ST, ctx) → html string` and optionally `bind(root, ST, ctx)`. Imports are relative
(`./js/…`). Module-relative resources use `new URL(…, import.meta.url)`; asset and fixture URLs are
document-relative.

**Backend (Task 6) must serve**, with API routes registered **before** the static mounts:
- `/api/*`;
- `StaticFiles` mounts:
  - `/images` → `artifacts/images` (or `fixtures/` in fixture mode);
  - `/cells` → `artifacts/cells`;
  - `/fixtures` → `fixtures/` (dev, so `?mock=1` also works from the backend);
  - finally `/` → `team44-triage-frontend/app` with `html=True`.
- Call `mimetypes.add_type('text/javascript', '.js')`. A wrong JS MIME type makes module scripts fail silently.
- Send `Cache-Control: no-store` on `/` assets during the demo build (stale modules).
- Add `CORSMiddleware` for `http://localhost:*` only when `PATHQUEUE_DEV_CORS=1`.

---

## 6. Risks and test plan

**Risks → mitigations**

| # | Risk | Mitigation |
|---|---|---|
| 1 | `file://`: module scripts and `fetch` are blocked, so the JS never runs | The static "serve over HTTP" text sits in `#app` before boot. README: `python3 -m http.server` at the **repo root**, then `/team44-triage-frontend/app/?mock=1` |
| 2 | CORS when the app is served separately from the API | Default to same origin (backend serves the app). `?api=` + dev CORS flag as the escape hatch |
| 3 | ES module paths | Imports resolve against the module file, fetch/img against the document. Use `import.meta.url` for mock manifest; never absolute `/js/...` (breaks the repo-root static serve) |
| 4 | XSS | LLM summary, rationale, and ids go through `innerHTML`, so `esc()` everything; finding ids are validated against `/^F\d+$/` |
| 5 | Polling clobbers UI | Fingerprint + timer-only tick + `ST.editing` guard + scroll restore |
| 6 | API drift (R2) | All field access goes through `toView`; T12 reconciles. Tolerate an envelope or a bare array, and `runtime` present or absent |
| 7 | Two tier implementations | Live mode trusts server order; mock never recomputes tiers |
| 8 | Ground-truth leak | The adapter and mock strip `eval_only`. Reference-case labels in Compared-to are train references, so the leak test checks payload keys, not the word "Invasive" |
| 9 | Timer drift | Durations are taken from server `elapsed_s`/`server_time` per fetch, not client wall-clock math alone |
| 10 | Responsiveness | 390 px: card rows; the SVG overlay scales with the image; `.grid` already collapses at 800 px |
| 11 | Fixture paths | Task 1's fixture `image.display` naming is unknown; `asset_map` in the manifest covers mismatches |
| 12 | Mock numbers in screenshots | `banner-mock` + `synthetic` badge. Task 13 screenshots must come from live mode |

**Playwright (`tests/e2e/frontend_smoke.spec.mjs`)**
- Runs against `make demo` (live) and, for CI without Python, `?mock=1` via a static server at the repo root.
- Viewports 390×844 and 1280×800. Both color schemes (`emulateMedia`).
- Assert `document.documentElement.scrollWidth <= innerWidth`.

Plan §14 steps:
1. Click `sim-start` and wait for ≥3 `worklist-row` elements. Note the hero row (`[data-tier="2"]`, latest
   `arrived`). Click `order-arrival` and check the hero is near the bottom; click `order-ai` and check the hero is
   above earlier arrivals and inside the tier-2 group. Check `reason-aggressive_in_conformal_set` is visible.
2. Click the hero row, then check the following in order:
   - `case-view[data-lane=biopsy]`;
   - `heatmap` img loaded; `heatmap-select` → `invasion`;
   - click `why-bar-invasion` → `rect.hl` count = `top_tiles.length`;
   - `src-measured`/`src-estimated` present;
   - `conformal-chip` texts = `[InSitu, Invasive]`;
   - `agent-order` texts contain p63, calponin, ER, PR, HER2, Ki-67;
   - `action-signout` is disabled, then click `action-approve` → `agent-status` = approved;
   - click `action-signout` → `verdict`;
   - `nav-oncologist` → `oncologist-item[data-case-id=<hero>]`.
3. Click `sim-inject-surgical`. The first `worklist-row` is `[data-lane=surgical][data-tier="1"]` with a live
   `[data-timer]` (text changes within 2 s). Open it, then:
   - `sim-banner` visible;
   - `margin-seg[data-margin=superior][data-involved=true]` ≥1;
   - `call-tumor` → `margin-pick-superior` checked → `call-confirm`;
   - `nav-surgeon` → `surgeon-result` contains "superior"; timer `[data-state=stopped]`.
4. Click `sim-inject-degraded` → open → `reliability-flag` → `call-defer` → `surgeon-result` contains "defer".
5. `nav-metrics` → `metrics-queue-sim svg`, `metrics-calibration svg`, `metrics-concepts`, `metrics-reliability`
   (or their `metrics-missing-*` in mock), plus `limitations`.

Extras:
- `sim-inject-composite` → margin segments render;
- `tab-cells` → `cells-overlay-toggle` swaps the img src, `cells-legend` shows 5 swatches, `cells-feature-*`
  (or `cells-disabled`);
- `finding-ref` of kind cells opens `cells-tab`;
- spot-check: set `spot-verdict-ok` ×2, click `spot-submit`, and assert the POST body;
- offline: stop the backend or route-abort `/api/*`, then check `state-offline`;
- **leak**: `page.on('response')` over `/api/*`; the JSON text must contain no `eval_only` or `true_label`, and
  neither may the DOM;
- **adapter unit block**: `page.evaluate(() => import('./js/adapter.js'))` checks:
  - tier labels 1–4;
  - the timer state at 14:59/15:00/20:00/stopped;
  - `eval_only` stripped;
  - a missing `margin`/`cells` gives null/'disabled';
  - `risk_score` 0.72 → 72.

**Requests for other tasks (send via the controller):**
- T6: the shapes in §3.2 (`runtime`, `sim` envelope, `allowed_actions`, `/fixtures` dev mount, MIME/no-store);
- T1: fixture `image.display` path + one surgical fixture with `margin.segments`;
- T12: run the Playwright spec in `make e2e`.
