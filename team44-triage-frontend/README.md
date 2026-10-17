# Team 44 Triage Worklist (frontend)

Frontend for Team 44's Ashby Prize Hackathon 2026 plan: a pathology worklist that triages cases with a foundation model, prepares workups with a guarded agent, and keeps the pathologist in charge.

Plan: https://team44-triage-plan.rishabh-rb.workers.dev/

Research prototype on public, de-identified data. Not clinically validated.

## Screens

- **Worklist**: arrival order vs AI priority toggle, tiers, timers, aging promotion at 75% of target, task list
- **Case view**: tile mosaic, concept words, look-alikes, calibrated risk, conformal set, reader agreement, agent actions with cancel, approval buttons
- **OR board**: surgeon-facing status and countdown per operating room

## Run locally

The app fetches JSON, so serve the `app` folder over HTTP:

```bash
cd app
python3 -m http.server 8000
```

Open http://localhost:8000. In VS Code you can also use the Live Server extension on `app/index.html`.

## Layout

```
app/
  index.html        page shell
  styles.css        styles (light and dark)
  app.js            worklist, case view, OR board, priority rule
  data/cases.json   case records (mock, replace with cached model/agent output)
```

## Swapping in real outputs

Each object in `cases.json` holds the case record plus model output. Replace the mock values with the cached JSON from the model and agent jobs, keeping field names, or adjust `app.js` to match the frozen contract. Do not show any mock number in a screenshot or slide.

## Priority rule (in `app.js`, function `sorted`)

1. Frozen-section cases first, by minutes left
2. Then cases that have used 75% of their target
3. Then tier: urgent, unsure, quality, routine
4. Then share of target used

## Deploy to GitHub Pages

Push the repo, then in Settings > Pages publish from the branch with the `/app` folder, or move `app/` contents to `/docs`.

## Connected to the PathQueue API (Team 44, Oct 17 2026)

The original vanilla-JS app (worklist, case view, OR board; author: adas286, Team 44) is now connected to the
PathQueue backend (`backend/`, `docs/API.md`) and extended for breast/BACH. Its ideas are kept and credited:
the **arrival-order vs AI-priority toggle**, the **aging promotion at 75% of target** (now computed server-side by
`core.tiers`), the **OR board** (surgeon screen with room and countdown), **visibly blocked agent actions** (no stains
in the frozen lane), the Tasks card, "Routine does not mean cleared", and the "Pathologist approval — the only place a
diagnosis is made" card. Its look (tokens, Plex/Iowan type, light/dark) is unchanged; new styles are appended at the
end of `styles.css`.

Run modes:

```bash
make demo CASES=fixtures                       # backend serves this app at http://127.0.0.1:8000
uv run python -m backend.dev_cases --out artifacts/dev && make demo ART=artifacts/dev   # 18 synthetic demo cases
python3 -m http.server 8000                    # from the REPO ROOT, then open
                                               # http://localhost:8000/team44-triage-frontend/app/?mock=1  (no backend)
```

`?mock=1` replays the committed synthetic fixtures in memory (`app/mock/`); `?api=http://host:port` points at a
backend served elsewhere (set `PATHQUEUE_DEV_CORS=1` on it). No external CDNs: the demo runs offline.

Layout added:

```
app/js/api.js        live client (polling every ui.poll_ms, backend-down state)
app/js/mock.js       ?mock=1 in-memory backend over fixtures (same method surface)
app/js/adapter.js    toView(): every contract field the UI reads; teammate helpers fmt/rem kept
app/js/types.js      JSDoc mirror of core/contract.py / frontend/src/types.ts
app/js/views/        worklist, case_biopsy, case_surgical, heatmap, why, agent, cells, surgeon, oncologist,
                     metrics, spotcheck
app/mock/            manifest.json (fixtures + patches), config.json (snapshot of GET /api/config, test-checked)
app/data/legacy_mock.json   the original colon/multi-organ mock (data/cases.json), untouched, no longer read
```

Wording on every screen: "suggested", "associated with", "simulated", "Research prototype — not for clinical use".
Ground truth is never fetched or rendered.
