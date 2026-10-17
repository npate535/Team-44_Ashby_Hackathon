# Team 44 Triage Worklist (frontend)

Frontend for Team 44's Ashby Prize Hackathon 2026 plan: a pathology worklist that triages cases with a foundation model, prepares workups with a guarded agent, and keeps the pathologist in charge.

Plan: https://team44-triage-plan.rishabh-rb.workers.dev/

Research prototype on public, de-identified data. Not clinically validated. All cases in `app/data/cases.json` are **mock data** until replaced with cached model outputs.

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
