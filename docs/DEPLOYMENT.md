# Sepia deployment

Sepia runs on Vercel in the **Sepia** team (`sepia2`), project `sepia-ashby`.
Production: https://sepia-ashby.xyz and https://sepia-ashby.vercel.app.
The Vercel Pro trial runs through November 1, 2026.

## Runtime and persistence

The frontend and immutable tissue images are served by Vercel's CDN. FastAPI
runs in `iad1` using Python 3.13. Model embeddings and training remain offline.
The private upload includes real case artifacts, measured cell evidence,
evaluation results, and cached interpretations. Case/evaluation JSONs continue
through the API's ground-truth filtering; only image assets go in `public/`.

Vercel's filesystem is temporary. `backend/vercel.py` keeps decisions, routing,
events, replay-clock state, edited spot checks, and new interpretations in the
connected **sepia-workspace** Upstash Redis database (free plan). Each API
request restores the shared workspace under a Redis lease; the response is sent
after an atomic save verifies ownership. Immutable case JSONs are excluded from
the saved SQLite snapshot and restored from packaged evidence. The initial
snapshot is about 48 KB. Changed case evidence requires an explicit workspace
migration; it must not silently reset saved decisions. Do not delete Redis to
deploy code.

The October 18 variant update explicitly starts `sepia:workspace:v2` with
the rebuilt UNI2-h 512-pixel case evidence. The previous `v1` workspace is
retained for recovery; no recorded routed results existed at migration time.
Deployment packages never copy the running laptop demo's SQLite database.

### October 18, 2026 (evening): look-alike detail and spot-checked summaries need workspace v3

Branch `jf/t4-contrast` changes the case evidence in two steps, both before
any deploy:

- Every one of the 328 case JSONs gains `contrast_case.detail`.
- Each concept finding gains `spotcheck` and `reliable`. The template
  summaries of 277 cases change, because concepts that failed their AI spot
  check are no longer given as reasons.

Predictions, flags, tiers, drafted orders and everything else are identical
to the previous version. `restore()` accepts a saved snapshot
only when its `evidence_hash` (SHA-256 of the sorted public case JSONs)
matches the packaged cases. Left on `v2`, every API request would raise and
return 503, including the deploy workflow's `/api/health` check. The default
key in `backend/vercel.py` is therefore `sepia:workspace:v3`, which starts
from the seed snapshot of the new evidence. `v2` stays in Redis, untouched,
for recovery. Nothing was deployed and Redis was not read or written when
this change was prepared.

Evidence hashes, computed locally with `create_app` over the packaged JSON
sections:

| Workspace | Case evidence | `evidence_hash` |
|---|---|---|
| `v2` | commit `ef43467` (unchanged since `ea008ee`) | `7f3f72e5b5fe52dd6149f4c6c1e6a96c5235b0dcbb426845b1b9e885aa5c52a7` |
| `v3` | with `contrast_case.detail`, concept-finding spot checks and the new summaries | `3baa496a29f194e44022034bb82c5fa7d1f50cca161b9f138ba85953fa6678be` |

An intermediate hash, `71f89232…`, covered the contrast detail alone. It was
never deployed and does not apply.

Before pushing this change to `main`:

1. Confirm that the Vercel project does not set `SEPIA_WORKSPACE_KEY`. An
   explicit value overrides the new default and keeps the site on the old
   evidence check.
2. Read `sepia:workspace:v2` without writing to it. Decode it with base64,
   then zlib, then JSON; the `sqlite` field is a base64 SQLite image. Count
   the rows in `decisions`, `orders` and `routed_results`. If they are empty,
   as at the `v2` migration, deploy as is.
3. If they hold reviews that must stay visible, migrate explicitly before the
   deploy. The evidence change is display data only (detail rows, spot-check
   fields and summary text), so re-encode the same
   snapshot with the `v3` hash above and `SET` it as `sepia:workspace:v3`.
   Never delete the `v2` key.
4. If another case-evidence change is merged before this push (for example,
   concept label edits), it shares the `v3` bump; recompute the `v3` hash
   from the final cases. A case change after `v3` is live needs `v4`.

The laptop demo's saved SQLite workspace rejects the new cases for the same
reason. Start it with `make demo-fresh`, which archives the old database.

Cursor's Linux CLI is compressed in the function bundle and extracted into
`/tmp` when a new interpretation is requested. Cached interpretations need no
model call. `CURSOR_API_KEY`, `CURSOR_API_KEY_BACKUP`, and
`PATHQUEUE_INTERPRET_MODEL` are encrypted Vercel environment variables. Redis
credentials come from the connected integration. Never commit tokens,
databases, `.env` files, or private deployment uploads.

## Code updates

Push to `main`. `.github/workflows/deploy.yml` verifies the web runtime,
packages the current case assets, and deploys to Vercel using the encrypted
`VERCEL_TOKEN` GitHub repository secret. `scripts/package_vercel.py` builds a
dedicated web Python project without the offline Torch/training dependencies.
`X-Sepia-Release` identifies the source commit.

For an authenticated manual deployment:

```sh
uv run python scripts/package_vercel.py --out /tmp/sepia-vercel
cd /tmp/sepia-vercel
vercel link --project sepia-ashby --scope sepia2
vercel deploy --prod
curl --fail https://sepia-ashby.vercel.app/api/health
```

The staging directory contains private case data. Keep it outside Git.

## DNS and former Fly deployment

The registrar is gen.xyz; authoritative DNS uses topdns.com nameservers.
Vercel recommends apex A records `216.198.79.1` and `64.29.17.1`, with
`www` CNAME `96e160bdaf0f8bbf.vercel-dns-017.com`. Remove the former Fly A
and AAAA records when switching; preserve unrelated mail/TXT records. Check
Vercel's Domains page for current recommendations.

The former Fly app `sepia-ashby` and `sepia_data` volume are retained for
recovery. The machine is stopped and GitHub Actions no longer deploys to Fly.
The volume has not been destroyed or replaced. Its latest review state could
not be read while the exhausted trial machine was stopped; the Vercel workspace
was seeded from the matching local artifact workspace.
