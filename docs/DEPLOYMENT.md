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
