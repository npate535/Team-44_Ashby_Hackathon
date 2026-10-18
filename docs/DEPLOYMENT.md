# Sepia deployment

The frontend and FastAPI API run together on Fly.io as `sepia-ashby`, in Chicago (`ord`). The public domain is https://sepia-ashby.xyz, with https://sepia-ashby.fly.dev as the Fly hostname.

## Cost boundary

This account was deployed using Fly's free trial without adding a payment method. Fly documents a limit of **7 days or 2 total VM hours, whichever comes first**. Trial machines also stop after five minutes. This is a temporary demo deployment, not permanent free hosting. Do not add billing, upgrade, allocate a dedicated IPv4, or increase resources automatically. Once the allowance expires, deployments and the app stop until the account's hosting arrangement changes. See https://docs.fly.io/about/free-trial.

## Code updates

Push application changes to `main`. The `Deploy Sepia` GitHub Actions workflow runs the web runtime tests and deploys one Fly machine, using the encrypted, app-scoped `FLY_API_TOKEN` repository secret. The token expires after one year and must be replaced before then. Cursor credentials are Fly secrets and are never passed through the build context or committed.

The workflow preserves the existing `sepia_data` volume. The first private deployment seeded the real case artifacts into this volume; subsequent builds contain no case dataset. `/data/pathqueue.sqlite`, images, measured evidence and cached interpretations stay on the mounted volume across code updates. Do not destroy or replace this volume. This is one machine with SQLite; do not scale it to multiple machines.

`X-Sepia-Release` on HTTP responses identifies the deployed commit. Check `/api/health` after a deployment and confirm the header matches the workflow's commit. The CI health check uses the Fly hostname so a registrar's DNS propagation delay does not block code updates. Check the custom-domain certificate with `fly certs check sepia-ashby.xyz --app sepia-ashby`. DNS uses an apex A and AAAA record and a `www` CNAME to the Fly hostname.

## Manual deployment

With the correct Fly account authenticated, run:

```sh
fly deploy --remote-only --depot=false --ha=false --build-arg RELEASE_SHA="$(git rev-parse HEAD)"
```

The repository's `deploy/seed/.gitkeep` intentionally holds an empty seed directory. Existing deployments read data from the volume. A completely new volume requires the private, curated case artifact seed before it can start; it must never be committed to Git.
