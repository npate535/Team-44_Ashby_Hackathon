# Ashby Hackathon resource map

Prepared 2026-10-17 for the terminal Claude Code agent in this repository.

## Local, readable without a browser
- official/HACKATHON_COMPUTE.md: current Delta login, data locations, GPU constraints, Lumen and HPC-GPT setup.
- official/STUDENT_GUIDE.md: hosted-model /health, /embed and /classify API.
- official/: official client.py, example_student_usage.py, run_embeddings.py, train_probe.py, model adapters, prompts, tests and Slurm examples. Source commit: 78576d3824c70679053db34458d79cc0df70d259. Full upstream: https://github.com/ncsa/ashby-hackathon-2026 .
- text/2026_Ashby_Hackathon_Opening_Session.md: today's opening deck, 49 slides.
- text/MUSK_Model_Deep_Dive.Claude.md: 28 slides.
- text/CONCH_Model_Deep_Dive.Claude.md: 26 slides.
- text/2026_Ashby_Hackathon_Welcome.md: 30 pre-briefing slides.
- text/2026_Ashby_Hackathon_Resource_Outline.md: older organizer outline, challenge, rubric, contacts and references.
- originals/: corresponding PowerPoint/Word originals, downloaded locally and git-ignored. Text extraction omits chart/image meaning and visual layout; inspect originals where needed.
- organizer-resource-canvas.md: model/dataset/paper/source links from the live Slack resource canvas.

## Terminal quick start
Run from the repository root:

```bash
./scripts/check-resources.sh       # read-only, does not submit jobs or print credentials
./scripts/delta.sh                # first Delta login: NCSA password + Duo
# After login, exit the shell; multiplexed connection lasts an hour.
./scripts/delta.sh 'hostname; whoami; accounts'
python3 scripts/model-api.py --model conch --health
# Small inference example, only once Delta login is working:
# python3 scripts/model-api.py --model conch tile.png > outputs/conch-embedding.json
# Or use images already on Delta:
# python3 scripts/model-api.py --model plip --remote-images /absolute/path/tile.tif
```

The model helper reads credentials from a private file outside Git and sends requests from Delta over SSH. It does not need requests/PyTorch locally, place keys in CLI arguments, or copy keys into shared cluster storage. --prompts prompts.json selects /classify instead of /embed. Use one model per experiment. No inference was run during setup.

## Verified access and remaining gates
- **Local instructions/materials:** present and readable; Claude Code automatically loads the root CLAUDE.md and private CLAUDE.local.md when launched in this folder.
- **DeltaAI:** existing SSH master was live; hostname gh-login02.delta.ncsa.illinois.edu and username rbansal3 verified. This is separate from Delta, and organizers reserve its GPU allocation as backup. No GPU jobs started.
- **Shared datasets:** NCT-CRC, PathGen, PathMMU and actual Quilt roots were readable through that session, along with personal NVMe storage. This is filesystem access, not proof of separate HF gate approval.
- **Delta SSH [blocked until login]:** no authenticated Delta session verified. The helper uses organizer-recommended dt-login02, with dt-login04 as fallback. The first interactive NCSA password + Duo login is needed. Do not use the hanging dt-login03 node reported today.
- **Hosted model APIs [blocked until Delta login]:** all four endpoint credentials saved privately outside this Git repo. They are intended for the Delta network; health requests from DeltaAI timed out. Do not describe this as a server outage. Hosted availability announced through Sunday Oct 18 at 6 PM CDT, not indefinitely.
- **HF gates:** earlier Oct 17 browser checks reported MUSK, CONCH and PathGen-CLIP approved. No default HF CLI token file found locally or on the inspected remote host. Downloading gated weights requires personal CLI authentication. Hosted MUSK/CONCH still require individual license approval.
- **PathMMU gate:** earlier check found a separate form + HF request still required (processing 1-2 days). Do not claim browser access granted.
- **Lumen API verified:** GET /v1/models returned HTTP 200. Credentials are stored outside Git; local path and current available model IDs are in CLAUDE.local.md. Listing models was tested, not inference. Base URL: https://lumen.ncsa.illinois.edu/v1 .
- **Illinois Chat:** NetID-authenticated browser tool, https://chat.illinois.edu/multimodal-pathology/chat . No unattended API integration is claimed.

## Important link corrections
- Working student Box folder: https://uofi.box.com/s/vsl57eshv213e6gzcovj01umzt61wp69 .
- Current opening deck is in that folder (file 2532937965034). The canvas's older standalone opening-slides share returned removed/unavailable on Oct 17; use the downloaded current deck.
- Canvas 'Quilt-1M' HF target is kzhang20/MedicalNarratives, not actual Quilt-1M. Actual Quilt is already on Delta at /work/nvme/bfqi/shared/datasets/quilt/.
- Shared project allocation totals are not a personal compute budget. Delta event rule is 10 GPU-hours per person and one GPU per team at once.

## Updates / help
- Slack canvas: https://app.slack.com/client/T0C8M0NR3J4/unified-files/doc/F0C9DK13A9M .
- #compute-issues: https://app.slack.com/client/T0C8M0NR3J4/C0C7SRT9QJ2 .
- #useful-resources: https://app.slack.com/client/T0C8M0NR3J4/C0C7NKW8ZEF .
- The local snapshots are reference material, not a live Slack feed. Check new organizer announcements before relying on time limits or final submission details.
- Do not push machine-local access notes, secrets, original slide files, large datasets, or generated results. No commit or push was performed by setup.
