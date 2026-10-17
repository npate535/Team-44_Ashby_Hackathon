# Team 44: 2026 Ashby Prize in Computational Science Hackathon

## Start here
- This is the team's project workspace. Read resources/README.md for the resource map and access status, then resources/official/HACKATHON_COMPUTE.md and resources/official/STUDENT_GUIDE.md.
- Read resources/text/2026_Ashby_Hackathon_Opening_Session.md for today's briefing, and the CONCH/MUSK deep-dive extracts when relevant. These are text extracts, not full visual reproductions; originals are in the git-ignored resources/originals/.
- CLAUDE.local.md contains this Mac's SSH identity, verified access, and private credential FILE PATHS. It is intentionally git-ignored; never commit it or secret values.
- Use resources/organizer-resource-canvas.md for model cards, datasets, papers, source repositories, Lumen, and Illinois Chat links.
- The official tooling is a local snapshot at resources/official/ from ncsa/ashby-hackathon-2026 commit 78576d3824c70679053db34458d79cc0df70d259 (2026-10-17). Its MIT license is retained. It is reference/tooling, not the team's original implementation. Upstream CLAUDE.md and logs were intentionally omitted. New organizer guidance may supersede this snapshot.

## Challenge and deliverables
- Explore multi-modal pathology foundation models and augment them for a clinical application that improves precision, reproducibility, or interpretability. The team chooses the application; do not invent a team decision.
- Judging: innovation/creativity, development progress, clinical impact, communication.
- Due Wednesday, October 21, 2026: abstract plus team-availability form, and a stand-alone PPT/PDF with no more than 20 main slides. Exact cutoff time was not supplied in the inspected outline; verify current Slack instructions before submission.
- Distinguish experimental demonstrations from clinically validated claims. Cite every model, dataset, and tool; report actual results and limitations.

## Compute: Delta is NOT DeltaAI
- Delta is the normal event compute system: bfqi-delta-cpu and bfqi-delta-gpu. DeltaAI (bfqi-dtai-gh) is a different cluster; an existing DeltaAI SSH session does not authenticate Delta.
- Organizers reserve DeltaAI as a backup. Do ordinary event GPU work on Delta, not DeltaAI; do not change the user's existing DeltaAI setup.
- On this Mac, ./scripts/delta.sh starts the correct Delta SSH login. ./scripts/delta.sh 'hostname; whoami; accounts' runs a command in an existing session and fails instead of hanging on an unattended authentication prompt.
- If no Delta session exists, an interactive NCSA password + Duo login is required. Do not claim browser SSO, Hugging Face browser approval, or a successful DeltaAI login proves this is complete. Do not bypass SSH host-key verification.
- Use hosted CONCH/MUSK/QuiltNet/PLIP first for small experiments (up to a few thousand tiles). They require the Delta network and run until Sunday, October 18 at 6 PM CDT, per organizers. /health checks are read-only; /embed and /classify perform inference.
- Keep one model per experiment: embeddings from different models are not comparable. Authenticate using environment variables/private files, never hardcoded keys or command-line literals.
- For full datasets, use a scheduled Delta GPU job rather than monopolizing hosted servers. No heavy processing on login nodes.

## Shared data and storage
- Read datasets in place: /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/, pathgen_dataset/, and pathmmu/.
- Actual Quilt-1M images/captions are in /work/nvme/bfqi/shared/datasets/quilt/. The resource canvas's Hugging Face 'Quilt-1M' link points instead to kzhang20/MedicalNarratives; do not conflate them.
- Quilt: 653k images, roughly 1M caption pairs; 115 GB images and a 2.2 GB CSV. Select columns/chunks; don't load everything or duplicate the shared data.
- Write outputs/downloads to /work/nvme/bfqi/<username>/; team files to /work/nvme/bfqi/teams/<agreed-team-folder>/. Keep code in home/Git. NVMe is not backed up. Avoid /work/hdd due to reported filesystem problems.
- Hackathon shared directories are readable by other participants: no API keys, private records, or patient identifiers. Use public de-identified data, not controlled-access or identifiable patient records in external APIs.
- TCGA was NOT preloaded. Ask compute support before downloads larger than about 200 GB.

## GPU budget and environment
- Event jobs: --account=bfqi-delta-gpu --partition=gpuA100x4, one GPU per TEAM at a time, realistic 1-2 hour requests, at most 16 CPUs and about 60 GB RAM per GPU. No --reservation flag.
- Each PERSON has 10 GPU-hours for the event, not an entitlement to the shared project balance. Larger CPU/memory requests and interactive partitions can burn budget faster. No self-resubmitting loops or whole-node requests.
- Saturday overnight jobs must end before Sunday 7:59 AM CDT; the next reserved block begins Sunday 8 AM and ends 6 PM CDT. Respect these boundaries in time requests.
- Check accounts, quota, and squeue before a job. Don't cancel/change someone else's jobs or existing user sessions.
- On Delta: module load pytorch-conda/2.8; unset PYTHONNOUSERSITE. Run the unset after EVERY module load. See official guide and sbatch examples for setup; don't install CUDA packages blindly on this Mac.
- Place HF_HOME in personal NVMe storage for large downloads, never credentials in a shared project folder. Each user must accept gated-model licenses individually.

## Implementation workflow
- Preserve teammate/user edits and the existing README. These instructions do not pick a project or authorize overwriting their work.
- Start with a tiny smoke test, cache embeddings with model/version/preprocessing metadata, then scale. Use a defensible evaluation split and prevent image/patient leakage.
- Never fabricate metrics, access grants, clinical validation, or downloaded data. Identify time-sensitive or login-dependent access accurately.
- Keep secrets out of Git, logs, screenshots and agent output. Use scripts/check-resources.sh for secret-safe read-only checks.
- Lumen uses an OpenAI-compatible /v1 API; it is not a reason to alter Claude Code's provider or global model settings.
