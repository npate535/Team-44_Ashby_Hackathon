#!/usr/bin/env bash
# Login-node pre-download of every encoder's weights into the PRIVATE home HF cache (BUILD_PLAN §A20; Agent 2).
#
#   bash pipeline/jobs/hf_download.sh                     # every encoder in config/encoders.yaml
#   bash pipeline/jobs/hf_download.sh conch,musk          # a subset
#
# Run from the code directory (~/pathqueue). Needs an existing `hf auth login` (read token in
# ~/.cache/huggingface/token); the token is never printed or passed on a command line. Files are fetched at the
# default revision `main` with the exact repo casing the loaders use (a pinned revision would not write refs/main and
# offline hf-hub loads would then fail). Writes $HF_HOME/team44_encoder_manifest.json (repo, file, commit, bytes) and
# finishes with an offline-readiness check (HF_HUB_OFFLINE=1: every file must resolve from the cache alone).
# I/O only: no model is loaded, so this is allowed on a login node. Gated repos the account cannot access yet are
# reported as GATED and the script exits 1 after downloading everything else.
set -euo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME="$HOME/.cache/huggingface"      # MUST equal the job scripts' HF_HOME
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
case "$HF_HOME" in /work/*|/projects/*) echo "HF_HOME must be in private home, not $HF_HOME" >&2; exit 1 ;; esac
mkdir -p "$HF_HOME"
chmod 700 "$HF_HOME"

ONLY="${1:-}"
set +e
python -I - "$ONLY" <<'PY'
import json, os, sys
import yaml
from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.errors import GatedRepoError

if not get_token():
    sys.exit("No Hugging Face token found. Run `hf auth login` first (the token is never printed).")
ecfg = yaml.safe_load(open("config/encoders.yaml"))
only = [s for s in sys.argv[1].split(",") if s]
manifest, failed = {}, []
for name, spec in ecfg["encoders"].items():
    if spec["loader"] == "official_test" or (only and name not in only):
        continue
    for fn in spec["files"]:
        repo = spec["repo"]
        try:
            path = hf_hub_download(repo_id=repo, filename=fn)
            commit = path.split("/snapshots/")[1].split("/")[0]
            size = os.path.getsize(path)
            manifest.setdefault(name, {"repo": repo, "files": {}})["files"][fn] = {"commit": commit, "bytes": size}
            print(f"OK    {name:12s} {repo}/{fn}  {size / 1e9:.2f} GB  @{commit[:7]}")
        except GatedRepoError:
            failed.append(f"{repo}/{fn}")
            print(f"GATED {name:12s} {repo}: request/accept access on huggingface.co with THIS account, then rerun")
        except Exception as exc:  # huggingface_hub errors never contain the token
            failed.append(f"{repo}/{fn}")
            print(f"FAIL  {name:12s} {repo}/{fn}: {type(exc).__name__}: {str(exc)[:200]}")
out = os.path.join(os.environ["HF_HOME"], "team44_encoder_manifest.json")
old = json.load(open(out)) if os.path.exists(out) else {}
old.update(manifest)
with open(out, "w") as handle:
    json.dump(old, handle, indent=1)
print(f"revision manifest -> {out}")
sys.exit(1 if failed else 0)
PY
RC_DOWNLOAD=$?
set -e

echo "== offline-readiness check (HF_HUB_OFFLINE=1)"
set +e
HF_HUB_OFFLINE=1 python - "$ONLY" <<'PY'
import sys
from pipeline import encoders as E
ecfg = E.load_encoders_config()
only = [s for s in sys.argv[1].split(",") if s]
bad = 0
for name, spec in ecfg["encoders"].items():
    if spec["loader"] == "official_test" or (only and name not in only):
        continue
    missing = E.missing_files(spec)
    bad += bool(missing)
    print(f"{'READY  ' if not missing else 'MISSING'} {name:12s} {', '.join(missing) if missing else E._revision(spec)}")
sys.exit(1 if bad else 0)
PY
RC_OFFLINE=$?
set -e
du -sh "$HF_HOME/hub" 2>/dev/null || true
[ "$RC_DOWNLOAD" -eq 0 ] && [ "$RC_OFFLINE" -eq 0 ] && echo "hf_download: OK" || { echo "hf_download: INCOMPLETE (see GATED/FAIL/MISSING above)"; exit 1; }
