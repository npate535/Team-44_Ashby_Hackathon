#!/bin/bash
#SBATCH --job-name=pq-prep-b
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
# prep_b (Agent 1, BUILD_PLAN Task 2 / §G): composites -> WSI tiles + margin windows (`wsi` venv) -> Quilt/PathGen
# caption mining. Each step rewrites only its own manifest sets. Needs prep_a's outputs (cases.csv, inventory.json).
# Composites are never cut; WSI and caption mining are cuttable (BUILD_PLAN §I items 3, 6): their failure is
# reported and makes the job exit 1 at the end, but does not stop the remaining steps.

cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PATHQUEUE_SITE=delta
WSI_VENV=${WSI_VENV:-/work/nvme/bfqi/rbansal3/venvs/wsi}
echo "== prep_b on $(hostname) at $(date -Is); code $(pwd); cpus ${SLURM_CPUS_PER_TASK:-?}"
declare -A STATUS

echo "== composites (smoke --limit 5, then full)"
python -m pipeline.composites --limit 5
python -m pipeline.composites
STATUS[composites]=ok

echo "== wsi_extract (wsi venv: $WSI_VENV)"
if [[ -x "$WSI_VENV/bin/python" ]]; then
    if "$WSI_VENV/bin/python" -m pipeline.wsi_extract; then
        STATUS[wsi_extract]=ok
    else
        STATUS[wsi_extract]="FAILED (exit $?)"
    fi
else
    STATUS[wsi_extract]="SKIPPED: no venv at $WSI_VENV (run pipeline/jobs/wsi_env.sh on a login node)"
fi

echo "== mine_captions (one pass over the Quilt CSV; --limit would only cap the selection, not the scan)"
if python -m pipeline.mine_captions; then
    STATUS[mine_captions]=ok
else
    STATUS[mine_captions]="FAILED (exit $?)"
fi

echo "== summary"
for step in composites wsi_extract mine_captions; do echo "$step: ${STATUS[$step]}"; done
python - <<'PY'
import pandas as pd
from core import config, schemas
root = config.data_root(config.load_settings())
m = pd.read_csv(schemas.embed_manifest_csv(root), dtype=str, keep_default_na=False, usecols=["set", "label"])
print("manifest rows per set:", m["set"].value_counts().sort_index().to_dict())
q = m[m["set"].isin(["quilt", "pathgen"])]
if len(q):
    print("concept images per label:", q.groupby(["set", "label"]).size().to_dict())
PY
echo "== prep_b done at $(date -Is)"
for step in composites wsi_extract mine_captions; do [[ "${STATUS[$step]}" == ok ]] || exit 1; done
