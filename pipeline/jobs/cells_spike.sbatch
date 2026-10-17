#!/bin/bash
#SBATCH --job-name=pq-cells-spike
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
# cells_spike (Agent 1; CELLS_SPEC §2 C0, BUILD_PLAN §A18): segment cells.spike_tiles tiles from different cases with
# the configured backend, write overlays + artifacts/eval/cells_spike.md (review template) and log seconds per tile
# for the throughput gate. Needs prep_a outputs (tiles.csv) only; no embeddings. Prereq: pipeline/jobs/cells_env.sh.
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
VENV=$(python -c "from core import config; print(config.load_settings()['cells']['venv'])")
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"   # activation without sourcing (set -u safe)
[[ -x "$VENV/bin/python" ]] || { echo "cells venv missing: $VENV (run pipeline/jobs/cells_env.sh)"; exit 2; }
unset PYTHONNOUSERSITE
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PATHQUEUE_SITE=delta
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
SCR=${TMPDIR:-/tmp}/cells_spike_${SLURM_JOB_ID:-$$}; mkdir -p "$SCR"; export MPLCONFIGDIR=$SCR/mpl
echo "== cells_spike on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,numpy,tiatoolbox,zarr;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'numpy',numpy.__version__,'tia',tiatoolbox.__version__,'zarr',zarr.__version__)"
python -m pipeline.04b_cells --spike --scratch "$SCR"
rm -rf "$SCR"
echo "== overlays: cells/<case_id>/<tile_idx>_overlay.png under the data root; template: artifacts/eval/cells_spike.md"
echo "== cells_spike done at $(date -Is)"
