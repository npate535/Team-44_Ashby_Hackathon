#!/bin/bash
#SBATCH --job-name=pq-cells
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
# cells (Agent 1; CELLS_SPEC §4-§6, BUILD_PLAN §A18/§G): 04a (out-of-fold tile selection from the on-disk CONCH
# probe embeddings, seconds) -> 04b on ~1,200 tiles in cells.shards processes sharing the GPU -> --verify -> 04c
# (CPU features). Resumable: rerunning skips finished tiles. Prereqs: cells_env.sh, embed_core (emb/conch/bach512_probe),
# a passed cells_spike review. Hard stop Sun 10:00 (cells.hard_stop): after that, set cells.enabled false instead.
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
setting() { python -c "from core import config; c = config.load_settings(); v = c
for k in '$1'.split('.'): v = v[k]
print(v)"; }
VENV=$(setting cells.venv); NS=${CELLS_SHARDS:-$(setting cells.shards)}; DATA=$(setting paths.delta.data_root)
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"   # activation without sourcing (set -u safe)
[[ -x "$VENV/bin/python" ]] || { echo "cells venv missing: $VENV (run pipeline/jobs/cells_env.sh)"; exit 2; }
unset PYTHONNOUSERSITE
THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / NS )); (( THREADS >= 1 )) || THREADS=1
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PATHQUEUE_SITE=delta
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS
SCR=${TMPDIR:-/tmp}/cells_${SLURM_JOB_ID:-$$}; mkdir -p "$SCR" "$DATA/logs"; export MPLCONFIGDIR=$SCR/mpl
echo "== cells on $(hostname) at $(date -Is); shards $NS x $THREADS threads"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,numpy,tiatoolbox,zarr;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'numpy',numpy.__version__,'tia',tiatoolbox.__version__,'zarr',zarr.__version__)"
echo "== 04a_select_tiles"
python -m pipeline.04a_select_tiles
echo "== 04b_cells ($NS shards)"
START=$(date +%s)
pids=()
for i in $(seq 0 $((NS - 1))); do
    python -m pipeline.04b_cells --shard "$i" --nshards "$NS" --scratch "$SCR" ${CELLS_LIMIT:+--limit "$CELLS_LIMIT"} \
        > "$DATA/logs/cells_${SLURM_JOB_ID:-$$}_${i}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
END=$(date +%s)
for i in $(seq 0 $((NS - 1))); do echo "-- shard $i (tail)"; tail -n 3 "$DATA/logs/cells_${SLURM_JOB_ID:-$$}_${i}.log" || true; done
verify=0
python -m pipeline.04b_cells --verify ${CELLS_LIMIT:+--limit "$CELLS_LIMIT"} || verify=1
N=$(python -c "import pandas as pd; from core import schemas; print(len(pd.read_csv(schemas.cell_tiles_csv('$DATA'))))")
echo "throughput: $N selected tiles, wall $(( END - START )) s for 04b ($(python -c "print(round(($END - $START) / max(1, $N), 2))") s/tile wall)"
echo "== 04c_cell_features"
python -m pipeline.04c_cell_features ${CELLS_LIMIT:+--limit "$CELLS_LIMIT"}
rm -rf "$SCR"
echo "== cells done at $(date -Is) (shard failures: $fail, verify failures: $verify)"
exit $(( fail || verify ))
