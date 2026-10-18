#!/bin/bash
#SBATCH --job-name=pq-embed-ctx
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# UNI2-h on the 512-px degraded variants (all types/severities) and composites (view 0), so every variant read can use
# the production model uni2h512_lr instead of the 256-px fallback (docs/ACCURACY_ROUND2.md). Existing manifest rows only.
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -m pipeline.02_embed --sets deg512,comp512 --encoders uni2h --max-views comp512=1 --device cuda
echo "== embed_ctx done $(date)"
