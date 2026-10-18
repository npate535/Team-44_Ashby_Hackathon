#!/bin/bash
#SBATCH --job-name=pq-embed-ctx
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Accuracy round 2 (docs/ACCURACY_ROUND2.md): the 512-px context scale for the 224-native encoders. Train-only CV
# showed UNI2-h on 512-px tiles (view 0 only) beating its 256-px production features; this job gives the 512 scale
# the same 8 dihedral views (train augmentation, cal/test TTA) and the degraded / composite variants the production
# model needs, so a 512-scale winner can be adopted without a variant fallback. Existing manifest rows only.
# Resumable (chunk files); resubmit to finish. Submit: pipeline/jobs/submit.sh embed_ctx [--after <id>]
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m pipeline.02_embed --sets bach512,deg512,comp512 --encoders uni2h --device cuda --keep-going
python -m pipeline.02_embed --sets bach512 --encoders hoptimus0,virchow2 --device cuda --skip-unavailable --keep-going
echo "== embed_ctx done $(date)"
