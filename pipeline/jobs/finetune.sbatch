#!/bin/bash
#SBATCH --job-name=pq-finetune
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Accuracy round 2 (docs/ACCURACY_ROUND2.md; BUILD_PLAN §10): LoRA UNI2-h on all 218 train images, then re-extracted
# uni2h_ft features for every set the pipeline reads at 512 px (8 views; degraded; composites). LoRA weights stay in
# private home (CC-BY-NC-ND derivative). Submit: pipeline/jobs/submit.sh finetune [--after <id>]
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m pipeline.05c_finetune final --max-minutes 22 --workers 14
python -m pipeline.02_embed --sets bach512,deg512,comp512 --encoders uni2h_ft --device cuda
echo "== finetune done $(date)"
