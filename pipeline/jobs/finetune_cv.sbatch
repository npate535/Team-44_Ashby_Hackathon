#!/bin/bash
#SBATCH --job-name=pq-finetune-cv
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Accuracy round 2 (docs/ACCURACY_ROUND2.md): LoRA UNI2-h trained fold by fold on round 1's CV repeat 0, so the
# fine-tuned model is judged by the same out-of-fold log-loss as every probe. Train split only. Resumable: finished
# folds are skipped, so a resubmission finishes the rest. Submit: pipeline/jobs/submit.sh finetune_cv [--after <id>]
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m pipeline.05c_finetune cv --max-minutes 48 --workers 14
echo "== finetune_cv done $(date)"
