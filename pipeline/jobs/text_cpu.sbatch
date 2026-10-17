#!/bin/bash
#SBATCH --job-name=pq-text-cpu
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Re-encode the prompt bank on CPU after a config/concepts.yaml edit (03_text; no GPU time). Text towers only do
# light work; MUSK runs fp32 on CPU. Up-to-date npz files are skipped. Submit: pipeline/jobs/submit.sh text_cpu
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
python -m pipeline.03_text --device cpu --encoders conch,musk,pathgenclip --skip-unavailable
echo "== text_cpu done $(date)"
