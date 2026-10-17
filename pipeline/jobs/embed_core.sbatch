#!/bin/bash
#SBATCH --job-name=pq-embed-core
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Core embeddings (BUILD_PLAN §G ~0.8 GPU-h): CONCH + MUSK, one forward pass feeding both heads, on bach512
# (train items keep their 8 dihedral views in *_views.npy; every item is TTA-averaged on the node), nct and deg512,
# then the prompt bank (03_text). Resumable: a killed or timed-out job continues from its float16 chunks when
# resubmitted. Needs Agent 1's embed_manifest.csv (prep_a). Submit only through: pipeline/jobs/submit.sh embed_core
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# CONCH first (the primary encoder: concepts, retrieval, OOD), then the prompt bank (seconds on a GPU), then MUSK, so
# a time-limit kill during MUSK still leaves CONCH + text complete; resubmitting finishes MUSK from its chunks.
# MUSK's image/text alignment is checked inside 02_embed (NCT zero-shot with the adapter's own text encoder).
python -m pipeline.02_embed --sets bach512,nct,deg512 --encoders conch --device cuda
python -m pipeline.03_text --encoders conch,musk,pathgenclip --device cuda --skip-unavailable
python -m pipeline.02_embed --sets bach512,nct,deg512 --encoders musk --device cuda
echo "== embed_core done $(date)"
