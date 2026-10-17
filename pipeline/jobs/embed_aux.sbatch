#!/bin/bash
#SBATCH --job-name=pq-embed-aux
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Auxiliary embeddings (BUILD_PLAN §G ~0.5 GPU-h): composites (comp512 CONCH/MUSK, comp256 margin-probe encoders),
# WSI tiles and margin windows (wsi512 CONCH/MUSK; wsi256 + wsi_margin256 margin-probe encoders), and the mined
# Quilt-1M / PathGen concept images (CONCH align only, for 04_cav). Sets missing from the manifest are reported as
# "no rows" and skipped (e.g. pathgen when inventory finds no patch images). Needs prep_b's appended manifest.
# Submit: pipeline/jobs/submit.sh embed_aux --after <embed_extra id>
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m pipeline.02_embed --sets comp512,wsi512,quilt,pathgen --encoders conch --device cuda
python -m pipeline.02_embed --sets comp256,wsi256,wsi_margin256 --encoders uni2h,hoptimus0 --device cuda --skip-unavailable --keep-going
python -m pipeline.02_embed --sets comp512,wsi512 --encoders musk --device cuda
echo "== embed_aux done $(date)"
