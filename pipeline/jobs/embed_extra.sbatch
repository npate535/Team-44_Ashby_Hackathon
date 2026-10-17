#!/bin/bash
#SBATCH --job-name=pq-embed-extra
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Extra encoders for the black-box ceiling and the fine-scale margin probe (BUILD_PLAN §A8, §A12, §G ~0.9 GPU-h).
# Priority order: UNI2-h and H-optimus-0 (the margin-probe encoders, settings encoders.margin_probe_encoders) on
# bach256 (4 dihedral views), bach512 (view 0 only) and deg256; then Virchow2, Phikon-v2 and PathGen-CLIP on
# bach256 + bach512 as time allows (cut order §I item 2). Encoders without cached weights are skipped. Resumable:
# if the time limit hits, resubmit and it finishes the rest. Submit: pipeline/jobs/submit.sh embed_extra --after <id>
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m pipeline.02_embed --sets bach256,bach512,deg256 --max-views bach512=1 --encoders uni2h,hoptimus0 --device cuda \
    --skip-unavailable --keep-going
python -m pipeline.02_embed --sets bach256,bach512 --max-views bach512=1 --encoders virchow2,phikonv2,pathgenclip --device cuda \
    --skip-unavailable --keep-going
echo "== embed_extra done $(date)"
