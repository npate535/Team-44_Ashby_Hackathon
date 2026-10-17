#!/bin/bash
#SBATCH --job-name=pq-smoke
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Smoke test for every available encoder (BUILD_PLAN §E step 4-5, §G: ~0.3 GPU-h). Needs no Agent 1 output: it builds
# its own tiny manifest (72 NCT tiles + up to 4 readable BACH images tiled 512 x 2 views + degraded rows when
# pipeline.degrade is implemented) under <data root>/smoke and runs every check: versions, dims, align norms,
# align == official adapter (parity), autocast vs fp32, NaN/Inf, NCT zero-shot vs the encoders.yaml floor,
# throughput. Encoders whose weights are not in the HF cache are skipped (gated access not granted yet).
# Submit only through: pipeline/jobs/submit.sh smoke
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
SMOKE=/work/nvme/bfqi/teams/team44/pathqueue/smoke
ENCODERS=conch,musk,uni2h,hoptimus0,virchow2,phikonv2,pathgenclip

echo "== host $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python - <<'PY'
import importlib, platform
print("python", platform.python_version())
for name in ("torch", "torchvision", "numpy", "timm", "transformers", "huggingface_hub", "open_clip", "conch", "musk",
             "einops", "fairscale", "sentencepiece", "safetensors", "pandas", "sklearn", "PIL", "yaml"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name:16s} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:  # report every missing package, then let the encoder loads fail loudly
        print(f"  {name:16s} MISSING ({type(exc).__name__}: {exc})")
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

rm -rf "$SMOKE/emb" "$SMOKE/text" "$SMOKE/meta"   # only this run's results are reported below
python -m pipeline.02_embed --data-root "$SMOKE" --smoke-manifest
set +e
python -m pipeline.02_embed --data-root "$SMOKE" --sets nct,bach512,deg512 --encoders "$ENCODERS" --device cuda \
    --skip-unavailable --keep-going --precision-check --force 2>&1 | tee "$SMOKE/embed_run.log"
RC=$?
set -e
echo "== QuickGELU A/B for PathGen-CLIP (comparison only; compare the two nct zero-shot accuracies)"
python -m pipeline.02_embed --data-root "$SMOKE" --sets nct --encoders pathgenclip_quickgelu --device cuda \
    --skip-unavailable --force || echo "pathgenclip_quickgelu A/B failed (comparison only)"
if [ -f pipeline/03_text.py ] && [ -f config/concepts.yaml ]; then
    set +e
    python -m pipeline.03_text --data-root "$SMOKE" --encoders conch,musk,pathgenclip --device cuda --skip-unavailable --force
    RC_TEXT=$?
    set -e
    [ "$RC" -ne 0 ] || RC=$RC_TEXT
    echo "== early prompt-bank signal: NCT cross-organ concept AUROCs per text-aligned encoder (+ per phrasing)"
    python -m pipeline.04_concepts --data-root "$SMOKE" --nct-only || echo "nct-only concept check failed (diagnostic only)"
fi
echo "== meta per encoder (dims, checks, throughput)"
for meta in "$SMOKE"/emb/*/meta.json; do
    python - "$meta" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["encoder"], "rev", m["revision"][:12], "compute", m["compute_dtype"], "heads", m["heads"])
for s, info in m["sets"].items():
    print("   ", s, "items", info["n_items"], "views", info["n_views"], "img/s", info["throughput_img_s"],
          "checks", m["checks"].get(s))
PY
done
echo "== weight-loading warnings (MUSK prints missing/unexpected keys instead of raising; CONCH loads strict=False)"
# MUSK's "Ignored weights of ... not initialized" line lists the deliberately ignored relative_position_index buffers
# (musk/utils.py load_state_dict ignore_missing): benign, so it is shown but not counted.
grep -i "^Ignored weights of" "$SMOKE/embed_run.log" || true
if grep -iE "missing key|unexpected key|not initialized from pretrained|not used in" "$SMOKE/embed_run.log" \
        | grep -iv "^Ignored weights of"; then
    echo "WEIGHT-LOADING WARNINGS above: inspect them before any full embedding job (exit code 3)"
    [ "$RC" -ne 0 ] || RC=3
else
    echo "none"
fi
echo "== smoke exit code $RC (0 ok, 2 a check failed or nothing embedded, 3 weight-loading warnings)"
exit "$RC"
