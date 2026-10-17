#!/usr/bin/env bash
# Login-node environment recipe for the encoders (BUILD_PLAN §A22, plan.md §2; Agent 2). Run ONCE by the master:
#
#   bash pipeline/jobs/enc_env.sh            # organizer pip --user installs, pinned to the module's core packages
#   bash pipeline/jobs/enc_env.sh --venv     # optional: also create the `enc` --system-site-packages venv
#                                            # (only if an encoder ever needs a newer timm; none does today)
#
# Login-node safe: pip installs and imports only, no model is loaded. Every command is printed before it runs.
# The module's torch / numpy / timm / transformers (+ hub, tokenizers, safetensors, torchvision) versions are written
# to a constraints file first and every install uses it, so CONCH / MUSK / open_clip can never up- or downgrade them;
# the script then re-checks those versions and fails loudly if anything moved.
set -euo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE

ENV_DIR="$HOME/.config/pathqueue-env"         # private home, never /work/nvme (event-readable)
VENV="/work/nvme/bfqi/rbansal3/venvs/enc"
mkdir -p "$ENV_DIR"
chmod 700 "$ENV_DIR"
CONS="$ENV_DIR/constraints-pytorch-conda-2.8.txt"
PINNED="torch torchvision numpy timm transformers huggingface_hub tokenizers safetensors"

run() { echo "+ $*"; "$@"; }

pins() {
    python - $PINNED <<'PY'
import sys
from importlib import metadata
for name in sys.argv[1:]:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
PY
}

pins > "$CONS"
echo "== constraints from the module ($CONS):"
cat "$CONS"

run python -m pip install --user -c "$CONS" "open_clip_torch>=2.24"
run python -m pip install --user -c "$CONS" git+https://github.com/mahmoodlab/CONCH.git
run python -m pip install --user -c "$CONS" git+https://github.com/lilab-stanford/MUSK.git fairscale sentencepiece
# MUSK declares no install dependencies; make sure the ones it imports exist (encoder_loading.md §2 gotcha b)
for mod in einops safetensors; do
    python -c "import $mod" 2>/dev/null || run python -m pip install --user -c "$CONS" "$mod"
done

echo "== verify pinned versions did not move"
if ! diff <(pins) "$CONS"; then
    echo "ERROR: an install changed a pinned module package (diff above). Undo it before any GPU job." >&2
    exit 1
fi
python - <<'PY'
import importlib
for name in ("open_clip", "conch", "musk", "einops", "fairscale", "sentencepiece", "safetensors", "timm",
             "transformers", "torch"):
    mod = importlib.import_module(name)
    print(f"  import ok  {name:14s} {getattr(mod, '__version__', '')}")
from conch.open_clip_custom import create_model_from_pretrained  # noqa: F401  (import only, nothing loaded)
from musk import modeling, utils  # noqa: F401
PY

if [ "${1:-}" = "--venv" ]; then
    echo "== optional enc venv at $VENV (--system-site-packages: sees the module + the --user installs)"
    run python -m venv --system-site-packages "$VENV"
    "$VENV/bin/python" - $PINNED <<'PY'
import sys
from importlib import metadata
for name in sys.argv[1:]:
    try:
        print(f"  venv sees {name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
PY
    echo "Install an encoder-specific package with: $VENV/bin/python -m pip install -c $CONS <pkg>"
    echo "(drop its own line from a copy of the constraints file first if it must differ from the module)"
fi
echo "enc_env: OK"
