#!/bin/bash
# LOGIN NODE ONLY (the master runs it once; Agent 1 never runs it): `cells` venv with TIAToolbox 2.1.3 + the
# hovernet_fast-pannuke weights, then an offline-readiness check. No inference here.
#   bash pipeline/jobs/cells_env.sh                  # build venv + weights + check
#   DRY_RUN=1 bash pipeline/jobs/cells_env.sh        # print the commands only
#   WITH_FALLBACK=1 bash pipeline/jobs/cells_env.sh  # also clone+patch vqdang/hover_net (fallback backend)
# Paths, weights repo/revision/sha256 come from config/settings.yaml (cells block). Recipe: docs/research/
# cells_backends.md §5. --system-site-packages venv; the module's torch stack and numpy (and timm/transformers/hub
# when inside TIAToolbox's ranges) are pinned via a generated constraints file so CONCH/MUSK never break.
# Weights are public (not gated): no token is needed or read. PanNuke-derived weights: CC BY-NC-SA 4.0.
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
run() { echo "+ $*"; if [[ -z "${DRY_RUN:-}" ]]; then "$@"; fi; }
setting() { python -c "from core import config; c = config.load_settings(); v = c
for k in '$1'.split('.'): v = v[k]
print(v)"; }
VENV=$(setting cells.venv)
CKPT=$(setting cells.checkpoint)
REPO_ID=$(setting cells.weights_repo); REV=$(setting cells.weights_revision); SHA=$(setting cells.weights_sha256)
HN_REPO=$(setting cells.hovernet_repo); HN_REPO=${HN_REPO/#\~/$HOME}; HN_CKPT=$(setting cells.hovernet_checkpoint)
run mkdir -p "$(dirname "$VENV")" "$(dirname "$CKPT")"
[[ -x "$VENV/bin/python" ]] || run python -m venv --system-site-packages "$VENV"
run "$VENV/bin/python" -m pip install --upgrade pip
if [[ -z "${DRY_RUN:-}" ]]; then
"$VENV/bin/python" - "$VENV/constraints.txt" <<'PY'
import re, sys
from importlib.metadata import distributions
try:
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name as cn
except ImportError:  # pip vendors packaging
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.utils import canonicalize_name as cn
always = re.compile(r"^(torch|torchvision|torchaudio|triton|numpy|nvidia-.+)$")
if_in_range = {"timm": ">=1.0.3,<1.0.28", "transformers": ">=4.51.1,<5.14.1", "huggingface-hub": ">=0.33.3,<1.24.0"}
pins = {}
for d in distributions():
    n = cn(d.metadata["Name"] or "")
    if always.match(n) or (n in if_in_range and d.version in SpecifierSet(if_in_range[n])):
        pins[n] = f"{n}=={d.version}"
open(sys.argv[1], "w").write("\n".join(sorted(pins.values())) + "\n")
print("constraints:", *sorted(pins.values()), sep="\n  ")
PY
fi
run env PIP_CONSTRAINT="$VENV/constraints.txt" "$VENV/bin/python" -m pip install "tiatoolbox==2.1.3" "docopt==0.6.2" termcolor
run "$VENV/bin/python" -m pip check || echo "pip check: review above (conflicts with module-only packages are expected)"
run "$VENV/bin/python" - <<'PY'
import numpy, torch, timm, transformers, skimage, zarr, tiatoolbox
import tiatoolbox.models  # pulls transformers.SamModel, openslide, zarr, dask
from tiatoolbox import rcParam
print({"torch": torch.__version__, "numpy": numpy.__version__, "timm": timm.__version__,
       "transformers": transformers.__version__, "tiatoolbox": tiatoolbox.__version__, "skimage": skimage.__version__,
       "zarr": zarr.__version__, "compile_default": rcParam["torch_compile_mode"]})
assert torch.__version__.startswith("2.8.0") and numpy.__version__ == "2.2.6", "module torch/numpy were replaced!"
PY
if [[ -f "$CKPT" ]] && echo "$SHA  $CKPT" | sha256sum -c - >/dev/null 2>&1; then
    echo "weights present (sha256 ok): $CKPT"
else
    run "$VENV/bin/python" - "$REPO_ID" "$REV" "$CKPT" "$SHA" <<'PY'
import hashlib, os, shutil, sys
from huggingface_hub import hf_hub_download
repo, rev, dest, sha = sys.argv[1:]
p = hf_hub_download(repo, os.path.basename(dest), revision=rev, local_dir=os.path.dirname(dest))
if os.path.abspath(p) != os.path.abspath(dest):
    shutil.copyfile(p, dest)
h = hashlib.sha256(open(dest, "rb").read()).hexdigest()
assert h == sha, f"sha256 mismatch {h}"
print("weights OK", dest)
PY
fi
# Offline readiness: construct the engine on CPU with the local weights and HF offline (no inference, seconds).
run env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VENV/bin/python" - "$CKPT" <<'PY'
import sys
from tiatoolbox import rcParam
rcParam["torch_compile_mode"] = "disable"
from tiatoolbox.models import MultiTaskSegmentor
MultiTaskSegmentor(model="hovernet_fast-pannuke", weights=sys.argv[1], batch_size=2, num_workers=0, device="cpu",
                   verbose=False)
print("offline engine construction OK")
PY
if [[ -n "${WITH_FALLBACK:-}" ]]; then
    [[ -d "$HN_REPO" ]] || run git clone https://github.com/vqdang/hover_net.git "$HN_REPO"
    run git -C "$HN_REPO" checkout 67e2ce5e3f1a64a2ece77ad1c24233653a9e0901
    run sed -i '/^from imgaug import imgaug as ia$/d' "$HN_REPO/run_utils/utils.py"
    run sed -i 's|torch.load(self.method\["model_path"\])\["desc"\]|torch.load(self.method["model_path"], map_location="cpu", weights_only=False)["desc"]|' "$HN_REPO/infer/base.py"
    if grep -q imgaug "$HN_REPO/run_utils/utils.py"; then echo "PATCH FAILED: imgaug still imported"; exit 1; fi
    run env PIP_CONSTRAINT="$VENV/constraints.txt" "$VENV/bin/python" -m pip install gdown
    [[ -f "$HN_CKPT" ]] || run "$VENV/bin/gdown" "https://drive.google.com/uc?id=1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR" -O "$HN_CKPT" \
        || echo "gdown failed: download the PanNuke checkpoint on a laptop and copy it to $HN_CKPT"
fi
echo "cells venv ready: $VENV · weights: $CKPT"
