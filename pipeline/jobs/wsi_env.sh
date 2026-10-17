#!/bin/bash
# Login-node recipe (the master runs it once; Agent 1 never runs it): the `wsi` venv for pipeline/wsi_extract.py.
#   bash pipeline/jobs/wsi_env.sh            # build/refresh the venv and check it
#   DRY_RUN=1 bash pipeline/jobs/wsi_env.sh  # print the commands only
#   CHECK_SLIDE=/work/nvme/bfqi/teams/team44/bach/ICIAR2018_BACH_Challenge/WSI/A03.svs bash pipeline/jobs/wsi_env.sh
#       additionally opens one slide and reads a 256-px region (JPEG 2000 decode check; light, login-node safe)
# --system-site-packages venv on NVMe with a constraints file generated from the module's own `pip freeze`, so
# torch/numpy/Pillow/pandas from pytorch-conda/2.8 can never be upgraded. Installs only openslide-python and the
# openslide-bin wheel (bundles libopenslide with OpenJPEG: Aperio JP2K SVS support). No secrets involved.
set -euo pipefail
VENV=${WSI_VENV:-/work/nvme/bfqi/rbansal3/venvs/wsi}
run() { echo "+ $*"; if [[ -z "${DRY_RUN:-}" ]]; then "$@"; fi; }

module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
run mkdir -p "$(dirname "$VENV")"
[[ -x "$VENV/bin/python" ]] || run python -m venv --system-site-packages "$VENV"
if [[ -z "${DRY_RUN:-}" ]]; then
    # constraints = exact versions of everything the module provides (generated, never hand-written)
    "$VENV/bin/python" -m pip freeze --all 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' \
        | grep -viE '^(openslide-python|openslide-bin)==' > "$VENV/constraints.txt"
    echo "constraints: $(wc -l < "$VENV/constraints.txt") pins (torch $(grep -i '^torch==' "$VENV/constraints.txt" || true))"
fi
run "$VENV/bin/python" -m pip install --upgrade-strategy only-if-needed -c "$VENV/constraints.txt" \
    "openslide-python>=1.3.1,<2" "openslide-bin>=4.0.0.2,<5"
run "$VENV/bin/python" - <<'PY'
import os, sys
import numpy, PIL, openslide
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| Pillow", PIL.__version__)
print("openslide-python", openslide.__version__, "| libopenslide", openslide.__library_version__)
path = os.environ.get("CHECK_SLIDE")
if path:
    s = openslide.OpenSlide(path)
    print("slide", path, "dims", s.dimensions, "levels", s.level_count, "downsamples",
          [round(d, 2) for d in s.level_downsamples], "mpp-x", s.properties.get("openslide.mpp-x"))
    w, h = s.dimensions
    region = s.read_region((w // 2, h // 2), 0, (256, 256)).convert("RGB")
    print("read_region ok:", region.size, "mean RGB", [round(float(v), 1) for v in numpy.asarray(region).mean((0, 1))])
PY
echo "wsi venv ready: $VENV (prep_b activates it for wsi_extract only)"
