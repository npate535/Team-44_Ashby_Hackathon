#!/bin/bash
#SBATCH --job-name=pq-prep-a
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail
# prep_a (Agent 1, BUILD_PLAN Task 2 / §G): official BACH (download only if not already unpacked), patient
# workbooks, 00_inventory (gate), 01_build_cases --limit 5 smoke, then the full 01_build_cases.
# CPU only. Paths, URLs and md5s come from config/settings.yaml. No secrets are read or written.

cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PATHQUEUE_SITE=delta
echo "== prep_a on $(hostname) at $(date -Is); code $(pwd); cpus ${SLURM_CPUS_PER_TASK:-?}"

setting() { python -c "from core import config; c = config.load_settings(); v = c
for k in '$1'.split('.'): v = v[k]
print(v)"; }
BACH=$(setting paths.delta.bach_official)
DATA=$(setting paths.delta.data_root)
ZIP=$(setting dataset.official.filename)
URL=$(setting dataset.official.api_url)
MD5=$(setting dataset.official.md5)
SIZE=$(setting dataset.official.bytes)
PHOTOS="$BACH/$(setting dataset.official.photos_subdir)"
MICRO_FILE=$(setting dataset.patients.micro_file); MICRO_URL=$(setting dataset.patients.micro_url)
MICRO_MD5=$(setting dataset.patients.micro_md5)
WSI_FILE=$(setting dataset.patients.wsi_file); WSI_URL=$(setting dataset.patients.wsi_url)
WSI_MD5=$(setting dataset.patients.wsi_md5)
mkdir -p "$BACH" "$DATA/logs"

# ---- 1. official BACH: skip when the 400 photos are already unpacked (download job 22956284) -------------------
n_photos() { find "$PHOTOS" -name '*.tif' 2>/dev/null | wc -l; }
if [[ "$(n_photos)" -eq 400 ]]; then
    echo "BACH already unpacked: $(n_photos) photos under $PHOTOS (download skipped)"
else
    echo "BACH photos found: $(n_photos); downloading/unpacking $ZIP"
    cd "$BACH"
    if ! curl -sSf --max-time 20 -o /dev/null https://zenodo.org/api/records/3632035; then
        echo "NO_INTERNET_ON_COMPUTE_NODE"; exit 3
    fi
    # Same logic as pipeline/jobs/download_bach.sbatch: 8 resumable byte ranges in parallel, joined, md5-verified.
    PARTS=8
    CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
    if [[ ! -f "$ZIP" ]] || ! echo "$MD5  $ZIP" | md5sum -c - >/dev/null 2>&1; then
        rm -f "$ZIP"
        for i in $(seq 0 $((PARTS - 1))); do
            start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 )); (( end >= SIZE )) && end=$(( SIZE - 1 ))
            want=$(( end - start + 1 ))
            (
                have=$(stat -c %s "part.$i" 2>/dev/null || echo 0)
                while (( have < want )); do
                    curl -sSfL --retry 5 -r "$(( start + have ))-$end" "$URL" >> "part.$i" || sleep 5
                    have=$(stat -c %s "part.$i")
                done
            ) &
        done
        wait
        cat $(for i in $(seq 0 $((PARTS - 1))); do echo "part.$i"; done) > "$ZIP"
        rm -f part.*
    fi
    echo "$MD5  $ZIP" | md5sum -c -
    unzip -q -n "$ZIP" -d "$BACH"
    cd "${PQ_CODE_DIR:-$HOME/pathqueue}"
    echo "photos after unzip: $(n_photos)"
fi

# ---- 2. patient workbooks (not in the zip) --------------------------------------------------------------------
if [[ -f "$BACH/$MICRO_FILE" ]] && echo "$MICRO_MD5  $BACH/$MICRO_FILE" | md5sum -c - >/dev/null 2>&1; then
    echo "micro patient workbook present (md5 ok)"
else
    curl -sSfL --retry 5 --max-time 120 -o "$BACH/.$MICRO_FILE.part" "$MICRO_URL"
    if echo "$MICRO_MD5  $BACH/.$MICRO_FILE.part" | md5sum -c - >/dev/null 2>&1; then
        mv "$BACH/.$MICRO_FILE.part" "$BACH/$MICRO_FILE"; echo "micro patient workbook fetched (md5 ok)"
    else
        mv "$BACH/.$MICRO_FILE.part" "$BACH/$MICRO_FILE.bad"
        echo "ERROR: micro patient workbook md5 mismatch (kept as $MICRO_FILE.bad); 01 needs it for the leakage check"
        exit 4
    fi
fi
if [[ -f "$BACH/$WSI_FILE" ]] && echo "$WSI_MD5  $BACH/$WSI_FILE" | md5sum -c - >/dev/null 2>&1; then
    echo "WSI patient workbook present (md5 ok)"
else
    # Dropbox serves the folder as a zip; extract the workbook member. Only prep_b needs it, so failure warns.
    if curl -sSfL --retry 5 --max-time 120 -o "$BACH/.wsi_patients.zip" "$WSI_URL" \
        && python - "$BACH/.wsi_patients.zip" "$WSI_FILE" "$BACH/$WSI_FILE" <<'PY'
import sys, zipfile, pathlib
src, want, dest = sys.argv[1:]
with zipfile.ZipFile(src) as z:
    names = [n for n in z.namelist() if pathlib.PurePosixPath(n).name == want]
    if not names:
        sys.exit(f"{want} not in {src}: {z.namelist()[:10]}")
    pathlib.Path(dest).write_bytes(z.read(names[0]))
PY
    then
        echo "$WSI_MD5  $BACH/$WSI_FILE" | md5sum -c - || echo "WARNING: WSI patient workbook md5 differs from settings"
    else
        echo "WARNING: could not fetch the WSI patient workbook (prep_b needs it for WSI folds)"
    fi
    rm -f "$BACH/.wsi_patients.zip"
fi

# ---- 3. inventory (exit 1 = gate failed) -> 01 smoke -> 01 full -------------------------------------------------
echo "== 00_inventory"
python -m pipeline.00_inventory
echo "== 01_build_cases --limit 5 (smoke)"
SMOKE=$(mktemp -d "$DATA/.smoke01.XXXXXX")
python -m pipeline.01_build_cases --limit 5 --data-root "$SMOKE"
ls -R "$SMOKE" | head -20
rm -rf "$SMOKE"
echo "== 01_build_cases (full)"
python -m pipeline.01_build_cases
echo "== outputs"
du -sh "$DATA/tiles" "$DATA/display" 2>/dev/null || true
wc -l "$DATA/cases.csv" "$DATA/tiles.csv" "$DATA/embed_manifest.csv"
echo "== prep_a done at $(date -Is)"
