#!/bin/bash
#SBATCH --job-name=pq-download-bach
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bfqi/teams/team44/pathqueue/logs/%x-%j.out
# Download, verify and unpack the official BACH release (Zenodo record 3632035, CC BY-NC-ND 4.0)
# into the team directory. CPU only; organizers ask for downloads in CPU jobs, not on login nodes.
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
set -euo pipefail

DEST=/work/nvme/bfqi/teams/team44/bach
URL=https://zenodo.org/api/records/3632035/files/ICIAR2018_BACH_Challenge.zip/content
MD5=8ae1801334aa943c44627c1eef3631b2   # from the Zenodo API, 10419148291 bytes

cd "$DEST"
if ! curl -sSf --max-time 20 -o /dev/null https://zenodo.org/api/records/3632035; then
    echo "NO_INTERNET_ON_COMPUTE_NODE"
    exit 3
fi
# Zenodo serves ~3 MB/s per connection, so fetch 8 byte ranges in parallel (each resumable), then join.
SIZE=10419148291
PARTS=8
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
if [[ ! -f ICIAR2018_BACH_Challenge.zip ]] || ! echo "$MD5  ICIAR2018_BACH_Challenge.zip" | md5sum -c - >/dev/null 2>&1; then
    rm -f ICIAR2018_BACH_Challenge.zip
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
    cat $(for i in $(seq 0 $((PARTS - 1))); do echo "part.$i"; done) > ICIAR2018_BACH_Challenge.zip
    rm -f part.*
fi
echo "$MD5  ICIAR2018_BACH_Challenge.zip" | md5sum -c -
unzip -q -n ICIAR2018_BACH_Challenge.zip -d "$DEST"
find "$DEST" -maxdepth 3 -type d | head -30
echo "photo tifs: $(find "$DEST" -path '*Photos*' -name '*.tif' | wc -l)"
echo "svs: $(find "$DEST" -name '*.svs' | wc -l)  xml: $(find "$DEST" -name '*.xml' | wc -l)"
