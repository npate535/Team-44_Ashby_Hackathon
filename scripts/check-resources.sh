#!/bin/bash
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
printf 'LOCAL FILES\n'
for p in CLAUDE.md CLAUDE.local.md resources/README.md resources/official/HACKATHON_COMPUTE.md resources/official/STUDENT_GUIDE.md resources/text/2026_Ashby_Hackathon_Opening_Session.md; do if [[ -r "$p" && -s "$p" ]]; then printf 'OK %s\n' "$p"; else printf 'MISSING %s\n' "$p"; fi; done
printf '\nPRIVATE CREDENTIAL PRESENCE (NO VALUES)\n'
for p in "$HOME/.config/ashby-2026/model-servers.env" "$HOME/.config/ashby-2026/lumen.env"; do if [[ -s "$p" ]]; then printf 'PRESENT %s\n' "$p"; else printf 'NOT CONFIGURED %s\n' "$p"; fi; done
printf '\nDELTA READ-ONLY ACCESS CHECK\n'
if ./scripts/delta.sh 'hostname; whoami; accounts; for p in /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples /projects/bfqi/shared/datasets/pathgen_dataset /projects/bfqi/shared/datasets/pathmmu /work/nvme/bfqi/shared/datasets/quilt; do if test -r "$p"; then printf "READABLE %s\n" "$p"; else printf "UNAVAILABLE %s\n" "$p"; fi; done; for port in 8000 8001 8002 8003; do printf "HEALTH %s " "$port"; curl --silent --show-error --max-time 4 --output /dev/null --write-out "%{http_code}\n" http://gpub055.delta.ncsa.illinois.edu:$port/health || true; done'; then printf 'Delta session reachable.\n'; else printf 'Delta is not authenticated/reachable. Run ./scripts/delta.sh interactively (password + Duo).\n'; fi
printf '\nEXISTING DELTAAI MASTER (SEPARATE CLUSTER)\n'
ssh -O check deltaai 2>&1 || true
