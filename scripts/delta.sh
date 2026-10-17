#!/bin/bash
set -euo pipefail
# Interactive without a command; command mode never hangs waiting for MFA.
opts=(-o "HostName=${ASHBY_DELTA_HOST:-dt-login02.delta.ncsa.illinois.edu}" -o "User=${ASHBY_NCSA_USER:-rbansal3}" -o ControlMaster=auto -o ControlPersist=3600 -o "ControlPath=$HOME/.ssh/ashby-delta-%h-%r-%p.sock" -o ServerAliveInterval=30 -o ConnectTimeout=10)
if (( $# )); then opts+=(-o BatchMode=yes); fi
exec ssh "${opts[@]}" ashby-delta "$@"
