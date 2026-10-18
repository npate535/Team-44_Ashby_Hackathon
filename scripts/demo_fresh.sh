#!/bin/bash
# Restart the PathQueue demo on a fresh workspace: stop the server on $PORT (default 8000), archive (never delete)
# the workspace database under <artifacts>/archive/db-<UTC>/, start uvicorn again and print /api/health.
#   scripts/demo_fresh.sh            # or: make demo-fresh [PORT=8000] [ART=artifacts]
set -u
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"; ART="${ART:-artifacts}"; HOST="${HOST:-127.0.0.1}"
pkill -f "uvicorn backend.app:app --host $HOST --port $PORT" 2>/dev/null; sleep 1
TS=$(date -u +%Y%m%dT%H%M%SZ)
if ls "$ART"/pathqueue.sqlite* >/dev/null 2>&1; then
  mkdir -p "$ART/archive/db-$TS" && mv "$ART"/pathqueue.sqlite* "$ART/archive/db-$TS/"
  echo "archived the previous workspace database to $ART/archive/db-$TS/"
fi
mkdir -p outputs
PATHQUEUE_SITE=local PATHQUEUE_ARTIFACTS_ROOT="$ART" PYTHONPATH="$PWD" \
  nohup .venv/bin/python -m uvicorn backend.app:app --host "$HOST" --port "$PORT" > "outputs/demo-$PORT.log" 2>&1 &
for _ in $(seq 1 30); do curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/api/health" 2>/dev/null | grep -q 200 && break; sleep 1; done
curl -s "http://$HOST:$PORT/api/health" | head -c 240; echo
echo "open http://$HOST:$PORT  (log: outputs/demo-$PORT.log)"
