#!/bin/sh
set -eu
mkdir -p /data
if [ ! -d /data/cases ]; then
    if [ ! -d /seed/cases ]; then
        echo 'Case artifacts are missing from the persistent /data volume.' >&2
        exit 1
    fi
    cp -a /seed/. /data/
fi
chown -R app:app /data
exec gosu app python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers
