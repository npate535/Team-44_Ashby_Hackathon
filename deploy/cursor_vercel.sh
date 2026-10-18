#!/bin/sh
set -eu
runtime=/tmp/sepia-cursor-runtime
if [ ! -x "$runtime/cursor-agent" ]; then
    mkdir -p "$runtime"
    tar -xzf "$(dirname "$0")/cursor-runtime.tar.gz" -C "$runtime"
fi
mkdir -p /tmp/sepia-home
# Cursor writes its configuration/cache only into this temporary home.
exec env HOME=/tmp/sepia-home "$runtime/cursor-agent" "$@"
