#!/bin/sh
set -eu
# Keep Cursor's JavaScript runtime and native modules, excluding duplicate
# standalone executables. No credentials are part of this build.
mkdir -p cursor-runtime
curl -fsSL https://downloads.cursor.com/lab/2026.10.16-e3cb207/linux/x64/agent-cli-package.tar.gz \
  | tar --strip-components=1 -xzf - -C cursor-runtime \
      --exclude='*cursor-agent-sea' --exclude='*cursor-agent-worker-sea' --exclude='*crepectl'
chmod +x cursor-runtime/cursor-agent cursor-runtime/node
tar -czf cursor-runtime.tar.gz -C cursor-runtime .
rm -rf cursor-runtime
