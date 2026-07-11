#!/bin/sh
set -e

if [ -n "$CLAUDE_CREDENTIALS_B64" ]; then
    mkdir -p /root/.claude
    echo "$CLAUDE_CREDENTIALS_B64" | base64 -d > /root/.claude/.credentials.json
    chmod 600 /root/.claude/.credentials.json
    echo "[entrypoint] wrote /root/.claude/.credentials.json from CLAUDE_CREDENTIALS_B64 ($(wc -c < /root/.claude/.credentials.json) bytes)" >&2
fi

exec "$@"
