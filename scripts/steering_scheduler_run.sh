#!/usr/bin/env bash
# Backward-compatible wrapper that forwards to the Python scheduler.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v uv >/dev/null 2>&1; then
  exec uv run python "${SCRIPT_DIR}/steering_scheduler_run.py" "$@"
else
  exec python "${SCRIPT_DIR}/steering_scheduler_run.py" "$@"
fi
