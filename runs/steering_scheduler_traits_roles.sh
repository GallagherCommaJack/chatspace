#!/usr/bin/env bash
# Recorded invocation for the steering scheduler sweep over traits and roles.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

"${ROOT_DIR}/scripts/steering_scheduler_run.sh" \
  --include-roles \
  --skip-existing \
  --reuse-base-model \
  --run-root /workspace/steering_runs_scheduler \
  --model Qwen/Qwen3-32B \
  --target-layer 31 \
  --datasets qwen-3-32b__role__leviathan \
  -- \
  --learning-rate 0.5 \
  --num-epochs 1 \
  "$@" \
  # avoid gpu0 to allow concurrent interactive exploration
  # --avoid-gpu0 

