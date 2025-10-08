#!/usr/bin/env bash
# Generate persona rollouts for all Qwen-3-32B traits/roles with steering scale sweep.

set -euo pipefail

# if .env exists, export its variables
if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

uv run python "${ROOT_DIR}/scripts/generate_behavior_rollouts.py" \
  --log /workspace/steering_runs/steering_sweep.log \
  --include-prefix qwen-3-32b__trait__ qwen-3-32b__role__ \
  --rollouts 1 \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  --steering-no-system \
  --normalize-steering \
  --trained-scales 200 400 800 -200 -400 -800 \
  --activation-scales 200 400 800 -200 -400 -800 \
  --activation-match-learned \
  --minilm-eval \
  --judge-eval
