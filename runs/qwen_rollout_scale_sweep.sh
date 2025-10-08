#!/usr/bin/env bash
# Generate persona rollouts for all Qwen-3-32B traits/roles with steering scale sweep.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set; judge evaluation requires API access." >&2
  exit 1
fi

python "${ROOT_DIR}/scripts/generate_behavior_rollouts.py" \
  --log /workspace/steering_runs/steering_sweep.log \
  --include-prefix qwen-3-32b__trait__ qwen-3-32b__role__ \
  --rollouts 1 \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --steering-no-system \
  --normalize-steering \
  --trained-scales -100 -200 -300 -400 -500 -600 -700 -800 -900 -1000 \
                   100 200 300 400 500 600 700 800 900 1000 \
  --activation-scales -100 -200 -300 -400 -500 -600 -700 -800 -900 -1000 \
                      100 200 300 400 500 600 700 800 900 1000 \
  --minilm-eval \
  --judge-eval
