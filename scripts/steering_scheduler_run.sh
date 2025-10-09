#!/usr/bin/env bash
# Kick off persona steering-vector jobs via the new chatspace scheduler.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Defaults mirror the earlier train_all_steering sweep.
TRAITS_FILE="${TRAITS_FILE:-/workspace/persona_traits_over_100k.txt}"
ROLES_FILE="${ROLES_FILE:-/workspace/persona_roles_over_100k.txt}"
RUN_ROOT="${RUN_ROOT:-/workspace/steering_runs_scheduler}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TARGET_LAYER="${TARGET_LAYER:-31}"

INCLUDE_TRAITS=1
INCLUDE_ROLES=0
SKIP_EXISTING=0
EXTRA_ARGS=()
EXCLUDE_GPUS=()

usage() {
  cat <<'EOF'
Usage: steering_scheduler_run.sh [options] [-- extra-job-args]

Options:
  --traits-file PATH     Override trait dataset list (default: $TRAITS_FILE)
  --roles-file PATH      Override role dataset list (default: $ROLES_FILE)
  --include-roles        Include roles in addition to traits
  --traits-only          Disable roles (default behaviour)
  --run-root PATH        Output root for scheduler attempts
  --model NAME           Base model to fine-tune (default: Qwen/Qwen3-32B)
  --target-layer INDEX   Residual layer to hook steering vector (default: 31)
  --skip-existing        Map to scheduler --skip-if-committed so finished runs are skipped
  --no-traits            Disable trait datasets entirely
  --reuse-base-model     Force reuse within job (enabled by default)
  --no-reuse-base-model  Disable reuse and reload per dataset
  --exclude-gpu INDEX    Exclude a GPU index from CUDA_VISIBLE_DEVICES (can repeat)
  --avoid-gpu0           Convenience flag equal to --exclude-gpu 0
  --datasets NAME...     Explicit dataset names (can be repeated)
  --help                 Show this message

Any arguments after "--" are forwarded to `chatspace steering-train`.
EOF
}

DATASETS=()
REUSE_BASE_MODEL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traits-file)
      TRAITS_FILE="$2"; shift 2 ;;
    --traits-file=*)
      TRAITS_FILE="${1#*=}"; shift ;;
    --roles-file)
      ROLES_FILE="$2"; shift 2 ;;
    --roles-file=*)
      ROLES_FILE="${1#*=}"; shift ;;
    --include-roles)
      INCLUDE_ROLES=1; shift ;;
    --traits-only|--no-roles)
      INCLUDE_ROLES=0; shift ;;
    --no-traits)
      INCLUDE_TRAITS=0; shift ;;
    --run-root)
      RUN_ROOT="$2"; shift 2 ;;
    --run-root=*)
      RUN_ROOT="${1#*=}"; shift ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --model=*)
      MODEL="${1#*=}"; shift ;;
    --target-layer)
      TARGET_LAYER="$2"; shift 2 ;;
    --target-layer=*)
      TARGET_LAYER="${1#*=}"; shift ;;
    --skip-existing)
      SKIP_EXISTING=1; shift ;;
    --reuse-base-model)
      REUSE_BASE_MODEL=1; shift ;;
    --no-reuse-base-model)
      REUSE_BASE_MODEL=0; shift ;;
    --exclude-gpu)
      EXCLUDE_GPUS+=("$2"); shift 2 ;;
    --exclude-gpu=*)
      EXCLUDE_GPUS+=("${1#*=}"); shift ;;
    --avoid-gpu0)
      EXCLUDE_GPUS+=("0"); shift ;;
    --datasets)
      shift
      while [[ $# -gt 0 ]] && [[ $1 != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    *)
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

# Populate dataset list from files only when explicit datasets were not provided.
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  if [[ $INCLUDE_TRAITS -eq 1 ]]; then
    if [[ -f "$TRAITS_FILE" ]]; then
      while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        DATASETS+=("$line")
      done <"$TRAITS_FILE"
    else
      echo "warning: traits file not found: $TRAITS_FILE" >&2
    fi
  fi
  if [[ $INCLUDE_ROLES -eq 1 ]]; then
    if [[ -f "$ROLES_FILE" ]]; then
      while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        DATASETS+=("$line")
      done <"$ROLES_FILE"
    else
      echo "warning: roles file not found: $ROLES_FILE" >&2
    fi
  fi
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "error: no datasets resolved; provide --datasets or a traits/roles file" >&2
  exit 1
fi

if [[ ${#EXCLUDE_GPUS[@]} -gt 0 ]]; then
  EXCLUDE_STR=$(IFS=,; echo "${EXCLUDE_GPUS[*]}")
  if ! CUDA_MASK=$(EXCLUDE_GPUS="$EXCLUDE_STR" python - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

exclude_raw = os.environ.get("EXCLUDE_GPUS", "")
excludes = {token.strip() for token in exclude_raw.split(",") if token.strip()}

def parse_visible(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]

visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible_env:
    base = parse_visible(visible_env)
else:
    count = torch.cuda.device_count()
    if count == 0:
        print("No GPUs detected to exclude from", file=sys.stderr)
        sys.exit(1)
    base = [str(i) for i in range(count)]

available = [gpu for gpu in base if gpu not in excludes]
if not available:
    print("No GPUs left after applying exclusions", file=sys.stderr)
    sys.exit(1)

print(",".join(available))
PY
  ); then
    echo "error: unable to compute CUDA_VISIBLE_DEVICES mask" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$CUDA_MASK"
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
fi

echo "Planning ${#DATASETS[@]} dataset(s) via chatspace steering-train scheduler" >&2

CMD=(
  "uv" "run" "chatspace" "steering-train" "--"
  "--run-root" "$RUN_ROOT"
  "--model" "$MODEL"
  "--target-layer" "$TARGET_LAYER"
  "--attempt" "1"
)

if [[ $REUSE_BASE_MODEL -eq 1 ]]; then
  CMD+=("--reuse-base-model")
fi

if [[ $SKIP_EXISTING -eq 1 ]]; then
  CMD+=("--skip-if-committed")
fi

CMD+=("--datasets")
for dataset in "${DATASETS[@]}"; do
  CMD+=("$dataset")
done

CMD+=("${EXTRA_ARGS[@]}")

echo "Executing: ${CMD[*]}" >&2
exec "${CMD[@]}"
