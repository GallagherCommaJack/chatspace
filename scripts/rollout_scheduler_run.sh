#!/usr/bin/env bash
# Distribute behavior rollout generation across GPUs using simple-gpu-scheduler.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

TRAITS_FILE="${TRAITS_FILE:-/workspace/persona_traits_over_100k.txt}"
ROLES_FILE="${ROLES_FILE:-/workspace/persona_roles_over_100k.txt}"
RUN_ROOT="${RUN_ROOT:-/workspace/steering_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/steering_rollouts}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TARGET_LAYER="${TARGET_LAYER:-22}"
TRAIT_PREFIX="${TRAIT_PREFIX:-qwen-3-32b__trait__}"
ROLE_PREFIX="${ROLE_PREFIX:-qwen-3-32b__role__}"
NUM_GPUS_REQUEST="${NUM_GPUS:-}"

INCLUDE_TRAITS=1
INCLUDE_ROLES=0
SKIP_EXISTING=0
EXTRA_ARGS=()
EXCLUDE_GPUS=()
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: rollout_scheduler_run.sh [options] [-- extra-rollout-args]

Options:
  --traits-file PATH     Traits dataset list (default: $TRAITS_FILE)
  --roles-file PATH      Roles dataset list (default: $ROLES_FILE)
  --trait-prefix PREFIX  Prefix for trait datasets (default: qwen-3-32b__trait__)
  --role-prefix  PREFIX  Prefix for role datasets (default: qwen-3-32b__role__)
  --include-roles        Include role datasets in addition to traits
  --traits-only          Use trait datasets only (default behaviour)
  --no-traits            Disable trait datasets entirely
  --run-root PATH        Steering vector root (default: $RUN_ROOT)
  --output-root PATH     Rollout output root (default: $OUTPUT_ROOT)
  --model NAME           Base model to load (default: Qwen/Qwen3-32B)
  --target-layer INDEX   Layer index for steering (default: 22)
  --skip-existing        Skip datasets with existing rollouts.jsonl
  --dry-run              Print planned commands without launching workers
  --exclude-gpu INDEX    Exclude a GPU index (repeatable)
  --avoid-gpu0           Shortcut for --exclude-gpu 0
  --num-gpus COUNT       Limit the number of GPU workers (auto-detect otherwise)
  --datasets NAME...     Explicit dataset names (repeatable, overrides trait/role files)
  --help                 Show this message

Arguments after "--" are forwarded to generate_behavior_rollouts.py.
EOF
}

DATASETS=()

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
    --trait-prefix)
      TRAIT_PREFIX="$2"; shift 2 ;;
    --trait-prefix=*)
      TRAIT_PREFIX="${1#*=}"; shift ;;
    --role-prefix)
      ROLE_PREFIX="$2"; shift 2 ;;
    --role-prefix=*)
      ROLE_PREFIX="${1#*=}"; shift ;;
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
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"; shift ;;
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
    --dry-run)
      DRY_RUN=1; shift ;;
    --exclude-gpu)
      EXCLUDE_GPUS+=("$2"); shift 2 ;;
    --exclude-gpu=*)
      EXCLUDE_GPUS+=("${1#*=}"); shift ;;
    --avoid-gpu0)
      EXCLUDE_GPUS+=("0"); shift ;;
    --num-gpus)
      NUM_GPUS_REQUEST="$2"; shift 2 ;;
    --num-gpus=*)
      NUM_GPUS_REQUEST="${1#*=}"; shift ;;
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

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  if [[ $INCLUDE_TRAITS -eq 1 ]]; then
    if [[ -f "$TRAITS_FILE" ]]; then
      while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        DATASETS+=("${TRAIT_PREFIX}${line}")
      done <"$TRAITS_FILE"
    else
      echo "warning: traits file not found: $TRAITS_FILE" >&2
    fi
  fi
  if [[ $INCLUDE_ROLES -eq 1 ]]; then
    if [[ -f "$ROLES_FILE" ]]; then
      while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        DATASETS+=("${ROLE_PREFIX}${line}")
      done <"$ROLES_FILE"
    else
      echo "warning: roles file not found: $ROLES_FILE" >&2
    fi
  fi
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "error: no datasets resolved; provide --datasets or trait/role files" >&2
  exit 1
fi

if [[ -n "$NUM_GPUS_REQUEST" ]]; then
  if ! [[ "$NUM_GPUS_REQUEST" =~ ^[0-9]+$ ]]; then
    echo "error: --num-gpus expects a positive integer" >&2
    exit 1
  fi
  if (( NUM_GPUS_REQUEST < 1 )); then
    echo "error: --num-gpus must be >= 1" >&2
    exit 1
  fi
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "error: python interpreter not found" >&2
  exit 1
fi

mapfile -t DETECTED_GPUS < <("$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA is not available on this host", file=sys.stderr)
    sys.exit(1)

count = torch.cuda.device_count()
if count == 0:
    print("Torch reports zero CUDA devices", file=sys.stderr)
    sys.exit(1)

for idx in range(count):
    print(idx)
PY
) || {
  echo "error: unable to enumerate GPUs via torch" >&2
  exit 1
}

AVAILABLE_GPUS=()
for gpu in "${DETECTED_GPUS[@]}"; do
  skip=0
  for excluded in "${EXCLUDE_GPUS[@]}"; do
    if [[ "$gpu" == "$excluded" ]]; then
      skip=1
      break
    fi
  done
  if [[ $skip -eq 0 ]]; then
    AVAILABLE_GPUS+=("$gpu")
  fi
done

if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
  echo "error: no GPUs available after applying exclusions" >&2
  exit 1
fi

if [[ -n "$NUM_GPUS_REQUEST" ]]; then
  if (( NUM_GPUS_REQUEST > ${#AVAILABLE_GPUS[@]} )); then
    echo "warning: requested $NUM_GPUS_REQUEST GPU(s) but only ${#AVAILABLE_GPUS[@]} available" >&2
  fi
  if (( ${#AVAILABLE_GPUS[@]} > NUM_GPUS_REQUEST )); then
    AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:0:NUM_GPUS_REQUEST}")
  fi
fi

WORKER_COUNT=${#AVAILABLE_GPUS[@]}
if (( WORKER_COUNT == 0 )); then
  echo "error: worker count resolved to zero" >&2
  exit 1
fi

echo "Planning ${#DATASETS[@]} dataset(s) across ${WORKER_COUNT} worker(s) on GPU(s): ${AVAILABLE_GPUS[*]}" >&2

declare -a COMMAND_LINES=()
TOTAL_ASSIGNED=0

join_command() {
  local result=""
  for arg in "$@"; do
    if [[ -z "$result" ]]; then
      result="$(printf '%q' "$arg")"
    else
      result+=" $(printf '%q' "$arg")"
    fi
  done
  printf '%s' "$result"
}

for (( worker_idx=0; worker_idx<WORKER_COUNT; worker_idx++ )); do
  worker_datasets=()
  for (( idx=0; idx<${#DATASETS[@]}; idx++ )); do
    if (( idx % WORKER_COUNT == worker_idx )); then
      worker_datasets+=("${DATASETS[idx]}")
    fi
  done

  if [[ ${#worker_datasets[@]} -eq 0 ]]; then
    continue
  fi

  TOTAL_ASSIGNED=$((TOTAL_ASSIGNED + ${#worker_datasets[@]}))
  echo "  worker ${worker_idx} (${AVAILABLE_GPUS[worker_idx]}): ${#worker_datasets[@]} dataset(s)" >&2

  CMD=(
    "uv" "run" "python" "scripts/generate_behavior_rollouts.py"
    "--run-root" "$RUN_ROOT"
    "--output-root" "$OUTPUT_ROOT"
    "--model" "$MODEL"
    "--target-layer" "$TARGET_LAYER"
    "--steering-no-system"
  )

  if [[ $SKIP_EXISTING -eq 1 ]]; then
    CMD+=("--skip-existing")
  fi

  CMD+=(
    "--dataset-stride" "$WORKER_COUNT"
    "--dataset-offset" "$worker_idx"
    "--datasets"
  )

  CMD+=("${worker_datasets[@]}")
  CMD+=("${EXTRA_ARGS[@]}")

  COMMAND_LINES+=("$(join_command "${CMD[@]}")")
done

if (( TOTAL_ASSIGNED == 0 )); then
  echo "warning: no datasets assigned after applying stride/offset logic; exiting." >&2
  exit 0
fi

if (( TOTAL_ASSIGNED != ${#DATASETS[@]} )); then
  echo "error: dataset assignment mismatch (${TOTAL_ASSIGNED}/${#DATASETS[@]}). Check stride logic." >&2
  exit 1
fi

if (( DRY_RUN == 1 )); then
  echo "Dry-run: printing planned commands only." >&2
  {
    for line in "${COMMAND_LINES[@]}"; do
      printf '%s\n' "$line"
    done
  } | sed 's/^/  /'
  exit 0
fi

echo "Dispatching ${TOTAL_ASSIGNED}/${#DATASETS[@]} dataset(s) via simple-gpu-scheduler" >&2

{
  for line in "${COMMAND_LINES[@]}"; do
    printf '%s\n' "$line"
  done
} | uv run python -m simple_gpu_scheduler.command_line --gpus "${AVAILABLE_GPUS[@]}"
