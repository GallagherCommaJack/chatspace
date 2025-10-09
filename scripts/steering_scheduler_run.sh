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
TRAIT_PREFIX="${TRAIT_PREFIX:-qwen-3-32b__trait__}"
ROLE_PREFIX="${ROLE_PREFIX:-qwen-3-32b__role__}"
NUM_GPUS_REQUEST="${NUM_GPUS:-}"
ATTEMPT="${ATTEMPT:-1}"

INCLUDE_TRAITS=1
INCLUDE_ROLES=0
SKIP_EXISTING=0
EXTRA_ARGS=()
EXCLUDE_GPUS=()
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: steering_scheduler_run.sh [options] [-- extra-job-args]

Options:
  --traits-file PATH     Override trait dataset list (default: $TRAITS_FILE)
  --roles-file PATH      Override role dataset list (default: $ROLES_FILE)
  --trait-prefix PREFIX  Prefix for trait datasets (default: qwen-3-32b__trait__)
  --role-prefix PREFIX   Prefix for role datasets (default: qwen-3-32b__role__)
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
  --num-gpus COUNT       Limit the number of GPUs/workers launched (default: detect all available)
  --dry-run              Print worker commands instead of launching
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

# Populate dataset list from files only when explicit datasets were not provided.
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
  echo "error: no datasets resolved; provide --datasets or a traits/roles file" >&2
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
  echo "error: python interpreter not found (expected .venv or system python)" >&2
  exit 1
fi

mapfile -t DETECTED_GPUS < <("$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover
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
    echo "warning: requested $NUM_GPUS_REQUEST GPU(s) but only ${#AVAILABLE_GPUS[@]} available after exclusions" >&2
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
    "uv" "run" "chatspace" "steering-train" "--"
    "--run-root" "$RUN_ROOT"
    "--model" "$MODEL"
    "--target-layer" "$TARGET_LAYER"
    "--attempt" "$ATTEMPT"
  )

  if [[ $REUSE_BASE_MODEL -eq 1 ]]; then
    CMD+=("--reuse-base-model")
  fi

  if [[ $SKIP_EXISTING -eq 1 ]]; then
    CMD+=("--skip-if-committed")
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    CMD+=("--dry-run")
  fi

  CMD+=("--dataset-stride" "$WORKER_COUNT" "--dataset-offset" "$worker_idx")
  CMD+=("--datasets")
  CMD+=("${DATASETS[@]}")
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
  echo "Dry-run mode: printing worker commands and exiting." >&2
  for line in "${COMMAND_LINES[@]}"; do
    printf '%s\n' "$line"
  done
  exit 0
fi

echo "Dispatching ${TOTAL_ASSIGNED}/${#DATASETS[@]} dataset(s) via simple-gpu-scheduler" >&2

{
  for line in "${COMMAND_LINES[@]}"; do
    printf '%s\n' "$line"
  done
} | uv run python -m simple_gpu_scheduler.command_line --gpus "${AVAILABLE_GPUS[@]}"
