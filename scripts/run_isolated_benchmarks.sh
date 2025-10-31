#!/bin/bash
# Run isolated capture benchmarks in sequence, each in fresh Python process

set -e

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%S")
RESULTS_DIR="/workspace/benchmarks/isolated_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Running isolated capture benchmarks"
echo "Results directory: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Configuration
MODEL="Qwen/Qwen2.5-3B"
BATCH=32
PREFILL=512
DECODE=128
ITERATIONS=3

# Run baseline
echo "=== Running baseline (no capture) ==="
uv run python scripts/isolated_capture_bench.py \
    --mode baseline \
    --model "$MODEL" \
    --batch "$BATCH" \
    --prefill "$PREFILL" \
    --decode "$DECODE" \
    --iterations "$ITERATIONS" \
    --output "$RESULTS_DIR/baseline.json"
echo ""

# Sleep to ensure clean GPU state
sleep 2

# Run zero-layer (metadata overhead only)
echo "=== Running zero-layer capture (metadata overhead) ==="
uv run python scripts/isolated_capture_bench.py \
    --mode zero-layer \
    --model "$MODEL" \
    --batch "$BATCH" \
    --prefill "$PREFILL" \
    --decode "$DECODE" \
    --iterations "$ITERATIONS" \
    --output "$RESULTS_DIR/zero-layer.json"
echo ""

sleep 2

# Run all-layers
echo "=== Running all-layers capture ==="
uv run python scripts/isolated_capture_bench.py \
    --mode all-layers \
    --model "$MODEL" \
    --batch "$BATCH" \
    --prefill "$PREFILL" \
    --decode "$DECODE" \
    --iterations "$ITERATIONS" \
    --output "$RESULTS_DIR/all-layers.json"
echo ""

sleep 2

# Run half-layers (middle 50%)
echo "=== Running half-layers capture ==="
# Qwen2.5-3B has 36 layers, so capture 9-26 (middle 18)
uv run python scripts/isolated_capture_bench.py \
    --mode custom \
    --layers "9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26" \
    --model "$MODEL" \
    --batch "$BATCH" \
    --prefill "$PREFILL" \
    --decode "$DECODE" \
    --iterations "$ITERATIONS" \
    --output "$RESULTS_DIR/half-layers.json"
echo ""

sleep 2

# Run single layer
echo "=== Running single layer capture ==="
uv run python scripts/isolated_capture_bench.py \
    --mode custom \
    --layers "18" \
    --model "$MODEL" \
    --batch "$BATCH" \
    --prefill "$PREFILL" \
    --decode "$DECODE" \
    --iterations "$ITERATIONS" \
    --output "$RESULTS_DIR/single-layer.json"
echo ""

# Analyze results
echo "==================================="
echo "Benchmark complete!"
echo "==================================="
echo ""
echo "Results summary:"
for file in "$RESULTS_DIR"/*.json; do
    mode=$(basename "$file" .json)
    gen_time=$(jq -r '.summary.mean_generation_time' "$file")
    fetch_time=$(jq -r '.summary.mean_fetch_time' "$file")
    total_time=$(jq -r '.summary.mean_total_time' "$file")
    num_layers=$(jq -r '.num_captured_layers' "$file")
    printf "%-15s: gen=%7.3fs  fetch=%7.3fs  total=%7.3fs  layers=%2d\n" \
        "$mode" "$gen_time" "$fetch_time" "$total_time" "$num_layers"
done
echo ""

# Calculate overhead relative to baseline
baseline_gen=$(jq -r '.summary.mean_generation_time' "$RESULTS_DIR/baseline.json")
echo "Overhead relative to baseline ($baseline_gen s):"
for file in "$RESULTS_DIR"/*.json; do
    mode=$(basename "$file" .json)
    if [ "$mode" = "baseline" ]; then
        continue
    fi
    gen_time=$(jq -r '.summary.mean_generation_time' "$file")
    overhead=$(echo "scale=2; ($gen_time / $baseline_gen - 1) * 100" | bc)
    num_layers=$(jq -r '.num_captured_layers' "$file")
    printf "%-15s: %+7.2f%%  (%d layers)\n" "$mode" "$overhead" "$num_layers"
done
echo ""

echo "Full results available in: $RESULTS_DIR"
