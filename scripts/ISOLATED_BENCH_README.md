# Isolated Capture Benchmarking

## Problem

Previous benchmarks ran multiple configurations in the same Python process, potentially causing cross-contamination. For example, we observed:
- Zero-layer capture: 169s → 86.2s after optimization (expected)
- All-layer capture: 175s → 125s after optimization (unexpected - why?)

The all-layer improvement is mysterious because the optimization only adds early-exit guards for when capture is NOT active.

## Solution

Run each configuration in a completely isolated Python process to eliminate cross-contamination.

## Usage

### Single Configuration

```bash
# Baseline (no capture machinery)
uv run python scripts/isolated_capture_bench.py --mode baseline --output /workspace/results/baseline.json

# Zero-layer (capture machinery active but no layers captured)
uv run python scripts/isolated_capture_bench.py --mode zero-layer --output /workspace/results/zero.json

# All layers
uv run python scripts/isolated_capture_bench.py --mode all-layers --output /workspace/results/all.json

# Custom layer selection
uv run python scripts/isolated_capture_bench.py --mode custom --layers 0,5,10,15,20 --output /workspace/results/custom.json
```

### Full Suite

```bash
# Run all configurations in sequence
bash scripts/run_isolated_benchmarks.sh

# Results will be in /workspace/benchmarks/isolated_TIMESTAMP/
```

## Output

Each run produces a JSON file with:
- Configuration details (model, batch size, tokens, layers)
- Per-iteration timings (generation, fetch, total)
- Summary statistics (mean, min, max)

Example:
```json
{
  "mode": "all-layers",
  "model": "Qwen/Qwen2.5-3B",
  "batch_size": 32,
  "prefill_tokens": 512,
  "decode_tokens": 128,
  "total_layers": 36,
  "capture_layers": [0, 1, 2, ..., 35],
  "num_captured_layers": 36,
  "iterations": [
    {
      "iteration": 1,
      "generation_time": 125.3,
      "fetch_time": 59.8,
      "total_time": 185.1,
      "throughput_tokens_per_sec": 163.4
    },
    ...
  ],
  "summary": {
    "mean_generation_time": 125.1,
    "mean_fetch_time": 60.2,
    "mean_total_time": 185.3
  }
}
```

## Modes Explained

### baseline
Pure baseline with NO capture machinery activated. This is the true baseline.

### zero-layer
Capture machinery is activated (model runner patched, buffers allocated) but zero layers are actually captured. Measures overhead of the metadata extraction and coordination without actual capture work.

### all-layers
Captures all model layers. Measures full capture overhead.

### custom
Captures specific layers via `--layers` flag. Useful for testing:
- Single layer: minimal capture cost
- Half layers: middle ground
- Specific patterns: e.g., every 4th layer

## Key Differences from Previous Benchmarks

1. **Process Isolation**: Each config runs in fresh Python process
2. **Clear Modes**: Explicit distinction between "no machinery" vs "machinery but no capture"
3. **Simple Workload**: Single batch size, prompt length, decode length
4. **Focused Measurement**: Just generation and fetch, no complex scenarios
5. **Clean Timing**: CUDA synchronization around each phase

## Interpreting Results

### Zero-Layer Overhead
```
overhead = (zero_layer_gen_time - baseline_gen_time) / baseline_gen_time
```
This tells us the cost of metadata extraction, patching, and coordination WITHOUT any actual capture.

Expected: Should be near zero with our optimization guards.

### All-Layer Overhead
```
gen_overhead = (all_layers_gen_time - baseline_gen_time) / baseline_gen_time
fetch_overhead = fetch_time / baseline_gen_time
total_overhead = gen_overhead + fetch_overhead
```

This tells us the full cost of capturing all layers.

### Per-Layer Cost
```
per_layer_cost = (all_layers_gen_time - zero_layer_gen_time) / num_layers
```

Expected: Should be roughly linear with layer count.

## Debugging Tips

### If zero-layer has high overhead:
- Check that `CHATSPACE_CAPTURE_METADATA=1` (should be default)
- Verify early-exit guards are working
- Look for metadata extraction in profile

### If all-layer overhead is high:
- Check fetch time separately (GPU→CPU transfer + RPC)
- Profile to see if GPU clone or transfer dominates
- Consider testing with fewer layers to find scaling curve

### If results vary across runs:
- Increase `--iterations` (default 3)
- Add more sleep between runs in shell script
- Check for GPU memory pressure or other processes

## Environment Variables

- `CHATSPACE_CAPTURE_METADATA=0`: Disable model runner patching entirely
- `CHATSPACE_HOOK_VARIANT=noop`: Use no-op capture hook (for overhead measurement)
- `CHATSPACE_PROFILE_FETCH=1`: Enable torch profiler for fetch operations

## Next Steps

After running isolated benchmarks:
1. Compare zero-layer vs baseline to confirm optimization worked
2. Check if all-layer overhead is still unexpectedly low
3. Plot overhead vs layer count to verify linear scaling
4. Use profiler to identify remaining bottlenecks if overhead >10%
