# Precision Testing Scripts

This directory contains diagnostic tools for investigating numerical precision issues in vLLM steering operations, particularly projection capping.

## Core Diagnostic Tool

### `debug_decode_precision.py`

Compare projection cap behavior across different precision settings.

**Usage:**
```bash
python scripts/debug_decode_precision.py \
  --layer 12 \
  --cap 5.0 \
  --max-tokens 32 \
  --compare-precision float32
```

**What it does:**
- Runs a prompt with projection capping in the model's native dtype (typically bf16)
- Runs again with `--compare-precision` override (e.g., float32)
- Compares hidden states, cap_delta tensors, and decode logprobs
- Shows where and how numerical drift occurs

**Key outputs:**
- Step-by-step hidden state drift (MAE, RMSE)
- cap_delta tensor comparisons
- Decode logprob divergence tracking

## Comprehensive Sweep Scripts

### `precision_sweep_with_captures.py`

Full vLLM precision sweep capturing hidden states and cap_delta across multiple dtypes.

**Usage:**
```bash
python scripts/precision_sweep_with_captures.py \
  --layer 12 \
  --cap 5.0 \
  --max-tokens 64 \
  --run-id my_investigation \
  --vector-path /path/to/projection_vector.pt
```

**Configs tested:**
- `bf16`: Native model dtype with bf16 cap math
- `bf16_cap_fp32`: Model in bf16, cap math in fp32
- `float16`: Full float16 pipeline
- `float32`: Reference baseline

**Outputs:**
- Parquet files with token IDs, hidden states, cap_deltas per prompt
- Summary statistics (token match ratios, first divergence points)
- Saved to `/workspace/cache/precision_sweeps/{run_id}/`

### `hf_sweep_with_captures.py`

Equivalent sweep for HuggingFace/persona baseline using `create_projection_cap_steerer`.

**Usage:**
```bash
python scripts/hf_sweep_with_captures.py \
  --layer 12 \
  --cap 5.0 \
  --max-tokens 64 \
  --run-id my_investigation \
  --vector-path /path/to/projection_vector.pt
```

**Note:** Requires `persona-subspace` repository in Python path.

## Analysis Tools

### `analyze_parity_sweeps.py`

Analyze sweep results to identify where and how divergence occurs.

**Usage:**
```bash
python scripts/analyze_parity_sweeps.py \
  --run-id my_investigation \
  --prompt-idx 0
```

**Outputs:**
- Token match statistics per backend
- Step-by-step MAE and cosine similarity
- First divergence points correlated with hidden state drift
- Cross-backend comparison (HF vs vLLM)

### `compare_cap_deltas.py`

Detailed comparison of cap_delta tensors between runs.

**Usage:**
```bash
python scripts/compare_cap_deltas.py \
  --hf-dir /workspace/cache/hf_precision_sweeps/run1 \
  --vllm-dir /workspace/cache/precision_sweeps/run1 \
  --config bf16 \
  --reference-config float32 \
  --prompt-idx 0
```

**What it shows:**
- Decode-step drift analysis with token correlation
- cap_delta norm evolution
- Cross-backend drift comparison

## Typical Workflow

### 1. Quick Diagnostic

```bash
# Check if precision is causing issues
python scripts/debug_decode_precision.py --layer 12 --cap 5.0
```

### 2. Comprehensive Sweep

```bash
# Generate shared projection vector
python -c "
import torch
vec = torch.randn(1024) / torch.randn(1024).norm()
torch.save(vec, 'projection_vector.pt')
"

# Run vLLM sweep
python scripts/precision_sweep_with_captures.py \
  --vector-path projection_vector.pt \
  --run-id investigation_001

# Run HF baseline (optional)
python scripts/hf_sweep_with_captures.py \
  --vector-path projection_vector.pt \
  --run-id investigation_001
```

### 3. Analysis

```bash
# Analyze results
python scripts/analyze_parity_sweeps.py --run-id investigation_001

# Deep dive on specific prompt
python scripts/compare_cap_deltas.py \
  --hf-dir /workspace/cache/hf_precision_sweeps/investigation_001 \
  --vllm-dir /workspace/cache/precision_sweeps/investigation_001 \
  --prompt-idx 1
```

## Understanding the Metrics

**Token Match Ratio**: Fraction of generated tokens that match the fp32 reference. <90% suggests significant drift.

**MAE (Mean Absolute Error)**: Average per-element absolute difference in hidden states. Typical thresholds:
- <0.01: Excellent parity
- 0.01-0.1: Acceptable drift
- 0.1-1.0: Noticeable divergence
- >1.0: Significant drift

**First Divergence**: Token index where first mismatch occurs. Earlier divergence = less stable.

**cap_delta norm**: Magnitude of the projection cap adjustment. Typical range 5-15 for threshold=5.0.

## Known Findings

Based on comprehensive testing (see `/workspace/cache/precision_sweeps/20251020_FINDINGS.md`):

- **bf16**: 40-60% token match, diverges after ~20-30 tokens
- **float16**: 90-95% token match, diverges after ~50+ tokens
- **float32**: Reference (100% match)

**Recommendation**: Use float16 for production steering with projection caps. Only use bf16 for <5 steered layers on short sequences.

## Troubleshooting

**"No decode captures recorded"**: The HF capture hook may need adjustment. Check that the hook is registered on the correct layer output.

**"Vector length mismatch"**: Ensure projection vector matches model hidden_size (e.g., 1024 for Qwen3-0.6B, varies by model).

**"Engine died unexpectedly"**: Check GPU memory. Reduce `--gpu-memory-utilization` or `--max-model-len`.

## Dependencies

- vLLM (for vLLM sweeps)
- transformers (for HF sweeps)
- persona-subspace (for HF projection cap steerer)
- torch
- pandas (for future parquet support)
