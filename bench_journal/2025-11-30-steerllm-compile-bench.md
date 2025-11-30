# steerllm torch.compile Benchmark

**Date**: 2025-11-30
**Hardware**: NVIDIA H200
**PyTorch**: 2.8.0+cu128
**CUDA**: 12.8

## Objective

Evaluate torch.compile with dynamic shapes for steerllm steering operations on the slow path (heterogeneous batches).

## Findings

### 1. torch.compile adds overhead for small tensor ops

Using `mode="reduce-overhead"` (initial attempt):
| Op | Compiled (μs) | Uncompiled (μs) | Ratio |
|---|---|---|---|
| Add | 82 | 15 | 5.7x slower |
| Projection cap | 92 | 60 | 1.5x slower |
| Ablation | 86 | 40 | 2.1x slower |

Using `mode="default"` (better):
| Op | Compiled (μs) | Uncompiled (μs) | Ratio |
|---|---|---|---|
| Add | 49 | 18 | 2.7x slower |
| Projection cap | 60 | 57 | ~same |
| Ablation | 56 | 40 | 1.4x slower |

**Conclusion**: `reduce-overhead` mode has too much per-call overhead for small operations. Default mode is better but still doesn't help.

### 2. Fast path (uniform batch) is the real optimization

| Path | Mean (μs) | Ratio |
|---|---|---|
| Fast path (uniform, 32 reqs) | 87 | 1x |
| Slow path (heterogeneous) | 1652 | 18.9x slower |

**Speedup: 18.9x** for uniform batches (all requests use same steering config).

### 3. steerllm vs chatspace comparison

| Op | steerllm (compiled) | chatspace (uncompiled) |
|---|---|---|
| Projection cap | 61 μs | 75 μs |
| Ablation | 56 μs | 42 μs |

steerllm is faster for projection cap, slightly slower for ablation.

### 4. Compilation warmup overhead

| Mode | First call | Subsequent | Amortization |
|---|---|---|---|
| reduce-overhead | 3.14 ms | 1.04 ms | 3 calls |
| default | 1.19 ms | 0.06 ms | 19 calls |

## Decision

- **Disabled torch.compile by default** (`STEERLLM_COMPILE_STEERING=0`)
- Compilation adds overhead for small tensor ops on H200
- **Fast path is the main optimization** (18.9x speedup)
- Left infrastructure in place for users to opt-in via env var

## Configuration

To enable compilation (not recommended based on benchmarks):
```bash
export STEERLLM_COMPILE_STEERING=1
```

## Test Commands

```bash
uv run python scripts/benchmark_steering_compile.py
```
