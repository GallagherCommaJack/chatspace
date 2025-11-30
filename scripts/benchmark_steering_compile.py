"""Benchmark torch.compile for steering operations.

Compares:
1. Compiled vs uncompiled steering ops
2. Fast path (uniform batch) vs slow path (heterogeneous)
3. steerllm vs chatspace runtime implementations

Usage:
    uv run python scripts/benchmark_steering_compile.py
"""

import os
import sys
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable

# Add parent dir to path for steerllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Import steerllm runtime
from steerllm.backends.vllm import runtime as steer_rt

# ============================================================================
# Benchmark utilities
# ============================================================================

@dataclass
class BenchResult:
    name: str
    mean_us: float
    std_us: float
    iterations: int


def bench_fn(fn: Callable, warmup: int = 10, iters: int = 100) -> BenchResult:
    """Benchmark a function, return mean/std in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # microseconds

    return BenchResult(
        name="",
        mean_us=np.mean(times),
        std_us=np.std(times),
        iterations=iters,
    )


def print_table(results: list[BenchResult], title: str):
    """Print results as a table."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Name':<40} | {'Mean (μs)':<12} | {'Std (μs)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<40} | {r.mean_us:<12.2f} | {r.std_us:<10.2f}")
    print()


# ============================================================================
# Test steerllm compiled vs uncompiled
# ============================================================================

def bench_steerllm_compiled_ops():
    """Benchmark steerllm compiled steering ops."""
    print("\n[1] Benchmarking steerllm compiled ops (torch.compile default mode)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Create test tensors - larger batch to see if compilation helps
    hidden_size = 2048
    seq_lens = [256, 512, 384, 320]  # Larger sequences
    total_tokens = sum(seq_lens)

    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    vec = torch.randn(hidden_size, device=device, dtype=dtype)
    vec = vec / vec.norm()  # Unit vector

    results = []

    # --- Additive steering ---
    # Compiled (clear cache to force fresh compilation)
    steer_rt._COMPILE_STEERING = True
    steer_rt._compiled_ops.clear()  # Clear cache
    add_compiled = steer_rt._get_compiled_add()

    def bench_add_compiled():
        return add_compiled(hidden, vec)

    r = bench_fn(bench_add_compiled)
    r.name = "Add (compiled)"
    results.append(r)

    # Uncompiled
    def bench_add_uncompiled():
        return hidden + vec

    r = bench_fn(bench_add_uncompiled)
    r.name = "Add (uncompiled)"
    results.append(r)

    # --- Projection cap ---
    cap_min, cap_max = -0.5, 0.5

    # Compiled (both bounds)
    cap_compiled = steer_rt._get_compiled_projection_cap(True, True)

    def bench_cap_compiled():
        return cap_compiled(hidden, vec, cap_min, cap_max)

    r = bench_fn(bench_cap_compiled)
    r.name = "Projection cap (compiled, both bounds)"
    results.append(r)

    # Uncompiled (using original function)
    config = steer_rt._ProjectionCapConfig(unit_vector=vec, min=cap_min, max=cap_max)

    def bench_cap_uncompiled():
        return steer_rt._apply_projection_cap(hidden, config)

    r = bench_fn(bench_cap_uncompiled)
    r.name = "Projection cap (uncompiled)"
    results.append(r)

    # --- Ablation ---
    scale = 0.5

    # Compiled
    ablation_compiled = steer_rt._get_compiled_ablation()

    def bench_ablation_compiled():
        return ablation_compiled(hidden, vec, scale)

    r = bench_fn(bench_ablation_compiled)
    r.name = "Ablation (compiled)"
    results.append(r)

    # Uncompiled
    ablation_config = steer_rt._AblationConfig(unit_vector=vec, scale=scale)

    def bench_ablation_uncompiled():
        return steer_rt._apply_ablation(hidden, ablation_config)

    r = bench_fn(bench_ablation_uncompiled)
    r.name = "Ablation (uncompiled)"
    results.append(r)

    print_table(results, "steerllm: Compiled vs Uncompiled Ops")
    return results


# ============================================================================
# Test fast path vs slow path
# ============================================================================

def bench_fast_vs_slow_path():
    """Benchmark uniform (fast) vs heterogeneous (slow) batch steering."""
    print("\n[2] Benchmarking fast path vs slow path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    hidden_size = 2048
    num_requests = 32
    seq_len_per_request = 64
    total_tokens = num_requests * seq_len_per_request

    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    vec = torch.randn(hidden_size, device=device, dtype=dtype)
    vec = vec / vec.norm()

    # Create mock state
    state = steer_rt._SteeringState(
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )

    # Create layer spec
    class LayerSpec:
        pass

    layer_spec = LayerSpec()
    layer_spec.operations = [("add", vec, None)]

    # Create identical specs for all requests (fast path)
    class Spec:
        pass

    uniform_spec = Spec()
    uniform_spec.layers = {0: layer_spec}

    # Register same spec for all requests (fast path - uses identity check)
    request_ids = [f"req_{i}" for i in range(num_requests)]
    seq_lens = [seq_len_per_request] * num_requests

    for req_id in request_ids:
        state.request_steering_specs[req_id] = uniform_spec  # Same object!

    results = []

    # Fast path benchmark
    steer_rt._COMPILE_STEERING = True

    def bench_fast_path():
        h = hidden.clone()
        # Check uniformity
        first_layer_spec = None
        all_same = True
        for req_id in request_ids:
            spec = state.request_steering_specs.get(req_id)
            if spec and 0 in spec.layers:
                ls = spec.layers[0]
                if first_layer_spec is None:
                    first_layer_spec = ls
                elif ls is not first_layer_spec:
                    all_same = False
                    break

        if all_same and first_layer_spec:
            total = sum(seq_lens)
            h[:total] = steer_rt._apply_layer_steering_to_hidden(h[:total], first_layer_spec, state)
        return h

    r = bench_fn(bench_fast_path)
    r.name = f"Fast path (uniform, {num_requests} reqs)"
    results.append(r)

    # Slow path benchmark - create separate spec objects
    for i, req_id in enumerate(request_ids):
        separate_spec = Spec()
        separate_layer_spec = LayerSpec()
        separate_layer_spec.operations = [("add", vec, None)]
        separate_spec.layers = {0: separate_layer_spec}
        state.request_steering_specs[req_id] = separate_spec  # Different objects

    def bench_slow_path():
        h = hidden.clone()
        start_idx = 0
        for i, req_id in enumerate(request_ids):
            seq_len = seq_lens[i]
            end_idx = start_idx + seq_len
            spec = state.request_steering_specs.get(req_id)
            if spec and 0 in spec.layers:
                h[start_idx:end_idx] = steer_rt._apply_layer_steering_to_hidden(
                    h[start_idx:end_idx], spec.layers[0], state
                )
            start_idx = end_idx
        return h

    r = bench_fn(bench_slow_path)
    r.name = f"Slow path (heterogeneous, {num_requests} reqs)"
    results.append(r)

    print_table(results, "Fast Path vs Slow Path")

    # Calculate speedup
    fast_time = results[0].mean_us
    slow_time = results[1].mean_us
    speedup = slow_time / fast_time
    print(f"  Fast path speedup: {speedup:.1f}x")

    return results


# ============================================================================
# Compare steerllm vs chatspace runtimes
# ============================================================================

def bench_steerllm_vs_chatspace():
    """Compare steerllm and chatspace runtime implementations."""
    print("\n[3] Benchmarking steerllm vs chatspace runtime")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    hidden_size = 2048
    seq_len = 512
    hidden = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    vec = torch.randn(hidden_size, device=device, dtype=dtype)
    vec = vec / vec.norm()

    results = []

    # steerllm projection cap
    steer_rt._COMPILE_STEERING = True
    steer_rt._compiled_ops.clear()

    steer_config = steer_rt._ProjectionCapConfig(unit_vector=vec, min=-0.5, max=0.5)
    steer_cap_compiled = steer_rt._get_compiled_projection_cap(True, True)

    def bench_steerllm_cap():
        return steer_cap_compiled(hidden, vec, -0.5, 0.5)

    r = bench_fn(bench_steerllm_cap)
    r.name = "steerllm: projection cap (compiled)"
    results.append(r)

    # chatspace projection cap
    try:
        from chatspace.vllm_steering import runtime as chat_rt

        chat_config = chat_rt._ProjectionCapConfig(unit_vector=vec, min=-0.5, max=0.5)

        def bench_chatspace_cap():
            return chat_rt._apply_projection_cap(hidden, chat_config)

        r = bench_fn(bench_chatspace_cap)
        r.name = "chatspace: projection cap (uncompiled)"
        results.append(r)
    except ImportError:
        print("  [SKIP] chatspace runtime not available")

    # steerllm ablation
    steer_ablation_compiled = steer_rt._get_compiled_ablation()

    def bench_steerllm_ablation():
        return steer_ablation_compiled(hidden, vec, 0.5)

    r = bench_fn(bench_steerllm_ablation)
    r.name = "steerllm: ablation (compiled)"
    results.append(r)

    # chatspace ablation
    try:
        from chatspace.vllm_steering import runtime as chat_rt

        chat_ablation_config = chat_rt._AblationConfig(unit_vector=vec, scale=0.5)

        def bench_chatspace_ablation():
            return chat_rt._apply_ablation(hidden, chat_ablation_config)

        r = bench_fn(bench_chatspace_ablation)
        r.name = "chatspace: ablation (uncompiled)"
        results.append(r)
    except ImportError:
        pass

    print_table(results, "steerllm vs chatspace Runtime")
    return results


# ============================================================================
# Compilation warmup overhead
# ============================================================================

def bench_compilation_overhead():
    """Measure first-call compilation overhead."""
    print("\n[4] Measuring compilation warmup overhead")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    hidden_size = 2048
    seq_len = 256
    hidden = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    vec = torch.randn(hidden_size, device=device, dtype=dtype)
    vec = vec / vec.norm()

    results = []

    # Clear cache to force recompilation
    steer_rt._compiled_ops.clear()
    steer_rt._COMPILE_STEERING = True

    # First call (includes compilation)
    torch.cuda.synchronize()
    start = time.perf_counter()
    cap_fn = steer_rt._get_compiled_projection_cap(True, True)
    _ = cap_fn(hidden, vec, -0.5, 0.5)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000

    # Subsequent calls
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = cap_fn(hidden, vec, -0.5, 0.5)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_subsequent_ms = np.mean(times)

    print(f"  First call (incl. compilation): {first_call_ms:.2f} ms")
    print(f"  Subsequent calls (avg):         {avg_subsequent_ms:.4f} ms")
    print(f"  Compilation overhead:           {first_call_ms - avg_subsequent_ms:.2f} ms")
    print(f"  Amortization point:             {int(first_call_ms / avg_subsequent_ms)} calls")

    return {
        "first_call_ms": first_call_ms,
        "avg_subsequent_ms": avg_subsequent_ms,
        "compilation_overhead_ms": first_call_ms - avg_subsequent_ms,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print(" Steering Compilation Benchmark")
    print("="*60)

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, benchmarks may not be representative")

    print(f"PyTorch version: {torch.__version__}")

    # Run benchmarks
    bench_steerllm_compiled_ops()
    bench_fast_vs_slow_path()
    bench_steerllm_vs_chatspace()
    bench_compilation_overhead()

    print("\n" + "="*60)
    print(" Benchmark Complete")
    print("="*60)


if __name__ == "__main__":
    main()
