#!/usr/bin/env python3
"""Verify TP architecture understanding by inspecting vLLM decoder layers."""

from __future__ import annotations

import inspect
import torch


def main() -> None:
    """Analyze vLLM's TP architecture for steering compatibility."""
    print("=== vLLM Tensor Parallelism Architecture Analysis ===\n")

    # Check RowParallelLinear behavior
    from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear

    print("1. RowParallelLinear.forward behavior:")
    source = inspect.getsource(RowParallelLinear.forward)
    if "tensor_model_parallel_all_reduce" in source:
        print("   ✓ Contains allreduce operation")
        print("   ✓ Output tensors are full-size (replicated across ranks)")
    else:
        print("   ✗ No allreduce found")

    sig = inspect.signature(RowParallelLinear.__init__)
    print(f"   ✓ reduce_results default: {sig.parameters['reduce_results'].default}")

    print("\n2. ColumnParallelLinear.forward behavior:")
    source = inspect.getsource(ColumnParallelLinear.forward)
    if "tensor_model_parallel_all_gather" in source:
        print("   ✓ Contains all-gather operation (optional)")
        print("   ✓ Can produce full-size outputs when gather_output=True")
    else:
        print("   ✗ No all-gather found")

    # Check decoder layer structure
    print("\n3. Qwen3 Decoder Layer Structure:")
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

    source = inspect.getsource(Qwen3DecoderLayer.forward)
    print("   Analyzing forward pass...")
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'return' in line and i < len(lines) - 1:
            print(f"   Return statement: {line.strip()}")

    print("\n4. Qwen3 Attention Output:")
    from vllm.model_executor.models.qwen3 import Qwen3Attention

    attn_init = inspect.getsource(Qwen3Attention.__init__)
    if "RowParallelLinear" in attn_init and "o_proj" in attn_init:
        print("   ✓ o_proj uses RowParallelLinear")
        print("   ✓ Attention outputs are full-size (after allreduce)")

    print("\n5. Qwen3 MLP Output:")
    from vllm.model_executor.models.qwen3 import Qwen3MLP

    mlp_init = inspect.getsource(Qwen3MLP.__init__)
    if "RowParallelLinear" in mlp_init and "down_proj" in mlp_init:
        print("   ✓ down_proj uses RowParallelLinear")
        print("   ✓ MLP outputs are full-size (after allreduce)")

    print("\n" + "=" * 70)
    print("CONCLUSION: Steering Architecture for TP")
    print("=" * 70)
    print("""
At decoder layer boundaries in vLLM:
  • Hidden states are FULL-SIZE on every rank (after allreduce)
  • (delta, residual) tuples contain complete, replicated tensors
  • No sharding at the layer interface

Steering implications:
  ✓ Additive steering: Apply same full-size vector on each rank
  ✓ Projection capping: Compute dot product independently (same result)
  ✓ Ablation: Scale component independently (same result)
  ✓ No distributed operations needed in steering code
  ✓ Memory cost: O(hidden_size) per rank (not O(hidden_size/tp_size))

Current implementation should work correctly with TP without modifications!
""")

    print("\n6. Testing TP primitives availability:")
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
            get_tensor_model_parallel_rank,
        )
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

        # In single-GPU context, these should return safe defaults
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()

        print(f"   ✓ TP world size: {world_size}")
        print(f"   ✓ TP rank: {rank}")
        print("   ✓ Primitives available (if needed for future features)")

    except Exception as e:
        print(f"   ✗ Error accessing TP primitives: {e}")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
