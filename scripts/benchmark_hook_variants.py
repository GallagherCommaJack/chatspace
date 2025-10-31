#!/usr/bin/env python3
"""
Benchmark hook variants to isolate capture overhead sources.

Usage:
    # Test all variants sequentially
    uv run python scripts/benchmark_hook_variants.py

    # Test single variant
    CHATSPACE_HOOK_VARIANT=noop uv run python scripts/benchmark_hook_variants.py --single
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VariantResult:
    """Results from benchmarking a single hook variant."""
    variant: str
    baseline_time: float
    generation_time: float
    generation_overhead_pct: float
    throughput_toks: float


async def benchmark_single_variant(variant_name: str) -> VariantResult:
    """Benchmark a specific hook variant."""
    logger.info("=" * 80)
    logger.info(f"Benchmarking hook variant: {variant_name}")
    logger.info("=" * 80)

    # Set environment variable
    os.environ["CHATSPACE_HOOK_VARIANT"] = variant_name

    # Initialize model (this will load the variant)
    logger.info("Initializing VLLMSteerModel...")
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen2.5-3B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        dtype="bfloat16",
    )

    model = VLLMSteerModel(cfg, enforce_eager=True)
    logger.info("Model initialized")

    # Test configuration (matching profiling script)
    batch_size = 16
    prefill_tokens = 512
    decode_tokens = 50

    prompts = [
        f"This is test prompt number {i}. " * (prefill_tokens // 20)
        for i in range(batch_size)
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=decode_tokens,
    )

    # Get layer indices (all layers)
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    total_layers = hf_config.num_hidden_layers
    capture_layers = list(range(total_layers))

    logger.info(f"Test config: batch={batch_size}, prefill={prefill_tokens}, "
                f"decode={decode_tokens}, layers={total_layers}")

    # Warmup
    logger.info("Warming up...")
    await model.generate(prompts[:2], sampling_params, capture_layers=[])

    # Phase 1: Baseline (no capture)
    logger.info("\nPhase 1: Baseline generation (no capture)")
    baseline_times = []

    for i in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        await model.generate(prompts, sampling_params, capture_layers=[])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        baseline_times.append(elapsed)
        logger.info(f"  Iteration {i+1}/3: {elapsed:.4f}s")

    baseline_time = sum(baseline_times) / len(baseline_times)
    total_tokens = batch_size * (prefill_tokens + decode_tokens)
    throughput = total_tokens / baseline_time

    logger.info(f"  → Average: {baseline_time:.4f}s ({throughput:.1f} tok/s)")

    # Phase 2: Generation with capture (using selected variant)
    logger.info(f"\nPhase 2: Generation with capture (variant='{variant_name}')")
    generation_times = []

    for i in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, handles = await model.generate(
            prompts,
            sampling_params,
            capture_layers=capture_layers
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generation_times.append(elapsed)
        logger.info(f"  Iteration {i+1}/3: {elapsed:.4f}s")

        # Clean up captures (don't fetch, just unregister)
        for handle in handles:
            try:
                await model._collective_rpc("unregister_capture_request", handle.request_id)
            except:
                pass

    generation_time = sum(generation_times) / len(generation_times)
    generation_overhead_pct = ((generation_time / baseline_time) - 1.0) * 100.0

    logger.info(f"  → Average: {generation_time:.4f}s (overhead: {generation_overhead_pct:.2f}%)")

    return VariantResult(
        variant=variant_name,
        baseline_time=baseline_time,
        generation_time=generation_time,
        generation_overhead_pct=generation_overhead_pct,
        throughput_toks=throughput,
    )


async def main():
    """Run benchmark for all variants."""
    single_mode = "--single" in sys.argv

    if single_mode:
        # Test only the variant specified in environment
        variant = os.environ.get("CHATSPACE_HOOK_VARIANT", "full")
        logger.info(f"\nRunning single variant: {variant}")
        result = await benchmark_single_variant(variant)

        logger.info("\n" + "=" * 80)
        logger.info("RESULT")
        logger.info("=" * 80)
        logger.info(f"Variant: {result.variant}")
        logger.info(f"Baseline: {result.baseline_time:.4f}s")
        logger.info(f"Generation: {result.generation_time:.4f}s")
        logger.info(f"Overhead: {result.generation_overhead_pct:.2f}%")

        return

    # Test all variants
    logger.info("\n" + "=" * 80)
    logger.info("HOOK VARIANT COMPARISON BENCHMARK")
    logger.info("=" * 80)

    variants = ["noop_notiming", "noop", "slice_only", "clone_only", "full"]
    results = []

    for variant in variants:
        try:
            result = await benchmark_single_variant(variant)
            results.append(result)

            # Short pause between variants
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Failed to benchmark variant '{variant}': {e}")
            continue

    if not results:
        logger.error("No successful results!")
        return

    # Save results
    output_dir = Path("/workspace/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "hook_variant_results.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("HOOK VARIANT COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Variant':<15} | {'Baseline':>10} | {'Generation':>12} | {'Overhead':>10} | {'Notes'}")
    logger.info("-" * 80)

    # Calculate incremental costs
    noop_notiming_overhead = next((r.generation_overhead_pct for r in results if r.variant == "noop_notiming"), 0.0)
    noop_overhead = next((r.generation_overhead_pct for r in results if r.variant == "noop"), 0.0)
    slice_overhead = next((r.generation_overhead_pct for r in results if r.variant == "slice_only"), 0.0)
    clone_overhead = next((r.generation_overhead_pct for r in results if r.variant == "clone_only"), 0.0)
    full_overhead = next((r.generation_overhead_pct for r in results if r.variant == "full"), 0.0)

    for result in results:
        notes = ""
        if result.variant == "noop_notiming":
            notes = "Metadata lookup only"
        elif result.variant == "noop":
            incremental = noop_overhead - noop_notiming_overhead
            notes = f"+{incremental:.1f}% from extraction (second+first)"
        elif result.variant == "slice_only":
            incremental = slice_overhead - noop_overhead
            notes = f"+{incremental:.1f}% from slicing"
        elif result.variant == "clone_only":
            incremental = clone_overhead - slice_overhead
            notes = f"+{incremental:.1f}% from cloning"
        elif result.variant == "full":
            incremental = full_overhead - clone_overhead
            notes = f"+{incremental:.1f}% from storage"

        logger.info(
            f"{result.variant:<15} | "
            f"{result.baseline_time:>9.3f}s | "
            f"{result.generation_time:>11.3f}s | "
            f"{result.generation_overhead_pct:>9.2f}% | "
            f"{notes}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 80)

    logger.info(f"\nOverhead breakdown:")
    logger.info(f"  Metadata lookup:        {noop_notiming_overhead:>6.2f}%")
    logger.info(f"  + Extraction (sec+fst): {noop_overhead - noop_notiming_overhead:>6.2f}%")
    logger.info(f"  + Slicing cost:         {slice_overhead - noop_overhead:>6.2f}%")
    logger.info(f"  + Cloning cost:         {clone_overhead - slice_overhead:>6.2f}%")
    logger.info(f"  + Storage cost:         {full_overhead - clone_overhead:>6.2f}%")
    logger.info(f"  ----------------------------------------")
    logger.info(f"  Total (full variant):   {full_overhead:>6.2f}%")

    # Identify the dominant cost
    costs = {
        "Metadata lookup": noop_notiming_overhead,
        "Extraction (second+first)": noop_overhead - noop_notiming_overhead,
        "Slicing": slice_overhead - noop_overhead,
        "Cloning": clone_overhead - slice_overhead,
        "Storage": full_overhead - clone_overhead,
    }

    dominant = max(costs.items(), key=lambda x: x[1])
    logger.info(f"\n🎯 Dominant cost: {dominant[0]} ({dominant[1]:.2f}% overhead)")

    if dominant[0] == "Extraction (second+first)":
        logger.info("\n💡 Optimization suggestion: Tensor addition (second+first) is the bottleneck!")
        logger.info("   Consider: cache the result per forward pass, or modify vLLM to return")
        logger.info("   the final hidden state directly instead of (delta, residual) tuple")
    elif dominant[0] == "Cloning":
        logger.info("\n💡 Optimization suggestion: Cloning is the bottleneck!")
        logger.info("   Consider: async copies, pinned memory, or reducing clone frequency")
    elif dominant[0] == "Storage":
        logger.info("\n💡 Optimization suggestion: Storage operations are the bottleneck!")
        logger.info("   Consider: pre-allocated buffers or more efficient data structures")
    elif dominant[0] == "Slicing":
        logger.info("\n💡 Optimization suggestion: Slicing is the bottleneck!")
        logger.info("   Consider: view-based slicing or batch processing")

    logger.info("\n" + "=" * 80)
    logger.info("Benchmark complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
