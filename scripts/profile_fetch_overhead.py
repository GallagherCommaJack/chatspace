#!/usr/bin/env python3
"""
Focused profiling script to isolate fetch overhead bottleneck.

Tests 3 workload sizes with detailed timing breakdown:
- Small: batch=8, prefill=128 tokens
- Medium: batch=8, prefill=1024 tokens
- Large: batch=32, prefill=1024 tokens

For each workload, measures:
1. Baseline generation (no capture)
2. Generation with capture (no fetch)
3. Fetch operation separately
4. Total time (generation + fetch)
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WorkloadConfig:
    """Configuration for a profiling workload."""
    name: str
    batch_size: int
    prefill_tokens: int
    decode_tokens: int
    layers_to_capture: list[int]


@dataclass
class ProfilingResult:
    """Results from profiling a single workload."""
    config: WorkloadConfig

    # Baseline (no capture)
    baseline_time: float
    baseline_throughput: float

    # With capture (no fetch)
    generation_time: float
    generation_overhead_pct: float

    # Fetch operation
    fetch_time: float
    fetch_data_mb: float
    fetch_rate_mbps: float

    # Total
    total_time: float
    total_overhead_pct: float

    # Breakdown percentages
    generation_pct_of_total: float
    fetch_pct_of_total: float


async def profile_workload(
    model: VLLMSteerModel,
    config: WorkloadConfig,
    iterations: int = 3
) -> ProfilingResult:
    """Profile a single workload configuration."""

    logger.info("="*80)
    logger.info(f"Profiling: {config.name}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Prefill tokens: {config.prefill_tokens}")
    logger.info(f"  Decode tokens: {config.decode_tokens}")
    logger.info(f"  Layers captured: {len(config.layers_to_capture)}")
    logger.info("="*80)

    # Generate prompts
    prompts = [
        f"This is test prompt number {i} for profiling. " * (config.prefill_tokens // 20)
        for i in range(config.batch_size)
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=config.decode_tokens,
    )

    # Warm up
    logger.info("Warming up...")
    await model.generate(prompts[:1], sampling_params)

    # Phase 1: Baseline (no capture)
    logger.info("\nPhase 1: Baseline generation (no capture)")
    baseline_times = []

    for i in range(iterations):
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        await model.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        baseline_times.append(elapsed)
        logger.info(f"  Iteration {i+1}/{iterations}: {elapsed:.4f}s")

    baseline_time = sum(baseline_times) / len(baseline_times)
    total_tokens = config.batch_size * (config.prefill_tokens + config.decode_tokens)
    baseline_throughput = total_tokens / baseline_time

    logger.info(f"  → Average: {baseline_time:.4f}s ({baseline_throughput:.1f} tok/s)")

    # Phase 2: Generation with capture (no fetch)
    logger.info("\nPhase 2: Generation with capture (no fetch)")
    generation_times = []

    for i in range(iterations):
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        _, handles = await model.generate(
            prompts,
            sampling_params,
            capture_layers=config.layers_to_capture
        )
        elapsed = time.perf_counter() - start

        generation_times.append(elapsed)
        logger.info(f"  Iteration {i+1}/{iterations}: {elapsed:.4f}s")

        # Clean up captures (don't fetch, just unregister)
        for handle in handles:
            try:
                await model._collective_rpc("unregister_capture_request", handle.request_id)
            except:
                pass

    generation_time = sum(generation_times) / len(generation_times)
    generation_overhead_pct = ((generation_time / baseline_time) - 1.0) * 100.0

    logger.info(f"  → Average: {generation_time:.4f}s (overhead: {generation_overhead_pct:.2f}%)")

    # Phase 3: Fetch operation (after generation)
    logger.info("\nPhase 3: Fetch operation (after generation with capture)")
    fetch_times = []
    fetch_data_sizes = []

    for i in range(iterations):
        # Generate with capture
        _, handles = await model.generate(
            prompts,
            sampling_params,
            capture_layers=config.layers_to_capture
        )

        # Time just the fetch
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        await model.fetch_captures_batch(handles)
        elapsed = time.perf_counter() - start

        fetch_times.append(elapsed)

        # Calculate data size from handles (fetch mutates them in-place)
        data_size = 0
        for handle in handles:
            if handle._captures:
                for layer_idx, layer_list in handle._captures.items():
                    for capture_dict in layer_list:
                        tensor = capture_dict["hidden"]
                        data_size += tensor.numel() * tensor.element_size()
        fetch_data_sizes.append(data_size)

        logger.info(f"  Iteration {i+1}/{iterations}: {elapsed:.4f}s ({data_size/(1024*1024):.1f}MB)")

    fetch_time = sum(fetch_times) / len(fetch_times)
    fetch_data_mb = sum(fetch_data_sizes) / len(fetch_data_sizes) / (1024 * 1024)
    fetch_rate_mbps = fetch_data_mb / fetch_time if fetch_time > 0 else 0.0

    logger.info(f"  → Average: {fetch_time:.4f}s ({fetch_data_mb:.1f}MB, {fetch_rate_mbps:.1f}MB/s)")

    # Phase 4: Total time
    total_time = generation_time + fetch_time
    total_overhead_pct = ((total_time / baseline_time) - 1.0) * 100.0

    generation_pct_of_total = (generation_time / total_time) * 100.0
    fetch_pct_of_total = (fetch_time / total_time) * 100.0

    logger.info("\nSummary:")
    logger.info(f"  Baseline:   {baseline_time:.4f}s")
    logger.info(f"  Generation: {generation_time:.4f}s (+{generation_overhead_pct:.1f}%) [{generation_pct_of_total:.1f}% of total]")
    logger.info(f"  Fetch:      {fetch_time:.4f}s [{fetch_pct_of_total:.1f}% of total]")
    logger.info(f"  Total:      {total_time:.4f}s (+{total_overhead_pct:.1f}%)")

    return ProfilingResult(
        config=config,
        baseline_time=baseline_time,
        baseline_throughput=baseline_throughput,
        generation_time=generation_time,
        generation_overhead_pct=generation_overhead_pct,
        fetch_time=fetch_time,
        fetch_data_mb=fetch_data_mb,
        fetch_rate_mbps=fetch_rate_mbps,
        total_time=total_time,
        total_overhead_pct=total_overhead_pct,
        generation_pct_of_total=generation_pct_of_total,
        fetch_pct_of_total=fetch_pct_of_total,
    )


async def main():
    logger.info("="*80)
    logger.info("Fetch Overhead Profiling - Qwen3-32B")
    logger.info("="*80)

    # Initialize model
    logger.info("\nInitializing VLLMSteerModel...")
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-32B",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
    )

    model = VLLMSteerModel(cfg, enforce_eager=True)
    logger.info("Model initialized successfully")

    # Get layer count
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    total_layers = hf_config.num_hidden_layers
    logger.info(f"Model has {total_layers} layers")

    # Define layers to capture (32 layers evenly spaced)
    capture_layers = list(range(0, total_layers, total_layers // 32))[:32]

    # Define workloads
    workloads = [
        WorkloadConfig(
            name="Small (batch=8, prefill=128)",
            batch_size=8,
            prefill_tokens=128,
            decode_tokens=50,
            layers_to_capture=capture_layers,
        ),
        WorkloadConfig(
            name="Medium (batch=8, prefill=1024)",
            batch_size=8,
            prefill_tokens=1024,
            decode_tokens=50,
            layers_to_capture=capture_layers,
        ),
        WorkloadConfig(
            name="Large (batch=32, prefill=1024)",
            batch_size=32,
            prefill_tokens=1024,
            decode_tokens=50,
            layers_to_capture=capture_layers,
        ),
    ]

    # Run profiling
    results = []
    for workload in workloads:
        result = await profile_workload(model, workload, iterations=3)
        results.append(result)

        # Short pause between workloads
        await asyncio.sleep(2)

    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY TABLE")
    logger.info("="*80)
    logger.info(f"{'Workload':<30} {'Baseline':<12} {'Generation':<15} {'Fetch':<15} {'Total OH':<12}")
    logger.info("-"*80)

    for result in results:
        logger.info(
            f"{result.config.name:<30} "
            f"{result.baseline_time:>6.2f}s      "
            f"{result.generation_overhead_pct:>6.1f}% "
            f"({result.generation_pct_of_total:>4.1f}%)  "
            f"{result.fetch_time:>6.2f}s "
            f"({result.fetch_pct_of_total:>4.1f}%)  "
            f"{result.total_overhead_pct:>7.1f}%"
        )

    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)

    # Compare fetch times
    small = results[0]
    medium = results[1]
    large = results[2]

    logger.info(f"\nFetch time scaling:")
    logger.info(f"  Small → Medium: {medium.fetch_time / small.fetch_time:.2f}x (tokens: {medium.config.prefill_tokens / small.config.prefill_tokens:.1f}x)")
    logger.info(f"  Medium → Large: {large.fetch_time / medium.fetch_time:.2f}x (batch: {large.config.batch_size / medium.config.batch_size:.1f}x)")
    logger.info(f"  Small → Large: {large.fetch_time / small.fetch_time:.2f}x (combined: {(large.config.batch_size * large.config.prefill_tokens) / (small.config.batch_size * small.config.prefill_tokens):.1f}x)")

    logger.info(f"\nFetch transfer rates:")
    logger.info(f"  Small:  {small.fetch_rate_mbps:.1f} MB/s ({small.fetch_data_mb:.1f}MB in {small.fetch_time:.2f}s)")
    logger.info(f"  Medium: {medium.fetch_rate_mbps:.1f} MB/s ({medium.fetch_data_mb:.1f}MB in {medium.fetch_time:.2f}s)")
    logger.info(f"  Large:  {large.fetch_rate_mbps:.1f} MB/s ({large.fetch_data_mb:.1f}MB in {large.fetch_time:.2f}s)")

    logger.info(f"\nFetch vs Generation overhead:")
    for result in results:
        logger.info(f"  {result.config.name}: Fetch is {result.fetch_time / result.generation_time:.2f}x slower than generation overhead")

    logger.info("\n" + "="*80)
    logger.info("Profiling complete!")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
