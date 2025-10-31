#!/usr/bin/env python3
"""
Comprehensive benchmark for vLLM activation capture performance.

Measures:
- Per-layer capture overhead
- Prefill-only vs mixed prefill+decode workflows
- High-throughput batching (8-32 requests)
- Concurrent request handling
- Identifies bottlenecks via conditional profiling (>20% overhead)

Usage:
    uv run python scripts/benchmark_vllm_capture.py [--quick]
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    scenario: str  # "prefill_only", "mixed", "concurrent"
    batch_size: int
    num_layers_captured: int
    layer_description: str  # "none", "single", "first_half", etc.
    prefill_tokens: int
    decode_tokens: int
    concurrency: int = 1  # Number of concurrent batches

    def __str__(self):
        return (f"{self.scenario:15s} | batch={self.batch_size:2d} | "
                f"layers={self.num_layers_captured:2d} ({self.layer_description:12s}) | "
                f"prefill={self.prefill_tokens:4d} | decode={self.decode_tokens:3d} | "
                f"concurrency={self.concurrency}")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig

    # Timing metrics (seconds)
    baseline_time: float
    capture_time: float
    fetch_time: float

    # Derived metrics
    total_time: float
    overhead_pct: float

    # Throughput metrics
    tokens_per_sec: float
    requests_per_sec: float

    # Memory metrics
    peak_memory_gb: float

    # Statistics (from multiple runs)
    baseline_stddev: float = 0.0
    capture_stddev: float = 0.0
    fetch_stddev: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "config": {
                "scenario": self.config.scenario,
                "batch_size": self.config.batch_size,
                "num_layers_captured": self.config.num_layers_captured,
                "layer_description": self.config.layer_description,
                "prefill_tokens": self.config.prefill_tokens,
                "decode_tokens": self.config.decode_tokens,
                "concurrency": self.config.concurrency,
            },
            "timing": {
                "baseline_time": self.baseline_time,
                "capture_time": self.capture_time,
                "fetch_time": self.fetch_time,
                "total_time": self.total_time,
                "baseline_stddev": self.baseline_stddev,
                "capture_stddev": self.capture_stddev,
                "fetch_stddev": self.fetch_stddev,
            },
            "metrics": {
                "overhead_pct": self.overhead_pct,
                "tokens_per_sec": self.tokens_per_sec,
                "requests_per_sec": self.requests_per_sec,
            },
            "memory": {
                "peak_memory_gb": self.peak_memory_gb,
            },
        }


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results."""

    timestamp: str
    model_name: str
    tensor_parallel_size: int
    total_model_layers: int

    results: list[BenchmarkResult] = field(default_factory=list)
    worst_case_config: Optional[BenchmarkConfig] = None
    worst_case_overhead: float = 0.0
    profiling_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "model_name": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "total_model_layers": self.total_model_layers,
            },
            "worst_case": {
                "config": str(self.worst_case_config) if self.worst_case_config else None,
                "overhead_pct": self.worst_case_overhead,
                "profiling_triggered": self.profiling_triggered,
            },
            "results": [r.to_dict() for r in self.results],
        }


# ============================================================================
# Layer Configuration Generator
# ============================================================================


def generate_layer_configs(total_layers: int = 64) -> list[tuple[list[int], str]]:
    """
    Generate layer capture configurations for benchmarking.

    Returns:
        List of (layer_indices, description) tuples.
    """
    configs = [
        ([], "none"),
        ([total_layers // 2], "single"),
        (list(range(total_layers // 2)), "first_half"),
        (list(range(total_layers // 2, total_layers)), "second_half"),
        (list(range(0, total_layers, 2)), "every_other"),
        (list(range(total_layers)), "all"),
    ]
    return configs


# ============================================================================
# Benchmark Runner
# ============================================================================


class CaptureBenchmark:
    """Main benchmark runner."""

    def __init__(
        self,
        model: VLLMSteerModel,
        num_iterations: int = 3,
        warmup_iterations: int = 2,
    ):
        self.model = model
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations

        # Get model metadata
        self.model_config = self.model.cfg

        # Dynamically determine number of layers from model
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(self.model_config.model_name, trust_remote_code=True)
        self.total_layers = hf_config.num_hidden_layers
        logger.info(f"Model has {self.total_layers} layers")

    async def run_single_benchmark(
        self,
        config: BenchmarkConfig,
        layer_indices: list[int],
        warmup: bool = False,
    ) -> Optional[BenchmarkResult]:
        """
        Run a single benchmark configuration.

        Returns:
            BenchmarkResult if not warmup, None otherwise.
        """
        # Generate prompts
        prompts = [
            f"This is test prompt number {i} for benchmarking purposes." * (config.prefill_tokens // 20)
            for i in range(config.batch_size)
        ]

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=config.decode_tokens,
        )

        # Collect measurements
        baseline_times = []
        capture_times = []
        fetch_times = []
        memories = []

        iterations = 1 if warmup else self.num_iterations

        for _ in range(iterations):
            torch.cuda.reset_peak_memory_stats()

            # Baseline (no capture)
            start = time.perf_counter()
            if config.scenario == "concurrent":
                # Run multiple concurrent batches
                tasks = [
                    self.model.generate(prompts, sampling_params)
                    for _ in range(config.concurrency)
                ]
                await asyncio.gather(*tasks)
            else:
                await self.model.generate(prompts, sampling_params)
            baseline_time = time.perf_counter() - start
            baseline_times.append(baseline_time)

            # With capture (skip if no layers)
            if layer_indices:
                torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()
                if config.scenario == "concurrent":
                    # Run multiple concurrent batches with capture
                    tasks = [
                        self.model.generate(prompts, sampling_params, capture_layers=layer_indices)
                        for _ in range(config.concurrency)
                    ]
                    results = await asyncio.gather(*tasks)
                    # Extract handles from each (texts, handles) tuple
                    all_handles = []
                    for texts, handles in results:
                        all_handles.extend(handles)
                else:
                    _, handles = await self.model.generate(
                        prompts, sampling_params, capture_layers=layer_indices
                    )
                    all_handles = handles

                capture_time = time.perf_counter() - start
                capture_times.append(capture_time)

                # Fetch captures
                start = time.perf_counter()
                await self.model.fetch_captures_batch(all_handles)
                fetch_time = time.perf_counter() - start
                fetch_times.append(fetch_time)

                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                memories.append(peak_memory)
            else:
                # No capture = no overhead
                capture_times.append(baseline_time)
                fetch_times.append(0.0)
                memories.append(torch.cuda.max_memory_allocated() / 1024**3)

        if warmup:
            return None

        # Calculate statistics
        baseline_median = statistics.median(baseline_times)
        capture_median = statistics.median(capture_times)
        fetch_median = statistics.median(fetch_times)

        baseline_stddev = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0.0
        capture_stddev = statistics.stdev(capture_times) if len(capture_times) > 1 else 0.0
        fetch_stddev = statistics.stdev(fetch_times) if len(fetch_times) > 1 else 0.0

        total_time = capture_median + fetch_median
        overhead_pct = ((total_time / baseline_median) - 1.0) * 100.0 if baseline_median > 0 else 0.0

        # Calculate throughput
        total_requests = config.batch_size * config.concurrency
        total_tokens = total_requests * (config.prefill_tokens + config.decode_tokens)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        requests_per_sec = total_requests / total_time if total_time > 0 else 0.0

        return BenchmarkResult(
            config=config,
            baseline_time=baseline_median,
            capture_time=capture_median,
            fetch_time=fetch_median,
            total_time=total_time,
            overhead_pct=overhead_pct,
            tokens_per_sec=tokens_per_sec,
            requests_per_sec=requests_per_sec,
            peak_memory_gb=statistics.median(memories),
            baseline_stddev=baseline_stddev,
            capture_stddev=capture_stddev,
            fetch_stddev=fetch_stddev,
        )

    async def run_warmup(self):
        """Run warmup iterations."""
        logger.info("Running warmup iterations...")

        warmup_config = BenchmarkConfig(
            scenario="mixed",
            batch_size=8,
            num_layers_captured=1,
            layer_description="warmup",
            prefill_tokens=128,
            decode_tokens=50,
            concurrency=1,
        )

        for i in range(self.warmup_iterations):
            await self.run_single_benchmark(warmup_config, [32], warmup=True)
            logger.info(f"  Warmup iteration {i+1}/{self.warmup_iterations} complete")

    async def run_benchmark_matrix(
        self,
        scenarios: list[str],
        batch_sizes: list[int],
        prefill_tokens: list[int],
        decode_tokens: list[int],
        concurrency_levels: list[int],
    ) -> BenchmarkSummary:
        """
        Run the full benchmark matrix.

        Args:
            scenarios: List of scenario names ("prefill_only", "mixed", "concurrent")
            batch_sizes: List of batch sizes to test
            prefill_tokens: List of prefill token counts
            decode_tokens: List of decode token counts (use 5 for prefill_only)
            concurrency_levels: List of concurrency levels (only used for "concurrent" scenario)
        """
        summary = BenchmarkSummary(
            timestamp=datetime.utcnow().isoformat(),
            model_name=self.model_config.model_name,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            total_model_layers=self.total_layers,
        )

        # Generate layer configurations
        layer_configs = generate_layer_configs(self.total_layers)

        # Run warmup
        await self.run_warmup()

        # Run benchmark matrix
        total_configs = (
            len(scenarios) * len(batch_sizes) * len(prefill_tokens) *
            len(decode_tokens) * len(layer_configs)
        )
        logger.info(f"Running {total_configs} benchmark configurations...")

        config_idx = 0
        for scenario in scenarios:
            for batch_size in batch_sizes:
                for prefill_tok in prefill_tokens:
                    for decode_tok in decode_tokens:
                        # Skip prefill-only with high decode tokens
                        if scenario == "prefill_only" and decode_tok > 5:
                            continue

                        for layer_indices, layer_desc in layer_configs:
                            config_idx += 1

                            # Determine concurrency
                            if scenario == "concurrent":
                                concurrency = max(concurrency_levels)
                            else:
                                concurrency = 1

                            config = BenchmarkConfig(
                                scenario=scenario,
                                batch_size=batch_size,
                                num_layers_captured=len(layer_indices),
                                layer_description=layer_desc,
                                prefill_tokens=prefill_tok,
                                decode_tokens=decode_tok,
                                concurrency=concurrency,
                            )

                            logger.info(f"[{config_idx}/{total_configs}] {config}")

                            result = await self.run_single_benchmark(config, layer_indices)
                            if result:
                                summary.results.append(result)

                                # Print result
                                logger.info(
                                    f"  → Overhead: {result.overhead_pct:6.2f}% | "
                                    f"Throughput: {result.tokens_per_sec:7.1f} tok/s | "
                                    f"Memory: {result.peak_memory_gb:.2f} GB"
                                )

                                # Track worst case
                                if result.overhead_pct > summary.worst_case_overhead:
                                    summary.worst_case_overhead = result.overhead_pct
                                    summary.worst_case_config = config

        return summary

    async def run_profiled_benchmark(
        self,
        config: BenchmarkConfig,
        layer_indices: list[int],
        output_path: Path,
    ):
        """
        Run a single benchmark with PyTorch profiler enabled.

        Args:
            config: Benchmark configuration
            layer_indices: Layer indices to capture
            output_path: Path to save profiler trace
        """
        logger.info(f"Running profiled benchmark: {config}")

        prompts = [
            f"This is test prompt number {i} for profiling." * (config.prefill_tokens // 20)
            for i in range(config.batch_size)
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=config.decode_tokens,
        )

        # Run with profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            if config.scenario == "concurrent":
                tasks = [
                    self.model.generate(prompts, sampling_params, capture_layers=layer_indices)
                    for _ in range(config.concurrency)
                ]
                results = await asyncio.gather(*tasks)
                # Extract handles from each (texts, handles) tuple
                all_handles = []
                for texts, handles in results:
                    all_handles.extend(handles)
            else:
                _, handles = await self.model.generate(
                    prompts, sampling_params, capture_layers=layer_indices
                )
                all_handles = handles

            await self.model.fetch_captures_batch(all_handles)

        # Export trace
        prof.export_chrome_trace(str(output_path))
        logger.info(f"  Profiler trace saved to: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================


async def main(quick: bool = False):
    """Main benchmark entry point."""

    logger.info("="*80)
    logger.info("vLLM Activation Capture Performance Benchmark")
    logger.info("="*80)

    # Initialize model
    logger.info("Initializing VLLMSteerModel (Qwen3-32B)...")
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-32B",
        tensor_parallel_size=2,  # 32B requires 2 GPUs
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
    )

    model = VLLMSteerModel(cfg, enforce_eager=True)
    logger.info("Model initialized successfully")

    # Create benchmark runner
    benchmark = CaptureBenchmark(
        model=model,
        num_iterations=3 if not quick else 1,
        warmup_iterations=2 if not quick else 1,
    )

    # Define benchmark parameters
    if quick:
        scenarios = ["mixed"]
        batch_sizes = [8, 16]
        prefill_tokens = [128, 512]
        decode_tokens = [50]
        concurrency_levels = [2]
    else:
        # Reduced matrix for overnight 32B benchmark
        scenarios = ["mixed", "concurrent"]
        batch_sizes = [8, 32]
        prefill_tokens = [128, 1024]
        decode_tokens = [50, 100]
        concurrency_levels = [1, 8]

    # Run benchmark matrix
    summary = await benchmark.run_benchmark_matrix(
        scenarios=scenarios,
        batch_sizes=batch_sizes,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        concurrency_levels=concurrency_levels,
    )

    # Save results
    output_dir = Path("/workspace/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"capture_overhead_{summary.timestamp.replace(':', '-')}.json"
    with open(results_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    logger.info(f"Total configurations tested: {len(summary.results)}")
    logger.info(f"Worst-case overhead: {summary.worst_case_overhead:.2f}%")
    if summary.worst_case_config:
        logger.info(f"Worst-case config: {summary.worst_case_config}")

    # Trigger profiling if needed
    if summary.worst_case_overhead > 20.0 and summary.worst_case_config:
        logger.info("\n" + "="*80)
        logger.info("PROFILING WORST-CASE SCENARIO (>20% overhead)")
        logger.info("="*80)

        # Find layer indices for worst case
        worst_result = max(summary.results, key=lambda r: r.overhead_pct)
        layer_configs = generate_layer_configs(benchmark.total_layers)
        worst_layers = [
            indices for indices, desc in layer_configs
            if desc == worst_result.config.layer_description
        ][0]

        profile_path = output_dir / f"profile_worst_case_{summary.timestamp.replace(':', '-')}.json"
        await benchmark.run_profiled_benchmark(
            config=worst_result.config,
            layer_indices=worst_layers,
            output_path=profile_path,
        )

        summary.profiling_triggered = True

        # Re-save results with profiling flag
        with open(results_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)

    # Print top 10 highest overhead configurations
    logger.info("\n" + "="*80)
    logger.info("TOP 10 HIGHEST OVERHEAD CONFIGURATIONS")
    logger.info("="*80)

    sorted_results = sorted(summary.results, key=lambda r: r.overhead_pct, reverse=True)[:10]
    for i, result in enumerate(sorted_results, 1):
        logger.info(f"{i:2d}. {result.config}")
        logger.info(
            f"    Overhead: {result.overhead_pct:6.2f}% | "
            f"Throughput: {result.tokens_per_sec:7.1f} tok/s | "
            f"Capture: {result.capture_time:.4f}s | Fetch: {result.fetch_time:.4f}s"
        )

    logger.info("\n" + "="*80)
    logger.info("Benchmark complete!")
    logger.info("="*80)


if __name__ == "__main__":
    import sys

    quick = "--quick" in sys.argv
    asyncio.run(main(quick=quick))
