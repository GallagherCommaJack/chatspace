#!/usr/bin/env python3
"""
Benchmark activation capture overhead across model scales.

Runs sequential workloads for Qwen3 models with consistent batch/prompt sizes,
measuring baseline generation, capture-enabled generation, and fetch timings
for different layer counts. Results stream to a JSONL file so partial progress
can be inspected while the sweep runs.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoConfig
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig
from chatspace.vllm_steering import runtime as steering_runtime


logger = logging.getLogger("capture_scaling")


@dataclass
class BenchmarkConfig:
    """Experiment parameters shared across models."""

    models: Sequence[str]
    tensor_parallel_size: int
    batch_size: int
    prefill_tokens: int
    decode_tokens: int
    iterations: int
    output_dir: Path
    profile_trace_model: str
    capture_buckets: Sequence[str]


def parse_args(argv: Sequence[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/indexes/perf") / time.strftime("%Y%m%dT%H%M%S"),
        help="Directory for JSONL results, logs, and profiler traces.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prompts.",
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=1024,
        help="Approximate number of prompt tokens.",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=256,
        help="Number of tokens to decode per request.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallel degree.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-32B",
        ],
        help="Model identifiers to benchmark sequentially.",
    )
    parser.add_argument(
        "--profile-trace-model",
        default="Qwen/Qwen3-32B",
        help="Model to rerun with torch profiler + trace export.",
    )
    parser.add_argument(
        "--capture-buckets",
        type=str,
        default="zero,one,eight,all",
        help="Comma-separated capture bucket names to run (subset of zero,one,eight,all).",
    )

    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    return BenchmarkConfig(
        models=args.models,
        tensor_parallel_size=args.tp,
        batch_size=args.batch_size,
        prefill_tokens=args.prefill_tokens,
        decode_tokens=args.decode_tokens,
        iterations=args.iterations,
        output_dir=output_dir,
        profile_trace_model=args.profile_trace_model,
        capture_buckets=tuple(
            bucket.strip().lower()
            for bucket in args.capture_buckets.split(",")
            if bucket.strip()
        ),
    )


def configure_logging(output_dir: Path) -> None:
    log_path = output_dir / "capture_scaling.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    stream.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    root.addHandler(stream)


def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_prompts(batch_size: int, prefill_tokens: int) -> list[str]:
    """Construct simple prompts with approximate token counts."""
    token = "benchmark"
    repeated = " ".join([token] * prefill_tokens)
    prompts = []
    for idx in range(batch_size):
        prompts.append(f"Request {idx}: {repeated}")
    return prompts


def evenly_spaced_layers(total_layers: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= total_layers:
        return list(range(total_layers))

    positions: list[int] = []
    for i in range(count):
        center = (i + 0.5) * total_layers / count
        idx = max(0, min(total_layers - 1, int(round(center - 0.5))))
        if idx not in positions:
            positions.append(idx)
    while len(positions) < count:
        candidate = len(positions) % total_layers
        if candidate not in positions:
            positions.append(candidate)
        else:
            positions.append((candidate + 1) % total_layers)
    return sorted(positions[:count])


def compute_bytes(handles: Sequence) -> int:
    total = 0
    for handle in handles:
        captures = getattr(handle, "_captures", None)
        if not captures:
            continue
        for layer_list in captures.values():
            for capture in layer_list:
                tensor = capture.get("hidden")
                if isinstance(tensor, torch.Tensor):
                    total += tensor.numel() * tensor.element_size()
    return total


def make_record_writer(path: Path):
    f = path.open("a", buffering=1)

    def write(record: dict) -> None:
        json.dump(record, f)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    return write


async def measure_baseline(
    model: VLLMSteerModel,
    prompts: Sequence[str],
    sampling_params: SamplingParams,
    iterations: int,
    total_tokens: int,
) -> dict:
    gen_times: list[float] = []
    throughputs: list[float] = []

    for iteration in range(iterations):
        cuda_synchronize()
        start = time.perf_counter()
        await model.generate(prompts, sampling_params)
        cuda_synchronize()
        elapsed = time.perf_counter() - start
        gen_times.append(elapsed)
        throughputs.append(total_tokens / elapsed)
        logger.info("Baseline iteration %d: %.4fs", iteration + 1, elapsed)

    mean_time = statistics.fmean(gen_times)
    stdev_time = statistics.pstdev(gen_times) if len(gen_times) > 1 else 0.0
    mean_tp = statistics.fmean(throughputs)
    stdev_tp = statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    return {
        "mean_time_s": mean_time,
        "stdev_time_s": stdev_time,
        "mean_throughput_toks_per_s": mean_tp,
        "stdev_throughput_toks_per_s": stdev_tp,
        "iterations": gen_times,
    }


async def measure_capture(
    model: VLLMSteerModel,
    prompts: Sequence[str],
    sampling_params: SamplingParams,
    iterations: int,
    capture_layers: Sequence[int],
    total_tokens: int,
    baseline_mean: float,
    collect_profile: bool,
) -> dict:
    gen_times: list[float] = []
    fetch_times: list[float] = []
    data_sizes: list[int] = []
    iteration_details = []

    capture_arg: Iterable[int] | int | None
    if capture_layers:
        capture_arg = tuple(int(idx) for idx in capture_layers)
    else:
        capture_arg = tuple()

    for iteration in range(iterations):
        cuda_synchronize()
        start_gen = time.perf_counter()
        _, handles = await model.generate(
            prompts,
            sampling_params,
            capture_layers=capture_arg,
        )
        cuda_synchronize()
        gen_elapsed = time.perf_counter() - start_gen
        gen_times.append(gen_elapsed)

        cuda_synchronize()
        start_fetch = time.perf_counter()
        await model.fetch_captures_batch(handles)
        cuda_synchronize()
        fetch_elapsed = time.perf_counter() - start_fetch
        fetch_times.append(fetch_elapsed)

        data_bytes = compute_bytes(handles)
        data_sizes.append(data_bytes)

        iteration_details.append(
            {
                "iteration": iteration + 1,
                "generation_s": gen_elapsed,
                "fetch_s": fetch_elapsed,
                "data_mb": data_bytes / (1024 * 1024),
            }
        )

        # Drop references promptly
        for handle in handles:
            handle._captures = None
        del handles

        logger.info(
            "Capture iteration %d: gen=%.4fs fetch=%.4fs data=%.2fMB",
            iteration + 1,
            gen_elapsed,
            fetch_elapsed,
            data_bytes / (1024 * 1024),
        )

    mean_gen = statistics.fmean(gen_times)
    stdev_gen = statistics.pstdev(gen_times) if len(gen_times) > 1 else 0.0
    mean_fetch = statistics.fmean(fetch_times)
    stdev_fetch = statistics.pstdev(fetch_times) if len(fetch_times) > 1 else 0.0
    mean_data = statistics.fmean(data_sizes)
    stdev_data = statistics.pstdev(data_sizes) if len(data_sizes) > 1 else 0.0

    total_mean = mean_gen + mean_fetch
    overhead_pct = ((mean_gen / baseline_mean) - 1.0) * 100.0 if baseline_mean else None
    total_overhead_pct = (
        ((total_mean / baseline_mean) - 1.0) * 100.0 if baseline_mean else None
    )
    throughput = total_tokens / mean_gen if mean_gen else None

    profile_summaries = None
    if collect_profile:
        profile_summaries = await model.fetch_last_profiler_summaries()

    return {
        "mean_generation_s": mean_gen,
        "stdev_generation_s": stdev_gen,
        "mean_fetch_s": mean_fetch,
        "stdev_fetch_s": stdev_fetch,
        "mean_total_s": total_mean,
        "generation_overhead_pct": overhead_pct,
        "total_overhead_pct": total_overhead_pct,
        "mean_data_mb": mean_data / (1024 * 1024),
        "stdev_data_mb": stdev_data / (1024 * 1024),
        "mean_generation_throughput_toks_per_s": throughput,
        "iterations": iteration_details,
        "profile_summaries": profile_summaries,
    }


async def benchmark_model(cfg: BenchmarkConfig, model_name: str, writer) -> None:
    logger.info("==============================================")
    logger.info("Benchmarking model: %s", model_name)
    logger.info("==============================================")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=cfg.decode_tokens,
    )
    prompts = build_prompts(cfg.batch_size, cfg.prefill_tokens)
    total_tokens = cfg.batch_size * (cfg.prefill_tokens + cfg.decode_tokens)

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    total_layers = int(hf_config.num_hidden_layers)

    capture_sets = {
        "zero": [],
        "one": [total_layers // 2],
        "eight": evenly_spaced_layers(total_layers, 8),
        "all": list(range(total_layers)),
    }

    selected_buckets = [b for b in cfg.capture_buckets if b in capture_sets]
    if not selected_buckets:
        raise ValueError(f"No valid capture buckets selected from {cfg.capture_buckets}")

    steering_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=cfg.prefill_tokens + cfg.decode_tokens + 64,
        dtype="bfloat16",
    )

    model = VLLMSteerModel(steering_cfg, enforce_eager=True)

    try:
        logger.info("Warm-up pass (single prompt) to stabilize kernels.")
        cuda_synchronize()
        await model.generate(prompts[:1], sampling_params)
        cuda_synchronize()

        baseline = await measure_baseline(
            model,
            prompts,
            sampling_params,
            cfg.iterations,
            total_tokens,
        )
        writer(
            {
                "timestamp": time.time(),
                "model": model_name,
                "phase": "baseline",
                "batch_size": cfg.batch_size,
                "prefill_tokens": cfg.prefill_tokens,
                "decode_tokens": cfg.decode_tokens,
                "tensor_parallel_size": cfg.tensor_parallel_size,
                "iterations": cfg.iterations,
                "metrics": baseline,
            }
        )

        for label in selected_buckets:
            layers = capture_sets[label]
            collect_profile = (
                model_name == cfg.profile_trace_model and label == "all"
            )
            result = await measure_capture(
                model,
                prompts,
                sampling_params,
                cfg.iterations,
                layers,
                total_tokens,
                baseline["mean_time_s"],
                collect_profile=collect_profile,
            )

            record = {
                "timestamp": time.time(),
                "model": model_name,
                "phase": "capture",
                "capture_bucket": label,
                "layer_indices": layers,
                "layer_count": len(layers),
                "batch_size": cfg.batch_size,
                "prefill_tokens": cfg.prefill_tokens,
                "decode_tokens": cfg.decode_tokens,
                "tensor_parallel_size": cfg.tensor_parallel_size,
                "iterations": cfg.iterations,
                "baseline_mean_time_s": baseline["mean_time_s"],
                "metrics": result,
            }
            writer(record)

            if label == "all":
                logger.info(
                    "All-layer capture completed: total_mean=%.4fs overhead=%.2f%%",
                    result["mean_total_s"],
                    (result["total_overhead_pct"] or 0.0),
                )
    finally:
        engine = model.llm
        if engine is not None and hasattr(engine, "shutdown"):
            try:
                result = engine.shutdown()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.warning("Failed to shutdown engine cleanly", exc_info=True)
        del model
        await asyncio.sleep(0)
        cuda_synchronize()
        torch.cuda.empty_cache()


async def configure_profiler(output_dir: Path, enable: bool) -> None:
    steering_runtime._PROFILE_FETCH_ENABLED = enable
    steering_runtime._PROFILE_FETCH_TRACE_DIR = str(output_dir / "traces") if enable else None
    steering_runtime._PROFILE_FETCH_TRACE_PREFIX = "capture_fetch"
    steering_runtime._PROFILE_FETCH_EVENT_LIMIT = 128 if enable else 32
    steering_runtime._PROFILE_FETCH_TOPK = 15 if enable else 5


async def main_async(cfg: BenchmarkConfig) -> None:
    writer = make_record_writer(cfg.output_dir / "capture_scaling.jsonl")
    await configure_profiler(cfg.output_dir, enable=False)

    for model_name in cfg.models:
        enable_profile = model_name == cfg.profile_trace_model
        await configure_profiler(cfg.output_dir, enable=enable_profile)
        await benchmark_model(cfg, model_name, writer)
        await configure_profiler(cfg.output_dir, enable=False)


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_args(argv)
    configure_logging(cfg.output_dir)
    try:
        asyncio.run(main_async(cfg))
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user.")


if __name__ == "__main__":
    main()
