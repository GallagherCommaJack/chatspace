#!/usr/bin/env python3
"""
High-batch benchmark focused on stress-testing capture at high concurrency.
Run on GPU 1 while full benchmark runs on GPU 0.
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("="*80)
    logger.info("High-Batch vLLM Capture Benchmark (GPU 1)")
    logger.info("="*80)

    # Initialize model
    logger.info("Initializing VLLMSteerModel...")
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen2.5-3B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
    )

    model = VLLMSteerModel(cfg, enforce_eager=True)

    # Get layer count
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    total_layers = hf_config.num_hidden_layers
    logger.info(f"Model has {total_layers} layers")

    # Test configurations - focus on high batch and concurrency
    batch_sizes = [32, 64, 128]
    prefill_lengths = [256, 512]
    decode_lengths = [50, 100]
    layer_configs = [
        ([], "none"),
        ([total_layers // 2], "single"),
        (list(range(total_layers)), "all"),
    ]
    concurrency_levels = [1, 2, 4, 8]

    results = []
    test_num = 0
    total_tests = (len(batch_sizes) * len(prefill_lengths) * len(decode_lengths) *
                   len(layer_configs) * len(concurrency_levels))

    logger.info(f"Running {total_tests} high-batch test configurations...")

    for batch_size in batch_sizes:
        for prefill_len in prefill_lengths:
            for decode_len in decode_lengths:
                for layers, layer_desc in layer_configs:
                    for concurrency in concurrency_levels:
                        test_num += 1

                        # Generate prompts
                        prompts = [
                            f"Test prompt {i} " * (prefill_len // 20)
                            for i in range(batch_size)
                        ]

                        sampling_params = SamplingParams(
                            temperature=0.0,
                            max_tokens=decode_len,
                        )

                        logger.info(
                            f"[{test_num}/{total_tests}] "
                            f"batch={batch_size:3d} | layers={len(layers):2d} ({layer_desc:12s}) | "
                            f"prefill={prefill_len:4d} | decode={decode_len:3d} | concurrent={concurrency}"
                        )

                        # Run 3 iterations
                        times = []
                        for _ in range(3):
                            torch.cuda.reset_peak_memory_stats()

                            # Baseline
                            start = time.perf_counter()
                            if concurrency > 1:
                                tasks = [
                                    model.generate(prompts, sampling_params)
                                    for _ in range(concurrency)
                                ]
                                await asyncio.gather(*tasks)
                            else:
                                await model.generate(prompts, sampling_params)
                            baseline_time = time.perf_counter() - start

                            # With capture
                            if layers:
                                start = time.perf_counter()
                                if concurrency > 1:
                                    tasks = [
                                        model.generate(prompts, sampling_params, capture_layers=layers)
                                        for _ in range(concurrency)
                                    ]
                                    results_list = await asyncio.gather(*tasks)
                                    all_handles = []
                                    for texts, handles in results_list:
                                        all_handles.extend(handles)
                                else:
                                    _, handles = await model.generate(
                                        prompts, sampling_params, capture_layers=layers
                                    )
                                    all_handles = handles

                                capture_time = time.perf_counter() - start

                                # Fetch
                                start = time.perf_counter()
                                await model.fetch_captures_batch(all_handles)
                                fetch_time = time.perf_counter() - start

                                total_time = capture_time + fetch_time
                            else:
                                total_time = baseline_time
                                fetch_time = 0.0

                            overhead = ((total_time / baseline_time) - 1.0) * 100.0 if baseline_time > 0 else 0.0
                            times.append(overhead)

                            peak_mem = torch.cuda.max_memory_allocated() / 1024**3

                        median_overhead = statistics.median(times)
                        total_tokens = batch_size * concurrency * (prefill_len + decode_len)
                        throughput = total_tokens / (baseline_time + median_overhead * baseline_time / 100.0)

                        logger.info(
                            f"  → Overhead: {median_overhead:6.2f}% | "
                            f"Throughput: {throughput:8.1f} tok/s | "
                            f"Memory: {peak_mem:.2f} GB"
                        )

                        results.append({
                            "batch_size": batch_size,
                            "num_layers": len(layers),
                            "layer_desc": layer_desc,
                            "prefill_len": prefill_len,
                            "decode_len": decode_len,
                            "concurrency": concurrency,
                            "overhead_pct": median_overhead,
                            "throughput_tok_s": throughput,
                            "peak_memory_gb": peak_mem,
                        })

    # Save results
    output_dir = Path("/workspace/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat()
    results_path = output_dir / f"high_batch_results_{timestamp.replace(':', '-')}.json"

    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "model_name": cfg.model_name,
                "total_layers": total_layers,
                "test_type": "high_batch",
            },
            "results": results,
        }, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    # Print summary
    sorted_results = sorted(results, key=lambda r: r["overhead_pct"], reverse=True)[:10]

    logger.info("\n" + "="*80)
    logger.info("TOP 10 HIGHEST OVERHEAD CONFIGURATIONS")
    logger.info("="*80)

    for i, r in enumerate(sorted_results, 1):
        logger.info(
            f"{i:2d}. batch={r['batch_size']:3d} | layers={r['num_layers']:2d} ({r['layer_desc']:12s}) | "
            f"prefill={r['prefill_len']:4d} | decode={r['decode_len']:3d} | concurrent={r['concurrency']}"
        )
        logger.info(
            f"    Overhead: {r['overhead_pct']:6.2f}% | Throughput: {r['throughput_tok_s']:8.1f} tok/s"
        )

    logger.info("\n" + "="*80)
    logger.info("High-batch benchmark complete!")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
