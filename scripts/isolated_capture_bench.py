#!/usr/bin/env python3
"""
Isolated capture benchmark - runs single config in fresh Python process.

Usage:
    # Baseline (no capture)
    uv run python scripts/isolated_capture_bench.py --mode baseline --output /workspace/results/baseline.json

    # Zero-layer capture (metadata overhead only)
    uv run python scripts/isolated_capture_bench.py --mode zero-layer --output /workspace/results/zero.json

    # All-layer capture
    uv run python scripts/isolated_capture_bench.py --mode all-layers --output /workspace/results/all.json

    # Specific layers
    uv run python scripts/isolated_capture_bench.py --mode custom --layers 0,5,10,15 --output /workspace/results/custom.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["baseline", "zero-layer", "all-layers", "custom"],
                        help="Benchmark mode")
    parser.add_argument("--layers", type=str, help="Comma-separated layer indices for custom mode (e.g. '0,5,10')")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Model name")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--prefill", type=int, default=512, help="Prefill tokens")
    parser.add_argument("--decode", type=int, default=128, help="Decode tokens")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    return parser.parse_args()


async def run_benchmark(args):
    """Run single isolated benchmark."""

    # Initialize model
    print(f"Initializing model: {args.model}")
    cfg = VLLMSteeringConfig(
        model_name=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        dtype="bfloat16",
    )
    model = VLLMSteerModel(cfg, enforce_eager=True)

    # Get layer count
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    total_layers = hf_config.num_hidden_layers

    # Determine capture layers
    if args.mode == "baseline":
        capture_layers = []
    elif args.mode == "zero-layer":
        capture_layers = []  # Will call generate with capture_layers=[] to trigger machinery
    elif args.mode == "all-layers":
        capture_layers = list(range(total_layers))
    elif args.mode == "custom":
        if not args.layers:
            raise ValueError("--layers required for custom mode")
        capture_layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Generate prompts
    prompts = [
        f"This is test prompt number {i}. " * (args.prefill // 20)
        for i in range(args.batch)
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode,
    )

    # Warmup
    print("Warming up...")
    await model.generate(prompts[:2], sampling_params, capture_layers=[])

    # Run benchmark
    print(f"\nRunning {args.iterations} iterations for mode={args.mode}")
    results = {
        "mode": args.mode,
        "model": args.model,
        "batch_size": args.batch,
        "prefill_tokens": args.prefill,
        "decode_tokens": args.decode,
        "total_layers": total_layers,
        "capture_layers": capture_layers,
        "num_captured_layers": len(capture_layers),
        "iterations": [],
    }

    for i in range(args.iterations):
        print(f"  Iteration {i+1}/{args.iterations}")

        torch.cuda.synchronize()
        gen_start = time.perf_counter()

        if args.mode == "baseline":
            # Pure baseline - no capture machinery
            texts = await model.generate(prompts, sampling_params)
            handles = None
        else:
            # Trigger capture machinery (even if zero layers)
            texts, handles = await model.generate(prompts, sampling_params, capture_layers=capture_layers)

        torch.cuda.synchronize()
        gen_time = time.perf_counter() - gen_start

        # Fetch if we have handles
        fetch_time = 0.0
        if handles:
            torch.cuda.synchronize()
            fetch_start = time.perf_counter()

            await model.fetch_captures_batch(handles)

            torch.cuda.synchronize()
            fetch_time = time.perf_counter() - fetch_start

            # Cleanup
            for handle in handles:
                try:
                    await model._collective_rpc("unregister_capture_request", handle.request_id)
                except:
                    pass

        total_tokens = args.batch * (args.prefill + args.decode)
        throughput = total_tokens / gen_time

        iteration_result = {
            "iteration": i + 1,
            "generation_time": gen_time,
            "fetch_time": fetch_time,
            "total_time": gen_time + fetch_time,
            "throughput_tokens_per_sec": throughput,
        }
        results["iterations"].append(iteration_result)

        print(f"    Generation: {gen_time:.4f}s, Fetch: {fetch_time:.4f}s, Total: {gen_time + fetch_time:.4f}s")

    # Compute stats
    gen_times = [it["generation_time"] for it in results["iterations"]]
    fetch_times = [it["fetch_time"] for it in results["iterations"]]
    total_times = [it["total_time"] for it in results["iterations"]]

    results["summary"] = {
        "mean_generation_time": sum(gen_times) / len(gen_times),
        "mean_fetch_time": sum(fetch_times) / len(fetch_times),
        "mean_total_time": sum(total_times) / len(total_times),
        "min_generation_time": min(gen_times),
        "max_generation_time": max(gen_times),
    }

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Mean generation time: {results['summary']['mean_generation_time']:.4f}s")
    print(f"Mean fetch time: {results['summary']['mean_fetch_time']:.4f}s")
    print(f"Mean total time: {results['summary']['mean_total_time']:.4f}s")

    return results


def main():
    args = parse_args()
    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
