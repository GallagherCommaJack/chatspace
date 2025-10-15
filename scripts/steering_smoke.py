"""Quick smoke test to confirm steering vectors affect vLLM outputs.

This script runs a baseline decode, applies a large random steering vector to
demonstrate behavior change, then clears the vector to restore the baseline.
"""

from __future__ import annotations

import argparse
import torch

from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
)
from vllm import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Question: What is the capital of France? Answer:")
    parser.add_argument("--scale", type=float, default=5000.0)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--init-scale", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        target_layer=args.layer,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        init_scale=args.init_scale,
    )

    vllm_kwargs = {}
    if args.enforce_eager:
        vllm_kwargs["enforce_eager"] = True

    model = VLLMSteerModel(cfg, **vllm_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    baseline = model.generate(args.prompt, params)[0]
    print("=== Baseline ===")
    print(repr(baseline))

    perturb = torch.randn(model.hidden_size) * args.scale
    model.set_vector(perturb)
    worker_vec = model._fetch_worker_vectors()[0]
    print(f"Applied vector norm: {worker_vec.float().norm().item():.2f}")
    steered = model.generate(args.prompt, params)[0]
    post_worker_vec = model._fetch_worker_vectors()[0]
    print(f"Vector norm after generate: {post_worker_vec.float().norm().item():.2f}")
    print("\n=== Steered ===")
    print(repr(steered))

    model.set_vector(torch.zeros(model.hidden_size))
    restored = model.generate(args.prompt, params)[0]
    print("\n=== Restored ===")
    print(repr(restored))


if __name__ == "__main__":
    main()
