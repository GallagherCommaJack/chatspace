"""Compare steering outputs across model dtypes.

Runs a small prompt suite in multiple precision configurations to quantify
token-level divergence relative to a float32 reference run.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
)


os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


DEFAULT_PROMPTS: Sequence[str] = (
    "Explain why consciousness presents a challenge for purely computational theories.",
    "Summarize the main takeaways from the double-slit experiment for a high-school audience.",
    "Given a 7% annual return, how long will it take to double an investment?",
    "Outline the ethical considerations of using AI for medical diagnoses.",
    "Describe how photosynthesis converts light into chemical energy.",
    "Provide a step-by-step solution to 3x + 5 = 20.",
    "Write a short paragraph about the significance of the printing press.",
    "What are the advantages and disadvantages of remote work?",
    "Explain the Pauli exclusion principle in simple terms.",
    "How does increasing interest rates affect housing markets?",
)


@dataclass
class EvalConfig:
    label: str
    dtype: str
    cap_precision: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--cap", type=float, default=5.0)
    parser.add_argument("--vector-index", type=int, default=0)
    parser.add_argument("--vector-path", type=str, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Prompts to generate for evaluation.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace, dtype: str) -> VLLMSteerModel:
    cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=dtype,
    )
    model = VLLMSteerModel(cfg, bootstrap_layers=(args.layer,), enforce_eager=True)
    if args.vector_path:
        vector = torch.load(args.vector_path, map_location="cpu").to(torch.float32)
        if vector.shape[0] != model.hidden_size:
            raise ValueError(
                f"Vector length {vector.shape[0]} does not match hidden size {model.hidden_size}"
            )
    else:
        vector = torch.zeros(model.hidden_size)
        if args.vector_index < 0 or args.vector_index >= model.hidden_size:
            raise ValueError(
                f"vector index {args.vector_index} outside hidden size {model.hidden_size}"
            )
        vector[args.vector_index] = 1.0
    model.set_layer_projection_cap(args.layer, vector, max=args.cap)
    return model


def run_config(
    args: argparse.Namespace,
    cfg: EvalConfig,
) -> list[dict[str, object]]:
    model = build_model(args, dtype=cfg.dtype)
    try:
        if cfg.cap_precision is not None:
            model.set_projection_cap_precision(cfg.cap_precision)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            logprobs=0,
            prompt_logprobs=None,
        )
        results: list[dict[str, object]] = []
        for prompt in args.prompts:
            output = model.llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
            results.append(
                {
                    "prompt": prompt,
                    "text": output.text,
                    "token_ids": list(output.token_ids),
                }
            )
        return results
    finally:
        try:
            model.llm.shutdown()
        except Exception:
            pass
        del model
        torch.cuda.empty_cache()


def token_match_stats(
    reference: list[int], candidate: list[int]
) -> dict[str, float | int | None]:
    if not reference:
        return {"match_ratio": 1.0, "first_diff": None, "length_delta": len(candidate)}
    length = min(len(reference), len(candidate))
    matches = sum(1 for i in range(length) if reference[i] == candidate[i])
    first_diff: int | None = None
    for idx in range(length):
        if reference[idx] != candidate[idx]:
            first_diff = idx
            break
    return {
        "match_ratio": matches / max(len(reference), 1),
        "first_diff": first_diff,
        "length_delta": len(candidate) - len(reference),
    }


def main() -> None:
    args = parse_args()
    configs = [
        EvalConfig(label="bf16", dtype="auto", cap_precision=None),
        EvalConfig(label="bf16_cap_fp32", dtype="auto", cap_precision="float32"),
        EvalConfig(label="float16", dtype="float16", cap_precision="float32"),
        EvalConfig(label="float32", dtype="float32", cap_precision="float32"),
    ]

    all_results: dict[str, list[dict[str, object]]] = {}
    for cfg in configs:
        print(f"\n=== Running {cfg.label} ===")
        all_results[cfg.label] = run_config(args, cfg)
        for idx, result in enumerate(all_results[cfg.label]):
            text = result["text"]
            print(f"[{cfg.label}] prompt {idx}: {text!r}")

    reference = all_results["float32"]
    summary: dict[str, list[dict[str, float | int | None]]] = defaultdict(list)
    for cfg in configs:
        if cfg.label == "float32":
            continue
        for idx, cand in enumerate(all_results[cfg.label]):
            ref_tokens = reference[idx]["token_ids"]
            cand_tokens = cand["token_ids"]
            stats = token_match_stats(ref_tokens, cand_tokens)
            stats["prompt"] = idx
            summary[cfg.label].append(stats)

    print("\n=== Token Match Metrics (vs float32) ===")
    for label, entries in summary.items():
        ratios = [entry["match_ratio"] for entry in entries if entry["match_ratio"] is not None]
        avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        first_diff = next(
            (entry["first_diff"] for entry in entries if entry["first_diff"] is not None),
            None,
        )
        length_shift = sum(entry["length_delta"] for entry in entries if isinstance(entry["length_delta"], int))
        print(
            f"{label}: avg match {avg_ratio:.3f}, "
            f"first diff at token {first_diff}, "
            f"total length delta {length_shift}"
        )


if __name__ == "__main__":
    main()
