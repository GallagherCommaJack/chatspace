"""Compare vLLM hidden-state deviations against an fp32 reference."""

from __future__ import annotations

import argparse
import os
from statistics import mean
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--cap", type=float, default=5.0)
    parser.add_argument("--vector-index", type=int, default=0)
    parser.add_argument("--vector-path", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Prompts to evaluate.",
    )
    parser.add_argument(
        "--cap-precision",
        default=None,
        help="Optional projection cap working precision (e.g. float32).",
    )
    return parser.parse_args()


def load_vector(args: argparse.Namespace, hidden_size: int) -> torch.Tensor:
    if args.vector_path:
        vec = torch.load(args.vector_path, map_location="cpu").to(torch.float32)
        if vec.shape[0] != hidden_size:
            raise ValueError(
                f"Vector length {vec.shape[0]} does not match hidden size {hidden_size}"
            )
    else:
        vec = torch.zeros(hidden_size, dtype=torch.float32)
        vec[args.vector_index] = 1.0
    return vec


def capture_hidden(
    model: VLLMSteerModel,
    prompts: Sequence[str],
    *,
    layer_idx: int,
    max_tokens: int,
) -> list[torch.Tensor]:
    hidden_states: list[torch.Tensor] = []
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for prompt in prompts:
        model.clear_hidden_states(layer_idx)
        model.generate(prompt, sampling_params=params)
        states = model.fetch_hidden_states(layer_idx=layer_idx)
        worker_states = states[0][layer_idx]
        if not worker_states:
            hidden_states.append(torch.empty(0))
            continue
        hidden_states.append(worker_states[-1]["after"].cpu().to(torch.float32))
    return hidden_states


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    configs = [
        ("fp32", "float32"),
        ("bf16", "bfloat16"),
        ("fp16", "float16"),
    ]

    reference_hidden: list[torch.Tensor] | None = None
    results: dict[str, list[torch.Tensor]] = {}
    tokens_reference: list[list[int]] | None = None

    for label, dtype in configs:
        cfg = VLLMSteeringConfig(
            model_name=args.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=dtype,
        )
        model = VLLMSteerModel(cfg, bootstrap_layers=(args.layer,), enforce_eager=True)

        if args.cap_precision is not None:
            model.set_projection_cap_precision(args.cap_precision)
        else:
            model.set_projection_cap_precision(None)

        vector = load_vector(args, model.hidden_size)
        model.set_layer_projection_cap(args.layer, vector, max=args.cap)
        model.enable_hidden_state_capture(
            args.layer, capture_before=False, capture_after=True, max_captures=None
        )

        hidden = capture_hidden(
            model,
            args.prompts,
            layer_idx=args.layer,
            max_tokens=args.max_tokens,
        )
        results[label] = hidden

        if label == "fp32":
            reference_hidden = hidden
            # Also store generated tokens for baseline comparison
            tokens_reference = []
            params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
            for prompt in args.prompts:
                output = model.llm.generate([prompt], params, use_tqdm=False)[0]
                tokens_reference.append(output.outputs[0].token_ids)

        model.clear_all_vectors()
        del model
        torch.cuda.empty_cache()

    if reference_hidden is None:
        raise RuntimeError("fp32 baseline capture failed.")

    for label in ("bf16", "fp16"):
        if label not in results:
            continue
        maes: list[float] = []
        cosines: list[float] = []
        for idx, (bf, ref) in enumerate(zip(results[label], reference_hidden)):
            if bf.numel() == 0 or ref.numel() == 0:
                continue
            diff = (bf - ref).abs()
            mae = diff.mean().item()
            cos = torch.nn.functional.cosine_similarity(
                bf.flatten().unsqueeze(0),
                ref.flatten().unsqueeze(0),
                dim=-1,
            ).item()
            maes.append(mae)
            cosines.append(cos)
            print(
                f"{label} prompt {idx:02d}: hidden MAE={mae:.6f} cos={cos:.9f} "
                f"shape={tuple(ref.shape)}"
            )

        if maes:
            print(
                f"{label} summary: avg MAE={mean(maes):.6f}, avg cos={mean(cosines):.9f}"
            )


if __name__ == "__main__":
    main()

