"""Measure hidden-state deviation between HF fp32 and bf16 projection capping."""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.append("/root/persona-subspace")


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
    parser.add_argument("--prompts", nargs="*", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=0)
    return parser.parse_args()


def load_vector(args: argparse.Namespace, hidden_size: int, device: torch.device) -> torch.Tensor:
    if args.vector_path:
        vec = torch.load(args.vector_path, map_location="cpu").to(torch.float32)
        if vec.shape[0] != hidden_size:
            raise ValueError(
                f"Vector length {vec.shape[0]} does not match hidden size {hidden_size}"
            )
    else:
        vec = torch.zeros(hidden_size, dtype=torch.float32)
        vec[args.vector_index] = 1.0
    return vec.to(device=device)


def capture_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    *,
    layer_idx: int,
    cap_vector: torch.Tensor,
    cap_value: float,
    max_tokens: int,
) -> tuple[list[list[int]], list[torch.Tensor]]:
    from utils.steering_utils import create_projection_cap_steerer  # type: ignore

    all_tokens: list[list[int]] = []
    all_hiddens: list[torch.Tensor] = []

    device = next(model.parameters()).device
    vector = cap_vector.to(device=device, dtype=next(model.parameters()).dtype)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        captures: list[torch.Tensor] = []

        with create_projection_cap_steerer(
            model,
            feature_directions=[vector],
            cap_thresholds=[cap_value],
            layer_indices=[layer_idx],
        ):
            layer_module = model.model.layers[layer_idx]

            def hook(_, __, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captures.append(hidden.detach().cpu().to(torch.float32))
                return output

            handle = layer_module.register_forward_hook(hook)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.0,
                )[0]
            handle.remove()

        prompt_length = inputs["input_ids"].shape[1]
        tokens = output_ids[prompt_length:].tolist()
        all_tokens.append(tokens)
        if captures:
            all_hiddens.append(captures[-1])
        else:
            all_hiddens.append(torch.empty(0))

    return all_tokens, all_hiddens


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    configs = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    tokens_by_dtype: dict[str, list[list[int]]] = {}
    hiddens_by_dtype: dict[str, list[torch.Tensor]] = {}

    for label, dtype in configs.items():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        hidden_size = model.config.hidden_size
        vector = load_vector(args, hidden_size, next(model.parameters()).device)
        tokens, hiddens = capture_hidden_states(
            model,
            tokenizer,
            args.prompts,
            layer_idx=args.layer,
            cap_vector=vector,
            cap_value=args.cap,
            max_tokens=args.max_tokens,
        )
        tokens_by_dtype[label] = tokens
        hiddens_by_dtype[label] = hiddens
        del model
        torch.cuda.empty_cache()

    ref_tokens = tokens_by_dtype["fp32"]
    bf_tokens = tokens_by_dtype["bf16"]
    ratios: list[float] = []
    first_diffs: list[int] = []
    for idx, (ref, cand) in enumerate(zip(ref_tokens, bf_tokens)):
        length = min(len(ref), len(cand))
        matches = sum(1 for i in range(length) if ref[i] == cand[i])
        first_diff = next((i for i in range(length) if ref[i] != cand[i]), length)
        ratios.append(matches / max(len(ref), 1))
        first_diffs.append(first_diff)
        print(
            f"prompt {idx:02d}: token_match={matches/ max(len(ref),1):.3f} "
            f"first_diff={first_diff}"
        )
    if ratios:
        print(f"avg token match: {mean(ratios):.3f}")
        print(f"avg first diff: {mean(first_diffs):.1f}")

    bf_hiddens = hiddens_by_dtype["bf16"]
    fp_hiddens = hiddens_by_dtype["fp32"]
    maes: list[float] = []
    for idx, (bf, fp) in enumerate(zip(bf_hiddens, fp_hiddens)):
        if bf.numel() == 0 or fp.numel() == 0:
            continue
        diff = (bf - fp).abs()
        mae = diff.mean().item()
        maes.append(mae)
        print(f"prompt {idx:02d}: hidden MAE={mae:.4f}")
    if maes:
        print(f"avg hidden MAE: {mean(maes):.4f}")


if __name__ == "__main__":
    main()
