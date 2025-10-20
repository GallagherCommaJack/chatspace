"""Compare HuggingFace steering outputs under projection caps."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("/root/persona-subspace")
from utils.steering_utils import create_projection_cap_steerer  # type: ignore

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
class RunConfig:
    label: str
    dtype: torch.dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--cap", type=float, default=5.0)
    parser.add_argument("--vector-index", type=int, default=0)
    parser.add_argument("--vector-path", type=str, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional file containing prompts (one per line).",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Prompts to evaluate.",
    )
    return parser.parse_args()


def generate_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int,
) -> tuple[list[int], str]:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
        )[0]
    new_tokens = output_ids[encoded["input_ids"].shape[1] :].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return new_tokens, text


def main() -> None:
    args = parse_args()
    configs = [
        RunConfig("bf16", torch.bfloat16),
        RunConfig("fp32", torch.float32),
    ]

    prompts: list[str]
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            prompts = [line.strip() for line in fh if line.strip()]
        if not prompts:
            raise ValueError("Prompt file is empty.")
    else:
        prompts = list(args.prompts)

    results: dict[str, list[dict[str, object]]] = {}

    for cfg in configs:
        print(f"\n=== Running {cfg.label} ===")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=cfg.dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        hidden_size = model.config.hidden_size
        if args.vector_path:
            raw_vec = torch.load(args.vector_path, map_location="cpu").to(cfg.dtype)
            if raw_vec.shape[0] != hidden_size:
                raise ValueError(
                    f"Vector length {raw_vec.shape[0]} does not match hidden size {hidden_size}"
                )
            vector = raw_vec.to(device=model.device)
        else:
            vector = torch.zeros(hidden_size, dtype=cfg.dtype, device=model.device)
            vector[args.vector_index] = 1.0
        captures: list[dict[str, object]] = []

        with create_projection_cap_steerer(
            model,
            feature_directions=[vector],
            cap_thresholds=[args.cap],
            layer_indices=[args.layer],
            positions="all",
        ):
            for idx, prompt in enumerate(args.prompts):
                token_ids, text = generate_tokens(model, tokenizer, prompt, args.max_tokens)
                print(f"[{cfg.label}] prompt {idx}: {text!r}")
                captures.append(
                    {
                        "prompt": prompt,
                        "token_ids": token_ids,
                        "text": text,
                    }
                )
        results[cfg.label] = captures
        del model
        torch.cuda.empty_cache()

    if "fp32" in results:
        reference = results["fp32"]
        for label, entries in results.items():
            if label == "fp32":
                continue
            print(f"\n=== Token Match vs fp32 ({label}) ===")
            totals: list[float] = []
            for idx, entry in enumerate(entries):
                ref_tokens = reference[idx]["token_ids"]
                cand_tokens = entry["token_ids"]
                length = min(len(ref_tokens), len(cand_tokens))
                matches = sum(
                    1 for i in range(length) if ref_tokens[i] == cand_tokens[i]
                )
                first_diff = next(
                    (i for i in range(length) if ref_tokens[i] != cand_tokens[i]), None
                )
                ratio = matches / max(len(ref_tokens), 1)
                totals.append(ratio)
                print(
                    f"prompt {idx:02d}: match={ratio:.3f} "
                    f"first_diff={first_diff if first_diff is not None else 'None'} "
                    f"len_ref={len(ref_tokens)} len_cmp={len(cand_tokens)}"
                )
            if totals:
                avg_ratio = sum(totals) / len(totals)
                print(f"average match: {avg_ratio:.3f}")

    path = os.environ.get("HF_PRECISION_DUMP")
    if path:
        torch.save(results, path)


if __name__ == "__main__":
    main()
