"""HuggingFace/persona precision sweep with hidden state capture.

Runs persona steering with projection caps across dtypes, capturing hidden
states for comparison against vLLM.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("/root/persona-subspace")
from utils.steering_utils import create_projection_cap_steerer  # type: ignore


DEFAULT_PROMPTS: Sequence[str] = (
    "Explain why consciousness presents a challenge for purely computational theories.",
    "Summarize the main takeaways from the double-slit experiment for a high-school audience.",
    "Given a 7% annual return, how long will it take to double an investment?",
)


@dataclass
class RunConfig:
    label: str
    dtype: torch.dtype


@dataclass
class CaptureHook:
    """Hook to capture hidden states during generation."""

    layer_idx: int
    captures: list[dict[str, torch.Tensor]]

    def __call__(
        self, module: torch.nn.Module, input: tuple, output: tuple
    ) -> None:
        """Capture hidden state from layer output."""
        hidden = output[0]  # First element is hidden states
        if hidden is None:
            return

        # Determine phase based on sequence length
        seq_len = hidden.shape[1]
        phase = "prefill" if seq_len > 1 else "decode"

        # Count decode steps
        decode_count = sum(1 for c in self.captures if c.get("meta", {}).get("phase") == "decode")
        step = decode_count if phase == "decode" else 0

        capture_entry = {
            "hidden": hidden.detach().cpu(),
            "meta": {
                "phase": phase,
                "step": step,
                "seq_len": seq_len,
            },
        }

        self.captures.append(capture_entry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--cap", type=float, default=5.0)
    parser.add_argument("--vector-index", type=int, default=0)
    parser.add_argument("--vector-path", type=str, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Prompts to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/cache/hf_precision_sweeps",
        help="Directory to save capture results.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier.",
    )
    return parser.parse_args()


def generate_with_capture(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int,
    layer_idx: int,
) -> tuple[list[int], str, list[dict[str, torch.Tensor]]]:
    """Generate tokens and capture hidden states."""
    # Create capture hook
    hook = CaptureHook(layer_idx=layer_idx, captures=[])

    # Register hook on target layer
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)

    try:
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

        return new_tokens, text, hook.captures

    finally:
        handle.remove()


def run_config(
    args: argparse.Namespace,
    cfg: RunConfig,
) -> list[dict[str, object]]:
    """Run all prompts for a single config."""
    print(f"\n{'=' * 60}")
    print(f"Running config: {cfg.label}")
    print(f"  dtype={cfg.dtype}")
    print(f"{'=' * 60}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=cfg.dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hidden_size = model.config.hidden_size

    # Load or create projection vector
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

    results: list[dict[str, object]] = []

    with create_projection_cap_steerer(
        model,
        feature_directions=[vector],
        cap_thresholds=[args.cap],
        layer_indices=[args.layer],
        positions="all",
    ):
        for idx, prompt in enumerate(args.prompts):
            token_ids, text, captures = generate_with_capture(
                model, tokenizer, prompt, args.max_tokens, args.layer
            )

            print(
                f"[{cfg.label}] prompt {idx}: "
                f"tokens={len(token_ids)} captures={len(captures)}"
            )

            results.append(
                {
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "token_ids": token_ids,
                    "text": text,
                    "captures": captures,
                }
            )

    del model
    torch.cuda.empty_cache()

    return results


def save_results(
    results: dict[str, list[dict[str, object]]],
    args: argparse.Namespace,
) -> Path:
    """Save all results to disk."""
    output_dir = Path(args.output_dir)
    if args.run_id:
        output_dir = output_dir / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each config separately
    for label, run_results in results.items():
        config_dir = output_dir / label
        config_dir.mkdir(exist_ok=True)

        for result in run_results:
            prompt_idx = result["prompt_idx"]
            prompt_file = config_dir / f"prompt_{prompt_idx:02d}.pt"

            save_data = {
                "prompt": result["prompt"],
                "text": result["text"],
                "token_ids": result["token_ids"],
                "config": {
                    "label": label,
                    "dtype": str(results),  # Will be parsed from label
                },
                "captures": result["captures"],
            }

            torch.save(save_data, prompt_file)

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Cap threshold: {args.cap}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Max tokens: {args.max_tokens}\n")
        f.write(f"Prompts: {len(args.prompts)}\n")
        f.write(f"\nConfigs run:\n")
        for label in results.keys():
            f.write(f"  - {label}\n")

    print(f"\nResults saved to: {output_dir}")
    return output_dir


def token_match_stats(
    reference: list[int], candidate: list[int]
) -> dict[str, float | int | None]:
    """Compute token match statistics."""
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


def print_summary(results: dict[str, list[dict[str, object]]]) -> None:
    """Print token match summary."""
    if "fp32" not in results:
        print("\nNo fp32 reference available for comparison.")
        return

    reference = results["fp32"]

    print("\n=== Token Match Metrics (vs fp32) ===")
    for label, run_results in results.items():
        if label == "fp32":
            continue

        ratios = []
        first_diffs = []
        length_deltas = []

        for idx, result in enumerate(run_results):
            ref_tokens = reference[idx]["token_ids"]
            cand_tokens = result["token_ids"]
            stats = token_match_stats(ref_tokens, cand_tokens)

            ratios.append(stats["match_ratio"])
            if stats["first_diff"] is not None:
                first_diffs.append(stats["first_diff"])
            length_deltas.append(stats["length_delta"])

        avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        avg_first_diff = sum(first_diffs) / len(first_diffs) if first_diffs else None
        total_length_delta = sum(length_deltas)

        print(
            f"{label:20s}: avg match {avg_ratio:.3f}, "
            f"avg first diff {avg_first_diff if avg_first_diff is not None else 'N/A':>6}, "
            f"total length delta {total_length_delta:+3d}"
        )


def main() -> None:
    args = parse_args()

    configs = [
        RunConfig("bf16", torch.bfloat16),
        RunConfig("fp32", torch.float32),
    ]

    all_results: dict[str, list[dict[str, object]]] = {}

    for cfg in configs:
        all_results[cfg.label] = run_config(args, cfg)

    print_summary(all_results)
    save_results(all_results, args)


if __name__ == "__main__":
    main()
