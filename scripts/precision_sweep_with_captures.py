"""Precision sweep with full hidden state and cap_delta capture.

Runs vLLM steering across multiple dtypes (bf16, float16, float32), capturing
hidden states and projection cap deltas at each decode step. Useful for:

- Quantifying numerical drift in different precision regimes
- Correlating hidden state MAE with token divergence
- Debugging projection capping precision issues
- Establishing baseline parity metrics

Each run generates capture files containing:
- Generated token IDs and text
- Hidden states (before/after steering) per decode step
- cap_delta tensors showing projection adjustments
- Metadata: phase (prefill/decode), step index, sequence length
- Dtype diagnostics for each tensor operation

Example:
    python scripts/precision_sweep_with_captures.py \\
        --layer 12 \\
        --cap 5.0 \\
        --max-tokens 64 \\
        --run-id my_investigation \\
        --vector-path projection_vector.pt

See scripts/PRECISION_TESTING_README.md for full workflow.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class EvalConfig:
    label: str
    dtype: str
    cap_precision: str | None


@dataclass
class RunResult:
    config: EvalConfig
    prompt_idx: int
    prompt: str
    text: str
    token_ids: list[int]
    captures: list[dict[str, torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--cap", type=float, default=5.0)
    parser.add_argument("--vector-index", type=int, default=0)
    parser.add_argument("--vector-path", type=str, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Prompts to generate for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/cache/precision_sweeps",
        help="Directory to save capture results.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier for organizing outputs.",
    )
    return parser.parse_args()


def build_model(
    args: argparse.Namespace,
    dtype: str,
    *,
    enable_capture: bool = True,
) -> VLLMSteerModel:
    cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=dtype,
    )
    model = VLLMSteerModel(cfg, bootstrap_layers=(args.layer,), enforce_eager=True)

    # Load or create projection vector
    if args.vector_path:
        vector = torch.load(args.vector_path, map_location="cpu").to(torch.float32)
        if vector.shape[0] != model.hidden_size:
            raise ValueError(
                f"Vector length {vector.shape[0]} does not match hidden size {model.hidden_size}"
            )
    else:
        vector = torch.zeros(model.hidden_size, dtype=torch.float32)
        if args.vector_index < 0 or args.vector_index >= model.hidden_size:
            raise ValueError(
                f"vector index {args.vector_index} outside hidden size {model.hidden_size}"
            )
        vector[args.vector_index] = 1.0

    model.set_layer_projection_cap(args.layer, vector, max=args.cap)

    if enable_capture:
        model.enable_hidden_state_capture(
            args.layer,
            capture_before=True,
            capture_after=True,
            max_captures=None,
        )

    return model


def run_single_prompt(
    model: VLLMSteerModel,
    args: argparse.Namespace,
    cfg: EvalConfig,
    prompt_idx: int,
    prompt: str,
) -> RunResult:
    """Run a single prompt and collect captures."""
    model.clear_hidden_states(args.layer)

    if cfg.cap_precision is not None:
        model.set_projection_cap_precision(cfg.cap_precision)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        logprobs=0,
        prompt_logprobs=None,
    )

    output = model.llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]

    # Fetch captures
    all_captures = model.fetch_hidden_states(layer_idx=args.layer)
    captures = all_captures[0][args.layer] if all_captures else []

    return RunResult(
        config=cfg,
        prompt_idx=prompt_idx,
        prompt=prompt,
        text=output.text,
        token_ids=list(output.token_ids),
        captures=captures,
    )


def run_config(
    args: argparse.Namespace,
    cfg: EvalConfig,
) -> list[RunResult]:
    """Run all prompts for a single config."""
    model = build_model(args, dtype=cfg.dtype, enable_capture=True)

    try:
        results: list[RunResult] = []
        for idx, prompt in enumerate(args.prompts):
            result = run_single_prompt(model, args, cfg, idx, prompt)
            results.append(result)
            print(
                f"[{cfg.label}] prompt {idx}: "
                f"tokens={len(result.token_ids)} captures={len(result.captures)}"
            )
        return results
    finally:
        try:
            model.llm.shutdown()
        except Exception:
            pass
        del model
        torch.cuda.empty_cache()


def save_results(
    results: dict[str, list[RunResult]],
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
            prompt_file = config_dir / f"prompt_{result.prompt_idx:02d}.pt"

            # Extract and save relevant data
            save_data = {
                "prompt": result.prompt,
                "text": result.text,
                "token_ids": result.token_ids,
                "config": {
                    "label": result.config.label,
                    "dtype": result.config.dtype,
                    "cap_precision": result.config.cap_precision,
                },
                "captures": [],
            }

            # Process captures
            for cap_idx, capture in enumerate(result.captures):
                cap_entry = {}

                # Extract tensors
                for key in ("before", "after", "cap_delta"):
                    if key in capture and isinstance(capture[key], torch.Tensor):
                        cap_entry[key] = capture[key].cpu()

                # Extract metadata
                if "meta" in capture:
                    cap_entry["meta"] = dict(capture["meta"])

                if "cap_meta" in capture:
                    cap_entry["cap_meta"] = dict(capture["cap_meta"])

                save_data["captures"].append(cap_entry)

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


def print_summary(results: dict[str, list[RunResult]]) -> None:
    """Print token match summary."""
    if "float32" not in results:
        print("\nNo float32 reference available for comparison.")
        return

    reference = results["float32"]
    summary: dict[str, list[dict[str, float | int | None]]] = defaultdict(list)

    for label, run_results in results.items():
        if label == "float32":
            continue

        for idx, result in enumerate(run_results):
            ref_tokens = reference[idx].token_ids
            cand_tokens = result.token_ids
            stats = token_match_stats(ref_tokens, cand_tokens)
            stats["prompt"] = idx
            summary[label].append(stats)

    print("\n=== Token Match Metrics (vs float32) ===")
    for label, entries in summary.items():
        ratios = [e["match_ratio"] for e in entries if e["match_ratio"] is not None]
        avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        first_diffs = [e["first_diff"] for e in entries if e["first_diff"] is not None]
        avg_first_diff = sum(first_diffs) / len(first_diffs) if first_diffs else None
        length_shift = sum(e["length_delta"] for e in entries if isinstance(e["length_delta"], int))

        print(
            f"{label:20s}: avg match {avg_ratio:.3f}, "
            f"avg first diff {avg_first_diff if avg_first_diff is not None else 'N/A':>6}, "
            f"total length delta {length_shift:+3d}"
        )


def main() -> None:
    args = parse_args()

    configs = [
        EvalConfig(label="bf16", dtype="auto", cap_precision=None),
        EvalConfig(label="bf16_cap_fp32", dtype="auto", cap_precision="float32"),
        EvalConfig(label="float16", dtype="float16", cap_precision=None),
        EvalConfig(label="float32", dtype="float32", cap_precision=None),
    ]

    all_results: dict[str, list[RunResult]] = {}

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"Running config: {cfg.label}")
        print(f"  dtype={cfg.dtype}, cap_precision={cfg.cap_precision}")
        print(f"{'=' * 60}")

        all_results[cfg.label] = run_config(args, cfg)

    print_summary(all_results)
    save_results(all_results, args)


if __name__ == "__main__":
    main()
