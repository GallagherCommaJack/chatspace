"""Collect decode-time hidden state diagnostics for projection caps.

This utility runs a prompt twice: once with the default projection cap
precision and once with a higher-precision override. Hidden states are
captured before/after the cap, and decode-time logprobs are summarised to
highlight any divergence across precisions.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
)


os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@dataclass
class RunResult:
    label: str
    text: str
    logprobs: list[dict[int, "LogprobLike"]]
    captures: list[dict[str, torch.Tensor]]


LogprobLike = Sequence  # protocol placeholder for vLLM Logprob objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default=(
            "Explain why consciousness presents a challenge for purely "
            "computational theories."
        ),
    )
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--cap", type=float, default=5.0)
    parser.add_argument("--vector-index", type=int, default=0)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument(
        "--compare-precision",
        default="float32",
        help="Floating dtype name to apply for the second run (e.g. float32). "
        "Use 'none' to skip the comparison run.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> VLLMSteerModel:
    cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    model = VLLMSteerModel(cfg, bootstrap_layers=(args.layer,), enforce_eager=True)
    basis = torch.zeros(model.hidden_size)
    if args.vector_index < 0 or args.vector_index >= model.hidden_size:
        raise ValueError(f"vector index {args.vector_index} outside hidden size {model.hidden_size}")
    basis[args.vector_index] = 1.0
    model.set_layer_projection_cap(args.layer, basis, max=args.cap)
    model.enable_hidden_state_capture(
        args.layer,
        capture_before=True,
        capture_after=True,
        max_captures=None,
    )
    return model


def run_once(
    model: VLLMSteerModel,
    args: argparse.Namespace,
    *,
    label: str,
    precision: str | None,
) -> RunResult:
    model.clear_hidden_states(args.layer)
    model.set_projection_cap_precision(None if precision is None else precision)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=True,
    )
    outputs = model.llm.generate([args.prompt], params, use_tqdm=False)
    text = outputs[0].outputs[0].text
    logprobs = outputs[0].outputs[0].logprobs or []
    captures = model.fetch_hidden_states(layer_idx=args.layer)[0][args.layer]
    return RunResult(label=label, text=text, logprobs=logprobs, captures=captures)


def classify_capture(entry: dict[str, torch.Tensor]) -> str:
    tensor = entry.get("after")
    if tensor is None:
        tensor = entry.get("before")
    if tensor is None:
        return "unknown"
    return "decode" if tensor.shape[1] == 1 else "prefill"


def summarise_diff(
    baseline: list[dict[str, torch.Tensor]],
    compare: list[dict[str, torch.Tensor]],
) -> list[dict[str, float | int | str]]:
    if len(baseline) != len(compare):
        raise RuntimeError(
            f"Capture length mismatch: baseline={len(baseline)} compare={len(compare)}"
        )
    summary: list[dict[str, float | int | str]] = []
    for idx, (base, comp) in enumerate(zip(baseline, compare)):
        phase = classify_capture(base)
        entry: dict[str, float | int | str] = {"capture": idx, "phase": phase}
        for key in ("before", "after"):
            base_tensor = base.get(key)
            comp_tensor = comp.get(key)
            if base_tensor is None or comp_tensor is None:
                continue
            diff = comp_tensor.to(torch.float32) - base_tensor.to(torch.float32)
            entry[f"{key}_max_abs"] = float(diff.abs().max().item())
            entry[f"{key}_mean_abs"] = float(diff.abs().mean().item())
            entry[f"{key}_rmse"] = float(diff.pow(2).mean().sqrt().item())
        summary.append(entry)
    return summary


def extract_decode_logprobs(
    run: RunResult, *, max_tokens: int
) -> list[list[tuple[str, float]]]:
    if not run.logprobs:
        return []
    decode_entries = run.logprobs[-max_tokens:]
    decode_tokens: list[list[tuple[str, float]]] = []
    for entry in decode_entries:
        ranked = sorted(entry.items(), key=lambda item: item[1].rank)
        decode_tokens.append(
            [(value.decoded_token, float(value.logprob)) for _, value in ranked]
        )
    return decode_tokens


def compare_logprobs(
    baseline: list[dict[int, LogprobLike]],
    compare: list[dict[int, LogprobLike]],
    *,
    max_tokens: int,
) -> list[dict[str, str | int | float]]:
    if len(baseline) != len(compare):
        raise RuntimeError(
            f"Logprob length mismatch: baseline={len(baseline)} compare={len(compare)}"
        )
    decode_base = baseline[-max_tokens:]
    decode_comp = compare[-max_tokens:]
    summary: list[dict[str, str | int | float]] = []
    for step, (base_entry, comp_entry) in enumerate(zip(decode_base, decode_comp)):
        base_best = min(base_entry.items(), key=lambda item: item[1].rank)
        comp_best = min(comp_entry.items(), key=lambda item: item[1].rank)
        shared = set(base_entry.keys()) & set(comp_entry.keys())
        best_id = base_best[0]
        best_delta = (
            float(comp_entry[best_id].logprob - base_entry[best_id].logprob)
            if best_id in comp_entry
            else float("nan")
        )
        entry: dict[str, str | int | float] = {
            "step": step,
            "baseline_top": base_best[1].decoded_token,
            "compare_top": comp_best[1].decoded_token,
            "baseline_logprob": float(base_best[1].logprob),
            "compare_logprob": float(comp_best[1].logprob),
            "shared_delta_top": best_delta,
            "shared_count": len(shared),
        }
        summary.append(entry)
    return summary


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    args = parse_args()
    model = build_model(args)
    baseline = run_once(model, args, label="bf16", precision=None)
    print_section("Baseline (bf16 projection cap)")
    print(baseline.text)

    compare_label = args.compare_precision.lower()
    comparison: RunResult | None = None
    if compare_label != "none":
        comparison = run_once(
            model,
            args,
            label=compare_label,
            precision=compare_label,
        )
        print_section(f"Override ({compare_label})")
        print(comparison.text)

    if comparison is None:
        return

    capture_summary = summarise_diff(baseline.captures, comparison.captures)
    print_section("Hidden State Diffs")
    decode_entries = [entry for entry in capture_summary if entry.get("phase") == "decode"]
    if not decode_entries:
        print("No decode captures recorded.")
    else:
        for entry in decode_entries:
            print(
                f"step={entry['capture']:02d} decode: "
                f"after_max_abs={entry.get('after_max_abs', float('nan')):.3e} "
                f"after_rmse={entry.get('after_rmse', float('nan')):.3e}"
            )
    prefill_entries = [entry for entry in capture_summary if entry.get("phase") == "prefill"]
    if prefill_entries:
        tail = prefill_entries[-1]
        print(
            f"prefill tail: capture={tail['capture']:02d} "
            f"after_max_abs={tail.get('after_max_abs', float('nan')):.3e} "
            f"after_rmse={tail.get('after_rmse', float('nan')):.3e}"
        )

    logprob_summary = compare_logprobs(
        baseline.logprobs, comparison.logprobs, max_tokens=args.max_tokens
    )
    print_section("Decode Logprob Comparison")
    for item in logprob_summary:
        print(
            f"t={item['step']:02d} "
            f"bf16={item['baseline_top']!r} ({item['baseline_logprob']:.3f}) "
            f"{compare_label}={item['compare_top']!r} ({item['compare_logprob']:.3f}) "
            f"shared Δ={item['shared_delta_top']:.3f} "
            f"shared={item['shared_count']}"
        )


if __name__ == "__main__":
    main()
