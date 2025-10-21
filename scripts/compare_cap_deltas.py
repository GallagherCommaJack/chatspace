"""Compare cap_delta tensors between HF and vLLM sweeps.

Loads saved capture data and performs step-by-step comparison of:
- Hidden state MAE drift over decode steps
- cap_delta tensor divergence
- Correlation with token mismatch indices
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CaptureData:
    """Loaded capture data from a single run."""

    source: str  # "hf" or "vllm"
    config: str  # e.g., "bf16", "fp32"
    prompt_idx: int
    prompt: str
    text: str
    token_ids: list[int]
    captures: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-dir",
        type=str,
        required=True,
        help="Directory with HF sweep results (e.g., /workspace/cache/hf_precision_sweeps/run1)",
    )
    parser.add_argument(
        "--vllm-dir",
        type=str,
        required=True,
        help="Directory with vLLM sweep results (e.g., /workspace/cache/precision_sweeps/run1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bf16",
        help="Config to compare (e.g., bf16, fp32)",
    )
    parser.add_argument(
        "--prompt-idx",
        type=int,
        default=0,
        help="Prompt index to analyze",
    )
    parser.add_argument(
        "--reference-config",
        type=str,
        default="fp32",
        help="Reference config for computing drift (default: fp32)",
    )
    return parser.parse_args()


def load_capture(
    base_dir: Path, source: str, config: str, prompt_idx: int
) -> CaptureData:
    """Load a single capture file."""
    capture_file = base_dir / config / f"prompt_{prompt_idx:02d}.pt"

    if not capture_file.exists():
        raise FileNotFoundError(f"Capture file not found: {capture_file}")

    data = torch.load(capture_file, map_location="cpu")

    return CaptureData(
        source=source,
        config=config,
        prompt_idx=prompt_idx,
        prompt=data["prompt"],
        text=data["text"],
        token_ids=data["token_ids"],
        captures=data["captures"],
    )


def classify_capture(capture: dict[str, Any]) -> tuple[str, int]:
    """Extract phase and step from capture metadata."""
    meta = capture.get("meta", {})
    phase = meta.get("phase", "unknown")
    step = meta.get("step", 0)
    return phase, step


def compute_hidden_mae(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> float:
    """Compute mean absolute error between two tensors."""
    diff = tensor1.to(torch.float32) - tensor2.to(torch.float32)
    return float(diff.abs().mean().item())


def compute_hidden_cosine(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> float:
    """Compute cosine similarity between two tensors."""
    t1_flat = tensor1.reshape(-1).to(torch.float32)
    t2_flat = tensor2.reshape(-1).to(torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)
    )
    return float(cos_sim.item())


def analyze_decode_drift(
    target_data: CaptureData,
    reference_data: CaptureData,
    token_reference: CaptureData,
) -> None:
    """Analyze decode-step drift against reference and token divergence."""

    print(f"\n{'=' * 70}")
    print(f"Decode Drift Analysis")
    print(f"{'=' * 70}")
    print(f"Target:    {target_data.source}/{target_data.config}")
    print(f"Reference: {reference_data.source}/{reference_data.config}")
    print(f"Tokens vs: {token_reference.source}/{token_reference.config}")
    print(f"{'=' * 70}\n")

    # Extract decode captures
    target_decode = [
        c for c in target_data.captures if classify_capture(c)[0] == "decode"
    ]
    reference_decode = [
        c for c in reference_data.captures if classify_capture(c)[0] == "decode"
    ]

    if not target_decode:
        print(f"⚠️  No decode captures in {target_data.source}/{target_data.config}")
        return

    if not reference_decode:
        print(f"⚠️  No decode captures in {reference_data.source}/{reference_data.config}")
        return

    print(f"Target decode captures: {len(target_decode)}")
    print(f"Reference decode captures: {len(reference_decode)}")

    # Token divergence analysis
    token_match = [
        t1 == t2
        for t1, t2 in zip(target_data.token_ids, token_reference.token_ids)
    ]
    first_mismatch = next(
        (i for i, match in enumerate(token_match) if not match), None
    )

    print(f"\nToken Statistics:")
    print(f"  Total tokens: {len(target_data.token_ids)}")
    print(f"  Match count: {sum(token_match)}/{len(token_match)}")
    print(f"  Match ratio: {sum(token_match) / len(token_match):.3f}")
    print(f"  First mismatch at: {first_mismatch if first_mismatch is not None else 'None'}")

    # Step-by-step comparison
    print(f"\n{'Step':<6} {'Hidden MAE':<12} {'Cosine':<8} {'Token':<6} {'Notes':<30}")
    print("-" * 70)

    max_steps = min(len(target_decode), len(reference_decode))

    for step in range(max_steps):
        target_cap = target_decode[step]
        ref_cap = reference_decode[step]

        # Get hidden states (prefer 'after' which includes cap effect)
        target_hidden = target_cap.get("after") or target_cap.get("hidden")
        ref_hidden = ref_cap.get("after") or ref_cap.get("hidden")

        if target_hidden is None or ref_hidden is None:
            print(f"{step:<6} {'N/A':<12} {'N/A':<8} {'?':<6} Missing hidden state")
            continue

        # Compute metrics
        mae = compute_hidden_mae(target_hidden, ref_hidden)
        cosine = compute_hidden_cosine(target_hidden, ref_hidden)

        # Token status
        token_status = "✓" if step < len(token_match) and token_match[step] else "✗"

        # Notes
        notes = []
        if first_mismatch is not None and step == first_mismatch:
            notes.append("← FIRST TOKEN MISMATCH")
        if mae > 1.0:
            notes.append("HIGH MAE")

        print(
            f"{step:<6} {mae:<12.4e} {cosine:<8.6f} {token_status:<6} {' '.join(notes):<30}"
        )

        # Show cap_delta info if available
        if "cap_delta" in target_cap:
            cap_delta = target_cap["cap_delta"]
            cap_norm = float(cap_delta.norm().item())
            cap_meta = target_cap.get("cap_meta", {})
            cap_dtype = cap_meta.get("target_dtype", "unknown")
            print(f"       ├─ cap_delta norm: {cap_norm:.4e} (dtype: {cap_dtype})")

    print("-" * 70)


def analyze_cap_delta_drift(
    target_data: CaptureData,
    reference_data: CaptureData,
) -> None:
    """Analyze cap_delta tensor differences between target and reference."""

    print(f"\n{'=' * 70}")
    print(f"Cap Delta Drift Analysis")
    print(f"{'=' * 70}")
    print(f"Target:    {target_data.source}/{target_data.config}")
    print(f"Reference: {reference_data.source}/{reference_data.config}")
    print(f"{'=' * 70}\n")

    target_decode = [
        c for c in target_data.captures if classify_capture(c)[0] == "decode"
    ]
    reference_decode = [
        c for c in reference_data.captures if classify_capture(c)[0] == "decode"
    ]

    # Check if cap_delta is available
    target_has_delta = any("cap_delta" in c for c in target_decode)
    ref_has_delta = any("cap_delta" in c for c in reference_decode)

    if not target_has_delta:
        print(f"⚠️  No cap_delta in {target_data.source}/{target_data.config}")
        return

    if not ref_has_delta:
        print(f"⚠️  No cap_delta in {reference_data.source}/{reference_data.config}")
        return

    print(f"{'Step':<6} {'Target Norm':<12} {'Ref Norm':<12} {'Delta MAE':<12} {'Notes':<20}")
    print("-" * 70)

    max_steps = min(len(target_decode), len(reference_decode))

    for step in range(max_steps):
        target_cap = target_decode[step]
        ref_cap = reference_decode[step]

        target_delta = target_cap.get("cap_delta")
        ref_delta = ref_cap.get("cap_delta")

        if target_delta is None or ref_delta is None:
            print(f"{step:<6} {'N/A':<12} {'N/A':<12} {'N/A':<12} Missing cap_delta")
            continue

        target_norm = float(target_delta.norm().item())
        ref_norm = float(ref_delta.norm().item())
        delta_mae = compute_hidden_mae(target_delta, ref_delta)

        notes = []
        if delta_mae > 1e-2:
            notes.append("DIVERGING")

        print(
            f"{step:<6} {target_norm:<12.4e} {ref_norm:<12.4e} {delta_mae:<12.4e} {' '.join(notes):<20}"
        )

    print("-" * 70)


def main() -> None:
    args = parse_args()

    hf_dir = Path(args.hf_dir)
    vllm_dir = Path(args.vllm_dir)

    if not hf_dir.exists():
        print(f"❌ HF directory not found: {hf_dir}")
        return

    if not vllm_dir.exists():
        print(f"❌ vLLM directory not found: {vllm_dir}")
        return

    print(f"Loading captures for prompt {args.prompt_idx}, config {args.config}...")

    try:
        # Load target config data
        hf_target = load_capture(hf_dir, "hf", args.config, args.prompt_idx)
        vllm_target = load_capture(vllm_dir, "vllm", args.config, args.prompt_idx)

        # Load reference config data
        hf_reference = load_capture(
            hf_dir, "hf", args.reference_config, args.prompt_idx
        )
        vllm_reference = load_capture(
            vllm_dir, "vllm", args.reference_config, args.prompt_idx
        )

        print(f"✓ Loaded HF/{args.config}: {len(hf_target.captures)} captures")
        print(f"✓ Loaded vLLM/{args.config}: {len(vllm_target.captures)} captures")
        print(f"✓ Loaded HF/{args.reference_config}: {len(hf_reference.captures)} captures")
        print(f"✓ Loaded vLLM/{args.reference_config}: {len(vllm_reference.captures)} captures")

        # Analyze HF drift
        analyze_decode_drift(
            target_data=hf_target,
            reference_data=hf_reference,
            token_reference=hf_reference,
        )

        # Analyze vLLM drift
        analyze_decode_drift(
            target_data=vllm_target,
            reference_data=vllm_reference,
            token_reference=vllm_reference,
        )

        # Compare HF vs vLLM (same config)
        print(f"\n{'=' * 70}")
        print(f"Cross-Backend Comparison: HF vs vLLM (both {args.config})")
        print(f"{'=' * 70}")
        analyze_decode_drift(
            target_data=hf_target,
            reference_data=vllm_target,
            token_reference=vllm_reference,
        )

        # Cap delta analysis (vLLM only, since HF doesn't expose cap_delta)
        if "cap_delta" in (vllm_target.captures[0] if vllm_target.captures else {}):
            analyze_cap_delta_drift(
                target_data=vllm_target,
                reference_data=vllm_reference,
            )

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
