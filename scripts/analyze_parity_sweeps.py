"""Analyze HF vs vLLM parity sweep results.

Step-by-step analysis of numerical drift in precision sweeps. Shows:

- Token match statistics (vs reference precision)
- Hidden state MAE/cosine similarity per decode step
- First divergence point correlation with hidden state drift
- Cross-backend comparison (HF vs vLLM)
- cap_delta norm evolution (vLLM only)

Helps answer:
- Where does bf16 divergence start? (first mismatch step)
- How fast does MAE accumulate? (gradual vs sudden)
- Does cap precision matter? (compare bf16 vs bf16_cap_fp32)
- Which backend is more stable? (HF vs vLLM comparison)

Example:
    # Analyze prompt 0 from a sweep run
    python scripts/analyze_parity_sweeps.py \\
        --run-id 20251020_investigation \\
        --prompt-idx 0

    # Analyze different prompt showing gradual divergence
    python scripts/analyze_parity_sweeps.py \\
        --run-id 20251020_investigation \\
        --prompt-idx 1

Output includes step-by-step table with MAE, cosine similarity, token match
status, and annotations for key events (first divergence, high MAE).

See scripts/PRECISION_TESTING_README.md for interpretation guide.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/workspace/cache",
        help="Base directory containing sweep results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="20251020_cap_delta_investigation",
        help="Run ID subdirectory",
    )
    parser.add_argument(
        "--prompt-idx",
        type=int,
        default=0,
        help="Prompt index to analyze",
    )
    return parser.parse_args()


def load_capture(base_dir: Path, config: str, prompt_idx: int) -> dict[str, Any]:
    """Load a capture file."""
    capture_file = base_dir / config / f"prompt_{prompt_idx:02d}.pt"
    if not capture_file.exists():
        raise FileNotFoundError(f"Not found: {capture_file}")
    return torch.load(capture_file, map_location="cpu")


def compute_mae(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """Compute mean absolute error."""
    diff = t1.to(torch.float32) - t2.to(torch.float32)
    return float(diff.abs().mean().item())


def compute_cosine(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """Compute cosine similarity."""
    t1_flat = t1.reshape(-1).to(torch.float32)
    t2_flat = t2.reshape(-1).to(torch.float32)
    cos = torch.nn.functional.cosine_similarity(
        t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)
    )
    return float(cos.item())


def analyze_backend(
    name: str,
    bf16_data: dict[str, Any],
    ref_data: dict[str, Any],
) -> None:
    """Analyze drift for a single backend."""
    print(f"\n{'=' * 70}")
    print(f"{name} Analysis: bf16 vs reference")
    print(f"{'=' * 70}")

    # Token-level divergence
    bf16_tokens = bf16_data["token_ids"]
    ref_tokens = ref_data["token_ids"]
    token_matches = [t1 == t2 for t1, t2 in zip(bf16_tokens, ref_tokens)]
    first_mismatch = next((i for i, m in enumerate(token_matches) if not m), None)

    print(f"Token Stats:")
    print(f"  Total tokens: {len(bf16_tokens)}")
    print(f"  Match count: {sum(token_matches)}/{len(token_matches)}")
    print(f"  Match ratio: {sum(token_matches) / len(token_matches):.3f}")
    print(f"  First mismatch: {first_mismatch if first_mismatch is not None else 'None'}")

    # Extract decode captures
    bf16_caps = bf16_data["captures"]
    ref_caps = ref_data["captures"]

    bf16_decode = [c for c in bf16_caps if c.get("meta", {}).get("phase") == "decode"]
    ref_decode = [c for c in ref_caps if c.get("meta", {}).get("phase") == "decode"]

    print(f"\nCapture Stats:")
    print(f"  Total captures: {len(bf16_caps)}")
    print(f"  Decode captures: {len(bf16_decode)}")

    # Decode-step analysis
    print(f"\n{'Step':<6} {'MAE':<12} {'Cosine':<10} {'Token':<8} {'Notes'}")
    print("-" * 70)

    max_steps = min(len(bf16_decode), len(ref_decode), 20)  # Show first 20 steps

    for step in range(max_steps):
        bf16_hidden = bf16_decode[step].get("after")
        ref_hidden = ref_decode[step].get("after")

        if bf16_hidden is None or ref_hidden is None:
            continue

        mae = compute_mae(bf16_hidden, ref_hidden)
        cosine = compute_cosine(bf16_hidden, ref_hidden)

        token_status = "✓" if step < len(token_matches) and token_matches[step] else "✗"

        notes = []
        if first_mismatch is not None and step == first_mismatch:
            notes.append("← FIRST DIVERGENCE")
        if mae > 1.0:
            notes.append("HIGH MAE")

        print(
            f"{step:<6} {mae:<12.4e} {cosine:<10.6f} {token_status:<8} {' '.join(notes)}"
        )

        # Show cap_delta info if available (vLLM only)
        if "cap_delta" in bf16_decode[step]:
            cap_delta = bf16_decode[step]["cap_delta"]
            cap_norm = float(cap_delta.norm().item())
            cap_meta = bf16_decode[step].get("cap_meta", {})
            cap_dtype = cap_meta.get("target_dtype", "unknown")
            if step < 5 or step == first_mismatch:  # Show details for early steps
                print(f"       └─ cap_delta norm: {cap_norm:.4e} (dtype: {cap_dtype})")

    # Summary stats
    all_maes = [
        compute_mae(bf16_decode[i].get("after"), ref_decode[i].get("after"))
        for i in range(min(len(bf16_decode), len(ref_decode)))
        if bf16_decode[i].get("after") is not None and ref_decode[i].get("after") is not None
    ]

    if all_maes:
        print(f"\nSummary:")
        print(f"  Mean MAE: {sum(all_maes) / len(all_maes):.4e}")
        print(f"  Max MAE: {max(all_maes):.4e}")
        print(f"  MAE at first token mismatch: {all_maes[first_mismatch]:.4e}" if first_mismatch and first_mismatch < len(all_maes) else "")


def compare_backends(
    hf_bf16: dict[str, Any],
    vllm_bf16: dict[str, Any],
    vllm_ref: dict[str, Any],
) -> None:
    """Compare HF vs vLLM directly."""
    print(f"\n{'=' * 70}")
    print(f"Cross-Backend Comparison: HF bf16 vs vLLM bf16")
    print(f"{'=' * 70}")

    hf_decode = [c for c in hf_bf16["captures"] if c.get("meta", {}).get("phase") == "decode"]
    vllm_decode = [c for c in vllm_bf16["captures"] if c.get("meta", {}).get("phase") == "decode"]

    print(f"HF decode captures: {len(hf_decode)}")
    print(f"vLLM decode captures: {len(vllm_decode)}")

    # Token comparison
    hf_tokens = hf_bf16["token_ids"]
    vllm_tokens = vllm_bf16["token_ids"]
    token_matches = [t1 == t2 for t1, t2 in zip(hf_tokens, vllm_tokens)]
    first_diff = next((i for i, m in enumerate(token_matches) if not m), None)

    print(f"\nToken Comparison:")
    print(f"  HF tokens: {len(hf_tokens)}")
    print(f"  vLLM tokens: {len(vllm_tokens)}")
    print(f"  Match count: {sum(token_matches)}/{len(token_matches)}")
    print(f"  First difference: {first_diff if first_diff is not None else 'None'}")

    # Hidden state comparison
    print(f"\n{'Step':<6} {'HF Hidden':<20} {'vLLM Hidden':<20} {'Cross MAE':<12}")
    print("-" * 70)

    max_steps = min(len(hf_decode), len(vllm_decode), 20)

    for step in range(max_steps):
        hf_hidden = hf_decode[step].get("hidden") or hf_decode[step].get("after")
        vllm_hidden = vllm_decode[step].get("after")

        if hf_hidden is None or vllm_hidden is None:
            continue

        # Compare to vLLM reference
        vllm_ref_hidden = vllm_ref["captures"][step + 1].get("after")  # +1 for prefill offset
        if vllm_ref_hidden is not None:
            hf_mae = compute_mae(hf_hidden, vllm_ref_hidden)
            vllm_mae = compute_mae(vllm_hidden, vllm_ref_hidden)
            cross_mae = compute_mae(hf_hidden, vllm_hidden)

            print(
                f"{step:<6} MAE={hf_mae:<8.4e} {'':>11} MAE={vllm_mae:<8.4e} {'':>11} {cross_mae:<12.4e}"
            )


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    hf_dir = run_dir / "hf_precision_sweeps" / args.run_id
    vllm_dir = run_dir / "precision_sweeps" / args.run_id

    print(f"Loading captures for prompt {args.prompt_idx}...")

    try:
        # Load HF data
        hf_bf16 = load_capture(hf_dir, "bf16", args.prompt_idx)
        hf_fp32 = load_capture(hf_dir, "fp32", args.prompt_idx)

        # Load vLLM data
        vllm_bf16 = load_capture(vllm_dir, "bf16", args.prompt_idx)
        vllm_float32 = load_capture(vllm_dir, "float32", args.prompt_idx)

        print("✓ All captures loaded successfully")

        # Analyze each backend
        analyze_backend("HuggingFace", hf_bf16, hf_fp32)
        analyze_backend("vLLM", vllm_bf16, vllm_float32)

        # Cross-backend comparison
        compare_backends(hf_bf16, vllm_bf16, vllm_float32)

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
