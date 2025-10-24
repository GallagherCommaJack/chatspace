"""Deep diagnostic comparison of HuggingFace vs vLLM hidden states.

This test suite examines all differences between HF and vLLM hidden states
to understand their impact on additive steering interventions.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_comprehensive_hidden_state_diagnostics():
    """Comprehensive analysis of differences between HF and vLLM hidden states."""
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 2
    prompt = "The quick brown fox jumps over the lazy dog"

    print("\n" + "="*80)
    print("COMPREHENSIVE HIDDEN STATE DIAGNOSTIC: HF vs vLLM")
    print("="*80)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    num_tokens = inputs.input_ids.shape[1]

    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {num_tokens}")

    # Capture hidden states from HF model
    captured_hf_state = None

    def capture_hook(module, args, output):
        nonlocal captured_hf_state
        hidden_states = output[0] if isinstance(output, tuple) else output
        if captured_hf_state is None:
            captured_hf_state = hidden_states.detach().cpu().clone()
        return output

    # Install hook on target layer
    layer = hf_model.model.layers[target_layer]
    hook_handle = layer.register_forward_hook(capture_hook)

    # Forward pass
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)

    hook_handle.remove()

    assert captured_hf_state is not None
    hf_hidden = captured_hf_state.to(dtype=torch.float32)
    if hf_hidden.shape[0] == 1:
        hf_hidden = hf_hidden.squeeze(0)

    # Load vLLM model
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )

    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))

    # Enable capture for vLLM
    vllm_model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=False)

    # Generate with vLLM
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
    vllm_model.generate([prompt], sampling_params=sampling)

    # Fetch captured states
    vllm_states = vllm_model.fetch_hidden_states(layer_idx=target_layer)
    vllm_captures = vllm_states[0][target_layer]

    assert len(vllm_captures) > 0
    vllm_hidden = vllm_captures[0]["before"].to(dtype=torch.float32)

    # Ensure shapes match
    assert hf_hidden.shape == vllm_hidden.shape

    # =========================================================================
    # ANALYSIS 1: Basic Statistics
    # =========================================================================
    print("\n" + "-"*80)
    print("1. BASIC STATISTICS")
    print("-"*80)

    hf_flat = hf_hidden.reshape(-1)
    vllm_flat = vllm_hidden.reshape(-1)

    print(f"\nHuggingFace:")
    print(f"  Mean:   {hf_flat.mean().item():12.6f}")
    print(f"  Std:    {hf_flat.std().item():12.6f}")
    print(f"  Min:    {hf_flat.min().item():12.6f}")
    print(f"  Max:    {hf_flat.max().item():12.6f}")
    print(f"  Median: {hf_flat.median().item():12.6f}")

    print(f"\nvLLM:")
    print(f"  Mean:   {vllm_flat.mean().item():12.6f}")
    print(f"  Std:    {vllm_flat.std().item():12.6f}")
    print(f"  Min:    {vllm_flat.min().item():12.6f}")
    print(f"  Max:    {vllm_flat.max().item():12.6f}")
    print(f"  Median: {vllm_flat.median().item():12.6f}")

    # =========================================================================
    # ANALYSIS 2: Element-wise Differences
    # =========================================================================
    print("\n" + "-"*80)
    print("2. ELEMENT-WISE DIFFERENCES (vLLM - HF)")
    print("-"*80)

    diff = vllm_flat - hf_flat
    abs_diff = torch.abs(diff)

    print(f"\nAbsolute differences:")
    print(f"  Mean:   {abs_diff.mean().item():12.6f}")
    print(f"  Std:    {abs_diff.std().item():12.6f}")
    print(f"  Min:    {abs_diff.min().item():12.6f}")
    print(f"  Max:    {abs_diff.max().item():12.6f}")
    print(f"  Median: {abs_diff.median().item():12.6f}")

    print(f"\nSigned differences:")
    print(f"  Mean:   {diff.mean().item():12.6f}")
    print(f"  Std:    {diff.std().item():12.6f}")
    print(f"  Min:    {diff.min().item():12.6f}")
    print(f"  Max:    {diff.max().item():12.6f}")

    # =========================================================================
    # ANALYSIS 3: Relative Differences
    # =========================================================================
    print("\n" + "-"*80)
    print("3. RELATIVE DIFFERENCES")
    print("-"*80)

    # Relative error: |vllm - hf| / |hf|
    hf_magnitude = torch.abs(hf_flat)
    relative_err = abs_diff / (hf_magnitude + 1e-8)

    print(f"\nRelative error (|diff| / |HF|):")
    print(f"  Mean:   {relative_err.mean().item():12.6f}")
    print(f"  Median: {relative_err.median().item():12.6f}")
    print(f"  90th percentile: {torch.quantile(relative_err, 0.9).item():12.6f}")
    print(f"  99th percentile: {torch.quantile(relative_err, 0.99).item():12.6f}")
    print(f"  Max:    {relative_err.max().item():12.6f}")

    # =========================================================================
    # ANALYSIS 4: Similarity Metrics
    # =========================================================================
    print("\n" + "-"*80)
    print("4. SIMILARITY METRICS")
    print("-"*80)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_flat.unsqueeze(0), vllm_flat.unsqueeze(0)
    ).item()

    # Pearson correlation
    hf_centered = hf_flat - hf_flat.mean()
    vllm_centered = vllm_flat - vllm_flat.mean()
    correlation = (hf_centered * vllm_centered).sum() / (
        hf_centered.norm() * vllm_centered.norm() + 1e-8
    ).item()

    # R-squared (explained variance)
    ss_res = ((vllm_flat - hf_flat) ** 2).sum()
    ss_tot = ((hf_flat - hf_flat.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot).item()

    print(f"\nCosine similarity:    {cos_sim:.10f}")
    print(f"Pearson correlation:  {correlation:.10f}")
    print(f"R² (explained var):   {r_squared:.10f}")

    # =========================================================================
    # ANALYSIS 5: Per-Token Position Analysis
    # =========================================================================
    print("\n" + "-"*80)
    print("5. PER-TOKEN POSITION ANALYSIS")
    print("-"*80)

    # Shape: [num_tokens, hidden_size]
    per_token_diff = torch.abs(vllm_hidden - hf_hidden).mean(dim=-1)  # Mean over hidden dim
    per_token_norm_hf = hf_hidden.norm(dim=-1)
    per_token_norm_vllm = vllm_hidden.norm(dim=-1)

    print(f"\nMean absolute difference by token position:")
    for i in range(min(num_tokens, 10)):  # Show first 10 tokens
        print(f"  Token {i:2d}: {per_token_diff[i].item():12.6f}  "
              f"(HF norm: {per_token_norm_hf[i].item():8.2f}, "
              f"vLLM norm: {per_token_norm_vllm[i].item():8.2f})")

    if num_tokens > 10:
        print(f"  ... ({num_tokens - 10} more tokens)")

    # =========================================================================
    # ANALYSIS 6: Per-Dimension Analysis
    # =========================================================================
    print("\n" + "-"*80)
    print("6. PER-DIMENSION ANALYSIS")
    print("-"*80)

    # Shape: [num_tokens, hidden_size] -> aggregate over tokens
    per_dim_diff = torch.abs(vllm_hidden - hf_hidden).mean(dim=0)  # Mean over tokens
    per_dim_hf_mean = hf_hidden.mean(dim=0)
    per_dim_vllm_mean = vllm_hidden.mean(dim=0)

    # Find dimensions with largest differences
    top_diff_dims = torch.argsort(per_dim_diff, descending=True)[:10]

    print(f"\nTop 10 dimensions with largest mean absolute difference:")
    print(f"{'Dim':<8} {'Diff':<12} {'HF Mean':<12} {'vLLM Mean':<12}")
    print("-" * 50)
    for dim in top_diff_dims:
        dim_idx = dim.item()
        print(f"{dim_idx:<8} {per_dim_diff[dim_idx].item():<12.6f} "
              f"{per_dim_hf_mean[dim_idx].item():<12.6f} "
              f"{per_dim_vllm_mean[dim_idx].item():<12.6f}")

    # =========================================================================
    # ANALYSIS 7: Distribution Comparison
    # =========================================================================
    print("\n" + "-"*80)
    print("7. DISTRIBUTION COMPARISON (Percentiles)")
    print("-"*80)

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print(f"\n{'Percentile':<12} {'HF':<15} {'vLLM':<15} {'Difference':<15}")
    print("-" * 60)
    for p in percentiles:
        hf_val = torch.quantile(hf_flat, p / 100.0).item()
        vllm_val = torch.quantile(vllm_flat, p / 100.0).item()
        diff_val = vllm_val - hf_val
        print(f"{p}th{'':<9} {hf_val:<15.6f} {vllm_val:<15.6f} {diff_val:<15.6f}")

    # =========================================================================
    # ANALYSIS 8: Systematic Bias Check
    # =========================================================================
    print("\n" + "-"*80)
    print("8. SYSTEMATIC BIAS CHECK")
    print("-"*80)

    # Check if vLLM is systematically higher or lower
    sign_diff = torch.sign(diff)
    num_positive = (sign_diff > 0).sum().item()
    num_negative = (sign_diff < 0).sum().item()
    num_zero = (sign_diff == 0).sum().item()
    total = len(diff)

    print(f"\nSigned difference distribution:")
    print(f"  vLLM > HF:  {num_positive:8d} ({100*num_positive/total:5.2f}%)")
    print(f"  vLLM < HF:  {num_negative:8d} ({100*num_negative/total:5.2f}%)")
    print(f"  vLLM = HF:  {num_zero:8d} ({100*num_zero/total:5.2f}%)")

    # Overall bias
    mean_signed_diff = diff.mean().item()
    print(f"\nMean signed difference (vLLM - HF): {mean_signed_diff:.6f}")
    if abs(mean_signed_diff) < 0.01:
        print("  → No significant systematic bias detected")
    else:
        print(f"  → Systematic bias detected: vLLM is {'higher' if mean_signed_diff > 0 else 'lower'} on average")

    # =========================================================================
    # ANALYSIS 9: Scale Factor Analysis
    # =========================================================================
    print("\n" + "-"*80)
    print("9. SCALE FACTOR ANALYSIS")
    print("-"*80)

    # Compute optimal scale factor: minimize ||vllm - scale * hf||
    # Optimal scale = (vllm · hf) / (hf · hf)
    optimal_scale = (vllm_flat * hf_flat).sum() / (hf_flat * hf_flat).sum()
    optimal_scale = optimal_scale.item()

    scaled_hf = hf_flat * optimal_scale
    scaled_residual = (vllm_flat - scaled_hf).norm().item()
    original_residual = (vllm_flat - hf_flat).norm().item()

    print(f"\nOptimal scale factor: {optimal_scale:.10f}")
    print(f"Residual norm (unscaled):  {original_residual:.6f}")
    print(f"Residual norm (scaled):    {scaled_residual:.6f}")
    print(f"Improvement:               {100*(1 - scaled_residual/original_residual):.4f}%")

    if abs(optimal_scale - 1.0) < 0.0001:
        print("  → No significant scaling difference")
    else:
        print(f"  → vLLM hidden states are {optimal_scale:.6f}x HF scale")

    # =========================================================================
    # SUMMARY FOR STEERING IMPLICATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: IMPLICATIONS FOR ADDITIVE STEERING")
    print("="*80)

    steering_compatible = True
    warnings = []

    # Check 1: Overall scale similarity
    scale_diff = abs(vllm_flat.std().item() / hf_flat.std().item() - 1.0)
    if scale_diff > 0.01:  # More than 1% difference in scale
        warnings.append(f"Scale difference: {scale_diff*100:.2f}%")
        steering_compatible = False

    # Check 2: Mean difference
    mean_diff_normalized = abs(diff.mean().item()) / hf_flat.std().item()
    if mean_diff_normalized > 0.01:  # Mean diff > 1% of std
        warnings.append(f"Mean shift: {mean_diff_normalized*100:.2f}% of std")
        steering_compatible = False

    # Check 3: High correlation
    if correlation < 0.999:
        warnings.append(f"Correlation only {correlation:.6f}")
        steering_compatible = False

    print(f"\nSteering compatibility: {'✓ GOOD' if steering_compatible else '⚠ POTENTIAL ISSUES'}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nNo significant differences detected.")
        print("Additive steering vectors should transfer well between HF and vLLM.")

    print("\nKey metrics for steering:")
    print(f"  - Scale ratio (vLLM/HF std): {vllm_flat.std().item() / hf_flat.std().item():.6f}")
    print(f"  - Mean absolute difference:  {abs_diff.mean().item():.6f}")
    print(f"  - Cosine similarity:         {cos_sim:.6f}")

    # Clean up
    vllm_model.clear_all_vectors()
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("END OF DIAGNOSTIC")
    print("="*80 + "\n")
