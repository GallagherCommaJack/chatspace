"""Parity checks for steering transforms between HuggingFace and vLLM backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

# vLLM >=0.11 requires enabling pickle-based serialization for custom RPCs.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@dataclass
class ProjectionCapParams:
    vector: torch.Tensor
    min: float | None
    max: float | None


@dataclass
class AblationParams:
    vector: torch.Tensor
    scale: float


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vector)
    if float(norm) <= 0:
        raise ValueError("Direction vector norm must be positive.")
    return vector / norm


def _apply_projection_cap(hidden: torch.Tensor, unit: torch.Tensor, *, minimum: float | None, maximum: float | None) -> torch.Tensor:
    if minimum is None and maximum is None:
        return hidden
    flat = hidden.reshape(-1, hidden.shape[-1])
    projection = flat @ unit
    clamp_kwargs: dict[str, Any] = {}
    if minimum is not None:
        clamp_kwargs["min"] = projection.new_tensor(float(minimum))
    if maximum is not None:
        clamp_kwargs["max"] = projection.new_tensor(float(maximum))
    clamped = torch.clamp(projection, **clamp_kwargs)  # type: ignore[arg-type]
    if clamped is projection:
        return hidden
    delta = (clamped - projection).unsqueeze(-1) * unit
    return (flat + delta).reshape_as(hidden)


def _apply_ablation(hidden: torch.Tensor, unit: torch.Tensor, *, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return hidden
    flat = hidden.reshape(-1, hidden.shape[-1])
    projection = flat @ unit
    component = projection.unsqueeze(-1) * unit
    return (flat + (scale - 1.0) * component).reshape_as(hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_hf_steering_combinations_match():
    torch.manual_seed(123)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5  # Check that steering propagates to subsequent layers
    prompt = "A curious squirrel contemplates the mysteries of the forest."

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
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size
    vector = torch.randn(hidden_size, dtype=torch.float32) * 0.05
    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)

    projection_spec = ProjectionCapParams(vector=cap_vector, min=-0.2, max=0.2)
    ablation_spec = AblationParams(vector=ablation_vector, scale=0.6)

    cases: list[dict[str, Any]] = [
        {"name": "vector_only", "vector": vector, "cap": None, "ablation": None},
        {"name": "projection_cap_only", "vector": None, "cap": projection_spec, "ablation": None},
        {"name": "ablation_only", "vector": None, "cap": None, "ablation": ablation_spec},
        {"name": "vector_and_cap", "vector": vector, "cap": projection_spec, "ablation": None},
        {"name": "vector_cap_ablation", "vector": vector, "cap": projection_spec, "ablation": ablation_spec},
    ]

    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))
    vllm_model.enable_hidden_state_capture([target_layer, downstream_layer], capture_before=False, capture_after=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        for case in cases:
            # ------------------------------------------------------------------
            # HuggingFace path
            # ------------------------------------------------------------------
            captured_hf_hiddens: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int):
                def hook(module, _args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    orig_dtype = hidden.dtype
                    hidden_fp32 = hidden.to(torch.float32)

                    # Only apply steering at target layer
                    if layer_idx == target_layer:
                        if case["vector"] is not None:
                            hidden_fp32 = hidden_fp32 + case["vector"].to(device=hidden_fp32.device)

                        if case["cap"] is not None:
                            unit = _normalize(case["cap"].vector).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                            hidden_fp32 = _apply_projection_cap(
                                hidden_fp32,
                                unit,
                                minimum=case["cap"].min,
                                maximum=case["cap"].max,
                            )

                        if case["ablation"] is not None:
                            unit = _normalize(case["ablation"].vector).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                            hidden_fp32 = _apply_ablation(hidden_fp32, unit, scale=case["ablation"].scale)

                    captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
                    hidden_out = hidden_fp32.to(dtype=orig_dtype)
                    if isinstance(output, tuple):
                        return (hidden_out,) + output[1:]
                    return hidden_out
                return hook

            # Install hooks on both layers
            target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
            downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

            with torch.no_grad():
                hf_model(**inputs)

            target_handle.remove()
            downstream_handle.remove()

            hf_target_hidden = captured_hf_hiddens[target_layer]
            hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

            # ------------------------------------------------------------------
            # vLLM path
            # ------------------------------------------------------------------
            vllm_model.clear_all_vectors()
            vllm_model.clear_layer_projection_cap(target_layer)
            vllm_model.clear_layer_ablation(target_layer)
            vllm_model.clear_hidden_states(target_layer)
            vllm_model.clear_hidden_states(downstream_layer)

            # Only set steering at target layer
            if case["vector"] is not None:
                vllm_model.set_layer_vector(target_layer, case["vector"])
            if case["cap"] is not None:
                vllm_model.set_layer_projection_cap(
                    target_layer,
                    case["cap"].vector,
                    min=case["cap"].min,
                    max=case["cap"].max,
                )
            if case["ablation"] is not None:
                vllm_model.set_layer_ablation(
                    target_layer,
                    case["ablation"].vector,
                    scale=case["ablation"].scale,
                )

            vllm_model.generate([prompt], sampling_params=sampling)
            states = vllm_model.fetch_hidden_states()
            vllm_target_hidden = states[0][target_layer][0]["after"].to(dtype=torch.float32)
            vllm_downstream_hidden = states[0][downstream_layer][0]["after"].to(dtype=torch.float32)

            # ------------------------------------------------------------------
            # Comparison
            # ------------------------------------------------------------------
            # Check target layer (where steering is applied)
            hf_target_flat = hf_target_hidden.reshape(-1)
            vllm_target_flat = vllm_target_hidden.reshape(-1)
            target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
            target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

            assert target_mae < 2e-3, f"{case['name']} (target layer): mean abs diff too large ({target_mae:.6f})"
            assert target_cos > 0.9995, f"{case['name']} (target layer): cosine similarity degraded ({target_cos:.6f})"

            # Check downstream layer (verify steering propagates)
            hf_downstream_flat = hf_downstream_hidden.reshape(-1)
            vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
            downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
            downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

            assert downstream_mae < 2e-3, f"{case['name']} (downstream layer): mean abs diff too large ({downstream_mae:.6f})"
            assert downstream_cos > 0.9995, f"{case['name']} (downstream layer): cosine similarity degraded ({downstream_cos:.6f})"
    finally:
        del vllm_model
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_decode_steering_matches_hf_prefill():
    """Test that vLLM decode-time steering matches HF prefill-time steering.

    This validates that steering applied during vLLM's autoregressive decode phase
    produces the same hidden states as HF's prefill phase with the same tokens.

    Strategy:
    1. HF: Forward pass with [prompt + generated_tokens] in prefill mode (with steering)
    2. vLLM: Generate tokens autoregressively starting from prompt (with steering)
    3. Compare hidden states at each token position

    If vLLM decode-time steering works correctly, the hidden states should match.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5  # Check that steering propagates to subsequent layers
    prompt = "The capital of France is"

    # Load HF model
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

    # Create steering vector
    hidden_size = hf_model.config.hidden_size
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 0.1

    # -------------------------------------------------------------------------
    # Step 1: vLLM generate with steering (decode mode)
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=64,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Set steering vector (only at target layer)
    vllm_model.set_layer_vector(target_layer, steering_vector)

    # Enable capture on both layers
    vllm_model.enable_hidden_state_capture([target_layer, downstream_layer], capture_before=False, capture_after=True)

    # Generate tokens
    sampling = SamplingParams(temperature=0.0, max_tokens=5, logprobs=0, ignore_eos=True)
    outputs = vllm_model.generate([prompt], sampling_params=sampling)

    # Get the generated text (outputs is list[str] of completion only)
    generated_text = outputs[0]
    full_text = prompt + generated_text

    # Fetch vLLM hidden states (decode mode)
    vllm_states = vllm_model.fetch_hidden_states()
    vllm_target_captures = vllm_states[0][target_layer]
    vllm_downstream_captures = vllm_states[0][downstream_layer]

    # vLLM captures: [0] = prefill, [1:] = decode steps
    num_decode_steps = len(vllm_target_captures) - 1
    assert num_decode_steps >= 3, f"Need at least 3 decode steps, got {num_decode_steps}"

    print(f"\nGenerated text: {repr(generated_text)}")
    print(f"Full text: {repr(full_text)}")
    print(f"vLLM captures: {len(vllm_target_captures)} (1 prefill + {num_decode_steps} decode)")

    # -------------------------------------------------------------------------
    # Step 2: HF forward pass with full text (prefill mode with steering)
    # -------------------------------------------------------------------------
    # Tokenize the full text (what vLLM generated)
    full_inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    prompt_len = prompt_inputs.input_ids.shape[1]
    full_len = full_inputs.input_ids.shape[1]

    print(f"Prompt tokens: {prompt_len}, Full tokens: {full_len}")

    # Capture hidden states from HF with steering
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def capture_hook(module, args, output):
            """Hook to capture and steer hidden states from HF model."""
            hidden = output[0] if isinstance(output, tuple) else output
            orig_dtype = hidden.dtype
            hidden_fp32 = hidden.to(torch.float32)

            # Apply steering only at target layer (same vector vLLM used)
            if layer_idx == target_layer:
                hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

            # Capture the steered hidden state
            captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()

            # Return steered hidden state
            hidden_out = hidden_fp32.to(dtype=orig_dtype)
            if isinstance(output, tuple):
                return (hidden_out,) + output[1:]
            return hidden_out
        return capture_hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**full_inputs)

    target_handle.remove()
    downstream_handle.remove()

    # HF captured the full sequence: [prompt_tokens + generated_tokens]
    assert len(captured_hf_hiddens) == 2, "Should have captured both layers"
    hf_target_hidden_full = captured_hf_hiddens[target_layer].to(dtype=torch.float32).squeeze(0)  # [seq_len, hidden_size]
    hf_downstream_hidden_full = captured_hf_hiddens[downstream_layer].to(dtype=torch.float32).squeeze(0)  # [seq_len, hidden_size]

    print(f"HF target hidden shape: {hf_target_hidden_full.shape}")
    print(f"HF downstream hidden shape: {hf_downstream_hidden_full.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Compare vLLM decode-time vs HF prefill-time hidden states
    # -------------------------------------------------------------------------
    print(f"\nComparing vLLM decode (steered) vs HF prefill (steered):")

    for decode_idx in range(num_decode_steps):
        # HF: token position in the full sequence
        hf_token_idx = prompt_len + decode_idx

        if hf_token_idx >= hf_target_hidden_full.shape[0]:
            print(f"  Decode step {decode_idx}: Skipping (out of bounds)")
            continue

        # -------------------------------------------------------------------------
        # Target layer comparison
        # -------------------------------------------------------------------------
        # vLLM: hidden state from decode step (after steering)
        vllm_target_hidden = vllm_target_captures[decode_idx + 1]["after"].to(dtype=torch.float32)
        if vllm_target_hidden.dim() > 1:
            vllm_target_hidden = vllm_target_hidden.squeeze(0)  # [hidden_size]

        # HF: hidden state from prefill at the corresponding token position
        hf_target_hidden = hf_target_hidden_full[hf_token_idx]  # [hidden_size]

        # Compare
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)

        target_cos_sim = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()
        target_mae = torch.mean(torch.abs(hf_target_flat - vllm_target_flat)).item()

        # -------------------------------------------------------------------------
        # Downstream layer comparison (verify steering propagates)
        # -------------------------------------------------------------------------
        vllm_downstream_hidden = vllm_downstream_captures[decode_idx + 1]["after"].to(dtype=torch.float32)
        if vllm_downstream_hidden.dim() > 1:
            vllm_downstream_hidden = vllm_downstream_hidden.squeeze(0)

        hf_downstream_hidden = hf_downstream_hidden_full[hf_token_idx]

        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)

        downstream_cos_sim = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()
        downstream_mae = torch.mean(torch.abs(hf_downstream_flat - vllm_downstream_flat)).item()

        print(f"  Decode step {decode_idx} (token position {hf_token_idx}):")
        print(f"    Target layer:     cos={target_cos_sim:.6f}, MAE={target_mae:.6f}")
        print(f"    Downstream layer: cos={downstream_cos_sim:.6f}, MAE={downstream_mae:.6f}")

        # Assert high similarity for target layer (steering is working during decode)
        assert target_cos_sim > 0.999, (
            f"Decode step {decode_idx} (target layer): Cosine similarity {target_cos_sim:.6f} should be >0.999. "
            f"This indicates decode-time steering may not match prefill-time steering."
        )
        assert target_mae < 0.01, (
            f"Decode step {decode_idx} (target layer): MAE {target_mae:.6f} should be <0.01"
        )

        # Assert high similarity for downstream layer (steering propagates)
        assert downstream_cos_sim > 0.999, (
            f"Decode step {decode_idx} (downstream layer): Cosine similarity {downstream_cos_sim:.6f} should be >0.999. "
            f"This indicates steering may not propagate correctly through layers."
        )
        assert downstream_mae < 0.01, (
            f"Decode step {decode_idx} (downstream layer): MAE {downstream_mae:.6f} should be <0.01"
        )

    print(f"\n✓ vLLM decode-time steering matches HF prefill-time steering (target + downstream layers)")

    # Cleanup
    vllm_model.clear_all_vectors()
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_hf_high_magnitude_steering():
    """Test that steering works correctly at high magnitudes (10x).

    This test validates that the fix works under aggressive steering conditions.
    High-magnitude steering can expose bugs in residual stream handling that
    might not be visible with subtle perturbations.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "The quick brown fox jumps over the lazy dog."

    # Load HF model
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
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # HIGH MAGNITUDE: 10.0 instead of 0.05-0.1
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 10.0

    print(f"\nSteering vector magnitude: {torch.norm(steering_vector).item():.2f}")

    # -------------------------------------------------------------------------
    # HuggingFace path
    # -------------------------------------------------------------------------
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            orig_dtype = hidden.dtype
            hidden_fp32 = hidden.to(torch.float32)

            # Apply steering only at target layer
            if layer_idx == target_layer:
                hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

            captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
            hidden_out = hidden_fp32.to(dtype=orig_dtype)
            if isinstance(output, tuple):
                return (hidden_out,) + output[1:]
            return hidden_out
        return hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**inputs)

    target_handle.remove()
    downstream_handle.remove()

    hf_target_hidden = captured_hf_hiddens[target_layer]
    hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

    # -------------------------------------------------------------------------
    # vLLM path
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Set high-magnitude steering vector
    vllm_model.set_layer_vector(target_layer, steering_vector)

    # Enable capture on both layers
    vllm_model.enable_hidden_state_capture([target_layer, downstream_layer], capture_before=False, capture_after=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        vllm_model.generate([prompt], sampling_params=sampling)

        states = vllm_model.fetch_hidden_states()
        vllm_target_hidden = states[0][target_layer][0]["after"].to(dtype=torch.float32)
        vllm_downstream_hidden = states[0][downstream_layer][0]["after"].to(dtype=torch.float32)

        # -------------------------------------------------------------------------
        # Comparison
        # -------------------------------------------------------------------------
        # Target layer
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)
        target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
        target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

        # Downstream layer
        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
        downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
        downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

        # Print statistics
        hf_target_norm = torch.norm(hf_target_flat).item()
        vllm_target_norm = torch.norm(vllm_target_flat).item()
        hf_downstream_norm = torch.norm(hf_downstream_flat).item()
        vllm_downstream_norm = torch.norm(vllm_downstream_flat).item()

        print(f"\nTarget layer (where steering is applied):")
        print(f"  HF norm: {hf_target_norm:.2f}, vLLM norm: {vllm_target_norm:.2f}")
        print(f"  Cosine similarity: {target_cos:.6f}")
        print(f"  MAE: {target_mae:.6f}")

        print(f"\nDownstream layer (steering propagated):")
        print(f"  HF norm: {hf_downstream_norm:.2f}, vLLM norm: {vllm_downstream_norm:.2f}")
        print(f"  Cosine similarity: {downstream_cos:.6f}")
        print(f"  MAE: {downstream_mae:.6f}")

        # Assert high similarity even at high magnitude
        assert target_cos > 0.999, (
            f"Target layer: Cosine similarity {target_cos:.6f} should be >0.999 even at high magnitude"
        )
        assert target_mae < 0.01, (
            f"Target layer: MAE {target_mae:.6f} should be <0.01"
        )

        assert downstream_cos > 0.999, (
            f"Downstream layer: Cosine similarity {downstream_cos:.6f} should be >0.999, "
            f"indicating steering propagates correctly even at high magnitude"
        )
        assert downstream_mae < 0.01, (
            f"Downstream layer: MAE {downstream_mae:.6f} should be <0.01"
        )

        print(f"\n✓ High-magnitude steering (10x) works correctly at target and downstream layers")

    finally:
        vllm_model.clear_all_vectors()
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_hf_high_magnitude_ablation_and_capping():
    """Test that ablation and projection capping work correctly at high magnitudes.

    This test validates that the dual-mode transform system (mode="hidden" for
    ablations and caps) works correctly under aggressive steering conditions.
    """
    torch.manual_seed(123)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "The capital of France is Paris, which is known for"

    # Load HF model
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
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # HIGH MAGNITUDE: 10.0 for vector, wide range for projection cap
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 10.0
    cap_direction = torch.randn(hidden_size, dtype=torch.float32)
    ablation_direction = torch.randn(hidden_size, dtype=torch.float32)

    # Extreme projection cap range and aggressive ablation
    projection_spec = ProjectionCapParams(vector=cap_direction, min=-50.0, max=50.0)  # Wide range
    ablation_spec = AblationParams(vector=ablation_direction, scale=0.1)  # Aggressive ablation (90% removal)

    print(f"\nSteering vector norm: {torch.norm(steering_vector).item():.2f}")
    print(f"Projection cap range: [{projection_spec.min}, {projection_spec.max}]")
    print(f"Ablation scale: {ablation_spec.scale}")

    # -------------------------------------------------------------------------
    # HuggingFace path
    # -------------------------------------------------------------------------
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            orig_dtype = hidden.dtype
            hidden_fp32 = hidden.to(torch.float32)

            # Apply all steering operations only at target layer
            if layer_idx == target_layer:
                # Vector addition
                hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

                # Projection cap
                unit_cap = _normalize(cap_direction).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                hidden_fp32 = _apply_projection_cap(
                    hidden_fp32,
                    unit_cap,
                    minimum=projection_spec.min,
                    maximum=projection_spec.max,
                )

                # Ablation
                unit_ablation = _normalize(ablation_direction).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                hidden_fp32 = _apply_ablation(hidden_fp32, unit_ablation, scale=ablation_spec.scale)

            captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
            hidden_out = hidden_fp32.to(dtype=orig_dtype)
            if isinstance(output, tuple):
                return (hidden_out,) + output[1:]
            return hidden_out
        return hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**inputs)

    target_handle.remove()
    downstream_handle.remove()

    hf_target_hidden = captured_hf_hiddens[target_layer]
    hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

    # -------------------------------------------------------------------------
    # vLLM path
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Set high-magnitude steering operations
    vllm_model.set_layer_vector(target_layer, steering_vector)
    vllm_model.set_layer_projection_cap(
        target_layer,
        cap_direction,
        min=projection_spec.min,
        max=projection_spec.max,
    )
    vllm_model.set_layer_ablation(
        target_layer,
        ablation_direction,
        scale=ablation_spec.scale,
    )

    # Enable capture on both layers
    vllm_model.enable_hidden_state_capture([target_layer, downstream_layer], capture_before=False, capture_after=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        vllm_model.generate([prompt], sampling_params=sampling)

        states = vllm_model.fetch_hidden_states()
        vllm_target_hidden = states[0][target_layer][0]["after"].to(dtype=torch.float32)
        vllm_downstream_hidden = states[0][downstream_layer][0]["after"].to(dtype=torch.float32)

        # -------------------------------------------------------------------------
        # Comparison
        # -------------------------------------------------------------------------
        # Target layer
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)
        target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
        target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

        # Downstream layer
        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
        downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
        downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

        # Print statistics
        hf_target_norm = torch.norm(hf_target_flat).item()
        vllm_target_norm = torch.norm(vllm_target_flat).item()
        hf_downstream_norm = torch.norm(hf_downstream_flat).item()
        vllm_downstream_norm = torch.norm(vllm_downstream_flat).item()

        print(f"\nTarget layer (vector + cap + ablation):")
        print(f"  HF norm: {hf_target_norm:.2f}, vLLM norm: {vllm_target_norm:.2f}")
        print(f"  Cosine similarity: {target_cos:.6f}")
        print(f"  MAE: {target_mae:.6f}")

        print(f"\nDownstream layer (all operations propagated):")
        print(f"  HF norm: {hf_downstream_norm:.2f}, vLLM norm: {vllm_downstream_norm:.2f}")
        print(f"  Cosine similarity: {downstream_cos:.6f}")
        print(f"  MAE: {downstream_mae:.6f}")

        # Assert high similarity even with aggressive ablation and capping
        assert target_cos > 0.999, (
            f"Target layer: Cosine similarity {target_cos:.6f} should be >0.999 "
            f"even with high-magnitude vector + cap + ablation"
        )
        assert target_mae < 0.01, (
            f"Target layer: MAE {target_mae:.6f} should be <0.01"
        )

        assert downstream_cos > 0.999, (
            f"Downstream layer: Cosine similarity {downstream_cos:.6f} should be >0.999, "
            f"indicating all operations propagate correctly"
        )
        assert downstream_mae < 0.01, (
            f"Downstream layer: MAE {downstream_mae:.6f} should be <0.01"
        )

        print(f"\n✓ High-magnitude ablation and projection capping work correctly")

    finally:
        vllm_model.clear_all_vectors()
        vllm_model.clear_layer_projection_cap(target_layer)
        vllm_model.clear_layer_ablation(target_layer)
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_hf_multi_magnitude_steering():
    """Test that steering works correctly across a wide spectrum of magnitudes.

    This test validates the fix works from very subtle (0.001) to very aggressive (10.0)
    steering, ensuring no magnitude-dependent bugs in the residual stream handling.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "In the beginning, there was"

    # Load HF model once
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
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # Test a spectrum of magnitudes
    magnitudes = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]

    # Load vLLM model once
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))
    vllm_model.enable_hidden_state_capture([target_layer, downstream_layer], capture_before=False, capture_after=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    print(f"\nTesting {len(magnitudes)} magnitudes:")

    try:
        for magnitude in magnitudes:
            # Create steering vector at this magnitude
            torch.manual_seed(42)  # Same direction for all magnitudes
            steering_vector = torch.randn(hidden_size, dtype=torch.float32) * magnitude

            # -------------------------------------------------------------------------
            # HuggingFace path
            # -------------------------------------------------------------------------
            captured_hf_hiddens: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int):
                def hook(module, _args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    orig_dtype = hidden.dtype
                    hidden_fp32 = hidden.to(torch.float32)

                    # Apply steering only at target layer
                    if layer_idx == target_layer:
                        hidden_fp32 = hidden_fp32 + steering_vector.to(device=hidden_fp32.device)

                    captured_hf_hiddens[layer_idx] = hidden_fp32.detach().cpu().clone()
                    hidden_out = hidden_fp32.to(dtype=orig_dtype)
                    if isinstance(output, tuple):
                        return (hidden_out,) + output[1:]
                    return hidden_out
                return hook

            # Install hooks on both layers
            target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
            downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

            with torch.no_grad():
                hf_model(**inputs)

            target_handle.remove()
            downstream_handle.remove()

            hf_target_hidden = captured_hf_hiddens[target_layer]
            hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

            # -------------------------------------------------------------------------
            # vLLM path
            # -------------------------------------------------------------------------
            vllm_model.clear_all_vectors()
            vllm_model.clear_hidden_states(target_layer)
            vllm_model.clear_hidden_states(downstream_layer)

            # Set steering vector at this magnitude
            vllm_model.set_layer_vector(target_layer, steering_vector)

            vllm_model.generate([prompt], sampling_params=sampling)

            states = vllm_model.fetch_hidden_states()
            vllm_target_hidden = states[0][target_layer][0]["after"].to(dtype=torch.float32)
            vllm_downstream_hidden = states[0][downstream_layer][0]["after"].to(dtype=torch.float32)

            # -------------------------------------------------------------------------
            # Comparison
            # -------------------------------------------------------------------------
            # Target layer
            hf_target_flat = hf_target_hidden.reshape(-1)
            vllm_target_flat = vllm_target_hidden.reshape(-1)
            target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
            target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()

            # Downstream layer
            hf_downstream_flat = hf_downstream_hidden.reshape(-1)
            vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
            downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
            downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()

            vector_norm = torch.norm(steering_vector).item()

            print(f"\n  Magnitude {magnitude:5.3f} (vector norm: {vector_norm:6.2f}):")
            print(f"    Target:     cos={target_cos:.6f}, MAE={target_mae:.6f}")
            print(f"    Downstream: cos={downstream_cos:.6f}, MAE={downstream_mae:.6f}")

            # Assert high similarity across all magnitudes
            assert target_cos > 0.999, (
                f"Magnitude {magnitude}: Target layer cosine similarity {target_cos:.6f} should be >0.999"
            )
            assert target_mae < 0.01, (
                f"Magnitude {magnitude}: Target layer MAE {target_mae:.6f} should be <0.01"
            )

            assert downstream_cos > 0.999, (
                f"Magnitude {magnitude}: Downstream layer cosine similarity {downstream_cos:.6f} should be >0.999"
            )
            assert downstream_mae < 0.01, (
                f"Magnitude {magnitude}: Downstream layer MAE {downstream_mae:.6f} should be <0.01"
            )

        print(f"\n✓ All {len(magnitudes)} magnitudes (0.001 - 10.0) work correctly")

    finally:
        vllm_model.clear_all_vectors()
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_hf_high_precision_steering():
    """Test steering with float32 precision to check for subtle numerical issues.

    If there's a bug in the residual stream handling, running both models in
    higher precision should still show errors. If errors decrease significantly,
    it suggests the implementation is correct but precision-limited.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    downstream_layer = 5
    prompt = "The meaning of life is"

    # Load HF model in FLOAT32
    print("\nLoading models in float32 precision...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # FLOAT32 instead of float16
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size

    # Normal magnitude steering
    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 1.0

    print(f"Steering vector norm: {torch.norm(steering_vector).item():.2f}")

    # -------------------------------------------------------------------------
    # HuggingFace path (float32)
    # -------------------------------------------------------------------------
    captured_hf_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Already float32, no conversion needed

            # Apply steering only at target layer
            if layer_idx == target_layer:
                hidden = hidden + steering_vector.to(device=hidden.device)

            captured_hf_hiddens[layer_idx] = hidden.detach().cpu().clone()

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    # Install hooks on both layers
    target_handle = hf_model.model.layers[target_layer].register_forward_hook(make_hook(target_layer))
    downstream_handle = hf_model.model.layers[downstream_layer].register_forward_hook(make_hook(downstream_layer))

    with torch.no_grad():
        hf_model(**inputs)

    target_handle.remove()
    downstream_handle.remove()

    hf_target_hidden = captured_hf_hiddens[target_layer]
    hf_downstream_hidden = captured_hf_hiddens[downstream_layer]

    # -------------------------------------------------------------------------
    # vLLM path (float32)
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,  # Slightly more memory for float32
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float32",  # FLOAT32 instead of float16
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer, downstream_layer))

    # Set steering vector
    vllm_model.set_layer_vector(target_layer, steering_vector)

    # Enable capture on both layers
    vllm_model.enable_hidden_state_capture([target_layer, downstream_layer], capture_before=False, capture_after=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        vllm_model.generate([prompt], sampling_params=sampling)

        states = vllm_model.fetch_hidden_states()
        vllm_target_hidden = states[0][target_layer][0]["after"]
        vllm_downstream_hidden = states[0][downstream_layer][0]["after"]

        # -------------------------------------------------------------------------
        # Comparison (everything in float32)
        # -------------------------------------------------------------------------
        # Target layer
        hf_target_flat = hf_target_hidden.reshape(-1)
        vllm_target_flat = vllm_target_hidden.reshape(-1)
        target_mae = torch.mean(torch.abs(vllm_target_flat - hf_target_flat)).item()
        target_cos = F.cosine_similarity(hf_target_flat.unsqueeze(0), vllm_target_flat.unsqueeze(0), dim=-1).item()
        target_max_diff = torch.max(torch.abs(vllm_target_flat - hf_target_flat)).item()

        # Downstream layer
        hf_downstream_flat = hf_downstream_hidden.reshape(-1)
        vllm_downstream_flat = vllm_downstream_hidden.reshape(-1)
        downstream_mae = torch.mean(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()
        downstream_cos = F.cosine_similarity(hf_downstream_flat.unsqueeze(0), vllm_downstream_flat.unsqueeze(0), dim=-1).item()
        downstream_max_diff = torch.max(torch.abs(vllm_downstream_flat - hf_downstream_flat)).item()

        print(f"\nTarget layer (float32 precision):")
        print(f"  Cosine similarity: {target_cos:.12f}")
        print(f"  MAE: {target_mae:.12f}")
        print(f"  Max diff: {target_max_diff:.12f}")

        print(f"\nDownstream layer (float32 precision):")
        print(f"  Cosine similarity: {downstream_cos:.12f}")
        print(f"  MAE: {downstream_mae:.12f}")
        print(f"  Max diff: {downstream_max_diff:.12f}")

        # With float32, errors should be MUCH smaller if implementation is correct
        # If MAE is still ~0.001-0.003, something might be wrong
        print(f"\n{'='*60}")
        if target_mae < 1e-5 and downstream_mae < 1e-5:
            print("✓ Excellent: MAE < 1e-5 in float32")
            print("  This suggests the implementation is correct and float16")
            print("  precision was the limiting factor.")
        elif target_mae < 1e-4:
            print("⚠ Good but not perfect: MAE ~1e-4 to 1e-5")
            print("  Implementation likely correct, but check for subtle issues.")
        else:
            print(f"⚠ WARNING: MAE still high ({target_mae:.6e}) in float32!")
            print("  This suggests there may be a bug in the implementation.")
        print(f"{'='*60}")

        # More lenient assertion for float32 - should be near machine precision
        assert target_cos > 0.99999, (
            f"Target layer: Cosine similarity {target_cos:.12f} should be >0.99999 in float32"
        )
        assert downstream_cos > 0.99999, (
            f"Downstream layer: Cosine similarity {downstream_cos:.12f} should be >0.99999 in float32"
        )

    finally:
        vllm_model.clear_all_vectors()
        del vllm_model
        del hf_model
        torch.cuda.empty_cache()
