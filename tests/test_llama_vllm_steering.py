"""CUDA-dependent smoke tests for vLLM steering with Llama models."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import (
    VLLMSteerModel,
    VLLMSteeringConfig,
)
from chatspace.steering.model import TransformerSteerModel, SteeringVectorConfig
from chatspace.vllm_steering import runtime as steering_runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "meta-llama/Llama-3.2-1B-Instruct",
])
def test_llama_vllm_steering_vector_round_trip(model_name: str):
    """Test basic steering vector operations with Llama models."""

    # Use smaller memory for 1B model, more for 70B
    gpu_mem = 0.1 if "1B" in model_name else 0.9

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
    )

    target_layer = 4  # Middle layer for 16-layer model

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    hidden_size = model.hidden_size
    vector = torch.randn(hidden_size, dtype=torch.float32)
    model.set_layer_vector(target_layer, vector)

    # Verify vector was broadcast to workers
    worker_maps = model._fetch_worker_vectors()
    assert worker_maps, "Expected at least one worker vector."
    for worker_map in worker_maps:
        worker_vec = worker_map[target_layer]
        assert torch.allclose(
            worker_vec, vector.to(dtype=worker_vec.dtype), atol=1e-5
        ), "Broadcast steering vector does not match worker copy."

    # Test projection cap
    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    model.set_layer_projection_cap(target_layer, cap_vector, min=-0.5, max=0.75)

    # Test ablation
    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)
    model.set_layer_ablation(target_layer, ablation_vector, scale=0.4)

    # Verify projection cap and ablation were set
    inspection = model._engine_client.collective_rpc(
        steering_runtime.STEERING_RPC_METHOD,
        args=steering_runtime.rpc_args("inspect_layer_vector", target_layer),
    )
    assert inspection, "Expected inspection data for target layer."
    layer_info = inspection[0]

    # Check output_types includes tuple (Llama uses tuple output)
    output_types = layer_info.get("output_types", [])
    assert "tuple" in output_types, f"Expected tuple output type for Llama, got {output_types}"

    projection_cap = layer_info.get("projection_cap")
    assert projection_cap is not None
    assert projection_cap["min"] == pytest.approx(-0.5)
    assert projection_cap["max"] == pytest.approx(0.75)

    ablation_info = layer_info.get("ablation")
    assert ablation_info is not None
    assert ablation_info["scale"] == pytest.approx(0.4)

    # Clear and verify
    model.clear_layer_projection_cap(target_layer)
    model.clear_layer_ablation(target_layer)
    inspection_after_clear = model._engine_client.collective_rpc(
        steering_runtime.STEERING_RPC_METHOD,
        args=steering_runtime.rpc_args("inspect_layer_vector", target_layer),
    )
    assert inspection_after_clear, "Expected inspection after clearing."
    layer_info_after_clear = inspection_after_clear[0]
    assert layer_info_after_clear.get("projection_cap") is None
    assert layer_info_after_clear.get("ablation") is None

    # Test multi-layer steering
    worker_state = model._engine_client.collective_rpc(
        steering_runtime.STEERING_RPC_METHOD,
        args=steering_runtime.rpc_args("fetch_worker_state"),
    )
    assert worker_state, "Expected worker state info."
    layer_count = int(worker_state[0].get("layer_count", 0) or 0)
    if layer_count > 1:
        other_layer = (target_layer + 2) % layer_count
        extra_vector = torch.randn(hidden_size, dtype=torch.float32)
        model.set_layer_vector(other_layer, extra_vector)
        expanded_maps = model._fetch_worker_vectors()
        for worker_map in expanded_maps:
            assert other_layer in worker_map, "Missing secondary layer vector."
            worker_vec = worker_map[other_layer]
            assert torch.allclose(
                worker_vec, extra_vector.to(dtype=worker_vec.dtype), atol=1e-5
            ), "Secondary layer vector mismatch."

    # Clear all vectors
    model.clear_all_vectors()
    cleared_maps = model._fetch_worker_vectors()
    for worker_map in cleared_maps:
        for worker_vec in worker_map.values():
            assert torch.allclose(worker_vec, torch.zeros_like(worker_vec))

    # Clean up to avoid leaving GPU memory allocated between tests.
    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "meta-llama/Llama-3.2-1B-Instruct",
])
def test_llama_vllm_chat_respects_steering(model_name: str):
    """Test that steering vectors actually modify Llama model outputs."""
    torch.manual_seed(0)

    gpu_mem = 0.1 if "1B" in model_name else 0.9

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
    )

    target_layer = 4

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    prompt = "The capital of France is"
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    # Baseline generation
    baseline_req = model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    baseline_out = baseline_req.outputs[0]
    baseline_token = baseline_out.token_ids[0]
    baseline_logprob = baseline_out.logprobs[0][baseline_token].logprob

    # Generate with strong steering
    scale = 5_000.0
    random_vector = torch.randn(model.hidden_size, dtype=torch.float32) * scale
    model.set_layer_vector(target_layer, random_vector)

    # Verify steering was applied
    worker_info = model._engine_client.collective_rpc(
        steering_runtime.STEERING_RPC_METHOD,
        args=steering_runtime.rpc_args("inspect_layer_vector", target_layer),
    )
    assert worker_info, "Expected worker diagnostics."
    assert worker_info[0]["has_vector"], "Patched layer missing steering vector."
    assert worker_info[0]["norm"] > 0, "Worker vector norm is zero unexpectedly."

    # Generate with steering
    steered_req = model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    steered_out = steered_req.outputs[0]
    steered_token = steered_out.token_ids[0]
    steered_logprob = steered_out.logprobs[0][steered_token].logprob

    # Clear and regenerate
    model.clear_all_vectors()
    reset_req = model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    reset_out = reset_req.outputs[0]
    reset_token = reset_out.token_ids[0]
    reset_logprob = reset_out.logprobs[0][reset_token].logprob

    # Test chat interface
    request = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_sampling = SamplingParams(temperature=0.0, max_tokens=4)
    chat_response = model.chat(request, sampling_params=chat_sampling)[0]

    # Verify steering had an effect
    assert not torch.isclose(
        torch.tensor(baseline_logprob), torch.tensor(steered_logprob)
    ), (
        "Steering did not perturb token logprobs. "
        f"worker_info={worker_info}"
    )

    # Verify clearing restored baseline
    assert torch.isclose(
        torch.tensor(baseline_logprob), torch.tensor(reset_logprob), atol=1e-6
    ), "Clearing the steering vector should restore baseline behaviour."

    # Verify chat works
    assert isinstance(chat_response, str) and len(chat_response) > 0

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "meta-llama/Llama-3.2-1B-Instruct",
])
def test_llama_hidden_state_capture(model_name: str):
    """Test hidden state capture functionality with Llama models."""
    torch.manual_seed(42)

    gpu_mem = 0.1 if "1B" in model_name else 0.9

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
    )

    target_layer = 4

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Enable capture
    model.enable_hidden_state_capture(
        target_layer,
        capture_before=True,
        capture_after=True,
        max_captures=3
    )

    # Apply steering and generate
    vector = torch.randn(model.hidden_size, dtype=torch.float32) * 0.5
    model.set_layer_vector(target_layer, vector)

    prompt = "Once upon a time"
    sampling = SamplingParams(temperature=0.0, max_tokens=5)
    _ = model.generate([prompt], sampling)

    # Fetch captured states
    states = model.fetch_hidden_states(layer_idx=target_layer)
    assert states, "Expected captured states"
    assert target_layer in states[0], f"Expected layer {target_layer} in captured states"

    captures = states[0][target_layer]
    assert len(captures) > 0, "Expected at least one capture"

    # Verify capture structure
    first_capture = captures[0]
    assert "before" in first_capture, "Expected 'before' in capture"
    assert "after" in first_capture, "Expected 'after' in capture"
    assert "meta" in first_capture, "Expected 'meta' in capture"

    # Verify shapes
    before = first_capture["before"]
    after = first_capture["after"]
    assert before.shape[-1] == model.hidden_size, "Before shape mismatch"
    assert after.shape[-1] == model.hidden_size, "After shape mismatch"

    # Verify steering had an effect
    assert not torch.allclose(before, after, atol=1e-3), "Steering should modify hidden states"

    # Test clearing
    model.clear_hidden_states(target_layer)
    cleared_states = model.fetch_hidden_states(layer_idx=target_layer)
    assert len(cleared_states[0][target_layer]) == 0, "Expected empty captures after clear"

    # Disable capture
    model.disable_hidden_state_capture(target_layer)

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering and HF baseline.")
@pytest.mark.parametrize("model_name", [
    "meta-llama/Llama-3.2-1B-Instruct",
])
def test_llama_vllm_matches_hf_logprob_shift(model_name: str):
    """Test that vLLM and HuggingFace Llama steering produce similar logprob shifts."""
    torch.manual_seed(42)
    target_layer = 4
    prompt = "In a quiet village, the baker"

    try:
        config = AutoConfig.from_pretrained(model_name)
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"Unable to load config ({exc}). Ensure weights are cached.")

    hidden_size = int(config.hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # HuggingFace baseline
    hf_cfg = SteeringVectorConfig(
        model_name=model_name,
        target_layer=target_layer,
        init_scale=0.0
    )

    try:
        hf_model = TransformerSteerModel(
            hf_cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"Unable to load HF model ({exc}). Ensure weights are cached.")
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"Missing optional dependency ({exc}).")

    with torch.no_grad():
        hf_baseline_outputs = hf_model(**inputs, use_cache=False)
        hf_baseline_logits = hf_baseline_outputs.logits
        hf_baseline_next_token_logprobs = torch.log_softmax(hf_baseline_logits[:, -1, :], dim=-1)

        steering_vector = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 100.0
        hf_model.set_vector(steering_vector)
        hf_steered_outputs = hf_model(**inputs, use_cache=False)
        hf_steered_logits = hf_steered_outputs.logits
        hf_steered_next_token_logprobs = torch.log_softmax(hf_steered_logits[:, -1, :], dim=-1)

        hf_logprob_shift = (hf_steered_next_token_logprobs - hf_baseline_next_token_logprobs).abs().max().item()

    del hf_model
    torch.cuda.empty_cache()

    # vLLM implementation
    gpu_mem = 0.1 if "1B" in model_name else 0.9
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
        dtype="bfloat16",
    )

    try:
        vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"Unable to load vLLM model ({exc}). Ensure weights are cached.")

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=100)

    vllm_baseline_req = vllm_model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    vllm_baseline_logprobs = vllm_baseline_req.outputs[0].logprobs[0]

    vllm_model.set_layer_vector(target_layer, steering_vector)
    vllm_steered_req = vllm_model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    vllm_steered_logprobs = vllm_steered_req.outputs[0].logprobs[0]

    # Calculate vLLM logprob shift
    common_tokens = set(vllm_baseline_logprobs.keys()) & set(vllm_steered_logprobs.keys())
    vllm_shifts = [
        abs(vllm_steered_logprobs[tok].logprob - vllm_baseline_logprobs[tok].logprob)
        for tok in common_tokens
    ]
    vllm_logprob_shift = max(vllm_shifts) if vllm_shifts else 0.0

    del vllm_model
    torch.cuda.empty_cache()

    # Verify both implementations show similar steering effects
    assert hf_logprob_shift > 0.1, f"HF steering too weak: {hf_logprob_shift}"
    assert vllm_logprob_shift > 0.1, f"vLLM steering too weak: {vllm_logprob_shift}"

    # Allow some divergence due to implementation differences, but should be similar order of magnitude
    ratio = vllm_logprob_shift / hf_logprob_shift if hf_logprob_shift > 0 else float('inf')
    assert 0.1 < ratio < 10.0, (
        f"vLLM and HF logprob shifts differ too much. "
        f"HF: {hf_logprob_shift:.4f}, vLLM: {vllm_logprob_shift:.4f}, ratio: {ratio:.4f}"
    )
