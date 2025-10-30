"""CUDA-dependent smoke tests for vLLM steering with Gemma models."""

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
    "google/gemma-2b-it",  # Gemma 1 (Gemma2 requires flash attention with softcapping)
])
@pytest.mark.asyncio
async def test_gemma_vllm_steering_vector_round_trip(model_name: str):
    """Test basic steering vector operations with Gemma models."""

    # Use smaller memory for 2B model
    gpu_mem = 0.2

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
    )

    target_layer = 10  # Middle layer for Gemma

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    hidden_size = model.hidden_size
    vector = torch.randn(hidden_size, dtype=torch.float32)
    await model.set_layer_vector(target_layer, vector)

    # Verify vector was broadcast to workers
    worker_maps = await model._fetch_worker_vectors()
    assert worker_maps, "Expected at least one worker vector."
    for worker_map in worker_maps:
        worker_vec = worker_map[target_layer]
        assert torch.allclose(
            worker_vec, vector.to(dtype=worker_vec.dtype), atol=1e-5
        ), "Broadcast steering vector does not match worker copy."

    # Test projection cap (clear additive first)
    await model.clear_layer_vector(target_layer)
    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    await model.set_layer_projection_cap(target_layer, cap_vector, min=-0.5, max=0.75)

    # Test ablation (clear everything first)
    await model.clear_layer_vector(target_layer)
    await model.clear_layer_projection_cap(target_layer)
    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)
    await model.set_layer_ablation(target_layer, ablation_vector, scale=0.4)

    # Run a generation to trigger patched forward method with all operations
    prompt = "Test"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    await model.generate([prompt], sampling_params=sampling_params, use_tqdm=False)

    # Check that patching works (output_types should include tuple)
    inspection = await model._engine_client.collective_rpc(
        steering_runtime.STEERING_RPC_METHOD,
        args=steering_runtime.rpc_args("inspect_layer_vector", target_layer),
    )
    assert inspection, "Expected inspection data for target layer."
    layer_info = inspection[0]
    output_types = layer_info.get("output_types", [])
    assert "tuple" in output_types, f"Expected tuple output type for Gemma, got {output_types}"

    # Test multi-layer steering
    layer_a, layer_b = target_layer - 1, target_layer + 1
    await model.set_layer_vector(layer_a, torch.randn(hidden_size, dtype=torch.float32))
    await model.set_layer_vector(layer_b, torch.randn(hidden_size, dtype=torch.float32))

    worker_maps = await model._fetch_worker_vectors()
    for worker_map in worker_maps:
        assert layer_a in worker_map
        assert layer_b in worker_map

    del model
    torch.cuda.empty_cache()


@pytest.mark.skip(reason="vLLM V1 engine doesn't populate logprobs for Gemma models")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "google/gemma-2b-it",
])
@pytest.mark.asyncio
async def test_gemma_vllm_chat_respects_steering(model_name: str):
    """Verify chat generation is perturbed by steering vectors."""
    torch.manual_seed(42)
    gpu_mem = 0.2

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=256,
    )

    target_layer = 10

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    prompt = "The capital of France is"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    # Baseline generation
    baseline_result = (await model.generate([prompt], sampling_params, raw_output=True, use_tqdm=False))[0]
    baseline_logprobs = {
        tok: data.logprob
        for tok, data in baseline_result.outputs[0].logprobs[0].items()
    }

    # Apply strong steering vector
    steering_vec = torch.randn(model.hidden_size, dtype=torch.float32) * 100.0
    await model.set_layer_vector(target_layer, steering_vec)

    # Steered generation
    steered_result = (await model.generate([prompt], sampling_params, raw_output=True, use_tqdm=False))[0]
    steered_logprobs = {
        tok: data.logprob
        for tok, data in steered_result.outputs[0].logprobs[0].items()
    }

    # Verify that logprobs changed significantly for at least some tokens
    common_tokens = set(baseline_logprobs.keys()) & set(steered_logprobs.keys())
    assert len(common_tokens) >= 3, "Need at least 3 common tokens for comparison."

    logprob_diffs = [
        abs(baseline_logprobs[tok] - steered_logprobs[tok])
        for tok in common_tokens
    ]
    max_diff = max(logprob_diffs)
    assert max_diff > 0.5, f"Expected significant logprob change, got max diff {max_diff:.3f}"

    # Clear steering and verify reset
    await model.clear_layer_vector(target_layer)
    cleared_result = (await model.generate([prompt], sampling_params, raw_output=True, use_tqdm=False))[0]
    cleared_logprobs = {
        tok: data.logprob
        for tok, data in cleared_result.outputs[0].logprobs[0].items()
    }

    # After clearing, logprobs should be close to baseline
    common_cleared = set(baseline_logprobs.keys()) & set(cleared_logprobs.keys())
    for tok in common_cleared:
        diff = abs(baseline_logprobs[tok] - cleared_logprobs[tok])
        assert diff < 0.01, f"Expected cleared logprobs to match baseline for token {tok}, got diff {diff:.3f}"

    del model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "google/gemma-2b-it",
])
@pytest.mark.asyncio
async def test_gemma_hidden_state_capture(model_name: str):
    """Test that we can capture hidden states before/after steering."""
    gpu_mem = 0.2

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=128,
    )

    target_layer = 10

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Generate with capture
    prompt = "Hello world"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
    texts, handles = await model.generate([prompt], sampling_params=sampling_params, use_tqdm=False, capture_layers=[target_layer])

    # Fetch captured states
    await model.fetch_captures_batch(handles)
    assert len(handles) > 0, "Expected at least one request"
    assert target_layer in handles[0].captures, f"Expected captures for layer {target_layer}"

    layer_captures = handles[0].captures[target_layer]
    assert len(layer_captures) > 0, "Expected at least one capture"

    # Check structure (new API has simpler structure)
    first_capture = layer_captures[0]
    assert "hidden" in first_capture, "Expected 'hidden' key in capture"

    # Check shapes
    hidden = first_capture["hidden"]
    assert isinstance(hidden, torch.Tensor)
    assert hidden.size(-1) == model.hidden_size

    del model
    torch.cuda.empty_cache()


@pytest.mark.skip(reason="vLLM V1 engine doesn't populate logprobs for Gemma models")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.parametrize("model_name", [
    "google/gemma-2b-it",
])
@pytest.mark.asyncio
async def test_gemma_vllm_matches_hf_logprob_shift(model_name: str):
    """Verify that vLLM steering produces similar logprob shifts as HuggingFace baseline."""
    torch.manual_seed(43)
    gpu_mem = 0.2
    target_layer = 10

    # vLLM setup
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=128,
    )

    try:
        vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    hidden_size = vllm_model.hidden_size
    steering_vec = torch.randn(hidden_size, dtype=torch.float32) * 20.0

    prompt = "The capital of Germany is"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    # vLLM baseline
    vllm_baseline = (await vllm_model.generate([prompt], sampling_params, raw_output=True, use_tqdm=False))[0]
    vllm_baseline_logprobs = {
        tok: data.logprob
        for tok, data in vllm_baseline.outputs[0].logprobs[0].items()
    }

    # vLLM steered
    await vllm_model.set_layer_vector(target_layer, steering_vec)
    vllm_steered = (await vllm_model.generate([prompt], sampling_params, raw_output=True, use_tqdm=False))[0]
    vllm_steered_logprobs = {
        tok: data.logprob
        for tok, data in vllm_steered.outputs[0].logprobs[0].items()
    }

    del vllm_model
    torch.cuda.empty_cache()

    # HuggingFace setup
    hf_cfg = SteeringVectorConfig(model_name=model_name, target_layer=target_layer, init_scale=0.0)

    try:
        hf_model = TransformerSteerModel(
            hf_cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    except OSError as exc:
        pytest.skip(f"Unable to load HF model ({exc}). Ensure weights are cached.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # HF baseline
    inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.model.device)
    with torch.no_grad():
        hf_baseline_output = hf_model(**inputs)
    hf_baseline_logits = hf_baseline_output.logits[0, -1, :]

    # HF steered
    with torch.no_grad():
        hf_model.steering.vector.data = steering_vec.to(
            device=hf_model.steering.vector.device,
            dtype=hf_model.steering.vector.dtype,
        )

    with torch.no_grad():
        hf_steered_output = hf_model(**inputs)
    hf_steered_logits = hf_steered_output.logits[0, -1, :]

    # Compare logit shifts
    hf_logit_shift = hf_steered_logits - hf_baseline_logits

    # Get vLLM logprob shifts for the same tokens
    common_tokens = set(vllm_baseline_logprobs.keys()) & set(vllm_steered_logprobs.keys())
    assert len(common_tokens) >= 3, "Need at least 3 common tokens for comparison."

    # Convert logprobs to logit shifts
    for tok_id in list(common_tokens)[:5]:  # Check top 5 tokens
        vllm_shift = vllm_steered_logprobs[tok_id] - vllm_baseline_logprobs[tok_id]
        hf_shift = hf_logit_shift[tok_id].item()

        # Shifts should have same direction (sign)
        assert (vllm_shift * hf_shift) >= 0, (
            f"Token {tok_id} shift direction mismatch: vLLM={vllm_shift:.3f}, HF={hf_shift:.3f}"
        )

        # Magnitudes should be comparable (within 50% relative difference)
        if abs(vllm_shift) > 0.1:  # Only check substantial shifts
            relative_diff = abs(vllm_shift - hf_shift) / abs(vllm_shift)
            assert relative_diff < 0.5, (
                f"Token {tok_id} shift magnitude differs: vLLM={vllm_shift:.3f}, HF={hf_shift:.3f}"
            )

    del hf_model
    torch.cuda.empty_cache()
