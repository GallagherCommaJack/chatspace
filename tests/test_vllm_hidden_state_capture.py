"""Tests for vLLM hidden state capture using CaptureHandle API."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_hidden_state_capture_basic():
    """Test basic hidden state capture with CaptureHandle API."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Generate with capture
    prompt = "The quick brown fox"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    texts, handles = await model.generate([prompt], sampling_params=sampling, capture_layers=target_layer)

    # Verify we got a handle
    assert len(handles) == 1, "Expected one capture handle"
    handle = handles[0]
    assert handle.request_id.startswith("capture_"), "Expected capture request ID"
    assert target_layer in handle.layer_indices, f"Expected layer {target_layer} in handle"

    # Fetch captured states
    await handle.fetch()
    captures_dict = handle.captures

    assert target_layer in captures_dict, f"Layer {target_layer} not in captured states"
    layer_captures = captures_dict[target_layer]
    assert len(layer_captures) > 0, "Expected at least one capture"

    first_capture = layer_captures[0]
    assert "hidden" in first_capture, "Expected 'hidden' key in capture"

    hidden_state = first_capture["hidden"]
    assert hidden_state.shape[-1] == model.hidden_size, "Hidden dimension should match model"
    assert hidden_state.dtype in (torch.float16, torch.bfloat16, torch.float32), "Expected float dtype"

    # Test idempotent fetch
    await handle.fetch()
    assert handle.captures is captures_dict, "Fetch should be idempotent"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_hidden_state_capture_multiple_layers():
    """Test capturing from multiple layers simultaneously."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    layers = [1, 2, 3]

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=layers)
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Generate with capture on multiple layers
    prompt = "Hello world"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    texts, handles = await model.generate([prompt], sampling_params=sampling, capture_layers=layers)

    # Verify we got a handle
    assert len(handles) == 1, "Expected one capture handle"
    handle = handles[0]

    # Fetch captured states
    await handle.fetch()
    captures_dict = handle.captures

    # Check all layers captured
    for layer_idx in layers:
        assert layer_idx in captures_dict, f"Layer {layer_idx} should have captures"
        layer_captures = captures_dict[layer_idx]
        assert len(layer_captures) > 0, f"Layer {layer_idx} should have at least one capture"

        # Verify hidden state shape
        hidden_state = layer_captures[0]["hidden"]
        assert hidden_state.shape[-1] == model.hidden_size

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_hidden_state_capture_batch_fetch():
    """Test batch fetching multiple capture handles efficiently."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    target_layer = 2

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Generate multiple requests with capture
    prompts = ["First prompt", "Second prompt", "Third prompt"]
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    texts, handles = await model.generate(prompts, sampling_params=sampling, capture_layers=target_layer)

    # Verify we got handles for all prompts
    assert len(handles) == len(prompts), f"Expected {len(prompts)} handles"

    # Batch fetch all handles at once
    await model.fetch_captures_batch(handles)

    # Verify all handles have captures
    for i, handle in enumerate(handles):
        assert handle._captures is not None, f"Handle {i} should have captures after batch fetch"
        assert target_layer in handle.captures, f"Handle {i} should have layer {target_layer}"
        assert len(handle.captures[target_layer]) > 0, f"Handle {i} should have captures"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_hidden_state_capture_no_capture():
    """Test that generate() without capture_layers works normally."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True)
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Generate without capture
    prompt = "Test prompt"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    result = await model.generate([prompt], sampling_params=sampling)

    # Should return just texts, not tuple
    assert isinstance(result, list), "Without capture_layers, should return list of texts"
    assert len(result) == 1, "Should have one text output"
    assert isinstance(result[0], str), "Output should be string"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_hidden_states_match_hf():
    """Test that captured hidden states match between HuggingFace and vLLM.

    This test compares the prefill hidden states from both implementations
    to ensure they produce similar representations at the layer level.

    Note: HuggingFace hidden_states[0] is the embedding output, and hidden_states[i+1]
    corresponds to decoder layer i's output. This must be accounted for when comparing.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 2
    prompt = "The quick brown fox jumps over"

    # Load HuggingFace model
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
        )
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"Unable to load HF model ({exc}). Ensure weights are cached.")

    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Capture HF hidden states
    with torch.no_grad():
        hf_outputs = hf_model(**inputs, output_hidden_states=True, use_cache=False)
        hf_hidden = hf_outputs.hidden_states[target_layer + 1]  # [batch, seq_len, hidden_size]
        # Note: hidden_states[0] is embedding output, hidden_states[i+1] is layer i output
        # Take last token's hidden state
        hf_last_token = hf_hidden[0, -1, :].cpu()

    del hf_model
    torch.cuda.empty_cache()

    # Load vLLM model
    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
        dtype="float16",
    )

    try:
        vllm_model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"Unable to load vLLM model ({exc}). Ensure weights are cached.")

    # Generate with capture
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
    texts, handles = await vllm_model.generate([prompt], sampling_params=sampling, capture_layers=target_layer)

    # Fetch vLLM hidden states
    await handles[0].fetch()
    vllm_captures = handles[0].captures[target_layer]

    # Extract hidden states from prefill
    assert len(vllm_captures) > 0, "Expected at least one capture"
    vllm_hidden = vllm_captures[0]["hidden"]  # [seq_len, hidden_size]
    vllm_hidden_full = vllm_hidden.cpu()
    vllm_last_token = vllm_hidden[-1, :].cpu()

    del vllm_model
    torch.cuda.empty_cache()

    # Verify shapes match
    hf_full = hf_hidden[0, :, :].cpu()  # [seq_len, hidden_size]
    assert hf_full.shape == vllm_hidden_full.shape, (
        f"Shape mismatch: HF {hf_full.shape} vs vLLM {vllm_hidden_full.shape}"
    )

    # Compare full hidden state sequence
    full_cos_sim = torch.nn.functional.cosine_similarity(
        hf_full.reshape(-1).unsqueeze(0),
        vllm_hidden_full.reshape(-1).unsqueeze(0)
    ).item()
    assert full_cos_sim > 0.999, (
        f"Full sequence cosine similarity {full_cos_sim:.6f} too low. "
        f"Expected >0.999 for float16 parity."
    )

    full_mae = (hf_full - vllm_hidden_full).abs().mean().item()
    assert full_mae < 1e-3, (
        f"Full sequence MAE {full_mae:.6f} too large. Expected <1e-3 for float16."
    )

    # Compare last token specifically (most important for generation)
    last_cos_sim = torch.nn.functional.cosine_similarity(
        hf_last_token.unsqueeze(0),
        vllm_last_token.unsqueeze(0)
    ).item()
    assert last_cos_sim > 0.999, (
        f"Last token cosine similarity {last_cos_sim:.6f} too low. "
        f"HF and vLLM outputs should match closely."
    )

    last_mae = (hf_last_token - vllm_last_token).abs().mean().item()
    assert last_mae < 1e-3, (
        f"Last token MAE {last_mae:.6f} too large. Expected <1e-3 for float16 parity."
    )

    last_rmse = torch.nn.functional.mse_loss(hf_last_token, vllm_last_token).sqrt().item()
    assert last_rmse < 2e-3, (
        f"Last token RMSE {last_rmse:.6f} too large. Expected <2e-3 for float16 parity."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
async def test_hidden_states_with_steering_applied():
    """Test that HF and vLLM produce similar hidden states when steering is applied.

    This test verifies that steering produces consistent effects on hidden states
    at the steering layer in both implementations. We compare:
    1. The modified hidden states after steering is applied
    2. The difference between baseline and steered states (the steering effect itself)

    Note: Uses forward hooks to capture HF states, as output_hidden_states captures
    before steering hooks run.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 2
    prompt = "The capital of France is"

    # Load HuggingFace model
    try:
        from chatspace.steering.model import TransformerSteerModel, SteeringVectorConfig

        hf_cfg = SteeringVectorConfig(
            model_name=model_name,
            target_layer=target_layer,
            init_scale=0.0
        )
        hf_model = TransformerSteerModel(
            hf_cfg,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
        )
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"Unable to load HF model ({exc}). Ensure weights are cached.")

    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate steering vector
    hidden_size = hf_model.config.hidden_size
    steering_vector = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 5.0

    # Capture hidden states using hooks (output_hidden_states doesn't see steering hook output)
    captured_states = []

    def capture_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured_states.append(hidden[0, -1, :].detach().cpu())

    hook_handle = hf_model.model.model.layers[target_layer].register_forward_hook(capture_hook)

    # Get HF baseline hidden states
    with torch.no_grad():
        captured_states.clear()
        hf_baseline_outputs = hf_model(**inputs, use_cache=False)
        hf_baseline_hidden = captured_states[0]

        # Apply steering and get modified hidden states
        hf_model.set_vector(steering_vector)
        captured_states.clear()
        hf_steered_outputs = hf_model(**inputs, use_cache=False)
        hf_steered_hidden = captured_states[0]

    hook_handle.remove()

    hf_steering_effect = hf_steered_hidden - hf_baseline_hidden

    del hf_model
    torch.cuda.empty_cache()

    # Load vLLM model
    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
        dtype="float16",
    )

    try:
        vllm_model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"Unable to load vLLM model ({exc}). Ensure weights are cached.")

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    # Get vLLM baseline
    texts_baseline, handles_baseline = await vllm_model.generate(
        [prompt], sampling_params=sampling, capture_layers=target_layer
    )
    await handles_baseline[0].fetch()
    vllm_baseline_hidden = handles_baseline[0].captures[target_layer][0]["hidden"][-1, :].cpu()

    # Apply steering and get modified hidden states
    await vllm_model.set_layer_vector(target_layer, steering_vector)
    texts_steered, handles_steered = await vllm_model.generate(
        [prompt], sampling_params=sampling, capture_layers=target_layer
    )
    await handles_steered[0].fetch()
    vllm_steered_hidden = handles_steered[0].captures[target_layer][0]["hidden"][-1, :].cpu()

    vllm_steering_effect = vllm_steered_hidden - vllm_baseline_hidden

    del vllm_model
    torch.cuda.empty_cache()

    # Compare steered hidden states
    steered_cos_sim = torch.nn.functional.cosine_similarity(
        hf_steered_hidden.unsqueeze(0),
        vllm_steered_hidden.unsqueeze(0)
    ).item()

    # With steering applied, we expect slightly lower similarity due to implementation differences
    # in how steering is applied, but should still be very close
    assert steered_cos_sim > 0.99, (
        f"Steered hidden state cos_sim {steered_cos_sim:.6f} too low. "
        f"Expected >0.99 with steering applied."
    )

    # Compare the steering effect itself (difference vectors)
    effect_cos_sim = torch.nn.functional.cosine_similarity(
        hf_steering_effect.unsqueeze(0),
        vllm_steering_effect.unsqueeze(0)
    ).item()

    # The steering effect direction should be very similar
    assert effect_cos_sim > 0.95, (
        f"Steering effect cos_sim {effect_cos_sim:.6f} too low. "
        f"HF and vLLM steering mechanisms should produce similar effects. "
        f"HF effect norm: {hf_steering_effect.norm():.4f}, "
        f"vLLM effect norm: {vllm_steering_effect.norm():.4f}"
    )
