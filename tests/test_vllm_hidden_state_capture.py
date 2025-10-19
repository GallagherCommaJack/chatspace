"""Tests for vLLM hidden state capture debug hooks."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig
from chatspace.vllm_steering import runtime as steering_runtime

# vLLM >=0.11 requires enabling pickle-based serialization for custom RPCs.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_state_capture_basic():
    """Test basic hidden state capture enable/disable/fetch."""
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

    # Enable capture
    model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=True)

    # Generate with capture enabled
    prompt = "The quick brown fox"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    model.generate([prompt], sampling_params=sampling)

    # Fetch captured states
    states = model.fetch_hidden_states(layer_idx=target_layer)
    assert len(states) > 0, "Expected at least one worker"
    worker_states = states[0]
    assert target_layer in worker_states, f"Layer {target_layer} not in captured states"

    layer_captures = worker_states[target_layer]
    assert len(layer_captures) > 0, "Expected at least one capture"

    first_capture = layer_captures[0]
    assert "before" in first_capture, "Expected 'before' key in capture"
    assert "after" in first_capture, "Expected 'after' key in capture"

    before_state = first_capture["before"]
    after_state = first_capture["after"]

    # Check shapes match
    assert before_state.shape == after_state.shape, "Before/after shapes should match"
    assert before_state.shape[-1] == model.hidden_size, "Hidden dimension should match model"

    # Clear captured states
    model.clear_hidden_states(target_layer)
    cleared_states = model.fetch_hidden_states(layer_idx=target_layer)
    assert len(cleared_states[0][target_layer]) == 0, "States should be cleared"

    # Disable capture
    model.disable_hidden_state_capture(target_layer)

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_state_capture_verifies_steering():
    """Test that captured states verify steering is actually applied."""
    torch.manual_seed(42)

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

    # Set a steering vector
    steering_vector = torch.randn(model.hidden_size, dtype=torch.float32) * 10.0
    model.set_layer_vector(target_layer, steering_vector)

    # Enable capture
    model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=True)

    # Generate
    prompt = "The capital of France is"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = model.fetch_hidden_states(layer_idx=target_layer)
    layer_captures = states[0][target_layer]
    assert len(layer_captures) > 0, "Expected captures"

    first_capture = layer_captures[0]
    before = first_capture["before"]
    after = first_capture["after"]

    # Compute the difference
    diff = after - before

    # Check that the steering was applied (non-zero difference)
    assert not torch.allclose(before, after, atol=1e-3), "Before and after should differ when steering is applied"

    # Check that the difference is substantial (not just numerical noise)
    # The norm of the difference should be non-trivial
    diff_norm = diff.norm().item()
    assert diff_norm > 1.0, f"Difference norm {diff_norm} should be substantial given steering magnitude"

    # Verify the difference correlates with the steering direction
    # Flatten and compute correlation
    diff_flat = diff.reshape(-1, diff.shape[-1]).mean(dim=0)
    steering_cpu = steering_vector.to(dtype=diff_flat.dtype)

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        diff_flat.unsqueeze(0), steering_cpu.unsqueeze(0)
    ).item()

    # The steering should have a positive effect (correlation > 0)
    # Allow for some variation due to model dynamics
    assert cos_sim > 0.5, f"Cosine similarity {cos_sim} should indicate steering was applied"

    model.clear_all_vectors()
    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_state_capture_max_captures():
    """Test that max_captures limit is respected."""
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

    # Enable capture with max_captures=2
    model.enable_hidden_state_capture(target_layer, max_captures=2)

    # Generate multiple times
    prompt = "Test"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)

    for _ in range(5):
        model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = model.fetch_hidden_states(layer_idx=target_layer)
    layer_captures = states[0][target_layer]

    # Should have at most 2 captures
    assert len(layer_captures) <= 2, f"Expected at most 2 captures, got {len(layer_captures)}"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_state_capture_multiple_layers():
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

    # Enable capture on multiple layers
    model.enable_hidden_state_capture(layers)

    # Generate
    prompt = "Hello world"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    model.generate([prompt], sampling_params=sampling)

    # Fetch all states
    states = model.fetch_hidden_states()
    worker_states = states[0]

    # Check all layers captured
    for layer_idx in layers:
        assert layer_idx in worker_states, f"Layer {layer_idx} should have captures"
        assert len(worker_states[layer_idx]) > 0, f"Layer {layer_idx} should have at least one capture"

    # Disable all layers
    model.disable_hidden_state_capture(layers)

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_state_capture_only_before():
    """Test capturing only before states."""
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

    # Enable capture only for before states
    model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=False)

    # Generate
    prompt = "Test prompt"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = model.fetch_hidden_states(layer_idx=target_layer)
    layer_captures = states[0][target_layer]

    assert len(layer_captures) > 0, "Expected captures"
    first_capture = layer_captures[0]

    assert "before" in first_capture, "Expected 'before' key"
    assert "after" not in first_capture, "Should not have 'after' key"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_state_capture_only_after():
    """Test capturing only after states."""
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

    # Enable capture only for after states
    model.enable_hidden_state_capture(target_layer, capture_before=False, capture_after=True)

    # Generate
    prompt = "Test prompt"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = model.fetch_hidden_states(layer_idx=target_layer)
    layer_captures = states[0][target_layer]

    assert len(layer_captures) > 0, "Expected captures"
    first_capture = layer_captures[0]

    assert "before" not in first_capture, "Should not have 'before' key"
    assert "after" in first_capture, "Expected 'after' key"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_state_capture_disable_all():
    """Test disabling capture for all layers at once."""
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

    # Enable capture on multiple layers
    model.enable_hidden_state_capture(layers)

    # Generate to create some captures
    prompt = "Test"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    model.generate([prompt], sampling_params=sampling)

    # Verify captures exist
    states = model.fetch_hidden_states()
    assert len(states[0]) > 0, "Should have some captures"

    # Disable all at once
    model.disable_hidden_state_capture(None)

    # Generate again
    model.generate([prompt], sampling_params=sampling)

    # Fetch states - should be empty after disable
    states_after = model.fetch_hidden_states()
    worker_states = states_after[0]

    # All layers should either be missing or have empty capture lists
    for layer_idx in layers:
        if layer_idx in worker_states:
            assert len(worker_states[layer_idx]) == 0, f"Layer {layer_idx} should have no captures after disable"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_hidden_states_match_hf():
    """Test that captured hidden states match between HuggingFace and vLLM.

    This test compares the prefill hidden states from both implementations
    to ensure they produce similar representations at the layer level.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 2
    prompt = "The quick brown fox jumps over"

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

    # Capture hidden states from HF model using forward pass (not generate)
    captured_hf_state = None

    def capture_hook(module, args, output):
        """Hook to capture hidden states from HF model."""
        nonlocal captured_hf_state
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Only capture first time (prefill phase)
        if captured_hf_state is None:
            captured_hf_state = hidden_states.detach().cpu().clone()
        return output

    # Install hook on target layer
    layer = hf_model.model.layers[target_layer]
    hook_handle = layer.register_forward_hook(capture_hook)

    # Do a forward pass (not generation) for cleaner comparison
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)

    # Remove hook
    hook_handle.remove()

    # Get the captured hidden state
    assert captured_hf_state is not None, "Should have captured hidden state from HF"
    hf_hidden = captured_hf_state.to(dtype=torch.float32)

    # HF includes batch dimension, squeeze it out for comparison
    # Shape: [batch, seq_len, hidden_size] -> [seq_len, hidden_size]
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

    try:
        vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Enable capture for vLLM (capture before steering, no steering applied)
    vllm_model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=False)

    # Generate with vLLM using the same prompt
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
    vllm_model.generate([prompt], sampling_params=sampling)

    # Fetch captured states
    vllm_states = vllm_model.fetch_hidden_states(layer_idx=target_layer)
    vllm_captures = vllm_states[0][target_layer]

    assert len(vllm_captures) > 0, "Should have captured at least one hidden state from vLLM"
    vllm_hidden = vllm_captures[0]["before"].to(dtype=torch.float32)

    # Compare shapes (should match after squeezing batch dimension)
    assert hf_hidden.shape == vllm_hidden.shape, (
        f"Hidden state shapes should match: HF {hf_hidden.shape} vs vLLM {vllm_hidden.shape}"
    )

    # Compare values - allow for some numerical differences due to implementation details
    # The key metric is cosine similarity, which measures directional similarity
    # Absolute values may differ due to different attention implementations, dtype handling, etc.

    # Flatten for easier comparison
    hf_flat = hf_hidden.reshape(-1)
    vllm_flat = vllm_hidden.reshape(-1)

    # Compute cosine similarity (directional similarity)
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_flat.unsqueeze(0), vllm_flat.unsqueeze(0)
    ).item()

    # Cosine similarity should be very high (>0.99)
    # This is the primary metric - it shows the hidden states point in the same direction
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim} should be >0.99, indicating similar representations"

    # Compute Pearson correlation for additional validation
    hf_centered = hf_flat - hf_flat.mean()
    vllm_centered = vllm_flat - vllm_flat.mean()
    correlation = (hf_centered * vllm_centered).sum() / (
        hf_centered.norm() * vllm_centered.norm() + 1e-8
    )
    correlation = correlation.item()

    # Correlation should also be very high
    assert correlation > 0.99, f"Pearson correlation {correlation} should be >0.99"

    # Check that the shapes and basic statistics are similar
    hf_mean = hf_flat.mean().item()
    vllm_mean = vllm_flat.mean().item()
    hf_std = hf_flat.std().item()
    vllm_std = vllm_flat.std().item()

    # Print diagnostic info
    print(f"\nHidden state comparison:")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Pearson correlation: {correlation:.6f}")
    print(f"  HF mean: {hf_mean:.4f}, std: {hf_std:.4f}")
    print(f"  vLLM mean: {vllm_mean:.4f}, std: {vllm_std:.4f}")
    print(f"  Shape: {hf_hidden.shape}")

    # The test passes if cosine similarity and correlation are both very high
    # This indicates the models are producing similar hidden representations

    # Clean up
    vllm_model.clear_all_vectors()
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()
