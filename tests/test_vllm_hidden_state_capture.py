"""Tests for vLLM hidden state capture debug hooks."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig
from chatspace.vllm_steering import runtime as steering_runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_state_capture_basic():
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
    await model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=True)

    # Generate with capture enabled
    prompt = "The quick brown fox"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    await model.generate([prompt], sampling_params=sampling)

    # Fetch captured states
    states = await model.fetch_hidden_states(layer_idx=target_layer)
    assert len(states) > 0, "Expected at least one worker"
    worker_states = states[0]
    assert target_layer in worker_states, f"Layer {target_layer} not in captured states"

    layer_captures = worker_states[target_layer]
    assert len(layer_captures) > 0, "Expected at least one capture"

    first_capture = layer_captures[0]
    assert "before" in first_capture, "Expected 'before' key in capture"
    assert "after" in first_capture, "Expected 'after' key in capture"
    assert "meta" in first_capture, "Expected capture metadata"

    before_state = first_capture["before"]
    after_state = first_capture["after"]

    # Check shapes match
    assert before_state.shape == after_state.shape, "Before/after shapes should match"
    assert before_state.shape[-1] == model.hidden_size, "Hidden dimension should match model"

    meta = first_capture["meta"]
    assert meta["phase"] == "prefill", "First capture should correspond to prefill phase"
    assert meta["step"] == 0, "First capture should have step index 0"

    # Clear captured states
    model.clear_hidden_states(target_layer)
    cleared_states = await model.fetch_hidden_states(layer_idx=target_layer)
    assert len(cleared_states[0][target_layer]) == 0, "States should be cleared"

    # Generate again to ensure counters reset after clearing
    await model.generate([prompt], sampling_params=sampling)
    refreshed_states = await model.fetch_hidden_states(layer_idx=target_layer)
    refreshed_capture = refreshed_states[0][target_layer][0]
    assert refreshed_capture["meta"]["step"] == 0, "Counters should reset after clearing"

    # Enable projection cap and verify cap delta instrumentation is present
    unit = torch.zeros(model.hidden_size, dtype=torch.float32)
    unit[0] = 1.0
    model.set_layer_projection_cap(target_layer, unit, max=0.0)
    model.clear_hidden_states(target_layer)
    await model.generate([prompt], sampling_params=sampling)
    capped_states = await model.fetch_hidden_states(layer_idx=target_layer)
    capped_capture = capped_states[0][target_layer][0]
    assert "cap_delta" in capped_capture, "Expected projection delta in capture entry"
    assert "cap_meta" in capped_capture, "Expected projection delta metadata"
    model.clear_layer_projection_cap(target_layer)

    # Disable capture
    await model.disable_hidden_state_capture(target_layer)

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_state_capture_verifies_steering():
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
    await model.set_layer_vector(target_layer, steering_vector)

    # Enable capture
    await model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=True)

    # Generate
    prompt = "The capital of France is"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    await model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = await model.fetch_hidden_states(layer_idx=target_layer)
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

    await model.clear_all_vectors()
    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_state_capture_max_captures():
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
    await model.enable_hidden_state_capture(target_layer, max_captures=2)

    # Generate multiple times
    prompt = "Test"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)

    for _ in range(5):
        await model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = await model.fetch_hidden_states(layer_idx=target_layer)
    layer_captures = states[0][target_layer]

    # Should have at most 2 captures
    assert len(layer_captures) <= 2, f"Expected at most 2 captures, got {len(layer_captures)}"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
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

    # Enable capture on multiple layers
    await model.enable_hidden_state_capture(layers)

    # Generate
    prompt = "Hello world"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    await model.generate([prompt], sampling_params=sampling)

    # Fetch all states
    states = await model.fetch_hidden_states()
    worker_states = states[0]

    # Check all layers captured
    for layer_idx in layers:
        assert layer_idx in worker_states, f"Layer {layer_idx} should have captures"
        assert len(worker_states[layer_idx]) > 0, f"Layer {layer_idx} should have at least one capture"

    # Disable all layers
    await model.disable_hidden_state_capture(layers)

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_state_capture_fetch_multiple_layers_subset():
    """Test fetching hidden states for a subset of layers in a single RPC."""
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=128,
    )

    layers = [1, 2, 3]
    subset = [layers[0], layers[2]]

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=layers)
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    await model.enable_hidden_state_capture(layers)

    prompt = "Subset fetch test"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    await model.generate([prompt], sampling_params=sampling)

    states = await model.fetch_hidden_states(layer_idx=subset)
    assert len(states) > 0, "Expected at least one worker payload"

    worker_states = states[0]
    assert set(worker_states.keys()) == set(subset), "Fetch should only include requested layers"

    for layer_idx in subset:
        captures = worker_states[layer_idx]
        assert len(captures) > 0, f"Layer {layer_idx} should include captures"

    await model.disable_hidden_state_capture(layers)
    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_state_capture_only_before():
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
    await model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=False)

    # Generate
    prompt = "Test prompt"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    await model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = await model.fetch_hidden_states(layer_idx=target_layer)
    layer_captures = states[0][target_layer]

    assert len(layer_captures) > 0, "Expected captures"
    first_capture = layer_captures[0]

    assert "before" in first_capture, "Expected 'before' key"
    assert "after" not in first_capture, "Should not have 'after' key"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_state_capture_only_after():
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
    await model.enable_hidden_state_capture(target_layer, capture_before=False, capture_after=True)

    # Generate
    prompt = "Test prompt"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    await model.generate([prompt], sampling_params=sampling)

    # Fetch states
    states = await model.fetch_hidden_states(layer_idx=target_layer)
    layer_captures = states[0][target_layer]

    assert len(layer_captures) > 0, "Expected captures"
    first_capture = layer_captures[0]

    assert "before" not in first_capture, "Should not have 'before' key"
    assert "after" in first_capture, "Expected 'after' key"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_state_capture_disable_all():
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
    await model.enable_hidden_state_capture(layers)

    # Generate to create some captures
    prompt = "Test"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    await model.generate([prompt], sampling_params=sampling)

    # Verify captures exist
    states = await model.fetch_hidden_states()
    assert len(states[0]) > 0, "Should have some captures"

    # Disable all at once
    await model.disable_hidden_state_capture(None)

    # Generate again
    await model.generate([prompt], sampling_params=sampling)

    # Fetch states - should be empty after disable
    states_after = await model.fetch_hidden_states()
    worker_states = states_after[0]

    # All layers should either be missing or have empty capture lists
    for layer_idx in layers:
        if layer_idx in worker_states:
            assert len(worker_states[layer_idx]) == 0, f"Layer {layer_idx} should have no captures after disable"

    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_states_match_hf():
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
    await vllm_model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=False)

    # Generate with vLLM using the same prompt
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)
    await vllm_model.generate([prompt], sampling_params=sampling)

    # Fetch captured states
    vllm_states = await vllm_model.fetch_hidden_states(layer_idx=target_layer)
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
    await vllm_model.clear_all_vectors()
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_hidden_states_match_hf_decode_phase():
    """Test that captured hidden states match between HuggingFace and vLLM during decode phase.

    This test validates that the hidden state capture fix works correctly during
    autoregressive generation (decode phase), not just prefill. It generates multiple
    tokens and compares the hidden states at each decode step.
    """
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 2
    prompt = "The quick brown fox"
    num_decode_tokens = 5  # Generate 5 tokens during decode phase

    # -------------------------------------------------------------------------
    # HuggingFace path: Manual autoregressive generation with KV cache
    # -------------------------------------------------------------------------
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

    # Perform manual autoregressive generation to capture decode-phase hidden states
    past_key_values = None
    next_token_ids = inputs.input_ids
    hf_decode_hiddens = []

    for step in range(num_decode_tokens + 1):  # +1 to include prefill
        captured_hidden = None

        def capture_hook(module, args, output):
            """Hook to capture hidden states from HF model."""
            nonlocal captured_hidden
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Capture the last token's hidden state (the new token in decode phase)
            captured_hidden = hidden_states[:, -1:, :].detach().cpu().clone()
            return output

        # Install hook on target layer
        layer = hf_model.model.layers[target_layer]
        hook_handle = layer.register_forward_hook(capture_hook)

        # Forward pass with KV cache
        with torch.no_grad():
            outputs = hf_model(
                next_token_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        # Remove hook
        hook_handle.remove()

        # Store hidden state (skip first iteration which is prefill)
        if past_key_values is not None:
            assert captured_hidden is not None, f"Should have captured hidden state at step {step}"
            hf_decode_hiddens.append(captured_hidden.to(dtype=torch.float32))

        # Get next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Update KV cache for next iteration
        past_key_values = outputs.past_key_values

    assert len(hf_decode_hiddens) == num_decode_tokens, (
        f"Should have {num_decode_tokens} decode hidden states, got {len(hf_decode_hiddens)}"
    )

    # -------------------------------------------------------------------------
    # vLLM path: Generate all tokens at once
    # -------------------------------------------------------------------------
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=len(tokenizer.encode(prompt)) + num_decode_tokens + 16,
        dtype="float16",
    )

    try:
        vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    # Enable capture for vLLM (capture before steering)
    await vllm_model.enable_hidden_state_capture(target_layer, capture_before=True, capture_after=False)

    # Generate with vLLM using the same prompt (ignore EOS to ensure we generate the requested number)
    sampling = SamplingParams(temperature=0.0, max_tokens=num_decode_tokens, logprobs=0, ignore_eos=True)
    await vllm_model.generate([prompt], sampling_params=sampling)

    # Fetch captured states
    vllm_states = await vllm_model.fetch_hidden_states(layer_idx=target_layer)
    vllm_captures = vllm_states[0][target_layer]

    # vLLM captures: [0] = prefill, [1:] = decode steps
    # Determine actual number of decode tokens generated
    num_vllm_decode_tokens = len(vllm_captures) - 1  # Subtract prefill capture

    print(f"\nvLLM generated {num_vllm_decode_tokens} decode tokens (captures: {len(vllm_captures)} total)")

    # We need at least some decode tokens to validate
    assert num_vllm_decode_tokens >= 3, (
        f"Need at least 3 decode tokens for meaningful comparison, got {num_vllm_decode_tokens}"
    )

    # Use the minimum of what we generated
    num_tokens_to_compare = min(num_decode_tokens, num_vllm_decode_tokens)

    # Prefill metadata should report phase information
    prefill_meta = vllm_captures[0]["meta"]
    assert prefill_meta["phase"] == "prefill"
    assert prefill_meta["step"] == 0

    # -------------------------------------------------------------------------
    # Compare decode-phase hidden states
    # -------------------------------------------------------------------------
    print(f"\nDecode phase hidden state comparison ({num_tokens_to_compare} tokens):")

    for step_idx in range(num_tokens_to_compare):
        # HF: hidden state from decode step
        hf_hidden = hf_decode_hiddens[step_idx].squeeze(0).squeeze(0)  # Remove batch and seq dims

        # vLLM: hidden state from decode step (+1 to skip prefill)
        capture_entry = vllm_captures[step_idx + 1]
        capture_meta = capture_entry["meta"]
        assert capture_meta["phase"] == "decode", "Decode captures should be tagged as decode"
        assert capture_meta["phase_index"] == step_idx, "Decode phase index should align with step"
        vllm_hidden = capture_entry["before"].to(dtype=torch.float32)

        # vLLM decode states are typically (1, hidden_size), squeeze to (hidden_size,)
        if vllm_hidden.dim() > 1:
            vllm_hidden = vllm_hidden.squeeze(0)

        # Ensure shapes match
        assert hf_hidden.shape == vllm_hidden.shape, (
            f"Step {step_idx}: Shape mismatch - HF {hf_hidden.shape} vs vLLM {vllm_hidden.shape}"
        )

        # Flatten for comparison
        hf_flat = hf_hidden.reshape(-1)
        vllm_flat = vllm_hidden.reshape(-1)

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            hf_flat.unsqueeze(0), vllm_flat.unsqueeze(0)
        ).item()

        # Compute mean absolute error
        mae = torch.mean(torch.abs(hf_flat - vllm_flat)).item()

        # Statistics
        hf_mean = hf_flat.mean().item()
        vllm_mean = vllm_flat.mean().item()
        hf_std = hf_flat.std().item()
        vllm_std = vllm_flat.std().item()

        print(f"  Step {step_idx}:")
        print(f"    Cosine similarity: {cos_sim:.6f}")
        print(f"    MAE: {mae:.6f}")
        print(f"    HF mean: {hf_mean:.4f}, std: {hf_std:.4f}")
        print(f"    vLLM mean: {vllm_mean:.4f}, std: {vllm_std:.4f}")

        # Assert high cosine similarity (decode phase should also match)
        assert cos_sim > 0.99, (
            f"Decode step {step_idx}: Cosine similarity {cos_sim:.6f} should be >0.99. "
            f"This indicates the fix may not work correctly during decode phase."
        )

        # Assert low mean absolute error
        assert mae < 0.01, (
            f"Decode step {step_idx}: MAE {mae:.6f} should be <0.01"
        )

    print(f"\n✓ All {num_tokens_to_compare} decode steps match between HF and vLLM")

    # Clean up
    await vllm_model.clear_all_vectors()
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()
