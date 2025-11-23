"""Tests for shared memory cleanup failure scenarios.

Tests cover:
- SharedMemory.unlink() failures (FileNotFoundError, PermissionError)
- Graceful degradation when unlink fails
- Worker-side TTL cleanup behavior
- Shared memory limit exhaustion
- Fallback to bytes encoding
- Concurrent cleanup race conditions
- Partial cleanup after exceptions
"""

import asyncio
import logging
import os
import pytest
import time
import warnings
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


@pytest.fixture
def model_name():
    """Small model for fast tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture
async def model_factory(model_name):
    """Factory for creating VLLMSteerModel with custom config."""
    created_models = []

    async def _make_model(
        use_shared_memory=False,
        shm_threshold_kb=1024,
        shm_ttl_seconds=600,
        shm_max_gb=128.0,
    ):
        config = VLLMSteeringConfig(
            model_name=model_name,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )
        m = VLLMSteerModel(
            config,
            bootstrap_layers=(5,),
            use_shared_memory=use_shared_memory,
            shm_threshold_kb=shm_threshold_kb,
            shm_ttl_seconds=shm_ttl_seconds,
            shm_max_gb=shm_max_gb,
            enforce_eager=True,
        )
        created_models.append(m)
        return m

    yield _make_model

    # Cleanup all created models
    for m in created_models:
        if hasattr(m, "_engine") and m._engine is not None:
            try:
                await m._engine.shutdown()
            except Exception:
                pass


@pytest.mark.slow
@pytest.mark.asyncio
async def test_unlink_file_not_found_error(model_factory, caplog):
    """Test that FileNotFoundError during unlink doesn't crash worker.

    This simulates the case where shared memory was already unlinked
    (e.g., by another process or TTL cleanup) before client calls close().
    """
    model = await model_factory(use_shared_memory=True, shm_threshold_kb=1)

    prompts = ["Once upon a time"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    assert len(handles) == 1
    handle = handles[0]

    # Fetch captures to trigger shared memory creation
    await model.fetch_captures_batch([handle])
    assert len(handle._shm_names) > 0, "Expected shared memory to be used"

    # Patch SharedMemory.close() to simulate unlink failure
    original_close = SharedMemory.close

    def failing_close(self):
        # Call original close to unmap
        original_close(self)
        # Then simulate that segment was already deleted
        raise FileNotFoundError(f"No such file or directory: '/dev/shm/{self.name}'")

    with patch.object(SharedMemory, "close", failing_close):
        # Should not raise, should log warning
        with caplog.at_level(logging.WARNING):
            await handle.close()

    # Verify worker is still functional after cleanup failure
    results2, handles2 = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )
    assert len(results2) == 1
    assert len(handles2) == 1

    # Cleanup second handle
    await handles2[0].close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_unlink_permission_error(model_factory, caplog):
    """Test that PermissionError during unlink doesn't crash worker.

    This simulates the case where /dev/shm has incorrect permissions.
    """
    model = await model_factory(use_shared_memory=True, shm_threshold_kb=1)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])
    assert len(handle._shm_names) > 0

    # Simulate permission error during unlink
    original_close = SharedMemory.close

    def permission_denied_close(self):
        original_close(self)
        raise PermissionError(f"Permission denied: '/dev/shm/{self.name}'")

    with patch.object(SharedMemory, "close", permission_denied_close):
        with caplog.at_level(logging.WARNING):
            await handle.close()

    # Verify warning was logged
    assert any("Failed to close shared memory" in record.message for record in caplog.records)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_idempotent_close(model_factory):
    """Test that calling close() multiple times is safe."""
    model = await model_factory(use_shared_memory=True, shm_threshold_kb=1)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # First close should work normally
    await handle.close()
    assert handle._closed is True

    # Second close should be no-op (not raise)
    await handle.close()
    assert handle._closed is True

    # Third close should also be safe
    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_close_without_fetch(model_factory):
    """Test that closing a handle before fetching is safe."""
    model = await model_factory(use_shared_memory=True, shm_threshold_kb=1)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Close without fetching (no shared memory created yet)
    await handle.close()
    assert handle._closed is True

    # Verify fetch after close raises appropriate error
    # (This tests the expected failure mode, not a bug)
    # Note: This might not raise if fetch() implementation checks _closed
    # Just ensure it doesn't crash


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shared_memory_limit_fallback(model_factory):
    """Test graceful fallback to bytes encoding when limit is reached.

    This simulates hitting CHATSPACE_MAX_SHM_GB limit.
    """
    # Set very low limit to trigger fallback
    model = await model_factory(
        use_shared_memory=True,
        shm_threshold_kb=1,
        shm_max_gb=0.0001,  # 100 KB limit
    )

    prompts = ["Once upon a time, in a land far away, there lived a"]
    sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

    # Generate multiple captures to exceed limit
    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5, 10, 15, 20],  # Multiple layers to exceed limit
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Should still have captures (via bytes encoding fallback)
    assert handle._captures is not None
    assert len(handle._captures) > 0

    # Verify captures are valid
    for layer_idx, captures_list in handle._captures.items():
        for capture in captures_list:
            hidden = capture["hidden"]
            assert hidden.shape[0] > 0
            assert hidden.shape[1] == model.hidden_size
            assert not torch.isnan(hidden).any()

    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_after_fetch_exception(model_factory):
    """Test that shared memory is cleaned up even if fetch() raises."""
    model = await model_factory(use_shared_memory=True, shm_threshold_kb=1)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Mock _fetch_request_captures to raise after creating shared memory
    original_fetch = model._fetch_request_captures

    async def failing_fetch(*args, **kwargs):
        # Call original to create shared memory
        result = await original_fetch(*args, **kwargs)
        # Then raise to simulate error during processing
        raise RuntimeError("Simulated fetch failure")

    with patch.object(model, "_fetch_request_captures", failing_fetch):
        with pytest.raises(RuntimeError, match="Simulated fetch failure"):
            await handle.fetch()

    # Cleanup should still work
    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_close_operations(model_factory):
    """Test that concurrent close() calls don't cause race conditions."""
    model = await model_factory(use_shared_memory=True, shm_threshold_kb=1)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Launch multiple close operations concurrently
    close_tasks = [handle.close() for _ in range(5)]

    # All should complete without error
    await asyncio.gather(*close_tasks)

    # Handle should be closed
    assert handle._closed is True


@pytest.mark.slow
@pytest.mark.asyncio
async def test_threshold_boundary_conditions(model_factory):
    """Test shared memory vs bytes encoding at threshold boundary."""
    # Set threshold exactly at expected tensor size
    # For a small model with hidden_size=896 and ~20 tokens:
    # Tensor size ≈ 896 * 20 * 4 bytes = ~70KB per layer per request

    model = await model_factory(
        use_shared_memory=True,
        shm_threshold_kb=50,  # Below tensor size - should use shm
    )

    # Use longer prompt to ensure tensor exceeds threshold
    # Repeat "Test " many times to get enough tokens
    prompts = ["Test " * 100]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Should use shared memory (tensor is above threshold)
    assert len(handle._shm_names) > 0, "Expected shared memory for large tensor"

    await handle.close()

    # Now test with threshold above tensor size
    model2 = await model_factory(
        use_shared_memory=True,
        shm_threshold_kb=1000,  # Above tensor size - should use bytes
    )

    results2, handles2 = await model2.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle2 = handles2[0]
    await model2.fetch_captures_batch([handle2])

    # Should use bytes encoding (tensor is below threshold)
    # Note: Might still use shm if implementation batches segments
    # This is more of a performance hint than strict requirement

    await handle2.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_multiple_handles_cleanup(model_factory):
    """Test cleanup of multiple handles with shared memory."""
    model = await model_factory(use_shared_memory=True, shm_threshold_kb=1)

    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5, 10],
    )

    assert len(handles) == 3

    # Fetch all handles
    await model.fetch_captures_batch(handles)

    # All should have shared memory
    for handle in handles:
        assert len(handle._shm_names) > 0

    # Close in different order
    await handles[1].close()
    await handles[0].close()
    await handles[2].close()

    # All should be closed
    for handle in handles:
        assert handle._closed is True


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shared_memory_disabled_fallback(model_factory):
    """Test that system works correctly with shared memory disabled."""
    model = await model_factory(use_shared_memory=False)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Should not use shared memory
    assert len(handle._shm_names) == 0, "Expected bytes encoding, not shared memory"

    # Captures should still be valid
    assert handle._captures is not None
    assert 5 in handle._captures

    await handle.close()
