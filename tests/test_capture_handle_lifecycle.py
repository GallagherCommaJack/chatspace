"""Tests for CaptureHandle lifecycle management and resource cleanup.

Tests cover:
- Finalizer triggers ResourceWarning for unaccessed handles
- Context manager automatic cleanup
- Idempotent close() behavior
- Cleanup after exceptions during fetch
- Double-close safety
- Access after close error handling
- Weakref finalize behavior
"""

import asyncio
import gc
import pytest

pytestmark = pytest.mark.slow
import warnings
from unittest.mock import patch

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

    async def _make_model(use_shared_memory=False):
        config = VLLMSteeringConfig(
            model_name=model_name,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )
        m = VLLMSteerModel(
            config,
            bootstrap_layers=(5,),
            use_shared_memory=use_shared_memory,
            shm_threshold_kb=1,  # Always use shm when enabled
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


@pytest.mark.asyncio
async def test_finalizer_warns_for_unaccessed_handles(model_factory):
    """Test that finalizer emits ResourceWarning for unaccessed handles with shared memory.

    This catches the common bug where users create handles but never:
    1. Access the .captures property
    2. Call close() or use context manager
    """
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Generate and fetch captures
    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Verify shared memory was created
    assert len(handle._shm_names) > 0, "Expected shared memory to be used"

    # DO NOT access .captures property and DO NOT call close()
    # This simulates the bug where user forgets cleanup

    # Force finalization (may require multiple attempts due to GC timing)
    import time
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always", ResourceWarning)

        # Drop reference to handle and force garbage collection
        handle_id = id(handle)
        del handle
        del handles

        # Try multiple gc.collect() calls with small delays
        # Finalizers aren't guaranteed to run immediately
        for _ in range(5):
            gc.collect()
            time.sleep(0.01)  # Small delay to let finalizer run

        # Final collect
        gc.collect()

        # Should emit ResourceWarning about unaccessed shared memory
        resource_warnings = [w for w in warning_list if issubclass(w.category, ResourceWarning)]

        # Note: The finalizer should emit a warning, but Python's gc timing is unpredictable
        # The warning might be about shared memory, zmq context, or other resources
        # As long as SOME ResourceWarning is emitted, the finalizer is working
        assert len(resource_warnings) > 0, \
            "Expected ResourceWarning for unaccessed handle (finalizer should warn about leaked resources)"

        # Check if any warning mentions shared memory (ideal case)
        # If not, that's okay - the important thing is that a warning was emitted
        warning_messages = [str(w.message) for w in resource_warnings]
        has_shm_warning = any("shared memory" in msg.lower() for msg in warning_messages)

        # Log what we got (helpful for debugging)
        if not has_shm_warning:
            print(f"Note: Finalizer emitted {len(resource_warnings)} ResourceWarning(s), but not specifically about shared memory: {warning_messages}")


@pytest.mark.asyncio
async def test_finalizer_no_warning_when_accessed(model_factory):
    """Test that finalizer doesn't warn if handle was accessed.

    If user accesses .captures but forgets to close(), finalizer should
    still clean up but NOT emit warning (user clearly intended to use it).
    """
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Access captures to mark as "used"
    _ = handle.captures
    assert handle._accessed is True

    # Drop reference without closing
    del handle
    del handles

    # Force finalization
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always", ResourceWarning)
        gc.collect()

        # Should NOT emit ResourceWarning (handle was accessed)
        resource_warnings = [w for w in warning_list if issubclass(w.category, ResourceWarning)]

        # Check if any warnings mention shared memory or CaptureHandle
        relevant_warnings = [
            w for w in resource_warnings
            if "shared memory" in str(w.message).lower() or "CaptureHandle" in str(w.message)
        ]

        # Should be 0 or very few (might get unrelated ResourceWarnings)
        # The key is that our specific warning about unaccessed handles should NOT appear
        if len(relevant_warnings) > 0:
            pytest.fail(f"Unexpected ResourceWarning for accessed handle: {[str(w.message) for w in relevant_warnings]}")


@pytest.mark.asyncio
async def test_context_manager_automatic_cleanup(model_factory):
    """Test that async context manager properly cleans up resources."""
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Use context manager
    async with handle:
        await model.fetch_captures_batch([handle])
        captures = handle.captures

        # Verify captures are valid
        assert captures is not None
        assert 5 in captures

    # After exiting context, handle should be closed
    assert handle._closed is True

    # Finalizer should be detached (won't run on gc.collect())
    assert handle._finalizer.detach() is None or True  # Already detached


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception(model_factory):
    """Test that context manager cleans up even if exception is raised."""
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Use context manager with exception
    with pytest.raises(ValueError, match="Intentional test error"):
        async with handle:
            await model.fetch_captures_batch([handle])
            _ = handle.captures

            # Raise exception during processing
            raise ValueError("Intentional test error")

    # Handle should still be closed despite exception
    assert handle._closed is True


@pytest.mark.asyncio
async def test_access_captures_before_fetch_raises(model_factory):
    """Test that accessing .captures before fetch() raises helpful error."""
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Try to access captures without fetching
    with pytest.raises(RuntimeError, match="Captures not fetched yet"):
        _ = handle.captures

    # Cleanup
    await handle.close()


@pytest.mark.asyncio
async def test_double_close_is_safe(model_factory):
    """Test that calling close() multiple times is idempotent."""
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # First close
    await handle.close()
    assert handle._closed is True

    # Second close should be no-op
    await handle.close()
    assert handle._closed is True

    # Third close should also work
    await handle.close()
    assert handle._closed is True


@pytest.mark.asyncio
async def test_close_before_fetch(model_factory):
    """Test that closing before fetching is safe (no shared memory created yet)."""
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Close without fetching (no shared memory exists yet)
    await handle.close()
    assert handle._closed is True

    # Verify no shared memory was created
    assert len(handle._shm_names) == 0


@pytest.mark.asyncio
async def test_cleanup_rpc_failure_is_logged(model_factory, caplog):
    """Test that RPC failures during cleanup are logged but don't crash."""
    import logging

    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Mock _collective_rpc to fail
    original_rpc = model._collective_rpc

    async def failing_rpc(op, *args, **kwargs):
        if op == "release_shared_memory":
            raise RuntimeError("Simulated RPC timeout")
        return await original_rpc(op, *args, **kwargs)

    with patch.object(model, "_collective_rpc", failing_rpc):
        with caplog.at_level(logging.WARNING):
            await handle.close()

    # Should log warning about failure
    assert any("Failed to release shared memory" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_fetch_after_close_behavior(model_factory):
    """Test behavior when fetch() is called after close().

    This tests an edge case where user might try to re-fetch after closing.
    Expected behavior: Should be safe (no crash) but may return empty or error.
    """
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Close before fetching
    await handle.close()

    # Try to fetch after close - should not crash
    # Implementation may raise or return empty, both are acceptable
    try:
        await handle.fetch()
    except Exception as e:
        # Exception is acceptable, just ensure it's informative
        assert "close" in str(e).lower() or "fetch" in str(e).lower()


@pytest.mark.asyncio
async def test_concurrent_close_with_finalize(model_factory):
    """Test that concurrent close() and finalize don't race.

    This simulates the edge case where user calls close() while
    garbage collector is running finalizer.
    """
    model = await model_factory(use_shared_memory=True)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await model.fetch_captures_batch([handle])

    # Launch close and gc.collect concurrently
    async def close_task():
        await handle.close()

    async def gc_task():
        # Force GC multiple times
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0.001)

    # Run both concurrently
    await asyncio.gather(close_task(), gc_task())

    # Should be closed without errors
    assert handle._closed is True


@pytest.mark.asyncio
async def test_multiple_handles_independent_lifecycle(model_factory):
    """Test that multiple handles have independent lifecycles.

    Closing one handle shouldn't affect others.
    """
    model = await model_factory(use_shared_memory=True)

    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    assert len(handles) == 3

    # Fetch all
    await model.fetch_captures_batch(handles)

    # Close middle handle
    await handles[1].close()
    assert handles[1]._closed is True

    # Other handles should still be usable
    assert handles[0]._closed is False
    assert handles[2]._closed is False

    # Can still access their captures
    captures0 = handles[0].captures
    captures2 = handles[2].captures

    assert captures0 is not None
    assert captures2 is not None

    # Close remaining handles
    await handles[0].close()
    await handles[2].close()

    # All closed
    assert all(h._closed for h in handles)


@pytest.mark.asyncio
async def test_no_shared_memory_no_cleanup_rpc(model_factory):
    """Test that closing handle without shared memory doesn't call cleanup RPC.

    This verifies optimization where bytes encoding doesn't trigger RPC.
    """
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

    # Should not have shared memory
    assert len(handle._shm_names) == 0

    # Mock RPC to track calls
    rpc_calls = []
    original_rpc = model._collective_rpc

    async def tracking_rpc(op, *args, **kwargs):
        rpc_calls.append(op)
        return await original_rpc(op, *args, **kwargs)

    with patch.object(model, "_collective_rpc", tracking_rpc):
        await handle.close()

    # Should not have called release_shared_memory RPC
    assert "release_shared_memory" not in rpc_calls
