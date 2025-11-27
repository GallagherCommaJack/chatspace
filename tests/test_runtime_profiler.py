"""Tests for torch profiler integration in chatspace.vllm_steering.runtime."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from chatspace.vllm_steering import runtime


@pytest.mark.skipif(runtime.torch_profile is None, reason="torch profiler unavailable")
def test_profile_fetch_batch_attaches_summary(monkeypatch):
    monkeypatch.setattr(runtime, "_PROFILE_FETCH_ENABLED", True)
    monkeypatch.setattr(runtime, "_PROFILE_FETCH_TRACE_DIR", None)
    metadata = {"request_count": 1}

    with runtime._profile_fetch_batch(None, metadata) as prof:
        assert prof is not None
        # Generate some measurable work inside profiler context
        torch.matmul(torch.randn(32, 32), torch.randn(32, 32))

    summary = getattr(prof, "_chatspace_summary", None)
    assert summary is not None, "Profiler summary should be attached to session"
    assert summary["metadata"]["request_count"] == 1
    assert summary["event_count"] >= 1
    assert summary["events"], "Expected at least one summarized event"


@pytest.mark.skipif(runtime.torch_profile is None, reason="torch profiler unavailable")
def test_fetch_batch_captures_records_profiler_summary(monkeypatch):
    import threading
    monkeypatch.setattr(runtime, "_PROFILE_FETCH_ENABLED", True)
    monkeypatch.setattr(runtime, "_PROFILE_FETCH_TRACE_DIR", None)

    state = runtime._SteeringState(
        hidden_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        active_capture_requests={"req": set()},
        request_captures={"req": {0: torch.randn(2, 4)}},
        request_prefill_buffers={"req": {}},
        request_decode_buffers={"req": {}},
        request_pending_transfers={"req": {}},
        request_last_phase={"req": "decode"},
        request_token_counts={"req": 2},
        step_metadata={},
        global_step=0,
        request_steering_specs={},
        # Shared memory fields (required for _create_shared_tensor)
        active_shared_memory={},
        shm_lock=threading.Lock(),
        shm_max_gb=128.0,
        shm_ttl_seconds=600,
    )

    worker = types.SimpleNamespace(_chatspace_steering=state)

    result = runtime.fetch_batch_captures(worker, ["req"])
    assert "req" in result
    assert 0 in result["req"]
    assert state.last_fetch_profile is not None
    assert state.last_fetch_profile["metadata"]["request_count"] == 1
    assert state.last_fetch_profile["event_count"] >= 1


@pytest.mark.skipif(runtime.torch_profile is None, reason="torch profiler unavailable")
def test_profile_fetch_batch_exports_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_PROFILE_FETCH_ENABLED", True)
    monkeypatch.setattr(runtime, "_PROFILE_FETCH_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "_PROFILE_FETCH_TRACE_PREFIX", "unittest_trace")
    metadata = {"request_count": 2}

    with runtime._profile_fetch_batch(None, metadata) as prof:
        assert prof is not None
        torch.matmul(torch.randn(16, 16), torch.randn(16, 16))

    summary = getattr(prof, "_chatspace_summary", None)
    assert summary is not None
    trace_path = summary.get("trace_path")
    assert trace_path is not None
    assert Path(trace_path).exists()
