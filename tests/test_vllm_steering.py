"""CUDA-dependent smoke tests for vLLM steering backend."""

from __future__ import annotations

import os

import pytest
import torch

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

# vLLM >=0.11 requires enabling pickle-based serialization for custom RPCs.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_steering_vector_round_trip():

    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        target_layer=2,
        init_scale=0.0,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        max_model_len=128,
    )

    try:
        model = VLLMSteerModel(cfg)
    except OSError as exc:  # pragma: no cover - allows offline environments
        pytest.skip(f"Unable to load model ({exc}). Ensure weights are cached.")

    hidden_size = model.hidden_size
    vector = torch.randn(hidden_size, dtype=torch.float32)
    model.set_vector(vector)

    worker_vectors = model._fetch_worker_vectors()
    assert worker_vectors, "Expected at least one worker vector."
    for worker_vec in worker_vectors:
        assert torch.allclose(
            worker_vec, vector.to(dtype=worker_vec.dtype), atol=1e-5
        ), "Broadcast steering vector does not match worker copy."

    model.set_vector(None)
    cleared_vectors = model._fetch_worker_vectors()
    for worker_vec in cleared_vectors:
        assert torch.allclose(worker_vec, torch.zeros_like(worker_vec))

    # Clean up to avoid leaving GPU memory allocated between tests.
    del model
