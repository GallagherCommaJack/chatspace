"""CUDA-dependent smoke tests for vLLM steering backend."""

from __future__ import annotations

import os

import pytest
import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig
from chatspace.vllm_steering import runtime as steering_runtime

# vLLM >=0.11 requires enabling pickle-based serialization for custom RPCs.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_steering_vector_round_trip():

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

    hidden_size = model.hidden_size
    vector = torch.randn(hidden_size, dtype=torch.float32)
    model.set_layer_vector(target_layer, vector)

    worker_maps = model._fetch_worker_vectors()
    assert worker_maps, "Expected at least one worker vector."
    for worker_map in worker_maps:
        worker_vec = worker_map[target_layer]
        assert torch.allclose(
            worker_vec, vector.to(dtype=worker_vec.dtype), atol=1e-5
        ), "Broadcast steering vector does not match worker copy."

    worker_state = model._engine_client.collective_rpc(steering_runtime.fetch_worker_state)
    assert worker_state, "Expected worker state info."
    layer_count = int(worker_state[0].get("layer_count", 0) or 0)
    if layer_count > 1:
        other_layer = (target_layer + 1) % layer_count
        if other_layer == target_layer:
            other_layer = (other_layer + 1) % layer_count
        extra_vector = torch.randn(hidden_size, dtype=torch.float32)
        model.set_layer_vector(other_layer, extra_vector)
        expanded_maps = model._fetch_worker_vectors()
        for worker_map in expanded_maps:
            assert other_layer in worker_map, "Missing secondary layer vector."
            worker_vec = worker_map[other_layer]
            assert torch.allclose(
                worker_vec, extra_vector.to(dtype=worker_vec.dtype), atol=1e-5
            ), "Secondary layer vector mismatch."

    model.clear_all_vectors()
    cleared_maps = model._fetch_worker_vectors()
    for worker_map in cleared_maps:
        for worker_vec in worker_map.values():
            assert torch.allclose(worker_vec, torch.zeros_like(worker_vec))

    # Clean up to avoid leaving GPU memory allocated between tests.
    del model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_chat_respects_steering():
    torch.manual_seed(0)

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

    prompt = "State the color of a clear daytime sky."
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)

    baseline_req = model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    baseline_out = baseline_req.outputs[0]
    baseline_token = baseline_out.token_ids[0]
    baseline_logprob = baseline_out.logprobs[0][baseline_token].logprob

    scale = 5_000.0
    random_vector = torch.randn(model.hidden_size, dtype=torch.float32) * scale
    model.set_layer_vector(target_layer, random_vector)
    worker_info = model._engine_client.collective_rpc(
        steering_runtime.inspect_layer_vector, args=(target_layer,)
    )
    assert worker_info, "Expected worker diagnostics."
    assert worker_info[0]["has_vector"], "Patched layer missing steering vector."
    assert worker_info[0]["norm"] > 0, "Worker vector norm is zero unexpectedly."

    steered_req = model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    steered_out = steered_req.outputs[0]
    steered_token = steered_out.token_ids[0]
    steered_logprob = steered_out.logprobs[0][steered_token].logprob
    worker_info_after = model._engine_client.collective_rpc(
        steering_runtime.inspect_layer_vector, args=(target_layer,)
    )

    model.clear_all_vectors()
    reset_req = model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    reset_out = reset_req.outputs[0]
    reset_token = reset_out.token_ids[0]
    reset_logprob = reset_out.logprobs[0][reset_token].logprob

    request = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_sampling = SamplingParams(temperature=0.0, max_tokens=4)
    prefilled = model.chat(
        request,
        sampling_params=chat_sampling,
        prefill_assistant="<think>\n</think>\n",
    )[0]

    assert not torch.isclose(
        torch.tensor(baseline_logprob), torch.tensor(steered_logprob)
    ), (
        "Steering did not perturb token logprobs. "
        f"worker_info={worker_info} worker_info_after={worker_info_after}"
    )
    assert torch.isclose(
        torch.tensor(baseline_logprob), torch.tensor(reset_logprob)
    ), "Clearing the steering vector should restore baseline behaviour."
    assert "ASSISTANT_PREFILL:" not in prefilled

    del model
