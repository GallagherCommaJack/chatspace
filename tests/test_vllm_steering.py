"""CUDA-dependent smoke tests for vLLM steering backend."""

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
from chatspace.steering.model import QwenSteerModel, SteeringVectorConfig
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

    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    model.set_layer_projection_cap(target_layer, cap_vector, cap_below=-0.5, cap_above=0.75)
    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)
    model.set_layer_ablation(target_layer, ablation_vector, scale=0.4)

    inspection = model._engine_client.collective_rpc(
        steering_runtime.inspect_layer_vector, args=(target_layer,)
    )
    assert inspection, "Expected inspection data for target layer."
    layer_info = inspection[0]
    projection_cap = layer_info.get("projection_cap")
    assert projection_cap is not None
    assert projection_cap["cap_below"] == pytest.approx(-0.5)
    assert projection_cap["cap_above"] == pytest.approx(0.75)
    ablation_info = layer_info.get("ablation")
    assert ablation_info is not None
    assert ablation_info["scale"] == pytest.approx(0.4)

    model.clear_layer_projection_cap(target_layer)
    model.clear_layer_ablation(target_layer)
    inspection_after_clear = model._engine_client.collective_rpc(
        steering_runtime.inspect_layer_vector, args=(target_layer,)
    )
    assert inspection_after_clear, "Expected inspection after clearing."
    layer_info_after_clear = inspection_after_clear[0]
    assert layer_info_after_clear.get("projection_cap") is None
    assert layer_info_after_clear.get("ablation") is None

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_matches_hf_logprob_shift():
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 2
    prompt = "In a quiet village, the baker"

    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hf_cfg = SteeringVectorConfig(model_name=model_name, target_layer=target_layer, init_scale=0.0)
    hf_model = QwenSteerModel(
        hf_cfg,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    hf_model.eval()

    with torch.no_grad():
        hf_outputs_base = hf_model(**inputs)
    base_logits = hf_outputs_base.logits[:, -1, :].float()
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    baseline_token = int(torch.argmax(base_logprobs, dim=-1).item())
    hf_baseline_lp = float(base_logprobs[0, baseline_token])

    steering_vector = torch.randn(hidden_size, dtype=torch.float32) * 0.01
    hf_model.set_vector(steering_vector)
    with torch.no_grad():
        hf_outputs_steered = hf_model(**inputs)
    steered_logits = hf_outputs_steered.logits[:, -1, :].float()
    steered_logprobs = torch.log_softmax(steered_logits, dim=-1)
    hf_steered_lp = float(steered_logprobs[0, baseline_token])
    hf_shift = hf_steered_lp - hf_baseline_lp

    hf_model.set_vector(None)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)
    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )

    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))

    def _extract_logprob(output, token_id):
        entry = output.logprobs[0].get(int(token_id))
        if entry is None:
            raise AssertionError(f"Token id {token_id} not present in returned logprobs")
        return float(entry.logprob)

    baseline_out = vllm_model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0].outputs[0]
    vllm_baseline_lp = _extract_logprob(baseline_out, baseline_token)

    vllm_model.set_layer_vector(target_layer, steering_vector)
    steered_out = vllm_model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0].outputs[0]
    vllm_steered_lp = _extract_logprob(steered_out, baseline_token)
    vllm_shift = vllm_steered_lp - vllm_baseline_lp

    baseline_delta = abs(hf_baseline_lp - vllm_baseline_lp)
    assert baseline_delta <= 2e-2, (
        f"Baseline logprob mismatch {baseline_delta:.4f} exceeds tolerance 0.02"
    )

    shift_delta = abs(hf_shift - vllm_shift)
    shift_tol = max(1e-3, baseline_delta + 5e-4)
    assert shift_delta <= shift_tol, (
        f"Steering logprob shift mismatch {shift_delta:.4f} exceeds tolerance {shift_tol:.4f}"
    )

    vllm_model.clear_all_vectors()
    del vllm_model
    del hf_model
    torch.cuda.empty_cache()
