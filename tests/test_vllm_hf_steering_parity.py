"""Parity checks for steering transforms between HuggingFace and vLLM backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

# vLLM >=0.11 requires enabling pickle-based serialization for custom RPCs.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@dataclass
class ProjectionCapParams:
    vector: torch.Tensor
    min: float | None
    max: float | None


@dataclass
class AblationParams:
    vector: torch.Tensor
    scale: float


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vector)
    if float(norm) <= 0:
        raise ValueError("Direction vector norm must be positive.")
    return vector / norm


def _apply_projection_cap(hidden: torch.Tensor, unit: torch.Tensor, *, minimum: float | None, maximum: float | None) -> torch.Tensor:
    if minimum is None and maximum is None:
        return hidden
    flat = hidden.reshape(-1, hidden.shape[-1])
    projection = flat @ unit
    clamp_kwargs: dict[str, Any] = {}
    if minimum is not None:
        clamp_kwargs["min"] = projection.new_tensor(float(minimum))
    if maximum is not None:
        clamp_kwargs["max"] = projection.new_tensor(float(maximum))
    clamped = torch.clamp(projection, **clamp_kwargs)  # type: ignore[arg-type]
    if clamped is projection:
        return hidden
    delta = (clamped - projection).unsqueeze(-1) * unit
    return (flat + delta).reshape_as(hidden)


def _apply_ablation(hidden: torch.Tensor, unit: torch.Tensor, *, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return hidden
    flat = hidden.reshape(-1, hidden.shape[-1])
    projection = flat @ unit
    component = projection.unsqueeze(-1) * unit
    return (flat + (scale - 1.0) * component).reshape_as(hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for vLLM steering.")
def test_vllm_hf_steering_combinations_match():
    torch.manual_seed(123)

    model_name = "Qwen/Qwen3-0.6B"
    target_layer = 4
    prompt = "A curious squirrel contemplates the mysteries of the forest."

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
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_size = hf_model.config.hidden_size
    vector = torch.randn(hidden_size, dtype=torch.float32) * 0.05
    cap_vector = torch.randn(hidden_size, dtype=torch.float32)
    ablation_vector = torch.randn(hidden_size, dtype=torch.float32)

    projection_spec = ProjectionCapParams(vector=cap_vector, min=-0.2, max=0.2)
    ablation_spec = AblationParams(vector=ablation_vector, scale=0.6)

    cases: list[dict[str, Any]] = [
        {"name": "vector_only", "vector": vector, "cap": None, "ablation": None},
        {"name": "projection_cap_only", "vector": None, "cap": projection_spec, "ablation": None},
        {"name": "ablation_only", "vector": None, "cap": None, "ablation": ablation_spec},
        {"name": "vector_and_cap", "vector": vector, "cap": projection_spec, "ablation": None},
        {"name": "vector_cap_ablation", "vector": vector, "cap": projection_spec, "ablation": ablation_spec},
    ]

    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.05,
        max_model_len=inputs.input_ids.shape[1] + 16,
        dtype="float16",
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    vllm_model.enable_hidden_state_capture(target_layer, capture_before=False, capture_after=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=0)

    try:
        for case in cases:
            # ------------------------------------------------------------------
            # HuggingFace path
            # ------------------------------------------------------------------
            captured_hf_hidden: dict[str, torch.Tensor] = {}

            def hook(module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                orig_dtype = hidden.dtype
                hidden_fp32 = hidden.to(torch.float32)

                if case["vector"] is not None:
                    hidden_fp32 = hidden_fp32 + case["vector"].to(device=hidden_fp32.device)

                if case["cap"] is not None:
                    unit = _normalize(case["cap"].vector).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                    hidden_fp32 = _apply_projection_cap(
                        hidden_fp32,
                        unit,
                        minimum=case["cap"].min,
                        maximum=case["cap"].max,
                    )

                if case["ablation"] is not None:
                    unit = _normalize(case["ablation"].vector).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                    hidden_fp32 = _apply_ablation(hidden_fp32, unit, scale=case["ablation"].scale)

                captured_hf_hidden["tensor"] = hidden_fp32.detach().cpu().clone()
                hidden_out = hidden_fp32.to(dtype=orig_dtype)
                if isinstance(output, tuple):
                    return (hidden_out,) + output[1:]
                return hidden_out

            layer = hf_model.model.layers[target_layer]
            handle = layer.register_forward_hook(hook)
            with torch.no_grad():
                hf_model(**inputs)
            handle.remove()

            hf_hidden = captured_hf_hidden["tensor"]

            # ------------------------------------------------------------------
            # vLLM path
            # ------------------------------------------------------------------
            vllm_model.clear_all_vectors()
            vllm_model.clear_layer_projection_cap(target_layer)
            vllm_model.clear_layer_ablation(target_layer)
            vllm_model.clear_hidden_states(target_layer)

            if case["vector"] is not None:
                vllm_model.set_layer_vector(target_layer, case["vector"])
            if case["cap"] is not None:
                vllm_model.set_layer_projection_cap(
                    target_layer,
                    case["cap"].vector,
                    min=case["cap"].min,
                    max=case["cap"].max,
                )
            if case["ablation"] is not None:
                vllm_model.set_layer_ablation(
                    target_layer,
                    case["ablation"].vector,
                    scale=case["ablation"].scale,
                )

            vllm_model.generate([prompt], sampling_params=sampling)
            states = vllm_model.fetch_hidden_states(layer_idx=target_layer)
            vllm_hidden = states[0][target_layer][0]["after"].to(dtype=torch.float32)

            # ------------------------------------------------------------------
            # Comparison
            # ------------------------------------------------------------------
            hf_flat = hf_hidden.reshape(-1)
            vllm_flat = vllm_hidden.reshape(-1)
            mae = torch.mean(torch.abs(vllm_flat - hf_flat)).item()
            cos = F.cosine_similarity(hf_flat.unsqueeze(0), vllm_flat.unsqueeze(0), dim=-1).item()

            assert mae < 2e-3, f"{case['name']}: mean abs diff too large ({mae:.6f})"
            assert cos > 0.9995, f"{case['name']}: cosine similarity degraded ({cos:.6f})"
    finally:
        del vllm_model
        torch.cuda.empty_cache()
