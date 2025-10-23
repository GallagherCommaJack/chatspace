#!/usr/bin/env python3
"""Quick test to verify Gemma decoder layer patching works."""

import os
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import VLLMSteerModel, VLLMSteeringConfig
from chatspace.vllm_steering import runtime as steering_runtime

def main():
    """Test Gemma patching."""
    model_name = "google/gemma-3-27b-it"  # Gemma3 27B on H200

    print(f"Testing Gemma patching with {model_name}")

    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,  # More memory for 27B model on H200
        max_model_len=256,
    )

    target_layer = 23  # Middle layer for 27B model (likely ~46 layers)

    try:
        model = VLLMSteerModel(cfg, enforce_eager=True, bootstrap_layers=(target_layer,))
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return

    print(f"Model loaded. Hidden size: {model.hidden_size}")

    # Set a steering vector
    steering_vec = torch.randn(model.hidden_size, dtype=torch.float32) * 10.0
    model.set_layer_vector(target_layer, steering_vec)
    print(f"Set steering vector on layer {target_layer}")

    # Test ablation
    ablation_vec = torch.randn(model.hidden_size, dtype=torch.float32)
    model.set_layer_ablation(target_layer, ablation_vec, scale=0.5)
    print(f"Set ablation on layer {target_layer}")

    # Run a generation to trigger patching
    prompt = "Hello"
    sampling = SamplingParams(temperature=0.0, max_tokens=2)
    output = model.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    print(f"Generated: {output.outputs[0].text}")

    # Now check inspection
    inspection = model._engine_client.collective_rpc(
        steering_runtime.inspect_layer_vector, args=(target_layer,)
    )

    if inspection:
        layer_info = inspection[0]
        output_types = layer_info.get("output_types", [])
        print(f"Output types after generation: {output_types}")

        if "tuple" in output_types:
            print("✓ Gemma decoder layer is using tuple output (expected)")
        else:
            print(f"⚠ Unexpected output types: {output_types}")
    else:
        print("⚠ No inspection data returned")

    del model
    torch.cuda.empty_cache()
    print("Test complete!")

if __name__ == "__main__":
    main()
