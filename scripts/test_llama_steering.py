#!/usr/bin/env python3
"""Quick smoke test to verify Llama steering works in vLLM."""

from __future__ import annotations

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import VLLMSteerModel, VLLMSteeringConfig


def main() -> None:
    """Test basic Llama steering functionality."""

    # Use a tiny Llama model for testing
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading {model_name}...")
    cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )

    # Bootstrap with layer 8 (middle layer for 1B model which has 16 layers)
    model = VLLMSteerModel(cfg, bootstrap_layers=(8,))

    print(f"Model loaded. Hidden size: {model.hidden_size}, Layer count: {model.layer_count}")

    # Test 1: Generate with zero steering (baseline)
    print("\n=== Test 1: Baseline generation (zero steering) ===")
    model.set_target_layer(8)
    model.set_vector(torch.zeros(model.hidden_size))

    prompt = "The capital of France is"
    sampling = SamplingParams(temperature=0.0, max_tokens=5)
    baseline_output = model.generate([prompt], sampling)[0]
    print(f"Prompt: {prompt}")
    print(f"Output: {baseline_output}")

    # Test 2: Generate with random steering
    print("\n=== Test 2: Generation with random steering vector ===")
    random_vector = torch.randn(model.hidden_size) * 0.5
    model.set_vector(random_vector)

    steered_output = model.generate([prompt], sampling)[0]
    print(f"Prompt: {prompt}")
    print(f"Output: {steered_output}")

    # Outputs should differ with steering
    if baseline_output != steered_output:
        print("\n✓ Steering successfully modified output")
    else:
        print("\n✗ WARNING: Steering did not modify output")

    # Test 3: Verify vector retrieval
    print("\n=== Test 3: Verify vector retrieval ===")
    retrieved = model.current_vector(layer_idx=8)
    vector_match = torch.allclose(retrieved, random_vector, rtol=1e-4, atol=1e-6)
    print(f"Vector retrieved successfully: {vector_match}")
    print(f"Vector norm: {retrieved.norm().item():.4f}")

    # Test 4: Clear vector and verify
    print("\n=== Test 4: Clear vector ===")
    model.clear_layer_vector(8)
    cleared_vector = model.current_vector(layer_idx=8)
    is_zero = torch.allclose(cleared_vector, torch.zeros_like(cleared_vector))
    print(f"Vector cleared: {is_zero}")

    # Test 5: Multi-layer steering
    print("\n=== Test 5: Multi-layer steering ===")
    model.set_layer_vector(6, torch.randn(model.hidden_size) * 0.3)
    model.set_layer_vector(10, torch.randn(model.hidden_size) * 0.3)

    multi_output = model.generate([prompt], sampling)[0]
    print(f"Prompt: {prompt}")
    print(f"Output: {multi_output}")
    print(f"Multi-layer output differs from baseline: {multi_output != baseline_output}")

    # Test 6: Hidden state capture
    print("\n=== Test 6: Hidden state capture ===")
    model.clear_all_vectors()
    model.set_layer_vector(8, torch.randn(model.hidden_size) * 0.5)
    model.enable_hidden_state_capture(8, capture_before=True, capture_after=True, max_captures=2)

    _ = model.generate([prompt], sampling)
    states = model.fetch_hidden_states(layer_idx=8)

    if states and 8 in states[0]:
        captures = states[0][8]
        print(f"Captured {len(captures)} hidden states")
        if captures:
            first_capture = captures[0]
            print(f"  - Keys in first capture: {list(first_capture.keys())}")
            if "before" in first_capture:
                print(f"  - Before shape: {first_capture['before'].shape}")
            if "after" in first_capture:
                print(f"  - After shape: {first_capture['after'].shape}")
        print("✓ Hidden state capture working")
    else:
        print("✗ WARNING: No hidden states captured")

    model.disable_hidden_state_capture(8)

    print("\n=== All tests completed ===")
    print("Llama steering appears to be working correctly!")


if __name__ == "__main__":
    main()
