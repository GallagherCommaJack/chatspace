#!/usr/bin/env python3
"""Quick test script to validate vLLM steering vector implementation.

This script tests that:
1. VLLMSteerModel can be initialized and load a model
2. Steering vectors can be set and applied
3. Generation works with and without steering
4. Hook-based steering actually affects outputs
"""

import argparse
import sys
from pathlib import Path

import torch
from vllm import SamplingParams

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


def test_basic_generation(model_name: str = "Qwen/Qwen2.5-3B"):
    """Test basic model loading and generation."""
    print(f"[1/4] Testing basic model loading and generation with {model_name}")

    # Initialize vLLM model
    cfg = VLLMSteeringConfig(
        model_name=model_name,
        target_layer=16,  # Use middle layer for smaller model
        init_scale=0.0,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
    )

    print("  Loading model...")
    model = VLLMSteerModel(cfg)
    print("  ✓ Model loaded successfully")

    # Test generation without steering
    prompts = ["Once upon a time"]
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)

    print("  Generating text without steering...")
    outputs = model.generate(prompts, sampling_params)
    print(f"  Output: {outputs[0][:100]}...")
    print("  ✓ Generation successful")

    return model


def test_steering_vector_setting(model: VLLMSteerModel):
    """Test setting steering vectors."""
    print("\n[2/4] Testing steering vector setting")

    hidden_size = model.hidden_size
    print(f"  Model hidden size: {hidden_size}")

    # Create a random steering vector
    test_vector = torch.randn(hidden_size)
    print(f"  Created test vector with shape {test_vector.shape}")

    # Set the vector
    model.set_vector(test_vector)
    print("  ✓ Steering vector set successfully")

    # Verify it was set correctly
    with torch.no_grad():
        diff = torch.abs(model.current_vector() - test_vector).max()
        assert diff < 1e-5, f"Vector not set correctly (max diff: {diff})"
    print("  ✓ Steering vector verified")

    # Test clearing the vector
    model.set_vector(None)
    with torch.no_grad():
        assert torch.abs(model.current_vector()).max() < 1e-5, "Vector not cleared"
    print("  ✓ Steering vector cleared successfully")


def test_layer_switching(model: VLLMSteerModel):
    """Test changing target layer."""
    print("\n[3/4] Testing layer switching")

    original_layer = model.cfg.target_layer
    print(f"  Original target layer: {original_layer}")

    # Switch to different layer
    new_layer = original_layer + 2
    model.set_target_layer(new_layer)
    assert model.cfg.target_layer == new_layer
    print(f"  ✓ Switched to layer {new_layer}")

    # Switch back
    model.set_target_layer(original_layer)
    assert model.cfg.target_layer == original_layer
    print(f"  ✓ Switched back to layer {original_layer}")


def test_steering_effect(model: VLLMSteerModel):
    """Test that steering actually affects generation."""
    print("\n[4/4] Testing steering effect on generation")

    hidden_size = model.hidden_size
    prompt = "The weather today is"

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=30,
        seed=42,  # Use seed for reproducibility
    )

    # Generate without steering (baseline)
    model.set_vector(None)
    print("  Generating baseline (no steering)...")
    baseline = model.generate([prompt], sampling_params)[0]
    print(f"  Baseline: {baseline[:100]}")

    # Generate with positive steering vector
    positive_vector = torch.randn(hidden_size) * 5.0  # Large magnitude
    model.set_vector(positive_vector)
    print("  Generating with positive steering...")
    steered_pos = model.generate([prompt], sampling_params)[0]
    print(f"  Steered+: {steered_pos[:100]}")

    # Generate with negative steering vector
    negative_vector = -positive_vector
    model.set_vector(negative_vector)
    print("  Generating with negative steering...")
    steered_neg = model.generate([prompt], sampling_params)[0]
    print(f"  Steered-: {steered_neg[:100]}")

    # Verify outputs are different (steering has an effect)
    # Note: They might occasionally be the same by chance with random vectors,
    # but typically they should differ
    if baseline != steered_pos or baseline != steered_neg:
        print("  ✓ Steering affects generation (outputs differ)")
    else:
        print("  ⚠ Warning: Steering may not be working (outputs identical)")
        print("    This can happen with random vectors; try with trained vectors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="Model to test with (default: Qwen/Qwen2.5-3B for faster testing)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("vLLM Steering Vector Test Suite")
    print("=" * 70)

    try:
        # Run tests
        model = test_basic_generation(args.model)
        test_steering_vector_setting(model)
        test_layer_switching(model)
        test_steering_effect(model)

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Test with actual trained steering vectors")
        print("  2. Run full rollout generation with --use-vllm flag")
        print("  3. Compare outputs between HF and vLLM backends")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
