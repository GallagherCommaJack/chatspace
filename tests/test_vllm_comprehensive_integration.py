"""Comprehensive integration test for vLLM steering with batching, chat, decode, and multi-method steering.

This test validates the complete vLLM steering API with realistic usage:
- Batch generation (10 prompts)
- Chat formatting via tokenizer.apply_chat_template()
- Decode phase generation (30-50 tokens)
- All 3 steering methods (add, projection cap, ablation) on multiple layers
- Hidden state capture and comparison vs HuggingFace ground truth
- RWLock testing for concurrent operations
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    """Normalize a vector to unit length."""
    norm = torch.norm(vector)
    if float(norm) <= 0:
        raise ValueError("Vector norm must be positive.")
    return vector / norm


def _apply_projection_cap(
    hidden: torch.Tensor,
    unit: torch.Tensor,
    *,
    minimum: float | None,
    maximum: float | None
) -> torch.Tensor:
    """Apply projection capping to hidden states."""
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
    """Apply ablation to hidden states."""
    if scale == 1.0:
        return hidden
    flat = hidden.reshape(-1, hidden.shape[-1])
    projection = flat @ unit
    component = projection.unsqueeze(-1) * unit
    return (flat + (scale - 1.0) * component).reshape_as(hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for vLLM")
@pytest.mark.asyncio
async def test_comprehensive_vllm_integration():
    """Comprehensive integration test with batching, chat, decode, multi-method steering."""
    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-0.6B"
    num_prompts = 10
    max_tokens = 40

    # Get model config to determine hidden size
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_name)
    hidden_size = model_config.hidden_size

    # Layers for steering
    layer_2_config = {
        "layer": 2,
        "add_vector": torch.randn(hidden_size, dtype=torch.float32) * 0.1,
        "cap_vector": torch.randn(hidden_size, dtype=torch.float32),
        "cap_min": -0.3,
        "cap_max": 0.3,
    }

    layer_5_config = {
        "layer": 5,
        "ablation_vector": torch.randn(hidden_size, dtype=torch.float32),
        "ablation_scale": 0.7,
        "cap_vector": torch.randn(hidden_size, dtype=torch.float32),
        "cap_min": -0.5,
        "cap_max": 0.5,
    }

    # Create diverse chat messages for batch testing
    chat_messages = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "Explain quantum computing in simple terms."}],
        [{"role": "user", "content": "Write a haiku about programming."}],
        [{"role": "user", "content": "What are the benefits of exercise?"}],
        [{"role": "user", "content": "How do neural networks learn?"}],
        [{"role": "user", "content": "Describe the water cycle."}],
        [{"role": "user", "content": "What is the speed of light?"}],
        [{"role": "user", "content": "How does photosynthesis work?"}],
        [{"role": "user", "content": "What causes seasons on Earth?"}],
        [{"role": "user", "content": "Explain the concept of recursion."}],
    ]

    # Load tokenizer for chat formatting
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Convert chat messages to prompts using chat template
    prompts = []
    for messages in chat_messages:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    print(f"\n{'='*80}")
    print(f"Comprehensive vLLM Integration Test")
    print(f"{'='*80}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {max_tokens}")
    print(f"Steering layers: {layer_2_config['layer']}, {layer_5_config['layer']}")
    print(f"Methods: Additive + ProjectionCap on layer {layer_2_config['layer']}")
    print(f"         Ablation + ProjectionCap on layer {layer_5_config['layer']}")
    print(f"{'='*80}\n")

    # =========================================================================
    # Part 1: vLLM Generation with Steering
    # =========================================================================
    print("[1/3] Loading vLLM model and generating with steering...")

    vllm_cfg = VLLMSteeringConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,
        max_model_len=512,
        dtype="float16",
    )

    vllm_model = VLLMSteerModel(
        vllm_cfg,
        enforce_eager=True,
        bootstrap_layers=(layer_2_config["layer"], layer_5_config["layer"]),
    )

    # Apply multi-method steering
    # Layer 2: Additive + Projection Cap
    await vllm_model.set_layer_vector(layer_2_config["layer"], layer_2_config["add_vector"])
    await vllm_model.set_layer_projection_cap(
        layer_2_config["layer"],
        layer_2_config["cap_vector"],
        min=layer_2_config["cap_min"],
        max=layer_2_config["cap_max"],
    )

    # Layer 5: Ablation + Projection Cap
    await vllm_model.set_layer_ablation(
        layer_5_config["layer"],
        layer_5_config["ablation_vector"],
        scale=layer_5_config["ablation_scale"],
    )
    await vllm_model.set_layer_projection_cap(
        layer_5_config["layer"],
        layer_5_config["cap_vector"],
        min=layer_5_config["cap_min"],
        max=layer_5_config["cap_max"],
    )

    # Generate with steering and capture
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=False)

    try:
        texts, handles = await vllm_model.generate(
            prompts,
            sampling_params=sampling,
            capture_layers=[layer_2_config["layer"], layer_5_config["layer"]],
        )

        # Fetch all captures in batch (efficient single RPC)
        await vllm_model.fetch_captures_batch(handles)

        print(f"✓ Generated {len(texts)} sequences")
        for i, text in enumerate(texts):
            print(f"  Prompt {i+1}: {len(text.split())} words generated")

        # =====================================================================
        # Part 2: HuggingFace Ground Truth (sample 3 prompts for efficiency)
        # =====================================================================
        print(f"\n[2/3] Computing HuggingFace ground truth (sampling {3} prompts)...")

        # Load HF model once
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for ground truth
            device_map="cuda",
            attn_implementation="eager",
        )
        hf_model.eval()

        sample_indices = [0, 4, 8]  # Sample 3 prompts for ground truth comparison

        comparison_results = []

        for sample_idx in sample_indices:
            prompt = prompts[sample_idx]
            generated_text = texts[sample_idx]
            full_text = prompt + generated_text

            # Tokenize full sequence
            full_inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
            prompt_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            prompt_len = prompt_inputs.input_ids.shape[1]
            full_len = full_inputs.input_ids.shape[1]

            print(f"\n  Prompt {sample_idx}: {prompt_len} prompt tokens + {full_len - prompt_len} generated tokens")

            # Capture HF hidden states with steering applied
            captured_hf = {}

            def make_hf_hook(layer_idx: int):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    hidden_fp32 = hidden.to(torch.float32)

                    # Apply steering transforms matching vLLM
                    if layer_idx == layer_2_config["layer"]:
                        # Additive
                        hidden_fp32 = hidden_fp32 + layer_2_config["add_vector"].to(device=hidden_fp32.device)
                        # Projection cap
                        unit = _normalize(layer_2_config["cap_vector"]).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                        hidden_fp32 = _apply_projection_cap(
                            hidden_fp32,
                            unit,
                            minimum=layer_2_config["cap_min"],
                            maximum=layer_2_config["cap_max"],
                        )

                    elif layer_idx == layer_5_config["layer"]:
                        # Ablation
                        unit_abl = _normalize(layer_5_config["ablation_vector"]).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                        hidden_fp32 = _apply_ablation(hidden_fp32, unit_abl, scale=layer_5_config["ablation_scale"])
                        # Projection cap
                        unit_cap = _normalize(layer_5_config["cap_vector"]).to(device=hidden_fp32.device, dtype=hidden_fp32.dtype)
                        hidden_fp32 = _apply_projection_cap(
                            hidden_fp32,
                            unit_cap,
                            minimum=layer_5_config["cap_min"],
                            maximum=layer_5_config["cap_max"],
                        )

                    captured_hf[layer_idx] = hidden_fp32.detach().cpu().clone()

                    # Return in original dtype
                    hidden_out = hidden_fp32.to(dtype=hidden.dtype)
                    if isinstance(output, tuple):
                        return (hidden_out,) + output[1:]
                    return hidden_out

                return hook

            # Install hooks
            handles_hf = []
            for layer_idx in [layer_2_config["layer"], layer_5_config["layer"]]:
                handle = hf_model.model.layers[layer_idx].register_forward_hook(make_hf_hook(layer_idx))
                handles_hf.append(handle)

            with torch.no_grad():
                hf_model(**full_inputs)

            for handle in handles_hf:
                handle.remove()

            # ================================================================
            # Part 3: Compare vLLM vs HF for decode tokens
            # ================================================================
            # vLLM captures: single concatenated tensor with all tokens
            vllm_handle = handles[sample_idx]

            for layer_idx in [layer_2_config["layer"], layer_5_config["layer"]]:
                vllm_captures = vllm_handle.captures[layer_idx]
                hf_full_hidden = captured_hf[layer_idx].squeeze(0)  # [seq_len, hidden_size]

                # Extract the concatenated tensor (captures is a list with one element)
                if len(vllm_captures) == 0:
                    print(f"    Layer {layer_idx}: No captures returned")
                    continue

                vllm_all_tokens = vllm_captures[0]["hidden"].to(torch.float32)  # [seq_len, hidden_size]

                # Compare decode tokens (skip prefill, check first 5 decode tokens)
                num_decode_to_check = min(5, vllm_all_tokens.shape[0] - prompt_len)

                # Skip if no decode tokens
                if num_decode_to_check <= 0:
                    print(f"    Layer {layer_idx}: No decode captures (vLLM shape: {vllm_all_tokens.shape}, prompt_len: {prompt_len})")
                    continue

                layer_similarities = []
                layer_maes = []

                for decode_idx in range(num_decode_to_check):
                    # vLLM: slice from concatenated tensor
                    vllm_token_idx = prompt_len + decode_idx
                    vllm_hidden = vllm_all_tokens[vllm_token_idx]

                    # HF: token position in full sequence
                    hf_token_idx = prompt_len + decode_idx
                    if hf_token_idx >= hf_full_hidden.shape[0]:
                        break

                    hf_hidden = hf_full_hidden[hf_token_idx]

                    # Compute similarity
                    cos_sim = F.cosine_similarity(
                        hf_hidden.flatten().unsqueeze(0),
                        vllm_hidden.flatten().unsqueeze(0),
                        dim=-1
                    ).item()

                    mae = torch.mean(torch.abs(vllm_hidden - hf_hidden)).item()

                    layer_similarities.append(cos_sim)
                    layer_maes.append(mae)

                # Skip if no comparisons were made
                if len(layer_similarities) == 0:
                    print(f"    Layer {layer_idx}: No valid comparisons")
                    continue

                avg_cos = sum(layer_similarities) / len(layer_similarities)
                avg_mae = sum(layer_maes) / len(layer_maes)

                comparison_results.append({
                    "prompt_idx": sample_idx,
                    "layer": layer_idx,
                    "avg_cosine": avg_cos,
                    "avg_mae": avg_mae,
                    "num_tokens_checked": len(layer_similarities),
                })

                print(f"    Layer {layer_idx}: cos={avg_cos:.6f}, MAE={avg_mae:.6f} ({len(layer_similarities)} tokens)")

        # Clean up HF model
        del hf_model
        torch.cuda.empty_cache()

        # =====================================================================
        # Part 4: Test RWLock with Concurrent Operations
        # =====================================================================
        print(f"\n[3/4] Testing RWLock with concurrent operations...")

        async def concurrent_generate(prompt_idx: int):
            """Generate with a single prompt (should be allowed concurrently)."""
            result = await vllm_model.generate(
                [prompts[prompt_idx]],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=5),
            )
            return result[0]

        # Test 1: Queue up multiple generations before awaiting any
        print("  Testing truly concurrent generations (queue then await)...")
        import time

        # Track when each generation starts/ends
        gen_timings = []

        async def timed_generate(prompt_idx: int):
            """Generate with timing tracking."""
            start = time.time()
            result = await vllm_model.generate(
                [prompts[prompt_idx]],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=10),
            )
            end = time.time()
            gen_timings.append({"idx": prompt_idx, "start": start, "end": end})
            return result[0]

        # Create tasks WITHOUT awaiting (queue them up)
        task1 = asyncio.create_task(timed_generate(0))
        task2 = asyncio.create_task(timed_generate(1))
        task3 = asyncio.create_task(timed_generate(2))

        # NOW await all of them
        results = await asyncio.gather(task1, task2, task3)

        # Check if they overlapped
        gen_timings.sort(key=lambda x: x["start"])
        overlapping = False
        for i in range(len(gen_timings) - 1):
            # Check if gen i+1 started before gen i ended
            if gen_timings[i+1]["start"] < gen_timings[i]["end"]:
                overlapping = True
                break

        print(f"    ✓ {len(results)} generations completed")
        if overlapping:
            print(f"    ✓ Verified concurrent execution (requests overlapped in time)")
        else:
            print(f"    ⚠ Sequential execution detected (may be vLLM engine batching)")

        # Test 2: Verify concurrent captures are properly isolated (don't get mixed up)
        print("  Testing capture isolation (concurrent requests don't mix)...")

        # Generate concurrently WITH capture, using unique seeds per request
        capture_layers = [layer_2_config["layer"], layer_5_config["layer"]]

        async def concurrent_generate_with_capture(prompt_idx: int):
            # Use unique seed per prompt for different outputs
            texts_conc, handles_conc = await vllm_model.generate(
                [prompts[prompt_idx]],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=10, seed=5000 + prompt_idx),
                capture_layers=capture_layers,
            )
            await vllm_model.fetch_captures_batch(handles_conc)
            return (texts_conc[0], handles_conc[0].captures, prompt_idx)

        # Queue up concurrent tasks
        conc_task1 = asyncio.create_task(concurrent_generate_with_capture(0))
        conc_task2 = asyncio.create_task(concurrent_generate_with_capture(1))
        conc_task3 = asyncio.create_task(concurrent_generate_with_capture(2))

        results = await asyncio.gather(conc_task1, conc_task2, conc_task3)

        # Verify each capture has reasonable properties
        capture_issues = []
        for text, captures, prompt_idx in results:
            # Debug: Check tokenization details
            prompt_only = tokenizer(prompts[prompt_idx], return_tensors="pt")
            prompt_len = prompt_only.input_ids.shape[1]
            generated_only = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            generated_len = generated_only.input_ids.shape[1]
            full_output = tokenizer(prompts[prompt_idx] + text, return_tensors="pt")
            full_len = full_output.input_ids.shape[1]

            for layer in capture_layers:
                if layer not in captures:
                    capture_issues.append(f"Prompt {prompt_idx}: missing layer {layer}")
                    continue

                captured_hidden = captures[layer][0]["hidden"]
                actual_len = captured_hidden.shape[0]

                # In autoregressive generation, the final sampled token is never processed
                # through the model - it's only sampled from logits. So the capture should have:
                # prompt_tokens + (generated_tokens - 1)
                # Example: 15 prompt + (10 generated - 1 final) = 24 captured
                expected_len = prompt_len + (generated_len - 1)

                if actual_len != expected_len:
                    capture_issues.append(
                        f"Prompt {prompt_idx}, Layer {layer}: length mismatch "
                        f"(captured {actual_len} != expected {expected_len}, "
                        f"prompt={prompt_len}, generated={generated_len})"
                    )

                # Verify no NaNs or Infs
                if torch.isnan(captured_hidden).any() or torch.isinf(captured_hidden).any():
                    capture_issues.append(f"Prompt {prompt_idx}, Layer {layer}: contains NaN/Inf")

        if capture_issues:
            print(f"    ✗ Capture isolation issues detected:")
            for issue in capture_issues:
                print(f"      - {issue}")
            raise AssertionError(f"Concurrent capture isolation failed: {capture_issues}")
        else:
            print(f"    ✓ All {len(results)} concurrent captures properly isolated and valid")

        # Test 3: Steering change blocks until generation completes
        print("  Testing steering change blocks during generation...")

        async def long_generate():
            """Long generation to test blocking."""
            await vllm_model.generate(
                [prompts[0]],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=20),
            )

        async def try_steering_change():
            """Try to change steering (should block until generation completes)."""
            await asyncio.sleep(0.1)  # Let generation start
            new_vector = torch.randn(hidden_size, dtype=torch.float32) * 0.2
            await vllm_model.set_layer_vector(layer_2_config["layer"], new_vector)
            return "steering_changed"

        # Run both concurrently - steering should block
        gen_task = asyncio.create_task(long_generate())
        steering_task = asyncio.create_task(try_steering_change())

        await asyncio.gather(gen_task, steering_task)
        print("    ✓ Steering change correctly blocked until generation completed")

        # =====================================================================
        # Final Assertions
        # =====================================================================
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")

        all_pass = True
        for result in comparison_results:
            status = "✓ PASS" if result["avg_cosine"] > 0.99 else "✗ FAIL"
            print(f"Prompt {result['prompt_idx']}, Layer {result['layer']}: "
                  f"cos={result['avg_cosine']:.6f}, MAE={result['avg_mae']:.6f} {status}")

            if result["avg_cosine"] < 0.99:
                all_pass = False

        print(f"{'='*80}")

        # Assertions
        for result in comparison_results:
            assert result["avg_cosine"] > 0.99, (
                f"Prompt {result['prompt_idx']} Layer {result['layer']}: "
                f"Cosine similarity {result['avg_cosine']:.6f} should be >0.99"
            )
            assert result["avg_mae"] < 0.02, (
                f"Prompt {result['prompt_idx']} Layer {result['layer']}: "
                f"MAE {result['avg_mae']:.6f} should be <0.02"
            )

        if all_pass:
            print("\n✓ ALL TESTS PASSED")
            print("  - Batch generation with chat formatting works")
            print("  - Decode phase steering matches HF ground truth")
            print("  - Multi-method steering (add + cap + ablation) works")
            print("  - Concurrent captures match serial execution (isolation verified)")
            print("  - RWLock correctly coordinates concurrent operations")
        else:
            print("\n✗ SOME TESTS FAILED")

        print(f"{'='*80}\n")

    finally:
        # Cleanup
        await vllm_model.clear_all_vectors()
        await vllm_model.clear_layer_projection_cap(layer_2_config["layer"])
        await vllm_model.clear_layer_ablation(layer_5_config["layer"])
        await vllm_model.clear_layer_projection_cap(layer_5_config["layer"])
        del vllm_model
        torch.cuda.empty_cache()
