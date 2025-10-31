# chatspace

A **vLLM steering runtime** for language model activation capture and interpretability research.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Steering Methods](#steering-methods)
- [Activation Capture](#activation-capture)
- [Concurrency Model](#concurrency-model)
- [Advanced Features](#advanced-features)
- [Development](#development)

---

## Overview

**chatspace** provides a production-ready system for applying steering vectors and capturing activations from vLLM-hosted language models with support for:

- **Additive steering vectors**: Inject concept directions into layer activations
- **Projection capping**: Clamp hidden state components along specific directions
- **Component ablation**: Scale or suppress features for circuit analysis
- **Per-request activation capture**: Capture hidden states during generation for analysis
- **Concurrent generation**: Thread-safe steering updates with async readers-writer lock
- **Tensor parallelism**: Works transparently with vLLM's multi-GPU parallelism

> **Note**: This repo also contains some research code related to dataset embedding and persona subspace analysis (collaboration with [persona-subspace](https://github.com/lu-christina/persona-subspace)), but the vLLM steering runtime is the primary public-facing feature.

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/chatspace.git
cd chatspace

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- vLLM 0.6.0+ (for steering features)
- CUDA-capable GPU (recommended)

---

## Quick Start

```python
import torch
import asyncio
from vllm import SamplingParams
from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
)

async def main():
    # Initialize model
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    model = VLLMSteerModel(cfg, bootstrap_layers=(2, 4, 6))

    # Generate baseline
    sampling = SamplingParams(temperature=0.0, max_tokens=64)
    baseline = await model.generate(
        "Question: What is the capital of France? Answer:",
        sampling
    )
    print("Baseline:", baseline[0])

    # Apply steering vector to layer 4
    steering_vector = torch.randn(model.hidden_size) * 100.0
    await model.set_layer_vector(4, steering_vector)

    # Generate with steering
    steered = await model.generate(
        "Question: What is the capital of France? Answer:",
        sampling
    )
    print("Steered:", steered[0])

    # Clear steering
    await model.clear_all_vectors()

if __name__ == "__main__":
    asyncio.run(main())
```

**Run the smoke test:**
```bash
uv run python scripts/steering_smoke.py \
    --model-name "Qwen/Qwen3-0.6B" \
    --layer 2 \
    --scale 5000.0 \
    --max-tokens 32
```

---

## Core Concepts

#### 1. VLLMSteerModel

The main interface for steering and generation. Wraps vLLM's `AsyncLLMEngine` and provides:

- **Async-first API**: All generation and steering operations are async
- **Multi-layer steering**: Apply different steering operations to different transformer layers
- **Thread-safe**: Concurrent generation requests are safe; steering updates block during generation
- **Worker coordination**: Steering vectors are broadcast to all tensor-parallel workers via RPC

#### 2. Steering Specifications

Steering state is organized into layer-wise specifications:

```python
from chatspace.generation.vllm_steer_model import (
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
)

# Create a steering spec
spec = SteeringSpec(layers={
    2: LayerSteeringSpec(
        add=AddSpec(vector=unit_vec, scale=50.0),
    ),
    4: LayerSteeringSpec(
        add=AddSpec(vector=unit_vec, scale=100.0),
        projection_cap=ProjectionCapSpec(
            vector=direction_vec,
            min=-10.0,
            max=10.0,
        ),
    ),
})

# Apply spec (all steering updates happen atomically)
await model.apply_steering_spec(spec)

# Export current steering state
current_spec = model.export_steering_spec()
```

#### 3. Eager Execution Requirement

**IMPORTANT**: vLLM steering requires `enforce_eager=True` (enabled by default). CUDA graph compilation skips the Python-side steering hooks.

```python
# This is the default and recommended:
model = VLLMSteerModel(cfg)  # enforce_eager=True by default

# If you try to disable it, you'll get a warning:
model = VLLMSteerModel(cfg, enforce_eager=False)
# WARNING: vLLM steering requires enforce_eager=True; overriding user-supplied value.
```

---

## Steering Methods

#### Additive Steering

Add a fixed vector to layer activations:

```python
# Apply to specific layer
steering_vec = torch.randn(model.hidden_size) * 50.0
await model.set_layer_vector(layer_idx=4, vector=steering_vec)

# Or use the active layer (set via set_target_layer)
model.set_target_layer(4)
await model.set_vector(steering_vec)

# Clear steering for a layer
await model.clear_layer_vector(4)
```

**Use cases:**
- Concept steering (e.g., "make outputs more formal")
- Feature injection from trained steering vectors
- Behavioral modification (e.g., refusal prevention)

#### Projection Capping

Clamp the component of activations along a direction:

```python
import torch

# Define direction (will be normalized automatically)
direction = torch.randn(model.hidden_size)

# Cap projection to [-10, 10] range
await model.set_layer_projection_cap(
    layer_idx=6,
    vector=direction,
    min=-10.0,
    max=10.0,
)

# Remove cap
await model.clear_layer_projection_cap(6)
```

**Use cases:**
- Prevent extreme activations in specific directions
- Constrain steering vector effects
- Stabilize generation under strong steering

**Note**: Projection capping operates on the full hidden state (after adding steering vectors), computing `hidden @ direction` and clamping to `[min, max]`.

#### Component Ablation

Scale (amplify or suppress) activations along a direction:

```python
# Suppress component (scale < 1.0)
await model.set_layer_ablation(
    layer_idx=8,
    vector=direction,
    scale=0.1,  # Reduce to 10%
)

# Amplify component (scale > 1.0)
await model.set_layer_ablation(
    layer_idx=8,
    vector=direction,
    scale=2.0,  # Double the component
)

# Clear ablation
await model.clear_layer_ablation(8)
```

**Use cases:**
- Interpretability research (what happens when we remove a feature?)
- Circuit analysis (ablate specific features)
- Causal intervention experiments

---

## Activation Capture

Capture hidden states during generation for analysis:

```python
# Capture activations from layers 2, 4, 6
results, handles = await model.generate(
    ["What is 2+2?", "What is the capital of France?"],
    sampling,
    capture_layers=[2, 4, 6],
)

# Fetch captures (batched fetch is more efficient)
await model.fetch_captures_batch(handles)

# Access captures for each request
for i, handle in enumerate(handles):
    print(f"\nPrompt {i}: {results[i]}")

    for layer_idx in handle.layer_indices:
        # Each layer has a list (one per TP worker)
        captures = handle.captures[layer_idx]
        hidden = captures[0]["hidden"]  # [seq_len, hidden_size]
        print(f"  Layer {layer_idx}: {hidden.shape}")
```

**Capture behavior:**
- **Concatenated format**: Returns a single tensor per layer containing all tokens processed (prefill + decode)
- **Length calculation**: Captured length is `prompt_tokens + (generated_tokens - 1)` because the final generated token is sampled but never processed through the model
- **Isolation**: Concurrent requests with capture enabled maintain proper per-request isolation
- **Thread safety**: Captures are accumulated during generation and fetched after completion

**Example: Analyze steering effects**

```python
# Capture without steering
baseline_results, baseline_handles = await model.generate(
    prompt,
    sampling,
    capture_layers=[4],
)
await model.fetch_captures_batch(baseline_handles)
baseline_acts = baseline_handles[0].captures[4][0]["hidden"]

# Apply steering and capture again
await model.set_layer_vector(4, steering_vec)
steered_results, steered_handles = await model.generate(
    prompt,
    sampling,
    capture_layers=[4],
)
await model.fetch_captures_batch(steered_handles)
steered_acts = steered_handles[0].captures[4][0]["hidden"]

# Compare activations
delta = steered_acts - baseline_acts
print(f"Mean activation change: {delta.mean().item():.4f}")
print(f"Max activation change: {delta.abs().max().item():.4f}")
```

---

## Concurrency Model

#### Readers-Writer Lock (AsyncRWLock)

`VLLMSteerModel` uses an async readers-writer lock to coordinate operations:

- **Read operations (concurrent)**: Multiple `generate()` calls can run simultaneously
- **Write operations (exclusive)**: Steering updates block until all in-flight requests complete

**Read operations** (acquire read lock):
- `generate()`
- `chat()`

**Write operations** (acquire write lock):
- `set_layer_vector()`, `set_vector()`
- `set_layer_projection_cap()`, `clear_layer_projection_cap()`
- `set_layer_ablation()`, `clear_layer_ablation()`
- `apply_steering_spec()`, `push_steering_spec()`, `pop_steering_spec()`
- `clear_layer_vector()`, `clear_all_vectors()`

#### Concurrent Generation Example

```python
import asyncio

async def generate_many(model, prompts, sampling):
    """Run multiple concurrent generation requests."""
    tasks = [
        model.generate(prompt, sampling)
        for prompt in prompts
    ]
    results = await asyncio.gather(*tasks)
    return [r[0] for r in results]

# This is safe and performant:
prompts = [f"Prompt {i}" for i in range(10)]
results = await generate_many(model, prompts, sampling)
```

**Important**: Steering changes during concurrent generation will wait for all in-flight requests to complete:

```python
async def concurrent_steer_test():
    # Start long generation
    gen_task = asyncio.create_task(
        model.generate("Write a long story...", sampling)
    )

    # Try to update steering (will block until generation completes)
    await model.set_layer_vector(4, new_steering_vec)

    result = await gen_task
    # Steering update applied AFTER generation completed
```

---

## Advanced Features

#### Steering Context Manager

Temporarily apply steering and restore previous state:

```python
# Save current steering state
spec = SteeringSpec(layers={
    4: LayerSteeringSpec(add=AddSpec(vector=vec, scale=100.0))
})

async with model.steering(spec):
    # Steering active within this block
    results = await model.generate(prompts, sampling)
    # ...

# Previous steering automatically restored after exiting block
```

#### Steering Stack

Push and pop steering configurations:

```python
# Save baseline
await model.push_steering_spec(baseline_spec)

# Apply intervention
await model.apply_steering_spec(intervention_spec)
results_1 = await model.generate(prompts, sampling)

# Try different steering
await model.apply_steering_spec(alternative_spec)
results_2 = await model.generate(prompts, sampling)

# Restore baseline
await model.pop_steering_spec()
results_baseline = await model.generate(prompts, sampling)
```

#### Chat-style Generation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

responses = await model.chat(
    messages,
    sampling_params=sampling,
)
print(responses[0])
```

#### Tensor Parallel Support

The steering runtime is designed to work with vLLM's tensor parallelism:

```python
cfg = VLLMSteeringConfig(
    model_name="Qwen/Qwen3-32B",
    tensor_parallel_size=4,  # Multi-GPU
)
model = VLLMSteerModel(cfg, bootstrap_layers=(0, 15, 31))

# Steering vectors are broadcast to all workers
await model.set_layer_vector(15, steering_vec)
```

**Implementation notes:**
- Steering vectors are broadcast to all TP ranks via `collective_rpc`
- Each worker stores the full-size vector (memory cost is `O(hidden_size)` per rank)
- No distributed operations needed in steering code (vLLM's `RowParallelLinear` handles allreduce)

---

## Development

### Running Tests

```bash
# All tests
uv run pytest tests/

# Specific test
uv run pytest tests/test_vllm_comprehensive_integration.py -v

# With coverage
uv run pytest tests/ --cov=chatspace --cov-report=html
```

**Important**: Always run tests with timeouts - bugs can cause GPU hangs.

### Project Structure

```
chatspace/
  chatspace/
    vllm_steering/       # vLLM steering runtime
      runtime.py       # Worker-side patching & RPC handlers
    generation/
      vllm_steer_model.py  # Client-side steering API
      base.py          # Abstract base classes
    hf_embed/            # SentenceTransformer embedding pipeline
    cli.py               # Command-line interface
  scripts/
    steering_smoke.py    # Quick steering verification
  tests/
    test_vllm_comprehensive_integration.py  # End-to-end tests
    test_*.py            # Unit tests
  README.md                # This file
```

---

## References

- **vLLM Documentation**: https://docs.vllm.ai/
- **Steering Vectors**: Representation Engineering papers (Li et al., 2023)
- **Activation Capture**: Interpretability research (Anthropic, OpenAI)

---

## License

[Your license here]

## Citation

```bibtex
@software{chatspace2025,
  title={chatspace: vLLM Steering Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/chatspace}
}
```
