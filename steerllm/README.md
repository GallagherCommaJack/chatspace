# steerllm

Multi-backend LLM steering library. Apply steering vectors, projection caps, and ablations to large language models during inference.

## Installation

```bash
# Core only (just specs and utilities)
pip install steerllm

# vLLM backend for production inference
pip install steerllm[vllm]

# HuggingFace for training
pip install steerllm[huggingface]

# Everything
pip install steerllm[all]
```

## Quick Start

### Basic Steering (vLLM)

```python
import asyncio
from steerllm import VLLMSteeringModel, SteeringSpec
import torch

async def main():
    model = VLLMSteeringModel("Qwen/Qwen3-0.6B")

    # Create steering spec
    direction = torch.randn(model.hidden_size)
    steering = SteeringSpec.simple_add(layer=5, vector=direction, scale=2.0)

    # Generate with steering
    texts, _ = await model.generate(
        ["What is consciousness?"],
        max_tokens=100,
        steering_spec=steering,
    )
    print(texts[0])

asyncio.run(main())
```

### Activation Capture

```python
async def capture_example():
    model = VLLMSteeringModel("Qwen/Qwen3-0.6B")

    texts, handles = await model.generate(
        ["The meaning of life is"],
        max_tokens=50,
        capture_layers=[5, 10, 15],
    )

    async with handles[0] as handle:
        await handle.fetch()
        layer_5 = handle.captures[5][0]["hidden"]
        print(f"Layer 5 shape: {layer_5.shape}")
```

### Training (HuggingFace)

```python
from steerllm.backends.huggingface import HFSteeringModel
from torch.optim import Adam

model = HFSteeringModel("Qwen/Qwen3-0.6B", target_layers=[5])
optimizer = Adam(model.get_trainable_parameters(), lr=1e-4)

for batch in dataloader:
    outputs = model(**batch)
    loss = compute_loss(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.save_steering("./checkpoint")
```

## Features

- **Per-request steering**: Different requests can use different steering configs
- **Composable operations**: Add + Cap + Ablation in any order per layer
- **Zero-copy capture**: Shared memory IPC for fast activation transfer (vLLM)
- **Multi-backend**: vLLM for production, HuggingFace for training
- **Tensor parallelism**: Works with vLLM's TP support

## Steering Operations

### AddSpec
Additive steering: `hidden += vector * scale`

### ProjectionCapSpec
Clamp projection onto a direction: bounds the component along a direction

### AblationSpec
Scale component along direction: `scale=0` removes it entirely

## License

MIT
