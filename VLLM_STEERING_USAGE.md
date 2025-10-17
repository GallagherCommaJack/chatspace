# vLLM Steering Vector Support - IMPORTANT COMPATIBILITY NOTE

## ⚠️ Current Status: vLLM 0.11+ V1 Engine Incompatibility

**The hook-based steering approach does NOT currently work with vLLM 0.11+ (V1 engine)** due to architectural changes.

### Why It Doesn't Work

vLLM 0.11+ uses a **V1 engine architecture** where the model runs in a **separate worker process** (`EngineCore_DP0`). This means:
- The main process cannot directly access the model to install PyTorch hooks
- The `engine.model_executor` attribute doesn't exist in V1
- Model access requires IPC (inter-process communication)

### Recommendations

1. **Use HuggingFace Transformers** (current, working implementation)
   - `QwenSteerModel` works perfectly with hook-based steering
   - Full control over model and hooks
   - Recommended for steering vector research

2. **Wait for vLLM V1 steering support** (future work)
   - Would require logit processor approach or worker process patching
   - More complex implementation
   - May have limitations compared to direct hooks

3. **Use vLLM 0.6.x** (requires torch downgrade)
   - Has V0 architecture with `model_executor`
   - Incompatible with torch 2.8+ (project requirement)

## Original Design (for reference)

The chatspace project was designed to support **two backends** for steering vector rollouts:

1. **HuggingFace Transformers** (✅ WORKING): Original implementation using `transformers.AutoModelForCausalLM`
2. **vLLM** (⚠️  NOT COMPATIBLE): High-throughput implementation - blocked by V1 architecture

Both backends were intended to use the same **hook-based steering** approach: a PyTorch forward hook intercepts the output of a specified transformer layer and adds a steering vector to the residual stream.

## Key Features

- ✅ **Same API**: Both backends implement the `SteerableModel` interface
- ✅ **Hook-based steering**: Consistent behavior across backends (no re-implementation needed)
- ✅ **High throughput**: vLLM's PagedAttention and continuous batching for faster generation
- ✅ **Multi-GPU support**: Tensor parallelism via `--tensor-parallel-size`
- ✅ **Quantization**: vLLM supports GPTQ, AWQ, and other quantization schemes
- ✅ **Backward compatible**: Existing rollout scripts work without modification

## Architecture

### New Module: `chatspace/generation/`

```
chatspace/generation/
├── __init__.py              # Package exports
├── base.py                  # SteerableModel abstract interface
├── config.py                # GenerationConfig dataclass
└── vllm_steer_model.py      # VLLMSteerModel implementation
```

### Key Classes

**`SteerableModel` (Abstract Base)**
```python
class SteerableModel(ABC):
    @abstractmethod
    def set_vector(self, vector: torch.Tensor | None) -> None: ...

    @abstractmethod
    def set_target_layer(self, layer_idx: int) -> None: ...

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any: ...
```

**`VLLMSteerModel`**
- Wraps `vllm.LLM` with steering capabilities
- Accesses underlying model layers via `llm.llm_engine.model_executor...`
- Installs forward hooks at specified layer
- Supports multi-layer steering through `set_layer_vector`
- Adds advanced controls via `set_layer_projection_cap` and `set_layer_ablation`
- Supports Qwen, Gemma, and other vLLM-compatible models
- Optional `bootstrap_layers` pre-allocates worker buffers for known layer IDs

**`LayerSteeringSpec` / `SteeringSpec` / `ProjectionCapSpec` / `AblationSpec`**
- Lightweight dataclasses that capture per-layer and full-model steering state
- Support cloning + serialization via `export_steering_spec` / `apply_steering_spec`
- Persisted alongside vectors when calling `save_pretrained`

**`QwenSteerModel` (Updated)**
- Now inherits from `SteerableModel` base class
- Maintains backward compatibility
- No functional changes to existing code

## Usage

### Basic Usage: Generate Rollouts with vLLM

```bash
# Use vLLM backend instead of HuggingFace
uv run python scripts/generate_behavior_rollouts.py \
  --datasets qwen-3-32b__trait__analytical \
  --model Qwen/Qwen3-32B \
  --use-vllm \
  --rollouts 5
```

### Multi-GPU with Tensor Parallelism

```bash
# Use 4 GPUs for tensor parallelism
uv run python scripts/generate_behavior_rollouts.py \
  --datasets qwen-3-32b__trait__analytical \
  --model Qwen/Qwen3-32B \
  --use-vllm \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --rollouts 10
```

### Compare HF vs vLLM Outputs

```bash
# Generate with HuggingFace (baseline)
uv run python scripts/generate_behavior_rollouts.py \
  --datasets qwen-3-32b__trait__analytical \
  --model Qwen/Qwen3-32B \
  --output-root /workspace/steering_rollouts_hf \
  --rollouts 3

# Generate with vLLM (comparison)
uv run python scripts/generate_behavior_rollouts.py \
  --datasets qwen-3-32b__trait__analytical \
  --model Qwen/Qwen3-32B \
  --use-vllm \
  --output-root /workspace/steering_rollouts_vllm \
  --rollouts 3
```

## Command-Line Arguments

### New vLLM-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-vllm` | flag | False | Enable vLLM backend instead of HuggingFace |
| `--tensor-parallel-size` | int | 1 | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | float | 0.9 | GPU memory fraction to use (0.0-1.0) |

### Existing Arguments (Work with Both Backends)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | Qwen/Qwen3-32B | Model name to load |
| `--target-layer` | int | 22 | Layer index for steering injection |
| `--rollouts` | int | 1 | Number of rollouts per variant |
| `--max-new-tokens` | int | 256 | Maximum generation length |
| `--temperature` | float | 0.7 | Sampling temperature |
| `--top-p` | float | 0.9 | Nucleus sampling parameter |
| `--trained-scales` | float[] | [1.0] | Scaling factors for trained steering vectors |
| `--activation-scales` | float[] | [1.0] | Scaling factors for activation vectors |
| `--normalize-steering` | flag | False | L2-normalize vectors before scaling |

## Testing

### Quick Test with Small Model

```bash
# Test basic vLLM steering with a small model (faster)
uv run python scripts/test_vllm_steering.py --model Qwen/Qwen2.5-3B
```

**Expected Output:**
```
======================================================================
vLLM Steering Vector Test Suite
======================================================================
[1/4] Testing basic model loading and generation with Qwen/Qwen2.5-3B
  Loading model...
  ✓ Model loaded successfully
  Generating text without steering...
  ✓ Generation successful

[2/4] Testing steering vector setting
  ✓ Steering vector set successfully
  ✓ Steering vector verified
  ✓ Steering vector cleared successfully

[3/4] Testing layer switching
  ✓ Switched to layer 18
  ✓ Switched back to layer 16

[4/4] Testing steering effect on generation
  ✓ Steering affects generation (outputs differ)

======================================================================
✓ All tests passed!
======================================================================
```

### Full Integration Test

```bash
# Generate rollouts for one dataset using vLLM
uv run python scripts/generate_behavior_rollouts.py \
  --datasets qwen-3-32b__trait__analytical \
  --model Qwen/Qwen3-32B \
  --use-vllm \
  --rollouts 1 \
  --max-new-tokens 128
```

## Performance Comparison

### Expected Speedups (Approximate)

| Scenario | HF Transformers | vLLM | Speedup |
|----------|----------------|------|---------|
| Single GPU, batch=1 | 1x | 1.5-2x | Faster inference |
| Single GPU, batch=8 | 1x | 2-3x | Better batching |
| 4 GPUs, TP=4 | N/A | 3-4x | Tensor parallelism |
| Large rollouts (1000s) | 1x | 3-5x | Continuous batching |

**Note:** Actual speedups depend on model size, sequence length, and hardware.

## Implementation Details

### Hook Installation

The HuggingFace backend attaches hooks directly from the driver process.  
The vLLM backend now mirrors that behaviour from inside the worker processes:

```python
from chatspace.vllm_steering import runtime

# Executed via EngineCore.collective_rpc on every worker
runtime.initialize_worker_state(worker, (target,))
runtime.set_worker_vector(worker, target, steering_vector)
runtime.set_worker_vector(worker, other_layer, other_vector)  # Optional extra layers
# Clear a specific layer when done
runtime.clear_worker_vector(worker, target)
```

Each worker keeps 1D tensors in device memory keyed by layer index and a forward hook that adds the active vectors to the residual stream before logits are computed. Drivers must explicitly manage layer indices—there is no implicit "default"—so `VLLMSteerModel` exposes explicit mutators: `set_layer_vector` for additive steering, `set_layer_projection_cap` for clamping the residual projection, and `set_layer_ablation` for scaling the projection component.

```python
from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

cfg = VLLMSteeringConfig(model_name="Qwen/Qwen3-0.6B")
model = VLLMSteerModel(cfg, bootstrap_layers=(22,))
model.set_layer_vector(22, steering_tensor)
model.set_layer_vector(30, other_tensor)   # Optional extra layer
model.set_layer_projection_cap(22, cap_tensor, cap_below=-1.0, cap_above=1.5)
model.set_layer_ablation(22, ablation_tensor, scale=0.25)
model.clear_layer_vector(22)
model.clear_layer_projection_cap(22)
model.clear_layer_ablation(22)
model.clear_all_vectors()
```

### Snapshot & Restore Steering

```python
from chatspace.generation import LayerSteeringSpec, SteeringSpec

# Capture the baseline configuration for later reuse.
baseline = model.export_steering_spec()

# Build an override that tweaks multiple layers.
override = SteeringSpec(
    layers={
        10: LayerSteeringSpec(vector=steering_a),
        18: LayerSteeringSpec(vector=steering_b, ablation=ablation_spec),
    }
)

# Apply temporarily; previous steering is automatically restored on exit.
with model.steering(override):
    run_eval(model)

# Or reapply manually at any time.
model.apply_steering_spec(baseline)
```

### Generation API Differences

**HuggingFace:**
```python
# Tokenize inputs
encoded = tokenizer(texts, return_tensors="pt", padding=True)
# Generate token IDs
outputs = model.generate(**encoded, max_new_tokens=256, ...)
# Decode outputs
texts = tokenizer.batch_decode(outputs[:, input_lens:])
```

**vLLM:**
```python
# Pass text prompts directly (vLLM handles tokenization)
sampling_params = SamplingParams(max_tokens=256, ...)
# Generate text completions directly
outputs = llm.generate(prompts, sampling_params)
texts = [output.outputs[0].text for output in outputs]
```

## Supported Models

Both Qwen and Gemma models are supported:

- ✅ **Qwen/Qwen3-32B** (default)
- ✅ **Qwen/Qwen2.5-3B** (for testing)
- ✅ **google/gemma-2-27b**
- ✅ Any model with a standard transformer architecture

**Layer access pattern:**
- Models with `model.model.layers` structure (Qwen, Gemma, Llama, etc.)
- Auto-detection of hidden size from `model.config.hidden_size`

## Troubleshooting

### Issue: "Could not access underlying model from vLLM engine"

**Cause:** vLLM's internal structure changed or model architecture is non-standard.

**Fix:** Update `_get_base_model()` in `vllm_steer_model.py` to handle new structure:
```python
def _get_base_model(self):
    executor = self.llm.llm_engine.model_executor
    # Try different access patterns
    if hasattr(executor, 'driver_worker'):
        return executor.driver_worker.model_runner.model
    # Add more fallbacks as needed
```

### Issue: Outputs identical between steered and baseline

**Cause:** Random steering vectors may not have semantic meaning.

**Fix:** Use **trained** or **activation-based** vectors instead:
```bash
# This automatically loads trained vectors from /workspace/steering_runs/
uv run python scripts/generate_behavior_rollouts.py \
  --datasets qwen-3-32b__trait__analytical \
  --use-vllm \
  --trained-scales 1.0 2.0
```

### Issue: Out of memory with vLLM

**Cause:** vLLM pre-allocates GPU memory for KV cache.

**Fix:** Reduce `--gpu-memory-utilization`:
```bash
--gpu-memory-utilization 0.7  # Use 70% instead of 90%
```

Or use tensor parallelism to split across GPUs:
```bash
--tensor-parallel-size 2  # Split across 2 GPUs
```

## Future Enhancements

Potential improvements (not yet implemented):

1. **Quantization support**: Add `--quantization` flag for AWQ/GPTQ
2. **Batch optimization**: Batch multiple steering scales in a single pass
3. **vLLM-specific sampling**: Explore logit processors as alternative to hooks
4. **Distributed rollouts**: Coordinate vLLM instances across multiple nodes
5. **Streaming generation**: Real-time token streaming with steering

## References

- **vLLM Documentation**: https://docs.vllm.ai/
- **Original steering paper**: [Steering language models with activation engineering]
- **Chatspace project**: `/root/chatspace/`
- **Implementation**: `chatspace/generation/vllm_steer_model.py`

## Examples

### Example 1: High-throughput trait rollouts

```bash
# Generate 100 rollouts for all analytical trait variants with 4 GPUs
uv run python scripts/generate_behavior_rollouts.py \
  --datasets qwen-3-32b__trait__analytical \
  --model Qwen/Qwen3-32B \
  --use-vllm \
  --tensor-parallel-size 4 \
  --rollouts 100 \
  --trained-scales 0.5 1.0 2.0 \
  --activation-scales -2.0 -1.0 1.0 2.0
```

### Example 2: Distributed rollouts across GPUs

```bash
# Worker 0: process datasets 0, 4, 8, ... (on GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_behavior_rollouts.py \
  --log /workspace/steering_runs/steering_sweep.log \
  --use-vllm \
  --dataset-stride 4 \
  --dataset-offset 0 &

# Worker 1: process datasets 1, 5, 9, ... (on GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/generate_behavior_rollouts.py \
  --log /workspace/steering_runs/steering_sweep.log \
  --use-vllm \
  --dataset-stride 4 \
  --dataset-offset 1 &

# Workers 2 and 3 (similar pattern)
```

## Questions or Issues?

- Check the test script: `scripts/test_vllm_steering.py`
- Review implementation: `chatspace/generation/vllm_steer_model.py`
- Compare with original: `chatspace/steering/model.py` (QwenSteerModel)
