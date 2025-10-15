# vLLM Steering Implementation - Status and Next Steps

## Current Situation

I implemented a complete vLLM steering solution using the same hook-based approach as `QwenSteerModel`, but discovered that **vLLM 0.11.0's V1 engine architecture is incompatible** with this approach.

### What Was Built

✅ **Complete implementation** (in `chatspace/generation/`):
- `base.py` - Abstract `SteerableModel` interface
- `vllm_steer_model.py` - `VLLMSteerModel` class with hook installation
- `config.py` - Unified generation config
- Updated `generate_behavior_rollouts.py` with `--use-vllm` flag

❌ **Cannot access model** in vLLM V1:
- Model runs in separate process (`EngineCore_DP0`)
- Main process has no direct model access
- `engine.model_executor` doesn't exist in V1

## Options Moving Forward

### Option 1: Use HF Transformers (✅ Recommended)

Your current `QwenSteerModel` implementation works perfectly and has:
- ✅ Direct model access for hooks
- ✅ Full control over steering injection
- ✅ Works with Qwen3-32B and Gemma-2-27B
- ✅ Proven approach from research

**Optimizations you can make:**
1. Increase batch size in rollouts
2. Use `torch.compile()` for faster inference
3. Parallelize across GPUs with `CUDA_VISIBLE_DEVICES`
4. Use bf16/fp16 for memory efficiency

### Option 2: Implement vLLM V1 Steering (🔧 Requires Significant Work)

To make vLLM V1 work, we'd need to:

**Approach A: Worker Process Patching**
```python
# Monkey-patch vLLM's ModelRunner to install hooks after model load
# This would require:
1. Hook into vLLM's model loading pipeline
2. Install steering hooks in the worker process
3. Pass steering vectors via RPC/IPC to worker
4. Handle synchronization between processes
```

**Approach B: Custom Logit Processors**
```python
# Instead of modifying hidden states, modify logits before sampling
# Limitations:
- Can't directly inject at specific layers
- Would need to approximate steering effect at logit level
- Less precise than hook-based approach
- May not achieve same behavioral control
```

**Approach C: Fork vLLM or Use Private Copy**
- Maintain a custom vLLM version with steering support
- High maintenance burden
- Breaks when vLLM updates

### Option 3: Use vLLM 0.6.x with Torch Downgrade (⚠️ Not Recommended)

This would require:
1. Downgrade torch from 2.8 to 2.5.1
2. Risk breaking other dependencies
3. Use outdated vLLM (missing optimizations)
4. May not support Qwen3-32B properly

## My Recommendation

**Stick with HuggingFace Transformers for now.**

Here's why:
1. It works perfectly for your steering research
2. You have full control over the model and hooks
3. The hook-based approach is clean and well-understood
4. Performance is adequate for research purposes
5. No risk of compatibility issues

**If you need higher throughput later:**
- We can implement multi-GPU parallelization with HF transformers
- Use `torch.compile()` for 2-3x speedup
- Batch more aggressively
- Profile and optimize bottlenecks

## Alternative: Investigate vLLM Worker Access

If you really want vLLM, I can investigate whether we can:
1. Access the model in the worker process via `engine_core.collective_rpc()`
2. Install hooks by patching vLLM's `GPUModelRunner` class
3. Pass steering vectors through vLLM's IPC mechanism

This would be experimental and might break with vLLM updates.

## What Would You Like To Do?

Please let me know your preference:

**A)** Stick with HF transformers + optimize current implementation
**B)** Attempt vLLM V1 integration (worker patching approach)
**C)** Implement logit processor approach (less precise steering)
**D)** Something else?

## Code Status

All the vLLM code I wrote is ready to use once we solve the model access problem. The architecture is solid:
- `chatspace/generation/` - Clean module structure
- `SteerableModel` interface - Works for both backends
- Hook installation logic - Already implemented
- Rollout script integration - Already done

We just need a way to install the hooks in vLLM's worker process.
