# Capture Hook Optimization Findings

**Date:** 2025-10-31
**Branch:** `optimize-capture-hotpath`
**Baseline:** `async_activation_capture_claude`

## Isolated Benchmark Results (Unoptimized)

**Configuration:** Qwen2.5-3B, batch=32, prefill=512, decode=128

### Baseline (No Capture)
- Mean generation: **36.53s**
- Zero overhead by definition

### Zero-Layer Capture (Metadata Overhead Only)
- Mean generation: **37.21s**
- Overhead: **+1.9%** (0.68s)
- Excellent! Metadata guard optimization working well

### All-Layers Capture (36 layers)
**Severe Performance Degradation Across Iterations:**
- Iteration 1: 57.24s gen + 13.94s fetch = 71.18s total
- Iteration 2: 85.63s gen + 5.01s fetch = 90.64s total (+27% worse)
- Iteration 3: 112.90s gen + 4.94s fetch = 117.84s total (+65% worse than iter 1!)

**Mean:** 85.26s gen + 7.96s fetch = 93.22s total
**vs Baseline:** +133% generation overhead, +155% total overhead

## Root Cause Analysis

### Problem: torch.cat() Per Decode Token

The decode loop was calling `torch.cat()` for EVERY single decode token:
```python
# 128 decode × 36 layers × 32 batch = 147,456 concatenations!
state.request_captures[req_id][layer_idx] = torch.cat([existing, new], dim=0)
```

**Why This Causes Degradation:**
1. Each cat creates a new tensor, old one becomes garbage
2. Memory fragmentation increases with each operation
3. Allocation overhead compounds over time
4. Python dict/list performance degrades as they grow

**Evidence:**
- Generation time increased 97% from iter 1 to iter 3
- Fetch time improved (fewer, larger tensors)
- Pattern consistent with memory pressure/fragmentation

## Optimizations Implemented

### 1. Reduce Dictionary Lookups (Commit c5544fb)
```python
# Before: Multiple lookups
if req_id not in state.active_capture_requests: ...
if layer_idx not in state.active_capture_requests[req_id]: ...

# After: Single lookup, cache reference
req_layers = active_reqs.get(req_id)
if req_layers is None: continue
if layer_idx not in req_layers: continue
```

### 2. Remove Debug Logging Overhead
- Removed logger.debug() calls from hot forward hook path
- Even at DEBUG level, these have overhead (string formatting, function calls)

### 3. Direct Hook Call
```python
# Before: Dict lookup every time
hook_fn = _HOOK_VARIANTS[_HOOK_VARIANT]
hook_fn(state, layer_idx, hidden, request_ids, seq_lens)

# After: Direct call
_capture_hook_full(state, layer_idx, hidden, request_ids, seq_lens)
```

### 4. Boolean Phase Instead of String
```python
# Before: String comparison
phase = "prefill" if req_seq_len > 1 else "decode"
if phase == "prefill": ...

# After: Boolean
is_prefill = req_seq_len > 1
if is_prefill: ...
```

### 5. Decode Buffer Batching (Commit 938ca89)

**The Big Win:** Batch decode token concatenations

```python
# Buffer tokens
decode_buf = state.request_decode_buffers[req_id][layer_idx]
decode_buf.append(req_hidden.detach().clone())

# Flush every 32 tokens (instead of every 1 token)
if len(decode_buf) >= 32:
    batched = torch.cat(decode_buf, dim=0)
    decode_buf.clear()
    # Single concatenation for 32 tokens
    req_captures[layer_idx] = torch.cat([existing, batched], dim=0)
```

**Impact:**
- Reduces cat operations by **32x**: 147,456 → 4,608
- Batches memory allocations
- Reduces fragmentation
- Should eliminate iteration-to-iteration degradation

## Expected Results

With all optimizations:
1. **Zero-layer:** Should remain ~1-2% overhead
2. **All-layer:** Should be **consistent across iterations** (no degradation)
3. **All-layer mean:** Target <60s generation (vs 85s unoptimized)
4. **Per-layer cost:** Should be roughly linear (~0.5-1s per layer)

## Next Steps

1. ✅ Complete baseline isolated benchmark
2. ⏳ Wait for benchmark completion
3. 🔄 Run new isolated benchmark with optimizations
4. 📊 Compare before/after
5. 📝 Update JOURNAL.md with results
6. 🧪 Run comprehensive tests
7. 🚀 Merge if successful

## Files Changed

### chatspace/vllm_steering/runtime.py
- Optimized `_capture_hook_full()` (dict lookups, boolean phase)
- Optimized `_patched_forward()` (removed debug logging, direct call)
- Added `request_decode_buffers` to `_SteeringState`
- Added `_flush_decode_buffers()` function
- Updated `register_capture_request()`, `unregister_capture_request()`
- Updated `fetch_batch_captures()` to flush decode buffers

### New Benchmark Infrastructure
- `scripts/isolated_capture_bench.py` - Single-config benchmark
- `scripts/run_isolated_benchmarks.sh` - Orchestrator
- `scripts/ISOLATED_BENCH_README.md` - Documentation
