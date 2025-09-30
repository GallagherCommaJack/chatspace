# Multiprocessing Migration Summary

This document summarizes the migration from multithreading to multiprocessing for the chatspace embedding pipeline.

## Motivation

The original implementation used Python's `threading` module, which is subject to the Global Interpreter Lock (GIL). This meant that:
- Model inference couldn't truly run in parallel with data loading/writing
- CPU-bound operations (tokenization, tensor manipulation) were serialized
- GPU utilization was suboptimal due to GIL contention

The multiprocessing migration enables:
- **True parallelism** for CPU/GPU-bound model inference
- **GIL-free independence** for each pipeline stage
- **Better resource utilization** with proper process isolation
- **Improved signal handling** with a central controller

## Architecture Changes

### Before (Threading)
```
Main Thread:
  ├─ Loader Thread → row_queue
  ├─ Encoder (main thread) → batch_queue
  └─ Writer Thread → Parquet files
```

### After (Multiprocessing)
```
Controller Process (main):
  ├─ Loader Process → row_queue
  ├─ Encoder Process → batch_queue + progress_queue
  └─ Writer Process → Parquet files + shard_metadata_queue

Controller monitors:
  - Process health
  - Progress updates
  - Signal handling (SIGINT/SIGTERM)
  - Error propagation
```

## Key Changes

### 1. New Module: `orchestrator.py`

**Purpose**: Central process lifecycle management and signal handling

**Key Components**:
- `ProcessController`: Manages process registration, startup, monitoring, and shutdown
- `ManagedProcess`: Tracks process metadata and status
- `ProgressUpdate`: Message format for progress reporting
- Signal handlers for graceful shutdown on Ctrl-C

**Features**:
- Automatic process cleanup on exit
- Timeout-based forceful termination for hung processes
- Centralized progress bar updates
- Exception propagation from worker processes

### 2. Updated: `metrics.py`

**Changes**:
- `threading.Lock` → `multiprocessing.Lock`
- All locks are now process-safe
- Metrics objects can be shared across processes

**Affected Classes**:
- `StageTimings`: Now uses `mp.Lock` for thread-safety
- `PipelineMetrics`: Uses `mp.Lock` for encoder encode time tracking

### 3. Refactored: `pipeline.py`

**Major Changes**:

#### Worker Functions

1. **`_dataset_loader_worker`**:
   - Now takes `SentenceTransformerConfig` instead of pre-loaded dataset
   - Loads dataset inside the process
   - Uses `mp.Queue` and `mp.Event` for coordination
   - Reports errors via `error_queue`

2. **`_encoder_worker`** (NEW):
   - Completely new function that replaces in-thread encoder loop
   - **Loads model inside the encoder process** (critical for GIL freedom)
   - Performs warmup and compilation in isolation
   - Handles shutdown events gracefully
   - Sends progress updates to controller via `progress_queue`

3. **`_encode_and_dispatch_batch`**:
   - Updated to use `mp.Queue` instead of `queue.Queue`
   - Sends progress updates via optional `progress_queue`
   - No longer updates tqdm directly (controller handles that)

#### Main Pipeline Function

**`run_sentence_transformer`**:
- Uses `mp.get_context("spawn")` for clean process initialization
- Creates multiple queues: `row_queue`, `batch_queue`, `progress_queue`, `error_queue`, `shard_metadata_queue`
- Uses `ProcessController` for lifecycle management
- Removes model loading from main process (encoder loads it)
- Retrieves shard metadata via queue after writer completes
- Improved error handling with error queue

### 4. Updated: `writer.py`

**Changes**:
- `ThreadState` → `ProcessState`
- `queue.Queue` → `mp.Queue`
- `_ShardWriter.run()` now accepts optional `shard_metadata_queue`
- Sends shard list back to main process after completion

## Inter-Process Communication

### Queues

1. **`row_queue`**: Loader → Encoder
   - Transfers dataset rows
   - Bounded by `prefetch_batches * batch_size`

2. **`batch_queue`**: Encoder → Writer
   - Transfers `EmbeddedBatch` objects (rows + embeddings)
   - Bounded by `prefetch_batches`

3. **`progress_queue`**: Encoder → Controller
   - Sends `ProgressUpdate` messages
   - Non-blocking, best-effort delivery
   - Max size: 100 messages

4. **`error_queue`**: All Workers → Controller
   - Sends `(stage_name, exception)` tuples
   - Enables exception propagation across processes

5. **`shard_metadata_queue`**: Writer → Main
   - Sends list of shard metadata after completion
   - Size 1 (only one message)

### Shared State

- **`shutdown_event`**: `mp.Event` for coordinated shutdown
- **`metrics`**: `PipelineMetrics` with process-safe locks
- **`stats`**: `PipelineStats` (currently not fully shared, may need Manager)

## Signal Handling

The `ProcessController` provides robust signal handling:

1. **First SIGINT/SIGTERM**:
   - Sets `shutdown_event`
   - Workers check event and exit gracefully
   - Flushes remaining data

2. **Second SIGINT** (force):
   - Immediately terminates all processes
   - Raises `KeyboardInterrupt`

3. **Timeout-based cleanup**:
   - Waits up to 30 seconds for graceful exit
   - Terminates hung processes automatically

## Migration Checklist

- [x] Create `orchestrator.py` with `ProcessController`
- [x] Update `metrics.py` to use `mp.Lock`
- [x] Refactor `_dataset_loader_worker` for process context
- [x] Create new `_encoder_worker` that loads model internally
- [x] Update `_encode_and_dispatch_batch` for progress reporting
- [x] Refactor `run_sentence_transformer` to use multiprocessing
- [x] Update `writer.py` to use `ProcessState` and return shard metadata
- [x] Add shard metadata queue for cross-process communication
- [x] Verify syntax and basic compilation

## Testing

### Test Script

A test script has been created at `/root/chatspace/test_multiprocessing.py`:

```bash
python test_multiprocessing.py
```

This tests:
- Small dataset embedding (20 rows)
- Process startup and coordination
- Progress tracking
- Graceful shutdown
- Manifest generation

### Manual Testing

To test Ctrl-C handling:

```bash
# Start a long-running job
uv run chatspace embed-hf --dataset HuggingFaceTB/smoltalk --max-rows 1000

# Press Ctrl-C once (graceful shutdown)
# Or press Ctrl-C twice (force shutdown)
```

## Known Limitations & Future Work

1. **Metrics/Stats Sharing**: Currently, `PipelineStats` may not properly aggregate across processes. Consider using `multiprocessing.Manager` for full cross-process state.

2. **Single Encoder Process**: The current design uses one encoder process. For multi-GPU setups, we could spawn multiple encoder processes with device affinity.

3. **Queue Serialization Overhead**: PyTorch tensors are serialized when passing through queues. For very large batches, consider shared memory tensors.

4. **Progress Tracking Granularity**: Progress updates are sent per batch. For very large batches, this may feel less responsive.

5. **Warmup in Encoder Process**: Each encoder process performs its own warmup. If using multiple encoders, this could be optimized.

## Compatibility Notes

- **Python 3.8+** required (for multiprocessing improvements)
- **spawn context** used for CUDA compatibility (works on Linux, macOS, Windows)
- **PyTorch tensors** are automatically serialized (slower than shared memory but more compatible)
- **Compiled models** (`torch.compile`) work because model is loaded inside encoder process

## Performance Expectations

Expected improvements:
- **10-30% faster** for CPU-bound tokenization
- **15-40% faster** for GPU inference (depending on GIL contention in threading version)
- **Better GPU utilization** due to GIL-free inference
- **More responsive** Ctrl-C handling

Trade-offs:
- **Slightly higher memory** usage (separate process memory spaces)
- **Process startup overhead** (~1-2 seconds for model loading in encoder)
- **Queue serialization** overhead for tensor data

## Rollback Plan

If issues arise, the threading version can be restored from git history:

```bash
git log --oneline --all -- chatspace/hf_embed/pipeline.py
git checkout <commit_hash> -- chatspace/hf_embed/
```

The threading implementation is preserved in commit history before this migration.