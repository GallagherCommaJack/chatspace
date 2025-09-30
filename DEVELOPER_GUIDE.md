# Developer Guide: Multiprocessing Pipeline

Quick reference for working with the multiprocessing embedding pipeline.

## Quick Start

### Running the Pipeline

```bash
# Standard usage
uv run chatspace embed-hf \
  --dataset HuggingFaceTB/smoltalk \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --max-rows 1000

# With compilation (faster after warmup)
uv run chatspace embed-hf \
  --dataset HuggingFaceTB/smoltalk \
  --compile-model \
  --max-rows 1000

# Debug mode (more logging)
LOGLEVEL=DEBUG uv run chatspace embed-hf --dataset ... --max-rows 100
```

### Testing Changes

```bash
# Quick syntax check
python -m py_compile chatspace/hf_embed/*.py

# Small integration test
python test_multiprocessing.py

# Full test with real model
uv run chatspace embed-hf \
  --dataset HuggingFaceTB/smoltalk \
  --max-rows 50 \
  --model sentence-transformers/all-MiniLM-L6-v2
```

## Architecture Overview

### Process Roles

| Process | Purpose | CPU/GPU Usage | Memory Usage |
|---------|---------|---------------|--------------|
| **Controller** | Orchestration, progress, signals | Minimal | Low (~100 MB) |
| **Loader** | Dataset streaming | Low (I/O bound) | Low-Medium |
| **Encoder** | Tokenization + inference | **High** | High (model weights) |
| **Writer** | Parquet serialization | Medium (I/O + compression) | Medium (buffer) |

### Queue Communication

```python
# row_queue: dict → dict
{"id": ..., "text": "...", ...}

# batch_queue: EmbeddedBatch → EmbeddedBatch
EmbeddedBatch(rows=[...], embeddings=torch.Tensor(...))

# progress_queue: ProgressUpdate → None
ProgressUpdate(rows_processed=32, stage="encoder")

# error_queue: (str, Exception) → None
("encoder", RuntimeError("CUDA OOM"))

# shard_metadata_queue: list[dict] → None
[{"path": "...", "rows": 8192, "sha256": "...", ...}, ...]
```

## Common Tasks

### Adding a New Pipeline Stage

1. **Define worker function**:
```python
def _new_stage_worker(
    cfg: SentenceTransformerConfig,
    input_queue: mp.Queue[Any],
    output_queue: mp.Queue[Any],
    error_queue: mp.Queue[tuple[str, Exception]],
    shutdown_event: mp.Event,
    metrics: PipelineMetrics,
) -> None:
    """Worker function for new stage."""
    try:
        while not shutdown_event.is_set():
            try:
                item = input_queue.get(timeout=0.1)
            except Exception:
                continue

            if item is _STOP:
                break

            # Process item
            result = process(item)
            output_queue.put(result)

    except Exception as exc:
        error_queue.put(("new_stage", exc))
    finally:
        output_queue.put(_STOP)
```

2. **Create process in pipeline**:
```python
new_stage_process = ctx.Process(
    target=_new_stage_worker,
    args=(cfg, input_queue, output_queue, error_queue, shutdown_event, metrics),
    name="new-stage",
)
controller.register_process("new_stage", new_stage_process)
```

3. **Update data flow**:
```python
loader → row_queue → encoder → intermediate_queue → new_stage → batch_queue → writer
```

### Modifying the Encoder

The encoder is the most complex stage. Key points:

```python
def _encoder_worker(...):
    # 1. Load model INSIDE process (critical for GIL freedom)
    model = _load_model(cfg)
    model_runner = _ModelRunner(model, cfg.compile_model, cfg.compile_mode)

    # 2. Warmup compiled models
    warmup_timings = model_runner.warmup(bucket_sizes)

    # 3. Main loop with shutdown checking
    while not shutdown_event.is_set():
        try:
            item = row_queue.get(timeout=0.1)  # Always use timeout
        except Exception:
            continue  # Check shutdown_event again

        # 4. Process batches, send progress
        result = encode_batch(item)
        progress_queue.put_nowait(ProgressUpdate(rows_processed=len(result)))
```

### Adding Metrics

1. **Define metric in StageTimings or PipelineMetrics**:
```python
@dataclass
class PipelineMetrics:
    # ... existing fields ...
    new_metric_seconds: float = 0.0
    _lock: mp.Lock = field(default_factory=mp.Lock, init=False, repr=False)

    def add_new_metric(self, delta: float) -> None:
        if delta <= 0:
            return
        with self._lock:
            self.new_metric_seconds += delta
```

2. **Track metric in worker**:
```python
start = time.perf_counter()
do_work()
metrics.add_new_metric(time.perf_counter() - start)
```

3. **Export in to_dict()**:
```python
def to_dict(self, total_duration: float) -> dict[str, Any]:
    return {
        # ... existing metrics ...
        "new_metric_seconds": self.new_metric_seconds,
    }
```

### Debugging Tips

#### Enable Debug Logging

```bash
LOGLEVEL=DEBUG uv run chatspace embed-hf --dataset ...
```

#### Check Process Status

```python
# In main process, before joining
for managed in controller._processes:
    print(f"{managed.name}: alive={managed.process.is_alive()}, exitcode={managed.process.exitcode}")
```

#### Inspect Queue Sizes

```python
# Add to main_work() function
print(f"row_queue: {row_queue.qsize()}")
print(f"batch_queue: {batch_queue.qsize()}")
print(f"progress_queue: {progress_queue.qsize()}")
```

#### Trace Deadlocks

If pipeline hangs:

1. **Check if processes are alive**:
```bash
ps aux | grep python
```

2. **Check queue states** (add logging):
```python
logging.info("Waiting on row_queue.get()...")
item = row_queue.get(timeout=5.0)
logging.info("Received item from row_queue")
```

3. **Check for missing STOP sentinels**:
- Every worker that reads from a queue must send `_STOP` to downstream queue
- Check that `_STOP` is sent even on exceptions (use `finally:`)

#### Memory Leaks

Track memory usage:
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")
```

## Common Pitfalls

### ❌ Passing Non-Serializable Objects

**Wrong**:
```python
# Model is not serializable!
process = ctx.Process(
    target=encoder_worker,
    args=(model, ...),  # ❌ Will fail
)
```

**Right**:
```python
# Pass config, load model inside process
process = ctx.Process(
    target=encoder_worker,
    args=(cfg, ...),  # ✅ Config is serializable
)

def encoder_worker(cfg, ...):
    model = _load_model(cfg)  # Load inside process
```

### ❌ Forgetting Timeout on Queue.get()

**Wrong**:
```python
while not shutdown_event.is_set():
    item = queue.get()  # ❌ Blocks forever, can't check shutdown_event
```

**Right**:
```python
while not shutdown_event.is_set():
    try:
        item = queue.get(timeout=0.1)  # ✅ Checks shutdown every 0.1s
    except Exception:
        continue
```

### ❌ Missing STOP Sentinel

**Wrong**:
```python
def worker(input_queue, output_queue):
    while True:
        item = input_queue.get()
        if item is _STOP:
            break
        output_queue.put(process(item))
    # ❌ Downstream worker will hang waiting for STOP
```

**Right**:
```python
def worker(input_queue, output_queue):
    try:
        while True:
            item = input_queue.get()
            if item is _STOP:
                break
            output_queue.put(process(item))
    finally:
        output_queue.put(_STOP)  # ✅ Always send STOP
```

### ❌ Unbounded Queue Puts

**Wrong**:
```python
for i in range(1000000):
    queue.put(item)  # ❌ If consumer is slow, fills memory
```

**Right**:
```python
# Use bounded queue (already done in pipeline)
queue = ctx.Queue(maxsize=100)  # ✅ Blocks when full

# Or use put with timeout
try:
    queue.put(item, timeout=1.0)
except Full:
    if shutdown_event.is_set():
        break
```

### ❌ Sharing Metrics Without Locks

**Wrong**:
```python
@dataclass
class Metrics:
    counter: int = 0

    def increment(self):
        self.counter += 1  # ❌ Race condition across processes
```

**Right**:
```python
@dataclass
class Metrics:
    counter: int = 0
    _lock: mp.Lock = field(default_factory=mp.Lock)

    def increment(self):
        with self._lock:  # ✅ Process-safe
            self.counter += 1
```

## Performance Tuning

### Batch Size Tuning

```bash
# Small batches (faster feedback, lower GPU util)
uv run chatspace embed-hf --batch-size 16 --dataset ...

# Large batches (better GPU util, more memory)
uv run chatspace embed-hf --batch-size 128 --dataset ...

# Token-based batching (adaptive)
uv run chatspace embed-hf --tokens-per-batch 131072 --dataset ...
```

### Queue Size Tuning

In [pipeline.py](chatspace/hf_embed/pipeline.py):

```python
# Increase prefetch for I/O bound workloads
row_queue = ctx.Queue(maxsize=max(1, max_bucket_batch * cfg.prefetch_batches * 2))

# Decrease for memory-constrained systems
batch_queue = ctx.Queue(maxsize=max(1, cfg.prefetch_batches // 2))
```

### Compilation

```bash
# Enable compilation (one-time warmup cost, then faster)
uv run chatspace embed-hf --compile-model --dataset ...

# Try different modes
uv run chatspace embed-hf --compile-model --compile-mode reduce-overhead --dataset ...
uv run chatspace embed-hf --compile-model --compile-mode max-autotune --dataset ...
```

### Multi-GPU (Future)

To scale to multiple GPUs:

1. Create multiple encoder processes
2. Set `CUDA_VISIBLE_DEVICES` for each
3. Round-robin row distribution

```python
# Pseudocode for multi-GPU
for gpu_id in range(num_gpus):
    encoder_process = ctx.Process(
        target=_encoder_worker,
        args=(cfg, row_queue, batch_queues[gpu_id], ...),
        name=f"encoder-gpu{gpu_id}",
    )
    # Set environment variable for CUDA device
    encoder_process.env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    encoder_process.start()
```

## Troubleshooting

### Pipeline Hangs

**Symptoms**: Process starts but no progress

**Diagnosis**:
1. Check if all processes are alive: `ps aux | grep python`
2. Add debug logging to see where each process is stuck
3. Check queue sizes (might be full or empty)

**Common causes**:
- Missing `_STOP` sentinel
- Deadlock on queue.get() without timeout
- Exception in worker not propagated

**Solution**:
- Add timeouts to all queue.get() calls
- Ensure `_STOP` sent in `finally:` blocks
- Check error_queue for exceptions

### "Killed" / OOM

**Symptoms**: Process terminates with "Killed" message

**Diagnosis**:
```bash
dmesg | tail  # Check for OOM killer
```

**Common causes**:
- Model too large for available memory
- Batch size too large
- Queue unbounded growth
- Memory leak in long runs

**Solution**:
- Reduce batch size: `--batch-size 16`
- Reduce prefetch: `--prefetch-batches 2`
- Use smaller model
- Use CPU inference: `--device cpu`

### Slow Performance

**Symptoms**: Much slower than expected

**Diagnosis**:
1. Check GPU utilization: `nvidia-smi`
2. Check CPU usage: `top` or `htop`
3. Enable metrics logging

**Common causes**:
- Queue too small (workers idle waiting)
- Batch size too small (GPU underutilized)
- I/O bottleneck (slow disk)
- GIL still active (check torch version)

**Solution**:
- Increase prefetch: `--prefetch-batches 8`
- Increase batch size: `--batch-size 64` or `--tokens-per-batch 131072`
- Use faster storage (SSD, tmpfs)
- Enable compilation: `--compile-model`

### Ctrl-C Doesn't Work

**Symptoms**: Pipeline ignores Ctrl-C

**Diagnosis**:
- Check if signal handler is installed correctly
- Look for blocking operations without timeout

**Solution**:
- Always use timeout on blocking calls
- Check shutdown_event regularly
- Use `try/except KeyboardInterrupt` in main thread

## References

- [MULTIPROCESSING_MIGRATION.md](MULTIPROCESSING_MIGRATION.md) - Full migration details
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture diagrams and flow
- [chatspace/hf_embed/pipeline.py](chatspace/hf_embed/pipeline.py) - Main pipeline code
- [chatspace/hf_embed/orchestrator.py](chatspace/hf_embed/orchestrator.py) - Process controller