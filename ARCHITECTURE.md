# chatspace Multiprocessing Architecture

## Process Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CONTROLLER PROCESS (Main)                    │
│                                                                       │
│  ┌──────────────────┐      ┌──────────────────┐                    │
│  │ Process Manager  │      │  Progress Bar    │                    │
│  │ - Lifecycle      │      │  (tqdm)          │                    │
│  │ - Signal Handler │      │                  │                    │
│  │ - Error Monitor  │      └──────────────────┘                    │
│  └──────────────────┘                                               │
│          │                                                           │
│          │ monitors / controls                                      │
│          ▼                                                           │
└──────────┼───────────────────────────────────────────────────────────┘
           │
           │
    ┌──────┴───────┬──────────────┬───────────────┐
    │              │              │               │
    ▼              ▼              ▼               ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐
│ Loader  │  │ Encoder  │  │ Writer   │  │ shutdown_  │
│ Process │  │ Process  │  │ Process  │  │ event      │
└─────────┘  └──────────┘  └──────────┘  └────────────┘
     │              │             │              │
     │              │             │              │
     │              └─────────────┴──────────────┘
     │                            │               │
     │                            │               │
     │                    monitors for shutdown   │
     │                                            │
     └────────────────────────────────────────────┘
```

## Data Flow

```
                    row_queue              batch_queue
Dataset  ──────►  (bounded)  ──────►     (bounded)     ──────►  Parquet
  ^                  │                       │                    Shards
  │                  │                       │                      │
  │                  ▼                       ▼                      │
  │            ┌──────────┐           ┌──────────┐                 │
  │            │  Loader  │           │ Encoder  │                 │
  │            │ Process  │           │ Process  │                 │
  │            └──────────┘           └──────────┘                 │
  │                                        │                        │
  │                                        │                        │
  │                                  Model Inference                │
  │                                   (GIL-free)                    │
  │                                        │                        │
  │                                        ▼                        │
  │                                  progress_queue                 │
  │                                        │                        │
  │                                        ▼                        │
  │                                  ┌──────────┐                  │
  │                                  │ Writer   │                  │
  │                                  │ Process  │                  │
  │                                  └──────────┘                  │
  │                                        │                        │
  │                                        ▼                        │
  │                              shard_metadata_queue              │
  │                                        │                        │
  └────────────────────────────────────────┴────────────────────────┘
                                           │
                                           ▼
                                     Controller
                                   (progress updates)
```

## Queue Details

### row_queue
- **Direction**: Loader → Encoder
- **Content**: Raw dataset rows (dict)
- **Bounded**: Yes (`max_batch_size * prefetch_batches`)
- **Backpressure**: Loader blocks when queue full

### batch_queue
- **Direction**: Encoder → Writer
- **Content**: `EmbeddedBatch` objects (rows + embeddings tensor)
- **Bounded**: Yes (`prefetch_batches`)
- **Backpressure**: Encoder blocks when queue full

### progress_queue
- **Direction**: Encoder → Controller
- **Content**: `ProgressUpdate` messages (row counts)
- **Bounded**: Yes (100 messages)
- **Backpressure**: None (non-blocking `put_nowait`, drops on full)

### error_queue
- **Direction**: All workers → Controller
- **Content**: `(stage_name: str, exception: Exception)` tuples
- **Bounded**: No (unbounded)
- **Backpressure**: None

### shard_metadata_queue
- **Direction**: Writer → Controller
- **Content**: `list[dict]` of shard metadata
- **Bounded**: Yes (1 message)
- **Backpressure**: Writer blocks until controller reads

## Process Lifecycle

### Startup Sequence

```
1. Controller creates all queues and shutdown_event
2. Controller creates Process objects
3. Controller installs signal handlers (SIGINT/SIGTERM)
4. Controller starts all processes
5. Each process:
   - Loader: Loads dataset, starts streaming
   - Encoder: Loads model, performs warmup, starts encoding loop
   - Writer: Initializes shard writer, starts write loop
6. Controller monitors progress_queue and updates progress bar
```

### Normal Shutdown

```
1. Loader exhausts dataset → sends _STOP to row_queue
2. Encoder receives _STOP → flushes buckets → sends _STOP to batch_queue + progress_queue
3. Writer receives _STOP → flushes remaining rows → sends shard metadata → exits
4. Controller detects all processes finished
5. Controller retrieves shard metadata
6. Controller joins all processes
7. Controller writes manifest
```

### Signal Shutdown (Ctrl-C)

```
1. User presses Ctrl-C (SIGINT)
2. Controller signal handler sets shutdown_event
3. All workers detect shutdown_event.is_set() → exit loops gracefully
4. Controller waits for processes (timeout: 30s)
5. If timeout exceeded → terminate processes forcefully
6. Controller checks error_queue for exceptions
7. Controller cleans up and propagates exceptions
```

### Force Shutdown (Double Ctrl-C)

```
1. User presses Ctrl-C twice
2. Controller signal handler calls _force_terminate_all()
3. All processes receive SIGTERM
4. Processes that don't respond receive SIGKILL
5. Controller raises KeyboardInterrupt
```

## Process Isolation Benefits

### GIL Independence

**Before (Threading)**:
```python
# All in same process, GIL serializes CPU work
Thread 1: tokenize() ─────────────────────────────────
Thread 2: model.forward() ───────────────────────────
Thread 3: write_parquet() ───────────────────────────
          └─ GIL forces serialization ─┘
```

**After (Multiprocessing)**:
```python
# Separate processes, true parallelism
Process 1: tokenize()         ████████████████
Process 2: model.forward()         █████████████████
Process 3: write_parquet()              ████████████
           └─ All run in parallel! ─┘
```

### Memory Isolation

- Each process has its own memory space
- Model weights loaded only in encoder process
- Dataset iterator only in loader process
- No memory contention or false sharing

### CUDA Context Isolation

- Encoder process owns GPU
- No CUDA context conflicts
- Clean GPU memory management
- Easy to extend to multi-GPU (multiple encoder processes)

## Extending to Multi-GPU

Future enhancement for multi-GPU systems:

```
                    row_queue              batch_queue_0
Dataset  ──────►  (bounded)  ─┬───────►     (bounded)     ──────►┐
                               │                                   │
                               │         batch_queue_1             │
                               ├───────►     (bounded)     ──────►┤
                               │                                   │
                               │         batch_queue_2             │  Merge
                               └───────►     (bounded)     ──────►┤  Queue
                                                                   │
                               ┌──────────┐  ┌──────────┐         │
                               │ Encoder  │  │ Encoder  │         ▼
                               │ (GPU 0)  │  │ (GPU 1)  │    ┌──────────┐
                               └──────────┘  └──────────┘    │  Writer  │
                                                              │ Process  │
                               ┌──────────┐                  └──────────┘
                               │ Encoder  │
                               │ (GPU 2)  │
                               └──────────┘
```

## Performance Characteristics

### Throughput

- **Loader**: I/O bound, minimal CPU (10-50 MB/s typical)
- **Encoder**: GPU/CPU bound, highest resource usage (throughput = model speed)
- **Writer**: I/O bound, Parquet compression (50-200 MB/s typical)

### Latency

- **Pipeline fill time**: 2-5 seconds (time to fill queues and warm up)
- **Batch latency**: Depends on batch size and model speed
- **End-to-end**: Loader → Encoder → Writer ≈ 0.5-2 seconds per batch

### Memory

- **Main process**: Minimal (~100 MB)
- **Loader process**: Dataset metadata + buffer (varies, often <500 MB)
- **Encoder process**: Model weights + batch buffers (2-8 GB typical)
- **Writer process**: Accumulation buffer (depends on `rows_per_shard`)

## Error Handling

### Error Propagation

```
Worker Process          error_queue          Controller
     │                       │                    │
     │  Exception!           │                    │
     ├──────────────────────►│                    │
     │  ("encoder", exc)     │                    │
     │  exit process         │                    │
     │                       │  Controller polls  │
     │                       ◄────────────────────┤
     │                       │  get_nowait()      │
     │                       ├───────────────────►│
     │                       │  ("encoder", exc)  │
     │                       │                    │
     │                       │         Raise RuntimeError
     │                       │         "Process encoder failed"
     │                       │                    │
```

### Deadlock Prevention

- All queues have bounded sizes (prevents infinite memory growth)
- Timeout on all blocking operations (prevents indefinite hangs)
- Shutdown event checked in all loops (enables cancellation)
- Force termination after timeout (guarantees exit)

## Testing Considerations

### Unit Tests
- Test each worker function in isolation
- Mock queues for deterministic behavior
- Test signal handling with synthetic signals

### Integration Tests
- Test full pipeline with small dataset
- Test Ctrl-C at various stages
- Test error injection in each worker
- Test queue overflow scenarios

### Load Tests
- Test with large datasets (>1M rows)
- Monitor memory usage over time
- Check for memory leaks in long runs
- Verify graceful degradation under load