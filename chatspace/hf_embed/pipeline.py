"""Main pipeline orchestration for embedding datasets."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Any, Optional

from .bucketing import (
    TokenBatch,
    _BucketBuffer,
    _effective_batch_size,
    _select_bucket_size,
    _token_sequence_length,
)
from .config import SentenceTransformerConfig
from .dataset import _load_dataset, _rows_from_dataset
from .metrics import PipelineMetrics, PipelineStats
from .model import _ModelRunner, _load_model
from .orchestrator import ProcessController, ProgressUpdate
from .utils import (
    _compute_norms,
    _enumerate_bucket_sizes,
    _iso_now,
    _observe_git_sha,
    _prepare_paths,
)
from .writer import EmbeddedBatch, ProcessState, _ShardWriter, write_manifest

# Sentinel for signaling process shutdown
_STOP = object()


def _dataset_loader_worker(
    cfg: SentenceTransformerConfig,
    row_queue: mp.Queue[Any],
    error_queue: mp.Queue[tuple[str, Exception]],
    shutdown_event: mp.Event,
    metrics: PipelineMetrics,
) -> None:
    """Background process for loading dataset rows."""
    try:
        ds = _load_dataset(cfg)
        last_timestamp = time.perf_counter()
        for row in _rows_from_dataset(ds, cfg):
            if shutdown_event.is_set():
                break
            before_put = time.perf_counter()
            metrics.loader.add_busy(before_put - last_timestamp)
            row_queue.put(row)
            after_put = time.perf_counter()
            metrics.loader.add_idle(after_put - before_put)
            last_timestamp = after_put
    except Exception as exc:
        logging.exception("Loader process error")
        error_queue.put(("loader", exc))
    finally:
        row_queue.put(_STOP)


def _encode_and_dispatch_batch(
    token_batch: TokenBatch,
    *,
    model_runner: _ModelRunner,
    cfg: SentenceTransformerConfig,
    created_at: str,
    run_id: str,
    stats: PipelineStats,
    metrics: PipelineMetrics,
    batch_queue: mp.Queue[Any],
    progress_queue: Optional[mp.Queue[Any]],
) -> bool:
    """Encode a batch and dispatch to writer. Returns True if max_rows limit reached."""
    rows = [dict(row) for row in token_batch.rows]
    if not rows:
        return False

    features = {key: value for key, value in token_batch.features.items()}

    remaining: Optional[int] = None
    reached_limit = False
    if cfg.max_rows is not None:
        remaining = max(cfg.max_rows - stats.total_rows, 0)
        if remaining <= 0:
            return True
        if len(rows) > remaining:
            rows = rows[:remaining]
            features = {key: value[:remaining] for key, value in features.items()}
            reached_limit = True

    num_rows = len(rows)
    if num_rows == 0:
        return True if cfg.max_rows is not None and remaining == 0 else reached_limit

    encode_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model_runner.forward(features)
    metrics.add_encoder_encode(time.perf_counter() - encode_start)

    embeddings = outputs.get("sentence_embedding")
    if embeddings is None:
        raise ValueError("Model forward pass did not return 'sentence_embedding'.")
    embeddings = embeddings.detach()[:num_rows]

    stats.update_embedding_dim(embeddings.shape[1])

    norms_tensor = _compute_norms(embeddings)
    batch_min_norm = float(norms_tensor.min().item()) if norms_tensor.numel() > 0 else None
    batch_max_norm = float(norms_tensor.max().item()) if norms_tensor.numel() > 0 else None

    stats.register_rows(num_rows)
    stats.update_norm_bounds(batch_min_norm, batch_max_norm)

    # Send progress update to controller
    if progress_queue is not None:
        try:
            progress_queue.put_nowait(ProgressUpdate(rows_processed=num_rows))
        except Exception:
            pass  # Non-critical if queue is full

    for row in rows:
        row["model"] = cfg.model_name
        row["created_at"] = created_at
        row["run_id"] = run_id
        # Normalize text field: ensure embedded text is always in "text" column
        if cfg.text_field != "text" and cfg.text_field in row:
            row["text"] = row[cfg.text_field]

    batch_queue.put(EmbeddedBatch(rows=rows, embeddings=embeddings))
    return reached_limit


def _encoder_worker(
    cfg: SentenceTransformerConfig,
    created_at: str,
    run_id: str,
    row_queue: mp.Queue[Any],
    batch_queue: mp.Queue[Any],
    progress_queue: mp.Queue[Any],
    error_queue: mp.Queue[tuple[str, Exception]],
    shutdown_event: mp.Event,
    stats: PipelineStats,
    metrics: PipelineMetrics,
) -> None:
    """Encoder process: load model, tokenize, bucket, batch, encode, dispatch."""
    try:
        # Import torch inside worker to avoid slow CLI startup
        import torch

        # Enable tf32 tensor cores
        torch.set_float32_matmul_precision('high')

        # Load model inside the encoder process
        logging.info("Encoder process: loading model %s", cfg.model_name)
        model = _load_model(cfg)
        try:
            model.max_seq_length = cfg.bucket_max_tokens
        except Exception:
            logging.debug("Unable to set model.max_seq_length explicitly; using model default.")

        model_runner = _ModelRunner(model, cfg.compile_model, cfg.compile_mode)

        # Warmup if compilation is enabled
        bucket_sizes = _enumerate_bucket_sizes(cfg.bucket_min_tokens, cfg.bucket_max_tokens)
        warmup_timings = model_runner.warmup(bucket_sizes)
        if warmup_timings:
            readable = {size: round(duration, 4) for size, duration in warmup_timings.items()}
            logging.info("Warmup compile timings (seconds) per bucket: %s", readable)
            logging.info("Warmup compile total: %.4fs", sum(warmup_timings.values()))

        buckets: dict[int, _BucketBuffer] = {}
        reached_limit = False

        while not shutdown_event.is_set():
            wait_start = time.perf_counter()
            try:
                item = row_queue.get(timeout=0.1)
            except Exception:
                continue
            wait_end = time.perf_counter()
            metrics.encoder.add_idle(wait_end - wait_start)

            if item is _STOP:
                break

            busy_start = time.perf_counter()

            if reached_limit:
                metrics.encoder.add_busy(time.perf_counter() - busy_start)
                continue

            text_value = str(item.get(cfg.text_field, ""))
            if not text_value:
                stats.register_skipped()
                metrics.encoder.add_busy(time.perf_counter() - busy_start)
                continue

            try:
                tokenized = model_runner.tokenize(text_value, max_length=cfg.bucket_max_tokens)
            except Exception as exc:
                logging.warning("Failed to tokenize row: %s", exc)
                stats.register_skipped()
                metrics.encoder.add_busy(time.perf_counter() - busy_start)
                continue

            if "attention_mask" not in tokenized and "input_ids" in tokenized:
                tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"], dtype=torch.long)

            seq_length = _token_sequence_length(tokenized)
            if seq_length == 0:
                stats.register_skipped()
                metrics.encoder.add_busy(time.perf_counter() - busy_start)
                continue

            bucket_size = _select_bucket_size(seq_length, cfg)
            bucket = buckets.setdefault(bucket_size, _BucketBuffer(bucket_size))
            bucket.add(dict(item), tokenized)

            batch_target = _effective_batch_size(bucket_size, cfg)
            while len(bucket) >= batch_target and not reached_limit:
                token_batch = bucket.pop(batch_target, model_runner.pad_values)
                if token_batch is None:
                    break
                reached_limit = _encode_and_dispatch_batch(
                    token_batch,
                    model_runner=model_runner,
                    cfg=cfg,
                    created_at=created_at,
                    run_id=run_id,
                    stats=stats,
                    metrics=metrics,
                    batch_queue=batch_queue,
                    progress_queue=progress_queue,
                ) and cfg.max_rows is not None

            metrics.encoder.add_busy(time.perf_counter() - busy_start)

        # Flush remaining buckets
        if not reached_limit and not shutdown_event.is_set():
            flush_start = time.perf_counter()
            for bucket_size in sorted(buckets):
                bucket = buckets[bucket_size]
                batch_target = _effective_batch_size(bucket_size, cfg)
                while len(bucket) > 0 and not reached_limit:
                    count = min(batch_target, len(bucket))
                    token_batch = bucket.pop(count, model_runner.pad_values)
                    if token_batch is None:
                        break
                    reached_limit = _encode_and_dispatch_batch(
                        token_batch,
                        model_runner=model_runner,
                        cfg=cfg,
                        created_at=created_at,
                        run_id=run_id,
                        stats=stats,
                        metrics=metrics,
                        batch_queue=batch_queue,
                        progress_queue=progress_queue,
                    ) and cfg.max_rows is not None
                if reached_limit:
                    break
            metrics.encoder.add_busy(time.perf_counter() - flush_start)

    except Exception as exc:
        logging.exception("Encoder process error")
        error_queue.put(("encoder", exc))
    finally:
        batch_queue.put(_STOP)
        progress_queue.put(_STOP)


def run_sentence_transformer(cfg: SentenceTransformerConfig) -> dict[str, Any]:
    """Main entry point for embedding pipeline with multiprocessing.

    Args:
        cfg: Configuration for the embedding run

    Returns:
        Dictionary with manifest, run summary, and paths
    """
    start_time = time.time()
    paths = _prepare_paths(
        cfg.output_root,
        cfg.model_name,
        cfg.dataset,
        cfg.subset,
        cfg.split,
        cfg.manifest_relpath,
    )

    git_sha = _observe_git_sha()
    created_at = _iso_now()
    run_id = cfg.run_id or created_at.replace(":", "").replace("-", "")

    # Use spawn method for multiprocessing (works well with CUDA)
    ctx = mp.get_context("spawn")

    # Shared state and queues
    max_bucket_batch = _effective_batch_size(cfg.bucket_min_tokens, cfg)
    row_queue: mp.Queue[Any] = ctx.Queue(maxsize=max(1, max_bucket_batch * cfg.prefetch_batches))
    batch_queue: mp.Queue[Any] = ctx.Queue(maxsize=max(1, cfg.prefetch_batches))
    progress_queue: mp.Queue[Any] = ctx.Queue(maxsize=100)
    error_queue: mp.Queue[tuple[str, Exception]] = ctx.Queue()
    shard_metadata_queue: mp.Queue[Any] = ctx.Queue(maxsize=1)
    shutdown_event = ctx.Event()

    # Shared metrics and stats with process-safe locks
    metrics = PipelineMetrics()
    stats = PipelineStats()

    # Create process controller
    controller = ProcessController()
    controller.set_shutdown_event(shutdown_event)

    # Create loader process
    loader_process = ctx.Process(
        target=_dataset_loader_worker,
        args=(cfg, row_queue, error_queue, shutdown_event, metrics),
        name="dataset-loader",
    )
    controller.register_process("loader", loader_process)

    # Create encoder process
    encoder_process = ctx.Process(
        target=_encoder_worker,
        args=(
            cfg,
            created_at,
            run_id,
            row_queue,
            batch_queue,
            progress_queue,
            error_queue,
            shutdown_event,
            stats,
            metrics,
        ),
        name="encoder",
    )
    controller.register_process("encoder", encoder_process)

    # Create writer process
    shard_writer = _ShardWriter(cfg, paths, git_sha)
    writer_state = ProcessState()
    writer_process = ctx.Process(
        target=shard_writer.run,
        args=(batch_queue, _STOP, writer_state, metrics, shard_metadata_queue),
        name="shard-writer",
    )
    controller.register_process("writer", writer_process)

    # Create progress bar in controller
    progress = controller.create_progress_bar(disable=not cfg.progress, unit="rows")

    def main_work() -> None:
        """Main work: monitor progress and check for errors."""
        # Monitor progress updates
        while not shutdown_event.is_set():
            try:
                msg = progress_queue.get(timeout=0.5)
                if msg is _STOP:
                    break
                if isinstance(msg, ProgressUpdate):
                    controller.update_progress(msg.rows_processed)
            except Exception:
                # Check if all processes are done
                if not any(p.is_alive() for p in [loader_process, encoder_process, writer_process]):
                    break
                # Check for errors
                try:
                    stage, exc = error_queue.get_nowait()
                    raise RuntimeError(f"Process {stage} failed: {exc}") from exc
                except Exception:
                    pass
                continue

    try:
        controller.run_with_monitoring(main_work)
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        raise
    finally:
        # Check for any errors in queue
        try:
            while not error_queue.empty():
                stage, exc = error_queue.get_nowait()
                logging.error("Process %s reported error: %s", stage, exc)
        except Exception:
            pass

    # Retrieve shard metadata from writer process
    try:
        shards = shard_metadata_queue.get(timeout=5.0)
    except Exception:
        logging.warning("Failed to retrieve shard metadata from writer, using empty list")
        shards = []

    duration = time.time() - start_time
    logging.info("Pipeline stage timings: %s", metrics.to_dict(duration))

    write_manifest(
        cfg=cfg,
        paths=paths,
        shards=shards,
        stats=stats,
        metrics=metrics,
        created_at=created_at,
        run_id=run_id,
        duration=duration,
        git_sha=git_sha,
    )

    logging.info("Wrote manifest: %s", paths["manifest_path"])
    logging.info("Wrote run summary: %s", paths["run_path"])

    manifest_path = paths["manifest_path"]
    with manifest_path.open("r", encoding="utf-8") as fh:
        import json
        manifest = json.load(fh)

    run_path = paths["run_path"]
    with run_path.open("r", encoding="utf-8") as fh:
        import json
        run_summary = json.load(fh)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_summary": run_summary,
        "run_path": run_path,
    }