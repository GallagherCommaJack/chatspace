"""Main pipeline orchestration for embedding datasets."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

import torch
from tqdm import tqdm

# Enable tf32 tensor cores
torch.set_float32_matmul_precision('high')

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
from .utils import (
    _compute_norms,
    _enumerate_bucket_sizes,
    _iso_now,
    _observe_git_sha,
    _prepare_paths,
)
from .writer import EmbeddedBatch, ThreadState, _ShardWriter, write_manifest

# Sentinel for signaling thread shutdown
_STOP = object()


def _dataset_loader_worker(
    ds: Any,
    cfg: SentenceTransformerConfig,
    row_queue: queue.Queue[Any],
    state: ThreadState,
    metrics: PipelineMetrics,
) -> None:
    """Background thread for loading dataset rows."""
    try:
        last_timestamp = time.perf_counter()
        for row in _rows_from_dataset(ds, cfg):
            before_put = time.perf_counter()
            metrics.loader.add_busy(before_put - last_timestamp)
            row_queue.put(row)
            after_put = time.perf_counter()
            metrics.loader.add_idle(after_put - before_put)
            last_timestamp = after_put
    except BaseException as exc:  # noqa: BLE001
        state.error = exc
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
    batch_queue: queue.Queue[Any],
    progress: tqdm,
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
    progress.update(num_rows)

    for row in rows:
        row["model"] = cfg.model_name
        row["created_at"] = created_at
        row["run_id"] = run_id
        # Normalize text field: ensure embedded text is always in "text" column
        if cfg.text_field != "text" and cfg.text_field in row:
            row["text"] = row[cfg.text_field]

    batch_queue.put(EmbeddedBatch(rows=rows, embeddings=embeddings))
    return reached_limit


def _embedding_pipeline(
    model_runner: _ModelRunner,
    cfg: SentenceTransformerConfig,
    created_at: str,
    run_id: str,
    *,
    row_queue: queue.Queue[Any],
    batch_queue: queue.Queue[Any],
    stats: PipelineStats,
    metrics: PipelineMetrics,
    progress: tqdm,
) -> None:
    """Main encoder loop: tokenize, bucket, batch, encode, and dispatch."""
    buckets: dict[int, _BucketBuffer] = {}
    reached_limit = False

    while True:
        wait_start = time.perf_counter()
        item = row_queue.get()
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
        except Exception as exc:  # noqa: BLE001
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
                progress=progress,
            ) and cfg.max_rows is not None

        metrics.encoder.add_busy(time.perf_counter() - busy_start)

    # Flush remaining buckets
    if not reached_limit:
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
                    progress=progress,
                ) and cfg.max_rows is not None
            if reached_limit:
                break
        metrics.encoder.add_busy(time.perf_counter() - flush_start)


def run_sentence_transformer(cfg: SentenceTransformerConfig) -> dict[str, Any]:
    """Main entry point for embedding pipeline.

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
    model = _load_model(cfg)

    try:
        model.max_seq_length = cfg.bucket_max_tokens
    except Exception:
        logging.debug("Unable to set model.max_seq_length explicitly; using model default.")

    model_runner = _ModelRunner(model, cfg.compile_model, cfg.compile_mode)

    bucket_sizes = _enumerate_bucket_sizes(cfg.bucket_min_tokens, cfg.bucket_max_tokens)
    warmup_timings = model_runner.warmup(bucket_sizes)
    if warmup_timings:
        readable = {size: round(duration, 4) for size, duration in warmup_timings.items()}
        logging.info("Warmup compile timings (seconds) per bucket: %s", readable)
        logging.info("Warmup compile total: %.4fs", sum(warmup_timings.values()))

    ds = _load_dataset(cfg)

    git_sha = _observe_git_sha()

    created_at = _iso_now()
    run_id = cfg.run_id or created_at.replace(":", "").replace("-", "")

    max_bucket_batch = _effective_batch_size(cfg.bucket_min_tokens, cfg)

    row_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, max_bucket_batch * cfg.prefetch_batches))
    batch_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, cfg.prefetch_batches))

    loader_state = ThreadState()
    writer_state = ThreadState()
    metrics = PipelineMetrics()

    loader_thread = threading.Thread(
        target=_dataset_loader_worker,
        args=(ds, cfg, row_queue, loader_state, metrics),
        name="dataset-loader",
        daemon=True,
    )
    loader_thread.start()

    shard_writer = _ShardWriter(cfg, paths, git_sha)
    writer_thread = threading.Thread(
        target=shard_writer.run,
        args=(batch_queue, _STOP, writer_state, metrics),
        name="shard-writer",
        daemon=True,
    )
    writer_thread.start()

    stats = PipelineStats()
    progress = tqdm(disable=not cfg.progress, unit="rows")
    embedding_error: Optional[BaseException] = None

    try:
        _embedding_pipeline(
            model_runner=model_runner,
            cfg=cfg,
            created_at=created_at,
            run_id=run_id,
            row_queue=row_queue,
            batch_queue=batch_queue,
            stats=stats,
            metrics=metrics,
            progress=progress,
        )
    except BaseException as exc:  # noqa: BLE001
        embedding_error = exc
    finally:
        progress.close()
        batch_queue.put(_STOP)
        loader_thread.join()
        writer_thread.join()

    if loader_state.error is not None:
        raise RuntimeError("Dataset loader thread failed") from loader_state.error
    if writer_state.error is not None:
        raise RuntimeError("Shard writer thread failed") from writer_state.error
    if embedding_error is not None:
        raise embedding_error

    duration = time.time() - start_time
    logging.info("Pipeline stage timings: %s", metrics.to_dict(duration))

    write_manifest(
        cfg=cfg,
        paths=paths,
        shards=shard_writer.shards,
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