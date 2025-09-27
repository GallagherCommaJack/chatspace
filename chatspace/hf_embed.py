from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import queue
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import IterableDataset, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import __version__ as CHATSPACE_VERSION


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_component(value: str) -> str:
    return value.replace("/", "__")


def _compute_sha256(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class SentenceTransformerConfig:
    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 32
    rows_per_shard: int = 8192
    max_rows: Optional[int] = None
    output_root: Path = Path("/workspace")
    manifest_relpath: Optional[Path] = None
    seed: Optional[int] = None
    dtype: Optional[str] = "bfloat16"
    device: Optional[str] = None
    attention_impl: Optional[str] = "flash_attention_2"
    tokenizer_padding: str = "left"
    trust_remote_code: bool = True
    num_workers: int = 1
    prefetch_batches: int = 4
    bucket_min_tokens: int = 128
    bucket_max_tokens: int = 32768
    tokens_per_batch: Optional[int] = None
    compile_model: bool = False
    compile_mode: Optional[str] = "default"
    progress: bool = True
    source_label: str = "huggingface"
    run_id: Optional[str] = None
    max_rows_per_file: Optional[int] = None
    resume: bool = False
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")
        if self.bucket_min_tokens <= 0:
            raise ValueError("bucket_min_tokens must be positive")
        if self.bucket_max_tokens <= 0:
            raise ValueError("bucket_max_tokens must be positive")
        if self.bucket_min_tokens > self.bucket_max_tokens:
            raise ValueError("bucket_min_tokens must be less than or equal to bucket_max_tokens")
        if self.tokens_per_batch is not None and self.tokens_per_batch <= 0:
            raise ValueError("tokens_per_batch must be positive if provided")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive if provided")


def _default_model_kwargs(cfg: SentenceTransformerConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if cfg.attention_impl:
        kwargs["attn_implementation"] = cfg.attention_impl
    if cfg.device:
        kwargs["device_map"] = cfg.device
    if cfg.dtype:
        kwargs["dtype"] = cfg.dtype
    kwargs.update(cfg.model_kwargs)
    return kwargs


def _default_tokenizer_kwargs(cfg: SentenceTransformerConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "padding": True,
        "truncation": True,
    }
    if cfg.tokenizer_padding:
        kwargs["padding_side"] = cfg.tokenizer_padding
    kwargs.update(cfg.tokenizer_kwargs)
    return kwargs


def _load_dataset(cfg: SentenceTransformerConfig) -> IterableDataset:
    dataset_kwargs: Dict[str, Any] = {}
    if cfg.subset:
        dataset_kwargs["name"] = cfg.subset
    logging.info("Loading dataset %s subset=%s split=%s (streaming)", cfg.dataset, cfg.subset, cfg.split)
    return load_dataset(cfg.dataset, split=cfg.split, streaming=True, **dataset_kwargs)


def _load_model(cfg: SentenceTransformerConfig) -> SentenceTransformer:
    logging.info("Loading SentenceTransformer model: %s", cfg.model_name)
    model_kwargs = _default_model_kwargs(cfg)
    tokenizer_kwargs = _default_tokenizer_kwargs(cfg)
    try:
        model = SentenceTransformer(
            cfg.model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=cfg.trust_remote_code,
        )
    except ValueError as exc:
        if cfg.attention_impl and "Flash Attention" in str(exc):
            logging.warning("Model %s does not support attention implementation '%s'; retrying with default.", cfg.model_name, cfg.attention_impl)
            model_kwargs.pop("attn_implementation", None)
            model = SentenceTransformer(
                cfg.model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                trust_remote_code=cfg.trust_remote_code,
            )
        else:
            raise
    model = model.eval()
    if hasattr(model, "requires_grad_"):
        model.requires_grad_(False)
    return model


def _prepare_paths(cfg: SentenceTransformerConfig) -> Dict[str, Path]:
    dataset_component = _sanitize_component(cfg.dataset)
    if cfg.subset:
        dataset_component = f"{dataset_component}__{_sanitize_component(cfg.subset)}"
    split_component = _sanitize_component(cfg.split)
    model_component = _sanitize_component(cfg.model_name)

    embeddings_dir = cfg.output_root / "embeddings" / model_component / dataset_component / split_component
    indexes_dir = cfg.output_root / "indexes" / model_component / dataset_component
    cache_dir = cfg.output_root / "cache"
    logs_dir = cfg.output_root / "logs"

    for path in [embeddings_dir, indexes_dir, cache_dir, logs_dir]:
        _ensure_dir(path)

    manifest_path = cfg.manifest_relpath
    if manifest_path is None:
        manifest_path = indexes_dir / f"manifest-{split_component}.json"

    run_path = indexes_dir / f"run-{split_component}.json"

    return {
        "embeddings_dir": embeddings_dir,
        "indexes_dir": indexes_dir,
        "manifest_path": manifest_path,
        "run_path": run_path,
        "logs_dir": logs_dir,
    }


def _observe_git_sha() -> Optional[str]:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _rows_from_dataset(ds: IterableDataset, cfg: SentenceTransformerConfig) -> Iterator[Dict[str, Any]]:
    for idx, row in enumerate(ds):
        if cfg.max_rows is not None and idx >= cfg.max_rows:
            break
        if cfg.seed is not None:
            # Deterministic hashing to decide keep/drop could be added here; for now no-op.
            pass
        yield dict(row)


def _compute_norms(embedding_batch: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(embedding_batch, dim=1)


@dataclass
class PipelineStats:
    total_rows: int = 0
    skipped_rows: int = 0
    embedding_dim: Optional[int] = None
    min_norm: Optional[float] = None
    max_norm: Optional[float] = None

    def register_rows(self, count: int) -> None:
        self.total_rows += count

    def register_skipped(self, count: int = 1) -> None:
        self.skipped_rows += count

    def update_embedding_dim(self, dim: int) -> None:
        if self.embedding_dim is None:
            self.embedding_dim = dim
        elif dim is not None and self.embedding_dim != dim:
            raise ValueError(f"Inconsistent embedding dimension: expected {self.embedding_dim}, received {dim}")

    def update_norm_bounds(self, min_norm: Optional[float], max_norm: Optional[float]) -> None:
        if min_norm is None or max_norm is None:
            return
        self.min_norm = min_norm if self.min_norm is None else min(self.min_norm, min_norm)
        self.max_norm = max_norm if self.max_norm is None else max(self.max_norm, max_norm)


@dataclass
class StageTimings:
    busy_seconds: float = 0.0
    idle_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def add_busy(self, delta: float) -> None:
        if delta <= 0:
            return
        with self._lock:
            self.busy_seconds += delta

    def add_idle(self, delta: float) -> None:
        if delta <= 0:
            return
        with self._lock:
            self.idle_seconds += delta

    def to_dict(self, total_duration: float) -> Dict[str, Any]:
        total_stage = self.busy_seconds + self.idle_seconds
        utilization = (self.busy_seconds / total_duration) if total_duration > 0 else None
        stage_busy_fraction = (self.busy_seconds / total_stage) if total_stage > 0 else None
        return {
            "busy_seconds": self.busy_seconds,
            "idle_seconds": self.idle_seconds,
            "busy_fraction_of_stage": stage_busy_fraction,
            "busy_fraction_of_run": utilization,
        }


@dataclass
class PipelineMetrics:
    loader: StageTimings = field(default_factory=StageTimings)
    encoder: StageTimings = field(default_factory=StageTimings)
    writer: StageTimings = field(default_factory=StageTimings)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    encoder_encode_seconds: float = 0.0

    def add_encoder_encode(self, delta: float) -> None:
        if delta <= 0:
            return
        with self._lock:
            self.encoder_encode_seconds += delta

    def to_dict(self, total_duration: float) -> Dict[str, Any]:
        return {
            "loader": self.loader.to_dict(total_duration),
            "encoder": {
                **self.encoder.to_dict(total_duration),
                "encode_call_seconds": self.encoder_encode_seconds,
            },
            "writer": self.writer.to_dict(total_duration),
        }


@dataclass
class TokenBatch:
    rows: List[Dict[str, Any]]
    features: Dict[str, torch.Tensor]
    bucket_size: int


class _BucketBuffer:
    def __init__(self, bucket_size: int) -> None:
        self.bucket_size = bucket_size
        self.rows: List[Dict[str, Any]] = []
        self.tokens: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def add(self, row: Dict[str, Any], tokenized: Dict[str, torch.Tensor]) -> None:
        self.rows.append(row)
        for key, tensor in tokenized.items():
            self.tokens[key].append(tensor)

    def __len__(self) -> int:
        return len(self.rows)

    def pop(self, count: int, pad_values: Dict[str, int]) -> Optional[TokenBatch]:
        if count <= 0 or not self.rows:
            return None
        count = min(count, len(self.rows))
        rows = self.rows[:count]
        token_slices = {key: value[:count] for key, value in self.tokens.items()}
        features = _pad_and_stack_tokens(token_slices, self.bucket_size, pad_values)
        self.rows = self.rows[count:]
        for key in list(self.tokens.keys()):
            self.tokens[key] = self.tokens[key][count:]
        return TokenBatch(rows=rows, features=features, bucket_size=self.bucket_size)

    def flush(self, pad_values: Dict[str, int]) -> Optional[TokenBatch]:
        return self.pop(len(self.rows), pad_values)


class _ModelRunner:
    def __init__(self, model: SentenceTransformer, compile_enabled: bool, compile_mode: Optional[str]) -> None:
        self.model = model
        self.device = model.device
        self.compile_enabled = compile_enabled
        self.compile_mode = compile_mode or "default"
        self._compiled_forward: Optional[Any] = None

        pad_token_id = getattr(getattr(model, "tokenizer", None), "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0
        self.pad_values: Dict[str, int] = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "token_type_ids": 0,
        }

        if self.compile_enabled and not hasattr(torch, "compile"):
            logging.warning("torch.compile requested but not available in this PyTorch build; disabling compilation.")
            self.compile_enabled = False

        if self.compile_enabled:
            try:
                def forward_fn(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                    return model.forward(features)

                self._compiled_forward = torch.compile(forward_fn, mode=self.compile_mode)
            except Exception:
                logging.warning("torch.compile failed; continuing without compilation", exc_info=True)
                self.compile_enabled = False
                self._compiled_forward = None

        self.model.eval()

    def tokenize(self, text: str, *, max_length: int) -> Dict[str, torch.Tensor]:
        tokenized = self.model.tokenize([text], max_length=max_length, padding=False, truncation=True)
        return {key: value.squeeze(0) for key, value in tokenized.items()}

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device_features = {key: value.to(self.device, non_blocking=True) for key, value in features.items()}
        if self.compile_enabled and self._compiled_forward is not None:
            return self._compiled_forward(device_features)
        return self.model.forward(device_features)


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def _select_bucket_size(length: int, cfg: SentenceTransformerConfig) -> int:
    bucket = max(cfg.bucket_min_tokens, _next_power_of_two(length))
    return min(bucket, cfg.bucket_max_tokens)


def _token_sequence_length(tokens: Dict[str, torch.Tensor]) -> int:
    if "attention_mask" in tokens:
        return int(tokens["attention_mask"].sum().item())
    if "input_ids" in tokens:
        return int(tokens["input_ids"].shape[-1])
    for value in tokens.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[-1])
    return 0


def _pad_and_stack_tokens(
    token_slices: Dict[str, List[torch.Tensor]], bucket_size: int, pad_values: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    features: Dict[str, torch.Tensor] = {}
    for key, tensors in token_slices.items():
        if not tensors:
            continue
        pad_value = pad_values.get(key, 0)
        padded: List[torch.Tensor] = []
        for tensor in tensors:
            if tensor.ndimension() == 0:
                tensor = tensor.unsqueeze(0)
            if tensor.ndimension() > 1:
                tensor = tensor.view(-1)
            current_len = tensor.shape[-1]
            if current_len > bucket_size:
                tensor = tensor[..., :bucket_size]
                current_len = bucket_size
            if current_len < bucket_size:
                pad_shape = (bucket_size - current_len,)
                pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad_tensor], dim=-1)
            padded.append(tensor)
        if padded:
            stacked = torch.stack(padded, dim=0)
            features[key] = stacked
    return features


def _effective_batch_size(bucket_size: int, cfg: SentenceTransformerConfig) -> int:
    if cfg.tokens_per_batch is not None:
        sequences = cfg.tokens_per_batch // max(bucket_size, 1)
        return max(sequences, 1)
    return max(cfg.batch_size, 1)


@dataclass
class EmbeddedBatch:
    rows: List[Dict[str, Any]]
    embeddings: torch.Tensor


@dataclass
class ThreadState:
    error: Optional[BaseException] = None


_STOP = object()


def _dataset_loader_worker(ds: IterableDataset, cfg: SentenceTransformerConfig, row_queue: "queue.Queue[Any]", state: ThreadState, metrics: PipelineMetrics) -> None:
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
    batch_queue: "queue.Queue[Any]",
    progress: tqdm,
) -> bool:
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

    batch_queue.put(EmbeddedBatch(rows=rows, embeddings=embeddings))
    return reached_limit


def _embedding_pipeline(
    model_runner: _ModelRunner,
    cfg: SentenceTransformerConfig,
    created_at: str,
    run_id: str,
    *,
    row_queue: "queue.Queue[Any]",
    batch_queue: "queue.Queue[Any]",
    stats: PipelineStats,
    metrics: PipelineMetrics,
    progress: tqdm,
) -> None:
    buckets: Dict[int, _BucketBuffer] = {}
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


class _ShardWriter:
    def __init__(self, cfg: SentenceTransformerConfig, paths: Dict[str, Path], git_sha: Optional[str]) -> None:
        self._cfg = cfg
        self._paths = paths
        self._git_sha = git_sha
        self._current_rows: List[Dict[str, Any]] = []
        self._current_norms: List[float] = []
        self._shards: List[Dict[str, Any]] = []
        self._shard_index = 0

    @property
    def shards(self) -> List[Dict[str, Any]]:
        return self._shards

    def run(self, batch_queue: "queue.Queue[Any]", stop_token: Any, state: ThreadState, metrics: PipelineMetrics) -> None:
        try:
            while True:
                wait_start = time.perf_counter()
                item = batch_queue.get()
                wait_end = time.perf_counter()
                metrics.writer.add_idle(wait_end - wait_start)

                if item is stop_token:
                    break

                busy_start = time.perf_counter()
                self._append_batch(item)
                metrics.writer.add_busy(time.perf_counter() - busy_start)

            busy_start = time.perf_counter()
            self._flush_remaining()
            metrics.writer.add_busy(time.perf_counter() - busy_start)
        except BaseException as exc:  # noqa: BLE001
            state.error = exc

    def _append_batch(self, batch: EmbeddedBatch) -> None:
        if not batch.rows:
            return

        embeddings_cpu = batch.embeddings.detach().to(device="cpu", dtype=torch.float32)
        if embeddings_cpu.shape[0] != len(batch.rows):
            raise ValueError(
                f"Mismatch between rows ({len(batch.rows)}) and embeddings ({embeddings_cpu.shape[0]})"
            )

        embeddings_list = embeddings_cpu.tolist()
        norms_tensor = torch.linalg.vector_norm(embeddings_cpu, dim=1)
        norms_list = norms_tensor.tolist()

        for row, embedding in zip(batch.rows, embeddings_list):
            row["embedding"] = embedding

        self._current_rows.extend(batch.rows)
        self._current_norms.extend(norms_list)
        while len(self._current_rows) >= self._cfg.rows_per_shard:
            self._write_shard(self._cfg.rows_per_shard)

    def _flush_remaining(self) -> None:
        if self._current_rows:
            self._write_shard(len(self._current_rows))

    def _write_shard(self, rows_to_write: int) -> None:
        shard_rows = self._current_rows[:rows_to_write]
        shard_norms = self._current_norms[:rows_to_write]
        self._current_rows = self._current_rows[rows_to_write:]
        self._current_norms = self._current_norms[rows_to_write:]

        shard_created_at = _iso_now()
        shard_path = self._paths["embeddings_dir"] / f"shard-{self._shard_index:05d}.parquet"
        table = pa.Table.from_pylist(shard_rows)
        pq.write_table(table, shard_path)
        file_size = shard_path.stat().st_size
        checksum = _compute_sha256(shard_path)

        shard_min_norm = min(shard_norms) if shard_norms else None
        shard_max_norm = max(shard_norms) if shard_norms else None
        embedding_dim = len(shard_rows[0]["embedding"]) if shard_rows and shard_rows[0].get("embedding") is not None else None

        self._shards.append(
            {
                "path": str(shard_path),
                "rows": len(shard_rows),
                "bytes": file_size,
                "sha256": checksum,
                "embedding_dim": embedding_dim,
                "min_norm": float(shard_min_norm) if shard_min_norm is not None else None,
                "max_norm": float(shard_max_norm) if shard_max_norm is not None else None,
                "created_at": shard_created_at,
                "shard_index": self._shard_index,
                "tool": {
                    "package": "chatspace",
                    "version": CHATSPACE_VERSION,
                    "git_sha": self._git_sha,
                },
            }
        )
        self._shard_index += 1


def _config_to_dict(cfg: SentenceTransformerConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    data["output_root"] = str(cfg.output_root)
    if cfg.manifest_relpath is not None:
        data["manifest_relpath"] = str(cfg.manifest_relpath)
    return data


def run_sentence_transformer(cfg: SentenceTransformerConfig) -> Dict[str, Any]:
    start_time = time.time()
    paths = _prepare_paths(cfg)
    ds = _load_dataset(cfg)
    model = _load_model(cfg)

    try:
        model.max_seq_length = cfg.bucket_max_tokens
    except Exception:
        logging.debug("Unable to set model.max_seq_length explicitly; using model default.")

    model_runner = _ModelRunner(model, cfg.compile_model, cfg.compile_mode)

    git_sha = _observe_git_sha()

    created_at = _iso_now()
    run_id = cfg.run_id or created_at.replace(":", "").replace("-", "")

    max_bucket_batch = _effective_batch_size(cfg.bucket_min_tokens, cfg)

    row_queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, max_bucket_batch * cfg.prefetch_batches))
    batch_queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, cfg.prefetch_batches))

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
    metrics_summary = metrics.to_dict(duration)
    logging.info("Pipeline stage timings: %s", metrics_summary)
    manifest = {
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "split": cfg.split,
        "model": cfg.model_name,
        "source": cfg.source_label,
        "rows_total": stats.total_rows,
        "rows_skipped": stats.skipped_rows,
        "embedding_dim": stats.embedding_dim,
        "rows_per_shard": cfg.rows_per_shard,
        "shards": shard_writer.shards,
        "created_at": created_at,
        "run_id": run_id,
        "min_norm": stats.min_norm,
        "max_norm": stats.max_norm,
        "run_config": _config_to_dict(cfg),
        "timings": metrics_summary,
    }

    manifest_path = paths["manifest_path"]
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    run_summary = {
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "split": cfg.split,
        "model": cfg.model_name,
        "created_at": created_at,
        "run_id": run_id,
        "duration_seconds": duration,
        "rows_total": stats.total_rows,
        "rows_skipped": stats.skipped_rows,
        "num_shards": len(shard_writer.shards),
        "embedding_dim": stats.embedding_dim,
        "git_sha": git_sha,
        "manifest_path": str(manifest_path),
        "min_norm": stats.min_norm,
        "max_norm": stats.max_norm,
        "tool_version": CHATSPACE_VERSION,
        "timings": metrics_summary,
    }

    run_path = paths["run_path"]
    with run_path.open("w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)

    logging.info("Wrote manifest: %s", manifest_path)
    logging.info("Wrote run summary: %s", run_path)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_summary": run_summary,
        "run_path": run_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed datasets with SentenceTransformer models")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., 'HuggingFaceFW/fineweb'")
    parser.add_argument("--subset", default=None, help="Dataset config/subset name (e.g., 'sample-10BT')")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--text-field", dest="text_field", default="text", help="Field in the dataset containing text")
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen3-Embedding-0.6B", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32, help="Batch size for embedding calls")
    parser.add_argument("--rows-per-shard", dest="rows_per_shard", type=int, default=8192, help="Number of rows per Parquet shard")
    parser.add_argument("--max-rows", dest="max_rows", type=int, default=None, help="Optional cap on processed rows (for testing)")
    parser.add_argument("--output-root", dest="output_root", default="/workspace", help="Base output directory")
    parser.add_argument("--dtype", dest="dtype", default="bfloat16", help="Model dtype hint (e.g., float16, bfloat16)")
    parser.add_argument("--device", dest="device", default=None, help="Device map hint (e.g., 'cuda', 'auto')")
    parser.add_argument("--attention", dest="attention_impl", default="flash_attention_2", help="Attention implementation hint")
    parser.add_argument("--prefetch-batches", dest="prefetch_batches", type=int, default=4, help="Number of dataset batches to buffer ahead of the encoder")
    parser.add_argument("--bucket-min-tokens", dest="bucket_min_tokens", type=int, default=128, help="Minimum sequence length bucket (tokens)")
    parser.add_argument("--bucket-max-tokens", dest="bucket_max_tokens", type=int, default=32768, help="Maximum sequence length bucket (tokens)")
    parser.add_argument("--tokens-per-batch", dest="tokens_per_batch", type=int, default=None, help="Target number of tokens per batch (overrides --batch-size when set)")
    parser.add_argument("--compile-model", dest="compile_model", action="store_true", help="Enable torch.compile for the model forward pass")
    parser.add_argument("--compile-mode", dest="compile_mode", default="default", help="torch.compile mode to use when compilation is enabled")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Optional run identifier")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Future placeholder for resuming runs")
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Logging level")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(asctime)s] %(levelname)s %(message)s")

    cfg = SentenceTransformerConfig(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        text_field=args.text_field,
        model_name=args.model_name,
        batch_size=args.batch_size,
        rows_per_shard=args.rows_per_shard,
        max_rows=args.max_rows,
        output_root=Path(args.output_root),
        dtype=args.dtype,
        device=args.device,
        attention_impl=args.attention_impl,
        prefetch_batches=args.prefetch_batches,
        progress=args.progress,
        run_id=args.run_id,
        resume=args.resume,
        bucket_min_tokens=args.bucket_min_tokens,
        bucket_max_tokens=args.bucket_max_tokens,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        tokens_per_batch=args.tokens_per_batch,
    )

    run_sentence_transformer(cfg)


if __name__ == "__main__":
    main()

