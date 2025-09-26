from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

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


def _batched(iterable: Iterable[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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


def _tensor_from_embeddings(embeddings: List[List[float]]) -> torch.Tensor:
    return torch.tensor(embeddings, dtype=torch.float32)


def _convert_embeddings(numpy_embeddings: Any) -> List[List[float]]:
    if hasattr(numpy_embeddings, "tolist"):
        return numpy_embeddings.astype("float32").tolist()
    return [[float(x) for x in row] for row in numpy_embeddings]


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

    git_sha = _observe_git_sha()

    created_at = _iso_now()
    run_id = cfg.run_id or created_at.replace(":", "").replace("-", "")

    shards: List[Dict[str, Any]] = []
    total_rows = 0
    skipped_rows = 0
    embedding_dim: Optional[int] = None
    min_norm: Optional[float] = None
    max_norm: Optional[float] = None

    current_shard_rows: List[Dict[str, Any]] = []
    shard_index = 0

    iterator = _rows_from_dataset(ds, cfg)
    progress = tqdm(disable=not cfg.progress, unit="rows")

    for batch in _batched(iterator, cfg.batch_size):
        texts = [str(row.get(cfg.text_field, "")) for row in batch]

        filtered_indices: List[int] = []
        filtered_rows: List[Dict[str, Any]] = []
        for idx, (row, text) in enumerate(zip(batch, texts)):
            if not text:
                skipped_rows += 1
                continue
            filtered_indices.append(idx)
            filtered_rows.append(row)

        if not filtered_rows:
            continue

        with torch.inference_mode():
            embeddings_array = model.encode(
                [texts[i] for i in filtered_indices],
                batch_size=cfg.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
            )

        embeddings_list = _convert_embeddings(embeddings_array)
        embeddings_tensor = _tensor_from_embeddings(embeddings_list)
        batch_norms = _compute_norms(embeddings_tensor)

        if embedding_dim is None:
            embedding_dim = embeddings_tensor.shape[1]

        for row, embedding, norm in zip(filtered_rows, embeddings_list, batch_norms.tolist()):
            augmented_row = dict(row)
            augmented_row["embedding"] = embedding
            augmented_row["model"] = cfg.model_name
            augmented_row["created_at"] = created_at
            augmented_row["run_id"] = run_id
            current_shard_rows.append(augmented_row)

            min_norm = norm if min_norm is None else min(min_norm, norm)
            max_norm = norm if max_norm is None else max(max_norm, norm)

        batch_size_effective = len(filtered_rows)
        total_rows += batch_size_effective
        progress.update(batch_size_effective)

        if cfg.max_rows is not None and total_rows >= cfg.max_rows:
            current_shard_rows = current_shard_rows[: cfg.max_rows - (total_rows - batch_size_effective)]
            total_rows = cfg.max_rows

        while current_shard_rows and len(current_shard_rows) >= cfg.rows_per_shard:
            shard_rows = current_shard_rows[: cfg.rows_per_shard]
            current_shard_rows = current_shard_rows[cfg.rows_per_shard :]
            shard_created_at = _iso_now()
            shard_path = paths["embeddings_dir"] / f"shard-{shard_index:05d}.parquet"
            table = pa.Table.from_pylist(shard_rows)
            pq.write_table(table, shard_path)
            file_size = shard_path.stat().st_size
            checksum = _compute_sha256(shard_path)

            shard_norms_tensor = _tensor_from_embeddings([row["embedding"] for row in shard_rows])
            shard_norms = _compute_norms(shard_norms_tensor)

            shards.append(
                {
                    "path": str(shard_path),
                    "rows": len(shard_rows),
                    "bytes": file_size,
                    "sha256": checksum,
                    "embedding_dim": embedding_dim,
                    "min_norm": float(shard_norms.min().item()),
                    "max_norm": float(shard_norms.max().item()),
                    "created_at": shard_created_at,
                    "shard_index": shard_index,
                    "tool": {
                        "package": "chatspace",
                        "version": CHATSPACE_VERSION,
                        "git_sha": git_sha,
                    },
                }
            )
            shard_index += 1

            if cfg.max_rows is not None and total_rows >= cfg.max_rows:
                break

        if cfg.max_rows is not None and total_rows >= cfg.max_rows:
            break

    progress.close()

    if current_shard_rows:
        shard_created_at = _iso_now()
        shard_path = paths["embeddings_dir"] / f"shard-{shard_index:05d}.parquet"
        table = pa.Table.from_pylist(current_shard_rows)
        pq.write_table(table, shard_path)
        file_size = shard_path.stat().st_size
        checksum = _compute_sha256(shard_path)
        shard_norms_tensor = _tensor_from_embeddings([row["embedding"] for row in current_shard_rows])
        shard_norms = _compute_norms(shard_norms_tensor)
        shards.append(
            {
                "path": str(shard_path),
                "rows": len(current_shard_rows),
                "bytes": file_size,
                "sha256": checksum,
                "embedding_dim": embedding_dim,
                "min_norm": float(shard_norms.min().item()),
                "max_norm": float(shard_norms.max().item()),
                "created_at": shard_created_at,
                "shard_index": shard_index,
                "tool": {
                    "package": "chatspace",
                    "version": CHATSPACE_VERSION,
                    "git_sha": git_sha,
                },
            }
        )

    duration = time.time() - start_time
    manifest = {
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "split": cfg.split,
        "model": cfg.model_name,
        "source": cfg.source_label,
        "rows_total": total_rows,
        "rows_skipped": skipped_rows,
        "embedding_dim": embedding_dim,
        "rows_per_shard": cfg.rows_per_shard,
        "shards": shards,
        "created_at": created_at,
        "run_id": run_id,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "run_config": _config_to_dict(cfg),
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
        "rows_total": total_rows,
        "rows_skipped": skipped_rows,
        "num_shards": len(shards),
        "embedding_dim": embedding_dim,
        "git_sha": git_sha,
        "manifest_path": str(manifest_path),
        "min_norm": min_norm,
        "max_norm": max_norm,
        "tool_version": CHATSPACE_VERSION,
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
        progress=args.progress,
        run_id=args.run_id,
        resume=args.resume,
    )

    run_sentence_transformer(cfg)


if __name__ == "__main__":
    main()

