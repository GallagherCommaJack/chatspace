# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**chatspace** is a dataset embedding toolkit for downloading datasets from various sources, sampling randomly, and embedding them with either OpenAI Embeddings API or local SentenceTransformer models. All embeddings and metadata are persisted to `/workspace` in columnar formats (Parquet) for later analysis.

## Storage Conventions

All large disk writes must go to `/workspace`:
- Raw datasets: `/workspace/datasets/raw/{source}/{dataset}/{version_or_split}/...`
- Processed datasets: `/workspace/datasets/processed/{dataset}/{version_or_split}/...`
- Embeddings output: `/workspace/embeddings/{model}/{dataset}/shard-<index>.parquet`
- Manifests / indexes: `/workspace/indexes/{model}/{dataset}/manifest.json`
- Caches / temp: `/workspace/cache/...`
- Logs: `/workspace/logs/...`

## Commands

### Running the CLI

The package is installed as a CLI command `chatspace` using uv:

```bash
# Download and describe a dataset (lightweight metadata only)
uv run chatspace download-dataset --name <dataset> [--split <split>] [--source huggingface]

# Embed dataset with OpenAI API (requires OPENAI_API_KEY in .env)
uv run chatspace embed-dataset --dataset <name> [--split <split>] [--n <count>|--p <fraction>] [--execute]

# Embed dataset with local SentenceTransformer model
uv run chatspace embed-hf --dataset <name> [--subset <config>] [--model <model>] [options...]
```

### Common Development Commands

```bash
# Run with uv
uv run python main.py

# Run a specific example script
bash runs/fineweb-10BT.sh
```

## Architecture

### Entry Points

- `main.py`: Simple entry point that lazy-imports `chatspace.cli:main`
- `chatspace/cli.py`: CLI argument parser and command handlers
  - `handle_download_dataset`: Scaffolds dataset metadata under `/workspace`
  - `handle_embed_dataset`: OpenAI embeddings pipeline (dry-run by default, requires `--execute`)
  - `handle_embed_hf`: Local SentenceTransformer embeddings pipeline

### Core Modules

- `chatspace/hf_embed/`: Full-featured embedding pipeline for SentenceTransformer models (modular package)
  - `config.py`: `SentenceTransformerConfig` dataclass with validation
  - `dataset.py`: Dataset loading, conversation extraction, row streaming
  - `model.py`: `_ModelRunner` class for tokenization, inference, warmup, and compilation
  - `bucketing.py`: Token bucketing (`_BucketBuffer`), batch sizing, padding logic
  - `pipeline.py`: Main orchestration (`run_sentence_transformer`), threading, encoder loop
  - `writer.py`: `_ShardWriter` for Parquet I/O, manifest generation
  - `metrics.py`: `PipelineStats`, `StageTimings`, `PipelineMetrics` for performance tracking
  - `utils.py`: Pure utility functions (paths, checksums, git SHA, ISO timestamps)

  **Key features:**
  - Streaming dataset loader with deterministic sampling
  - Token-based bucketing (power-of-2 sizes from `bucket_min_tokens` to `bucket_max_tokens`)
  - Adaptive batch sizing based on `tokens_per_batch` (overrides `batch_size` when set)
  - Multi-threaded pipeline: loader → encoder (main thread) → writer
  - Optional `torch.compile` with per-bucket compilation and warmup
  - Detailed stage timings (busy/idle time for loader, encoder, writer)
  - Parquet shards with checksums, norms, and metadata
  - Manifest generation with shard stats, git SHA, tool version

- `chatspace/env.py`: Environment variable utilities
  - `load_environment()`: Loads `.env` without overriding existing vars
  - `get_env(name, default, required)`: Safe retrieval with validation

### Pipeline Stages

The `embed-hf` command uses a three-stage pipeline:

1. **Loader** (background thread): Streams dataset rows and enqueues them
2. **Encoder** (main thread): Tokenizes, buckets by sequence length, batches, encodes, and enqueues embedded batches
3. **Writer** (background thread): Accumulates rows and writes Parquet shards when `rows_per_shard` threshold is reached

### Key Configuration

- **Token batching**: Use `--tokens-per-batch` to control batch size by total token count (e.g., 131072) instead of fixed sequence count
- **Bucketing**: Sequences are padded to the next power-of-2 length between `bucket_min_tokens` (default 128) and `bucket_max_tokens` (default 32768)
- **Compilation**: Pass `--compile-model` to enable `torch.compile` with per-bucket caching
- **Warmup**: When compilation is enabled, all bucket sizes are warmed up before the main pipeline runs

## Environment Variables

Required for OpenAI embeddings:
- `OPENAI_API_KEY`: API key for OpenAI embeddings

Optional:
- `OPENAI_BASE_URL`: Override base URL (e.g., Azure or gateway)
- `OPENAI_EMBED_MODEL`: Default model name (default: `text-embedding-3-small`)
- `OPENAI_TIMEOUT`: Request timeout in seconds

## Data Model

Each embedding row includes:
- `id`: Stable identifier for the sample
- `source`: Dataset source (e.g., "huggingface")
- `dataset`: Dataset name
- `split`: Optional split (train/test/validation)
- `text`: Input string used for embedding
- `metadata`: JSON object with provenance and field info
- `embedding`: Float vector (fixed dimensionality per model)
- `model`: Embedding model name
- `created_at`: ISO timestamp (UTC)
- `run_id`: Run identifier for tracking

## Reproducibility

Each run records:
- Git commit SHA (if available)
- Tool version (`chatspace.__version__`)
- CLI arguments in `run_config`
- Shard-level checksums (SHA256) in manifest
- Sampling seed and parameters
- Pipeline stage timings and utilization metrics