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

## Testing Guidelines

- Add or update tests in `tests/` that cover new code paths; mirror naming (`test_<module>.py`) and use `pytest` fixtures
- Validate concurrency changes with `python test_multiprocessing.py`
- Include small `uv run chatspace embed-hf --max-rows` dry runs when touching dataset or writer logic
- Guard against regressions by checking embedding dimension, norm bounds, and manifest integrity
- **IMPORTANT**: Always run tests with timeouts - bugs can cause hangs and GPU memory recovery is tricky

### vLLM Steering Tests

- `tests/test_vllm_comprehensive_integration.py`: End-to-end integration test covering:
  - Batch generation with chat formatting (10 prompts, 40 tokens each)
  - Multi-method steering (additive, projection cap, ablation on multiple layers)
  - Hidden state capture during prefill AND decode
  - HuggingFace parity validation (cosine similarity ~1.0, MAE <0.02)
  - Concurrent generation with temporal overlap verification
  - Capture isolation (concurrent requests don't mix)
  - RWLock coordination (steering changes block during generation)
- Run this test when modifying steering logic, capture mechanisms, or concurrency handling
- Expected runtime: ~20-25 seconds with CUDA available

## Coding Style

- Follow PEP 8 with 4-space indentation, descriptive snake_case for functions, UpperCamelCase for classes
- Prefer type hints and module-level docstrings; mirror existing tone in `chatspace/hf_embed/pipeline.py`
- Route diagnostics through `logging` module; avoid bare `print` except in CLI entry points
- Keep shard and manifest writers immutable: create new files rather than mutating outputs in-place

## Commit Guidelines

- Use concise, imperative commit subjects (e.g., "Fix steering vector training pipeline")
- **DO NOT commit with --amend unless explicitly asked**
- PRs should describe motivation, summarize changes, list validation commands, and link issues

## vLLM Steering Runtime Notes

- `chatspace/vllm_steering/runtime.py` monkey-patches decoder layer `forward` to add steering vectors
- **Requires eager execution**: Always launch `VLLMSteerModel` with `enforce_eager=True` (default)
- CUDA-graph capture breaks steering because compiled graphs ignore the Python-side patch
- Running via `uv run` keeps repo root on `sys.path`, so `sitecustomize.py` patch triggers automatically
- Use `scripts/steering_smoke.py` for quick verification of steering behavior
- **Qwen decoder layer fusion**: vLLM fuses RMSNorm with skip connection, returns `(mlp_delta, residual_before_mlp)` - must add `delta + residual` to mirror HuggingFace captures

### Concurrency and Threading Model

**AsyncRWLock for Steering Configuration:**
- `VLLMSteerModel` uses a readers-writer lock (`AsyncRWLock`) to coordinate concurrent operations
- **Read operations** (concurrent): Multiple `generate()` calls can run simultaneously
- **Write operations** (exclusive): Steering configuration changes block until all in-flight requests complete
- Write operations include:
  - `set_layer_vector()`, `set_layer_projection_cap()`, `set_layer_ablation()`
  - `clear_layer_projection_cap()`, `clear_layer_ablation()`, `clear_all_vectors()`
  - `apply_steering_spec()`
- Writers signal intent via `_writer_waiting` flag to prevent reader starvation
- Concurrent generation is safe and performant - requests don't block each other

### Hidden State Capture Behavior

**Capture API Structure:**
- vLLM captures return a **single concatenated tensor** per layer containing all processed tokens
- Format: `captures[layer_idx][0]["hidden"]` with shape `[seq_len, hidden_size]`
- This tensor includes both prefill and decode tokens in sequence order
- To extract specific tokens, slice the tensor by position (e.g., `captures[2][0]["hidden"][prompt_len:]` for decode-only)

**Critical: Autoregressive Generation Length**
- Captured tensors have length `prompt_tokens + (generated_tokens - 1)`, NOT `prompt_tokens + generated_tokens`
- **Why**: In autoregressive generation, the final sampled token is never processed through the model
  1. Prefill: Process all prompt tokens
  2. Decode iterations 1..(N-1): Each iteration processes a token and produces logits for the next
  3. Final iteration N: Sample from logits only - the Nth token never flows through the model
- **Example**: 15-token prompt generating 10 tokens
  - Output text: 25 tokens total
  - Captured hidden states: 24 tokens (15 prefill + 9 decode)
  - Missing: The 10th generated token (sampled but never processed)
- **When validating**: Always use `expected_len = prompt_len + (generated_len - 1)`
- This is universal LLM behavior, not vLLM-specific

**Capture Isolation:**
- Concurrent requests with capture enabled maintain proper per-request isolation
- Each request's captures are tracked independently via request IDs
- Validated in `tests/test_vllm_comprehensive_integration.py` with temporal overlap verification

## Journaling Practices

- Scratch notes and active investigations go in `TEMP_JOURNAL.md` (gitignored)
- Capture UTC timestamp with `date -u` before editing logs
- Once work stabilizes, move key findings to canonical `JOURNAL.md`
- Note tmux sessions, long-running jobs, and `/workspace` artifact paths for resumability
- **IMPORTANT**: Implementation summaries, feature documentation, and completion notes should be added to `JOURNAL.md`, NOT as separate markdown files in the repo root. Keep the repo clean by consolidating documentation in the journal.