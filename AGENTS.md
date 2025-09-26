## Project Notes

### Overview
- **Goal**: Download datasets from various sources, sample randomly, embed with the OpenAI Embeddings API, and persist embeddings for later analysis.
- **Policy**: All large disk writes must go to the network drive at `/workspace`.

### Storage Conventions (under `/workspace`)
- **Raw datasets**: `/workspace/datasets/raw/{source}/{dataset}/{version_or_split}/...`
- **Processed datasets**: `/workspace/datasets/processed/{dataset}/{version_or_split}/...`
- **Embeddings output**: `/workspace/embeddings/{model}/{dataset}/shard-<index>.parquet`
- **Manifests / indexes**: `/workspace/indexes/{model}/{dataset}/manifest.json`
- **Caches / temp**: `/workspace/cache/...`
- **Logs**: `/workspace/logs/...`

Notes:
- Prefer columnar formats (Parquet/Arrow) for embeddings and metadata to enable efficient slicing and vector length validation.
- Each embedding shard should be immutable and accompanied by shard-level stats (row count, embedding dimension, min/max norms, created_at, tool version, git sha).
- A dataset-level `manifest.json` should list shards with checksums and byte sizes for resumability.

### Data Model (per row)
- `id`: stable identifier for the sample
- `source`: dataset source (e.g., huggingface, kaggle, http)
- `dataset`: dataset name
- `split`: optional (train/test/validation)
- `text`: input string used for embedding (after any preprocessing)
- `metadata`: JSON object with provenance, chunking info, fields used to form `text`
- `embedding`: float vector (fixed dimensionality per `model`)
- `model`: embedding model name (e.g., `text-embedding-3-small`)
- `created_at`: ISO timestamp (UTC)

### Sampling Strategy
- Use a deterministic RNG seed for reproducibility.
- Support sample count (`n`) or fraction (`p`) and optional stratification by `split`.
- Record `seed`, `n`/`p`, and selection criteria in the manifest.

### Embedding Pipeline (initial design)
1. Resolve dataset and download to `/workspace/datasets/raw/...` (streaming where possible).
2. Optionally preprocess/normalize text to `/workspace/datasets/processed/...`.
3. Sample deterministically (record seed and counts).
4. Batch inputs and call OpenAI Embeddings API with retry/backoff and rate-limit handling.
5. Write shards to `/workspace/embeddings/{model}/{dataset}/...` as Parquet with a matching manifest in `/workspace/indexes/...`.
6. Validate: embedding dim, NaNs, shard sizes; append shard stats to manifest.

### OpenAI Configuration
- Environment variables:
  - `OPENAI_API_KEY` (required)
  - `OPENAI_BASE_URL` (optional, e.g., Azure or gateway)
  - `OPENAI_EMBED_MODEL` (default model name)
  - `OPENAI_TIMEOUT` (seconds)
- Practical defaults:
  - Start with `text-embedding-3-small` for cost, allow override to `text-embedding-3-large`.
  - Token limits: ensure text chunking stays within model context window; record chunking policy in metadata.

### Throughput, Cost, and Reliability
- Batching: tune batch sizes for throughput while respecting rate limits.
- Retries: exponential backoff with jitter; classify retryable vs non-retryable errors.
- Resumability: shard by time or count; skip existing shards when resuming.
- Cost control: dry-run mode (no API calls), sample caps per run, per-run budget guardrail.

### Reproducibility & Audit
- Record: git commit sha, tool version, `uv.lock` hash excerpt, CLI args.
- Persist a `run.json` alongside outputs summarizing counts, failures, and timing.

### CLI Sketch (future)
- `download-dataset --name <org/dataset> [--split ...]`
- `embed-dataset --dataset <name> --model <model> [--n <int>|--p <float>] [--seed <int>]`
- `verify-manifest --dataset <name> --model <model>`
- `list-datasets`

### Next Steps
- Enumerate initial dataset sources and access methods.
- Decide default text fields per dataset and chunking policy.
- Implement downloader and embedder modules with the above conventions.

