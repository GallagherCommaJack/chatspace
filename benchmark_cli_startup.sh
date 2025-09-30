#!/bin/bash
# Benchmark CLI startup latency

echo "=== CLI Startup Latency Benchmark ==="
echo ""

echo "1. Help command (no heavy imports):"
hyperfine --warmup 3 --runs 10 "uv run chatspace --help" 2>&1 | grep -E "(Time|Mean)"

echo ""
echo "2. Subcommand help (still no heavy imports):"
hyperfine --warmup 3 --runs 10 "uv run chatspace embed-hf --help" 2>&1 | grep -E "(Time|Mean)"

echo ""
echo "3. Config-only import (Python):"
hyperfine --warmup 3 --runs 10 "uv run python -c 'from chatspace.hf_embed import SentenceTransformerConfig'" 2>&1 | grep -E "(Time|Mean)"

echo ""
echo "4. Full import with torch (Python):"
hyperfine --warmup 3 --runs 5 "uv run python -c 'from chatspace.hf_embed import SentenceTransformerConfig, run_sentence_transformer'" 2>&1 | grep -E "(Time|Mean)"

echo ""
echo "=== Summary ==="
echo "The CLI should be fast (~70-100ms) for help and argument parsing."
echo "Heavy imports (torch, transformers) only happen when actually running embed-hf."