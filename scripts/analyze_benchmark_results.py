#!/usr/bin/env python3
"""
Analyze benchmark results and generate summary report.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_results(results_file):
    """Analyze a single benchmark results file."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    print("="*80)
    print(f"Analyzing: {results_file.name}")
    print("="*80)
    print(f"Model: {data['metadata']['model_name']}")
    print(f"Total layers: {data['metadata']['total_model_layers']}")
    print(f"Configurations tested: {len(data['results'])}\n")

    # Collect statistics
    all_overheads = [r['metrics']['overhead_pct'] for r in data['results']]

    print(f"Overall statistics:")
    print(f"  Min overhead: {min(all_overheads):6.2f}%")
    print(f"  Max overhead: {max(all_overheads):6.2f}%")
    print(f"  Avg overhead: {sum(all_overheads)/len(all_overheads):6.2f}%")
    print(f"  Median overhead: {sorted(all_overheads)[len(all_overheads)//2]:6.2f}%\n")

    # Analyze by layer count
    by_layers = defaultdict(list)
    for r in data['results']:
        nlayers = r['config']['num_layers_captured']
        by_layers[nlayers].append(r['metrics']['overhead_pct'])

    print("Average overhead by layer count:")
    for nlayers in sorted(by_layers.keys()):
        overheads = by_layers[nlayers]
        avg = sum(overheads) / len(overheads)
        print(f"  {nlayers:2d} layers: {avg:6.2f}%  (n={len(overheads):3d}, range: {min(overheads):.2f}%-{max(overheads):.2f}%)")

    # Analyze by scenario if present
    by_scenario = defaultdict(list)
    for r in data['results']:
        scenario = r['config'].get('scenario', 'unknown')
        by_scenario[scenario].append(r['metrics']['overhead_pct'])

    if len(by_scenario) > 1:
        print("\nAverage overhead by scenario:")
        for scenario in sorted(by_scenario.keys()):
            overheads = by_scenario[scenario]
            avg = sum(overheads) / len(overheads)
            print(f"  {scenario:15s}: {avg:6.2f}%  (n={len(overheads):3d})")

    # Analyze by batch size
    by_batch = defaultdict(list)
    for r in data['results']:
        batch = r['config']['batch_size']
        by_batch[batch].append(r['metrics']['overhead_pct'])

    print("\nAverage overhead by batch size:")
    for batch in sorted(by_batch.keys()):
        overheads = by_batch[batch]
        avg = sum(overheads) / len(overheads)
        print(f"  Batch {batch:3d}: {avg:6.2f}%  (n={len(overheads):3d})")

    # Top 10 highest overhead
    sorted_results = sorted(data['results'], key=lambda r: r['metrics']['overhead_pct'], reverse=True)
    print("\nTop 10 highest overhead configurations:")
    for i, r in enumerate(sorted_results[:10], 1):
        cfg = r['config']
        m = r['metrics']
        scenario = cfg.get('scenario', cfg.get('layer_desc', 'unknown'))
        batch = cfg['batch_size']
        layers = cfg['num_layers_captured']
        prefill = cfg.get('prefill_tokens', cfg.get('prefill_len', '?'))
        decode = cfg.get('decode_tokens', cfg.get('decode_len', '?'))
        overhead = m['overhead_pct']
        throughput = m.get('throughput_tok_s', m.get('tokens_per_sec', 0))

        print(f"  {i:2d}. {scenario:15s} | batch={batch:3d} | layers={layers:2d} | "
              f"prefill={str(prefill):>4s} | decode={str(decode):>3s} | overhead={overhead:6.2f}% | "
              f"throughput={throughput:8.1f} tok/s")

    print("\n")

def main():
    # Find all result files
    benchmark_dir = Path("/workspace/benchmarks")
    result_files = sorted(benchmark_dir.glob("*.json"))

    if not result_files:
        print("No benchmark results found in /workspace/benchmarks/")
        sys.exit(1)

    print("Found benchmark result files:")
    for f in result_files:
        print(f"  - {f.name}")
    print()

    # Analyze each file
    for f in result_files:
        analyze_results(f)

if __name__ == "__main__":
    main()
