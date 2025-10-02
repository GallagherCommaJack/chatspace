#!/usr/bin/env python3
"""
Merge token statistics from multiple analysis runs by reading plot filenames.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess


def get_git_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    plots_dir = Path("/workspace/persona_token_plots")

    # Get all plot files
    plot_files = list(plots_dir.glob("*.png"))

    print(f"Found {len(plot_files)} plot files")

    # Build minimal stats structure
    datasets = {}

    for plot_file in plot_files:
        dataset_name = plot_file.stem

        # Parse dataset name
        parts = dataset_name.split("__")
        if len(parts) >= 3:
            model = parts[0]
            dtype = parts[1]
            name = "__".join(parts[2:])  # Handle names with __ in them

            datasets[dataset_name] = {
                "model": model,
                "type": dtype,
                "name": name,
                "plot": str(plot_file),
            }

    output = {
        "metadata": {
            "total_datasets": len(datasets),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": get_git_sha(),
            "note": "Reconstructed from plot filenames - full token stats available in per-model runs",
        },
        "datasets": datasets,
    }

    output_file = Path("/workspace/persona_token_stats.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    models = {}
    types = {}
    for ds in datasets.values():
        models[ds["model"]] = models.get(ds["model"], 0) + 1
        types[ds["type"]] = types.get(ds["type"], 0) + 1

    print(f"\n✓ Reconstructed stats for {len(datasets)} datasets")
    print(f"\nBy model:")
    for model, count in sorted(models.items()):
        print(f"  {model}: {count}")
    print(f"\nBy type:")
    for dtype, count in sorted(types.items()):
        print(f"  {dtype}: {count}")
    print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
