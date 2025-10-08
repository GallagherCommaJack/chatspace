#!/usr/bin/env python3
"""
Analyze token counts in persona datasets, grouped by extract_score.

For each dataset, computes:
- Token count histogram by score (0-3 for roles, 0-100 for traits)
- Uses model-specific tokenizers with chat templates
- Generates matplotlib visualizations
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import subprocess

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Model to tokenizer mapping
MODEL_TOKENIZERS = {
    "gemma-2-27b": "google/gemma-2-27b-it",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen-3-32b": "Qwen/Qwen3-32B",
}


def get_git_sha() -> str:
    """Get current git SHA if available."""
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


def parse_dataset_name(filename: str) -> Tuple[str, str, str]:
    """
    Parse dataset filename to extract model, type, and name.

    Example: gemma-2-27b__role__accountant__min3 -> (gemma-2-27b, role, accountant)
    """
    parts = filename.split("__")
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")

    model = parts[0]
    dtype = parts[1]  # role or trait
    name = parts[2]

    return model, dtype, name


def tokenize_conversations(
    conversations: List[List[Dict[str, str]]],
    tokenizer,
) -> List[int]:
    """
    Tokenize a batch of conversations using chat template.

    Args:
        conversations: List of message lists
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        List of token counts for each conversation
    """
    token_counts = []

    for messages in conversations:
        try:
            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Tokenize
            tokens = tokenizer.encode(formatted, add_special_tokens=True)
            token_counts.append(len(tokens))
        except Exception as e:
            # If chat template fails, fall back to simple concatenation
            text = " ".join(msg["content"] for msg in messages)
            tokens = tokenizer.encode(text, add_special_tokens=True)
            token_counts.append(len(tokens))

    return token_counts


def analyze_dataset(
    parquet_path: Path,
    tokenizer,
    dataset_name: str,
) -> Dict:
    """
    Analyze a single dataset: compute token histogram by score.

    Returns:
        Dictionary with statistics
    """
    # Read parquet file
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Parse dataset name
    model, dtype, name = parse_dataset_name(dataset_name)

    # Group by extract_score
    score_groups = defaultdict(list)
    for _, row in df.iterrows():
        score = row["extract_score"]
        # Handle None/NaN scores
        if score is None or (isinstance(score, float) and pd.isna(score)):
            score = "null"
        else:
            score = str(int(score))

        score_groups[score].append(row["messages"])

    # Compute token counts for each score group
    score_histogram = {}
    total_conversations = 0
    total_tokens = 0

    for score, conversations in score_groups.items():
        token_counts = tokenize_conversations(conversations, tokenizer)
        score_tokens = sum(token_counts)
        score_convs = len(conversations)

        score_histogram[score] = {
            "conversations": score_convs,
            "tokens": score_tokens,
        }

        total_conversations += score_convs
        total_tokens += score_tokens

    return {
        "model": model,
        "type": dtype,
        "name": name,
        "total_conversations": total_conversations,
        "total_tokens": total_tokens,
        "score_histogram": score_histogram,
    }


def plot_histogram(
    stats: Dict,
    output_path: Path,
):
    """
    Create matplotlib bar chart of token counts by score.
    """
    histogram = stats["score_histogram"]

    # Sort scores (handle null separately)
    scores = sorted(
        [k for k in histogram.keys() if k != "null"],
        key=lambda x: int(x)
    )
    if "null" in histogram:
        scores.append("null")

    # Extract data
    token_counts = [histogram[score]["tokens"] for score in scores]
    conv_counts = [histogram[score]["conversations"] for score in scores]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top plot: token counts
    ax1.bar(scores, token_counts, color='steelblue', alpha=0.8)
    ax1.set_xlabel("Extract Score")
    ax1.set_ylabel("Total Tokens")
    ax1.set_title(f"{stats['model']} - {stats['type']} - {stats['name']}")
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (score, tokens) in enumerate(zip(scores, token_counts)):
        ax1.text(i, tokens, f'{tokens:,}', ha='center', va='bottom', fontsize=8)

    # Bottom plot: conversation counts
    ax2.bar(scores, conv_counts, color='coral', alpha=0.8)
    ax2.set_xlabel("Extract Score")
    ax2.set_ylabel("Conversations")
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (score, count) in enumerate(zip(scores, conv_counts)):
        ax2.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def process_model_datasets(
    model: str,
    tokenizer,
    dataset_dir: Path,
    output_dir: Path,
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Process all datasets for a single model.

    Returns:
        (stats_dict, errors)
    """
    # Find all parquet files for this model
    parquet_files = sorted(dataset_dir.glob(f"{model}__*.parquet"))

    stats = {}
    errors = []

    print(f"\n=== Processing {model} ({len(parquet_files)} datasets) ===")

    for parquet_path in tqdm(parquet_files, desc=model):
        dataset_name = parquet_path.stem

        try:
            # Analyze dataset
            dataset_stats = analyze_dataset(parquet_path, tokenizer, dataset_name)
            stats[dataset_name] = dataset_stats

            # Create plot
            plot_path = output_dir / f"{dataset_name}.png"
            plot_histogram(dataset_stats, plot_path)

        except Exception as e:
            errors.append(f"{dataset_name}: {str(e)}")

    return stats, errors


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze token counts in persona datasets")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/workspace/datasets/processed/persona"),
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/workspace/persona_token_stats.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--output-plots",
        type=Path,
        default=Path("/workspace/persona_token_plots"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma-2-27b", "llama-3.3-70b", "qwen-3-32b"],
        help="Models to process"
    )

    args = parser.parse_args()

    # Create output directories
    args.output_plots.mkdir(parents=True, exist_ok=True)

    # Process each model sequentially (with their datasets in parallel)
    all_stats = {}
    all_errors = []

    for model in args.models:
        print(f"\nLoading tokenizer: {MODEL_TOKENIZERS[model]}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZERS[model])
        except Exception as e:
            print(f"ERROR: Failed to load tokenizer for {model}: {e}")
            continue

        # Process all datasets for this model
        model_stats, model_errors = process_model_datasets(
            model,
            tokenizer,
            args.dataset_dir,
            args.output_plots,
        )

        all_stats.update(model_stats)
        all_errors.extend(model_errors)

        print(f"✓ Processed {len(model_stats)} datasets for {model}")
        if model_errors:
            print(f"✗ {len(model_errors)} errors")

    # Load existing stats if file exists (for merging multiple runs)
    existing_stats = {}
    if args.output_json.exists():
        try:
            with open(args.output_json, "r") as f:
                existing_data = json.load(f)
                existing_stats = existing_data.get("datasets", {})
            print(f"\nMerging with {len(existing_stats)} existing datasets")
        except Exception:
            pass

    # Merge with existing
    existing_stats.update(all_stats)

    # Save results
    output = {
        "metadata": {
            "total_datasets": len(existing_stats),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": get_git_sha(),
        },
        "datasets": existing_stats,
    }

    if all_errors:
        output["errors"] = all_errors

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved statistics to {args.output_json}")
    print(f"✓ Saved {len(all_stats)} plots to {args.output_plots}")

    if all_errors:
        print(f"\n✗ {len(all_errors)} errors occurred:")
        for error in all_errors[:10]:
            print(f"  {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more")


if __name__ == "__main__":
    import pandas as pd  # Import here to catch missing dependency
    main()
