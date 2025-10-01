#!/usr/bin/env python3
"""
Generate HuggingFace datasets for all (model, role) and (model, trait) pairs.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatspace.persona_to_hf import (
    load_single_role_conversations,
    load_single_trait_conversations,
    list_available_roles,
    list_available_traits,
    save_dataset,
)


def create_role_dataset_wrapper(args):
    """Wrapper for parallel execution."""
    model, role, output_dir, min_score = args
    try:
        dataset = load_single_role_conversations(
            model_name=model,
            role_name=role,
            min_score=min_score,
        )

        output_name = f"{model}__role__{role}"
        if min_score:
            output_name += f"__min{min_score}"

        save_dataset(dataset, output_dir, output_name)
        return (model, role, len(dataset), None)
    except Exception as e:
        return (model, role, 0, str(e))


def create_trait_dataset_wrapper(args):
    """Wrapper for parallel execution."""
    model, trait, output_dir, min_score, label_filter = args
    try:
        dataset = load_single_trait_conversations(
            model_name=model,
            trait_name=trait,
            min_score=min_score,
            label_filter=label_filter,
        )

        output_name = f"{model}__trait__{trait}"
        if min_score:
            output_name += f"__min{min_score}"
        if label_filter:
            output_name += f"__{label_filter}"

        save_dataset(dataset, output_dir, output_name)
        return (model, trait, len(dataset), None)
    except Exception as e:
        return (model, trait, 0, str(e))


def generate_all_roles(
    models=["gemma-2-27b", "llama-3.3-70b", "qwen-3-32b"],
    output_dir=Path("/workspace/datasets/processed/persona"),
    min_score=None,
    max_workers=8,
):
    """Generate datasets for all (model, role) pairs."""

    # Collect all tasks
    tasks = []
    for model in models:
        roles = list_available_roles(model)
        for role in roles:
            tasks.append((model, role, output_dir, min_score))

    print(f"\nGenerating {len(tasks)} role datasets across {len(models)} models...")
    print(f"Workers: {max_workers}")

    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(create_role_dataset_wrapper, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Role datasets") as pbar:
            for future in as_completed(futures):
                model, role, count, error = future.result()
                if error:
                    errors.append((model, role, error))
                else:
                    results.append((model, role, count))
                pbar.update(1)

    # Summary
    print(f"\n✓ Created {len(results)} role datasets")
    print(f"  Total conversations: {sum(r[2] for r in results):,}")
    if errors:
        print(f"\n✗ {len(errors)} errors:")
        for model, role, error in errors[:5]:
            print(f"  {model}/{role}: {error}")

    return results, errors


def generate_all_traits(
    models=["gemma-2-27b", "llama-3.3-70b", "qwen-3-32b"],
    output_dir=Path("/workspace/datasets/processed/persona"),
    min_score=None,
    label_filter=None,
    max_workers=8,
):
    """Generate datasets for all (model, trait) pairs."""

    # Collect all tasks
    tasks = []
    for model in models:
        traits = list_available_traits(model)
        for trait in traits:
            tasks.append((model, trait, output_dir, min_score, label_filter))

    print(f"\nGenerating {len(tasks)} trait datasets across {len(models)} models...")
    print(f"Workers: {max_workers}")

    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(create_trait_dataset_wrapper, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Trait datasets") as pbar:
            for future in as_completed(futures):
                model, trait, count, error = future.result()
                if error:
                    errors.append((model, trait, error))
                else:
                    results.append((model, trait, count))
                pbar.update(1)

    # Summary
    print(f"\n✓ Created {len(results)} trait datasets")
    print(f"  Total conversations: {sum(r[2] for r in results):,}")
    if errors:
        print(f"\n✗ {len(errors)} errors:")
        for model, trait, error in errors[:5]:
            print(f"  {model}/{trait}: {error}")

    return results, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate all persona datasets")
    parser.add_argument(
        "--type",
        choices=["roles", "traits", "both"],
        default="both",
        help="Generate role, trait, or both dataset types"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma-2-27b", "llama-3.3-70b", "qwen-3-32b"],
        help="Models to process"
    )
    parser.add_argument(
        "--min-score",
        type=int,
        help="Minimum extract score (roles: 0-3, traits: 0-100)"
    )
    parser.add_argument(
        "--label-filter",
        choices=["pos", "neg"],
        help="For traits: only pos or neg prompted conversations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/datasets/processed/persona"),
        help="Output directory"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    if args.type in ["roles", "both"]:
        generate_all_roles(
            models=args.models,
            output_dir=args.output_dir,
            min_score=args.min_score,
            max_workers=args.workers,
        )

    if args.type in ["traits", "both"]:
        generate_all_traits(
            models=args.models,
            output_dir=args.output_dir,
            min_score=args.min_score,
            label_filter=args.label_filter,
            max_workers=args.workers,
        )

    print("\n✓ All datasets generated!")
