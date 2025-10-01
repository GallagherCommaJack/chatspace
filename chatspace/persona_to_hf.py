"""
Convert persona-subspace data to HuggingFace datasets.

Converts role and trait conversation data from persona-subspace project
into HuggingFace dataset format for fine-tuning and analysis.
"""

import json
from pathlib import Path
from typing import Literal, Optional
import datasets
from datasets import Dataset, DatasetDict


def load_single_role_conversations(
    model_name: str,
    role_name: str,
    persona_data_dir: Path = Path("/workspace/persona-data"),
    min_score: Optional[int] = None,
) -> Dataset:
    """
    Load conversations for a single role into HuggingFace dataset.

    Args:
        model_name: Model name (gemma-2-27b, llama-3.3-70b, qwen-3-32b)
        role_name: Specific role name (e.g., 'accountant', 'activist')
        persona_data_dir: Root directory containing persona data
        min_score: Minimum extract_score to include (0-3, None for all)

    Returns:
        HuggingFace Dataset with role conversations
    """
    responses_dir = persona_data_dir / model_name / "roles_240" / "responses"
    scores_dir = persona_data_dir / model_name / "roles_240" / "extract_scores"

    # Load extract scores if filtering requested
    scores = None
    score_file = scores_dir / f"{role_name}.json"
    if score_file.exists():
        with open(score_file) as f:
            scores = json.load(f)

    # Collect all conversations for this role
    rows = []
    response_file = responses_dir / f"{role_name}.jsonl"

    if not response_file.exists():
        raise FileNotFoundError(f"Role '{role_name}' not found in {model_name}")

    # Load conversations
    with open(response_file) as f:
        for line in f:
            conv = json.loads(line)

            # Filter by score if needed
            if min_score is not None and scores is not None:
                key = f"{conv['label']}_p{conv['prompt_index']}_q{conv['question_index']}"
                score = scores.get(key, 0)
                if score < min_score:
                    continue

            # Extract conversation
            messages = conv["conversation"]

            rows.append({
                "model": model_name,
                "role": role_name,
                "system_prompt": conv["system_prompt"],
                "label": conv["label"],
                "prompt_index": conv["prompt_index"],
                "question_index": conv["question_index"],
                "question": conv["question"],
                "messages": messages,
                "extract_score": scores.get(
                    f"{conv['label']}_p{conv['prompt_index']}_q{conv['question_index']}",
                    None
                ) if scores else None,
            })

    return Dataset.from_list(rows)


def load_single_trait_conversations(
    model_name: str,
    trait_name: str,
    persona_data_dir: Path = Path("/workspace/persona-data"),
    min_score: Optional[int] = None,
    label_filter: Optional[Literal["pos", "neg"]] = None,
) -> Dataset:
    """
    Load conversations for a single trait into HuggingFace dataset.

    Args:
        model_name: Model name (gemma-2-27b, llama-3.3-70b, qwen-3-32b)
        trait_name: Specific trait name (e.g., 'absolutist', 'analytical')
        persona_data_dir: Root directory containing persona data
        min_score: Minimum extract_score to include (0-100, None for all)
        label_filter: Only include 'pos' or 'neg' prompted conversations

    Returns:
        HuggingFace Dataset with trait conversations
    """
    responses_dir = persona_data_dir / model_name / "traits_240" / "responses"
    scores_dir = persona_data_dir / model_name / "traits_240" / "extract_scores"

    # Load extract scores if filtering requested
    scores = None
    score_file = scores_dir / f"{trait_name}.json"
    if score_file.exists():
        with open(score_file) as f:
            scores = json.load(f)

    # Collect all conversations for this trait
    rows = []
    response_file = responses_dir / f"{trait_name}.jsonl"

    if not response_file.exists():
        raise FileNotFoundError(f"Trait '{trait_name}' not found in {model_name}")

    # Load conversations
    with open(response_file) as f:
        for line in f:
            conv = json.loads(line)

            # Filter by label if requested
            if label_filter is not None and conv["label"] != label_filter:
                continue

            # Filter by score if needed
            if min_score is not None and scores is not None:
                key = f"{conv['label']}_p{conv['prompt_index']}_q{conv['question_index']}"
                score = scores.get(key, 0)
                if score < min_score:
                    continue

            # Extract conversation
            messages = conv["conversation"]

            rows.append({
                "model": model_name,
                "trait": trait_name,
                "system_prompt": conv["system_prompt"],
                "label": conv["label"],
                "prompt_index": conv["prompt_index"],
                "question_index": conv["question_index"],
                "question": conv["question"],
                "messages": messages,
                "extract_score": scores.get(
                    f"{conv['label']}_p{conv['prompt_index']}_q{conv['question_index']}",
                    None
                ) if scores else None,
            })

    return Dataset.from_list(rows)


def list_available_roles(
    model_name: str,
    persona_data_dir: Path = Path("/workspace/persona-data"),
) -> list[str]:
    """List all available roles for a model."""
    responses_dir = persona_data_dir / model_name / "roles_240" / "responses"
    return sorted([f.stem for f in responses_dir.glob("*.jsonl")])


def list_available_traits(
    model_name: str,
    persona_data_dir: Path = Path("/workspace/persona-data"),
) -> list[str]:
    """List all available traits for a model."""
    responses_dir = persona_data_dir / model_name / "traits_240" / "responses"
    return sorted([f.stem for f in responses_dir.glob("*.jsonl")])


def save_dataset(
    dataset: Dataset,
    output_dir: Path,
    name: str,
):
    """
    Save dataset to disk.

    Args:
        dataset: Dataset to save
        output_dir: Directory to save dataset files
        name: Dataset name for directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as HF dataset
    dataset_dict = DatasetDict({"train": dataset})
    save_path = output_dir / name
    dataset_dict.save_to_disk(str(save_path))
    print(f"Saved dataset to {save_path}")

    # Also save as parquet for easier inspection
    parquet_file = output_dir / f"{name}.parquet"
    dataset.to_parquet(str(parquet_file))
    print(f"Saved parquet to {parquet_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert persona data to HuggingFace datasets")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List available roles or traits")
    list_parser.add_argument("--type", choices=["roles", "traits"], required=True)
    list_parser.add_argument("--model", default="gemma-2-27b", help="Model name")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create dataset for single persona/trait")
    create_parser.add_argument(
        "--type",
        choices=["role", "trait"],
        required=True,
        help="Dataset type"
    )
    create_parser.add_argument(
        "--model",
        default="gemma-2-27b",
        help="Model name"
    )
    create_parser.add_argument(
        "--name",
        required=True,
        help="Role or trait name (e.g., 'accountant', 'absolutist')"
    )
    create_parser.add_argument(
        "--min-score",
        type=int,
        help="Minimum extract score (roles: 0-3, traits: 0-100)"
    )
    create_parser.add_argument(
        "--label-filter",
        choices=["pos", "neg"],
        help="For traits: only include pos or neg prompted conversations"
    )
    create_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/datasets/processed/persona"),
        help="Output directory"
    )

    args = parser.parse_args()

    if args.command == "list":
        if args.type == "roles":
            items = list_available_roles(args.model)
            print(f"\n{len(items)} roles available for {args.model}:")
        else:
            items = list_available_traits(args.model)
            print(f"\n{len(items)} traits available for {args.model}:")

        for item in items:
            print(f"  {item}")

    elif args.command == "create":
        # Create dataset
        if args.type == "role":
            dataset = load_single_role_conversations(
                model_name=args.model,
                role_name=args.name,
                min_score=args.min_score,
            )
            output_name = f"{args.model}__role__{args.name}"
            if args.min_score:
                output_name += f"__min{args.min_score}"
        else:
            dataset = load_single_trait_conversations(
                model_name=args.model,
                trait_name=args.name,
                min_score=args.min_score,
                label_filter=args.label_filter,
            )
            output_name = f"{args.model}__trait__{args.name}"
            if args.min_score:
                output_name += f"__min{args.min_score}"
            if args.label_filter:
                output_name += f"__{args.label_filter}"

        # Print stats
        print(f"\nDataset: {output_name}")
        print(f"Model: {args.model}")
        print(f"Type: {args.type}")
        print(f"Name: {args.name}")
        print(f"Total conversations: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")

        if len(dataset) > 0:
            print(f"\nSample (first message only):")
            sample = dataset[0]
            print(f"  System: {sample['system_prompt'][:100]}...")
            print(f"  Question: {sample['question']}")
            print(f"  Extract score: {sample['extract_score']}")
            print(f"  Messages: {len(sample['messages'])} turns")

            # Save
            save_dataset(dataset, args.output_dir, output_name)
        else:
            print("\nNo conversations match the filters!")
