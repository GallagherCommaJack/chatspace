"""Comprehensive evaluation: trained steering vectors + activation vectors (where available).

Optimized to extract hidden states once per dataset, then evaluate multiple vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatspace.steering import extract_layer_hidden_states, load_activation_vector, runs as run_utils


def load_dataset_with_all_scores(
    dataset_name: str,
    data_root: Path,
    dataset_type: Literal["role", "trait"],
) -> list[dict]:
    """Load dataset including all examples (pos and neg for proper evaluation)."""

    def _load_single_dataset(name: str):
        """Helper to load a single dataset."""
        dataset_path = data_root / name
        if not dataset_path.exists():
            parquet_path = dataset_path.with_suffix(".parquet")
            if parquet_path.exists():
                from datasets import load_dataset
                return load_dataset("parquet", data_files=str(parquet_path))["train"]
            else:
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        else:
            return load_from_disk(str(dataset_path))["train"]

    records = []

    # For roles: load both positive role dataset AND negative default dataset
    if dataset_type == "role":
        # Load positive examples (score==3) from role dataset
        try:
            pos_ds = _load_single_dataset(dataset_name)
            for row in pos_ds:
                score = row.get("extract_score")
                if score is None:
                    continue
                try:
                    score = int(score)
                except (TypeError, ValueError):
                    continue

                # Only keep score==3 (successful role exhibition)
                if score != 3:
                    continue

                messages = [m for m in row["messages"] if m.get("role") != "system"]
                if not messages:
                    continue

                records.append({
                    "messages": messages,
                    "score": score,
                    "label": "pos",
                })
        except FileNotFoundError:
            print(f"Warning: Could not load positive dataset {dataset_name}")
            return []

        # Load negative examples from default dataset (baseline responses)
        if "__role__" in dataset_name:
            model_prefix = dataset_name.split("__role__")[0]
            # Try both 0_default and 1_default
            for default_suffix in ["0_default", "1_default"]:
                default_name = f"{model_prefix}__role__{default_suffix}"
                try:
                    neg_ds = _load_single_dataset(default_name)
                    neg_before = len([r for r in records if r['label']=='neg'])
                    for row in neg_ds:
                        messages = [m for m in row["messages"] if m.get("role") != "system"]
                        if not messages:
                            continue

                        records.append({
                            "messages": messages,
                            "score": 1,  # Assign score=1 for baseline (no role prompting)
                            "label": "neg",
                        })
                    neg_count = len([r for r in records if r['label']=='neg']) - neg_before
                    if neg_count > 0:
                        print(f"  Loaded {len([r for r in records if r['label']=='pos'])} pos, {neg_count} neg examples")
                        break  # Found valid negatives
                except FileNotFoundError:
                    continue

    # For traits: already has pos/neg in the same dataset
    elif dataset_type == "trait":
        ds = _load_single_dataset(dataset_name)
        for row in ds:
            score = row.get("extract_score")
            if score is None:
                continue
            try:
                score = int(score)
            except (TypeError, ValueError):
                continue

            label = row.get("label")
            if label not in ["pos", "neg"]:
                continue

            messages = [m for m in row["messages"] if m.get("role") != "system"]
            if not messages:
                continue

            records.append({
                "messages": messages,
                "score": score,
                "label": label,
            })

    return records
def eval_vector_as_classifier(
    steering_vector: np.ndarray,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Evaluate steering vector as binary classifier for pos vs neg examples."""
    # Project hidden states onto steering vector
    projections = hidden_states @ steering_vector  # [n_samples]

    # Reshape for sklearn
    X = projections.reshape(-1, 1)
    y = labels

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Fit logistic regression (only optimizes scale & offset)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_prob_train = clf.predict_proba(X_train)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "train_f1": float(f1_score(y_train, y_pred_train)),
        "test_f1": float(f1_score(y_test, y_pred_test)),
        "train_roc_auc": float(roc_auc_score(y_train, y_prob_train)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob_test)),
        "scale": float(clf.coef_[0, 0]),
        "offset": float(clf.intercept_[0]),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_positive_train": int(y_train.sum()),
        "n_positive_test": int(y_test.sum()),
    }

    return metrics


def find_steering_vectors(
    steering_root: Path,
    dataset_type: Literal["role", "trait"] | None = None,
) -> list[tuple[str, Path]]:
    """Find all steering vectors in the given root directory."""
    index = run_utils.collect_run_dirs(steering_root)
    results: list[tuple[str, Path]] = []
    for dataset, run_dir in index.items():
        if dataset_type == "role" and "__role__" not in dataset:
            continue
        if dataset_type == "trait" and "__trait__" not in dataset:
            continue
        vec_path = run_dir / "steering_vector.pt"
        if vec_path.exists():
            results.append((dataset, vec_path))
    return sorted(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steering-root",
        type=Path,
        default=Path("/workspace/steering_runs_scheduler_prod_acc1"),
        help="Root directory containing trained steering vectors",
    )
    parser.add_argument(
        "--activation-root",
        type=Path,
        default=Path("/workspace/persona-data/qwen-3-32b/roles_240/vectors"),
        help="Directory containing activation vectors (optional)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/datasets/processed/persona"),
        help="Root directory containing persona datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/classifier_eval_comprehensive"),
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--type",
        choices=["role", "trait", "both"],
        default="role",
        help="Which dataset type to evaluate",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--target-layer", type=int, default=31)
    parser.add_argument("--activation-layer", type=int, default=22, help="Layer for activation vectors")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--truncate-after-target",
        action="store_true",
        help="Stop the forward pass after the target layer when extracting hidden states",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit number of datasets to process")
    parser.add_argument("--eval-activation-vectors", action="store_true", help="Also evaluate activation vectors")
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Find steering vectors
    dataset_type_filter = None if args.type == "both" else args.type
    datasets = find_steering_vectors(args.steering_root, dataset_type_filter)

    if not datasets:
        print(f"No steering vectors found in {args.steering_root}")
        return

    print(f"Found {len(datasets)} steering vectors to evaluate")

    if args.limit:
        datasets = datasets[: args.limit]
        print(f"Limited to {len(datasets)} datasets")

    # Load model once
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=False,
    )
    model.eval()

    # Process each dataset
    summary = []
    for i, (dataset_name, steering_vec_path) in enumerate(datasets, 1):
        output_path = args.output_root / f"{dataset_name}.json"

        if output_path.exists() and args.skip_existing:
            print(f"[{i}/{len(datasets)}] Skipping {dataset_name} (exists)")
            continue

        print(f"\n[{i}/{len(datasets)}] Evaluating {dataset_name}")

        if args.dry_run:
            continue

        # Determine dataset type
        if "__role__" in dataset_name:
            dataset_type = "role"
        elif "__trait__" in dataset_name:
            dataset_type = "trait"
        else:
            print(f"  ERROR: Cannot determine dataset type")
            continue

        # Check if dataset exists
        dataset_path = args.data_root / dataset_name
        if not dataset_path.exists():
            parquet_path = dataset_path.with_suffix(".parquet")
            if not parquet_path.exists():
                print(f"  ERROR: Dataset not found at {dataset_path}")
                summary.append({
                    "dataset": dataset_name,
                    "status": "dataset_not_found",
                })
                continue

        try:
            # Load dataset
            print(f"  Loading dataset...")
            records = load_dataset_with_all_scores(dataset_name, args.data_root, dataset_type)
            if not records:
                print(f"  ERROR: No valid records loaded")
                summary.append({"dataset": dataset_name, "status": "no_records"})
                continue

            labels_str = np.array([rec["label"] for rec in records])
            labels = (labels_str == "pos").astype(int)

            # Extract hidden states ONCE
            print(f"  Extracting hidden states at layer {args.target_layer}...")
            hidden_states = extract_layer_hidden_states(
                records,
                model=model,
                tokenizer=tokenizer,
                target_layer=args.target_layer,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                truncate_after=args.truncate_after_target,
                add_generation_prompt=False,
                use_tqdm=True,
            )

            # Evaluate trained steering vector
            print(f"  Loading steering vector: {steering_vec_path}")
            state = torch.load(steering_vec_path, map_location="cpu")
            steering_vector = state["steering_vector"].float().numpy()

            print(f"  Evaluating trained steering vector...")
            steering_metrics = eval_vector_as_classifier(
                steering_vector,
                hidden_states,
                labels,
                test_size=args.test_size,
                seed=args.seed,
            )

            result = {
                "dataset": dataset_name,
                "dataset_type": dataset_type,
                "n_records": len(records),
                "n_positive": int(labels.sum()),
                "steering_vector": {
                    "path": str(steering_vec_path),
                    "metrics": steering_metrics,
                },
            }

            # Optionally evaluate activation vector
            if args.eval_activation_vectors:
                activation_vec = None
                activation_vec_path = None

                try:
                    print(f"  Loading activation vector via library...")
                    # Use library function with production defaults:
                    # - Traits: pos_neg_50 (contrast vector)
                    # - Roles: pos_3 - default_1 (difference vector)
                    activation_vec_tensor = load_activation_vector(
                        dataset_name,
                        persona_root=args.activation_root.parent.parent,  # Go up from vectors/ to persona-data/
                        target_layer=args.target_layer,
                        role_contrast_default=True,  # Always compute role differences
                    )

                    if activation_vec_tensor is not None:
                        activation_vec = activation_vec_tensor.numpy()
                        # Determine path for logging
                        if dataset_type == "trait":
                            _, trait_name = dataset_name.split("__trait__", 1)
                            activation_vec_path = args.activation_root / f"{trait_name}.pt"
                        elif dataset_type == "role":
                            _, role_name = dataset_name.split("__role__", 1)
                            activation_vec_path = args.activation_root / f"{role_name}.pt"
                except Exception as exc:
                    print(f"  Warning: Failed to load activation vector: {exc}")

                # Evaluate activation vector if loaded successfully
                if activation_vec is not None:
                    try:
                        print(f"  Evaluating activation vector...")
                        activation_metrics = eval_vector_as_classifier(
                            activation_vec,
                            hidden_states,
                            labels,
                            test_size=args.test_size,
                            seed=args.seed,
                        )

                        result["activation_vector"] = {
                            "path": str(activation_vec_path),
                            "layer": args.target_layer,
                            "metrics": activation_metrics,
                        }
                    except Exception as exc:
                        print(f"  Warning: Failed to evaluate activation vector: {exc}")
                        result["activation_vector"] = {
                            "path": str(activation_vec_path),
                            "error": str(exc),
                        }
                else:
                    print(f"  No activation vector found for {dataset_name}")

            # Save result
            output_path.write_text(json.dumps(result, indent=2))
            print(f"  SUCCESS - Steering test ROC-AUC: {steering_metrics['test_roc_auc']:.4f}")

            summary.append({
                "dataset": dataset_name,
                "status": "success",
                "test_roc_auc": steering_metrics['test_roc_auc'],
            })

        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()
            summary.append({
                "dataset": dataset_name,
                "status": "error",
                "error": str(exc),
            })

    # Save summary
    summary_path = args.output_root / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n\nSummary saved to {summary_path}")

    # Print summary stats
    status_counts = {}
    for item in summary:
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nSummary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    if "success" in status_counts:
        success_items = [item for item in summary if item["status"] == "success"]
        aucs = [item["test_roc_auc"] for item in success_items if "test_roc_auc" in item]
        if aucs:
            print(f"\nMean test ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
