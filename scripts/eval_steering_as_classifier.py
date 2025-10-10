"""Evaluate trained steering vectors as classifiers on the original dataset.

For roles: Binary logistic regression for "judge score == 3" vs other scores.
For traits: Ridge regression vs 0-100 score with MSE.

The steering vector is used as the weight vector, with only scale & offset optimized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatspace.steering import extract_layer_hidden_states


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
            print(f"  Loaded {len([r for r in records if r['label']=='pos'])} positive examples (score=3)")
        except FileNotFoundError:
            print(f"Warning: Could not load positive dataset {dataset_name}")
            return []

        # Load negative examples from default dataset (baseline responses)
        # Default datasets have no role prompting and were not judged, use all as negatives
        # Extract model prefix (e.g., "qwen-3-32b" from "qwen-3-32b__role__absurdist")
        if "__role__" in dataset_name:
            model_prefix = dataset_name.split("__role__")[0]
            # Try both 0_default and 1_default
            for default_suffix in ["0_default", "1_default"]:
                default_name = f"{model_prefix}__role__{default_suffix}"
                try:
                    neg_ds = _load_single_dataset(default_name)
                    neg_before = len([r for r in records if r['label']=='neg'])
                    for row in neg_ds:
                        # Default datasets don't have extract_score, use all examples as baseline
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
                        print(f"  Loaded {neg_count} negative examples (baseline) from {default_name}")
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
def eval_role_classifier(
    steering_vector: np.ndarray,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Evaluate steering vector as binary classifier for pos vs neg examples.

    We only optimize scale & offset:
        logit = scale * dot(steering_vector, hidden_state) + offset
    """
    # Labels should already be 0/1 for neg/pos

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


def eval_trait_regressor(
    steering_vector: np.ndarray,
    hidden_states: np.ndarray,
    scores: np.ndarray,
    test_size: float = 0.2,
    alpha: float = 1.0,
    seed: int = 42,
) -> dict:
    """Evaluate steering vector as ridge regressor for 0-100 score.

    We only optimize scale & offset:
        score = scale * dot(steering_vector, hidden_state) + offset
    """
    # Project hidden states onto steering vector
    projections = hidden_states @ steering_vector  # [n_samples]

    # Reshape for sklearn
    X = projections.reshape(-1, 1)
    y = scores

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Fit ridge regression (only optimizes scale & offset)
    reg = Ridge(alpha=alpha, random_state=seed)
    reg.fit(X_train, y_train)

    # Evaluate
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    metrics = {
        "train_mse": float(mean_squared_error(y_train, y_pred_train)),
        "test_mse": float(mean_squared_error(y_test, y_pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "scale": float(reg.coef_[0]),
        "offset": float(reg.intercept_),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "ridge_alpha": alpha,
    }

    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., qwen-3-32b__role__accountant)")
    parser.add_argument("--steering-vector-path", type=Path, required=True, help="Path to steering_vector.pt")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/processed/persona"))
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--target-layer", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--truncate-after-target",
        action="store_true",
        help="Stop the forward pass after the target layer when extracting hidden states",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path for metrics")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regression alpha for traits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    # Determine dataset type
    if "__role__" in args.dataset:
        dataset_type = "role"
    elif "__trait__" in args.dataset:
        dataset_type = "trait"
    else:
        raise ValueError(f"Cannot determine dataset type from name: {args.dataset}")

    print(f"Loading steering vector from {args.steering_vector_path}")
    state = torch.load(args.steering_vector_path, map_location="cpu")
    steering_vector = state["steering_vector"].float().numpy()  # [hidden_dim]

    print(f"Loading dataset: {args.dataset} (type: {dataset_type})")
    records = load_dataset_with_all_scores(args.dataset, args.data_root, dataset_type)
    print(f"Loaded {len(records)} records")

    # Extract labels and scores
    labels_str = np.array([rec["label"] for rec in records])
    scores = np.array([rec["score"] for rec in records])

    # Convert labels to binary: pos=1, neg=0
    labels = (labels_str == "pos").astype(int)

    print(f"Label distribution: pos={labels.sum()}, neg={(1-labels).sum()}")
    print(f"Score distribution: min={scores.min()}, max={scores.max()}, mean={scores.mean():.2f}")

    if dataset_type == "role":
        print(f"  Pos scores - mean={scores[labels==1].mean():.2f}, std={scores[labels==1].std():.2f}")
        print(f"  Neg scores - mean={scores[labels==0].mean():.2f}, std={scores[labels==0].std():.2f}")

    # Load model and extract hidden states
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

    print("Extracting hidden states...")
    hidden_states = extract_layer_hidden_states(
        records,
        model=model,
        tokenizer=tokenizer,
        target_layer=args.target_layer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        truncate_after=args.truncate_after_target,
        add_generation_prompt=False,
        use_tqdm=True,
    )
    print(f"Extracted hidden states: {hidden_states.shape}")

    # Free up memory
    del model
    torch.cuda.empty_cache()

    # Evaluate as classifier/regressor
    print(f"\nEvaluating as {dataset_type} classifier/regressor...")

    if dataset_type == "role":
        metrics = eval_role_classifier(
            steering_vector,
            hidden_states,
            labels,
            test_size=args.test_size,
            seed=args.seed,
        )
        print("\nRole Classifier Metrics:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test F1: {metrics['test_f1']:.4f}")
        print(f"  Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        print(f"  Scale: {metrics['scale']:.4f}, Offset: {metrics['offset']:.4f}")
        print(f"  Train: {metrics['n_positive_train']}/{metrics['n_train']} positive")
        print(f"  Test: {metrics['n_positive_test']}/{metrics['n_test']} positive")
    else:  # trait
        metrics = eval_trait_regressor(
            steering_vector,
            hidden_states,
            scores,
            test_size=args.test_size,
            alpha=args.ridge_alpha,
            seed=args.seed,
        )
        print("\nTrait Regressor Metrics:")
        print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test MSE: {metrics['test_mse']:.4f}")
        print(f"  Scale: {metrics['scale']:.4f}, Offset: {metrics['offset']:.4f}")
        print(f"  Train: {metrics['n_train']}, Test: {metrics['n_test']}")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "dataset": args.dataset,
            "dataset_type": dataset_type,
            "steering_vector_path": str(args.steering_vector_path),
            "metrics": metrics,
            "config": {
                "model": args.model,
                "target_layer": args.target_layer,
                "test_size": args.test_size,
                "seed": args.seed,
            },
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
