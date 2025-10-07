"""Compare trained steering vectors to activation-averaged baselines."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_LOG = Path("/workspace/steering_runs/steering_sweep.log")
DEFAULT_RUN_ROOT = Path("/workspace/steering_runs")
PERSONA_ROOT = Path("/workspace/persona-data")

TARGET_LAYER = 22  # zero-based index


def _iter_log_runs(log_path: Path) -> Iterable[dict[str, object]]:
    current: Optional[dict[str, object]] = None
    for line in log_path.read_text().splitlines():
        if line.startswith("=== Training "):
            if current:
                yield current
            name = line.split("Training ", 1)[1].split(" ===", 1)[0]
            current = {"dataset": name}
        elif line.startswith("Validation metrics:") and current is not None:
            metrics = ast.literal_eval(line.split(":", 1)[1].strip())
            current["eval_metrics"] = metrics
        elif line.startswith("Prompted baseline metrics:") and current is not None:
            metrics = ast.literal_eval(line.split(":", 1)[1].strip())
            current["baseline_metrics"] = metrics
    if current:
        yield current


def _load_trained_vector(run_dir: Path) -> Optional[torch.Tensor]:
    vec_path = run_dir / "steering_vector.pt"
    if not vec_path.exists():
        return None
    data = torch.load(vec_path, map_location="cpu")
    vec = data.get("steering_vector") if isinstance(data, dict) else data
    return vec.float()


def _load_activation_vector(dataset: str) -> Optional[torch.Tensor]:
    if "__trait__" in dataset:
        model_prefix = dataset.split("__trait__", 1)[0]
        trait = dataset.split("__trait__", 1)[1]
        vec_file = PERSONA_ROOT / f"{model_prefix}/traits_240/vectors/{trait}.pt"
        if not vec_file.exists():
            return None
        data: Dict[str, torch.Tensor] = torch.load(vec_file, map_location="cpu")
        key = "pos_70" if "pos_70" in data else next(iter(data))
        vec = data[key][TARGET_LAYER]
        return vec.float()
    if "__role__" in dataset:
        model_prefix = "qwen-3-32b"  # use Qwen activation vectors for roles
        role = dataset.split("__role__", 1)[1]
        vec_file = PERSONA_ROOT / f"{model_prefix}/roles_240/vectors/{role}.pt"
        if not vec_file.exists():
            return None
        data: Dict[str, torch.Tensor] = torch.load(vec_file, map_location="cpu")
        key = "pos_3" if "pos_3" in data else next(iter(data))
        vec = data[key][TARGET_LAYER]
        return vec.float()
    return None


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = F.normalize(a.view(-1), dim=0)
    b_n = F.normalize(b.view(-1), dim=0)
    return float((a_n * b_n).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=Path("/workspace/steering_runs/steering_vector_comparison.parquet"))
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(args.log)

    records = []
    for entry in _iter_log_runs(args.log):
        dataset = entry["dataset"]
        run_dir = args.runs / dataset
        trained_vec = _load_trained_vector(run_dir)
        activation_vec = _load_activation_vector(dataset)

        cos_sim = None
        norm_trained = None
        norm_activation = None

        if trained_vec is not None:
            norm_trained = float(trained_vec.norm().item())
        if activation_vec is not None:
            norm_activation = float(activation_vec.norm().item())

        if trained_vec is not None and activation_vec is not None and trained_vec.shape == activation_vec.shape:
            cos_sim = cosine_similarity(trained_vec, activation_vec)

        eval_metrics = entry.get("eval_metrics") or {}
        baseline_metrics = entry.get("baseline_metrics") or {}

        eval_loss = eval_metrics.get("eval_loss")
        eval_ppl = eval_metrics.get("eval_ppl")
        if eval_loss is None:
            manual = next((h for h in json.load(open(run_dir / "trainer_state.json"))["log_history"] if "eval_loss_manual" in h), None)
            if manual:
                eval_loss = manual["eval_loss_manual"]
                eval_ppl = math.exp(eval_loss)

        records.append(
            {
                "dataset": dataset,
                "trained_vector_path": str(run_dir / "steering_vector.pt") if trained_vec is not None else None,
                "activation_source": str(activation_vec.shape if activation_vec is not None else None),
                "cos_similarity": cos_sim,
                "trained_norm": norm_trained,
                "activation_norm": norm_activation,
                "trained_eval_loss": eval_loss,
                "trained_eval_ppl": eval_ppl,
                "baseline_eval_loss": baseline_metrics.get("eval_loss"),
                "baseline_eval_ppl": baseline_metrics.get("eval_ppl"),
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_parquet(args.output)
    print(f"Wrote {len(df)} rows to {args.output}")

    summary = df.groupby(df["dataset"].str.contains("__trait__"))[["cos_similarity", "trained_eval_ppl", "baseline_eval_ppl"]].mean()
    print(summary)


if __name__ == "__main__":
    main()
