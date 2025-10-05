"""Batch-train steering vectors for persona traits/roles with ≥100k tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

from chatspace.steering import train as train_module


DEFAULT_TRAITS_FILE = Path("/workspace/persona_traits_over_100k.txt")
DEFAULT_ROLES_FILE = Path("/workspace/persona_roles_over_100k.txt")
DEFAULT_DATA_ROOT = Path("/workspace/datasets/processed/persona")
DEFAULT_RUN_ROOT = Path("/workspace/steering_runs")


def _read_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _iter_datasets(
    names: Iterable[str],
    prefix: str,
    data_root: Path,
) -> list[str]:
    valid: list[str] = []
    for raw in names:
        dataset_name = f"{prefix}{raw}"
        dataset_path = data_root / dataset_name
        if dataset_path.exists():
            valid.append(dataset_name)
    return sorted(set(valid))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits-file", type=Path, default=DEFAULT_TRAITS_FILE)
    parser.add_argument("--roles-file", type=Path, default=DEFAULT_ROLES_FILE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--target-layer", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-tokens", type=int, default=100_000)
    parser.add_argument("--val-tokens", type=int, default=10_000)
    parser.add_argument("--trait-prefix", default="qwen-3-32b__trait__")
    parser.add_argument("--role-prefix", default="gemma-2-27b__role__")
    parser.add_argument("--num-workers", type=int, default=1, help="Processed sequentially; argument reserved")
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def _run_single(dataset: str, args: argparse.Namespace) -> None:
    output_dir = args.output_root / dataset
    if output_dir.exists() and args.skip_existing:
        print(f"Skipping {dataset} (exists)")
        return

    cmd = [
        "--datasets", dataset,
        "--output-dir", str(output_dir),
        "--model", args.model,
        "--target-layer", str(args.target_layer),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--num-epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--max-length", str(args.max_length),
        "--target-tokens", str(args.train_tokens),
        "--val-target-tokens", str(args.val_tokens),
        "--bf16",
        "--gradient-checkpointing",
        "--logging-steps", "1",
        "--lr-scheduler", "cosine",
        "--early-stop-patience", "2",
        "--compare-prompted",
    ]

    print("\n=== Training", dataset, "===")
    print("Output dir:", output_dir)
    if args.dry_run:
        print("Dry run (command not executed):", "chatspace.steering.train", " ".join(cmd))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_module.main(cmd)
    except ValueError as exc:
        print(f"Failed on {dataset}: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error on {dataset}: {exc}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    trait_names = _read_names(args.traits_file)
    role_names = _read_names(args.roles_file)
    datasets = _iter_datasets(trait_names, args.trait_prefix, args.data_root)
    datasets += _iter_datasets(role_names, args.role_prefix, args.data_root)
    datasets = sorted(set(datasets))

    print(f"Found {len(datasets)} datasets with available processed data")
    for name in datasets:
        _run_single(name, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
