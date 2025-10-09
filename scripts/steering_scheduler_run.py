#!/usr/bin/env python3
"""Plan and dispatch steering training jobs across GPUs using simple-gpu-scheduler."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatspace.steering import runs as run_utils
from chatspace.utils import scheduling  # noqa: E402
DEFAULT_TRAITS_FILE = Path("/workspace/persona_traits_over_100k.txt")
DEFAULT_ROLES_FILE = Path("/workspace/persona_roles_over_100k.txt")
DEFAULT_RUN_ROOT = Path("/workspace/steering_runs_scheduler")

VALID_PREFIX_MARKERS = ("__trait__", "__role__")


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries: list[str] = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(value)
    return entries


def build_dataset_list(
    traits_file: Path,
    roles_file: Path,
    trait_prefix: str,
    role_prefix: str,
    include_traits: bool,
    include_roles: bool,
    explicit: Sequence[str],
) -> list[str]:
    if explicit:
        return list(explicit)

    datasets: list[str] = []
    if include_traits:
        for name in _read_list(traits_file):
            datasets.append(f"{trait_prefix}{name}")
    if include_roles:
        for name in _read_list(roles_file):
            datasets.append(f"{role_prefix}{name}")
    return datasets


def filter_existing(datasets: Iterable[str], run_root: Path, skip_existing: bool) -> list[str]:
    if not skip_existing:
        return list(datasets)
    pending: list[str] = []
    for dataset in datasets:
        if not run_utils.has_successful_run(run_root, dataset):
            pending.append(dataset)
    return pending


def build_worker_commands(
    datasets: Sequence[str],
    plans: Sequence[scheduling.WorkerPlan],
    worker_count: int,
    args: argparse.Namespace,
    extra_args: Sequence[str],
) -> list[list[str]]:
    commands: list[list[str]] = []
    for plan in plans:
        cmd = [
            "uv",
            "run",
            "chatspace",
            "steering-train",
            "--",
            "--run-root",
            str(args.run_root),
            "--model",
            args.model,
            "--target-layer",
            str(args.target_layer),
            "--attempt",
            str(args.attempt),
            "--dataset-stride",
            str(worker_count),
            "--dataset-offset",
            str(plan.index),
            "--datasets",
            *datasets,
        ]
        if args.reuse_base_model:
            cmd.append("--reuse-base-model")
        if args.skip_existing:
            cmd.append("--skip-if-committed")
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(extra_args)
        commands.append(cmd)
    return commands


def launch_scheduler(commands: Sequence[Sequence[str]], gpus: Sequence[int], dry_run: bool) -> None:
    if dry_run:
        for line in commands:
            print(shlex.join(line))
        return

    command_lines = [shlex.join(cmd) for cmd in commands]
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "simple_gpu_scheduler.command_line",
            "--gpus",
            *map(str, gpus),
        ],
        input="\n".join(command_lines) + "\n",
        text=True,
        cwd=str(ROOT_DIR),
    )
    proc.check_returncode()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits-file", type=Path, default=DEFAULT_TRAITS_FILE)
    parser.add_argument("--roles-file", type=Path, default=DEFAULT_ROLES_FILE)
    parser.add_argument("--trait-prefix", default="qwen-3-32b__trait__")
    parser.add_argument("--role-prefix", default="qwen-3-32b__role__")
    parser.add_argument("--include-traits", dest="include_traits", action="store_true", default=True)
    parser.add_argument("--no-traits", dest="include_traits", action="store_false")
    parser.add_argument("--include-roles", dest="include_roles", action="store_true", default=False)
    parser.add_argument("--traits-only", dest="include_roles", action="store_false")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--target-layer", type=int, default=31)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-base-model", action="store_true", default=True)
    parser.add_argument("--no-reuse-base-model", dest="reuse_base_model", action="store_false")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--exclude-gpu", type=int, action="append", default=None)
    parser.add_argument("--avoid-gpu0", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to chatspace steering-train",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    datasets = build_dataset_list(
        traits_file=args.traits_file,
        roles_file=args.roles_file,
        trait_prefix=args.trait_prefix,
        role_prefix=args.role_prefix,
        include_traits=args.include_traits,
        include_roles=args.include_roles,
        explicit=args.datasets or [],
    )
    datasets = filter_existing(datasets, args.run_root, args.skip_existing)
    if not datasets:
        print("No datasets to process.")
        return

    if args.avoid_gpu0:
        exclude = set(args.exclude_gpu or []) | {0}
    else:
        exclude = set(args.exclude_gpu or [])

    available_gpus = scheduling.detect_available_gpus(exclude=exclude, limit=args.num_gpus)
    if not available_gpus:
        raise RuntimeError("No GPUs available")

    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    worker_count = len(available_gpus)
    plans = scheduling.assign_by_stride(datasets, worker_count)

    print(
        f"Planning {len(datasets)} dataset(s) across {worker_count} GPU worker(s): {', '.join(map(str, available_gpus))}"
    )
    for plan in plans:
        print(f"  worker {plan.index} -> {len(plan.datasets)} dataset(s)")

    commands = build_worker_commands(datasets, plans, worker_count, args, extra_args)
    if not commands:
        print("No worker commands generated; check dataset selection and GPU availability.")
        return
    launch_scheduler(commands, available_gpus, args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])
