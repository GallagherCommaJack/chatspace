#!/usr/bin/env python3
"""Distribute persona rollout generation across GPUs using simple-gpu-scheduler."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatspace.constants import (
    PERSONA_ROLES_FILE,
    PERSONA_TRAITS_FILE,
    STEERING_ROLLOUT_ROOT,
    STEERING_RUN_ROOT,
)
from chatspace.persona import resolve_persona_datasets  # noqa: E402
from chatspace.utils import scheduling  # noqa: E402

DEFAULT_TRAITS_FILE = PERSONA_TRAITS_FILE
DEFAULT_ROLES_FILE = PERSONA_ROLES_FILE
DEFAULT_RUN_ROOT = STEERING_RUN_ROOT
DEFAULT_OUTPUT_ROOT = STEERING_ROLLOUT_ROOT


def build_worker_commands(
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
            "python",
            "scripts/generate_behavior_rollouts.py",
            "--run-root",
            str(args.run_root),
            "--output-root",
            str(args.output_root),
            "--model",
            args.model,
            "--target-layer",
            str(args.target_layer),
            "--dataset-stride",
            str(worker_count),
            "--dataset-offset",
            str(plan.index),
            "--datasets",
            *plan.datasets,
            "--steering-no-system",
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        cmd.extend(extra_args)
        commands.append(cmd)
    return commands


def launch_scheduler(commands: Sequence[Sequence[str]], gpus: Sequence[int], dry_run: bool) -> None:
    if dry_run:
        for cmd in commands:
            print(shlex.join(cmd))
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--target-layer", type=int, default=22)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--exclude-gpu", type=int, action="append", default=None)
    parser.add_argument("--avoid-gpu0", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments forwarded to generate_behavior_rollouts.py")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    datasets = resolve_persona_datasets(
        explicit=args.datasets,
        traits_file=args.traits_file if args.include_traits else None,
        roles_file=args.roles_file if args.include_roles else None,
        trait_prefix=args.trait_prefix,
        role_prefix=args.role_prefix,
        include_traits=args.include_traits,
        include_roles=args.include_roles,
    )
    if not datasets:
        print("No datasets to process.")
        return

    exclude = set(args.exclude_gpu or [])
    if args.avoid_gpu0:
        exclude.add(0)

    gpus = scheduling.detect_available_gpus(exclude=exclude, limit=args.num_gpus)
    if not gpus:
        raise RuntimeError("No GPUs available")

    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    worker_count = len(gpus)
    plans = scheduling.assign_by_stride(datasets, worker_count)

    print(f"Planning {len(datasets)} dataset(s) across {worker_count} GPU worker(s): {', '.join(map(str, gpus))}")
    for plan in plans:
        print(f"  worker {plan.index} -> {len(plan.datasets)} dataset(s)")

    commands = build_worker_commands(plans, worker_count, args, extra_args)
    if not commands:
        print("No worker commands generated; check dataset selection and GPU availability.")
        return

    launch_scheduler(commands, gpus, args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])
