"""Filesystem-coordinated steering training job runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

from .model import SteeringVectorConfig
from .train import add_training_arguments, build_model, prepare_tokenizer, run_training


DEFAULT_RUN_ROOT = Path("/workspace/steering_runs")
JOB_ONLY_FIELDS = {
    "run_root",
    "run_id",
    "attempt",
    "skip_if_committed",
    "skip_if_locked",
    "force",
    "dry_run",
    "reuse_base_model",
    "dataset_stride",
    "dataset_offset",
}


class RunLockedError(RuntimeError):
    """Raised when a run is already claimed by another process."""


class RunCommittedError(RuntimeError):
    """Raised when a run already has a commit artifact."""


class AttemptExistsError(RuntimeError):
    """Raised when an attempt directory already exists without --force."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_component(value: str) -> str:
    lowered = value.lower().strip()
    if not lowered:
        return "unnamed"
    safe = []
    for char in lowered:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        elif char == "/":
            safe.append("__")
        else:
            safe.append("-")
    sanitized = "".join(safe)
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    while sanitized.startswith("-"):
        sanitized = sanitized[1:]
    return sanitized or "unnamed"


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "item"):
        try:
            return _json_ready(obj.item())
        except Exception:  # pragma: no cover - defensive fallback
            return str(obj)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)


def _gpu_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "torch_cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return snapshot

    snapshot["device_count"] = torch.cuda.device_count()
    snapshot["devices"] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        snapshot["devices"].append(
            {
                "index": index,
                "name": props.name,
                "total_memory": props.total_memory,
            }
        )
    try:
        snapshot["current_device"] = torch.cuda.current_device()
    except Exception:
        snapshot["current_device"] = None
    return snapshot


@dataclass(frozen=True)
class RunLayout:
    model: str
    dataset: str
    run_id: str
    attempt: int
    base_dir: Path
    attempt_dir: Path
    lock_path: Path
    commit_path: Path
    error_path: Path
    config_path: Path
    run_manifest_path: Path


def build_run_layout(
    model_name: str,
    dataset_name: str,
    run_root: Path,
    run_id: str,
    attempt: int,
    explicit_output: Path | None,
) -> RunLayout:
    if explicit_output is not None:
        attempt_dir = explicit_output
        base_dir = explicit_output
    else:
        model_slug = _sanitize_component(model_name)
        dataset_slug = _sanitize_component(dataset_name)
        base_dir = run_root / model_slug / dataset_slug / run_id
        attempt_dir = base_dir / f"try{attempt:02d}"

    lock_name = f"run.try{attempt}.lock"
    commit_name = f"run.try{attempt}.commit.json"
    error_name = f"run.try{attempt}.error.json"

    return RunLayout(
        model=model_name,
        dataset=dataset_name,
        run_id=run_id,
        attempt=attempt,
        base_dir=base_dir,
        attempt_dir=attempt_dir,
        lock_path=attempt_dir / lock_name,
        commit_path=attempt_dir / commit_name,
        error_path=attempt_dir / error_name,
        config_path=attempt_dir / "train_args.json",
        run_manifest_path=base_dir / "run.json",
    )


class RunCoordinator:
    def __init__(self, layout: RunLayout) -> None:
        self.layout = layout

    def prepare(self, force: bool = False) -> None:
        if self.layout.commit_path.exists() and not force:
            raise RunCommittedError(
                f"Commit already exists at {self.layout.commit_path}. Choose a new --run-id or --attempt."
            )

        if self.layout.attempt_dir.exists() and not force:
            raise AttemptExistsError(
                f"Attempt directory already exists: {self.layout.attempt_dir}. Pick a new --attempt."
            )

        self.layout.attempt_dir.mkdir(parents=True, exist_ok=True)

        if not self.layout.run_manifest_path.exists():
            manifest = {
                "dataset": self.layout.dataset,
                "model": self.layout.model,
                "run_id": self.layout.run_id,
                "created_at": _iso_now(),
                "attempt_root": str(self.layout.base_dir),
            }
            _write_json(self.layout.run_manifest_path, manifest)

    def acquire_lock(self, payload: Dict[str, Any]) -> None:
        try:
            fd = os.open(self.layout.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:  # pragma: no cover - race condition coverage env dependent
            raise RunLockedError(f"Lock exists at {self.layout.lock_path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)

    def release_lock(self) -> None:
        if self.layout.lock_path.exists():
            self.layout.lock_path.unlink()

    def write_commit(self, payload: Dict[str, Any]) -> None:
        _write_json(self.layout.commit_path, payload)
        self.release_lock()

    def write_error(self, payload: Dict[str, Any]) -> None:
        _write_json(self.layout.error_path, payload)
        self.release_lock()


def add_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="Base directory for steering runs")
    parser.add_argument("--run-id", default=None, help="Override autogenerated run identifier (UTC timestamp)")
    parser.add_argument("--attempt", type=int, default=1, help="Attempt counter for retries (>=1)")
    parser.add_argument(
        "--skip-if-committed",
        action="store_true",
        help="Skip execution when a commit artifact already exists",
    )
    parser.add_argument(
        "--skip-if-locked",
        action="store_true",
        help="Skip execution when a lock file already exists",
    )
    parser.add_argument("--force", action="store_true", help="Override existing artifacts (unsafe)")
    parser.add_argument("--dry-run", action="store_true", help="Create run directories and lock, then exit")
    parser.add_argument(
        "--reuse-base-model",
        action="store_true",
        help="Load the base model/tokenizer once and reuse across datasets processed sequentially",
    )
    parser.add_argument(
        "--dataset-stride",
        type=int,
        default=None,
        help="Stride to select datasets from the provided list (combine with --dataset-offset)",
    )
    parser.add_argument(
        "--dataset-offset",
        type=int,
        default=None,
        help="Offset when applying --dataset-stride (defaults to 0)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    add_job_arguments(parser)
    return parser


def _prepare_train_args(args: argparse.Namespace, attempt_dir: Path, dataset_name: str) -> argparse.Namespace:
    job_free = {k: v for k, v in vars(args).items() if k not in JOB_ONLY_FIELDS}
    job_free["output_dir"] = attempt_dir
    job_free["datasets"] = [dataset_name]
    return argparse.Namespace(**job_free)


def _dump_training_config(path: Path, args: argparse.Namespace) -> None:
    payload = {k: _json_ready(v) for k, v in vars(args).items()}
    _write_json(path, payload)


def _validate_datasets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    datasets = list(args.datasets)
    if not datasets:
        parser.error("--datasets requires at least one dataset name")

    multi = len(datasets) > 1
    if multi and args.output_dir is not None:
        parser.error("--output-dir cannot be combined with multiple datasets; allow runner to manage directories")

    if multi and not args.reuse_base_model:
        parser.error("Multiple datasets require --reuse-base-model so the base model is reused in-process")

    return datasets


def _apply_dataset_stride(
    datasets: list[str],
    stride: int | None,
    offset: int,
) -> list[str]:
    if stride is None or stride <= 1:
        return datasets
    return [name for idx, name in enumerate(datasets) if idx % stride == offset]


def _load_shared_components(args: argparse.Namespace):
    tokenizer = prepare_tokenizer(args.model)
    cfg = SteeringVectorConfig(
        model_name=args.model,
        target_layer=args.target_layer,
        init_scale=args.init_scale,
    )
    model = build_model(cfg, args.device_map)
    return model, tokenizer


def _run_single_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    run_id: str,
    shared_model,
    shared_tokenizer,
) -> None:
    layout = build_run_layout(
        model_name=args.model,
        dataset_name=dataset_name,
        run_root=args.run_root,
        run_id=run_id,
        attempt=args.attempt,
        explicit_output=args.output_dir,
    )

    if layout.commit_path.exists() and args.skip_if_committed and not args.force:
        print(f"Commit exists at {layout.commit_path}; skipping {dataset_name}")
        return

    if layout.lock_path.exists() and args.skip_if_locked and not args.force:
        print(f"Lock exists at {layout.lock_path}; skipping {dataset_name}")
        return

    coordinator = RunCoordinator(layout)
    try:
        coordinator.prepare(force=args.force)
    except RunCommittedError as err:
        if args.skip_if_committed and not args.force:
            print(str(err))
            return
        raise

    if layout.lock_path.exists() and not args.force:
        raise RunLockedError(f"Lock exists at {layout.lock_path}; run with --force to override")

    gpu_info = _gpu_snapshot()
    started_at = _iso_now()
    host = socket.gethostname()
    lock_payload = {
        "dataset": dataset_name,
        "model": args.model,
        "run_id": run_id,
        "attempt": args.attempt,
        "pid": os.getpid(),
        "host": host,
        "started_at": started_at,
        "gpu": gpu_info,
    }

    coordinator.acquire_lock(lock_payload)

    if args.dry_run:
        print(f"Dry-run: created lock at {layout.lock_path}. Exiting without training.")
        coordinator.release_lock()
        return

    train_args = _prepare_train_args(args, layout.attempt_dir, dataset_name)
    _dump_training_config(layout.config_path, train_args)

    shared_model_arg = shared_model if args.reuse_base_model else None
    shared_tokenizer_arg = shared_tokenizer if args.reuse_base_model else None

    print(
        f"Starting steering training: dataset={dataset_name} model={args.model} "
        f"run_dir={layout.attempt_dir} gpu={gpu_info.get('cuda_visible_devices')}"
    )

    start_time = time.monotonic()
    try:
        summary = run_training(
            train_args,
            model=shared_model_arg,
            tokenizer=shared_tokenizer_arg,
            reset_vector=True,
        )
    except FileNotFoundError as exc:
        duration = time.monotonic() - start_time
        coordinator.write_error(
            {
                "status": "skipped_missing_dataset",
                "dataset": dataset_name,
                "model": args.model,
                "run_id": run_id,
                "attempt": args.attempt,
                "started_at": started_at,
                "ended_at": _iso_now(),
                "duration_seconds": duration,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "gpu": gpu_info,
            }
        )
        print(f"Skipping dataset {dataset_name}: missing dataset ({exc}).")
        return
    except ValueError as exc:
        duration = time.monotonic() - start_time
        coordinator.write_error(
            {
                "status": "skipped_insufficient_examples",
                "dataset": dataset_name,
                "model": args.model,
                "run_id": run_id,
                "attempt": args.attempt,
                "started_at": started_at,
                "ended_at": _iso_now(),
                "duration_seconds": duration,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "gpu": gpu_info,
            }
        )
        print(f"Skipping dataset {dataset_name}: unable to prepare examples ({exc}).")
        return
    except Exception as exc:  # pragma: no cover - requires failing training
        duration = time.monotonic() - start_time
        coordinator.write_error(
            {
                "status": "failed",
                "dataset": dataset_name,
                "model": args.model,
                "run_id": run_id,
                "attempt": args.attempt,
                "started_at": started_at,
                "ended_at": _iso_now(),
                "duration_seconds": duration,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "gpu": gpu_info,
            }
        )
        raise

    duration = time.monotonic() - start_time
    commit_payload = {
        "status": "success",
        "dataset": dataset_name,
        "model": args.model,
        "run_id": run_id,
        "attempt": args.attempt,
        "started_at": started_at,
        "ended_at": _iso_now(),
        "duration_seconds": duration,
        "summary": summary,
        "gpu": gpu_info,
        "host": host,
        "pid": os.getpid(),
        "lock_path": str(layout.lock_path),
    }
    coordinator.write_commit(commit_payload)
    print(f"Finished steering training. Commit written to {layout.commit_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.attempt < 1:
        parser.error("--attempt must be >= 1")

    if args.dataset_stride is not None and args.dataset_stride < 1:
        parser.error("--dataset-stride must be >= 1")

    if args.dataset_offset is not None and args.dataset_stride is None:
        parser.error("--dataset-offset requires --dataset-stride")

    dataset_offset = args.dataset_offset if args.dataset_offset is not None else 0
    if dataset_offset < 0:
        parser.error("--dataset-offset must be >= 0")

    datasets = _validate_datasets(args, parser)
    if args.dataset_stride is not None and dataset_offset >= args.dataset_stride:
        parser.error("--dataset-offset must be < --dataset-stride")

    datasets = _apply_dataset_stride(datasets, args.dataset_stride, dataset_offset)
    if not datasets:
        print(
            f"No datasets matched this worker (stride={args.dataset_stride}, offset={dataset_offset}). "
            "Nothing to do."
        )
        return

    args.datasets = datasets
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    shared_model = None
    shared_tokenizer = None
    if args.reuse_base_model and not args.dry_run:
        shared_model, shared_tokenizer = _load_shared_components(args)

    for dataset in datasets:
        _run_single_dataset(args, dataset, run_id, shared_model, shared_tokenizer)


if __name__ == "__main__":  # pragma: no cover
    main()
