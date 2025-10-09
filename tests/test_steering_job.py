from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from chatspace.steering import job
from chatspace.steering import train as train_mod
from chatspace import cli


def test_sanitize_component_handles_mixed_characters() -> None:
    assert job._sanitize_component("Qwen/Qwen3-32B") == "qwen__qwen3-32b"
    assert job._sanitize_component(" Role Persona ") == "role-persona"
    assert job._sanitize_component("") == "unnamed"


def test_build_run_layout_default_structure(tmp_path: Path) -> None:
    layout = job.build_run_layout(
        model_name="Qwen/Qwen3-32B",
        dataset_name="qwen-3-32b__trait__analytical",
        run_root=tmp_path,
        run_id="20240101T000000Z",
        attempt=2,
        explicit_output=None,
    )

    expected_base = tmp_path / "qwen__qwen3-32b" / "qwen-3-32b__trait__analytical" / "20240101T000000Z"
    expected_attempt = expected_base / "try02"

    assert layout.base_dir == expected_base
    assert layout.attempt_dir == expected_attempt
    assert layout.lock_path == expected_attempt / "run.try2.lock"
    assert layout.commit_path == expected_attempt / "run.try2.commit.json"
    assert layout.error_path == expected_attempt / "run.try2.error.json"
    assert layout.config_path == expected_attempt / "train_args.json"
    assert layout.run_manifest_path == expected_base / "run.json"
    assert layout.run_id == "20240101T000000Z"
    assert layout.attempt == 2


def test_build_run_layout_explicit_output(tmp_path: Path) -> None:
    explicit = tmp_path / "custom"
    layout = job.build_run_layout(
        model_name="Gemma",
        dataset_name="dataset",
        run_root=tmp_path,
        run_id="unused",
        attempt=3,
        explicit_output=explicit,
    )

    assert layout.base_dir == explicit
    assert layout.attempt_dir == explicit
    assert layout.lock_path == explicit / "run.try3.lock"
    assert layout.commit_path == explicit / "run.try3.commit.json"


def test_run_coordinator_creates_lock_and_commit(tmp_path: Path) -> None:
    layout = job.build_run_layout(
        model_name="Qwen",
        dataset_name="dataset",
        run_root=tmp_path,
        run_id="run",
        attempt=1,
        explicit_output=None,
    )
    coord = job.RunCoordinator(layout)

    coord.prepare()
    assert layout.attempt_dir.exists()
    assert layout.run_manifest_path.exists()

    coord.acquire_lock({"status": "running"})
    assert layout.lock_path.exists()

    coord.write_commit({"status": "success"})
    assert layout.commit_path.exists()
    assert not layout.lock_path.exists()

    with layout.commit_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["status"] == "success"


def test_run_coordinator_error_file(tmp_path: Path) -> None:
    layout = job.build_run_layout(
        model_name="Qwen",
        dataset_name="dataset",
        run_root=tmp_path,
        run_id="run",
        attempt=5,
        explicit_output=None,
    )
    coord = job.RunCoordinator(layout)
    coord.prepare()
    coord.acquire_lock({})
    coord.write_error({"status": "failed"})

    assert layout.error_path.exists()
    assert not layout.lock_path.exists()


def test_prepare_train_args_replaces_dataset(tmp_path: Path) -> None:
    args = argparse.Namespace(
        datasets=["original"],
        model="model",
        run_root=tmp_path,
        run_id="run",
        attempt=1,
        skip_if_committed=False,
        skip_if_locked=False,
        force=False,
        dry_run=False,
        reuse_base_model=False,
        output_dir=None,
        extra="value",
    )
    attempt_dir = tmp_path / "attempt"
    result = job._prepare_train_args(args, attempt_dir, "replacement")
    assert result.datasets == ["replacement"]
    assert result.output_dir == attempt_dir
    assert result.extra == "value"


class _DummyParser:
    def error(self, message: str):  # pragma: no cover - simple shim
        raise SystemExit(message)


def test_validate_datasets_allows_single_dataset() -> None:
    args = argparse.Namespace(
        datasets=["only"],
        output_dir=None,
        reuse_base_model=False,
    )
    datasets = job._validate_datasets(args, _DummyParser())
    assert datasets == ["only"]


def test_validate_datasets_requires_reuse_for_multiple() -> None:
    args = argparse.Namespace(
        datasets=["a", "b"],
        output_dir=None,
        reuse_base_model=False,
    )
    with pytest.raises(SystemExit):
        job._validate_datasets(args, _DummyParser())


def test_validate_datasets_blocks_output_dir_combo() -> None:
    args = argparse.Namespace(
        datasets=["a", "b"],
        output_dir=Path("/tmp/out"),
        reuse_base_model=True,
    )
    with pytest.raises(SystemExit):
        job._validate_datasets(args, _DummyParser())


def test_validate_datasets_accepts_multiple_with_reuse() -> None:
    args = argparse.Namespace(
        datasets=["a", "b"],
        output_dir=None,
        reuse_base_model=True,
    )
    datasets = job._validate_datasets(args, _DummyParser())
    assert datasets == ["a", "b"]


def test_scheduler_matches_direct_training(monkeypatch, tmp_path: Path) -> None:
    dataset = "mock-dataset"
    vector_size = 8

    def fake_run_training(args, model=None, tokenizer=None, reset_vector=True):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_val = float(len(getattr(args, "datasets", [])))
        tensor = torch.arange(vector_size, dtype=torch.float32) + base_val
        torch.save({"steering_vector": tensor}, output_dir / "steering_vector.pt")
        (output_dir / "steering_config.json").write_text(
            json.dumps({"datasets": getattr(args, "datasets", [])})
        )
        return {"train_metrics": {"vector_sum": float(tensor.sum())}}

    monkeypatch.setattr(train_mod, "run_training", fake_run_training, raising=False)
    monkeypatch.setattr(job, "run_training", fake_run_training, raising=False)

    direct_dir = tmp_path / "direct"
    parser = train_mod.build_argparser()
    direct_args = parser.parse_args([
        "--datasets",
        dataset,
        "--output-dir",
        str(direct_dir),
    ])
    train_mod.run_training(direct_args)

    scheduled_dir = tmp_path / "scheduled"
    job_args = [
        "--run-root",
        str(tmp_path / "runs"),
        "--model",
        "stub-model",
        "--target-layer",
        "0",
        "--attempt",
        "1",
        "--datasets",
        dataset,
        "--output-dir",
        str(scheduled_dir),
    ]
    job.main(job_args)

    direct_tensor = torch.load(direct_dir / "steering_vector.pt")["steering_vector"]
    scheduled_tensor = torch.load(scheduled_dir / "steering_vector.pt")["steering_vector"]

    if not torch.equal(direct_tensor, scheduled_tensor):
        diff = (direct_tensor - scheduled_tensor).abs()
        stats = {
            "min": float(diff.min()),
            "max": float(diff.max()),
            "mean": float(diff.mean()),
            "var": float(diff.var()),
        }
        pytest.fail(f"Scheduled steering vector differs from direct run: {stats}")


def test_cli_forward_strips_double_dash(monkeypatch):
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv

    monkeypatch.setattr(job, "main", fake_main)

    args = argparse.Namespace(job_args=["--", "--datasets", "foo"])
    cli.handle_steering_train_forward(args)
    assert captured["argv"] == ["--datasets", "foo"]
