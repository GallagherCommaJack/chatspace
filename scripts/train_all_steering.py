"""Batch-train steering vectors while reusing a single Qwen base model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import SFTConfig, SFTTrainer

from chatspace.persona import resolve_persona_datasets
from chatspace.steering.data import (
    PersonaSteeringDatasetConfig,
    prepare_persona_token_budget,
)
from chatspace.steering.model import QwenSteerModel, SteeringVectorConfig
from chatspace.steering.train import EarlyStopCallback, _compute_average_loss


DEFAULT_TRAITS_FILE = Path("/workspace/persona_traits_over_100k.txt")
DEFAULT_ROLES_FILE = Path("/workspace/persona_roles_over_100k.txt")
DEFAULT_DATA_ROOT = Path("/workspace/datasets/processed/persona")
DEFAULT_RUN_ROOT = Path("/workspace/steering_runs")

SKIP_DATASET_SUFFIXES = {"__role__1_default"}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits-file", type=Path, default=DEFAULT_TRAITS_FILE)
    parser.add_argument("--roles-file", type=Path, default=DEFAULT_ROLES_FILE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--target-layer", type=int, default=22)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-tokens", type=int, default=100_000)
    parser.add_argument("--val-tokens", type=int, default=10_000)
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument(
        "--lr-scheduler",
        default="cosine",
        choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
    )
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-threshold", type=float, default=0.0)
    parser.add_argument("--role-score", type=int, default=3)
    parser.add_argument("--trait-score", type=int, default=75)
    parser.add_argument("--compare-prompted", action="store_true")
    parser.add_argument("--trait-prefix", default="qwen-3-32b__trait__")
    parser.add_argument("--role-prefix", default="qwen-3-32b__role__")
    parser.add_argument("--skip-traits", action="store_true", help="Skip training trait steering vectors")
    parser.add_argument("--skip-roles", action="store_true", help="Skip training role steering vectors")
    parser.add_argument("--num-workers", type=int, default=1, help="Processed sequentially; argument reserved")
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def _reset_vector(model: QwenSteerModel, init_scale: float) -> None:
    if init_scale == 0.0:
        model.steering.vector.data.zero_()
    else:
        torch.nn.init.normal_(model.steering.vector, mean=0.0, std=init_scale)
    if model.steering.vector.grad is not None:
        model.steering.vector.grad.zero_()


def _prepare_split(dataset_name: str, args: argparse.Namespace, tokenizer) -> tuple:
    cfg = PersonaSteeringDatasetConfig(
        dataset_names=[dataset_name],
        train_tokens=args.train_tokens,
        val_tokens=max(args.val_tokens, 0),
        seed=args.seed,
        tokenizer_name=args.model,
        max_length=args.max_length,
        role_min_score=args.role_score,
        trait_min_score=args.trait_score,
    )
    result = prepare_persona_token_budget(cfg, tokenizer)

    train_dataset = result.splits["train"]
    train_tokens = result.token_counts.get("train", 0)

    val_dataset = result.splits.get("val")
    val_token_count = result.token_counts.get("val", 0)
    if val_dataset is not None and len(val_dataset) == 0:
        val_dataset = None

    print(
        f"Prepared {dataset_name}: {len(train_dataset)} train seq / {train_tokens} tokens"
        + (f"; val {len(val_dataset)} seq / {val_token_count} tokens." if val_dataset is not None else ".")
    )

    return train_dataset, val_dataset


def _train_single(
    dataset: str,
    args: argparse.Namespace,
    model: QwenSteerModel,
    tokenizer,
) -> None:
    output_dir = args.output_root / dataset
    if output_dir.exists() and args.skip_existing:
        print(f"Skipping {dataset} (exists)")
        return

    print(f"\n=== Training {dataset} ===")
    print("Output dir:", output_dir)

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = _prepare_split(dataset, args, tokenizer)

    _reset_vector(model, args.init_scale)
    torch.manual_seed(args.seed)

    eval_strategy = "steps" if val_dataset is not None else "no"

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        seed=args.seed,
        do_eval=val_dataset is not None,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        bf16=args.bf16,
        num_train_epochs=args.epochs,
        logging_steps=max(1, args.logging_steps),
        eval_strategy=eval_strategy,
        eval_steps=max(1, args.eval_steps),
        warmup_ratio=args.warmup_ratio,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        lr_scheduler_type=args.lr_scheduler,
        save_strategy="no",
        save_only_model=True,
        save_total_limit=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.create_model_card = lambda *_, **__: None

    def _save_model(target: str | None = None, _internal_call: bool = False) -> None:
        dest = Path(target) if target is not None else output_dir
        model.save_pretrained(dest)

    trainer.save_model = _save_model  # type: ignore[assignment]

    early_cb = None
    if val_dataset is not None and args.early_stop_patience > 0:
        early_cb = EarlyStopCallback(trainer, args.early_stop_patience, args.early_stop_threshold)
        trainer.add_callback(early_cb)

    metrics: dict[str, float | str] = {"dataset": dataset, "learning_rate": args.learning_rate}

    try:
        train_result = trainer.train()
    except ValueError as exc:  # dataset too short etc.
        print(f"Failed on {dataset}: {exc}")
        return

    metrics.update(
        {
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_loss": train_result.metrics.get("train_loss"),
            "epoch": train_result.metrics.get("epoch"),
        }
    )

    if early_cb is not None and getattr(early_cb, "best_vector", None) is not None:
        best_vec = early_cb.best_vector.to(model.steering.vector.device)
        model.steering.vector.data.copy_(best_vec)

    if val_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")
        if eval_loss is None:
            eval_loss = _compute_average_loss(model, trainer.get_eval_dataloader())
            eval_metrics["eval_loss"] = eval_loss
        eval_metrics["eval_ppl"] = math.exp(eval_loss)
        metrics.update(eval_metrics)

        if args.compare_prompted:
            stored_vec = model.steering.vector.detach().clone()
            model.steering.vector.data.zero_()
            base_loss = _compute_average_loss(model, trainer.get_eval_dataloader())
            metrics["baseline_loss"] = base_loss
            metrics["baseline_ppl"] = math.exp(base_loss)
            model.steering.vector.data.copy_(stored_vec)

    model.save_pretrained(output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.free_memory()
    del trainer
    torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    include_traits = not args.skip_traits
    include_roles = not args.skip_roles

    if not include_traits:
        print("[INFO] Skipping trait datasets (--skip-traits)")
    if not include_roles:
        print("[INFO] Skipping role datasets (--skip-roles)")

    datasets = resolve_persona_datasets(
        traits_file=args.traits_file if include_traits else None,
        roles_file=args.roles_file if include_roles else None,
        trait_prefix=args.trait_prefix,
        role_prefix=args.role_prefix,
        include_traits=include_traits,
        include_roles=include_roles,
    )

    filtered: list[str] = []
    for dataset in datasets:
        if any(dataset.endswith(suffix) for suffix in SKIP_DATASET_SUFFIXES):
            print(f"Skipping dataset {dataset} (excluded suffix)")
            continue
        dataset_path = args.data_root / dataset
        if not dataset_path.exists():
            continue
        filtered.append(dataset)

    datasets = sorted(set(filtered))

    print(f"Found {len(datasets)} datasets with available processed data")
    if args.dry_run:
        for name in datasets:
            print(f"[DRY RUN] Would train {name} -> {args.output_root / name}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_cfg = SteeringVectorConfig(
        model_name=args.model,
        target_layer=args.target_layer,
        init_scale=args.init_scale,
    )
    model_kwargs: dict[str, object] = {"torch_dtype": "auto"}
    if args.device_map == "cuda":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = args.device_map
        if args.device_map == "auto":
            model_kwargs["low_cpu_mem_usage"] = False

    model = QwenSteerModel(model_cfg, **model_kwargs)
    if args.device_map == "cuda" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    for name in datasets:
        _train_single(name, args, model, tokenizer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
