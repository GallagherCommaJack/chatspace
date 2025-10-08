#!/usr/bin/env python3
"""Sweep constant learning rates for steering-vector training without reloading the base model."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import SFTConfig, SFTTrainer

from chatspace.steering.data import PersonaSteeringDatasetConfig, load_persona_steering_dataset
from chatspace.steering.model import QwenSteerModel, SteeringVectorConfig
from chatspace.steering.train import EarlyStopCallback, _compute_average_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True, help="Persona dataset names to train on")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to store sweep outputs")
    parser.add_argument("--learning-rates", type=float, nargs="+", required=True, help="Learning rates to evaluate")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Base model to steer")
    parser.add_argument("--target-layer", type=int, default=22, help="Residual layer index for steering")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--init-scale", type=float, default=0.0, help="Stddev for steering vector init (0 => zeros)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=4096, help="Max tokens per sequence")
    parser.add_argument("--target-tokens", type=int, default=100_000, help="Training token budget")
    parser.add_argument("--val-target-tokens", type=int, default=10_000, help="Validation token budget (0 disables)")
    parser.add_argument("--role-score", type=int, default=3, help="Minimum role extract_score filter")
    parser.add_argument("--trait-score", type=int, default=75, help="Minimum trait extract_score filter")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio for scheduler")
    parser.add_argument("--num-epochs", type=float, default=1.0, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--device-map", default="auto", help="Device map passed to model loading")
    parser.add_argument("--logging-steps", type=int, default=10, help="Trainer logging frequency (optimizer steps)")
    parser.add_argument("--eval-steps", type=int, default=100, help="Validation frequency when eval set is present")
    parser.add_argument(
        "--lr-scheduler",
        default="constant",
        choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
        help="Learning-rate scheduler",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop after N non-improving evals (0 disables)")
    parser.add_argument("--early-stop-threshold", type=float, default=0.0, help="Min improvement to reset patience")
    parser.add_argument("--compare-prompted", action="store_true", help="Report perplexity for the unsteered model as well")
    return parser.parse_args()


def reset_vector(model: QwenSteerModel, init_scale: float) -> None:
    if init_scale == 0.0:
        model.steering.vector.data.zero_()
    else:
        torch.nn.init.normal_(model.steering.vector, mean=0.0, std=init_scale)
    if model.steering.vector.grad is not None:
        model.steering.vector.grad.zero_()


def prepare_datasets(args, tokenizer) -> tuple:
    cfg = PersonaSteeringDatasetConfig(
        dataset_names=args.datasets,
        target_tokens=args.target_tokens + max(args.val_target_tokens, 0),
        seed=args.seed,
        tokenizer_name=args.model,
        max_length=args.max_length,
        role_min_score=args.role_score,
        trait_min_score=args.trait_score,
    )
    full_dataset = load_persona_steering_dataset(cfg, tokenizer)

    token_lengths = list(full_dataset["length"])
    target_tokens = args.target_tokens
    val_tokens = max(args.val_target_tokens, 0)

    cumulative = 0
    train_indices: list[int] = []
    val_indices: list[int] = []

    for idx, length in enumerate(token_lengths):
        cumulative += int(length)
        if cumulative <= target_tokens:
            train_indices.append(idx)
        elif val_tokens > 0 and cumulative <= target_tokens + val_tokens:
            val_indices.append(idx)
        else:
            break

    if not train_indices:
        raise ValueError("No training examples selected; increase target tokens or relax filters")

    train_dataset = full_dataset.select(train_indices)
    train_tokens = sum(int(full_dataset[i]["length"]) for i in train_indices)

    val_dataset = None
    val_selected_tokens = 0
    if val_indices:
        val_dataset = full_dataset.select(val_indices)
        val_selected_tokens = sum(int(full_dataset[i]["length"]) for i in val_indices)

    print(
        f"Prepared dataset: {len(train_dataset)} train seq / {train_tokens} tokens"
        + (
            f"; validation {len(val_dataset)} seq / {val_selected_tokens} tokens." if val_dataset is not None else "."
        )
    )

    return train_dataset, val_dataset


def build_model(args) -> QwenSteerModel:
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

    return model


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset, val_dataset = prepare_datasets(args, tokenizer)
    model = build_model(args)

    results: list[dict[str, float]] = []

    for lr in args.learning_rates:
        run_dir = args.output_root / f"lr_{lr:g}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Training at learning rate {lr} ===")

        reset_vector(model, args.init_scale)
        torch.manual_seed(args.seed)

        eval_strategy = "steps" if val_dataset is not None else "no"

        sft_config = SFTConfig(
            output_dir=str(run_dir),
            seed=args.seed,
            do_eval=val_dataset is not None,
            learning_rate=lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_steps=args.max_steps,
            bf16=args.bf16,
            num_train_epochs=args.num_epochs,
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

        def _save_model(output_dir: str | None = None, _internal_call: bool = False) -> None:
            target = Path(output_dir) if output_dir is not None else run_dir
            model.save_pretrained(target)

        trainer.save_model = _save_model  # type: ignore[assignment]

        early_cb = None
        if val_dataset is not None and args.early_stop_patience > 0:
            early_cb = EarlyStopCallback(trainer, args.early_stop_patience, args.early_stop_threshold)
            trainer.add_callback(early_cb)

        start = time.time()
        train_result = trainer.train()
        runtime = time.time() - start

        if early_cb is not None and getattr(early_cb, "best_vector", None) is not None:
            best_vec = early_cb.best_vector.to(model.steering.vector.device)
            model.steering.vector.data.copy_(best_vec)

        metrics = {
            "learning_rate": lr,
            "train_runtime": train_result.metrics.get("train_runtime", runtime),
            "train_loss": train_result.metrics.get("train_loss"),
        }

        if val_dataset is not None:
            eval_metrics = trainer.evaluate()
            eval_loss = eval_metrics.get("eval_loss")
            if eval_loss is None:
                eval_loss = _compute_average_loss(model, trainer.get_eval_dataloader())
                eval_metrics["eval_loss"] = eval_loss
            eval_metrics["eval_ppl"] = math.exp(eval_loss)
            metrics.update(eval_metrics)

            if args.compare_prompted:
                base_model = QwenSteerModel(
                    SteeringVectorConfig(model_name=args.model, target_layer=args.target_layer, init_scale=0.0),
                    torch_dtype="auto",
                    device_map=args.device_map,
                )
                base_loss = _compute_average_loss(base_model, trainer.get_eval_dataloader())
                metrics["baseline_loss"] = base_loss
                metrics["baseline_ppl"] = math.exp(base_loss)
                del base_model

        model.save_pretrained(run_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        results.append(metrics)

        trainer.accelerator.free_memory()
        torch.cuda.empty_cache()
        del trainer
        model.zero_grad(set_to_none=True)
        model.train()

    summary_path = args.output_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSweep complete; metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
