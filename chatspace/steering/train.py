"""Training entry point for steering vectors using TRL's SFTTrainer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import SFTConfig, SFTTrainer

from .data import PersonaSteeringDatasetConfig, load_persona_steering_dataset
from .model import QwenSteerModel, SteeringVectorConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train steering vectors with TRL SFTTrainer")
    parser.add_argument("--datasets", nargs="+", required=True, help="Persona dataset names")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct", help="Base model name")
    parser.add_argument("--target-layer", type=int, default=22, help="Residual layer index for steering")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for steering vector")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=4096, help="Max tokens per sequence")
    parser.add_argument("--target-tokens", type=int, default=100_000, help="Total training tokens")
    parser.add_argument("--role-score", type=int, default=3, help="Minimum role extract_score")
    parser.add_argument("--trait-score", type=int, default=75, help="Minimum trait extract_score")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio for scheduler")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    return parser


def prepare_dataset(args, tokenizer) -> Dataset:
    cfg = PersonaSteeringDatasetConfig(
        dataset_names=args.datasets,
        target_tokens=args.target_tokens,
        seed=args.seed,
        tokenizer_name=args.model,
        max_length=args.max_length,
        role_min_score=args.role_score,
        trait_min_score=args.trait_score,
    )
    dataset = load_persona_steering_dataset(cfg, tokenizer)
    return dataset


def build_trainer(args) -> SFTTrainer:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = prepare_dataset(args, tokenizer)

    model_cfg = SteeringVectorConfig(model_name=args.model, target_layer=args.target_layer)
    model = QwenSteerModel(model_cfg, torch_dtype="auto", device_map="auto")

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=getattr(args, "max_steps", -1),
        bf16=getattr(args, "bf16", False),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        warmup_ratio=args.warmup_ratio,
        report_to=[],
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    return trainer


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(args)
    trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()


