"""Training entry point for steering vectors using TRL's SFTTrainer."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.trainer.sft_trainer import SFTConfig, SFTTrainer

from .data import PersonaSteeringDatasetConfig, load_persona_steering_dataset
from .model import QwenSteerModel, SteeringVectorConfig


def _compute_average_loss(
    model: torch.nn.Module,
    dataloader,
) -> float:
    """Compute average token-level negative log likelihood over a dataloader."""

    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            token_count = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if model_was_training:
        model.train()

    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


class EarlyStopCallback(TrainerCallback):
    """Manual early stopping based on validation loss/perplexity."""

    def __init__(self, trainer: SFTTrainer, patience: int, threshold: float) -> None:
        self.trainer = trainer
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.wait = 0
        self.best_vector: torch.Tensor | None = None

    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
        if self.trainer.eval_dataset is None:
            return

        dataloader = kwargs.get("eval_dataloader") or self.trainer.get_eval_dataloader()
        loss = _compute_average_loss(self.trainer.model, dataloader)
        state.log_history.append({"epoch": state.epoch, "eval_loss_manual": loss})

        if loss + self.threshold < self.best_loss:
            self.best_loss = loss
            self.wait = 0
            self.best_vector = self.trainer.model.steering.vector.detach().cpu().clone()
        else:
            self.wait += 1
            if self.patience and self.wait >= self.patience:
                control.should_training_stop = True



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train steering vectors with TRL SFTTrainer")
    parser.add_argument("--datasets", nargs="+", required=True, help="Persona dataset names")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct", help="Base model name")
    parser.add_argument("--target-layer", type=int, default=22, help="Residual layer index for steering")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for steering vector")
    parser.add_argument("--init-scale", type=float, default=0.0, help="Initialization scale for steering vector")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=4096, help="Max tokens per sequence")
    parser.add_argument("--target-tokens", type=int, default=100_000, help="Total training tokens")
    parser.add_argument("--val-target-tokens", type=int, default=0, help="Validation tokens (0 disables split)")
    parser.add_argument("--role-score", type=int, default=3, help="Minimum role extract_score")
    parser.add_argument("--trait-score", type=int, default=75, help="Minimum trait extract_score")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio for scheduler")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    parser.add_argument("--num-epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the frozen base model",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to base model loading (e.g. 'auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Frequency (in optimizer steps) for Trainer logging",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="constant",
        choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if validation loss does not improve for this many evaluations (0 disables)",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=0.0,
        help="Minimum improvement in validation loss to reset patience",
    )
    parser.add_argument(
        "--compare-prompted",
        action="store_true",
        help="After training, report validation perplexity for baseline persona prompts",
    )
    return parser


def prepare_dataset(args, tokenizer) -> Dataset:
    cfg = PersonaSteeringDatasetConfig(
        dataset_names=args.datasets,
        target_tokens=args.target_tokens + max(args.val_target_tokens, 0),
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

    full_dataset = prepare_dataset(args, tokenizer)
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
        raise ValueError("Unable to allocate any training examples; increase target tokens or relax filters")

    train_dataset = full_dataset.select(train_indices)
    train_tokens = sum(int(full_dataset[i]["length"]) for i in train_indices)

    val_dataset = None
    val_selected_tokens = 0
    if val_indices:
        val_dataset = full_dataset.select(val_indices)
        val_selected_tokens = sum(int(full_dataset[i]["length"]) for i in val_indices)

    msg = (
        f"Prepared dataset with {len(train_dataset)} train sequences / {train_tokens} tokens"
    )
    if val_dataset is not None:
        msg += f"; validation {len(val_dataset)} sequences / {val_selected_tokens} tokens"
    print(msg + ".")

    model_cfg = SteeringVectorConfig(
        model_name=args.model,
        target_layer=args.target_layer,
        init_scale=args.init_scale,
    )
    device_map = args.device_map
    model_kwargs: dict[str, object] = {"torch_dtype": "auto"}
    if device_map == "cuda":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = device_map
        if device_map == "auto":
            # Avoid meta tensors so Trainer's .to() call succeeds.
            model_kwargs["low_cpu_mem_usage"] = False

    model = QwenSteerModel(model_cfg, **model_kwargs)

    if device_map == "cuda" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    gradient_checkpointing = getattr(args, "gradient_checkpointing", False)

    eval_strategy = "epoch" if val_dataset is not None else "no"
    save_strategy = "no"

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        do_eval=val_dataset is not None,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=getattr(args, "max_steps", -1),
        bf16=getattr(args, "bf16", False),
        num_train_epochs=getattr(args, "num_epochs", 1.0),
        logging_steps=max(1, args.logging_steps),
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        warmup_ratio=args.warmup_ratio,
        report_to=[],
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        lr_scheduler_type=args.lr_scheduler,
        save_only_model=True,
        save_total_limit=1,
        # NOTE: assistant_only_loss disabled - Qwen tokenizer doesn't support {% generation %}
        # assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Avoid Hugging Face model card writes that can fail on quota-restricted filesystems.
    trainer.create_model_card = lambda *_, **__: None
    # Persist only the steering vector + config to keep checkpoints small and resumable.
    def _save_model(output_dir: str | None = None, _internal_call: bool = False) -> None:
        target_dir = Path(output_dir) if output_dir is not None else args.output_dir
        model.save_pretrained(target_dir)

    trainer.save_model = _save_model  # type: ignore[assignment]
    trainer._val_dataset = val_dataset  # type: ignore[attr-defined]
    trainer._tokenizer = tokenizer  # type: ignore[attr-defined]

    if val_dataset is not None and args.early_stop_patience > 0:
        stop_callback = EarlyStopCallback(trainer, args.early_stop_patience, args.early_stop_threshold)
        trainer.add_callback(stop_callback)
        trainer._early_stop_callback = stop_callback  # type: ignore[attr-defined]
    else:
        trainer._early_stop_callback = None  # type: ignore[attr-defined]

    return trainer


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(args)
    train_result = trainer.train()
    val_dataset = getattr(trainer, "_val_dataset", None)
    early_cb = getattr(trainer, "_early_stop_callback", None)

    if early_cb is not None and getattr(early_cb, "best_vector", None) is not None:
        vector = early_cb.best_vector.to(trainer.model.steering.vector.device)
        trainer.model.steering.vector.data.copy_(vector)

    if val_dataset is not None:
        eval_metrics = trainer.evaluate()
        if "eval_loss" not in eval_metrics:
            eval_loss = _compute_average_loss(trainer.model, trainer.get_eval_dataloader())
            eval_metrics["eval_loss"] = eval_loss
        else:
            eval_loss = eval_metrics["eval_loss"]
        eval_metrics["eval_ppl"] = math.exp(eval_loss)
        print("Validation metrics:", eval_metrics)

        if args.compare_prompted:
            if torch.cuda.is_available():
                trainer.model.to("cpu")
                torch.cuda.empty_cache()
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=False,
            )
            base_model.eval()
            base_loss = _compute_average_loss(base_model, trainer.get_eval_dataloader())
            base_metrics = {
                "eval_loss": base_loss,
                "eval_ppl": math.exp(base_loss),
            }
            print("Prompted baseline metrics:", base_metrics)

    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
