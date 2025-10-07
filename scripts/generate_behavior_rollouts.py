"""Generate persona question rollouts for prompted, trained, and activation steering."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatspace.steering.model import QwenSteerModel, SteeringVectorConfig


HOME = Path.home()
DEFAULT_LOG = Path("/workspace/steering_runs/steering_sweep.log")
DEFAULT_RUN_ROOT = Path("/workspace/steering_runs")
PERSONA_ROOT = Path("/workspace/persona-data")
INSTRUCTIONS_ROOT = HOME / "persona-subspace"

TARGET_LAYER = 22


def read_log_datasets(log_path: Path, prefixes: Sequence[str]) -> list[str]:
    datasets: list[str] = []
    tuples = tuple(prefixes)
    for line in log_path.read_text().splitlines():
        if line.startswith("=== Training "):
            name = line.split("Training ", 1)[1].split(" ===", 1)[0]
            if name.startswith(tuples) and name not in datasets:
                datasets.append(name)
    return datasets


def load_instructions(dataset: str) -> tuple[list[str], list[str]]:
    if "__trait__" in dataset:
        name = dataset.split("__trait__", 1)[1]
        path = INSTRUCTIONS_ROOT / "traits" / "data" / "instructions" / f"{name}.json"
    elif "__role__" in dataset:
        name = dataset.split("__role__", 1)[1]
        path = INSTRUCTIONS_ROOT / "roles" / "data" / "instructions" / f"{name}.json"
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset}")

    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text())
    prompts = []
    for entry in payload.get("instruction", []):
        if isinstance(entry, dict):
            prompt = entry.get("pos") or entry.get("prompt")
            if prompt:
                prompts.append(prompt)
        elif isinstance(entry, str):
            prompts.append(entry)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError(f"No questions in {path}")
    return prompts, questions


def load_activation_vector(dataset: str) -> torch.Tensor | None:
    if "__trait__" in dataset:
        model_prefix, trait = dataset.split("__trait__", 1)
        vec_file = PERSONA_ROOT / f"{model_prefix}/traits_240/vectors/{trait}.pt"
        if not vec_file.exists():
            return None
        data = torch.load(vec_file, map_location="cpu")
        key = "pos_70" if "pos_70" in data else next(iter(data))
        return data[key][TARGET_LAYER].float()
    if "__role__" in dataset:
        role = dataset.split("__role__", 1)[1]
        vec_file = PERSONA_ROOT / f"qwen-3-32b/roles_240/vectors/{role}.pt"
        if not vec_file.exists():
            return None
        data = torch.load(vec_file, map_location="cpu")
        key = "pos_3" if "pos_3" in data else next(iter(data))
        return data[key][TARGET_LAYER].float()
    return None


def make_messages(system_prompt: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate_variants(
    dataset: str,
    prompts: list[str],
    questions: list[str],
    args,
    tokenizer,
) -> Iterable[dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_model(model, msgs):
        chat_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(chat_text, return_tensors="pt").to(device)
        input_len = encoded["input_ids"].shape[1]
        outputs = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token,
        )
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return text

    base_kwargs = dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto", low_cpu_mem_usage=False)

    baseline_model = AutoModelForCausalLM.from_pretrained(args.model, **base_kwargs).eval()

    steering_dir = args.run_root / dataset
    trained_header = None
    if (steering_dir / "steering_vector.pt").exists():
        trained_model = QwenSteerModel.from_pretrained(steering_dir, **base_kwargs).eval()
        trained_header = trained_model
    else:
        trained_model = None

    activation_vec = load_activation_vector(dataset)
    if activation_vec is not None:
        activation_cfg = SteeringVectorConfig(model_name=args.model, target_layer=args.target_layer, init_scale=0.0)
        activation_model = QwenSteerModel(activation_cfg, **base_kwargs).eval()
        activation_model.steering.vector.data.copy_(activation_vec.to(activation_model.steering.vector.device))
    else:
        activation_model = None

    for prompt_idx, prompt in enumerate(prompts):
        for question_idx, question in enumerate(questions):
            msgs = make_messages(prompt, question)
            for rollout_idx in range(args.rollouts):
                yield {
                    "dataset": dataset,
                    "variant": "prompted",
                    "prompt_index": prompt_idx,
                    "question_index": question_idx,
                    "rollout_index": rollout_idx,
                    "question": question,
                    "system_prompt": prompt,
                    "response": run_model(baseline_model, msgs),
                }
                if trained_model is not None:
                    yield {
                        "dataset": dataset,
                        "variant": "trained",
                        "prompt_index": prompt_idx,
                        "question_index": question_idx,
                        "rollout_index": rollout_idx,
                        "question": question,
                        "system_prompt": prompt,
                        "response": run_model(trained_model, msgs),
                    }
                if activation_model is not None:
                    yield {
                        "dataset": dataset,
                        "variant": "activation",
                        "prompt_index": prompt_idx,
                        "question_index": question_idx,
                        "rollout_index": rollout_idx,
                        "question": question,
                        "system_prompt": prompt,
                        "response": run_model(activation_model, msgs),
                    }

    del baseline_model
    if trained_header is not None:
        del trained_header
    if activation_model is not None:
        del activation_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/steering_rollouts"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--target-layer", type=int, default=22)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--include-prefix", nargs="*", default=["qwen-3-32b__trait__", "gemma-2-27b__role__"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = read_log_datasets(args.log, args.include_prefix)

    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        run_dir = args.output_root / dataset
        output_path = run_dir / "rollouts.jsonl"
        if output_path.exists() and args.skip_existing:
            print(f"Skipping {dataset} (rollouts exist)")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            prompts, questions = load_instructions(dataset)
        except FileNotFoundError:
            print(f"Missing instructions for {dataset}")
            continue
        except ValueError as exc:
            print(f"Skipping {dataset}: {exc}")
            continue

        print(f"Generating {dataset} -> {output_path}")
        with output_path.open("w", encoding="utf-8") as fout:
            for record in generate_variants(dataset, prompts, questions, args, tokenizer):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
