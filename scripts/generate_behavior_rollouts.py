"""Generate persona question rollouts for prompted, trained, and activation steering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm, trange


HOME = Path.home()
DEFAULT_LOG = Path("/workspace/steering_runs/steering_sweep.log")
DEFAULT_RUN_ROOT = Path("/workspace/steering_runs")
PERSONA_ROOT = Path("/workspace/persona-data")
INSTRUCTIONS_ROOT = HOME / "persona-subspace"

TARGET_LAYER = 22


class SteeringController:
    """Attach a single residual hook and swap steering vectors on demand."""

    def __init__(self, model: AutoModelForCausalLM) -> None:
        self.model = model
        self.layer_idx: int | None = None
        self._handle = None
        self.vector: torch.Tensor | None = None

    def _hook(self, module, args, output):
        if self.vector is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        vec = self.vector
        if vec.device != hidden.device or vec.dtype != hidden.dtype:
            vec = vec.to(device=hidden.device, dtype=hidden.dtype)
            self.vector = vec
        steered = hidden + vec
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def set_layer(self, layer_idx: int) -> None:
        if self.layer_idx == layer_idx:
            return
        if self._handle is not None:
            self._handle.remove()
        layer = self.model.model.layers[layer_idx]
        self._handle = layer.register_forward_hook(self._hook)
        self.layer_idx = layer_idx

    def set_vector(self, vector: torch.Tensor | None) -> None:
        if vector is None:
            self.vector = None
            return
        if vector.ndim != 1:
            raise ValueError("Steering vector must be 1D")
        self.vector = vector

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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


def make_messages(
    system_prompt: str | None,
    question: str,
    include_system: bool = True,
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if include_system and system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    return msgs


def generate_variants(
    dataset: str,
    prompts: list[str],
    questions: list[str],
    args,
    tokenizer,
    baseline_model,
    controller: SteeringController,
    device: torch.device,
) -> Iterable[dict[str, object]]:
    def run_batch(
        message_batches: list[list[dict[str, str]]],
        vector: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> list[str]:
        target_layer = layer_idx if layer_idx is not None else args.target_layer
        controller.set_layer(target_layer)
        controller.set_vector(vector)

        chat_texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in message_batches
        ]
        encoded = tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            input_lens = torch.tensor([enc.size(0) for enc in encoded["input_ids"]], device=device)
        else:
            input_lens = attention_mask.sum(dim=1)
        outputs = baseline_model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        texts: list[str] = []
        for idx in range(outputs.size(0)):
            seq = outputs[idx]
            offset = int(input_lens[idx])
            decoded = tokenizer.decode(seq[offset:], skip_special_tokens=True).strip()
            texts.append(decoded)
        return texts

    steering_dir = args.run_root / dataset
    trained_vector: torch.Tensor | None = None
    trained_layer = args.target_layer
    vector_path = steering_dir / "steering_vector.pt"
    if vector_path.exists():
        state = torch.load(vector_path, map_location="cpu")
        tensor = state.get("steering_vector")
        if tensor is None:
            raise ValueError(f"steering_vector.pt missing 'steering_vector' key at {vector_path}")
        trained_vector = tensor.float()
        if torch.cuda.is_available():
            trained_vector = trained_vector.to(device)
        config_path = steering_dir / "steering_config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            trained_layer = int(cfg.get("target_layer", trained_layer))

    activation_vec = load_activation_vector(dataset)
    activation_layer = TARGET_LAYER
    if activation_vec is not None:
        activation_vec = activation_vec.float()
        if torch.cuda.is_available():
            activation_vec = activation_vec.to(device)
    else:
        activation_layer = args.target_layer

    baseline_layer = args.target_layer
    if trained_vector is not None:
        baseline_layer = trained_layer
    elif activation_vec is not None:
        baseline_layer = activation_layer

    steering_include_system = not args.steering_no_system

    # Baseline generations: loop over prompts, batch questions per rollout
    for prompt_idx, prompt in enumerate(prompts):
        baseline_batches = [make_messages(prompt, question) for question in questions]
        progress_desc = f"{dataset} prompted {prompt_idx + 1}/{len(prompts)}"
        for rollout_idx in trange(args.rollouts, desc=progress_desc, leave=False):
            responses = run_batch(baseline_batches, vector=None, layer_idx=baseline_layer)
            for question_idx, (question, response) in enumerate(zip(questions, responses)):
                yield {
                    "dataset": dataset,
                    "variant": "prompted",
                    "prompt_index": prompt_idx,
                    "question_index": question_idx,
                    "rollout_index": rollout_idx,
                    "question": question,
                    "system_prompt": prompt,
                    "response": response,
                }

    if prompts:
        print(
            f"[{dataset}] Prompted rollouts complete: {len(prompts)} prompts x {args.rollouts} rollouts."
        )

    steering_include_system = not args.steering_no_system
    steering_prompt = prompts[0] if prompts else None
    steering_batches = [
        make_messages(steering_prompt, question, include_system=steering_include_system)
        for question in questions
    ]

    if trained_vector is not None:
        for rollout_idx in trange(args.rollouts, desc=f"{dataset} trained", leave=False):
            responses = run_batch(steering_batches, vector=trained_vector, layer_idx=trained_layer)
            for question_idx, (question, response) in enumerate(zip(questions, responses)):
                yield {
                    "dataset": dataset,
                    "variant": "trained",
                    "prompt_index": None,
                    "question_index": question_idx,
                    "rollout_index": rollout_idx,
                    "question": question,
                    "system_prompt": steering_prompt if steering_include_system else None,
                    "response": response,
                }
        print(
            f"[{dataset}] Trained steering rollouts complete: {args.rollouts} rollouts x {len(questions)} questions."
        )

    if activation_vec is not None:
        for rollout_idx in trange(args.rollouts, desc=f"{dataset} activation", leave=False):
            responses = run_batch(steering_batches, vector=activation_vec, layer_idx=activation_layer)
            for question_idx, (question, response) in enumerate(zip(questions, responses)):
                yield {
                    "dataset": dataset,
                    "variant": "activation",
                    "prompt_index": None,
                    "question_index": question_idx,
                    "rollout_index": rollout_idx,
                    "question": question,
                    "system_prompt": steering_prompt if steering_include_system else None,
                    "response": response,
                }
        print(
            f"[{dataset}] Activation steering rollouts complete: {args.rollouts} rollouts x {len(questions)} questions."
        )


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
    parser.add_argument(
        "--steering-no-system",
        action="store_true",
        help="Drop the system prompt for trained/activation generations",
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=False,
    )
    baseline_model = AutoModelForCausalLM.from_pretrained(args.model, **base_kwargs).eval()
    controller = SteeringController(baseline_model)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = read_log_datasets(args.log, args.include_prefix)

    args.output_root.mkdir(parents=True, exist_ok=True)

    try:
        for dataset in tqdm(datasets, desc="Datasets"):
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
                for record in generate_variants(
                    dataset,
                    prompts,
                    questions,
                    args,
                    tokenizer,
                    baseline_model,
                    controller,
                    device,
                ):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        controller.close()
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
