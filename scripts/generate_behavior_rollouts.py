"""Generate persona question rollouts and optionally score them with MiniLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm, trange


class RateLimiter:
    """Simple async rate limiter with token bucket semantics."""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            wait_time = (1 - self.tokens) / self.rate if self.rate > 0 else 0
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.tokens = max(0.0, self.tokens - 1)


HOME = Path.home()
DEFAULT_LOG = Path("/workspace/steering_runs/steering_sweep.log")
DEFAULT_RUN_ROOT = Path("/workspace/steering_runs")
PERSONA_ROOT = Path("/workspace/persona-data")
INSTRUCTIONS_ROOT = HOME / "persona-subspace"

TARGET_LAYER = 22


@dataclass
class InstructionData:
    prompts: list[str]
    questions: list[str]
    eval_prompt: str | None


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


def _train_minilm_classifier(dataset, model_name: str, batch_size: int, seed: int):
    import numpy as np
    import random
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    embedder = SentenceTransformer(model_name)

    texts: list[str] = []
    labels: list[int] = []
    for row in dataset:
        assistant_msgs = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        if not assistant_msgs:
            continue
        texts.append(assistant_msgs[-1])
        labels.append(1 if row.get("label") == "pos" else 0)

    if not texts:
        raise ValueError("Dataset does not contain assistant messages for MiniLM training")

    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    return embedder, clf, metrics


def _score_texts(embedder, clf, texts: Sequence[str], batch_size: int) -> list[float]:
    import numpy as np

    if not texts:
        return []
    embeddings = embedder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    probs = clf.predict_proba(embeddings)[:, 1]
    return [float(p) for p in probs]


def _summarize_scores(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    if not values:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.array(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_minilm_outputs(dataset: str, records: list[dict], metrics: dict, run_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    score_rows: list[dict] = []
    variant_groups: dict[str, list[float]] = defaultdict(list)
    question_groups: dict[tuple[str, int], list[float]] = defaultdict(list)

    for rec in records:
        score = rec.get("minilm_score")
        if score is None:
            continue
        question_index = rec.get("question_index")
        row = {
            "dataset": dataset,
            "variant": rec.get("variant"),
            "prompt_index": rec.get("prompt_index"),
            "question_index": question_index,
            "rollout_index": rec.get("rollout_index"),
            "steering_scale": rec.get("steering_scale"),
            "minilm_score": float(score),
        }
        score_rows.append(row)
        if row["variant"] is not None:
            variant_groups[row["variant"]].append(float(score))
        if question_index is not None and row["variant"] is not None:
            question_groups[(row["variant"], question_index)].append(float(score))

    if not score_rows:
        print(f"[{dataset}] MiniLM eval produced no scores; skipping write")
        return

    pq.write_table(pa.Table.from_pylist(score_rows), run_dir / "minilm_scores.parquet")

    per_question_rows = []
    for (variant, q_idx), vals in question_groups.items():
        summary = _summarize_scores(vals)
        per_question_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "question_index": q_idx,
                **summary,
            }
        )
    if per_question_rows:
        pq.write_table(pa.Table.from_pylist(per_question_rows), run_dir / "minilm_per_question.parquet")

    variant_summary = {variant: _summarize_scores(vals) for variant, vals in variant_groups.items()}
    summary_payload = {
        "dataset": dataset,
        "classifier_metrics": metrics,
        "variant_summary": variant_summary,
    }
    (run_dir / "minilm_summary.json").write_text(json.dumps(summary_payload, indent=2))

    human = ", ".join(f"{variant}={stats['mean']:.3f}" for variant, stats in variant_summary.items())
    print(f"[{dataset}] MiniLM summary => {human}")


def _run_minilm_eval(dataset: str, records: list[dict], args, run_dir: Path) -> None:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        print(f"[{dataset}] Skipping MiniLM eval (datasets import failed: {exc})")
        return

    data_path = args.data_root / dataset
    if not data_path.exists():
        print(f"[{dataset}] Skipping MiniLM eval (missing dataset at {data_path})")
        return

    dataset_dict = load_from_disk(str(data_path))
    split = dataset_dict["train"] if "train" in dataset_dict else dataset_dict[next(iter(dataset_dict.keys()))]

    try:
        embedder, clf, metrics = _train_minilm_classifier(
            split,
            args.minilm_model,
            args.minilm_batch_size,
            args.minilm_seed,
        )
    except ValueError as exc:
        print(f"[{dataset}] MiniLM eval failed: {exc}")
        return

    scores = _score_texts(embedder, clf, [rec["response"] for rec in records], args.minilm_batch_size)
    for rec, score in zip(records, scores):
        rec["minilm_score"] = score

    _write_minilm_outputs(dataset, records, metrics, run_dir)

    del embedder, clf
    torch.cuda.empty_cache()


def _parse_judge_score(response_text: str):
    if not response_text:
        return None
    upper = response_text.upper()
    if "REFUSAL" in upper:
        return "REFUSAL"
    import re

    numbers = re.findall(r"\b(\d{1,3})\b", response_text)
    for num in numbers:
        try:
            value = int(num)
        except ValueError:
            continue
        if 0 <= value <= 100:
            return value
    return None


async def _judge_call(client, prompt: str, args, rate_limiter: RateLimiter):
    await rate_limiter.acquire()
    try:
        response = await client.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Judge call failed: {exc}")
        return None

    if not response.choices:
        return None
    return response.choices[0].message.content


async def _judge_dataset_async(prompts: list[str], args) -> list[str | None]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(f"openai package not installed: {exc}") from exc

    client = AsyncOpenAI()
    rate_limiter = RateLimiter(args.judge_requests_per_second)
    results: list[str | None] = []

    for idx in range(0, len(prompts), args.judge_batch_size):
        batch = prompts[idx : idx + args.judge_batch_size]
        tasks = [
            asyncio.create_task(_judge_call(client, prompt, args, rate_limiter))
            for prompt in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in batch_results:
            if isinstance(res, Exception):
                print(f"Judge batch error: {res}")
                results.append(None)
            else:
                results.append(res)

    return results


def _write_judge_outputs(dataset: str, records: list[dict], args, run_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    variant_scores: dict[str, list[float]] = defaultdict(list)
    variant_refusals: dict[str, int] = defaultdict(int)
    variant_missing: dict[str, int] = defaultdict(int)
    question_scores: dict[tuple[str, int], list[float]] = defaultdict(list)
    question_refusals: dict[tuple[str, int], int] = defaultdict(int)

    for rec in records:
        if "judge_score" not in rec and "judge_refusal" not in rec:
            continue
        variant = rec.get("variant")
        score = rec.get("judge_score")
        refusal = bool(rec.get("judge_refusal", False))
        question_index = rec.get("question_index")
        row = {
            "dataset": dataset,
            "variant": variant,
            "prompt_index": rec.get("prompt_index"),
            "question_index": question_index,
            "rollout_index": rec.get("rollout_index"),
            "steering_scale": rec.get("steering_scale"),
            "judge_score": float(score) if score is not None else None,
            "judge_refusal": refusal,
            "judge_response": rec.get("judge_response"),
        }
        rows.append(row)
        if variant is None:
            continue
        if refusal:
            variant_refusals[variant] += 1
            if question_index is not None:
                question_refusals[(variant, question_index)] += 1
        elif score is None:
            variant_missing[variant] += 1
        else:
            variant_scores[variant].append(float(score))
            if question_index is not None:
                question_scores[(variant, question_index)].append(float(score))

    if not rows:
        print(f"[{dataset}] Judge eval produced no rows; skipping write")
        return

    pq.write_table(pa.Table.from_pylist(rows), run_dir / "judge_scores.parquet")

    per_question_rows = []
    for (variant, q_idx), scores in question_scores.items():
        summary = _summarize_scores(scores)
        per_question_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "question_index": q_idx,
                **summary,
                "refusals": question_refusals.get((variant, q_idx), 0),
            }
        )
    if per_question_rows:
        pq.write_table(pa.Table.from_pylist(per_question_rows), run_dir / "judge_per_question.parquet")

    all_variants = set(variant_scores) | set(variant_refusals) | set(variant_missing)
    variant_summary = {}
    for variant in sorted(all_variants):
        stats = _summarize_scores(variant_scores.get(variant, []))
        stats["refusals"] = variant_refusals.get(variant, 0)
        stats["missing"] = variant_missing.get(variant, 0)
        variant_summary[variant] = stats

    summary_payload = {
        "dataset": dataset,
        "judge_model": args.judge_model,
        "variant_summary": variant_summary,
    }
    (run_dir / "judge_summary.json").write_text(json.dumps(summary_payload, indent=2))

    printable = ", ".join(
        f"{variant}={variant_summary[variant]['mean']:.1f} (ref {variant_summary[variant]['refusals']})"
        for variant in sorted(variant_summary)
    )
    print(f"[{dataset}] Judge summary => {printable}")


def _run_judge_eval(
    dataset: str,
    instructions: InstructionData,
    records: list[dict],
    args,
    run_dir: Path,
) -> None:
    eval_template = instructions.eval_prompt
    if not eval_template:
        print(f"[{dataset}] Skipping judge eval (instruction missing eval_prompt)")
        return

    prompts: list[str] = []
    target_records: list[dict] = []
    for rec in records:
        response = rec.get("response")
        question = rec.get("question")
        if not response or not question:
            continue
        try:
            prompt = eval_template.format(question=question, answer=response)
        except KeyError as exc:
            print(f"[{dataset}] Judge prompt format error: {exc}")
            continue
        prompts.append(prompt)
        target_records.append(rec)

    if not prompts:
        print(f"[{dataset}] No judge prompts to evaluate")
        return

    print(f"[{dataset}] Running judge eval on {len(prompts)} responses with {args.judge_model}")
    try:
        judge_outputs = asyncio.run(_judge_dataset_async(prompts, args))
    except RuntimeError as exc:
        print(f"[{dataset}] Judge eval skipped: {exc}")
        return

    for rec, raw in zip(target_records, judge_outputs):
        rec["judge_response"] = raw
        parsed = _parse_judge_score(raw) if raw is not None else None
        if parsed == "REFUSAL":
            rec["judge_score"] = None
            rec["judge_refusal"] = True
        elif isinstance(parsed, int):
            rec["judge_score"] = float(parsed)
            rec["judge_refusal"] = False
        else:
            rec["judge_score"] = None
            rec["judge_refusal"] = False

    _write_judge_outputs(dataset, records, args, run_dir)


def read_log_datasets(log_path: Path, prefixes: Sequence[str]) -> list[str]:
    datasets: list[str] = []
    tuples = tuple(prefixes)
    for line in log_path.read_text().splitlines():
        if line.startswith("=== Training "):
            name = line.split("Training ", 1)[1].split(" ===", 1)[0]
            if name.startswith(tuples) and name not in datasets:
                datasets.append(name)
    return datasets


def load_instructions(dataset: str) -> InstructionData:
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
    prompts: list[str] = []
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
    eval_prompt = payload.get("eval_prompt")
    return InstructionData(prompts=prompts, questions=questions, eval_prompt=eval_prompt)


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


def _prepare_scaled_variants(
    base_name: str,
    vector: torch.Tensor | None,
    scales: Sequence[float],
    normalize: bool,
    *,
    include_learned: bool = False,
) -> list[tuple[str, torch.Tensor, float]]:
    if vector is None:
        return []

    vec = vector
    norm = float(torch.linalg.norm(vec).item())
    if normalize:
        if norm > 0:
            vec = vec / norm
        else:
            normalize = False

    variants: list[tuple[str, torch.Tensor, float]] = []
    if include_learned:
        learned_name = f"{base_name}_scale_learned"
        if normalize and norm > 0:
            variants.append((learned_name, vec * norm, norm))
        else:
            variants.append((learned_name, vector, 1.0))
    existing_names = {name for name, _, _ in variants}
    for scale in scales:
        scaled = vec * float(scale)
        scaled_name = base_name
        if not (len(scales) == 1 and abs(scale - 1.0) < 1e-6):
            scaled_name = f"{base_name}_scale_{scale:g}"
        if scaled_name in existing_names:
            continue
        variants.append((scaled_name, scaled, float(scale)))
        existing_names.add(scaled_name)
    return variants


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
    instructions: InstructionData,
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

    trained_variants = _prepare_scaled_variants(
        "trained",
        trained_vector,
        args.trained_scales,
        args.normalize_steering,
        include_learned=True,
    )
    activation_variants = _prepare_scaled_variants(
        "activation",
        activation_vec,
        args.activation_scales,
        args.normalize_steering,
        include_learned=False,
    )

    baseline_layer = args.target_layer
    if trained_variants:
        baseline_layer = trained_layer
    elif activation_variants:
        baseline_layer = activation_layer

    steering_include_system = not args.steering_no_system

    # Baseline generations: loop over prompts, batch questions per rollout
    for prompt_idx, prompt in enumerate(instructions.prompts):
        baseline_batches = [make_messages(prompt, question) for question in instructions.questions]
        progress_desc = f"{dataset} prompted {prompt_idx + 1}/{len(instructions.prompts)}"
        for rollout_idx in trange(args.rollouts, desc=progress_desc, leave=False):
            responses = run_batch(baseline_batches, vector=None, layer_idx=baseline_layer)
            for question_idx, (question, response) in enumerate(zip(instructions.questions, responses)):
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

    if instructions.prompts:
        print(
            f"[{dataset}] Prompted rollouts complete: {len(instructions.prompts)} prompts x {args.rollouts} rollouts."
        )

    steering_include_system = not args.steering_no_system
    steering_prompt = instructions.prompts[0] if instructions.prompts else None
    steering_batches = [
        make_messages(steering_prompt, question, include_system=steering_include_system)
        for question in instructions.questions
    ]

    for variant_name, variant_vec, scale in trained_variants:
        for rollout_idx in trange(args.rollouts, desc=f"{dataset} {variant_name}", leave=False):
            responses = run_batch(steering_batches, vector=variant_vec, layer_idx=trained_layer)
            for question_idx, (question, response) in enumerate(zip(instructions.questions, responses)):
                yield {
                    "dataset": dataset,
                    "variant": variant_name,
                    "prompt_index": None,
                    "question_index": question_idx,
                    "rollout_index": rollout_idx,
                    "question": question,
                    "system_prompt": steering_prompt if steering_include_system else None,
                    "steering_scale": scale,
                    "response": response,
                }
        print(
            f"[{dataset}] {variant_name} rollouts complete: {args.rollouts} rollouts x {len(instructions.questions)} questions."
        )

    for variant_name, variant_vec, scale in activation_variants:
        for rollout_idx in trange(args.rollouts, desc=f"{dataset} {variant_name}", leave=False):
            responses = run_batch(steering_batches, vector=variant_vec, layer_idx=activation_layer)
            for question_idx, (question, response) in enumerate(zip(instructions.questions, responses)):
                yield {
                    "dataset": dataset,
                    "variant": variant_name,
                    "prompt_index": None,
                    "question_index": question_idx,
                    "rollout_index": rollout_idx,
                    "question": question,
                    "system_prompt": steering_prompt if steering_include_system else None,
                    "steering_scale": scale,
                    "response": response,
                }
        print(
            f"[{dataset}] {variant_name} rollouts complete: {args.rollouts} rollouts x {len(instructions.questions)} questions."
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/steering_rollouts"))
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/processed/persona"))
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
    parser.add_argument("--minilm-eval", action="store_true", help="Compute MiniLM scores for generated rollouts")
    parser.add_argument("--minilm-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--minilm-batch-size", type=int, default=64)
    parser.add_argument("--minilm-seed", type=int, default=17)
    parser.add_argument("--judge-eval", action="store_true", help="Score responses with persona LLM judge")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-max-tokens", type=int, default=10)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--judge-requests-per-second", type=float, default=10.0)
    parser.add_argument("--normalize-steering", action="store_true", help="L2-normalize steering vectors before scaling")
    parser.add_argument("--steering-scales", type=float, nargs="*", default=[1.0], help="Default scale factors for steering vectors")
    parser.add_argument("--trained-scales", type=float, nargs="*", help="Override scale factors for trained vectors")
    parser.add_argument("--activation-scales", type=float, nargs="*", help="Override scale factors for activation vectors")
    args = parser.parse_args(argv)

    if not args.steering_scales:
        args.steering_scales = [1.0]
    args.steering_scales = [float(s) for s in args.steering_scales]
    args.trained_scales = [float(s) for s in (args.trained_scales if args.trained_scales else args.steering_scales)]
    args.activation_scales = [float(s) for s in (args.activation_scales if args.activation_scales else args.steering_scales)]

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
                instructions = load_instructions(dataset)
            except FileNotFoundError:
                print(f"Missing instructions for {dataset}")
                continue
            except ValueError as exc:
                print(f"Skipping {dataset}: {exc}")
                continue

            print(f"Generating {dataset} -> {output_path}")
            records: list[dict] = []
            with output_path.open("w", encoding="utf-8") as fout:
                for record in generate_variants(
                    dataset,
                    instructions,
                    args,
                    tokenizer,
                    baseline_model,
                    controller,
                    device,
                ):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)

            if args.minilm_eval:
                _run_minilm_eval(dataset, records, args, run_dir)

            if args.judge_eval:
                _run_judge_eval(dataset, instructions, records, args, run_dir)

            if args.minilm_eval or args.judge_eval:
                for rec in records:
                    rec.pop("response", None)
    finally:
        controller.close()
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
