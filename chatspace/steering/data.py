"""Persona dataset utilities for steering vector training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk


@dataclass
class PersonaSteeringDatasetConfig:
    """Configuration for assembling persona datasets for steering training."""

    dataset_names: List[str]
    dataset_root: Path = Path("/workspace/datasets/processed/persona")
    target_tokens: int = 100_000
    seed: int = 17
    tokenizer_name: str = "Qwen/Qwen2.5-32B-Instruct"
    max_length: int = 4096
    role_min_score: int = 3
    trait_min_score: int = 75


def _load_split(name: str, root: Path) -> Dataset:
    dir_path = root / name
    if dir_path.exists():
        return load_from_disk(str(dir_path))["train"]
    parquet_path = dir_path.with_suffix(".parquet")
    if parquet_path.exists():
        return load_dataset("parquet", data_files=str(parquet_path))["train"]
    raise FileNotFoundError(f"Dataset {name} not found under {root}")


def _example_passes(row: Mapping[str, object], cfg: PersonaSteeringDatasetConfig) -> bool:
    score = int(row.get("extract_score", 0))
    if "role" in row:
        return score >= cfg.role_min_score
    if "trait" in row:
        label = row.get("label")
        return label == "pos" and score >= cfg.trait_min_score
    return False


def _strip_system(messages: List[Mapping[str, str]]) -> List[Mapping[str, str]]:
    return [m for m in messages if m.get("role") != "system"]


def load_persona_steering_dataset(cfg: PersonaSteeringDatasetConfig, tokenizer) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    records = []

    for name in cfg.dataset_names:
        ds = _load_split(name, cfg.dataset_root)
        for row in ds:
            if not _example_passes(row, cfg):
                continue
            messages = _strip_system(row["messages"])
            if not messages:
                continue
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(
                chat,
                truncation=True,
                max_length=cfg.max_length,
                return_attention_mask=True,
            )
            length = len(tokenized["input_ids"])
            records.append({
                "messages": messages,
                "length": length,
                "source_dataset": name,
            })

    if not records:
        raise ValueError("No examples found matching the filters")

    rng.shuffle(records)

    selected = []
    total_tokens = 0
    for record in records:
        if total_tokens >= cfg.target_tokens:
            break
        selected.append(record)
        total_tokens += record["length"]

    if not selected:
        raise ValueError("Unable to reach target tokens with provided datasets")

    dataset = Dataset.from_list(selected)
    return dataset

 
