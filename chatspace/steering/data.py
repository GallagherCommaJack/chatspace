"""Persona dataset utilities for steering vector training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk


@dataclass
class PersonaSteeringDatasetConfig:
    """Configuration for assembling persona datasets for steering training."""

    dataset_names: List[str]
    dataset_root: Path = Path("/workspace/datasets/processed/persona")
    target_tokens: int = 100_000
    seed: int = 17
    tokenizer_name: str = "Qwen/Qwen3-32B"
    max_length: int = 4096
    role_min_score: int = 3
    trait_min_score: int = 75


def _parse_dataset_spec(name: str) -> Tuple[str, Dict[str, str]]:
    if ":" not in name:
        return name, {}
    dataset, spec = name.split(":", 1)
    spec = spec.strip()
    if not spec:
        return dataset, {}
    if "=" in spec:
        key, value = spec.split("=", 1)
    else:
        key, value = "role", spec
    return dataset, {key.strip(): value.strip()}


def _load_split(name: str, root: Path) -> Dataset:
    dataset_name, filters = _parse_dataset_spec(name)
    dir_path = root / dataset_name
    if dir_path.exists():
        dataset = load_from_disk(str(dir_path))["train"]
    else:
        parquet_path = dir_path.with_suffix(".parquet")
        if parquet_path.exists():
            dataset = load_dataset("parquet", data_files=str(parquet_path))["train"]
        else:
            raise FileNotFoundError(f"Dataset {name} not found under {root}")

    role_filter = filters.get("role")
    if role_filter is not None:
        dataset = dataset.filter(lambda row: row.get("role") == role_filter)

    return dataset


def _example_passes(row: Mapping[str, object], cfg: PersonaSteeringDatasetConfig) -> bool:
    raw_score = row.get("extract_score", 0)
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        # Datasets without scores (e.g., default/baseline responses) should be filtered out
        # when score-based filtering is enabled, as they are control conditions.
        return False
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

 
