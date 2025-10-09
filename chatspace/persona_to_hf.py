"""
Convert persona-subspace data to HuggingFace datasets.

These legacy helpers now delegate to :mod:`chatspace.persona` so downstream scripts
can keep importing the existing API while sharing the centralised implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from datasets import Dataset

from .persona import (
    DEFAULT_PERSONA_DATA_ROOT,
    list_available_roles as _list_available_roles,
    list_available_traits as _list_available_traits,
    load_persona_dataset,
    save_persona_dataset,
)


def load_single_role_conversations(
    model_name: str,
    role_name: str,
    persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT,
    min_score: Optional[int] = None,
) -> Dataset:
    """Load conversations for a single role into a Hugging Face dataset."""
    return load_persona_dataset(
        model=model_name,
        dataset_type="role",
        name=role_name,
        persona_data_dir=persona_data_dir,
        min_score=min_score,
    )


def load_single_trait_conversations(
    model_name: str,
    trait_name: str,
    persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT,
    min_score: Optional[int] = None,
    label_filter: Optional[Literal["pos", "neg"]] = None,
) -> Dataset:
    """Load conversations for a single trait into a Hugging Face dataset."""
    return load_persona_dataset(
        model=model_name,
        dataset_type="trait",
        name=trait_name,
        persona_data_dir=persona_data_dir,
        min_score=min_score,
        label_filter=label_filter,
    )


def list_available_roles(
    model_name: str,
    persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT,
) -> list[str]:
    """List all available roles for a model."""
    return _list_available_roles(model=model_name, persona_data_dir=persona_data_dir)


def list_available_traits(
    model_name: str,
    persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT,
) -> list[str]:
    """List all available traits for a model."""
    return _list_available_traits(model=model_name, persona_data_dir=persona_data_dir)


def save_dataset(
    dataset: Dataset,
    output_dir: Path,
    name: str,
) -> None:
    """Save dataset to disk in Hugging Face and Parquet formats."""
    save_persona_dataset(dataset, output_dir, name)
    dataset_path = output_dir / name
    parquet_path = output_dir / f"{name}.parquet"
    print(f"Saved dataset to {dataset_path}")
    print(f"Saved parquet to {parquet_path}")
