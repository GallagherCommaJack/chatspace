"""Steering vector utilities with optional training helpers."""

from __future__ import annotations

from .activations import load_activation_vector
from .constants import ERROR_FILENAME, SUMMARY_FILENAME
from .runs import collect_run_dirs, has_successful_run, latest_run_dir, list_trained_datasets

__all__ = [
    "ERROR_FILENAME",
    "SUMMARY_FILENAME",
    "collect_run_dirs",
    "has_successful_run",
    "load_activation_vector",
    "latest_run_dir",
    "list_trained_datasets",
]

try:  # pragma: no cover - optional dependencies (datasets, transformers)
    from .data import PersonaSteeringDatasetConfig, load_persona_steering_dataset, prepare_persona_token_budget
    from .model import QwenSteerModel, SteeringVectorConfig
except ImportError:  # pragma: no cover - gracefully degrade when optional deps missing
    pass
else:  # pragma: no cover - executed when optional deps available
    __all__.extend(
        [
            "PersonaSteeringDatasetConfig",
            "load_persona_steering_dataset",
            "prepare_persona_token_budget",
            "SteeringVectorConfig",
            "QwenSteerModel",
        ]
    )
