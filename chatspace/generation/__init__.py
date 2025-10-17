"""Text generation utilities with steering vector support."""

from .base import SteerableModel
from .compat import LegacyExperiment, load_legacy_role_trait_config
from .config import GenerationConfig
from .vllm_steer_model import (
    AddSpec,
    AblationSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
    VLLMSteerModel,
    VLLMSteeringConfig,
)

__all__ = [
    "SteerableModel",
    "GenerationConfig",
    "VLLMSteerModel",
    "VLLMSteeringConfig",
    "AddSpec",
    "LayerSteeringSpec",
    "ProjectionCapSpec",
    "AblationSpec",
    "SteeringSpec",
    "LegacyExperiment",
    "load_legacy_role_trait_config",
]
