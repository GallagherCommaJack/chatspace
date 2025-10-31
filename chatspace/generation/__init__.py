"""Text generation utilities with steering vector support."""

from .base import SteerableModel
from .compat import LegacyExperiment, load_legacy_role_trait_config
from .config import GenerationConfig
from .vllm_steer_model import (
    AddSpec,
    AblationSpec,
    CaptureHandle,
    LayerSteeringSpec,
    MessageBoundary,
    ProjectionCapSpec,
    SteeringSpec,
    VLLMSteerModel,
    VLLMSteeringConfig,
    compute_message_boundaries,
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
    "CaptureHandle",
    "MessageBoundary",
    "compute_message_boundaries",
    "LegacyExperiment",
    "load_legacy_role_trait_config",
]
