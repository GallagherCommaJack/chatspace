"""Abstract base class for steerable language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class SteerableModel(ABC):
    """Abstract interface for language models that support steering vector injection.

    Steering is performed by adding a learned or activation-based vector to
    the residual stream at a specified transformer layer during inference.
    """

    @abstractmethod
    def set_vector(self, vector: torch.Tensor | None) -> None:
        """Set the steering vector for inference.

        Parameters
        ----------
        vector : torch.Tensor | None
            Steering vector to inject at the target layer. If None, clears
            the steering (sets to zero). Shape should match model's hidden_size.
        """
        pass

    @abstractmethod
    def set_target_layer(self, layer_idx: int) -> None:
        """Change the target layer for steering vector injection.

        Parameters
        ----------
        layer_idx : int
            Index of the transformer layer where the steering vector
            should be injected into the residual stream.
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Generate text with the current steering configuration.

        The signature and return type depend on the underlying model
        implementation (HuggingFace Transformers vs vLLM).
        """
        pass
