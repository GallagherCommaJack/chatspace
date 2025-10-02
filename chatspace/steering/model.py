"""Trainable steering vector module for Qwen models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM


class ResidualHook(nn.Module):
    """A module that injects a trainable vector into the residual stream."""

    def __init__(self, hidden_size: int, init_scale: float = 0.01) -> None:
        super().__init__()
        self.vector = nn.Parameter(torch.randn(hidden_size) * init_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.vector


@dataclass
class SteeringVectorConfig:
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    target_layer: int = 22
    init_scale: float = 0.01


class QwenSteerModel(nn.Module):
    """Wrap a Qwen causal LM with an additive steering vector at a residual layer."""

    def __init__(self, cfg: SteeringVectorConfig, **model_kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        self.model = base_model

        for param in self.model.parameters():
            param.requires_grad_(False)

        hidden_size = base_model.config.hidden_size
        self.steering = ResidualHook(hidden_size, cfg.init_scale)

        layer = self.model.model.layers[cfg.target_layer]
        original_forward = layer.forward

        def hooked_forward(*args, **kwargs):
            hidden_states = original_forward(*args, **kwargs)
            return self.steering(hidden_states)

        layer.forward = hooked_forward

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


