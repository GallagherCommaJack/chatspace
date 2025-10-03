"""Trainable steering vector module for Qwen models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
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
        self.config = base_model.config  # Expose config for TRL compatibility

        for param in self.model.parameters():
            param.requires_grad_(False)

        hidden_size = base_model.config.hidden_size
        self.steering = ResidualHook(hidden_size, cfg.init_scale)
        self._hook_handle = None
        self._install_hook()

    def _install_hook(self) -> None:
        """Install forward hook to inject steering vector at target layer."""
        layer = self.model.model.layers[self.cfg.target_layer]

        def hook_fn(module, args, output):
            # output is a tuple: (hidden_states, *optional_outputs)
            hidden_states = output[0] if isinstance(output, tuple) else output
            if (
                self.steering.vector.device != hidden_states.device
                or self.steering.vector.dtype != hidden_states.dtype
            ):
                with torch.no_grad():
                    self.steering.vector.data = self.steering.vector.data.to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
            steered = self.steering(hidden_states)
            # Return tuple in same format as input
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        self._hook_handle = layer.register_forward_hook(hook_fn)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Expose generate method for inference."""
        return self.model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Proxy gradient checkpointing to base model."""
        return self.model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        """Proxy gradient checkpointing to base model."""
        return self.model.gradient_checkpointing_disable(*args, **kwargs)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str | Path, **_) -> None:
        """Persist only the steering vector and configuration."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        vector_path = path / "steering_vector.pt"
        torch.save({"steering_vector": self.steering.vector.detach().cpu()}, vector_path)

        config_path = path / "steering_config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self.cfg), fh, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **model_kwargs) -> "QwenSteerModel":
        path = Path(save_directory)
        config_path = path / "steering_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing steering configuration at {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            cfg_dict = json.load(fh)

        cfg = SteeringVectorConfig(**cfg_dict)
        model = cls(cfg, **model_kwargs)

        vector_path = path / "steering_vector.pt"
        if vector_path.exists():
            state = torch.load(vector_path, map_location="cpu")
            tensor = state.get("steering_vector")
            if tensor is None:
                raise ValueError(f"steering_vector.pt missing 'steering_vector' key at {vector_path}")
            model.steering.vector.data.copy_(tensor)

        return model
