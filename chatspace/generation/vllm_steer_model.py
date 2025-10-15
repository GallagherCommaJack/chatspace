"""vLLM-based steerable language model that manages steering inside workers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams

from chatspace.vllm_steering import runtime as steering_runtime

from .base import SteerableModel


@dataclass
class VLLMSteeringConfig:
    """Configuration for vLLM-based steerable model."""

    model_name: str = "Qwen/Qwen3-32B"
    target_layer: int = 22
    init_scale: float = 0.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str.startswith("torch."):
        raise ValueError(f"Unexpected dtype format: {dtype_str}")
    name = dtype_str.split(".", maxsplit=1)[1]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype


class VLLMSteerModel(SteerableModel):
    """Wrap a vLLM LLM with steering vector control via worker RPCs."""

    def __init__(self, cfg: VLLMSteeringConfig, **vllm_kwargs) -> None:
        self.cfg = cfg

        llm_kwargs = {
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "dtype": cfg.dtype,
        }
        if cfg.max_model_len is not None:
            llm_kwargs["max_model_len"] = cfg.max_model_len
        llm_kwargs.update(vllm_kwargs)

        self.llm = LLM(model=cfg.model_name, **llm_kwargs)
        self._engine_client = self.llm.llm_engine.engine_core

        setup_info = self._engine_client.collective_rpc(
            steering_runtime.initialize_worker_state,
            args=(int(cfg.target_layer), float(cfg.init_scale)),
        )
        if not setup_info:
            raise RuntimeError("Failed to initialize steering state on workers.")

        first = setup_info[0]
        self.hidden_size = int(first["hidden_size"])
        self._vector_dtype = _parse_dtype(first["dtype"])
        self._cached_vector = torch.zeros(self.hidden_size, dtype=self._vector_dtype)

    def _broadcast_vector(self, vector: torch.Tensor) -> None:
        vec = vector.to(dtype=self._vector_dtype, device="cpu").contiguous()
        payload = steering_runtime.serialize_tensor(vec)
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_vector, args=(payload,)
        )
        self._cached_vector = vec.clone()

    def _broadcast_clear(self) -> None:
        self._engine_client.collective_rpc(steering_runtime.clear_worker_vector)
        self._cached_vector.zero_()

    def set_target_layer(self, layer_idx: int) -> None:
        layer_idx = int(layer_idx)
        if layer_idx == self.cfg.target_layer:
            return
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_layer, args=(layer_idx,)
        )
        self.cfg.target_layer = layer_idx

    def set_vector(self, vector: torch.Tensor | None) -> None:
        if vector is None:
            self._broadcast_clear()
            return

        vec = vector.detach().view(-1)
        if vec.shape[0] != self.hidden_size:
            raise ValueError(
                f"Steering vector shape mismatch: expected {(self.hidden_size,)}, got {tuple(vec.shape)}"
            )
        self._broadcast_vector(vec)

    def current_vector(self) -> torch.Tensor:
        """Return a CPU copy of the last vector broadcast to workers."""
        return self._cached_vector.clone()

    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    def save_pretrained(self, save_directory: str | Path, **_) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        vector_path = path / "steering_vector.pt"
        torch.save({"steering_vector": self._cached_vector.clone()}, vector_path)

        config_path = path / "steering_config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self.cfg), fh, indent=2)

    @classmethod
    def from_pretrained(
        cls, save_directory: str | Path, **vllm_kwargs: Any
    ) -> "VLLMSteerModel":
        path = Path(save_directory)
        config_path = path / "steering_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing steering configuration at {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            cfg_dict = json.load(fh)

        cfg = VLLMSteeringConfig(**cfg_dict)
        model = cls(cfg, **vllm_kwargs)

        vector_path = path / "steering_vector.pt"
        if vector_path.exists():
            state = torch.load(vector_path, map_location="cpu")
            tensor = state.get("steering_vector")
            if tensor is None:
                raise ValueError(
                    f"steering_vector.pt missing 'steering_vector' key at {vector_path}"
                )
            model.set_vector(tensor)
        return model

    # ------------------------------------------------------------------
    # Internal debugging helpers (used in tests)
    # ------------------------------------------------------------------

    def _fetch_worker_vectors(self) -> list[torch.Tensor]:
        """Retrieve current worker vectors for validation."""
        payloads = self._engine_client.collective_rpc(
            steering_runtime.fetch_worker_vector
        )
        return [
            steering_runtime.deserialize_tensor(
                payload, device=torch.device("cpu"), dtype=self._vector_dtype
            )
            for payload in payloads
        ]
