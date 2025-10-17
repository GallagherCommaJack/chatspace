"""vLLM-based steerable language model that manages steering inside workers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence, cast
import logging

import torch
from vllm import LLM, SamplingParams

from chatspace.vllm_steering import runtime as steering_runtime


logger = logging.getLogger(__name__)

@dataclass
class VLLMSteeringConfig:
    """Configuration for vLLM-based steerable model."""

    model_name: str = "Qwen/Qwen3-32B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"
    bootstrap_layers: tuple[int, ...] = ()


@dataclass
class SteeringSpec:
    """Layer steering vector parameters."""

    vector: torch.Tensor


@dataclass
class ProjectionCapSpec:
    """Layer projection capping parameters."""

    vector: torch.Tensor
    cap_below: float | None = None
    cap_above: float | None = None


@dataclass
class AblationSpec:
    """Layer ablation parameters."""

    vector: torch.Tensor
    scale: float = 1.0


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str.startswith("torch."):
        raise ValueError(f"Unexpected dtype format: {dtype_str}")
    name = dtype_str.split(".", maxsplit=1)[1]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype


class VLLMSteerModel:
    """Wrap a vLLM LLM with steering vector control via worker RPCs."""

    def __init__(
        self,
        cfg: VLLMSteeringConfig,
        *,
        bootstrap_layers: Sequence[int] | None = None,
        **vllm_kwargs,
    ) -> None:
        self.cfg = cfg

        enforce_eager = bool(vllm_kwargs.get("enforce_eager", True))

        llm_kwargs = {
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "dtype": cfg.dtype,
            "enforce_eager": enforce_eager,
        }
        if cfg.max_model_len is not None:
            llm_kwargs["max_model_len"] = cfg.max_model_len
        llm_kwargs.update(vllm_kwargs)

        steering_runtime.ensure_layer_patch_installed()
        self.llm = LLM(model=cfg.model_name, **llm_kwargs)
        if not enforce_eager:
            logger.warning(
                "vLLM steering currently requires enforce_eager=True to apply layer hooks."
            )
        self._engine_client = self.llm.llm_engine.engine_core

        init_layers: tuple[int, ...]
        if bootstrap_layers is not None:
            init_layers = tuple(int(idx) for idx in bootstrap_layers)
        else:
            init_layers = tuple(int(idx) for idx in cfg.bootstrap_layers)

        setup_info = self._engine_client.collective_rpc(
            steering_runtime.initialize_worker_state,
            args=(init_layers,),
        )
        if not setup_info:
            raise RuntimeError("Failed to initialize steering state on workers.")

        first = setup_info[0]
        self.hidden_size = int(first["hidden_size"])
        self._vector_dtype = _parse_dtype(first["dtype"])
        layer_count = int(first["layer_count"])
        self.layer_count = layer_count
        self._steering_specs: dict[int, SteeringSpec] = {}
        self._projection_caps: dict[int, ProjectionCapSpec] = {}
        self._ablations: dict[int, AblationSpec] = {}
        for idx in init_layers:
            if idx < 0 or idx >= layer_count:
                raise ValueError(
                    f"bootstrap layer {idx} is out of range for model with {layer_count} layers"
                )
            self._ensure_cached_layer(idx)

    def _ensure_cached_layer(self, layer_idx: int) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.layer_count:
            raise ValueError(
                f"Layer index {layer_idx} out of range for model with {self.layer_count} layers"
            )
        spec = self._steering_specs.get(layer_idx)
        if spec is None:
            vector = torch.zeros(self.hidden_size, dtype=self._vector_dtype)
            spec = SteeringSpec(vector=vector)
            self._steering_specs[layer_idx] = spec
        return spec.vector

    def _prepare_layer_vector(self, vector: torch.Tensor, *, context: str) -> torch.Tensor:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{context} requires a torch.Tensor input.")
        vec = vector.detach().view(-1)
        if vec.shape[0] != self.hidden_size:
            raise ValueError(
                f"{context} shape mismatch: expected {(self.hidden_size,)}, got {tuple(vec.shape)}"
            )
        return vec.to(dtype=self._vector_dtype, device="cpu").contiguous()

    def _prepare_direction_vector(self, vector: torch.Tensor, *, context: str) -> torch.Tensor:
        prepared = self._prepare_layer_vector(vector, context=context)
        norm = float(prepared.norm().item())
        if norm <= 0:
            raise ValueError(f"{context} vector must have non-zero norm.")
        return prepared

    def _broadcast_vector(self, layer_idx: int, vector: torch.Tensor) -> None:
        payload = steering_runtime.serialize_tensor(vector)
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_vector,
            args=(int(layer_idx), payload),
        )
        self._steering_specs[int(layer_idx)] = SteeringSpec(vector=vector.clone())

    def _broadcast_projection_cap(self, layer_idx: int, spec: ProjectionCapSpec) -> None:
        payload = {
            "vector": steering_runtime.serialize_tensor(spec.vector),
            "cap_below": spec.cap_below,
            "cap_above": spec.cap_above,
        }
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_projection_cap,
            args=(int(layer_idx), payload),
        )
        self._projection_caps[int(layer_idx)] = ProjectionCapSpec(
            vector=spec.vector.clone(),
            cap_below=spec.cap_below,
            cap_above=spec.cap_above,
        )

    def _broadcast_ablation(self, layer_idx: int, spec: AblationSpec) -> None:
        payload = {
            "vector": steering_runtime.serialize_tensor(spec.vector),
            "scale": float(spec.scale),
        }
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_ablation,
            args=(int(layer_idx), payload),
        )
        self._ablations[int(layer_idx)] = AblationSpec(
            vector=spec.vector.clone(),
            scale=float(spec.scale),
        )

    def _broadcast_clear(self, layer_idx: int | None = None) -> None:
        if layer_idx is None:
            self._engine_client.collective_rpc(steering_runtime.clear_worker_vector)
            self._engine_client.collective_rpc(steering_runtime.clear_worker_projection_cap)
            self._engine_client.collective_rpc(steering_runtime.clear_worker_ablation)
            for spec in self._steering_specs.values():
                spec.vector.zero_()
            self._projection_caps.clear()
            self._ablations.clear()
        else:
            target = int(layer_idx)
            self._engine_client.collective_rpc(
                steering_runtime.clear_worker_vector, args=(target,)
            )
            self._engine_client.collective_rpc(
                steering_runtime.clear_worker_projection_cap, args=(target,)
            )
            self._engine_client.collective_rpc(
                steering_runtime.clear_worker_ablation, args=(target,)
            )
            self._ensure_cached_layer(target).zero_()
            self._projection_caps.pop(target, None)
            self._ablations.pop(target, None)

    def set_layer_vector(
        self, layer_idx: int, vector: torch.Tensor | None
    ) -> None:
        layer_idx = int(layer_idx)
        if vector is None:
            self._broadcast_clear(layer_idx)
            return

        prepared = self._prepare_layer_vector(vector, context="Steering vector update")
        self._broadcast_vector(layer_idx, prepared)

    def set_layer_projection_cap(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        *,
        cap_below: float | None = None,
        cap_above: float | None = None,
    ) -> None:
        target = int(layer_idx)
        self._ensure_cached_layer(target)
        if cap_below is None and cap_above is None:
            raise ValueError("Projection cap requires at least one bound.")
        lower = float(cap_below) if cap_below is not None else None
        upper = float(cap_above) if cap_above is not None else None
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("cap_below cannot exceed cap_above.")
        prepared = self._prepare_direction_vector(
            vector, context="Projection cap vector"
        )
        spec = ProjectionCapSpec(vector=prepared, cap_below=lower, cap_above=upper)
        self._broadcast_projection_cap(target, spec)

    def clear_layer_projection_cap(self, layer_idx: int) -> None:
        target = int(layer_idx)
        self._engine_client.collective_rpc(
            steering_runtime.clear_worker_projection_cap, args=(target,)
        )
        self._projection_caps.pop(target, None)

    def current_projection_cap(self, layer_idx: int) -> ProjectionCapSpec | None:
        spec = self._projection_caps.get(int(layer_idx))
        if spec is None:
            return None
        return ProjectionCapSpec(
            vector=spec.vector.clone(),
            cap_below=spec.cap_below,
            cap_above=spec.cap_above,
        )

    def set_layer_ablation(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        scale: float,
    ) -> None:
        target = int(layer_idx)
        self._ensure_cached_layer(target)
        if not isinstance(scale, (int, float)):
            raise TypeError("Ablation scale must be a numeric value.")
        scale_value = float(scale)
        if not math.isfinite(scale_value):
            raise ValueError("Ablation scale must be finite.")
        prepared = self._prepare_direction_vector(
            vector, context="Ablation vector"
        )
        spec = AblationSpec(vector=prepared, scale=scale_value)
        self._broadcast_ablation(target, spec)

    def clear_layer_ablation(self, layer_idx: int) -> None:
        target = int(layer_idx)
        self._engine_client.collective_rpc(
            steering_runtime.clear_worker_ablation, args=(target,)
        )
        self._ablations.pop(target, None)

    def current_ablation(self, layer_idx: int) -> AblationSpec | None:
        spec = self._ablations.get(int(layer_idx))
        if spec is None:
            return None
        return AblationSpec(vector=spec.vector.clone(), scale=spec.scale)

    def current_vector(self, layer_idx: int | None = None) -> torch.Tensor:
        """Return a CPU copy of the last vector broadcast to workers."""
        if layer_idx is None:
            if not self._steering_specs:
                raise ValueError("No steering vectors have been cached yet.")
            if len(self._steering_specs) != 1:
                raise ValueError(
                    "Multiple layer vectors cached. Provide layer_idx explicitly."
                )
            layer_idx = next(iter(self._steering_specs.keys()))
        return self._ensure_cached_layer(int(layer_idx)).clone()

    def clear_layer_vector(self, layer_idx: int) -> None:
        self.set_layer_vector(layer_idx, None)

    def clear_all_vectors(self) -> None:
        self._broadcast_clear()

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

    def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        prefill_assistant: str | bool | None = None,
        **sampling_kwargs: Any,
    ) -> list[str]:
        """Execute chat-style generation with optional sampling overrides.

        Parameters
        ----------
        messages : list[dict[str, Any]] | list[list[dict[str, Any]]]
            Conversation messages using the OpenAI-style schema. A single
            conversation may be provided (list of messages) or a batch of
            conversations (list of conversation lists).
        sampling_params : SamplingParams | None
            Optional sampling configuration. If omitted, ``sampling_kwargs``
            are used to instantiate a ``SamplingParams`` object.
        use_tqdm : bool, default False
            Whether to display the progress bar during generation.
        chat_options : dict[str, Any] | None
            Additional keyword arguments forwarded to ``LLM.chat`` (for
            example ``chat_template`` or ``add_generation_prompt``).
        prefill_assistant : str | bool | None, default None
            Optional prefix inserted as the final assistant message before
            generation. When set to ``True`` the helper injects an empty
            ``<think></think>`` block compatible with the hybrid chat template.
            String inputs allow custom prefixes; any whitespace-only think blocks
            are normalized to match template formatting. The helper automatically
            strips the prefix (including the sentinel) from returned outputs. Set
            to ``None`` or ``False`` to disable prefilling.
        **sampling_kwargs : Any
            Keyword arguments used to build a ``SamplingParams`` instance when
            ``sampling_params`` is not supplied.
        """
        if sampling_params is None:
            sampling_params = SamplingParams(**sampling_kwargs)
        elif sampling_kwargs:
            raise ValueError(
                "Provide either sampling_params or sampling keyword overrides, not both."
            )

        single_conversation = (
            isinstance(messages, list)
            and (len(messages) == 0 or isinstance(messages[0], dict))
        )
        if single_conversation:
            batched_messages = [cast(list[dict[str, Any]], messages)]
        else:
            batched_messages = cast(list[list[dict[str, Any]]], messages)

        prepared_messages: list[list[dict[str, Any]]]
        chat_kwargs = dict(chat_options or {})
        chat_kwargs.setdefault("chat_template_content_format", "string")

        sentinel = "ASSISTANT_PREFILL:"
        trim_prefix: str | None = None

        def _normalize_prefill(raw: str) -> str:
            stripped = raw.strip()
            if stripped.startswith("<think>") and "</think>" in stripped:
                head, tail = stripped.split("</think>", 1)
                inside = head[len("<think>") :].strip()
                if not inside:
                    tail = tail.lstrip("\n")
                    return "<think>\n\n</think>\n\n" + tail
            return raw

        prefill_base: str | None
        if isinstance(prefill_assistant, bool):
            prefill_base = "<think>\n\n</think>\n\n" if prefill_assistant else None
        elif isinstance(prefill_assistant, str):
            prefill_base = _normalize_prefill(prefill_assistant)
        else:
            prefill_base = None

        if prefill_base is not None:
            prefill_payload = f"{prefill_base}{sentinel}"
            trim_prefix = prefill_payload
            if chat_kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "Cannot prefill assistant when add_generation_prompt=True. "
                    "Disable prefilling or override chat_options."
                )
            if chat_kwargs.get("continue_final_message") is False:
                raise ValueError(
                    "Cannot prefill assistant when continue_final_message=False. "
                    "Disable prefilling or override chat_options."
                )
            chat_kwargs.setdefault("add_generation_prompt", False)
            chat_kwargs.setdefault("continue_final_message", True)

            prepared_messages = []
            for conv in batched_messages:
                conv_copy = [dict(msg) for msg in conv]
                conv_copy.append({"role": "assistant", "content": prefill_payload})
                prepared_messages.append(conv_copy)
        elif prefill_assistant not in (None, False):
            raise TypeError("prefill_assistant must be a string, boolean, or None.")
        else:
            prepared_messages = batched_messages

        outputs = self.llm.chat(
            prepared_messages,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            **chat_kwargs,
        )
        texts: list[str] = []
        for output in outputs:
            text = output.outputs[0].text
            if trim_prefix and text.startswith(trim_prefix):
                text = text[len(trim_prefix) :].lstrip()
            texts.append(text)
        return texts

    def save_pretrained(self, save_directory: str | Path, **_) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        vector_path = path / "steering_vector.pt"
        serialized_vectors = {
            int(layer_idx): spec.vector.detach().cpu().clone()
            for layer_idx, spec in self._steering_specs.items()
        }
        serialized_caps = {
            int(layer_idx): {
                "vector": spec.vector.detach().cpu().clone(),
                "cap_below": spec.cap_below,
                "cap_above": spec.cap_above,
            }
            for layer_idx, spec in self._projection_caps.items()
        }
        serialized_ablations = {
            int(layer_idx): {
                "vector": spec.vector.detach().cpu().clone(),
                "scale": float(spec.scale),
            }
            for layer_idx, spec in self._ablations.items()
        }
        torch.save(
            {
                "layer_vectors": serialized_vectors,
                "steering_vector": (
                    next(iter(serialized_vectors.values())).clone()
                    if serialized_vectors
                    else None
                ),
                "projection_caps": serialized_caps,
                "ablations": serialized_ablations,
            },
            vector_path,
        )

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

        allowed_keys = {f.name for f in fields(VLLMSteeringConfig)}
        filtered_cfg = {key: value for key, value in cfg_dict.items() if key in allowed_keys}
        cfg = VLLMSteeringConfig(**filtered_cfg)

        model = cls(cfg, **vllm_kwargs)

        vector_path = path / "steering_vector.pt"
        if vector_path.exists():
            state = torch.load(vector_path, map_location="cpu")
            layer_vectors = state.get("layer_vectors")
            if isinstance(layer_vectors, dict) and layer_vectors:
                for layer_idx, tensor in layer_vectors.items():
                    model.set_layer_vector(int(layer_idx), tensor)
            else:
                tensor = state.get("steering_vector")
                if tensor is None:
                    raise ValueError(
                        f"steering_vector.pt missing steering data at {vector_path}"
                    )
                model.set_layer_vector(0, tensor)
            projection_caps = state.get("projection_caps") or {}
            if isinstance(projection_caps, dict):
                for layer_idx, payload in projection_caps.items():
                    vector = payload.get("vector")
                    if vector is None:
                        continue
                    model.set_layer_projection_cap(
                        int(layer_idx),
                        vector,
                        cap_below=payload.get("cap_below"),
                        cap_above=payload.get("cap_above"),
                    )
            ablations = state.get("ablations") or {}
            if isinstance(ablations, dict):
                for layer_idx, payload in ablations.items():
                    vector = payload.get("vector")
                    scale = payload.get("scale")
                    if vector is None or scale is None:
                        continue
                    model.set_layer_ablation(
                        int(layer_idx),
                        vector,
                        float(scale),
                    )
        return model

    # ------------------------------------------------------------------
    # Internal debugging helpers (used in tests)
    # ------------------------------------------------------------------

    def _fetch_worker_vectors(self) -> list[dict[int, torch.Tensor]]:
        """Retrieve current worker vectors for validation."""
        payloads = self._engine_client.collective_rpc(
            steering_runtime.fetch_worker_vectors
        )
        worker_vectors: list[dict[int, torch.Tensor]] = []
        for payload in payloads:
            worker_map: dict[int, torch.Tensor] = {}
            for layer_idx, tensor in payload.items():
                worker_map[int(layer_idx)] = (
                    tensor.detach().to(dtype=self._vector_dtype).cpu().clone()
                )
            worker_vectors.append(worker_map)
        return worker_vectors
