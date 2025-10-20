"""Steering helpers for coordinating vLLM worker state with chatspace.

The helpers in this module wrap ``vllm.LLM`` so steering vectors can be injected
into Qwen-style decoder layers without breaking CUDA graph capture.  The primary
entry point is :class:`VLLMSteerModel`, which mirrors the
``SteerableModel`` interface used by the HuggingFace implementation:

Typical usage::

    cfg = VLLMSteeringConfig(model_name="Qwen/Qwen3-0.6B")
    model = VLLMSteerModel(cfg, bootstrap_layers=(target_layer,))
    model.set_target_layer(target_layer)
    model.set_vector(torch.zeros(model.hidden_size))
    outputs = model.generate(["...prompt..."], sampling_params)

``VLLMSteerModel`` internally broadcasts steering updates to every worker via
collective RPCs.  ``enforce_eager=True`` should remain enabled unless you have
verified the steering patch still executes inside compiled graphs.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator, Sequence, cast
import logging

import torch
from vllm import LLM, SamplingParams

from chatspace.vllm_steering import runtime as steering_runtime
from .base import SteerableModel


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
class AddSpec:
    """Describe an additive steering vector with its magnitude.

    Parameters
    ----------
    vector :
        L2-normalised direction stored on CPU in ``float32`` for numerical
        stability.  The norm is maintained separately via ``scale`` so the
        direction can be reused.
    scale :
        Magnitude applied to the direction when broadcasting to workers.
    The helper L2-normalises ``vector`` when constructing the spec; ``scale``
    therefore captures the original norm.  Supplying ``scale=0`` represents a
    cleared steering vector.
    """

    vector: torch.Tensor
    scale: float = 1.0

    def clone(self) -> "AddSpec":
        return AddSpec(vector=self.vector.detach().clone(), scale=self.scale)

    def materialize(self) -> torch.Tensor:
        """Return the scaled steering vector."""
        return (self.vector * self.scale).contiguous()


@dataclass
class ProjectionCapSpec:
    """Describe projection capping applied after steering is injected.

    Parameters
    ----------
    vector :
        Direction (unit vector) used to measure the hidden-state component that
        should be clamped.  The helper L2-normalises the provided tensor and
        stores it in ``float32`` when constructing the spec.
    min :
        Optional minimum bound for that component.  ``None`` leaves the lower
        side unconstrained.
    max :
        Optional maximum bound for that component.  ``None`` leaves the upper
        side unconstrained.
    """

    vector: torch.Tensor
    min: float | None = None
    max: float | None = None

    def clone(self) -> "ProjectionCapSpec":
        return ProjectionCapSpec(
            vector=self.vector.detach().clone(),
            min=self.min,
            max=self.max,
        )


@dataclass
class AblationSpec:
    """Describe multiplicative ablation along a residual direction.

    Parameters
    ----------
    vector :
        Direction to project onto before rescaling.  The helper L2-normalises
        this tensor and stores it in ``float32`` when constructing the spec.
    scale :
        Multiplicative factor applied to the projected component.  Values under
        ``1.0`` diminish the component; values over ``1.0`` amplify it.
    """

    vector: torch.Tensor
    scale: float = 1.0

    def clone(self) -> "AblationSpec":
        return AblationSpec(vector=self.vector.detach().clone(), scale=self.scale)


@dataclass
class LayerSteeringSpec:
    """All steering controls for a single transformer layer.

    Parameters
    ----------
    add :
        Optional :class:`AddSpec` describing the additive steering vector.
    projection_cap :
        Optional :class:`ProjectionCapSpec` applied after the steering addition.
    ablation :
        Optional :class:`AblationSpec` applied after the steering addition.
    """

    add: AddSpec | None = None
    projection_cap: ProjectionCapSpec | None = None
    ablation: AblationSpec | None = None

    def clone(self) -> "LayerSteeringSpec":
        return LayerSteeringSpec(
            add=self.add.clone() if self.add else None,
            projection_cap=self.projection_cap.clone() if self.projection_cap else None,
            ablation=self.ablation.clone() if self.ablation else None,
        )

    def is_empty(self) -> bool:
        add_active = False
        if self.add is not None:
            scale = float(self.add.scale)
            add_active = math.isfinite(scale) and not math.isclose(scale, 0.0, rel_tol=0.0, abs_tol=1e-12)
        return (not add_active) and self.projection_cap is None and self.ablation is None


@dataclass
class SteeringSpec:
    """Bundle steering metadata for multiple layers.

    Parameters
    ----------
    layers :
        Mapping of layer indices to :class:`LayerSteeringSpec` instances.
    """

    layers: dict[int, LayerSteeringSpec] = field(default_factory=dict)

    def clone(self) -> "SteeringSpec":
        return SteeringSpec(
            layers={layer: spec.clone() for layer, spec in self.layers.items()}
        )

    def is_empty(self) -> bool:
        return all(spec.is_empty() for spec in self.layers.values())


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str.startswith("torch."):
        raise ValueError(f"Unexpected dtype format: {dtype_str}")
    name = dtype_str.split(".", maxsplit=1)[1]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype


class VLLMSteerModel(SteerableModel):
    """Steerable wrapper around ``vllm.LLM`` for Qwen-family models.

    The wrapper keeps a small cache of per-layer steering metadata and mirrors
    the :class:`chatspace.generation.base.SteerableModel` contract so higher
    level pipeline code can swap between HuggingFace and vLLM backends.  When a
    steering vector is updated we serialize the payload, broadcast it to all
    worker processes with ``collective_rpc`` and leave a CPU copy in
    ``_layer_specs`` for quick inspection.

    Parameters
    ----------
    cfg :
        High-level configuration for vLLM (model name, tensor parallel size,
        bootstrap layers, etc).
    bootstrap_layers :
        Optional explicit list of layer indices that should be pre-initialised.
        Supplying a layer ensures the worker patches allocate buffers before
        the first call to :meth:`set_vector`.
    **vllm_kwargs :
        Extra keyword arguments forwarded to ``vllm.LLM``.  ``enforce_eager``
        defaults to ``True`` and attempts to disable it are overridden because
        compiled graphs would otherwise skip the Python-side steering hook.
    """

    def __init__(
        self,
        cfg: VLLMSteeringConfig,
        *,
        bootstrap_layers: Sequence[int] | None = None,
        **vllm_kwargs,
    ) -> None:
        self.cfg = cfg

        enforce_eager_raw = vllm_kwargs.get("enforce_eager", True)
        enforce_eager = bool(enforce_eager_raw)
        if not enforce_eager:
            logger.warning(
                "vLLM steering requires enforce_eager=True; overriding user-supplied value."
            )
            enforce_eager = True

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
        self._layer_specs: dict[int, LayerSteeringSpec] = {}
        self._steering_stack: list[SteeringSpec] = []
        for idx in init_layers:
            if idx < 0 or idx >= layer_count:
                raise ValueError(
                    f"bootstrap layer {idx} is out of range for model with {layer_count} layers"
                )
            self._ensure_layer_spec(idx)
        self._active_layer: int | None = None
        if init_layers:
            self.set_target_layer(init_layers[0])
        elif layer_count > 0:
            self.set_target_layer(0)

    def _ensure_layer_spec(self, layer_idx: int) -> LayerSteeringSpec:
        idx = int(layer_idx)
        if idx < 0 or idx >= self.layer_count:
            raise ValueError(
                f"Layer index {idx} out of range for model with {self.layer_count} layers"
            )
        spec = self._layer_specs.get(idx)
        if spec is None:
            spec = LayerSteeringSpec()
            self._layer_specs[idx] = spec
        return spec

    def _prune_layer_entry(self, layer_idx: int) -> None:
        idx = int(layer_idx)
        spec = self._layer_specs.get(idx)
        if spec is None:
            return
        if spec.is_empty():
            self._layer_specs.pop(idx, None)

    def _prepare_layer_vector(
        self, vector: torch.Tensor, *, context: str
    ) -> torch.Tensor:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{context} requires a torch.Tensor input.")
        vec = vector.detach().view(-1)
        if vec.shape[0] != self.hidden_size:
            raise ValueError(
                f"{context} shape mismatch: expected {(self.hidden_size,)}, got {tuple(vec.shape)}"
            )
        return vec.to(dtype=torch.float32, device="cpu").contiguous()

    def _normalize_vector(
        self, vector: torch.Tensor, *, context: str
    ) -> tuple[torch.Tensor, float]:
        prepared = self._prepare_layer_vector(vector, context=context)
        norm = float(prepared.norm().item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"{context} vector must have positive finite norm.")
        unit = (prepared / norm).to(dtype=torch.float32).contiguous()
        return unit, norm

    def _prepare_direction_vector(
        self, vector: torch.Tensor, *, context: str
    ) -> torch.Tensor:
        unit, _ = self._normalize_vector(vector, context=context)
        return unit

    def _broadcast_add(self, layer_idx: int, add_spec: AddSpec) -> None:
        payload = steering_runtime.serialize_tensor(
            add_spec.materialize().to(dtype=self._vector_dtype)
        )
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_vector,
            args=(int(layer_idx), payload),
        )
        layer_spec = self._ensure_layer_spec(int(layer_idx))
        layer_spec.add = add_spec.clone()

    def _broadcast_projection_cap(
        self, layer_idx: int, spec: ProjectionCapSpec
    ) -> None:
        payload = {
            "vector": steering_runtime.serialize_tensor(
                spec.vector.to(dtype=torch.float32)
            ),
            "min": spec.min,
            "max": spec.max,
        }
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_projection_cap,
            args=(int(layer_idx), payload),
        )
        layer_spec = self._ensure_layer_spec(int(layer_idx))
        layer_spec.projection_cap = spec.clone()

    def _broadcast_ablation(self, layer_idx: int, spec: AblationSpec) -> None:
        payload = {
            "vector": steering_runtime.serialize_tensor(
                spec.vector.to(dtype=self._vector_dtype)
            ),
            "scale": float(spec.scale),
        }
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_ablation,
            args=(int(layer_idx), payload),
        )
        layer_spec = self._ensure_layer_spec(int(layer_idx))
        layer_spec.ablation = spec.clone()

    def _broadcast_clear(self, layer_idx: int | None = None) -> None:
        if layer_idx is None:
            self._engine_client.collective_rpc(steering_runtime.clear_worker_vector)
            self._engine_client.collective_rpc(
                steering_runtime.clear_worker_projection_cap
            )
            self._engine_client.collective_rpc(steering_runtime.clear_worker_ablation)
            self._layer_specs.clear()
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
            layer_spec = self._layer_specs.get(target)
            if layer_spec is not None:
                layer_spec.add = None
                layer_spec.projection_cap = None
                layer_spec.ablation = None
            self._prune_layer_entry(target)

    def set_layer_vector(self, layer_idx: int, vector: torch.Tensor | None) -> None:
        target = int(layer_idx)
        if vector is None:
            self._broadcast_clear(target)
            return

        prepared = self._prepare_layer_vector(
            vector, context="Steering vector update"
        )
        magnitude = float(prepared.norm().item())
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            self._broadcast_clear(target)
            return
        unit = (prepared / magnitude).contiguous()
        add_spec = AddSpec(vector=unit, scale=magnitude)
        self._broadcast_add(target, add_spec)

    def set_vector(self, vector: torch.Tensor | None) -> None:
        """Set the steering vector for the active layer."""
        if self._active_layer is None:
            raise ValueError("Target layer not set. Call set_target_layer first.")
        self.set_layer_vector(self._active_layer, vector)

    def set_target_layer(self, layer_idx: int) -> None:
        """Select which layer receives ``set_vector`` updates."""
        idx = int(layer_idx)
        self._ensure_layer_spec(idx)
        self._active_layer = idx

    def set_layer_projection_cap(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> None:
        target = int(layer_idx)
        self._ensure_layer_spec(target)
        if min is None and max is None:
            raise ValueError("Projection cap requires at least one bound.")
        lower = float(min) if min is not None else None
        upper = float(max) if max is not None else None
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("min cannot exceed max.")
        prepared = self._prepare_direction_vector(
            vector, context="Projection cap vector"
        )
        spec = ProjectionCapSpec(vector=prepared, min=lower, max=upper)
        self._broadcast_projection_cap(target, spec)

    def clear_layer_projection_cap(self, layer_idx: int) -> None:
        target = int(layer_idx)
        self._engine_client.collective_rpc(
            steering_runtime.clear_worker_projection_cap, args=(target,)
        )
        layer_spec = self._layer_specs.get(target)
        if layer_spec is not None:
            layer_spec.projection_cap = None
        self._prune_layer_entry(target)

    def current_projection_cap(self, layer_idx: int) -> ProjectionCapSpec | None:
        layer_spec = self._layer_specs.get(int(layer_idx))
        if layer_spec is None or layer_spec.projection_cap is None:
            return None
        return layer_spec.projection_cap.clone()

    def set_projection_cap_precision(
        self, dtype: torch.dtype | str | None
    ) -> None:
        """Override the working precision used for projection cap math."""
        if dtype is None:
            dtype_name: str | None = None
        elif isinstance(dtype, torch.dtype):
            dtype_name = str(dtype).removeprefix("torch.")
        elif isinstance(dtype, str):
            dtype_name = dtype.removeprefix("torch.")
        else:
            raise TypeError("dtype must be a torch.dtype, string, or None.")
        self._engine_client.collective_rpc(
            steering_runtime.set_projection_cap_precision, args=(dtype_name,)
        )

    def set_layer_ablation(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        scale: float,
    ) -> None:
        target = int(layer_idx)
        self._ensure_layer_spec(target)
        if not isinstance(scale, (int, float)):
            raise TypeError("Ablation scale must be a numeric value.")
        scale_value = float(scale)
        if not math.isfinite(scale_value):
            raise ValueError("Ablation scale must be finite.")
        prepared = self._prepare_direction_vector(vector, context="Ablation vector")
        spec = AblationSpec(vector=prepared, scale=scale_value)
        self._broadcast_ablation(target, spec)

    def clear_layer_ablation(self, layer_idx: int) -> None:
        target = int(layer_idx)
        self._engine_client.collective_rpc(
            steering_runtime.clear_worker_ablation, args=(target,)
        )
        layer_spec = self._layer_specs.get(target)
        if layer_spec is not None:
            layer_spec.ablation = None
        self._prune_layer_entry(target)

    def current_ablation(self, layer_idx: int) -> AblationSpec | None:
        layer_spec = self._layer_specs.get(int(layer_idx))
        if layer_spec is None or layer_spec.ablation is None:
            return None
        return layer_spec.ablation.clone()

    def export_steering_spec(self) -> SteeringSpec:
        """Capture the current steering configuration across all layers."""
        layers: dict[int, LayerSteeringSpec] = {}
        for layer_idx, spec in self._layer_specs.items():
            cloned = spec.clone()
            if cloned.is_empty():
                continue
            layers[int(layer_idx)] = cloned
        return SteeringSpec(layers=layers)

    def apply_steering_spec(
        self, spec: SteeringSpec, *, clear_missing: bool = True
    ) -> None:
        """Apply a previously captured steering specification.

        Each layer entry replaces the additive vector, projection cap, and
        ablation settings. Layers omitted from the spec are cleared when
        ``clear_missing`` is ``True``. When a layer entry omits the additive
        vector the helper clears any existing steering state before applying
        projection caps or ablations present in the spec.
        """
        target_layers = {int(idx) for idx in spec.layers.keys()}
        if clear_missing:
            existing_layers = set(self._layer_specs.keys())
            for layer_idx in sorted(existing_layers - target_layers):
                self.clear_layer_vector(layer_idx)

        for layer_idx_raw, layer_spec in spec.layers.items():
            layer_idx = int(layer_idx_raw)
            add_spec = layer_spec.add
            cleared = add_spec is None
            if cleared:
                self.clear_layer_vector(layer_idx)
            else:
                self._broadcast_add(layer_idx, add_spec)

            projection_spec = layer_spec.projection_cap
            if projection_spec is not None:
                self.set_layer_projection_cap(
                    layer_idx,
                    projection_spec.vector,
                    min=projection_spec.min,
                    max=projection_spec.max,
                )
            elif not cleared:
                self.clear_layer_projection_cap(layer_idx)

            ablation_spec = layer_spec.ablation
            if ablation_spec is not None:
                self.set_layer_ablation(
                    layer_idx, ablation_spec.vector, ablation_spec.scale
                )
            elif not cleared:
                self.clear_layer_ablation(layer_idx)

    def push_steering_spec(
        self, spec: SteeringSpec, *, clear_missing: bool = True
    ) -> None:
        """Push current steering onto a stack and apply ``spec``."""
        baseline = self.export_steering_spec().clone()
        self._steering_stack.append(baseline)
        self.apply_steering_spec(spec, clear_missing=clear_missing)

    def pop_steering_spec(self) -> SteeringSpec:
        """Restore the most recently pushed steering spec."""
        if not self._steering_stack:
            raise RuntimeError("No steering spec to pop.")
        baseline = self._steering_stack.pop()
        self.apply_steering_spec(baseline, clear_missing=True)
        return baseline

    @contextmanager
    def steering(
        self, spec: SteeringSpec, *, clear_missing: bool = True
    ) -> Iterator[None]:
        """Context manager that reapplies previous steering on exit."""
        self.push_steering_spec(spec, clear_missing=clear_missing)
        try:
            yield
        finally:
            self.pop_steering_spec()

    def current_vector(self, layer_idx: int | None = None) -> torch.Tensor:
        """Return a CPU copy of the last vector broadcast to workers."""
        if layer_idx is None:
            if not self._layer_specs:
                raise ValueError("No steering vectors have been cached yet.")
            if len(self._layer_specs) != 1:
                raise ValueError(
                    "Multiple layer vectors cached. Provide layer_idx explicitly."
                )
            layer_idx = next(iter(self._layer_specs.keys()))
        spec = self._ensure_layer_spec(int(layer_idx))
        if spec.add is None:
            return torch.zeros(self.hidden_size, dtype=self._vector_dtype)
        vector = spec.add.materialize().to(dtype=self._vector_dtype)
        return vector.clone()

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

        single_conversation = isinstance(messages, list) and (
            len(messages) == 0 or isinstance(messages[0], dict)
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
            int(layer_idx): spec.add.materialize().detach().cpu().clone()
            for layer_idx, spec in self._layer_specs.items()
            if spec.add is not None
        }
        serialized_caps = {
            int(layer_idx): {
                "vector": spec.projection_cap.vector.detach().cpu().clone(),
                "min": spec.projection_cap.min,
                "max": spec.projection_cap.max,
            }
            for layer_idx, spec in self._layer_specs.items()
            if spec.projection_cap is not None
        }
        serialized_ablations = {
            int(layer_idx): {
                "vector": spec.ablation.vector.detach().cpu().clone(),
                "scale": float(spec.ablation.scale),
            }
            for layer_idx, spec in self._layer_specs.items()
            if spec.ablation is not None
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
        filtered_cfg = {
            key: value for key, value in cfg_dict.items() if key in allowed_keys
        }
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
                        min=payload.get("min"),
                        max=payload.get("max"),
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

    def enable_hidden_state_capture(
        self,
        layer_idx: int | Sequence[int],
        *,
        capture_before: bool = True,
        capture_after: bool = True,
        max_captures: int | None = None,
    ) -> None:
        """Enable hidden state capture for debugging and analysis.

        Captured states are stored on workers and can be retrieved via
        :meth:`fetch_hidden_states`. States remain in memory until cleared
        with :meth:`clear_hidden_states` or disabled with
        :meth:`disable_hidden_state_capture`.

        Parameters
        ----------
        layer_idx :
            Layer index or sequence of indices to enable capture for.
        capture_before :
            Capture hidden states before steering is applied.
        capture_after :
            Capture hidden states after steering is applied.
        max_captures :
            Maximum number of capture entries per layer. ``None`` means unlimited.
            When the limit is reached, new captures are ignored.

        Examples
        --------
        >>> model.enable_hidden_state_capture(2, capture_before=True, capture_after=True)
        >>> model.generate(["test prompt"], sampling_params)
        >>> states = model.fetch_hidden_states()
        >>> print(states[0][2][0]["before"].shape)  # worker 0, layer 2, first capture
        """
        indices = (
            [int(layer_idx)] if isinstance(layer_idx, int) else [int(i) for i in layer_idx]
        )
        for idx in indices:
            if idx < 0 or idx >= self.layer_count:
                raise ValueError(
                    f"Layer index {idx} out of range for model with {self.layer_count} layers"
                )
            self._engine_client.collective_rpc(
                steering_runtime.enable_hidden_state_capture,
                args=(idx,),
                kwargs={
                    "capture_before": capture_before,
                    "capture_after": capture_after,
                    "max_captures": max_captures,
                },
            )

    def disable_hidden_state_capture(
        self, layer_idx: int | Sequence[int] | None = None
    ) -> None:
        """Disable hidden state capture for one or more layers.

        This also clears any captured states for the affected layers.

        Parameters
        ----------
        layer_idx :
            Layer index, sequence of indices, or ``None`` to disable all layers.
        """
        if layer_idx is None:
            self._engine_client.collective_rpc(
                steering_runtime.disable_hidden_state_capture, args=(None,)
            )
        else:
            indices = (
                [int(layer_idx)] if isinstance(layer_idx, int) else [int(i) for i in layer_idx]
            )
            for idx in indices:
                self._engine_client.collective_rpc(
                    steering_runtime.disable_hidden_state_capture, args=(idx,)
                )

    def fetch_hidden_states(
        self, layer_idx: int | None = None
    ) -> list[dict[int, list[dict[str, torch.Tensor]]]]:
        """Retrieve captured hidden states from workers.

        Returns a list of capture maps, one per worker. Each map is keyed by
        layer index and contains a list of capture entries. Each entry has
        ``"before"`` and/or ``"after"`` keys with CPU tensors.

        Parameters
        ----------
        layer_idx :
            Layer index to fetch, or ``None`` to fetch all layers.

        Returns
        -------
        list[dict[int, list[dict[str, torch.Tensor]]]]
            List of worker capture maps. Length equals the number of workers
            (tensor parallel size).

        Examples
        --------
        >>> model.enable_hidden_state_capture(2)
        >>> model.generate(["test"], sampling_params)
        >>> states = model.fetch_hidden_states(layer_idx=2)
        >>> worker_0_captures = states[0][2]
        >>> first_capture = worker_0_captures[0]
        >>> before_steering = first_capture["before"]  # shape: [batch, seq_len, hidden_size]
        >>> after_steering = first_capture["after"]
        """
        payloads = self._engine_client.collective_rpc(
            steering_runtime.fetch_captured_hidden_states,
            args=(layer_idx,) if layer_idx is not None else (),
        )
        return cast(list[dict[int, list[dict[str, torch.Tensor]]]], payloads)

    def clear_hidden_states(self, layer_idx: int | None = None) -> None:
        """Clear captured hidden states without disabling capture.

        This frees memory while keeping capture enabled for future generations.

        Parameters
        ----------
        layer_idx :
            Layer index to clear, or ``None`` to clear all layers.
        """
        self._engine_client.collective_rpc(
            steering_runtime.clear_captured_hidden_states,
            args=(layer_idx,) if layer_idx is not None else (),
        )
