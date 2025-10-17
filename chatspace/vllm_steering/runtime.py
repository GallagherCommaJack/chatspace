"""Worker-side utilities for steering vector control inside vLLM workers.

These helpers are executed inside vLLM worker processes via collective RPCs.
They patch the target Qwen3 transformer layer so steering vectors participate
in CUDA-graph captures, and provide APIs to update vectors or retarget layers
at runtime.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

import torch
from torch import nn
try:  # pragma: no cover - torch._dynamo optional at runtime
    import torch._dynamo as _dynamo
except Exception:  # pragma: no cover - fallback when unavailable
    _dynamo = None

LayerLike = Any


@dataclass
class _SteeringState:
    """Track steering metadata for a worker."""

    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    layer_vectors: dict[int, torch.Tensor]
    projection_caps: dict[int, "_ProjectionCapConfig"]
    ablations: dict[int, "_AblationConfig"]


@dataclass
class _ProjectionCapConfig:
    """Maintain projection capping parameters for a layer."""

    unit_vector: torch.Tensor
    cap_below: float | None
    cap_above: float | None


@dataclass
class _AblationConfig:
    """Maintain ablation parameters for a layer."""

    unit_vector: torch.Tensor
    scale: float


class _SteeredModelWrapper(nn.Module):
    """Wrap vLLM model to apply steering after forward execution."""

    def __init__(self, model: nn.Module, state: _SteeringState) -> None:
        super().__init__()
        object.__setattr__(self, "_wrapped_model", model)
        self._steering_state = state

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        if name in {"_wrapped_model", "_steering_state"}:
            return object.__getattribute__(self, name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._wrapped_model, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._wrapped_model(*args, **kwargs)

    def unwrap(self) -> nn.Module:
        return self._wrapped_model


_PATCH_TARGETS: Sequence[tuple[str, str]] = (
    ("vllm.model_executor.models.qwen2", "Qwen2DecoderLayer"),
    ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeDecoderLayer"),
    ("vllm.model_executor.models.qwen2_vl", "Qwen2VLDecoderLayer"),
    ("vllm.model_executor.models.qwen3", "Qwen3DecoderLayer"),
    ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeDecoderLayer"),
    ("vllm.model_executor.models.qwen3_next", "Qwen3NextDecoderLayer"),
    ("vllm.model_executor.models.qwen3_vl", "Qwen3DecoderLayer"),
)

_PATCHED_CLASSES: set[type] = set()
_PATCH_INSTALLED = False
_SEEN_OUTPUT_TYPES: set[str] = set()


def _transform_output(
    output: Any,
    transform: Callable[[torch.Tensor], torch.Tensor],
    *,
    fallback: Callable[[Any], Any] | None = None,
) -> Any:
    """Apply ``transform`` to the primary hidden state within ``output``."""
    _SEEN_OUTPUT_TYPES.add(type(output).__name__)
    if isinstance(output, torch.Tensor):
        return transform(output)
    if isinstance(output, tuple):
        if not output:
            return output
        hidden = output[0]
        steered_hidden = transform(hidden)
        if steered_hidden is hidden:
            return output
        return (steered_hidden,) + output[1:]
    if isinstance(output, list):
        if not output:
            return output
        hidden = output[0]
        steered_hidden = transform(hidden)
        if steered_hidden is hidden:
            return output
        patched = list(output)
        patched[0] = steered_hidden
        return patched
    if isinstance(output, dict) and "last_hidden_state" in output:
        patched = dict(output)
        patched["last_hidden_state"] = transform(patched["last_hidden_state"])
        return patched
    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state  # type: ignore[assignment]
        output.last_hidden_state = transform(hidden)  # type: ignore[assignment]
        return output
    if fallback is not None:
        return fallback(output)
    raise TypeError(f"Unsupported output type for steering transform: {type(output)}")


def _apply_vector_to_output(output: Any, vector: torch.Tensor) -> Any:
    return _transform_output(output, lambda hidden: hidden + vector, fallback=lambda value: value + vector)


def _apply_projection_cap_to_output(output: Any, config: _ProjectionCapConfig) -> Any:
    def _cap(hidden: torch.Tensor) -> torch.Tensor:
        return _apply_projection_cap(hidden, config)

    return _transform_output(output, _cap, fallback=None)


def _apply_ablation_to_output(output: Any, config: _AblationConfig) -> Any:
    def _ablate(hidden: torch.Tensor) -> torch.Tensor:
        return _apply_ablation(hidden, config)

    return _transform_output(output, _ablate, fallback=None)


def _reshape_for_component_ops(hidden: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if hidden.shape[-1] != expected_dim:
        raise ValueError(
            f"Hidden state last dimension {hidden.shape[-1]} does not match steering dimension {expected_dim}"
        )
    return hidden.reshape(-1, expected_dim)


def _normalize_direction(vector: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        norm = torch.norm(vector)
    if not torch.isfinite(norm):
        raise ValueError("Direction vector norm is not finite.")
    if float(norm) <= 0:
        raise ValueError("Direction vector norm must be positive.")
    return vector / norm


def _deserialize_direction_payload(
    payload: Any, *, dest: torch.Tensor, state: _SteeringState
) -> torch.Tensor:
    try:
        vector = deserialize_tensor(
            payload,
            device=dest.device,
            dtype=dest.dtype,
        )
    except TypeError:
        if not isinstance(payload, torch.Tensor):
            raise
        vector = payload.to(device=dest.device, dtype=dest.dtype)
    if vector.ndim != 1:
        raise ValueError("Direction vector must be 1D.")
    if vector.shape[0] != state.hidden_size:
        raise ValueError(
            f"Direction vector dimension mismatch: expected {state.hidden_size}, got {vector.shape[0]}"
        )
    return _normalize_direction(vector)


def _apply_projection_cap(hidden: torch.Tensor, config: _ProjectionCapConfig) -> torch.Tensor:
    if config.cap_below is None and config.cap_above is None:
        return hidden
    flat = _reshape_for_component_ops(hidden, config.unit_vector.shape[0])
    unit = config.unit_vector
    projection = flat @ unit
    if config.cap_below is not None:
        lower = projection.new_tensor(float(config.cap_below))
        mask = projection < lower
        if mask.any():
            delta = (lower - projection[mask]).unsqueeze(-1) * unit
            flat[mask] = flat[mask] + delta
            projection = flat @ unit
    if config.cap_above is not None:
        upper = projection.new_tensor(float(config.cap_above))
        mask = projection > upper
        if mask.any():
            delta = (upper - projection[mask]).unsqueeze(-1) * unit
            flat[mask] = flat[mask] + delta
    return flat.reshape_as(hidden)


def _apply_ablation(hidden: torch.Tensor, config: _AblationConfig) -> torch.Tensor:
    if config.scale == 1.0:
        return hidden
    flat = _reshape_for_component_ops(hidden, config.unit_vector.shape[0])
    unit = config.unit_vector
    projection = flat @ unit
    component = projection.unsqueeze(-1) * unit
    flat = flat + (config.scale - 1.0) * component
    return flat.reshape_as(hidden)


if _dynamo is not None:
    _apply_projection_cap = _dynamo.disable(_apply_projection_cap)  # type: ignore[assignment]
    _apply_ablation = _dynamo.disable(_apply_ablation)  # type: ignore[assignment]


if _dynamo is not None:
    _transform_output = _dynamo.disable(_transform_output)  # type: ignore[assignment]
    _apply_vector_to_output = _dynamo.disable(_apply_vector_to_output)  # type: ignore[assignment]
    _apply_projection_cap_to_output = _dynamo.disable(_apply_projection_cap_to_output)  # type: ignore[assignment]
    _apply_ablation_to_output = _dynamo.disable(_apply_ablation_to_output)  # type: ignore[assignment]


def _patch_decoder_layer_class(layer_cls: type) -> None:
    if layer_cls in _PATCHED_CLASSES:
        return
    original_init = layer_cls.__init__
    original_forward = layer_cls.forward

    @wraps(original_init)
    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if hasattr(self, "_chatspace_steering_vector"):
            with torch.no_grad():
                self._chatspace_steering_vector.zero_()  # type: ignore[attr-defined]
        else:
            hidden_size = getattr(self, "hidden_size", None)
            if hidden_size is None:
                # Fallback: infer hidden size from first parameter.
                param = next(self.parameters(), None)
                if param is None:
                    raise RuntimeError(
                        f"Cannot infer hidden size for steering patch on {layer_cls!r}"
                    )
                hidden_size = param.shape[-1]
            ref_param = next(self.parameters(), None)
            dtype = ref_param.dtype if ref_param is not None else torch.float32
            device = ref_param.device if ref_param is not None else torch.device("cpu")
            parameter = nn.Parameter(
                torch.zeros(hidden_size, dtype=dtype, device=device),
                requires_grad=True,
            )
            self.register_parameter("_chatspace_steering_vector", parameter)
        if not hasattr(self, "_chatspace_projection_cap"):
            self._chatspace_projection_cap = None  # type: ignore[attr-defined]
        if not hasattr(self, "_chatspace_ablation"):
            self._chatspace_ablation = None  # type: ignore[attr-defined]

    @wraps(original_forward)
    def _patched_forward(self, *args: Any, **kwargs: Any) -> Any:
        output = original_forward(self, *args, **kwargs)
        vector = getattr(self, "_chatspace_steering_vector", None)
        if vector is not None:
            output = _apply_vector_to_output(output, vector)
        cap_config = getattr(self, "_chatspace_projection_cap", None)
        if isinstance(cap_config, _ProjectionCapConfig):
            output = _apply_projection_cap_to_output(output, cap_config)
        ablation_config = getattr(self, "_chatspace_ablation", None)
        if isinstance(ablation_config, _AblationConfig):
            output = _apply_ablation_to_output(output, ablation_config)
        return output

    layer_cls.__init__ = _patched_init  # type: ignore[assignment]
    layer_cls.forward = _patched_forward  # type: ignore[assignment]
    _PATCHED_CLASSES.add(layer_cls)


def ensure_layer_patch_installed() -> None:
    """Patch known Qwen decoder layers so CUDA graphs include steering."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    for module_name, class_name in _PATCH_TARGETS:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        layer_cls = getattr(module, class_name, None)
        if layer_cls is None:
            continue
        try:
            _patch_decoder_layer_class(layer_cls)
        except Exception:
            continue
    _PATCH_INSTALLED = True


def _resolve_layers(model: Any) -> list[LayerLike]:
    """Return the list of transformer layers for Qwen-like architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError(f"Could not resolve layers for model of type {type(model)}")


def _ensure_state(worker: Any) -> _SteeringState:
    state = getattr(worker, "_chatspace_steering", None)
    if state is None:
        raise RuntimeError("Steering state not initialized on worker.")
    return state


def _ensure_layer_entry(worker: Any, state: _SteeringState, layer_idx: int) -> torch.Tensor:
    """Ensure the requested layer owns a steering buffer and return it."""
    layer_idx = int(layer_idx)
    model = worker.model_runner.model
    layers = _resolve_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"Target layer {layer_idx} out of range for model with {len(layers)} layers"
        )
    layer = layers[layer_idx]
    vector = getattr(layer, "_chatspace_steering_vector", None)
    if not isinstance(vector, torch.Tensor) or vector.numel() != state.hidden_size:
        parameter = nn.Parameter(
            torch.zeros(
                state.hidden_size,
                dtype=state.dtype,
                device=state.device,
            ),
            requires_grad=True,
        )
        layer.register_parameter("_chatspace_steering_vector", parameter)
        vector = parameter
    if vector.ndim != 1:
        raise ValueError("Steering vector buffer must be 1-dimensional.")
    if vector.numel() != state.hidden_size:
        raise ValueError(
            f"Steering buffer size mismatch: expected {state.hidden_size}, got {vector.numel()}"
        )
    state.layer_vectors[layer_idx] = vector
    return vector


def deserialize_tensor(
    payload: Any, *, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Best-effort reconstruction of a tensor serialized via msgpack."""
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        try:
            dtype_str = payload["dtype"]
            shape = payload["shape"]
            data = payload["data"]
        except KeyError as exc:
            raise TypeError(f"Missing tensor metadata for key {exc}") from exc
        target_dtype = getattr(torch, dtype_str, None)
        if target_dtype is None:
            raise TypeError(f"Unsupported tensor dtype payload: {dtype_str}")
        shape_tuple = tuple(int(dim) for dim in shape)
        tensor = torch.tensor(data, dtype=target_dtype).reshape(shape_tuple)
    elif (
        isinstance(payload, (list, tuple))
        and len(payload) == 3
        and isinstance(payload[0], str)
        and isinstance(payload[1], Sequence)
    ):
        dtype_str, shape, data = payload
        target_dtype = getattr(torch, dtype_str, None)
        if target_dtype is None:
            raise TypeError(f"Unsupported tensor dtype payload: {dtype_str}")
        shape_tuple = tuple(int(dim) for dim in shape)
        if isinstance(data, memoryview):  # type: ignore[name-defined]
            raw = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        elif isinstance(data, list):
            tensor = torch.tensor(data, dtype=target_dtype)
            return tensor.reshape(shape_tuple).to(device=device, dtype=dtype or target_dtype)
        else:
            raise TypeError(f"Unsupported tensor data payload: {type(data)}")
        if len(raw) == 0:
            tensor = torch.empty(shape_tuple, dtype=target_dtype)
        else:
            tensor = torch.frombuffer(raw, dtype=target_dtype).clone().reshape(shape_tuple)
    else:
        raise TypeError(f"Unexpected tensor payload type: {type(payload)}")
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Serialize tensor into a JSON-friendly structure for RPC transport."""
    arr = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(arr.dtype).removeprefix("torch."),
        "shape": list(arr.shape),
        "data": arr.view(-1).tolist(),
    }


def initialize_worker_state(
    worker: Any, layer_indices: Sequence[int] | None = None
) -> dict[str, Any]:
    """Install steering patch on worker after model load."""
    ensure_layer_patch_installed()
    model = worker.model_runner.model
    layers = _resolve_layers(model)

    hidden_size = model.config.hidden_size
    first_param = next(model.parameters(), None)
    if first_param is None:
        raise RuntimeError("Model has no parameters to infer device/dtype.")
    device = first_param.device
    dtype = first_param.dtype
    state = _SteeringState(
        hidden_size=int(hidden_size),
        dtype=dtype,
        device=device,
        layer_vectors={},
        projection_caps={},
        ablations={},
    )
    worker._chatspace_steering = state

    if not isinstance(worker.model_runner.model, _SteeredModelWrapper):
        worker.model_runner.model = _SteeredModelWrapper(model, state)

    initial_layers = tuple(int(idx) for idx in (layer_indices or ()))
    seen: set[int] = set()
    for layer_idx in initial_layers:
        if layer_idx in seen:
            continue
        seen.add(layer_idx)
        _ensure_layer_entry(worker, state, layer_idx)
        with torch.no_grad():
            state.layer_vectors[layer_idx].zero_()

    return {
        "hidden_size": hidden_size,
        "layer_count": len(layers),
        "dtype": str(dtype),
        "device": str(device),
    }


def set_worker_vector(worker: Any, layer_idx: int, vector: torch.Tensor) -> None:
    """Replace or install the steering vector for a specific layer."""
    state = _ensure_state(worker)
    dest = _ensure_layer_entry(worker, state, int(layer_idx))
    try:
        vector = deserialize_tensor(
            vector,
            device=dest.device,
            dtype=dest.dtype,
        )
    except TypeError:
        # Fall back for already-deserialized tensors.
        if not isinstance(vector, torch.Tensor):
            raise
    if vector.ndim != 1:
        raise ValueError("Steering vector must be 1D.")
    if vector.shape != dest.shape:
        raise ValueError(
            f"Steering vector shape mismatch: expected {tuple(dest.shape)}, "
            f"got {tuple(vector.shape)}"
        )
    with torch.no_grad():
        dest.copy_(
            vector.to(device=dest.device, dtype=dest.dtype)
        )


def set_worker_projection_cap(worker: Any, layer_idx: int, payload: dict[str, Any]) -> None:
    """Configure projection capping for a specific layer."""
    state = _ensure_state(worker)
    target_idx = int(layer_idx)
    dest = _ensure_layer_entry(worker, state, target_idx)
    vector_payload = payload.get("vector")
    if vector_payload is None:
        raise ValueError("Projection cap payload missing 'vector'.")
    unit = _deserialize_direction_payload(vector_payload, dest=dest, state=state)
    cap_below = payload.get("cap_below")
    cap_above = payload.get("cap_above")
    cap_below_float = float(cap_below) if cap_below is not None else None
    cap_above_float = float(cap_above) if cap_above is not None else None
    if (
        cap_below_float is not None
        and cap_above_float is not None
        and cap_below_float > cap_above_float
    ):
        raise ValueError("cap_below cannot exceed cap_above.")
    config = _ProjectionCapConfig(
        unit_vector=unit.detach().clone().contiguous(),
        cap_below=cap_below_float,
        cap_above=cap_above_float,
    )
    state.projection_caps[target_idx] = config
    layer = _resolve_layers(worker.model_runner.model)[target_idx]
    layer._chatspace_projection_cap = config  # type: ignore[attr-defined]


def clear_worker_projection_cap(worker: Any, layer_idx: int | None = None) -> None:
    """Disable projection capping for one or all layers."""
    state = _ensure_state(worker)
    if layer_idx is None:
        targets = tuple(state.projection_caps.keys())
    else:
        targets = (int(layer_idx),)
    if not targets:
        return
    layers = _resolve_layers(worker.model_runner.model)
    for idx in targets:
        state.projection_caps.pop(idx, None)
        if 0 <= idx < len(layers):
            layer = layers[idx]
            if hasattr(layer, "_chatspace_projection_cap"):
                layer._chatspace_projection_cap = None  # type: ignore[attr-defined]


def set_worker_ablation(worker: Any, layer_idx: int, payload: dict[str, Any]) -> None:
    """Configure ablation scaling for a specific layer."""
    state = _ensure_state(worker)
    target_idx = int(layer_idx)
    dest = _ensure_layer_entry(worker, state, target_idx)
    vector_payload = payload.get("vector")
    if vector_payload is None:
        raise ValueError("Ablation payload missing 'vector'.")
    scale = payload.get("scale")
    if scale is None:
        raise ValueError("Ablation payload missing 'scale'.")
    scale_float = float(scale)
    if not math.isfinite(scale_float):
        raise ValueError("Ablation scale must be finite.")
    unit = _deserialize_direction_payload(vector_payload, dest=dest, state=state)
    config = _AblationConfig(
        unit_vector=unit.detach().clone().contiguous(),
        scale=scale_float,
    )
    state.ablations[target_idx] = config
    layer = _resolve_layers(worker.model_runner.model)[target_idx]
    layer._chatspace_ablation = config  # type: ignore[attr-defined]


def clear_worker_ablation(worker: Any, layer_idx: int | None = None) -> None:
    """Disable ablation scaling for one or all layers."""
    state = _ensure_state(worker)
    if layer_idx is None:
        targets = tuple(state.ablations.keys())
    else:
        targets = (int(layer_idx),)
    if not targets:
        return
    layers = _resolve_layers(worker.model_runner.model)
    for idx in targets:
        state.ablations.pop(idx, None)
        if 0 <= idx < len(layers):
            layer = layers[idx]
            if hasattr(layer, "_chatspace_ablation"):
                layer._chatspace_ablation = None  # type: ignore[attr-defined]


def clear_worker_vector(worker: Any, layer_idx: int | None = None) -> None:
    """Zero out steering vectors for one or all layers."""
    state = _ensure_state(worker)
    if layer_idx is None:
        clear_worker_projection_cap(worker, None)
        clear_worker_ablation(worker, None)
    else:
        clear_worker_projection_cap(worker, int(layer_idx))
        clear_worker_ablation(worker, int(layer_idx))
    targets: Sequence[torch.Tensor]
    if layer_idx is None:
        targets = tuple(state.layer_vectors.values())
    else:
        vector = state.layer_vectors.get(int(layer_idx))
        if vector is None:
            try:
                vector = _ensure_layer_entry(worker, state, int(layer_idx))
            except ValueError:
                return
        targets = (vector,)
    if not targets:
        return
    with torch.no_grad():
        for vec in targets:
            vec.zero_()


def fetch_worker_vectors(worker: Any) -> dict[int, torch.Tensor]:
    """Return CPU copies of steering vectors keyed by layer index."""
    state = _ensure_state(worker)
    return {
        layer_idx: vector.detach().cpu().clone()
        for layer_idx, vector in state.layer_vectors.items()
    }


def fetch_worker_vector(worker: Any, layer_idx: int) -> torch.Tensor:
    """Return a CPU copy of a steering vector for the requested layer."""
    state = _ensure_state(worker)
    vector = state.layer_vectors.get(int(layer_idx))
    if vector is None:
        vector = _ensure_layer_entry(worker, state, int(layer_idx))
    return vector.detach().cpu().clone()


def fetch_worker_state(worker: Any) -> dict[str, Any]:
    """Inspect the worker steering state."""
    state = _ensure_state(worker)
    model = worker.model_runner.model
    layer_count = len(_resolve_layers(model))
    return {
        "active_layers": sorted(state.layer_vectors.keys()),
        "shape": (state.hidden_size,),
        "dtype": str(state.dtype),
        "device": str(state.device),
        "layer_count": layer_count,
    }


def inspect_layer_vector(worker: Any, layer_idx: int | None = None) -> dict[str, Any]:
    """Return diagnostics for the patched layer (for debugging)."""
    state = _ensure_state(worker)
    if layer_idx is None:
        if not state.layer_vectors:
            return {"has_vector": False}
        target_idx = int(next(iter(state.layer_vectors.keys())))
    else:
        target_idx = int(layer_idx)
    try:
        vector = _ensure_layer_entry(worker, state, target_idx)
    except ValueError:
        return {"has_vector": False}
    layers = _resolve_layers(worker.model_runner.model)
    layer = layers[target_idx]
    layer_type = type(layer).__name__
    with torch.no_grad():
        norm = float(vector.norm().item())
        sum_val = float(vector.sum().item())
    try:
        forward_name = type(layer.forward).__name__
    except AttributeError:  # pragma: no cover - callable without __name__
        forward_name = layer.forward.__class__.__name__
    has_instance_forward = "forward" in layer.__dict__
    projection_cap = getattr(layer, "_chatspace_projection_cap", None)
    if isinstance(projection_cap, _ProjectionCapConfig):
        proj_info = {
            "cap_below": projection_cap.cap_below,
            "cap_above": projection_cap.cap_above,
        }
    else:
        proj_info = None
    ablation = getattr(layer, "_chatspace_ablation", None)
    if isinstance(ablation, _AblationConfig):
        ablation_info = {
            "scale": ablation.scale,
        }
    else:
        ablation_info = None
    return {
        "has_vector": True,
        "norm": norm,
        "sum": sum_val,
        "dtype": str(vector.dtype),
        "device": str(vector.device),
        "output_types": list(_SEEN_OUTPUT_TYPES),
        "layer_type": layer_type,
        "forward_type": forward_name,
        "instance_forward": has_instance_forward,
        "layer_module": layer.__class__.__module__,
        "projection_cap": proj_info,
        "ablation": ablation_info,
    }
