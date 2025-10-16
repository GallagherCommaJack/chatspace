"""Worker-side utilities for steering vector control inside vLLM workers.

These helpers are executed inside vLLM worker processes via collective RPCs.
They patch the target Qwen3 transformer layer so steering vectors participate
in CUDA-graph captures, and provide APIs to update vectors or retarget layers
at runtime.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import wraps
from typing import Any, Sequence

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

    layer_idx: int
    steering_vector: torch.Tensor


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


def _apply_vector_to_output(output: Any, vector: torch.Tensor) -> Any:
    _SEEN_OUTPUT_TYPES.add(type(output).__name__)
    if isinstance(output, torch.Tensor):
        return output + vector
    if isinstance(output, tuple):
        if not output:
            return output
        steered_hidden = output[0] + vector
        return (steered_hidden,) + output[1:]
    if isinstance(output, list):
        if not output:
            return output
        patched = list(output)
        patched[0] = patched[0] + vector
        return patched
    if isinstance(output, dict) and "last_hidden_state" in output:
        patched = dict(output)
        patched["last_hidden_state"] = patched["last_hidden_state"] + vector
        return patched
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = output.last_hidden_state + vector  # type: ignore[assignment]
        return output
    return output + vector

if _dynamo is not None:
    _apply_vector_to_output = _dynamo.disable(_apply_vector_to_output)  # type: ignore[assignment]


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
            return
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

    @wraps(original_forward)
    def _patched_forward(self, *args: Any, **kwargs: Any) -> Any:
        output = original_forward(self, *args, **kwargs)
        vector = getattr(self, "_chatspace_steering_vector", None)
        if vector is None:
            return output
        return _apply_vector_to_output(output, vector)

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


def initialize_worker_state(worker: Any, layer_idx: int, init_scale: float) -> dict[str, Any]:
    """Install steering patch on worker after model load."""
    ensure_layer_patch_installed()
    model = worker.model_runner.model
    layers = _resolve_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"Target layer {layer_idx} out of range for model with {len(layers)} layers"
        )

    hidden_size = model.config.hidden_size
    first_param = next(model.parameters(), None)
    if first_param is None:
        raise RuntimeError("Model has no parameters to infer device/dtype.")
    device = first_param.device
    dtype = first_param.dtype

    layer = layers[layer_idx]
    if not hasattr(layer, "_chatspace_steering_vector"):
        parameter = nn.Parameter(
            torch.zeros(hidden_size, dtype=dtype, device=device),
            requires_grad=True,
        )
        layer.register_parameter("_chatspace_steering_vector", parameter)
    vector = layer._chatspace_steering_vector  # type: ignore[attr-defined]
    if not isinstance(vector, torch.Tensor):
        raise RuntimeError("Failed to attach steering parameter to target layer.")
    if vector.numel() != hidden_size:
        raise ValueError(
            f"Steering buffer size mismatch: expected {hidden_size}, got {vector.numel()}"
        )
    with torch.no_grad():
        if init_scale > 0:
            torch.nn.init.normal_(vector, mean=0.0, std=init_scale)
        else:
            vector.zero_()

    state = _SteeringState(
        layer_idx=layer_idx,
        steering_vector=vector,
    )
    worker._chatspace_steering = state

    if not isinstance(worker.model_runner.model, _SteeredModelWrapper):
        worker.model_runner.model = _SteeredModelWrapper(model, state)

    return {
        "hidden_size": hidden_size,
        "layer_count": len(layers),
        "dtype": str(dtype),
        "device": str(device),
    }


def set_worker_vector(worker: Any, vector: torch.Tensor) -> None:
    """Replace the steering vector with the provided tensor."""
    state = _ensure_state(worker)
    try:
        vector = deserialize_tensor(
            vector,
            device=state.steering_vector.device,
            dtype=state.steering_vector.dtype,
        )
    except TypeError:
        # Fall back for already-deserialized tensors.
        if not isinstance(vector, torch.Tensor):
            raise
    if vector.ndim != 1:
        raise ValueError("Steering vector must be 1D.")
    if vector.shape != state.steering_vector.shape:
        raise ValueError(
            f"Steering vector shape mismatch: expected {tuple(state.steering_vector.shape)}, "
            f"got {tuple(vector.shape)}"
        )
    with torch.no_grad():
        state.steering_vector.copy_(
            vector.to(device=state.steering_vector.device, dtype=state.steering_vector.dtype)
        )


def clear_worker_vector(worker: Any) -> None:
    """Zero out the steering vector."""
    state = _ensure_state(worker)
    with torch.no_grad():
        state.steering_vector.zero_()


def set_worker_layer(worker: Any, layer_idx: int) -> None:
    """Move steering patch to a different transformer layer."""
    state = _ensure_state(worker)
    model = worker.model_runner.model
    layers = _resolve_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"Target layer {layer_idx} out of range for model with {len(layers)} layers"
        )
    if layer_idx == state.layer_idx:
        return
    old_layer = layers[state.layer_idx]
    new_layer = layers[layer_idx]
    old_vector = getattr(old_layer, "_chatspace_steering_vector", None)
    if isinstance(old_vector, torch.Tensor):
        with torch.no_grad():
            old_vector.zero_()
    if not hasattr(new_layer, "_chatspace_steering_vector"):
        template = state.steering_vector
        parameter = nn.Parameter(
            torch.zeros_like(template, device=template.device, dtype=template.dtype),
            requires_grad=True,
        )
        new_layer.register_parameter("_chatspace_steering_vector", parameter)
    vector = new_layer._chatspace_steering_vector  # type: ignore[attr-defined]
    if vector.shape != state.steering_vector.shape:
        raise ValueError(
            f"Steering buffer shape mismatch on new layer: expected {tuple(state.steering_vector.shape)}, "
            f"got {tuple(vector.shape)}"
        )
    with torch.no_grad():
        vector.zero_()
    state.layer_idx = int(layer_idx)
    state.steering_vector = vector


def fetch_worker_vector(worker: Any) -> torch.Tensor:
    """Return a CPU copy of the current steering vector (for debugging/tests)."""
    state = _ensure_state(worker)
    return state.steering_vector.detach().cpu().clone()


def fetch_worker_state(worker: Any) -> dict[str, Any]:
    """Inspect the worker steering state."""
    state = _ensure_state(worker)
    vector = state.steering_vector
    return {
        "layer_idx": state.layer_idx,
        "shape": tuple(vector.shape),
        "dtype": str(vector.dtype),
        "device": str(vector.device),
    }


def inspect_layer_vector(worker: Any) -> dict[str, Any]:
    """Return diagnostics for the patched layer (for debugging)."""
    state = _ensure_state(worker)
    layers = _resolve_layers(worker.model_runner.model)
    layer = layers[state.layer_idx]
    vector = getattr(layer, "_chatspace_steering_vector", None)
    layer_type = type(layer).__name__
    if vector is None:
        return {"has_vector": False}
    with torch.no_grad():
        norm = float(vector.norm().item())
        sum_val = float(vector.sum().item())
    try:
        forward_name = type(layer.forward).__name__
    except AttributeError:  # pragma: no cover - callable without __name__
        forward_name = layer.forward.__class__.__name__
    has_instance_forward = "forward" in layer.__dict__
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
    }
