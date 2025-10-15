"""Worker-side utilities for steering vector control inside vLLM workers.

These helpers are executed inside vLLM worker processes via collective RPCs.
They install forward hooks on the target Qwen3 transformer layer and provide
APIs to update steering vectors and retarget layers at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.hooks import RemovableHandle

LayerLike = Any


@dataclass
class _SteeringState:
    """Track steering metadata for a worker."""

    layer_idx: int
    vector: torch.Tensor
    hook_handle: RemovableHandle | None = None


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


def _register_hook(layer: LayerLike, state: _SteeringState) -> RemovableHandle:
    """Attach forward hook that adds the steering vector to residual stream."""

    def _hook(module, args, output):
        del module, args  # Unused.
        hidden = output[0] if isinstance(output, tuple) else output
        vector = state.vector
        if vector.device != hidden.device or vector.dtype != hidden.dtype:
            state.vector = vector = vector.to(device=hidden.device, dtype=hidden.dtype)
        if vector.shape[-1] != hidden.shape[-1]:
            raise ValueError(
                f"Steering vector hidden size mismatch: expected {hidden.shape[-1]}, "
                f"got {vector.shape[-1]}"
            )
        steered = hidden + vector
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return layer.register_forward_hook(_hook)


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
    """Install steering hook on worker after model load."""
    model = worker.model_runner.model
    layers = _resolve_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"Target layer {layer_idx} out of range for model with {len(layers)} layers"
        )

    hidden_size = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    vector = torch.zeros(hidden_size, device=device, dtype=dtype)
    if init_scale > 0:
        torch.nn.init.normal_(vector, mean=0.0, std=init_scale)

    state = _SteeringState(layer_idx=layer_idx, vector=vector)
    state.hook_handle = _register_hook(layers[layer_idx], state)
    worker._chatspace_steering = state

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
        vector = deserialize_tensor(vector, device=state.vector.device, dtype=state.vector.dtype)
    except TypeError:
        # Fall back for already-deserialized tensors.
        if not isinstance(vector, torch.Tensor):
            raise
    if vector.ndim != 1:
        raise ValueError("Steering vector must be 1D.")
    if vector.shape != state.vector.shape:
        raise ValueError(
            f"Steering vector shape mismatch: expected {tuple(state.vector.shape)}, "
            f"got {tuple(vector.shape)}"
        )
    state.vector.copy_(vector.to(device=state.vector.device, dtype=state.vector.dtype))


def clear_worker_vector(worker: Any) -> None:
    """Zero out the steering vector."""
    state = _ensure_state(worker)
    state.vector.zero_()


def set_worker_layer(worker: Any, layer_idx: int) -> None:
    """Move steering hook to a different transformer layer."""
    state = _ensure_state(worker)
    model = worker.model_runner.model
    layers = _resolve_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"Target layer {layer_idx} out of range for model with {len(layers)} layers"
        )
    if state.hook_handle is not None:
        state.hook_handle.remove()
    state.layer_idx = int(layer_idx)
    state.hook_handle = _register_hook(layers[layer_idx], state)


def fetch_worker_vector(worker: Any) -> torch.Tensor:
    """Return a CPU copy of the current steering vector (for debugging/tests)."""
    state = _ensure_state(worker)
    return state.vector.detach().cpu()


def fetch_worker_state(worker: Any) -> dict[str, Any]:
    """Inspect the worker steering state."""
    state = _ensure_state(worker)
    return {
        "layer_idx": state.layer_idx,
        "shape": tuple(state.vector.shape),
        "dtype": str(state.vector.dtype),
        "device": str(state.vector.device),
    }
