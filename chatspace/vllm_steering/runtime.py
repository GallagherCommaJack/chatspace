"""Worker-side utilities for steering vector control inside vLLM workers.

These helpers are executed inside vLLM worker processes via collective RPCs.
They patch decoder layers in Qwen, Llama, and Gemma models so steering vectors
participate in CUDA-graph captures, and provide APIs to update vectors or
retarget layers at runtime.

Tensor Parallelism Support
---------------------------
Steering operations are designed to work transparently with vLLM's tensor parallelism
(TP). At decoder layer boundaries, vLLM's ``RowParallelLinear`` layers perform
allreduce operations, ensuring hidden states are full-size (replicated) across all TP
ranks (verified via source code inspection).

**Broadcasting:** The ``collective_rpc`` mechanism broadcasts steering vectors to all
TP workers via a shared message queue (``rpc_broadcast_mq``). Each worker receives the
identical full-size vector and applies it independently.

**Steering Semantics:**
- **Additive steering**: Each rank independently adds the same full-size vector,
  maintaining consistency across ranks without coordination.
- **Projection capping**: Each rank computes dot products on full-size hidden
  states, yielding identical projections without requiring distributed reductions.
- **Ablation**: Component scaling operates on full-size states independently with
  consistent results.

**Implementation:** No distributed operations or vector sharding are needed in the
steering code. The implementation stores full-size steering vectors on each rank,
with memory cost ``O(hidden_size)`` per rank rather than ``O(hidden_size / tp_size)``.

**Note:** Single-GPU (TP=1) behavior has been empirically verified. Multi-GPU (TP≥2)
correctness is expected based on architectural analysis but requires hardware testing
to confirm.
"""

from __future__ import annotations

import importlib
import math
import queue
import threading
import time
from contextlib import nullcontext
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

# Optional override for projection cap math precision. When set the cap
# operations are evaluated in the requested dtype before casting back.
_PROJECTION_CAP_PRECISION: torch.dtype | None = None


@dataclass
class _CaptureConfig:
    """Configuration for capturing hidden states at a layer."""

    capture_before: bool = True
    capture_after: bool = True
    max_captures: int | None = None


@dataclass
class _CaptureCounters:
    """Track capture step indices for prefill/decode phases."""

    total: int = 0
    prefill: int = 0
    decode: int = 0
    prefill_complete: bool = False

    def reset(self) -> None:
        self.total = 0
        self.prefill = 0
        self.decode = 0
        self.prefill_complete = False

    def next(self, seq_len: int) -> tuple[str, int, int]:
        """Advance counters and return (phase, phase_index, step)."""
        if not self.prefill_complete and seq_len > 1:
            phase = "prefill"
            phase_index = self.prefill
            self.prefill += 1
        else:
            phase = "decode"
            phase_index = self.decode
            self.decode += 1
            self.prefill_complete = True
        step = self.total
        self.total += 1
        return phase, phase_index, step


@dataclass
class _SteeringState:
    """Track steering metadata for a worker."""

    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    layer_vectors: dict[int, torch.Tensor]
    projection_caps: dict[int, "_ProjectionCapConfig"]
    ablations: dict[int, "_AblationConfig"]
    capture_configs: dict[int, _CaptureConfig]
    captured_states: dict[int, list[dict[str, torch.Tensor]]]
    capture_counters: dict[int, _CaptureCounters]
    capture_task_queue: "queue.SimpleQueue[_PendingCaptureTask] | None"
    capture_worker: threading.Thread | None
    capture_stream: torch.cuda.Stream | None
    capture_lock: threading.Lock | None
    capture_pending: dict[int, int]


@dataclass
class _ProjectionCapConfig:
    """Maintain projection capping parameters for a layer."""

    unit_vector: torch.Tensor
    min: float | None
    max: float | None


@dataclass
class _AblationConfig:
    """Maintain ablation parameters for a layer."""

    unit_vector: torch.Tensor
    scale: float


@dataclass
class _PendingCaptureTask:
    """Describe an asynchronous GPU→CPU capture transfer."""

    layer_idx: int
    entry: dict[str, Any]
    key: str
    gpu_tensor: torch.Tensor
    ready_event: torch.cuda.Event


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
    # Qwen models
    ("vllm.model_executor.models.qwen2", "Qwen2DecoderLayer"),
    ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeDecoderLayer"),
    ("vllm.model_executor.models.qwen2_vl", "Qwen2VLDecoderLayer"),
    ("vllm.model_executor.models.qwen3", "Qwen3DecoderLayer"),
    ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeDecoderLayer"),
    ("vllm.model_executor.models.qwen3_next", "Qwen3NextDecoderLayer"),
    ("vllm.model_executor.models.qwen3_vl", "Qwen3DecoderLayer"),
    # Llama models
    ("vllm.model_executor.models.llama", "LlamaDecoderLayer"),
    ("vllm.model_executor.models.llama4", "Llama4DecoderLayer"),
    ("vllm.model_executor.models.llama_eagle", "LlamaDecoderLayer"),
    ("vllm.model_executor.models.llama_eagle3", "LlamaDecoderLayer"),
    ("vllm.model_executor.models.llama4_eagle", "Llama4DecoderLayer"),
    # Gemma models
    ("vllm.model_executor.models.gemma", "GemmaDecoderLayer"),
    ("vllm.model_executor.models.gemma2", "Gemma2DecoderLayer"),
    ("vllm.model_executor.models.gemma3", "Gemma3DecoderLayer"),
)

_PATCHED_CLASSES: set[type] = set()
_PATCH_INSTALLED = False
_SEEN_OUTPUT_TYPES: set[str] = set()

STEERING_RPC_METHOD = "_chatspace_steering_rpc"
STEERING_WORKER_EXTENSION = "chatspace.vllm_steering.runtime.SteeringWorkerExtension"
_RPC_HANDLERS: dict[str, Callable[..., Any]] = {}
_RPC_GATEWAY_INSTALLED = False


def rpc_args(op: str, *args: Any) -> tuple[Any, ...]:
    return (op, *args)


def _register_rpc(name: str, func: Callable[..., Any]) -> None:
    _RPC_HANDLERS[name] = func


def ensure_collective_rpc_gateway_installed() -> None:
    """Expose steering RPC handlers via vLLM's msgpack-safe interface."""
    global _RPC_GATEWAY_INSTALLED
    if _RPC_GATEWAY_INSTALLED:
        return
    try:
        from vllm.worker.worker_base import WorkerWrapperBase
    except Exception:  # pragma: no cover - vLLM not available
        return
    if hasattr(WorkerWrapperBase, STEERING_RPC_METHOD):
        _RPC_GATEWAY_INSTALLED = True
        return

    def _dispatch(self: Any, op: str, *args: Any, **kwargs: Any) -> Any:
        handler = _RPC_HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"Unknown chatspace steering RPC: {op}")
        target = getattr(self, "worker", self)
        return handler(target, *args, **kwargs)

    setattr(WorkerWrapperBase, STEERING_RPC_METHOD, _dispatch)
    _RPC_GATEWAY_INSTALLED = True


class SteeringWorkerExtension:
    """Worker mixin providing msgpack-safe steering RPC handling."""

    def _chatspace_steering_rpc(self, op: str, *args: Any, **kwargs: Any) -> Any:
        handler = _RPC_HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"Unknown chatspace steering RPC: {op}")
        return handler(self, *args, **kwargs)


def _setup_async_capture_infrastructure(state: _SteeringState) -> None:
    """Ensure background capture machinery is initialised for a worker."""
    if state.capture_lock is None:
        state.capture_lock = threading.Lock()
    if state.device.type != "cuda":
        state.capture_task_queue = None
        state.capture_stream = None
        state.capture_worker = None
        return
    if state.capture_task_queue is None:
        state.capture_task_queue = queue.SimpleQueue()
    if state.capture_stream is None:
        state.capture_stream = torch.cuda.Stream(device=state.device)
    if state.capture_worker is None:
        worker = threading.Thread(
            target=_capture_worker_loop,
            args=(state,),
            name="chatspace-capture-worker",
            daemon=True,
        )
        state.capture_worker = worker
        worker.start()


def _capture_worker_loop(state: _SteeringState) -> None:
    """Background worker that migrates captured tensors to CPU."""
    if state.device.type == "cuda":
        torch.cuda.set_device(state.device)
    stream = state.capture_stream
    queue_ref = state.capture_task_queue
    if queue_ref is None or stream is None:
        return
    while True:
        task = queue_ref.get()
        try:
            task.ready_event.wait()
            transfer_event = torch.cuda.Event(blocking=False)
            with torch.cuda.stream(stream):
                cpu_tensor = task.gpu_tensor.to("cpu", non_blocking=True)
                stream.record_event(transfer_event)
            transfer_event.wait()
            del task.gpu_tensor
            cpu_tensor = cpu_tensor.detach().clone()
            entry = task.entry
            entry[task.key] = cpu_tensor
            pending = entry.get("_pending", 0) - 1
            if pending > 0:
                entry["_pending"] = pending
                continue
            entry.pop("_pending", None)
            _finalize_capture_entry(state, task.layer_idx, entry)
        except Exception:
            continue


def _queue_capture_tensor(
    tensor: torch.Tensor,
    *,
    entry: dict[str, Any],
    key: str,
    layer_idx: int | None,
    state: _SteeringState | None,
    capture_config: _CaptureConfig | None,
) -> None:
    """Queue tensor for asynchronous CPU materialisation or clone immediately."""
    if tensor is None:
        return
    max_captures = capture_config.max_captures if capture_config is not None else None
    if "_max" not in entry:
        entry["_max"] = max_captures
    if state is None or layer_idx is None:
        entry[key] = tensor.detach().cpu().clone()
        return
    if tensor.device.type != "cuda":
        entry[key] = tensor.detach().cpu().clone()
        return
    _setup_async_capture_infrastructure(state)
    if state.capture_task_queue is None or state.capture_stream is None:
        entry[key] = tensor.detach().cpu().clone()
        return
    gpu_clone = tensor.detach().clone()
    ready_event = torch.cuda.Event(blocking=False)
    torch.cuda.current_stream(device=gpu_clone.device).record_event(ready_event)
    pending_before = entry.get("_pending", 0)
    pending = pending_before + 1
    entry["_pending"] = pending
    if pending_before == 0:
        entry["_async"] = True
        state.capture_pending[layer_idx] = state.capture_pending.get(layer_idx, 0) + 1
    task = _PendingCaptureTask(
        layer_idx=layer_idx,
        entry=entry,
        key=key,
        gpu_tensor=gpu_clone,
        ready_event=ready_event,
    )
    state.capture_task_queue.put(task)


def _finalize_capture_entry(
    state: _SteeringState | None,
    layer_idx: int | None,
    entry: dict[str, Any],
) -> None:
    """Append capture entry when all pending tensor copies have completed."""
    if state is None or layer_idx is None:
        return
    if not entry:
        return
    pending = entry.get("_pending", 0)
    if pending > 0:
        return
    entry.pop("_pending", None)
    async_flag = entry.pop("_async", False)
    max_captures = entry.pop("_max", None)
    captures = state.captured_states.setdefault(layer_idx, [])
    lock = state.capture_lock
    context = lock if lock is not None else nullcontext()
    with context:
        if max_captures is not None and len(captures) >= max_captures:
            captures.pop(0)
        captures.append(entry)
    if async_flag:
        pending = state.capture_pending.get(layer_idx, 0)
        if pending > 0:
            remaining = pending - 1
            if remaining > 0:
                state.capture_pending[layer_idx] = remaining
            else:
                state.capture_pending.pop(layer_idx, None)


def _wait_for_pending_captures(state: _SteeringState, indices: Sequence[int]) -> None:
    if state.capture_task_queue is None:
        return
    if not indices:
        return
    attempts = 0
    while True:
        if all(state.capture_pending.get(idx, 0) == 0 for idx in indices):
            return
        attempts += 1
        if attempts > 2000:  # ~1s worst-case fallback
            return
        time.sleep(0.0005)


def _extract_hidden_from_output(output: Any) -> torch.Tensor | None:
    """Extract the primary hidden state tensor from layer output.

    For vLLM Qwen and Llama layers, the forward returns (delta, residual) where:
    - delta (output[0]): The per-layer update that will be added to the residual.
    - residual (output[1]): The running residual stream before the delta is applied.

    To mirror HuggingFace decoder outputs we need ``residual + delta`` which is the
    hidden state propagated to the next layer. Fall back to the first tensor when the
    structure does not match this tuple form.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            # vLLM Qwen layers return (delta, residual)
            first = output[0]
            second = output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                # Combine residual stream with delta to match HF hidden state
                return second + first
        if len(output) > 0:
            # Single element or fallback: return first element
            hidden = output[0]
            if isinstance(hidden, torch.Tensor):
                return hidden
    if isinstance(output, dict) and "last_hidden_state" in output:
        return output["last_hidden_state"]
    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state  # type: ignore[assignment]
        if isinstance(hidden, torch.Tensor):
            return hidden
    return None


def _infer_sequence_length(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 1
    if tensor.ndim >= 2:
        return int(tensor.shape[-2])
    return 1


def _prepare_capture_metadata(
    counter: _CaptureCounters | None, reference: torch.Tensor | None
) -> dict[str, int | str]:
    if counter is None:
        return {}
    seq_len = _infer_sequence_length(reference)
    phase, phase_index, step = counter.next(seq_len)
    return {
        "phase": phase,
        "phase_index": phase_index,
        "step": step,
        "seq_len": seq_len,
    }


def _is_qwen_layer_output(output: Any) -> bool:
    return (
        isinstance(output, tuple)
        and len(output) >= 2
        and isinstance(output[0], torch.Tensor)
        and isinstance(output[1], torch.Tensor)
    )


def _transform_output(
    output: Any,
    transform: Callable[[torch.Tensor], torch.Tensor],
    *,
    fallback: Callable[[Any], Any] | None = None,
    mode: str = "delta",
    debug_hook: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Apply ``transform`` to the primary hidden state within ``output``.

    Parameters
    ----------
    mode :
        ``"delta"`` applies the transform to the layer delta (output[0]) and returns
        an updated tuple. ``"hidden"`` materializes HuggingFace-equivalent hidden
        state ``residual + delta`` (when available), applies ``transform``, and
        re-expresses the result as an updated delta so downstream layers receive
        the transformed hidden state.
    debug_hook :
        Optional callback receiving intermediate tensors (as a dict). Used for
        instrumentation such as projection delta dumps.
    """
    _SEEN_OUTPUT_TYPES.add(type(output).__name__)
    if isinstance(output, torch.Tensor):
        return transform(output)
    if isinstance(output, tuple):
        if not output:
            return output
        if mode == "hidden" and _is_qwen_layer_output(output):
            delta, residual = output[0], output[1]
            target_dtype = _PROJECTION_CAP_PRECISION
            if target_dtype is not None:
                working_residual = residual.to(target_dtype)
                working_delta = delta.to(target_dtype)
            else:
                working_residual = residual
                working_delta = delta
            hidden = working_residual + working_delta
            transformed = transform(hidden)
            if transformed is hidden:
                return output
            if target_dtype is not None:
                projected_delta = transformed - working_residual
                if debug_hook is not None:
                    debug_hook(
                        {
                            "projected_delta": projected_delta,
                            "target_dtype": target_dtype,
                            "input_delta_dtype": delta.dtype,
                            "residual_dtype": working_residual.dtype,
                            "hidden_before_dtype": hidden.dtype,
                            "hidden_after_dtype": transformed.dtype,
                        }
                    )
                new_delta = projected_delta.to(delta.dtype)
            else:
                projected_delta = transformed - residual
                if debug_hook is not None:
                    debug_hook(
                        {
                            "projected_delta": projected_delta,
                            "target_dtype": residual.dtype,
                            "input_delta_dtype": delta.dtype,
                            "residual_dtype": residual.dtype,
                            "hidden_before_dtype": hidden.dtype,
                            "hidden_after_dtype": transformed.dtype,
                        }
                    )
                new_delta = projected_delta
            return (new_delta,) + output[1:]
        hidden = output[0]
        transformed = transform(hidden)
        if transformed is hidden:
            return output
        return (transformed,) + output[1:]
    if isinstance(output, list):
        if not output:
            return output
        hidden = output[0]
        transformed = transform(hidden)
        if transformed is hidden:
            return output
        patched = list(output)
        patched[0] = transformed
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
    return _transform_output(
        output,
        lambda delta: delta + vector,
        fallback=lambda value: value + vector,
        mode="delta",
    )


def _apply_projection_cap_to_output(
    output: Any,
    config: _ProjectionCapConfig,
    *,
    debug_hook: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    def _cap(hidden: torch.Tensor) -> torch.Tensor:
        return _apply_projection_cap(hidden, config)

    return _transform_output(
        output,
        _cap,
        fallback=None,
        mode="hidden",
        debug_hook=debug_hook,
    )


def _apply_ablation_to_output(output: Any, config: _AblationConfig) -> Any:
    def _ablate(hidden: torch.Tensor) -> torch.Tensor:
        return _apply_ablation(hidden, config)

    return _transform_output(output, _ablate, fallback=None, mode="hidden")


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
    payload: Any,
    *,
    dest: torch.Tensor,
    state: _SteeringState,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    try:
        vector = deserialize_tensor(
            payload,
            device=dest.device,
            dtype=target_dtype,
        )
    except TypeError:
        if not isinstance(payload, torch.Tensor):
            raise
        vector = payload.to(device=dest.device)
        if target_dtype is not None and vector.dtype != target_dtype:
            vector = vector.to(dtype=target_dtype)
    if vector.ndim != 1:
        raise ValueError("Direction vector must be 1D.")
    if vector.shape[0] != state.hidden_size:
        raise ValueError(
            f"Direction vector dimension mismatch: expected {state.hidden_size}, got {vector.shape[0]}"
        )
    return _normalize_direction(vector)


def _apply_projection_cap(hidden: torch.Tensor, config: _ProjectionCapConfig) -> torch.Tensor:
    if config.min is None and config.max is None:
        return hidden
    target_dtype = _PROJECTION_CAP_PRECISION
    working = hidden
    cast_back = False
    unit = config.unit_vector
    if target_dtype is None:
        target_dtype = unit.dtype
        if hidden.dtype != target_dtype:
            working = hidden.to(target_dtype)
            cast_back = True
    elif hidden.dtype != target_dtype:
        working = hidden.to(target_dtype)
        cast_back = True
    if unit.dtype != target_dtype:
        unit = unit.to(target_dtype)
    flat = _reshape_for_component_ops(working, unit.shape[0])
    projection = flat @ unit
    clamp_kwargs: dict[str, torch.Tensor] = {}
    if config.min is not None:
        clamp_kwargs["min"] = projection.new_tensor(float(config.min))
    if config.max is not None:
        clamp_kwargs["max"] = projection.new_tensor(float(config.max))
    clamped = torch.clamp(projection, **clamp_kwargs)  # type: ignore[arg-type]
    if clamped is not projection:
        delta = (clamped - projection).unsqueeze(-1) * unit
        flat = flat + delta
    result = flat.reshape_as(working)
    if cast_back:
        return result.to(hidden.dtype)
    return result


def _apply_ablation(hidden: torch.Tensor, config: _AblationConfig) -> torch.Tensor:
    if config.scale == 1.0:
        return hidden
    flat = _reshape_for_component_ops(hidden, config.unit_vector.shape[0])
    unit = config.unit_vector.to(dtype=flat.dtype)  # Match hidden state dtype
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

        # Check if this layer should capture hidden states
        capture_config = getattr(self, "_chatspace_capture_config", None)
        capture_enabled = isinstance(capture_config, _CaptureConfig)
        captured_before: torch.Tensor | None = None
        cap_debug_payload: dict[str, Any] | None = None

        def _capture_cap_debug(payload: dict[str, Any]) -> None:
            nonlocal cap_debug_payload
            projected_delta = payload.get("projected_delta")
            if isinstance(projected_delta, torch.Tensor):
                payload = dict(payload)
                payload["projected_delta"] = projected_delta.detach().cpu().clone()
            cap_debug_payload = payload

        if capture_enabled and capture_config.capture_before:
            # Extract and capture hidden state before steering
            try:
                captured_before = _extract_hidden_from_output(output)
            except Exception:  # pragma: no cover - graceful degradation
                pass

        # Apply steering operations
        vector = getattr(self, "_chatspace_steering_vector", None)
        if vector is not None:
            output = _apply_vector_to_output(output, vector)
        cap_config = getattr(self, "_chatspace_projection_cap", None)
        if isinstance(cap_config, _ProjectionCapConfig):
            debug_hook = _capture_cap_debug if capture_enabled else None
            output = _apply_projection_cap_to_output(
                output,
                cap_config,
                debug_hook=debug_hook,
            )
        ablation_config = getattr(self, "_chatspace_ablation", None)
        if isinstance(ablation_config, _AblationConfig):
            output = _apply_ablation_to_output(output, ablation_config)

        # Capture hidden state after steering if requested
        if capture_enabled:
            try:
                captured_after: torch.Tensor | None = None
                if capture_config.capture_after:
                    captured_after = _extract_hidden_from_output(output)
                if captured_before is not None or captured_after is not None:
                    capture_entry: dict[str, Any] = {}
                    state = getattr(self, "_chatspace_steering_state", None)
                    layer_idx = getattr(self, "_chatspace_layer_index", None)
                    if captured_before is not None:
                        _queue_capture_tensor(
                            captured_before,
                            entry=capture_entry,
                            key="before",
                            layer_idx=layer_idx,
                            state=state,
                            capture_config=capture_config,
                        )
                    if captured_after is not None:
                        _queue_capture_tensor(
                            captured_after,
                            entry=capture_entry,
                            key="after",
                            layer_idx=layer_idx,
                            state=state,
                            capture_config=capture_config,
                        )
                    counter = getattr(self, "_chatspace_capture_counter", None)
                    reference_tensor = captured_after if captured_after is not None else captured_before
                    metadata = _prepare_capture_metadata(counter, reference_tensor)
                    if metadata:
                        capture_entry["meta"] = metadata
                    if cap_debug_payload is not None:
                        projected_delta = cap_debug_payload.get("projected_delta")
                        if isinstance(projected_delta, torch.Tensor):
                            capture_entry["cap_delta"] = projected_delta
                        cap_meta = {
                            key: str(value)
                            for key, value in cap_debug_payload.items()
                            if key in {
                                "target_dtype",
                                "input_delta_dtype",
                                "residual_dtype",
                                "hidden_before_dtype",
                                "hidden_after_dtype",
                            }
                            and value is not None
                        }
                        if cap_meta:
                            capture_entry["cap_meta"] = cap_meta
                        cap_debug_payload = None
                    if ("before" in capture_entry or "after" in capture_entry or capture_entry.get("_pending")):
                        _finalize_capture_entry(state, layer_idx, capture_entry)
            except Exception:  # pragma: no cover - graceful degradation
                pass
        cap_debug_payload = None

        return output

    layer_cls.__init__ = _patched_init  # type: ignore[assignment]
    layer_cls.forward = _patched_forward  # type: ignore[assignment]
    _PATCHED_CLASSES.add(layer_cls)


def ensure_layer_patch_installed() -> None:
    """Patch known Qwen and Llama decoder layers so CUDA graphs include steering."""
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
    """Return the list of transformer layers for Qwen, Llama, and Gemma architectures."""
    # Multimodal models (e.g., Gemma3ForConditionalGeneration)
    if hasattr(model, "language_model"):
        if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            return list(model.language_model.model.layers)
        if hasattr(model.language_model, "layers"):
            return list(model.language_model.layers)
    # Standard text-only models
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
    setattr(layer, "_chatspace_steering_state", state)
    setattr(layer, "_chatspace_layer_index", layer_idx)
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
        storage_dtype_str = payload.get("storage_dtype")
        encoding = payload.get("encoding")
        if encoding == "bytes":
            if isinstance(data, memoryview):  # type: ignore[name-defined]
                raw = data.tobytes()
            elif isinstance(data, (bytes, bytearray)):
                raw = bytes(data)
            else:
                raise TypeError(f"Unsupported tensor data payload for encoding=bytes: {type(data)}")
            if len(raw) == 0:
                tensor = torch.empty(shape_tuple, dtype=target_dtype)
            else:
                storage_dtype = target_dtype
                if storage_dtype_str:
                    storage_dtype = getattr(torch, storage_dtype_str, None)
                    if not isinstance(storage_dtype, torch.dtype):
                        raise TypeError(f"Unsupported storage dtype payload: {storage_dtype_str}")
                tensor = torch.frombuffer(raw, dtype=storage_dtype).clone().reshape(shape_tuple)
                if storage_dtype != target_dtype:
                    tensor = tensor.to(dtype=target_dtype)
        elif encoding in (None, "list"):
            tensor = torch.tensor(data, dtype=target_dtype).reshape(shape_tuple)
        else:
            raise TypeError(f"Unsupported tensor encoding: {encoding}")
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
    dtype_name = str(arr.dtype).removeprefix("torch.")
    storage = arr
    storage_dtype_name = dtype_name
    if arr.numel() == 0:
        buffer = b""
    else:
        try:
            buffer = storage.numpy().tobytes()
        except TypeError:
            storage = arr.to(dtype=torch.float32)
            storage_dtype_name = str(storage.dtype).removeprefix("torch.")
            buffer = storage.numpy().tobytes()
    payload: dict[str, Any] = {
        "dtype": dtype_name,
        "shape": list(arr.shape),
        "data": buffer,
        "encoding": "bytes",
    }
    if storage_dtype_name != dtype_name:
        payload["storage_dtype"] = storage_dtype_name
    return payload


def initialize_worker_state(
    worker: Any, layer_indices: Sequence[int] | None = None
) -> dict[str, Any]:
    """Install steering patch on worker after model load."""
    ensure_layer_patch_installed()
    model = worker.model_runner.model
    layers = _resolve_layers(model)

    # Handle multimodal models (e.g., Gemma3) where config is nested
    config = model.config
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        hidden_size = config.text_config.hidden_size
    elif hasattr(config, "hidden_size"):
        hidden_size = config.hidden_size
    else:
        raise RuntimeError(f"Could not resolve hidden_size from config of type {type(config)}")

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
        capture_configs={},
        captured_states={},
        capture_counters={},
        capture_task_queue=None,
        capture_worker=None,
        capture_stream=None,
        capture_lock=None,
        capture_pending={},
    )
    worker._chatspace_steering = state
    _setup_async_capture_infrastructure(state)

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
    unit = _deserialize_direction_payload(
        vector_payload, dest=dest, state=state, target_dtype=torch.float32
    )
    min_val = payload.get("min")
    max_val = payload.get("max")
    min_float = float(min_val) if min_val is not None else None
    max_float = float(max_val) if max_val is not None else None
    if (
        min_float is not None
        and max_float is not None
        and min_float > max_float
    ):
        raise ValueError("min cannot exceed max.")
    config = _ProjectionCapConfig(
        unit_vector=unit.detach().clone().contiguous(),
        min=min_float,
        max=max_float,
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


def set_projection_cap_precision(worker: Any, dtype_name: str | None) -> None:
    """Adjust the working precision used for projection cap math."""
    global _PROJECTION_CAP_PRECISION
    if dtype_name is None:
        _PROJECTION_CAP_PRECISION = None
        return
    if not isinstance(dtype_name, str):
        raise TypeError("dtype_name must be a string or None.")
    attr = dtype_name.removeprefix("torch.")
    target = getattr(torch, attr, None)
    if not isinstance(target, torch.dtype):
        raise ValueError(f"Unsupported projection cap dtype: {dtype_name}")
    _PROJECTION_CAP_PRECISION = target


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


def fetch_worker_vectors(worker: Any) -> dict[int, dict[str, Any]]:
    """Return serialized steering vectors keyed by layer index."""
    state = _ensure_state(worker)
    return {
        layer_idx: serialize_tensor(vector)
        for layer_idx, vector in state.layer_vectors.items()
    }


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
            "min": projection_cap.min,
            "max": projection_cap.max,
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


def enable_hidden_state_capture(
    worker: Any,
    layer_idx: int,
    *,
    capture_before: bool = True,
    capture_after: bool = True,
    max_captures: int | None = None,
) -> None:
    """Enable hidden state capture for a specific layer.

    Parameters
    ----------
    worker :
        The vLLM worker instance.
    layer_idx :
        Layer index to enable capture for.
    capture_before :
        Whether to capture hidden states before steering is applied.
    capture_after :
        Whether to capture hidden states after steering is applied.
    max_captures :
        Maximum number of capture entries to store. ``None`` means unlimited.
    """
    state = _ensure_state(worker)
    target_idx = int(layer_idx)
    _ensure_layer_entry(worker, state, target_idx)

    capture_config = _CaptureConfig(
        capture_before=bool(capture_before),
        capture_after=bool(capture_after),
        max_captures=int(max_captures) if max_captures is not None else None,
    )
    state.capture_configs[target_idx] = capture_config
    state.captured_states[target_idx] = []
    counter = _CaptureCounters()
    state.capture_counters[target_idx] = counter

    layers = _resolve_layers(worker.model_runner.model)
    layer = layers[target_idx]
    layer._chatspace_capture_config = capture_config  # type: ignore[attr-defined]
    layer._chatspace_capture_queue = state.captured_states[target_idx]  # type: ignore[attr-defined]
    layer._chatspace_capture_counter = counter  # type: ignore[attr-defined]


def disable_hidden_state_capture(
    worker: Any, layer_idx: int | None = None
) -> None:
    """Disable hidden state capture for one or all layers.

    Parameters
    ----------
    worker :
        The vLLM worker instance.
    layer_idx :
        Layer index to disable capture for. If ``None``, disables for all layers.
    """
    state = _ensure_state(worker)
    if layer_idx is None:
        targets = tuple(state.capture_configs.keys())
    else:
        targets = (int(layer_idx),)

    if not targets:
        return

    layers = _resolve_layers(worker.model_runner.model)
    for idx in targets:
        state.capture_configs.pop(idx, None)
        lock = state.capture_lock
        context = lock if lock is not None else nullcontext()
        with context:
            state.captured_states.pop(idx, None)  # Also clear captured data
        state.capture_pending.pop(idx, None)
        state.capture_counters.pop(idx, None)
        if 0 <= idx < len(layers):
            layer = layers[idx]
            if hasattr(layer, "_chatspace_capture_config"):
                layer._chatspace_capture_config = None  # type: ignore[attr-defined]
            if hasattr(layer, "_chatspace_capture_queue"):
                layer._chatspace_capture_queue = None  # type: ignore[attr-defined]
            if hasattr(layer, "_chatspace_capture_counter"):
                layer._chatspace_capture_counter = None  # type: ignore[attr-defined]


def fetch_captured_hidden_states(
    worker: Any, layer_idx: int | Sequence[int] | None = None
) -> dict[int, list[dict[str, Any]]]:
    """Retrieve serialized hidden states from one or all layers.

    Parameters
    ----------
    worker :
        The vLLM worker instance.
    layer_idx :
        Layer index or iterable of indices to fetch captures from. If ``None``,
        fetches captures from all layers that currently have stored entries.

    Returns
    -------
    dict[int, list[dict[str, Any]]]
        Mapping of layer indices to lists of capture entries. Each entry is a dict
        with ``"before"``/``"after"`` tensors serialized via :func:`serialize_tensor`
        plus optional ``"meta"``, ``"cap_delta"``, and ``"cap_meta"`` diagnostics.
    """
    state = _ensure_state(worker)

    def _serialize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in entry.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = serialize_tensor(value)
            elif isinstance(value, dict):
                serialized[key] = dict(value)
            else:
                serialized[key] = value
        return serialized

    if layer_idx is None:
        indices: Sequence[int] | None = None
    elif isinstance(layer_idx, Sequence) and not isinstance(layer_idx, (str, bytes, bytearray)):
        indices = [int(idx) for idx in layer_idx]
    else:
        indices = [int(layer_idx)]

    wait_targets = list(state.capture_pending.keys()) if indices is None else list(indices)
    if wait_targets:
        _wait_for_pending_captures(state, wait_targets)

    if indices is None:
        return {
            idx: [
                _serialize_entry(entry)
                for entry in captures
            ]
            for idx, captures in state.captured_states.items()
        }

    indices = list(indices)

    result: dict[int, list[dict[str, Any]]] = {}
    for target_idx in indices:
        captures = state.captured_states.get(target_idx, [])
        result[target_idx] = [
            _serialize_entry(entry)
            for entry in captures
        ]
    return result


def clear_captured_hidden_states(
    worker: Any, layer_idx: int | None = None
) -> None:
    """Clear captured hidden states for one or all layers without disabling capture.

    Parameters
    ----------
    worker :
        The vLLM worker instance.
    layer_idx :
        Layer index to clear captures from. If ``None``, clears all layers.
    """
    state = _ensure_state(worker)
    if layer_idx is None:
        lock = state.capture_lock
        context = lock if lock is not None else nullcontext()
        with context:
            for captures in state.captured_states.values():
                captures.clear()
        for counter in state.capture_counters.values():
            counter.reset()
    else:
        target_idx = int(layer_idx)
        lock = state.capture_lock
        context = lock if lock is not None else nullcontext()
        with context:
            captures = state.captured_states.get(target_idx)
            if captures is not None:
                captures.clear()
        counter = state.capture_counters.get(target_idx)
        if counter is not None:
            counter.reset()


for _rpc_fn in (
    initialize_worker_state,
    set_worker_vector,
    set_worker_projection_cap,
    clear_worker_projection_cap,
    set_projection_cap_precision,
    set_worker_ablation,
    clear_worker_ablation,
    clear_worker_vector,
    fetch_worker_vectors,
    fetch_worker_state,
    inspect_layer_vector,
    enable_hidden_state_capture,
    disable_hidden_state_capture,
    fetch_captured_hidden_states,
    clear_captured_hidden_states,
):
    _register_rpc(_rpc_fn.__name__, _rpc_fn)

ensure_collective_rpc_gateway_installed()
