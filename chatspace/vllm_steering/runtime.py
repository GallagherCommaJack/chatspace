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
import logging
import math
import os
import queue
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

import torch
from torch import nn

# Use vLLM's logger to ensure logs appear in worker processes
try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
try:  # pragma: no cover - torch._dynamo optional at runtime
    import torch._dynamo as _dynamo
except Exception:  # pragma: no cover - fallback when unavailable
    _dynamo = None

try:  # pragma: no cover - profiler optional depending on torch build
    from torch.profiler import ProfilerActivity, profile as torch_profile
except Exception:  # pragma: no cover - profiler unavailable
    ProfilerActivity = None
    torch_profile = None


def _get_env_int(name: str, default: int) -> int:
    """Helper to parse integer environment variables safely."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """Normalize truthy environment flags."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value not in {"", "0", "false", "no", "off"}


_PROFILE_FETCH_ENABLED = _env_flag("CHATSPACE_PROFILE_FETCH", False)
_PROFILE_FETCH_TOPK = _get_env_int("CHATSPACE_PROFILE_FETCH_TOPK", 5)
_PROFILE_FETCH_EVENT_LIMIT = _get_env_int("CHATSPACE_PROFILE_FETCH_EVENT_LIMIT", 32)
_PROFILE_FETCH_TRACE_DIR = os.getenv("CHATSPACE_PROFILE_FETCH_TRACE_DIR")
_PROFILE_FETCH_TRACE_PREFIX = os.getenv("CHATSPACE_PROFILE_FETCH_TRACE_PREFIX", "fetch_trace")
_CAPTURE_METADATA_ENABLED = _env_flag("CHATSPACE_CAPTURE_METADATA", True)

LayerLike = Any

# Optional override for projection cap math precision. When set the cap
# operations are evaluated in the requested dtype before casting back.
_PROJECTION_CAP_PRECISION: torch.dtype | None = None

_PROFILE_FETCH_WARNED = False

_TRANSFER_EVENT_TOKENS = ("::copy", "::_to", "::to")


def _is_transfer_event(name: str) -> bool:
    """Heuristic to detect GPU→CPU transfer ops in profiler output."""
    lowered = name.lower()
    return any(token in lowered for token in _TRANSFER_EVENT_TOKENS)


def _summarize_profiler(
    prof: Any, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convert a torch profiler session into structured summary data."""
    rows: list[dict[str, Any]] = []
    total_cpu = 0.0
    total_cuda = 0.0
    for event in prof.key_averages():
        name = getattr(event, "key", getattr(event, "name", "unknown"))
        cpu_total = getattr(event, "cpu_time_total", 0.0) / 1_000_000.0
        cpu_self = getattr(event, "self_cpu_time_total", 0.0) / 1_000_000.0
        cuda_total = getattr(event, "cuda_time_total", 0.0) / 1_000_000.0
        cuda_self = getattr(event, "self_cuda_time_total", 0.0) / 1_000_000.0
        row = {
            "name": name,
            "count": int(getattr(event, "count", 0)),
            "cpu_time_s": cpu_total,
            "self_cpu_time_s": cpu_self,
            "cuda_time_s": cuda_total,
            "self_cuda_time_s": cuda_self,
        }
        rows.append(row)
        total_cpu += cpu_total
        total_cuda += cuda_total

    rows.sort(key=lambda item: item["self_cuda_time_s"], reverse=True)
    event_count = len(rows)
    limited_rows = rows[: max(_PROFILE_FETCH_EVENT_LIMIT, _PROFILE_FETCH_TOPK)]
    transfer_cuda_s = sum(
        row["self_cuda_time_s"] for row in rows if _is_transfer_event(row["name"])
    )
    summary = {
        "metadata": dict(metadata or {}),
        "events": limited_rows,
        "top_events": limited_rows[: _PROFILE_FETCH_TOPK],
        "event_count": event_count,
        "total_cpu_s": total_cpu,
        "total_cuda_s": total_cuda,
        "transfer_cuda_s": transfer_cuda_s,
    }
    return summary


def _export_profiler_trace(prof: Any, metadata: dict[str, Any]) -> str | None:
    """Persist profiler trace to disk when configured."""
    if not _PROFILE_FETCH_TRACE_DIR:
        return None
    trace_dir = Path(_PROFILE_FETCH_TRACE_DIR).expanduser()
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - filesystem errors
        logger.warning(
            "Unable to create profiler trace directory %s; skipping export.",
            trace_dir,
            exc_info=True,
        )
        return None

    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    suffix = f"{time.time():.6f}".replace(".", "")
    filename = f"{_PROFILE_FETCH_TRACE_PREFIX}_{timestamp}_{suffix}.json"
    path = trace_dir / filename
    try:
        prof.export_chrome_trace(str(path))
    except Exception:  # pragma: no cover - torch profiler export errors
        logger.warning(
            "Failed to export torch profiler trace to %s", path, exc_info=True
        )
        return None
    if metadata is not None:
        metadata.setdefault("trace_path", str(path))
    return str(path)


def _log_profiler_summary(summary: dict[str, Any]) -> None:
    """Emit a concise log entry for profiler sessions."""
    top_events = summary.get("top_events") or []
    if not top_events:
        logger.info(
            "[Torch Profiler] fetch_batch_captures events=%d total_cuda=%.4fs total_cpu=%.4fs (no hot ops)",
            summary.get("event_count", 0),
            summary.get("total_cuda_s", 0.0),
            summary.get("total_cpu_s", 0.0),
        )
        return

    preview = ", ".join(
        f"{row['name']}:{row['self_cuda_time_s']:.4f}s"
        for row in top_events[: min(3, len(top_events))]
    )
    logger.info(
        "[Torch Profiler] fetch_batch_captures events=%d total_cuda=%.4fs total_cpu=%.4fs top_cuda=%s",
        summary.get("event_count", 0),
        summary.get("total_cuda_s", 0.0),
        summary.get("total_cpu_s", 0.0),
        preview,
    )


@contextmanager
def _profile_fetch_batch(
    state: "_SteeringState | None", metadata: dict[str, Any] | None = None
) -> Any:
    """Profile fetch_batch_captures with torch.profiler when enabled."""
    global _PROFILE_FETCH_WARNED

    if not _PROFILE_FETCH_ENABLED:
        yield None
        return
    if torch_profile is None or ProfilerActivity is None:
        if not _PROFILE_FETCH_WARNED:
            logger.warning(
                "Torch profiler instrumentation requested but torch.profiler is unavailable; "
                "set CHATSPACE_PROFILE_FETCH=0 to disable profiling."
            )
            _PROFILE_FETCH_WARNED = True
        yield None
        return

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Mutate metadata in-place so caller updates propagate into the summary
    metadata = metadata or {}
    try:
        metadata["activities"] = [activity.name.lower() for activity in activities]
    except AttributeError:  # pragma: no cover - older torch enums
        metadata["activities"] = [str(activity) for activity in activities]

    with torch_profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
    ) as prof:
        yield prof

    summary = _summarize_profiler(prof, metadata)
    trace_path = _export_profiler_trace(prof, summary["metadata"])
    if trace_path:
        summary["trace_path"] = trace_path
    setattr(prof, "_chatspace_summary", summary)
    if state is not None:
        state.last_fetch_profile = summary
    _log_profiler_summary(summary)




@dataclass
class _StepContext:
    """Batch metadata computed once per scheduler step."""

    step_index: int
    request_ids: tuple[str, ...]
    scheduled_tokens: tuple[int, ...]
    slice_ranges: tuple[tuple[int, int], ...]  # Precomputed cumulative offsets


@dataclass
class _SteeringState:
    """Track steering metadata for a worker."""

    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    layer_vectors: dict[int, torch.Tensor]
    projection_caps: dict[int, "_ProjectionCapConfig"]
    ablations: dict[int, "_AblationConfig"]

    # Per-request activation capture
    active_capture_requests: dict[str, set[int]] = None  # request_id -> layer indices
    request_captures: dict[str, dict[int, torch.Tensor]] = None  # request_id -> layer -> tensor
    request_prefill_buffers: dict[str, dict[int, list[torch.Tensor]]] = None  # request_id -> layer -> chunks
    request_decode_buffers: dict[str, dict[int, list[torch.Tensor]]] = None  # request_id -> layer -> decode tokens
    request_last_phase: dict[str, str] = None  # request_id -> "prefill" or "decode"
    request_token_counts: dict[str, int] = None  # request_id -> token count

    # Per-step batch metadata (captured from model runner)
    step_metadata: dict[int, dict[str, Any]] = None  # step_number -> {request_ids, seq_lens, ...}
    global_step: int = 0  # Monotonically increasing step counter

    # Future: current_step_context for optimized routing
    current_step_context: _StepContext | None = None

    # Async transfer infrastructure
    transfer_stream: torch.cuda.Stream | None = None  # Stream for non-blocking GPU→CPU transfers
    request_pending_transfers: dict[str, dict[int, tuple[torch.Tensor, torch.cuda.Event]]] = None  # request_id -> layer -> (cpu_tensor, event)

    # Profiling summaries
    last_fetch_profile: dict[str, Any] | None = None


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
                # NOTE: This tensor addition happens on every forward pass when capturing!
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


def _extract_hidden_from_output_timed(output: Any) -> tuple[torch.Tensor | None, float, float]:
    """Timed version of _extract_hidden_from_output for profiling.

    Returns:
        (hidden_tensor, time_spent_in_addition, time_spent_total)
    """
    import time
    start_total = time.perf_counter()

    if isinstance(output, torch.Tensor):
        return output, 0.0, time.perf_counter() - start_total

    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            first = output[0]
            second = output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                # Time the tensor addition specifically
                start_add = time.perf_counter()
                result = second + first
                time_add = time.perf_counter() - start_add

                time_total = time.perf_counter() - start_total
                return result, time_add, time_total

        if len(output) > 0:
            hidden = output[0]
            if isinstance(hidden, torch.Tensor):
                time_total = time.perf_counter() - start_total
                return hidden, 0.0, time_total

    if isinstance(output, dict) and "last_hidden_state" in output:
        time_total = time.perf_counter() - start_total
        return output["last_hidden_state"], 0.0, time_total

    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state  # type: ignore[assignment]
        if isinstance(hidden, torch.Tensor):
            time_total = time.perf_counter() - start_total
            return hidden, 0.0, time_total

    time_total = time.perf_counter() - start_total
    return None, 0.0, time_total


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


# ============================================================================
# Hook Variant System for Profiling
# ============================================================================
# Environment variable to select hook variant for performance testing
_HOOK_VARIANT = os.environ.get("CHATSPACE_HOOK_VARIANT", "full")


def _capture_hook_noop(
    state: _SteeringState,
    layer_idx: int,
    hidden: torch.Tensor,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """No-op hook: measures baseline overhead of hook invocation."""
    pass


def _capture_hook_noop_notiming(
    state: _SteeringState,
    layer_idx: int,
    hidden: torch.Tensor,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """No-op hook with NO hidden extraction: measures pure metadata lookup overhead."""
    pass


def _capture_hook_slice_only(
    state: _SteeringState,
    layer_idx: int,
    hidden: torch.Tensor,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """Slice-only hook: performs slicing but no clone/store."""
    start_idx = 0
    for i, req_id in enumerate(request_ids):
        if req_id not in state.active_capture_requests:
            if seq_lens and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        if layer_idx not in state.active_capture_requests[req_id]:
            if seq_lens and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        # Slice but don't clone
        if seq_lens and i < len(seq_lens):
            seq_len = seq_lens[i]
            end_idx = start_idx + seq_len
            req_hidden = hidden[start_idx:end_idx]  # SLICE ONLY
            start_idx = end_idx
        else:
            req_hidden = hidden  # noqa: F841


def _capture_hook_clone_only(
    state: _SteeringState,
    layer_idx: int,
    hidden: torch.Tensor,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """Clone-only hook: slices and clones but doesn't store."""
    start_idx = 0
    for i, req_id in enumerate(request_ids):
        if req_id not in state.active_capture_requests:
            if seq_lens and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        if layer_idx not in state.active_capture_requests[req_id]:
            if seq_lens and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        # Slice and clone but don't store
        if seq_lens and i < len(seq_lens):
            seq_len = seq_lens[i]
            end_idx = start_idx + seq_len
            req_hidden = hidden[start_idx:end_idx]
            start_idx = end_idx
        else:
            req_hidden = hidden

        # Clone but discard immediately
        _ = req_hidden.detach().clone()  # CLONE ONLY


def _capture_hook_full(
    state: _SteeringState,
    layer_idx: int,
    hidden: torch.Tensor,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """Full hook: complete capture implementation (slice + clone + store)."""
    start_idx = 0
    active_reqs = state.active_capture_requests  # Cache dict reference

    for i, req_id in enumerate(request_ids):
        # Check if request has any active captures (single lookup)
        req_layers = active_reqs.get(req_id)
        if req_layers is None:
            if seq_lens and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        # Check if this layer is captured for this request
        if layer_idx not in req_layers:
            if seq_lens and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        # Determine slice for this request
        if seq_lens and i < len(seq_lens):
            seq_len = seq_lens[i]
            end_idx = start_idx + seq_len
            req_hidden = hidden[start_idx:end_idx]
            start_idx = end_idx
        else:
            req_hidden = hidden

        # Determine phase (using shape check - prefill=seq_len>1, decode=seq_len==1)
        req_seq_len = req_hidden.shape[0]
        is_prefill = req_seq_len > 1

        if is_prefill:
            # Buffer chunks during prefill
            state.request_prefill_buffers[req_id][layer_idx].append(
                req_hidden.detach().clone()
            )
            state.request_last_phase[req_id] = "prefill"
        else:
            # Check if transitioning from prefill to decode
            if state.request_last_phase.get(req_id) == "prefill":
                _coalesce_prefill_chunks(state, req_id)
                state.request_last_phase[req_id] = "decode"

            # Buffer decode tokens to reduce concatenation overhead
            decode_buf = state.request_decode_buffers[req_id][layer_idx]
            decode_buf.append(req_hidden.detach().clone())

            # Flush buffer every 32 tokens to reduce cat operations
            if len(decode_buf) >= 32:
                batched = torch.cat(decode_buf, dim=0)
                decode_buf.clear()

                req_captures = state.request_captures[req_id]
                if layer_idx in req_captures:
                    final_tensor = torch.cat([req_captures[layer_idx], batched], dim=0)
                else:
                    final_tensor = batched
                req_captures[layer_idx] = final_tensor

                # Phase 2: Start async GPU→CPU transfer immediately during generation
                if state.transfer_stream is not None:
                    with torch.cuda.stream(state.transfer_stream):
                        gpu_tensor = final_tensor.detach().contiguous()
                        cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(state.transfer_stream)

                        # Store for later synchronization in fetch
                        if req_id not in state.request_pending_transfers:
                            state.request_pending_transfers[req_id] = {}
                        state.request_pending_transfers[req_id][layer_idx] = (cpu_tensor, event)


# Hook variant registry
_HOOK_VARIANTS: dict[str, Callable] = {
    "noop_notiming": _capture_hook_noop_notiming,
    "noop": _capture_hook_noop,
    "slice_only": _capture_hook_slice_only,
    "clone_only": _capture_hook_clone_only,
    "full": _capture_hook_full,
}

# Log selected variant for debugging
if _HOOK_VARIANT not in _HOOK_VARIANTS:
    logger.warning(
        f"Invalid CHATSPACE_HOOK_VARIANT='{_HOOK_VARIANT}'. "
        f"Valid options: {list(_HOOK_VARIANTS.keys())}. Defaulting to 'full'."
    )
    _HOOK_VARIANT = "full"
logger.info(f"Activation capture hook variant: {_HOOK_VARIANT}")


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

        # Apply steering operations
        vector = getattr(self, "_chatspace_steering_vector", None)
        if vector is not None:
            output = _apply_vector_to_output(output, vector)
        cap_config = getattr(self, "_chatspace_projection_cap", None)
        if isinstance(cap_config, _ProjectionCapConfig):
            output = _apply_projection_cap_to_output(output, cap_config, debug_hook=None)
        ablation_config = getattr(self, "_chatspace_ablation", None)
        if isinstance(ablation_config, _AblationConfig):
            output = _apply_ablation_to_output(output, ablation_config)

        # Per-request activation capture (fast path)
        try:
            state = getattr(self, "_chatspace_steering_state", None)
            if state is None or not state.active_capture_requests:
                return output

            # Get current step metadata (fast path - no debug logging)
            current_step = state.global_step - 1  # Most recent step
            if current_step < 0:
                return output

            metadata = state.step_metadata.get(current_step)
            if metadata is None:
                return output

            request_ids = metadata.get("request_ids")
            if not request_ids:
                return output

            layer_idx = getattr(self, "_chatspace_layer_index", None)
            if layer_idx is None:
                return output

            # Extract hidden state (after steering)
            hidden = _extract_hidden_from_output(output)
            if hidden is None or hidden.dim() != 2:
                return output

            # Call capture hook directly (avoid dict lookup in hot path)
            seq_lens = metadata.get("seq_lens")
            _capture_hook_full(state, layer_idx, hidden, request_ids, seq_lens)
        except Exception:  # pragma: no cover - graceful degradation
            pass

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
                # Create writable copy to avoid PyTorch warning about non-writable buffers
                tensor = torch.frombuffer(bytearray(raw), dtype=storage_dtype).clone().reshape(shape_tuple)
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


def _record_step_context(
    state: _SteeringState,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """Record batch context with precomputed slice ranges for efficient routing."""
    if not request_ids:
        state.current_step_context = None
        return

    # Use seq_lens if available, otherwise assume 1 token per request (decode phase)
    if seq_lens is None or len(seq_lens) != len(request_ids):
        scheduled_tokens = tuple([1] * len(request_ids))
    else:
        scheduled_tokens = tuple(seq_lens)

    # Precompute cumulative slice ranges ONCE per step
    slice_ranges = []
    offset = 0
    for token_count in scheduled_tokens:
        start = offset
        end = offset + token_count
        slice_ranges.append((start, end))
        offset = end

    state.current_step_context = _StepContext(
        step_index=state.global_step,
        request_ids=tuple(request_ids),
        scheduled_tokens=scheduled_tokens,
        slice_ranges=tuple(slice_ranges),
    )
    state.global_step += 1


def _patch_model_runner(worker: Any, state: _SteeringState) -> None:
    """Patch GPUModelRunner.execute_model to capture per-step batch metadata."""
    if not _CAPTURE_METADATA_ENABLED:
        logger.info("CHATSPACE_CAPTURE_METADATA=0; skipping model runner patch.")
        return
    model_runner = worker.model_runner
    logger.info(
        f"_patch_model_runner: model_runner type={type(model_runner).__name__}, "
        f"has execute_model={hasattr(model_runner, 'execute_model')}"
    )
    
    if not hasattr(model_runner, "execute_model"):
        logger.error(f"model_runner does not have execute_model method! Available methods: {[m for m in dir(model_runner) if not m.startswith('_')][:20]}")
        return
    
    original_execute = getattr(model_runner, "_original_execute_model", None)

    if original_execute is not None:
        # Already patched
        logger.info("Model runner already patched, skipping")
        return

    original_execute = model_runner.execute_model
    logger.info(
        f"Patching model_runner.execute_model: original={original_execute}, "
        f"model_runner type={type(model_runner).__name__}"
    )

    def patched_execute_model(model_input: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept execute_model to capture batch metadata."""
        if not state.active_capture_requests:
            return original_execute(model_input, *args, **kwargs)
        logger.debug(
            f"patched_execute_model called: model_input type={type(model_input).__name__}, "
            f"has scheduled_new_reqs={hasattr(model_input, 'scheduled_new_reqs')}, "
            f"has scheduled_cached_reqs={hasattr(model_input, 'scheduled_cached_reqs')}, "
            f"global_step={state.global_step}"
        )
        # Log all attributes of model_input for debugging
        attrs = [attr for attr in dir(model_input) if not attr.startswith("_")]
        logger.debug(f"model_input attributes: {attrs[:20]}")  # Limit to first 20
            
        # Extract request IDs and sequence lengths from model_input
        try:
            request_ids = None
            seq_lens = None

            # V1 engine: model_input is SchedulerOutput
            # It has scheduled_new_reqs (list) and scheduled_cached_reqs (CachedRequestData object)
            if hasattr(model_input, "scheduled_new_reqs") and hasattr(model_input, "scheduled_cached_reqs"):
                logger.debug("Found scheduled_new_reqs and scheduled_cached_reqs attributes")
                
                # Extract from NEW requests (prefill phase)
                new_reqs_val = model_input.scheduled_new_reqs
                if new_reqs_val:
                    new_reqs = new_reqs_val
                    logger.debug(f"scheduled_new_reqs: type={type(new_reqs)}, len={len(new_reqs) if isinstance(new_reqs, list) else 'N/A'}")
                    if not isinstance(new_reqs, list):
                        new_reqs = [new_reqs]

                    request_ids = []
                    seq_lens = []
                    for req in new_reqs:
                        logger.debug(f"Processing new_req: type={type(req).__name__}, has req_id={hasattr(req, 'req_id')}, has prompt_token_ids={hasattr(req, 'prompt_token_ids')}")
                        if hasattr(req, "req_id"):
                            request_ids.append(req.req_id)
                        # For NewRequestData, use len(prompt_token_ids)
                        if hasattr(req, "prompt_token_ids"):
                            seq_lens.append(len(req.prompt_token_ids))

                # Also handle CACHED requests (decode phase)
                cached_reqs_val = model_input.scheduled_cached_reqs
                if cached_reqs_val and hasattr(cached_reqs_val, "req_ids"):
                    cached = cached_reqs_val
                    num_reqs = getattr(cached, "num_reqs", None)
                    logger.debug(f"scheduled_cached_reqs: type={type(cached)}, num_reqs={num_reqs}, has req_ids={hasattr(cached, 'req_ids')}")
                    if cached.num_reqs > 0 and cached.req_ids:
                        if not request_ids:
                            request_ids = []
                            seq_lens = []

                        # Cached requests are generating 1 token each (decode)
                        request_ids.extend(cached.req_ids)
                        seq_lens.extend([1] * len(cached.req_ids))
            else:
                logger.debug(
                    f"model_input does not have expected attributes. "
                    f"has scheduled_new_reqs={hasattr(model_input, 'scheduled_new_reqs')}, "
                    f"has scheduled_cached_reqs={hasattr(model_input, 'scheduled_cached_reqs')}"
                )

            if request_ids:
                # Store metadata for this step (legacy)
                current_step = state.global_step
                state.step_metadata[current_step] = {
                    "request_ids": request_ids,
                    "seq_lens": seq_lens,  # Can be None if not extractable
                    "step": current_step,
                }
                logger.debug(
                    f"Stored step_metadata for step {current_step}: "
                    f"request_ids={request_ids}, seq_lens={seq_lens}"
                )
                # Record step context with precomputed slicing (new)
                _record_step_context(state, request_ids, seq_lens)
            else:
                logger.debug("No request_ids extracted from model_input")
        except Exception as e:
            # Log exception details instead of silently failing
            logger.warning(
                f"Failed to extract metadata from model_input: {type(e).__name__}: {e}",
                exc_info=True
            )

        # Call original execute_model
        return original_execute(model_input, *args, **kwargs)

    model_runner.execute_model = patched_execute_model
    model_runner._original_execute_model = original_execute


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
        active_capture_requests={},
        request_captures={},
        request_prefill_buffers={},
        request_decode_buffers={},
        request_last_phase={},
        request_token_counts={},
        step_metadata={},
        global_step=0,
        current_step_context=None,
        transfer_stream=torch.cuda.Stream(device=device) if device.type == 'cuda' else None,
        request_pending_transfers={},
    )
    worker._chatspace_steering = state

    # Patch model runner to capture batch metadata for per-request activation tracking
    logger.info(f"About to patch model runner. worker.model_runner type: {type(worker.model_runner).__name__}")
    logger.info(f"worker.model_runner has execute_model: {hasattr(worker.model_runner, 'execute_model')}")
    _patch_model_runner(worker, state)
    logger.info(f"After patching, execute_model type: {type(worker.model_runner.execute_model).__name__}")

    if not isinstance(worker.model_runner.model, _SteeredModelWrapper):
        worker.model_runner.model = _SteeredModelWrapper(model, state)

    # Attach state and layer indices to all layers so forward hooks can access them
    layers = _resolve_layers(worker.model_runner.model)
    for layer_idx, layer in enumerate(layers):
        setattr(layer, "_chatspace_steering_state", state)
        setattr(layer, "_chatspace_layer_index", layer_idx)

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


def _coalesce_prefill_chunks(state: _SteeringState, request_id: str) -> None:
    """Concatenate buffered prefill chunks into final tensor."""
    if request_id not in state.request_prefill_buffers:
        return

    buffers = state.request_prefill_buffers[request_id]
    for layer_idx, chunks in buffers.items():
        if not chunks:
            continue

        # Concatenate all chunks along sequence dimension
        coalesced = torch.cat(chunks, dim=0)  # [total_prefill_tokens, hidden_size]
        state.request_captures[request_id][layer_idx] = coalesced

        # Clear buffer
        chunks.clear()


def _flush_decode_buffers(state: _SteeringState, request_id: str, start_async_transfer: bool = False) -> None:
    """Flush any remaining decode tokens from buffers into final captures.

    Parameters
    ----------
    start_async_transfer : bool
        If True and CUDA stream available, immediately start async GPU→CPU transfer.
        This allows transfers to overlap with generation for Phase 2 streaming.
    """
    if state.request_decode_buffers is None or request_id not in state.request_decode_buffers:
        return

    decode_buffers = state.request_decode_buffers[request_id]
    for layer_idx, decode_buf in decode_buffers.items():
        if not decode_buf:
            continue

        # Concatenate buffered decode tokens
        batched = torch.cat(decode_buf, dim=0) if len(decode_buf) > 1 else decode_buf[0]
        decode_buf.clear()

        # Append to existing captures or create new
        req_captures = state.request_captures[request_id]
        if layer_idx in req_captures:
            final_tensor = torch.cat([req_captures[layer_idx], batched], dim=0)
        else:
            final_tensor = batched
        req_captures[layer_idx] = final_tensor

        # Phase 2: Start async transfer immediately if requested
        if start_async_transfer and state.transfer_stream is not None:
            with torch.cuda.stream(state.transfer_stream):
                # Make contiguous on GPU before async transfer
                gpu_tensor = final_tensor.detach().contiguous()
                cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
                event = torch.cuda.Event()
                event.record(state.transfer_stream)

                # Store for later synchronization in fetch
                if request_id not in state.request_pending_transfers:
                    state.request_pending_transfers[request_id] = {}
                state.request_pending_transfers[request_id][layer_idx] = (cpu_tensor, event)


def register_capture_request(
    worker: Any, request_id: str, layer_indices: list[int]
) -> None:
    """Register request and prepare buffers for per-request capture."""
    if not layer_indices:
        return
    state = _ensure_state(worker)

    # Ensure decode buffers dict exists (migration from older state)
    if not hasattr(state, 'request_decode_buffers') or state.request_decode_buffers is None:
        state.request_decode_buffers = {}

    # Ensure pending transfers dict exists (migration from older state)
    if not hasattr(state, 'request_pending_transfers') or state.request_pending_transfers is None:
        state.request_pending_transfers = {}

    logger.debug(f"register_capture_request: request_id={request_id}, layer_indices={layer_indices}")
    state.active_capture_requests[request_id] = set(layer_indices)
    state.request_captures[request_id] = {}
    state.request_prefill_buffers[request_id] = {idx: [] for idx in layer_indices}
    state.request_decode_buffers[request_id] = {idx: [] for idx in layer_indices}
    state.request_pending_transfers[request_id] = {}
    state.request_last_phase[request_id] = "prefill"
    state.request_token_counts[request_id] = 0


def fetch_request_activations(worker: Any, request_id: str) -> dict[int, Any]:
    """Fetch activations, coalesce any remaining chunks, serialize, and cleanup."""
    state = _ensure_state(worker)

    # Coalesce any remaining prefill chunks
    _coalesce_prefill_chunks(state, request_id)
    # Flush any remaining decode buffers
    _flush_decode_buffers(state, request_id)

    # Serialize captures
    captures = state.request_captures.pop(request_id, {})
    serialized = {
        layer_idx: serialize_tensor(tensor)
        for layer_idx, tensor in captures.items()
    }

    # Cleanup
    state.active_capture_requests.pop(request_id, None)
    state.request_prefill_buffers.pop(request_id, None)
    if state.request_decode_buffers is not None:
        state.request_decode_buffers.pop(request_id, None)
    state.request_last_phase.pop(request_id, None)
    state.request_token_counts.pop(request_id, None)

    return serialized


def fetch_batch_captures(worker: Any, request_ids: list[str]) -> dict[str, dict[int, Any]]:
    """Fetch multiple requests' captures in one RPC call.

    Parameters
    ----------
    request_ids : list[str]
        List of request IDs to fetch captures for.

    Returns
    -------
    dict[str, dict[int, Any]]
        Mapping of request_id to layer captures. Each capture is a serialized tensor.
    """
    import time

    state = _ensure_state(worker)
    result: dict[str, dict[int, Any]] = {}

    # Timing accumulators
    total_coalesce_time = 0.0
    total_numpy_time = 0.0
    total_bytes = 0
    layer_count = 0

    profiling_metadata: dict[str, Any] = {
        "request_count": len(request_ids),
        "device": str(state.device),
    }

    with _profile_fetch_batch(state, profiling_metadata) as profiler:
        # Phase 1: Coalesce all captures
        t_coalesce_start = time.perf_counter()
        for req_id in request_ids:
            _coalesce_prefill_chunks(state, req_id)
            _flush_decode_buffers(state, req_id)
        total_coalesce_time = time.perf_counter() - t_coalesce_start

        # Phase 2: Collect pre-transferred data and start new transfers for remaining data
        transfer_data: dict[tuple[str, int], tuple[torch.Tensor, torch.cuda.Event | None, int]] = {}

        for req_id in request_ids:
            # First check for pre-transferred data from Phase 2 streaming
            pending = state.request_pending_transfers.get(req_id, {}) if state.request_pending_transfers else {}

            # Get current captures (may include data not yet transferred)
            captures = state.request_captures.get(req_id, {})

            for layer_idx, tensor in captures.items():
                layer_count += 1
                tensor_bytes = tensor.numel() * tensor.element_size()
                total_bytes += tensor_bytes

                # Check if this layer was pre-transferred during generation
                if layer_idx in pending:
                    # Use pre-transferred data (already in flight or complete)
                    cpu_tensor, event = pending[layer_idx]
                    transfer_data[(req_id, layer_idx)] = (cpu_tensor, event, tensor_bytes)
                else:
                    # This layer wasn't pre-transferred, start transfer now
                    # (e.g., remaining decode tokens that didn't reach flush threshold)
                    if state.transfer_stream is not None:
                        with torch.cuda.stream(state.transfer_stream):
                            gpu_tensor = tensor.detach().contiguous()
                            cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(state.transfer_stream)
                            transfer_data[(req_id, layer_idx)] = (cpu_tensor, event, tensor_bytes)
                    else:
                        # Fallback for non-CUDA devices
                        cpu_tensor = tensor.detach().cpu().contiguous()
                        transfer_data[(req_id, layer_idx)] = (cpu_tensor, None, tensor_bytes)

        # Phase 3: Synchronize all events before serialization
        if state.transfer_stream is not None:
            for cpu_tensor, event, _ in transfer_data.values():
                if event is not None:
                    event.synchronize()

        # Phase 4: Serialize all transferred tensors
        for req_id in request_ids:
            captures = state.request_captures.pop(req_id, {})
            serialized = {}

            for layer_idx in captures.keys():
                cpu_tensor, _, tensor_bytes = transfer_data[(req_id, layer_idx)]

                # Time numpy serialization
                t_numpy_start = time.perf_counter()
                dtype_name = str(cpu_tensor.dtype).removeprefix("torch.")
                storage = cpu_tensor
                storage_dtype_name = dtype_name
                if cpu_tensor.numel() == 0:
                    buffer = b""
                else:
                    try:
                        buffer = storage.numpy().tobytes()
                    except TypeError:
                        storage = cpu_tensor.to(dtype=torch.float32)
                        storage_dtype_name = str(storage.dtype).removeprefix("torch.")
                        buffer = storage.numpy().tobytes()
                total_numpy_time += time.perf_counter() - t_numpy_start

                payload: dict[str, Any] = {
                    "dtype": dtype_name,
                    "shape": list(cpu_tensor.shape),
                    "data": buffer,
                    "encoding": "bytes",
                }
                if storage_dtype_name != dtype_name:
                    payload["storage_dtype"] = storage_dtype_name

                serialized[layer_idx] = payload

            # Cleanup
            state.active_capture_requests.pop(req_id, None)
            state.request_prefill_buffers.pop(req_id, None)
            state.request_last_phase.pop(req_id, None)
            state.request_token_counts.pop(req_id, None)
            if state.request_decode_buffers is not None:
                state.request_decode_buffers.pop(req_id, None)
            if state.request_pending_transfers is not None:
                state.request_pending_transfers.pop(req_id, None)

            result[req_id] = serialized

        # Populate profiling metadata for summary/logging
        profiling_metadata["layer_count"] = layer_count
        profiling_metadata["total_bytes"] = total_bytes
        profiling_metadata["total_numpy_time_s"] = total_numpy_time
        profiling_metadata["total_coalesce_time_s"] = total_coalesce_time

    profile_summary = (
        getattr(profiler, "_chatspace_summary", None) if profiler is not None else None
    )
    transfer_cuda_s = (
        float(profile_summary.get("transfer_cuda_s", 0.0))
        if profile_summary
        else 0.0
    )

    # Log detailed timing breakdown
    if layer_count > 0:
        total_mb = total_bytes / (1024 * 1024)
        log_parts = [
            f"Requests={len(request_ids)}",
            f"Layers={layer_count}",
            f"Data={total_mb:.1f}MB",
            f"Coalesce={total_coalesce_time:.4f}s",
            f"NumPy={total_numpy_time:.4f}s",
        ]
        if transfer_cuda_s > 0.0:
            log_parts.append(f"GPU→CPU(cuda)={transfer_cuda_s:.4f}s")
            if total_mb > 0:
                log_parts.append(f"Transfer rate={total_mb/transfer_cuda_s:.1f}MB/s")
        logger.info("[Fetch Timing] " + ", ".join(log_parts))

    return result


def unregister_capture_request(worker: Any, request_id: str) -> None:
    """Clean up aborted or cancelled capture request."""
    state = _ensure_state(worker)
    state.request_captures.pop(request_id, None)
    state.active_capture_requests.pop(request_id, None)
    state.request_prefill_buffers.pop(request_id, None)
    if state.request_decode_buffers is not None:
        state.request_decode_buffers.pop(request_id, None)
    state.request_last_phase.pop(request_id, None)
    state.request_token_counts.pop(request_id, None)
    if state.request_pending_transfers is not None:
        state.request_pending_transfers.pop(request_id, None)


def fetch_last_profile(worker: Any) -> dict[str, Any]:
    """Return the most recent torch profiler summary, if available."""
    state = _ensure_state(worker)
    summary = state.last_fetch_profile
    if summary is None:
        return {}
    return summary


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
    register_capture_request,
    fetch_request_activations,
    fetch_batch_captures,
    fetch_last_profile,
    unregister_capture_request,
):
    _register_rpc(_rpc_fn.__name__, _rpc_fn)

ensure_collective_rpc_gateway_installed()
