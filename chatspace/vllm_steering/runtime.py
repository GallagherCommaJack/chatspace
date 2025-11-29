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

import atexit
import importlib
import logging
import math
import os
import queue
import threading
import time
import uuid
import warnings
from contextlib import contextmanager, nullcontext
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

# Use vLLM's logger to ensure logs appear in worker processes
try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

try:
    import torch._dynamo as _dynamo
except ImportError:
    _dynamo = None

try:
    from torch.profiler import ProfilerActivity, profile as torch_profile
except ImportError:
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

# Performance counters for profiling hot path (disabled by default, no overhead)
_PERF_COUNTERS_ENABLED = _env_flag("CHATSPACE_PERF_COUNTERS", False)
_PERF_COUNTERS: dict[str, int] = {}
_PERF_TIMINGS: dict[str, float] = {}  # Cumulative time in seconds


def _perf_incr(name: str, delta: int = 1) -> None:
    """Increment a counter (no-op if profiling disabled)."""
    if _PERF_COUNTERS_ENABLED:
        _PERF_COUNTERS[name] = _PERF_COUNTERS.get(name, 0) + delta


def _perf_add_time(name: str, seconds: float) -> None:
    """Add cumulative time to a timing bucket (no-op if profiling disabled)."""
    if _PERF_COUNTERS_ENABLED:
        _PERF_TIMINGS[name] = _PERF_TIMINGS.get(name, 0.0) + seconds


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
    except OSError as e:
        logger.debug(f"Failed to create trace directory {trace_dir}: {e}")
        return None

    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    suffix = f"{time.time():.6f}".replace(".", "")
    filename = f"{_PROFILE_FETCH_TRACE_PREFIX}_{timestamp}_{suffix}.json"
    path = trace_dir / filename
    try:
        prof.export_chrome_trace(str(path))
    except (OSError, IOError) as e:
        logger.debug(f"Failed to save trace to {path}: {e}")
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
    metadata["activities"] = [activity.name.lower() for activity in activities]

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
class _SteeringState:
    """Track steering metadata for a worker."""

    hidden_size: int
    dtype: torch.dtype
    device: torch.device

    # Per-request activation capture
    active_capture_requests: dict[str, set[int]] = None  # request_id -> layer indices
    request_captures: dict[str, dict[int, torch.Tensor]] = None  # request_id -> layer -> tensor
    request_prefill_buffers: dict[str, dict[int, list[torch.Tensor]]] = None  # request_id -> layer -> chunks
    request_decode_buffers: dict[str, dict[int, list[torch.Tensor]]] = None  # request_id -> layer -> decode tokens
    request_last_phase: dict[str, str] = None  # request_id -> "prefill" or "decode"
    request_token_counts: dict[str, int] = None  # request_id -> token count

    # Per-request steering
    request_steering_specs: dict[str, Any] = None  # request_id -> SteeringSpec (Any to avoid circular import)

    # Per-step batch metadata (captured from model runner)
    step_metadata: dict[int, dict[str, Any]] = None  # step_number -> {request_ids, seq_lens, ...}
    global_step: int = 0  # Monotonically increasing step counter

    # Async transfer infrastructure
    transfer_stream: torch.cuda.Stream | None = None  # Stream for non-blocking GPU→CPU transfers
    request_pending_transfers: dict[str, dict[int, tuple[torch.Tensor, torch.cuda.Event]]] = None  # request_id -> layer -> (cpu_tensor, event)

    # Profiling summaries
    last_fetch_profile: dict[str, Any] | None = None

    # Shared memory IPC for zero-copy activation extraction
    active_shared_memory: dict[str, tuple[Any, float]] = None  # shm_name -> (SharedMemory, timestamp)
    shm_lock: threading.Lock = None  # Protects active_shared_memory dict access
    shm_cleanup_thread: Any = None  # threading.Thread for TTL cleanup

    # Shared memory configuration (SHM is always enabled for captures)
    shm_ttl_seconds: int = 600
    shm_max_gb: float = 128.0

    # Capture configuration
    decode_buffer_size: int = 128  # Flush decode buffer every N tokens


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

    def __getattr__(self, name: str) -> Any:
        if name in {"_wrapped_model", "_steering_state"}:
            return object.__getattribute__(self, name)
        return getattr(self._wrapped_model, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
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
    except ImportError:
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
        hidden = output.last_hidden_state
        if isinstance(hidden, torch.Tensor):
            return hidden
    raise TypeError(f"Cannot extract hidden state from output of type {type(output).__name__}")


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
        hidden = output.last_hidden_state
        output.last_hidden_state = transform(hidden)
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
    vector = deserialize_tensor(
        payload,
        device=dest.device,
        dtype=target_dtype,
    )
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
    _apply_projection_cap = _dynamo.disable(_apply_projection_cap)
    _apply_ablation = _dynamo.disable(_apply_ablation)


if _dynamo is not None:
    _transform_output = _dynamo.disable(_transform_output)
    _apply_vector_to_output = _dynamo.disable(_apply_vector_to_output)
    _apply_projection_cap_to_output = _dynamo.disable(_apply_projection_cap_to_output)
    _apply_ablation_to_output = _dynamo.disable(_apply_ablation_to_output)



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
                req_hidden.detach()
            )
            state.request_last_phase[req_id] = "prefill"
        else:
            # Check if transitioning from prefill to decode
            if state.request_last_phase.get(req_id) == "prefill":
                _coalesce_prefill_chunks(state, req_id)
                state.request_last_phase[req_id] = "decode"

            # Buffer decode tokens to reduce concatenation overhead
            decode_buf = state.request_decode_buffers[req_id][layer_idx]
            decode_buf.append(req_hidden.detach())

            # Flush buffer periodically to reduce cat operations
            if len(decode_buf) >= state.decode_buffer_size:
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



def _apply_per_request_steering(
    output: Any,
    state: _SteeringState,
    layer_idx: int,
    request_ids: list[str],
    seq_lens: list[int] | None,
    cached_hidden: torch.Tensor | None = None,
) -> Any:
    """Apply per-request steering by slicing, transforming, and concatenating.

    This function extracts the hidden state tensor, slices it per request,
    applies each request's steering configuration (if present), and
    reconstructs the output.

    Args:
        cached_hidden: Pre-extracted hidden state to avoid double extraction
    """
    # Use cached hidden if available, otherwise extract
    hidden = cached_hidden if cached_hidden is not None else _extract_hidden_from_output(output)
    if hidden is None or hidden.dim() != 2:
        return output

    # Handle case where seq_lens is None (single request or no metadata)
    if seq_lens is None or len(seq_lens) != len(request_ids):
        # Fall back to treating entire batch as single request
        if len(request_ids) == 1:
            req_id = request_ids[0]
            spec = state.request_steering_specs.get(req_id)
            if spec is not None and layer_idx in spec.layers:
                layer_spec = spec.layers[layer_idx]
                if layer_spec.operations:  # Non-empty operations list
                    return _apply_layer_steering_to_output(output, layer_spec, state)
        return output

    # Apply steering in-place to avoid shape mismatch with padded tensors
    # vLLM may pad tensors to larger sizes internally, but seq_lens only covers actual tokens
    transformed_hidden = hidden.clone()
    start_idx = 0

    for i, req_id in enumerate(request_ids):
        seq_len = seq_lens[i]
        end_idx = start_idx + seq_len

        # Apply steering if this request has a spec for this layer
        spec = state.request_steering_specs.get(req_id)
        if spec is not None and layer_idx in spec.layers:
            layer_spec = spec.layers[layer_idx]
            if layer_spec.operations:  # Non-empty operations list
                # Apply steering to the slice and write back
                transformed_hidden[start_idx:end_idx] = _apply_layer_steering_to_hidden(
                    hidden[start_idx:end_idx], layer_spec, state
                )

        start_idx = end_idx

    # Reconstruct output with transformed hidden state (same shape as original)
    return _reconstruct_output_with_hidden(output, hidden, transformed_hidden)


def _apply_layer_steering_to_hidden(
    hidden: torch.Tensor,
    layer_spec: Any,  # DeserializedLayerSpec from register_steering_spec
    state: _SteeringState,
) -> torch.Tensor:
    """Apply steering operations to a hidden state tensor.

    Operations are applied in sequence order from layer_spec.operations.
    Each operation is a tuple: (op_type, vector, params)
    - "add": (vector, None)
    - "cap": (vector, (min, max))
    - "ablation": (vector, scale)
    """
    ops = layer_spec.operations
    if not ops:
        return hidden

    # Fast path: single operation (most common case)
    if len(ops) == 1:
        op_type, vec, params = ops[0]
        if op_type == "add":
            return hidden + vec
        elif op_type == "cap":
            cap_min, cap_max = params
            return _apply_projection_cap(
                hidden, _ProjectionCapConfig(unit_vector=vec, min=cap_min, max=cap_max)
            )
        else:  # ablation
            return _apply_ablation(
                hidden, _AblationConfig(unit_vector=vec, scale=params)
            )

    # General path: multiple operations in sequence
    for op_type, vec, params in ops:
        if op_type == "add":
            hidden = hidden + vec
        elif op_type == "cap":
            cap_min, cap_max = params
            hidden = _apply_projection_cap(
                hidden, _ProjectionCapConfig(unit_vector=vec, min=cap_min, max=cap_max)
            )
        else:  # ablation
            hidden = _apply_ablation(
                hidden, _AblationConfig(unit_vector=vec, scale=params)
            )

    return hidden


def _apply_layer_steering_to_output(
    output: Any,
    layer_spec: Any,  # LayerSteeringSpec
    state: _SteeringState,
) -> Any:
    """Apply layer steering to the entire output (for single-request case)."""
    def transform(hidden: torch.Tensor) -> torch.Tensor:
        return _apply_layer_steering_to_hidden(hidden, layer_spec, state)

    return _transform_output(output, transform, fallback=None, mode="hidden")


def _reconstruct_output_with_hidden(
    output: Any,
    original_hidden: torch.Tensor,
    transformed_hidden: torch.Tensor,
) -> Any:
    """Reconstruct output structure with transformed hidden state."""
    # Handle Qwen-style (delta, residual) output
    if _is_qwen_layer_output(output):
        delta, residual = output[0], output[1]
        # transformed_hidden = residual + new_delta
        # new_delta = transformed_hidden - residual
        # Must be contiguous for vLLM's subsequent operations (rotary embedding)
        new_delta = (transformed_hidden - residual).contiguous()
        return (new_delta,) + output[1:]

    # Handle tensor output
    if isinstance(output, torch.Tensor):
        return transformed_hidden

    # Handle tuple/list output
    if isinstance(output, (tuple, list)):
        if isinstance(output, tuple):
            return (transformed_hidden,) + output[1:]
        else:
            return [transformed_hidden] + output[1:]

    # Handle dict output
    if isinstance(output, dict) and "last_hidden_state" in output:
        patched = dict(output)
        patched["last_hidden_state"] = transformed_hidden
        return patched

    # Handle object with last_hidden_state attribute
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = transformed_hidden
        return output

    # Fallback: return transformed_hidden directly
    return transformed_hidden


def _patch_decoder_layer_class(layer_cls: type) -> None:
    if layer_cls in _PATCHED_CLASSES:
        return
    original_init = layer_cls.__init__
    original_forward = layer_cls.forward

    @wraps(original_init)
    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # No longer need module parameters for global steering

    @wraps(original_forward)
    def _patched_forward(self, *args: Any, **kwargs: Any) -> Any:
        _perf_incr("forward_calls")

        output = original_forward(self, *args, **kwargs)

        state = getattr(self, "_chatspace_steering_state", None)
        layer_idx = getattr(self, "_chatspace_layer_index", None)

        # Fast path: no state attached means no steering/capture possible
        if state is None:
            _perf_incr("exit_no_state")
            return output

        # Extract request metadata for per-request steering
        request_ids = None
        seq_lens = None
        current_step = state.global_step - 1
        if current_step >= 0:
            metadata = state.step_metadata.get(current_step)
            if metadata is not None:
                request_ids = metadata.get("request_ids")
                seq_lens = metadata.get("seq_lens")

        if not request_ids:
            _perf_incr("exit_no_requests")
            return output

        # Check if we have per-request steering specs for any request in the batch
        has_per_request_steering = False
        if layer_idx is not None:
            for req_id in request_ids:
                spec = state.request_steering_specs.get(req_id)
                if spec is not None and layer_idx in spec.layers:
                    layer_spec = spec.layers[layer_idx]
                    if layer_spec.operations:  # Non-empty operations list
                        has_per_request_steering = True
                        break

        # Check if capture is active (empty dict is falsy, so this is correct)
        has_active_capture = bool(state.active_capture_requests)

        # Fast path: no steering and no capture for this forward
        if not has_per_request_steering and not has_active_capture:
            _perf_incr("exit_no_work")
            return output

        # Extract hidden state once for both steering and capture
        # This avoids double extraction when both are active
        _perf_incr("extraction_triggered")
        if _PERF_COUNTERS_ENABLED:
            _t0 = time.perf_counter()
        cached_hidden = _extract_hidden_from_output(output)
        if _PERF_COUNTERS_ENABLED:
            _perf_add_time("extraction_time", time.perf_counter() - _t0)

        # Apply per-request steering if needed
        if has_per_request_steering:
            _perf_incr("steering_applied")
            output = _apply_per_request_steering(
                output, state, layer_idx, request_ids, seq_lens, cached_hidden=cached_hidden
            )
            # After steering, output has changed, so invalidate cached_hidden
            # We need to re-extract for capture
            if has_active_capture:
                _perf_incr("extraction_post_steering")
                cached_hidden = _extract_hidden_from_output(output)

        # Handle activation capture
        if not has_active_capture:
            return output

        if layer_idx is None:
            _perf_incr("exit_capture_no_layer_idx")
            return output

        _perf_incr("capture_processed")
        hidden = cached_hidden if cached_hidden is not None else _extract_hidden_from_output(output)
        if hidden is None or hidden.dim() != 2:
            _perf_incr("exit_capture_bad_hidden")
            return output

        _capture_hook_full(state, layer_idx, hidden, request_ids, seq_lens)

        return output

    layer_cls.__init__ = _patched_init
    layer_cls.forward = _patched_forward
    _PATCHED_CLASSES.add(layer_cls)


def ensure_layer_patch_installed() -> None:
    """Patch known Qwen and Llama decoder layers so CUDA graphs include steering."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    for module_name, class_name in _PATCH_TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        layer_cls = getattr(module, class_name, None)
        if layer_cls is None:
            continue
        _patch_decoder_layer_class(layer_cls)
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




def deserialize_tensor(
    payload: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    shm_objects_list: list[SharedMemory] | None = None
) -> torch.Tensor:
    """Reconstruct a tensor from serialized format.

    Supports three payload formats:
    1. torch.Tensor - Direct passthrough (msgpack deserialization fallback)
    2. dict with encoding="bytes" - Steering vectors (client→worker RPC)
    3. dict with encoding="shm" - Activation captures (worker→client, uint8 view)

    Parameters
    ----------
    payload : Any
        Serialized tensor data (torch.Tensor or dict with encoding metadata)
    device : torch.device, optional
        Target device for the tensor
    dtype : torch.dtype, optional
        Target dtype for the tensor (overrides payload dtype if specified)
    shm_objects_list : list[SharedMemory], optional
        If provided and payload uses shared memory (encoding=="shm"),
        the SharedMemory object will be appended to this list to keep it alive
        and prevent garbage collection that would invalidate tensor views
    """
    # Format 1: Raw tensor (msgpack deserialization fallback)
    if isinstance(payload, torch.Tensor):
        tensor = payload
    # Format 2 & 3: Dict with encoding metadata
    elif isinstance(payload, dict):
        encoding = payload.get("encoding")
        dtype_str = payload.get("dtype")
        shape = payload.get("shape")

        if dtype_str is None or shape is None:
            raise TypeError(f"Missing required tensor metadata: dtype={dtype_str}, shape={shape}")

        target_dtype = getattr(torch, dtype_str, None)
        if target_dtype is None:
            raise TypeError(f"Unsupported tensor dtype: {dtype_str}")
        shape_tuple = tuple(int(dim) for dim in shape)

        # Format 2: Bytes encoding (steering vectors via RPC)
        if encoding == "bytes":
            data = payload.get("data")
            if not isinstance(data, bytes):
                raise TypeError(f"Expected bytes for encoding=bytes, got {type(data)}")
            storage_dtype_str = payload.get("storage_dtype")
            storage_dtype = target_dtype
            if storage_dtype_str:
                storage_dtype = getattr(torch, storage_dtype_str, None)
                if not isinstance(storage_dtype, torch.dtype):
                    raise TypeError(f"Unsupported storage dtype: {storage_dtype_str}")
            # Create writable copy to avoid PyTorch warning about non-writable buffers
            tensor = torch.frombuffer(bytearray(data), dtype=storage_dtype).clone().reshape(shape_tuple)
            if storage_dtype != target_dtype:
                tensor = tensor.to(dtype=target_dtype)

        # Format 3: Shared memory encoding (activation captures, uint8 view)
        elif encoding == "shm":
            shm_name = payload.get("shm_name")
            nbytes = payload.get("nbytes")
            if shm_name is None:
                raise TypeError("Missing shm_name in shared memory payload")

            # Open existing shared memory segment
            shm = SharedMemory(name=shm_name)

            # Track SharedMemory object to keep it alive if list provided
            if shm_objects_list is not None:
                shm_objects_list.append(shm)

            # Read as uint8 bytes, reinterpret to target dtype
            # If nbytes is provided (new format), use it; otherwise compute from shape
            if nbytes is not None:
                np_array = np.ndarray(nbytes, dtype=np.uint8, buffer=shm.buf)
            else:
                # Legacy fallback: compute nbytes from shape and dtype
                element_size = torch.empty(0, dtype=target_dtype).element_size()
                computed_nbytes = int(np.prod(shape_tuple)) * element_size
                np_array = np.ndarray(computed_nbytes, dtype=np.uint8, buffer=shm.buf)

            # Reconstruct tensor from uint8 buffer
            tensor = torch.frombuffer(bytearray(np_array), dtype=torch.uint8)
            tensor = tensor.view(target_dtype).reshape(shape_tuple).clone()
        else:
            raise TypeError(f"Unsupported tensor encoding: {encoding}")
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


def _create_shared_tensor(
    tensor: torch.Tensor,
    state: _SteeringState,
) -> dict[str, Any]:
    """Create shared memory segment for tensor and return metadata.

    All tensors are transferred via shared memory using a dtype-agnostic uint8 view.
    This removes the dependency on ml-dtypes and works with any numpy version.

    Args:
        tensor: Tensor to share (can be on any device, will be moved to CPU)
        state: Worker steering state for tracking active segments

    Returns:
        Metadata dict with encoding="shm"

    Raises:
        RuntimeError: If shared memory creation fails (no fallback)
    """
    # Ensure tensor is on CPU and contiguous
    cpu_tensor = tensor.detach().cpu().contiguous()
    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()

    # Check memory limit
    max_gb = state.shm_max_gb
    with state.shm_lock:
        total_shm_bytes = sum(
            shm.size for shm, _ in state.active_shared_memory.values()
        )
    if (total_shm_bytes + nbytes) / (1024**3) > max_gb:
        raise RuntimeError(
            f"Shared memory limit reached ({max_gb}GB, current={total_shm_bytes/(1024**3):.2f}GB, "
            f"requested={nbytes/(1024**3):.2f}GB). Increase CHATSPACE_MAX_SHM_GB or reduce batch size."
        )

    # Generate unique name
    shm_name = f"chatspace_{uuid.uuid4().hex}"

    try:
        # Create shared memory segment
        shm = SharedMemory(create=True, size=nbytes, name=shm_name)

        # View tensor as flat uint8 bytes (works for any dtype including bfloat16)
        byte_view = cpu_tensor.view(torch.uint8).flatten()

        # Create uint8 numpy array backed by shared memory and copy data
        shm_array = np.ndarray(nbytes, dtype=np.uint8, buffer=shm.buf)
        shm_array[:] = byte_view.numpy()

        # Track in state with timestamp
        with state.shm_lock:
            state.active_shared_memory[shm_name] = (shm, time.time())

        # Return metadata with dtype as string for reconstruction
        dtype_name = str(cpu_tensor.dtype).removeprefix("torch.")
        return {
            "encoding": "shm",
            "shm_name": shm_name,
            "shape": list(cpu_tensor.shape),
            "dtype": dtype_name,
            "nbytes": nbytes,
        }
    except Exception as e:
        # Clean up partial state if segment was created
        shm_to_cleanup = None
        with state.shm_lock:
            if shm_name in state.active_shared_memory:
                shm_to_cleanup, _ = state.active_shared_memory.pop(shm_name)
        if shm_to_cleanup is not None:
            try:
                shm_to_cleanup.close()
                shm_to_cleanup.unlink()
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to create shared memory segment ({nbytes} bytes): {e}. "
            f"Check /dev/shm space and CHATSPACE_MAX_SHM_GB setting."
        ) from e


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
        import sys  # For debug printing
        if not state.active_capture_requests and not state.request_steering_specs:
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
            # IMPORTANT: vLLM V1 orders the hidden state tensor as [CACHED, NEW], not [NEW, CACHED]!
            if hasattr(model_input, "scheduled_new_reqs") and hasattr(model_input, "scheduled_cached_reqs"):
                logger.debug("Found scheduled_new_reqs and scheduled_cached_reqs attributes")

                request_ids = []
                seq_lens = []

                # Process CACHED requests FIRST (they appear first in the tensor)
                cached_reqs_val = model_input.scheduled_cached_reqs
                if cached_reqs_val and hasattr(cached_reqs_val, "req_ids"):
                    cached = cached_reqs_val
                    num_reqs = getattr(cached, "num_reqs", None)
                    logger.debug(f"scheduled_cached_reqs: type={type(cached)}, num_reqs={num_reqs}, has req_ids={hasattr(cached, 'req_ids')}")
                    if cached.num_reqs > 0 and cached.req_ids:
                        request_ids.extend(cached.req_ids)
                        seq_lens.extend([1] * len(cached.req_ids))

                # Process NEW requests SECOND (they appear after cached in the tensor)
                new_reqs_val = model_input.scheduled_new_reqs
                if new_reqs_val:
                    new_reqs = new_reqs_val
                    logger.debug(f"scheduled_new_reqs: type={type(new_reqs)}, len={len(new_reqs) if isinstance(new_reqs, list) else 'N/A'}")
                    if not isinstance(new_reqs, list):
                        new_reqs = [new_reqs]

                    for req in new_reqs:
                        logger.debug(f"Processing new_req: type={type(req).__name__}, has req_id={hasattr(req, 'req_id')}, has prompt_token_ids={hasattr(req, 'prompt_token_ids')}")
                        if hasattr(req, "req_id"):
                            request_ids.append(req.req_id)
                        if hasattr(req, "prompt_token_ids"):
                            seq_lens.append(len(req.prompt_token_ids))
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
                # Increment global_step so next batch stores at a different index
                state.global_step += 1

                # Clean up old metadata to prevent unbounded growth (keep last 1000 steps)
                if len(state.step_metadata) > 1000:
                    old_steps = sorted(state.step_metadata.keys())[:-1000]
                    for step in old_steps:
                        state.step_metadata.pop(step, None)
                    logger.debug(f"Cleaned up {len(old_steps)} old step_metadata entries")
            else:
                logger.debug("No request_ids extracted from model_input")
        except (AttributeError, TypeError, KeyError, IndexError) as e:
            # Catch expected exceptions from object introspection and data access
            # These can occur if vLLM's internal structure changes
            logger.warning(
                f"Failed to extract metadata from model_input: {type(e).__name__}: {e}",
                exc_info=True
            )
        except Exception as e:
            # Unexpected exception - log at ERROR level and re-raise
            logger.error(
                f"Unexpected error in metadata extraction: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise

        # Call original execute_model
        return original_execute(model_input, *args, **kwargs)

    model_runner.execute_model = patched_execute_model
    model_runner._original_execute_model = original_execute


def initialize_worker_state(
    worker: Any,
    layer_indices: Sequence[int] | None = None,
    shm_ttl_seconds: int = 600,
    shm_max_gb: float = 128.0,
    decode_buffer_size: int = 128,
) -> dict[str, Any]:
    """Install steering patch on worker after model load.

    Shared memory is always used for activation captures. There is no bytes fallback.
    """
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
        active_capture_requests={},
        request_captures={},
        request_prefill_buffers={},
        request_decode_buffers={},
        request_last_phase={},
        request_token_counts={},
        request_steering_specs={},
        step_metadata={},
        global_step=0,
        transfer_stream=torch.cuda.Stream(device=device) if device.type == 'cuda' else None,
        request_pending_transfers={},
        active_shared_memory={},
        shm_lock=threading.Lock(),
        shm_cleanup_thread=None,
        shm_ttl_seconds=shm_ttl_seconds,
        shm_max_gb=shm_max_gb,
        decode_buffer_size=decode_buffer_size,
    )
    worker._chatspace_steering = state

    # Start shared memory cleanup thread (always enabled)
    stop_event = threading.Event()
    cleanup_thread = threading.Thread(
        target=_cleanup_stale_shared_memory,
        args=(state, stop_event),
        daemon=True,
        name="chatspace-shm-cleanup"
    )
    cleanup_thread.start()
    state.shm_cleanup_thread = (cleanup_thread, stop_event)

    # Register atexit handler to stop thread on shutdown
    def _stop_cleanup_thread():
        if state.shm_cleanup_thread:
            thread, event = state.shm_cleanup_thread
            event.set()
            thread.join(timeout=5.0)
            logger.info("Shared memory cleanup thread stopped (atexit)")

    atexit.register(_stop_cleanup_thread)
    logger.info("Started shared memory cleanup thread")

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

    return {
        "hidden_size": hidden_size,
        "layer_count": len(layers),
        "dtype": str(dtype),
        "device": str(device),
    }








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

    logger.debug(f"register_capture_request: request_id={request_id}, layer_indices={layer_indices}")
    state.active_capture_requests[request_id] = set(layer_indices)
    state.request_captures[request_id] = {}
    state.request_prefill_buffers[request_id] = {idx: [] for idx in layer_indices}
    state.request_decode_buffers[request_id] = {idx: [] for idx in layer_indices}
    state.request_pending_transfers[request_id] = {}
    state.request_last_phase[request_id] = "prefill"
    state.request_token_counts[request_id] = 0


def register_steering_spec(
    worker: Any, request_id: str, serialized_spec: dict[str, Any]
) -> None:
    """Register a per-request steering spec on the worker.

    Parameters
    ----------
    worker : Any
        The vLLM worker instance.
    request_id : str
        Unique request identifier.
    serialized_spec : dict[str, Any]
        Serialized SteeringSpec with structure:
        {
            "layers": {
                layer_idx: {
                    "operations": [
                        {"type": "add", "vector": bytes},
                        {"type": "cap", "vector": bytes, "min": float, "max": float},
                        {"type": "ablation", "vector": bytes, "scale": float},
                    ]
                }
            }
        }
    """
    state = _ensure_state(worker)
    logger.debug(f"register_steering_spec: request_id={request_id}")

    # Deserialize the steering spec
    from dataclasses import dataclass, field
    from typing import Any as AnyType

    # Simple dataclass to hold deserialized layer specs
    # operations: list of (op_type, vector, params) tuples
    # - "add": (vector, None)
    # - "cap": (vector, (min, max))
    # - "ablation": (vector, scale)
    @dataclass
    class DeserializedLayerSpec:
        operations: list[tuple[str, torch.Tensor, AnyType]] = field(default_factory=list)

    @dataclass
    class DeserializedSteeringSpec:
        layers: dict[int, DeserializedLayerSpec]

    deserialized_layers = {}
    for layer_idx_str, layer_data in serialized_spec["layers"].items():
        layer_idx = int(layer_idx_str)
        operations = []

        for op in layer_data.get("operations", []):
            op_type = op["type"]
            if op_type == "add":
                vector = deserialize_tensor(op["vector"], dtype=state.dtype, device=state.device)
                operations.append(("add", vector, None))
            elif op_type == "cap":
                vector = deserialize_tensor(op["vector"], dtype=torch.float32, device=state.device)
                operations.append(("cap", vector, (op["min"], op["max"])))
            elif op_type == "ablation":
                vector = deserialize_tensor(op["vector"], dtype=torch.float32, device=state.device)
                operations.append(("ablation", vector, op["scale"]))

        deserialized_layers[layer_idx] = DeserializedLayerSpec(operations=operations)

    # Store the deserialized spec in the state
    state.request_steering_specs[request_id] = DeserializedSteeringSpec(
        layers=deserialized_layers
    )


def unregister_steering_spec(worker: Any, request_id: str) -> None:
    """Unregister a per-request steering spec from the worker."""
    state = _ensure_state(worker)
    # Raise if request_id doesn't exist - indicates bug in caller's request tracking
    state.request_steering_specs.pop(request_id)
    logger.debug(f"unregister_steering_spec: request_id={request_id}")


def fetch_request_activations(worker: Any, request_id: str) -> dict[int, Any]:
    """Fetch activations, coalesce any remaining chunks, serialize via SHM, and cleanup."""
    state = _ensure_state(worker)

    # Coalesce any remaining prefill chunks
    _coalesce_prefill_chunks(state, request_id)
    # Flush any remaining decode buffers
    _flush_decode_buffers(state, request_id)

    # Serialize captures via shared memory
    captures = state.request_captures.pop(request_id, {})
    serialized = {
        layer_idx: _create_shared_tensor(tensor, state)
        for layer_idx, tensor in captures.items()
    }

    # Cleanup
    state.active_capture_requests.pop(request_id, None)
    state.request_prefill_buffers.pop(request_id, None)
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

        # Hoist transfer stream check outside loop to avoid repeated branching
        if state.transfer_stream is not None:
            # CUDA path: use async transfers
            for req_id in request_ids:
                # First check for pre-transferred data from Phase 2 streaming
                pending = state.request_pending_transfers.get(req_id, {})

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
                        with torch.cuda.stream(state.transfer_stream):
                            gpu_tensor = tensor.detach().contiguous()
                            cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(state.transfer_stream)
                            transfer_data[(req_id, layer_idx)] = (cpu_tensor, event, tensor_bytes)
        else:
            # Non-CUDA path: synchronous transfers
            for req_id in request_ids:
                pending = state.request_pending_transfers.get(req_id, {})
                captures = state.request_captures.get(req_id, {})

                for layer_idx, tensor in captures.items():
                    layer_count += 1
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    total_bytes += tensor_bytes

                    if layer_idx in pending:
                        cpu_tensor, event = pending[layer_idx]
                        transfer_data[(req_id, layer_idx)] = (cpu_tensor, event, tensor_bytes)
                    else:
                        cpu_tensor = tensor.detach().cpu().contiguous()
                        transfer_data[(req_id, layer_idx)] = (cpu_tensor, None, tensor_bytes)

        # Phase 3: Synchronize all events before serialization
        if state.transfer_stream is not None:
            state.transfer_stream.synchronize()

        # Phase 4: Serialize all transferred tensors
        for req_id in request_ids:
            captures = state.request_captures.pop(req_id, {})
            serialized = {}

            for layer_idx in captures.keys():
                cpu_tensor, _, tensor_bytes = transfer_data[(req_id, layer_idx)]

                # Time serialization to shared memory
                t_numpy_start = time.perf_counter()
                payload = _create_shared_tensor(cpu_tensor, state)

                total_numpy_time += time.perf_counter() - t_numpy_start
                serialized[layer_idx] = payload

            # Cleanup
            state.active_capture_requests.pop(req_id, None)
            state.request_prefill_buffers.pop(req_id, None)
            state.request_last_phase.pop(req_id, None)
            state.request_token_counts.pop(req_id, None)
            state.request_decode_buffers.pop(req_id, None)
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
    state.request_decode_buffers.pop(request_id, None)
    state.request_last_phase.pop(request_id, None)
    state.request_token_counts.pop(request_id, None)
    state.request_pending_transfers.pop(request_id, None)


def fetch_last_profile(worker: Any) -> dict[str, Any]:
    """Return the most recent torch profiler summary, if available."""
    state = _ensure_state(worker)
    summary = state.last_fetch_profile
    if summary is None:
        return {}
    return summary


def release_shared_memory(worker: Any, shm_names: list[str]) -> dict[str, Any]:
    """Release shared memory segments created for activation captures.

    Parameters
    ----------
    shm_names : list[str]
        List of shared memory segment names to release.

    Returns
    -------
    dict[str, Any]
        Status information about the release operation.
    """
    state = _ensure_state(worker)

    released = []
    not_found = []
    errors = []

    for shm_name in shm_names:
        # Pop from dict with lock held
        with state.shm_lock:
            if shm_name not in state.active_shared_memory:
                not_found.append(shm_name)
                continue
            shm, _ = state.active_shared_memory.pop(shm_name)

        # Close and unlink outside lock
        try:
            shm.close()
            shm.unlink()
            released.append(shm_name)
            logger.debug(f"Released shared memory: {shm_name}")
        except Exception as e:
            errors.append({"name": shm_name, "error": str(e)})
            logger.warning(f"Failed to release shared memory {shm_name}: {e}")

    return {
        "released": len(released),
        "not_found": len(not_found),
        "errors": len(errors),
        "error_details": errors if errors else None,
    }


def _cleanup_stale_shared_memory(state: _SteeringState, stop_event: threading.Event) -> None:
    """Background thread that periodically cleans up stale shared memory segments.

    Parameters
    ----------
    state : _SteeringState
        Worker steering state containing active_shared_memory dict.
    stop_event : threading.Event
        Event to signal thread shutdown.
    """
    ttl_seconds = state.shm_ttl_seconds
    scan_interval = 60  # Scan every 60 seconds

    logger.info(f"Starting shared memory cleanup thread (TTL: {ttl_seconds}s, scan: {scan_interval}s)")

    while not stop_event.wait(scan_interval):
        # Snapshot current shared memory state with lock
        with state.shm_lock:
            if not state.active_shared_memory:
                continue
            active_items = list(state.active_shared_memory.items())

        now = time.time()
        stale_names = []

        # Find stale segments (outside lock)
        for shm_name, (shm, timestamp) in active_items:
            age = now - timestamp
            if age > ttl_seconds:
                stale_names.append(shm_name)

        # Clean up stale segments
        for shm_name in stale_names:
            # Pop from dict with lock held
            with state.shm_lock:
                if shm_name not in state.active_shared_memory:
                    continue  # Already cleaned up
                shm, timestamp = state.active_shared_memory.pop(shm_name)

            # Close and unlink outside lock
            try:
                age = now - timestamp
                shm.close()
                shm.unlink()
                logger.warning(
                    f"TTL expired: {shm_name} (age: {age:.1f}s, "
                    f"size: {shm.size / (1024**2):.2f}MB)"
                )
            except Exception as e:
                logger.error(f"Failed to cleanup stale shared memory {shm_name}: {e}")

    logger.info("Shared memory cleanup thread stopped")


def fetch_perf_counters(worker: Any) -> dict[str, Any]:
    """Return current performance counters and timings.

    Only populated when CHATSPACE_PERF_COUNTERS=1 is set.
    """
    return {
        "counters": dict(_PERF_COUNTERS),
        "timings": dict(_PERF_TIMINGS),
        "enabled": _PERF_COUNTERS_ENABLED,
    }


for _rpc_fn in (
    initialize_worker_state,
    set_projection_cap_precision,
    register_capture_request,
    register_steering_spec,
    unregister_steering_spec,
    fetch_request_activations,
    fetch_batch_captures,
    fetch_last_profile,
    fetch_perf_counters,
    unregister_capture_request,
    release_shared_memory,
):
    _register_rpc(_rpc_fn.__name__, _rpc_fn)

ensure_collective_rpc_gateway_installed()
