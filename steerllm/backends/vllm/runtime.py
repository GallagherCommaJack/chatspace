"""Worker-side utilities for steering vector control inside vLLM workers.

These helpers are executed inside vLLM worker processes via collective RPCs.
They patch decoder layers in Qwen, Llama, and Gemma models so steering vectors
can be injected and activations captured at runtime.

Note: This is a standalone reimplementation for steerllm, not dependent on chatspace.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


# =============================================================================
# Tensor Serialization
# =============================================================================

def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Serialize tensor for RPC transport."""
    arr = tensor.detach().cpu().contiguous()
    dtype_name = str(arr.dtype).removeprefix("torch.")

    if arr.numel() == 0:
        buffer = b""
    else:
        try:
            buffer = arr.numpy().tobytes()
        except TypeError:
            # Some dtypes (like bfloat16) need conversion
            arr = arr.to(dtype=torch.float32)
            dtype_name = str(arr.dtype).removeprefix("torch.")
            buffer = arr.numpy().tobytes()

    return {
        "dtype": dtype_name,
        "shape": list(arr.shape),
        "data": buffer,
    }


def deserialize_tensor(data: dict[str, Any]) -> torch.Tensor:
    """Deserialize tensor from RPC transport."""
    dtype_name = data["dtype"]
    shape = data["shape"]
    buffer = data["data"]

    dtype = getattr(torch, dtype_name)
    arr = np.frombuffer(buffer, dtype=np.dtype(str(dtype).replace("torch.", "")))
    return torch.from_numpy(arr.copy()).reshape(shape).to(dtype=dtype)


def _get_env_int(name: str, default: int) -> int:
    """Parse integer environment variable safely."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


# Configuration from environment
_CAPTURE_METADATA_ENABLED = _env_flag("STEERLLM_CAPTURE_METADATA", True)

# Optional precision override for projection caps
_PROJECTION_CAP_PRECISION: torch.dtype | None = None


@dataclass
class _ProjectionCapConfig:
    """Projection capping parameters for a layer."""
    unit_vector: torch.Tensor
    min: float | None
    max: float | None


@dataclass
class _AblationConfig:
    """Ablation parameters for a layer."""
    unit_vector: torch.Tensor
    scale: float


@dataclass
class _SteeringState:
    """Track steering metadata for a worker.

    This is the per-worker state that maintains:
    - Active capture requests and their buffers
    - Per-request steering specifications
    - Shared memory tracking for zero-copy IPC
    """

    hidden_size: int
    dtype: torch.dtype
    device: torch.device

    # Per-request activation capture
    active_capture_requests: dict[str, set[int]] = field(default_factory=dict)
    request_captures: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    request_prefill_buffers: dict[str, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    request_decode_buffers: dict[str, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    request_last_phase: dict[str, str] = field(default_factory=dict)
    request_token_counts: dict[str, int] = field(default_factory=dict)

    # Per-request steering (request_id -> deserialized spec)
    request_steering_specs: dict[str, Any] = field(default_factory=dict)

    # Per-step batch metadata from model runner
    step_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)
    global_step: int = 0

    # Async transfer infrastructure
    transfer_stream: torch.cuda.Stream | None = None
    request_pending_transfers: dict[str, dict[int, tuple[torch.Tensor, torch.cuda.Event]]] = field(default_factory=dict)

    # Shared memory IPC
    active_shared_memory: dict[str, tuple[Any, float]] = field(default_factory=dict)
    shm_lock: threading.Lock = field(default_factory=threading.Lock)
    shm_cleanup_thread: Any = None

    # Shared memory configuration
    shm_ttl_seconds: int = 600
    shm_max_gb: float = 128.0

    # Capture configuration
    decode_buffer_size: int = 128


# Module-level state
_WORKER_STATE: _SteeringState | None = None


def get_worker_state() -> _SteeringState | None:
    """Get the current worker's steering state."""
    return _WORKER_STATE


def set_worker_state(state: _SteeringState) -> None:
    """Set the current worker's steering state."""
    global _WORKER_STATE
    _WORKER_STATE = state


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


# =============================================================================
# Decoder Layer Patching
# =============================================================================

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


def _extract_hidden_from_output(output: Any) -> torch.Tensor | None:
    """Extract hidden state tensor from layer output.

    vLLM Qwen/Llama layers return (delta, residual) where the full hidden
    state is residual + delta. We combine them to match HuggingFace outputs.
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            first, second = output[0], output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                # vLLM returns (delta, residual), we want residual + delta
                return second + first
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]

    if isinstance(output, dict) and "last_hidden_state" in output:
        return output["last_hidden_state"]

    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
        if isinstance(hidden, torch.Tensor):
            return hidden

    raise TypeError(f"Cannot extract hidden state from {type(output).__name__}")


def _reconstruct_output_with_hidden(
    output: Any,
    original_hidden: torch.Tensor,
    new_hidden: torch.Tensor,
) -> Any:
    """Reconstruct layer output with modified hidden states.

    For vLLM (delta, residual) format, we compute new_delta = new_hidden - residual.
    """
    if isinstance(output, torch.Tensor):
        return new_hidden

    if isinstance(output, tuple):
        if len(output) >= 2:
            first, second = output[0], output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                # output is (delta, residual)
                # original: delta + residual = original_hidden
                # new: new_delta + residual = new_hidden
                # new_delta = new_hidden - residual
                new_delta = new_hidden - second
                return (new_delta,) + output[1:]
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return (new_hidden,) + output[1:]

    if isinstance(output, list):
        if len(output) >= 2:
            first, second = output[0], output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                new_delta = new_hidden - second
                return [new_delta] + output[1:]
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return [new_hidden] + output[1:]

    return output


# =============================================================================
# Steering Operations
# =============================================================================

def _apply_projection_cap(
    hidden: torch.Tensor,
    config: _ProjectionCapConfig,
) -> torch.Tensor:
    """Apply projection capping to hidden states.

    Clamps the component of hidden states along the direction.
    """
    vec = config.unit_vector
    if vec.device != hidden.device or vec.dtype != hidden.dtype:
        vec = vec.to(device=hidden.device, dtype=hidden.dtype)

    # Compute projection: (hidden @ vec)
    proj = hidden @ vec  # [seq_len]

    # Clamp projection
    clamped = proj.clone()
    if config.min is not None:
        clamped = torch.clamp(clamped, min=config.min)
    if config.max is not None:
        clamped = torch.clamp(clamped, max=config.max)

    # Apply correction: hidden += (clamped - proj) * vec
    diff = (clamped - proj).unsqueeze(-1)  # [seq_len, 1]
    return hidden + diff * vec


def _apply_ablation(
    hidden: torch.Tensor,
    config: _AblationConfig,
) -> torch.Tensor:
    """Apply ablation (component scaling) to hidden states.

    Scales the component along the direction by the given factor.
    """
    vec = config.unit_vector
    if vec.device != hidden.device or vec.dtype != hidden.dtype:
        vec = vec.to(device=hidden.device, dtype=hidden.dtype)

    # Compute projection
    proj = hidden @ vec  # [seq_len]

    # Apply scaling: hidden += (scale - 1) * proj * vec
    adjustment = ((config.scale - 1) * proj).unsqueeze(-1)  # [seq_len, 1]
    return hidden + adjustment * vec


def _apply_layer_steering_to_hidden(
    hidden: torch.Tensor,
    layer_spec: Any,
    state: _SteeringState,
) -> torch.Tensor:
    """Apply steering operations to hidden states.

    Operations are applied in sequence order.
    Each operation is a tuple: (op_type, vector, params)
    """
    ops = layer_spec.operations
    if not ops:
        return hidden

    for op_type, vec, params in ops:
        if op_type == "add":
            if vec.device != hidden.device or vec.dtype != hidden.dtype:
                vec = vec.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + vec
        elif op_type == "cap":
            cap_min, cap_max = params
            hidden = _apply_projection_cap(
                hidden, _ProjectionCapConfig(unit_vector=vec, min=cap_min, max=cap_max)
            )
        elif op_type == "ablation":
            hidden = _apply_ablation(
                hidden, _AblationConfig(unit_vector=vec, scale=params)
            )

    return hidden


def _apply_per_request_steering(
    output: Any,
    state: _SteeringState,
    layer_idx: int,
    request_ids: list[str],
    seq_lens: list[int] | None,
    cached_hidden: torch.Tensor | None = None,
) -> Any:
    """Apply per-request steering by slicing and transforming hidden states."""
    hidden = cached_hidden if cached_hidden is not None else _extract_hidden_from_output(output)
    if hidden is None or hidden.dim() != 2:
        return output

    # Single request without seq_lens
    if seq_lens is None or len(seq_lens) != len(request_ids):
        if len(request_ids) == 1:
            req_id = request_ids[0]
            spec = state.request_steering_specs.get(req_id)
            if spec is not None and layer_idx in spec.layers:
                layer_spec = spec.layers[layer_idx]
                if layer_spec.operations:
                    new_hidden = _apply_layer_steering_to_hidden(hidden, layer_spec, state)
                    return _reconstruct_output_with_hidden(output, hidden, new_hidden)
        return output

    # Multiple requests: slice, transform, concatenate
    request_slices = []
    start_idx = 0

    for i, req_id in enumerate(request_ids):
        seq_len = seq_lens[i]
        end_idx = start_idx + seq_len
        req_hidden = hidden[start_idx:end_idx]

        spec = state.request_steering_specs.get(req_id)
        if spec is not None and layer_idx in spec.layers:
            layer_spec = spec.layers[layer_idx]
            if layer_spec.operations:
                req_hidden = _apply_layer_steering_to_hidden(req_hidden, layer_spec, state)

        request_slices.append(req_hidden)
        start_idx = end_idx

    transformed_hidden = torch.cat(request_slices, dim=0)
    return _reconstruct_output_with_hidden(output, hidden, transformed_hidden)


# =============================================================================
# Capture Logic
# =============================================================================

def _capture_hook_full(
    state: _SteeringState,
    layer_idx: int,
    hidden: torch.Tensor,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """Capture activations for requests that have capture enabled."""
    if not state.active_capture_requests:
        return

    start_idx = 0
    for i, req_id in enumerate(request_ids):
        if req_id not in state.active_capture_requests:
            if seq_lens is not None and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        capture_layers = state.active_capture_requests[req_id]
        if layer_idx not in capture_layers:
            if seq_lens is not None and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        # Extract this request's hidden states
        if seq_lens is not None and i < len(seq_lens):
            seq_len = seq_lens[i]
            end_idx = start_idx + seq_len
            req_hidden = hidden[start_idx:end_idx].detach().clone()
            start_idx = end_idx
        else:
            req_hidden = hidden.detach().clone()

        # Store in request captures
        if req_id not in state.request_captures:
            state.request_captures[req_id] = {}

        if layer_idx not in state.request_captures[req_id]:
            state.request_captures[req_id][layer_idx] = req_hidden
        else:
            # Concatenate with existing captures (for decode tokens)
            existing = state.request_captures[req_id][layer_idx]
            state.request_captures[req_id][layer_idx] = torch.cat([existing, req_hidden], dim=0)


# =============================================================================
# Layer Patching
# =============================================================================

def _patch_decoder_layer_class(layer_cls: type) -> None:
    """Patch a decoder layer class to support steering and capture."""
    if layer_cls in _PATCHED_CLASSES:
        return

    original_forward = layer_cls.forward

    @wraps(original_forward)
    def _patched_forward(self, *args: Any, **kwargs: Any) -> Any:
        output = original_forward(self, *args, **kwargs)

        state = getattr(self, "_steerllm_state", None)
        layer_idx = getattr(self, "_steerllm_layer_index", None)

        if state is None or layer_idx is None:
            return output

        # Get batch metadata
        request_ids = None
        seq_lens = None
        current_step = state.global_step - 1
        if current_step >= 0:
            metadata = state.step_metadata.get(current_step)
            if metadata is not None:
                request_ids = metadata.get("request_ids")
                seq_lens = metadata.get("seq_lens")

        if not request_ids:
            return output

        # Check for per-request steering
        has_steering = False
        for req_id in request_ids:
            spec = state.request_steering_specs.get(req_id)
            if spec is not None and layer_idx in spec.layers:
                layer_spec = spec.layers[layer_idx]
                if layer_spec.operations:
                    has_steering = True
                    break

        # Extract hidden state once for both steering and capture
        cached_hidden = None
        if has_steering or state.active_capture_requests:
            cached_hidden = _extract_hidden_from_output(output)

        # Apply steering
        if has_steering:
            output = _apply_per_request_steering(
                output, state, layer_idx, request_ids, seq_lens, cached_hidden
            )
            if state.active_capture_requests:
                cached_hidden = _extract_hidden_from_output(output)

        # Capture activations
        if state.active_capture_requests:
            hidden = cached_hidden if cached_hidden is not None else _extract_hidden_from_output(output)
            if hidden is not None and hidden.dim() == 2:
                _capture_hook_full(state, layer_idx, hidden, request_ids, seq_lens)

        return output

    layer_cls.forward = _patched_forward
    _PATCHED_CLASSES.add(layer_cls)


def ensure_layer_patch_installed() -> None:
    """Patch known decoder layers to support steering."""
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


def _resolve_layers(model: Any) -> list[Any]:
    """Return transformer layers for Qwen, Llama, and Gemma architectures."""
    # Multimodal models
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)

    # Standard architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "layers"):
        return list(model.layers)

    raise ValueError(f"Cannot find layers in model {type(model).__name__}")


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


def _patch_model_runner(worker: Any, state: _SteeringState) -> None:
    """Patch GPUModelRunner.execute_model to capture per-step batch metadata."""
    if not _CAPTURE_METADATA_ENABLED:
        logger.info("STEERLLM_CAPTURE_METADATA=0; skipping model runner patch.")
        return

    model_runner = worker.model_runner
    logger.info(
        f"_patch_model_runner: model_runner type={type(model_runner).__name__}, "
        f"has execute_model={hasattr(model_runner, 'execute_model')}"
    )

    if not hasattr(model_runner, "execute_model"):
        logger.error(
            f"model_runner does not have execute_model method! "
            f"Available methods: {[m for m in dir(model_runner) if not m.startswith('_')][:20]}"
        )
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
        if not state.active_capture_requests and not state.request_steering_specs:
            return original_execute(model_input, *args, **kwargs)

        # Extract request IDs and sequence lengths from model_input
        try:
            request_ids = None
            seq_lens = None

            # V1 engine: model_input is SchedulerOutput
            # IMPORTANT: vLLM V1 orders the hidden state tensor as [CACHED, NEW]
            if hasattr(model_input, "scheduled_new_reqs") and hasattr(model_input, "scheduled_cached_reqs"):
                request_ids = []
                seq_lens = []

                # Process CACHED requests FIRST (they appear first in the tensor)
                cached_reqs_val = model_input.scheduled_cached_reqs
                if cached_reqs_val and hasattr(cached_reqs_val, "req_ids"):
                    cached = cached_reqs_val
                    if cached.num_reqs > 0 and cached.req_ids:
                        request_ids.extend(cached.req_ids)
                        seq_lens.extend([1] * len(cached.req_ids))

                # Process NEW requests SECOND (they appear after cached in the tensor)
                new_reqs_val = model_input.scheduled_new_reqs
                if new_reqs_val:
                    new_reqs = new_reqs_val
                    if not isinstance(new_reqs, list):
                        new_reqs = [new_reqs]

                    for req in new_reqs:
                        if hasattr(req, "req_id"):
                            request_ids.append(req.req_id)
                        if hasattr(req, "prompt_token_ids"):
                            seq_lens.append(len(req.prompt_token_ids))

            if request_ids:
                # Store metadata for this step
                current_step = state.global_step
                state.step_metadata[current_step] = {
                    "request_ids": request_ids,
                    "seq_lens": seq_lens,
                    "step": current_step,
                }
                state.global_step += 1

                # Clean up old metadata (keep last 1000 steps)
                if len(state.step_metadata) > 1000:
                    old_steps = sorted(state.step_metadata.keys())[:-1000]
                    for step in old_steps:
                        state.step_metadata.pop(step, None)

        except (AttributeError, TypeError, KeyError, IndexError) as e:
            logger.warning(
                f"Failed to extract metadata from model_input: {type(e).__name__}: {e}",
                exc_info=True
            )
        except Exception as e:
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

    This is called via RPC on each worker to initialize steering infrastructure.

    Parameters
    ----------
    worker :
        vLLM worker instance with model_runner attribute.
    layer_indices :
        Optional list of layer indices to initialize (unused, for compatibility).
    shm_ttl_seconds :
        TTL for shared memory segments in seconds.
    shm_max_gb :
        Maximum shared memory usage in GB.
    decode_buffer_size :
        Buffer size for decode token batching.

    Returns
    -------
    dict[str, Any]
        Metadata about the initialized worker state.
    """
    ensure_layer_patch_installed()
    model = worker.model_runner.model
    layers = _resolve_layers(model)

    # Handle multimodal models where config is nested
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
        shm_ttl_seconds=shm_ttl_seconds,
        shm_max_gb=shm_max_gb,
        decode_buffer_size=decode_buffer_size,
    )

    # Create CUDA stream for async transfers if available
    if device.type == "cuda":
        state.transfer_stream = torch.cuda.Stream(device=device)

    worker._steerllm_steering = state
    set_worker_state(state)

    # Start shared memory cleanup thread
    stop_event = threading.Event()
    cleanup_thread = threading.Thread(
        target=_cleanup_stale_shared_memory,
        args=(state, stop_event),
        daemon=True,
        name="steerllm-shm-cleanup",
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

    # Patch model runner to capture batch metadata
    _patch_model_runner(worker, state)

    # Wrap model if not already wrapped
    if not isinstance(worker.model_runner.model, _SteeredModelWrapper):
        worker.model_runner.model = _SteeredModelWrapper(model, state)

    # Attach state and layer indices to all layers
    layers = _resolve_layers(worker.model_runner.model)
    for layer_idx, layer in enumerate(layers):
        setattr(layer, "_steerllm_state", state)
        setattr(layer, "_steerllm_layer_index", layer_idx)

    return {
        "hidden_size": hidden_size,
        "layer_count": len(layers),
        "dtype": str(dtype),
        "device": str(device),
    }


# =============================================================================
# RPC Handlers
# =============================================================================

_RPC_HANDLERS: dict[str, Callable[..., Any]] = {}
_RPC_GATEWAY_INSTALLED = False

STEERING_RPC_METHOD = "_steerllm_steering_rpc"
STEERING_WORKER_EXTENSION = "steerllm.backends.vllm.runtime.SteeringWorkerExtension"


def _register_rpc(name: str, func: Callable[..., Any]) -> None:
    """Register an RPC handler."""
    _RPC_HANDLERS[name] = func


def rpc_args(op: str, *args: Any) -> tuple[Any, ...]:
    """Create RPC arguments tuple."""
    return (op, *args)


class SteeringWorkerExtension:
    """Worker mixin providing steering RPC handling."""

    def _steerllm_steering_rpc(self, op: str, *args: Any, **kwargs: Any) -> Any:
        handler = _RPC_HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"Unknown steerllm steering RPC: {op}")
        return handler(self, *args, **kwargs)


def ensure_collective_rpc_gateway_installed() -> None:
    """Install RPC gateway on vLLM WorkerWrapperBase."""
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
            raise ValueError(f"Unknown steerllm steering RPC: {op}")
        target = getattr(self, "worker", self)
        return handler(target, *args, **kwargs)

    setattr(WorkerWrapperBase, STEERING_RPC_METHOD, _dispatch)
    _RPC_GATEWAY_INSTALLED = True


# =============================================================================
# RPC Handler Implementations
# =============================================================================

def _rpc_register_capture(
    worker: Any,
    request_id: str,
    layer_indices: list[int],
) -> bool:
    """Register capture for a request."""
    state = get_worker_state()
    if state is None:
        return False

    state.active_capture_requests[request_id] = set(layer_indices)
    state.request_captures[request_id] = {}
    return True


_register_rpc("register_capture", _rpc_register_capture)


def _rpc_unregister_capture(
    worker: Any,
    request_id: str,
) -> bool:
    """Unregister capture for a request."""
    state = get_worker_state()
    if state is None:
        return False

    state.active_capture_requests.pop(request_id, None)
    state.request_captures.pop(request_id, None)
    return True


_register_rpc("unregister_capture", _rpc_unregister_capture)


def _rpc_register_steering_spec(
    worker: Any,
    request_id: str,
    spec_data: dict[str, Any],
) -> bool:
    """Register a steering spec for a request.

    spec_data is a dict with:
    - layers: dict[int, layer_spec]
    - Each layer_spec has operations: list of (op_type, vector_bytes, dtype, params)
    """
    state = get_worker_state()
    if state is None:
        return False

    # Deserialize spec
    layers = {}
    for layer_idx_str, layer_data in spec_data.get("layers", {}).items():
        layer_idx = int(layer_idx_str)
        ops = []
        for op_data in layer_data.get("operations", []):
            op_type = op_data["type"]
            vec_bytes = op_data["vector"]
            dtype_str = op_data["dtype"]
            params = op_data.get("params")

            # Reconstruct tensor (uint8 view encoding for bfloat16 support)
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            vec = torch.frombuffer(bytearray(vec_bytes), dtype=torch.uint8).view(dtype).clone()
            vec = vec.to(device=state.device, dtype=state.dtype)

            # Scale vector for additive steering
            if op_type == "add" and params is not None:
                vec = vec * params  # params is scale for add
                params = None  # Clear params after applying

            ops.append((op_type, vec, params))

        # Create a simple namespace for the layer spec
        class LayerSpec:
            pass
        layer_spec = LayerSpec()
        layer_spec.operations = ops
        layers[layer_idx] = layer_spec

    # Create full spec
    class Spec:
        pass
    spec = Spec()
    spec.layers = layers

    state.request_steering_specs[request_id] = spec
    return True


_register_rpc("register_steering_spec", _rpc_register_steering_spec)


def _rpc_unregister_steering_spec(
    worker: Any,
    request_id: str,
) -> bool:
    """Unregister steering spec for a request."""
    state = get_worker_state()
    if state is None:
        return False

    state.request_steering_specs.pop(request_id, None)
    return True


_register_rpc("unregister_steering_spec", _rpc_unregister_steering_spec)


def _rpc_fetch_captures(
    worker: Any,
    request_id: str,
) -> dict[str, Any] | None:
    """Fetch captures for a request, using shared memory for zero-copy."""
    state = get_worker_state()
    if state is None:
        return None

    captures = state.request_captures.get(request_id)
    if captures is None:
        return None

    result = {}
    for layer_idx, tensor in captures.items():
        # Create shared memory segment
        shm_name = f"steerllm_{request_id}_{layer_idx}_{uuid.uuid4().hex[:8]}"
        # Use uint8 view for bfloat16 support (numpy doesn't support bfloat16)
        tensor_cpu = tensor.cpu().contiguous()
        tensor_bytes = tensor_cpu.view(torch.uint8).numpy().tobytes()

        try:
            shm = SharedMemory(name=shm_name, create=True, size=len(tensor_bytes))
            shm.buf[:len(tensor_bytes)] = tensor_bytes

            # Track for cleanup
            with state.shm_lock:
                state.active_shared_memory[shm_name] = (shm, time.time())

            result[str(layer_idx)] = {
                "shm_name": shm_name,
                "shape": list(tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype),
                "nbytes": len(tensor_bytes),
            }
        except Exception as e:
            logger.warning(f"Failed to create shared memory: {e}")
            # Fall back to bytes
            result[str(layer_idx)] = {
                "data": tensor_bytes,
                "shape": list(tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype),
            }

    return result


_register_rpc("fetch_captures", _rpc_fetch_captures)


def _rpc_release_shared_memory(
    worker: Any,
    shm_names: list[str],
) -> bool:
    """Release shared memory segments."""
    state = get_worker_state()
    if state is None:
        return False

    with state.shm_lock:
        for name in shm_names:
            entry = state.active_shared_memory.pop(name, None)
            if entry is not None:
                shm, _ = entry
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.debug(f"Error releasing shared memory {name}: {e}")

    return True


_register_rpc("release_shared_memory", _rpc_release_shared_memory)


def _rpc_set_step_metadata(
    worker: Any,
    step: int,
    metadata: dict[str, Any],
) -> bool:
    """Set batch metadata for a step."""
    state = get_worker_state()
    if state is None:
        return False

    state.step_metadata[step] = metadata
    state.global_step = step + 1

    # Cleanup old metadata (keep last 10 steps)
    if len(state.step_metadata) > 10:
        old_steps = sorted(state.step_metadata.keys())[:-10]
        for s in old_steps:
            state.step_metadata.pop(s, None)

    return True


_register_rpc("set_step_metadata", _rpc_set_step_metadata)


# Register initialize_worker_state as an RPC handler
_register_rpc("initialize_worker_state", initialize_worker_state)


# Install RPC gateway at module load time
ensure_collective_rpc_gateway_installed()
