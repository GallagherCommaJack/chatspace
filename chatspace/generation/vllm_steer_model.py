"""Steering helpers for coordinating vLLM worker state with chatspace.

The helpers in this module wrap ``vllm.LLM`` so steering vectors can be injected
into Qwen and Llama decoder layers without breaking CUDA graph capture.  The primary
entry point is :class:`VLLMSteerModel`, which supports per-request steering:

Typical usage::

    cfg = VLLMSteeringConfig(model_name="Qwen/Qwen3-0.6B")
    model = VLLMSteerModel(cfg, bootstrap_layers=(target_layer,))

    # Create steering spec
    from chatspace.generation.vllm_steer_model import SteeringSpec, LayerSteeringSpec, AddSpec
    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(
            add=AddSpec(vector=torch.randn(model.hidden_size), scale=1.0)
        )
    })

    # Generate with per-request steering
    outputs, handles = model.generate(
        ["...prompt..."],
        sampling_params,
        steering_spec=steering_spec,
        capture_layers=target_layer
    )

``VLLMSteerModel`` internally broadcasts steering specs to every worker via
collective RPCs.  ``enforce_eager=True`` should remain enabled unless you have
verified the steering patch still executes inside compiled graphs.
"""

from __future__ import annotations

import math
import asyncio
import os
import weakref
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Literal, Sequence, cast, overload
import logging

import numpy as np
import torch
from vllm import SamplingParams

from chatspace.vllm_steering import runtime as steering_runtime
# SteerableModel ABC removed - only vLLM implementation exists


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
class MessageBoundary:
    """Token boundary information for a single message in a chat conversation.

    Attributes
    ----------
    role : str
        Message role (e.g., "system", "user", "assistant").
    content : str
        Original message content.
    start_token : int
        Starting token index in the formatted prompt (inclusive).
    end_token : int
        Ending token index in the formatted prompt (exclusive).
    """

    role: str
    content: str
    start_token: int
    end_token: int

    @property
    def num_tokens(self) -> int:
        """Number of tokens in this message."""
        return self.end_token - self.start_token


@dataclass
class ChatResponse:
    """Response from chat() API with explicit prefill/generated separation.

    When using assistant response prefilling with continue_final_message=True,
    this dataclass makes it clear which text was prefilled vs. generated by
    the model. This is especially useful when combined with activation capture,
    as the message boundaries will include prefill tokens but the vLLM response
    text contains only generated tokens.

    Attributes
    ----------
    prefill : str
        Text that was prefilled (from partial assistant message). Empty string
        if no prefill was used.
    generated : str
        Text generated by the model (always present, may be empty).
    """

    prefill: str
    generated: str

    def full_text(self) -> str:
        """Return complete response text (prefill + generated)."""
        return self.prefill + self.generated

    def __str__(self) -> str:
        """String representation is the full text."""
        return self.full_text()

    @property
    def has_prefill(self) -> bool:
        """Whether this response includes a prefill."""
        return len(self.prefill) > 0

    def to_message(self, role: str = "assistant") -> dict[str, str]:
        """Convert to OpenAI-style message dict for building conversation history."""
        return {"role": role, "content": self.full_text()}


def _cleanup_capture_handle_and_warn(
    model_ref: weakref.ref,
    shm_names: list[str],
    accessed_container: list[bool],
) -> None:
    """Cleanup callback for CaptureHandle finalization.

    Warns if handle held shared memory but was never accessed.
    Attempts to release shared memory if not already released.

    Parameters
    ----------
    model_ref : weakref.ref
        Weak reference to the VLLMSteerModel.
    shm_names : list[str]
        List of shared memory segment names (mutable, updated by handle).
    accessed_container : list[bool]
        Mutable container with [accessed] flag.
    """
    accessed = accessed_container[0]

    if shm_names and not accessed:
        warnings.warn(
            f"CaptureHandle held {len(shm_names)} shared memory regions "
            f"but was never accessed! This wastes memory. "
            f"Use 'async with handle:' or call 'await handle.close()' explicitly.",
            ResourceWarning,
            stacklevel=2
        )

    # Attempt cleanup (may fail if model is gone)
    model = model_ref()
    if model is not None and shm_names:
        try:
            # We're in a finalizer, can't use async/await
            # The shared memory will be cleaned up by worker-side TTL
            logger.debug(f"Finalizer: {len(shm_names)} shm segments will be cleaned by TTL")
        except Exception as e:
            logger.debug(f"Finalizer cleanup note: {e}")


class CaptureHandle:
    """Handle for lazily fetching activation captures for a single request.

    Attributes
    ----------
    request_id : str
        Internal request identifier used for fetching captures.
    layer_indices : tuple[int, ...]
        Layer indices that were captured for this request.
    message_boundaries : tuple[MessageBoundary, ...] | None
        Optional message boundary information for chat-style captures.
        When set, allows slicing activations by message via get_message_activations().

    Usage
    -----
    Recommended usage with async context manager for automatic cleanup::

        async with handle:
            captures = handle.captures
            # Process captures...
        # Shared memory automatically released

    Or manual cleanup::

        captures = handle.captures
        await handle.close()  # Explicit cleanup
    """

    def __init__(
        self,
        request_id: str,
        layer_indices: tuple[int, ...],
        model: "VLLMSteerModel",
        message_boundaries: tuple[MessageBoundary, ...] | None = None,
    ):
        self.request_id = request_id
        self.layer_indices = layer_indices
        self._model_ref = weakref.ref(model)
        self._captures: dict[int, list[dict[str, Any]]] | None = None
        self.message_boundaries = message_boundaries

        # Shared memory tracking
        self._shm_names: list[str] = []
        self._shm_objects: list[SharedMemory] = []  # Keep SharedMemory objects alive
        self._accessed = False
        self._closed = False

        # Use mutable container for accessed flag so finalizer can see updates
        self._accessed_container = [False]  # [accessed]

        # Register finalize callback for backup cleanup
        self._finalizer = weakref.finalize(
            self,
            _cleanup_capture_handle_and_warn,
            weakref.ref(model),
            self._shm_names,  # Direct reference to list (will be updated)
            self._accessed_container,  # Reference to mutable container
        )

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args):
        """Exit async context manager and release shared memory."""
        await self.close()

    async def close(self):
        """Explicitly release shared memory resources."""
        if self._closed:
            return

        self._closed = True

        # Client-side cleanup: unmap shared memory
        for shm in self._shm_objects:
            try:
                shm.close()  # Unmap from client process
            except Exception as e:
                logger.warning(f"Failed to close shared memory {shm.name}: {e}")

        # Worker-side cleanup: send RPC to unlink shared memory segments
        model = self._model_ref()
        if model is not None and self._shm_names:
            try:
                await model._collective_rpc("release_shared_memory", self._shm_names)
                logger.debug(f"Released {len(self._shm_names)} shared memory segments")
            except Exception as e:
                logger.warning(f"Failed to release shared memory: {e}")

        # Clear references
        self._shm_objects.clear()

        # Detach finalizer since we cleaned up explicitly
        self._finalizer.detach()

    async def fetch(self) -> dict[int, list[dict[str, Any]]]:
        """Fetch captures from workers (idempotent).

        Returns
        -------
        dict[int, list[dict[str, Any]]]
            Mapping of layer indices to lists of capture entries. Each entry
            contains "before", "after", "meta" keys.
        """
        if self._captures is None:
            model = self._model_ref()
            if model is None:
                raise RuntimeError("Model has been garbage collected")
            self._captures = await model._fetch_request_captures(
                self.request_id,
                shm_objects_list=self._shm_objects
            )
            # Extract names for cleanup RPC
            self._shm_names = [shm.name for shm in self._shm_objects]
        return self._captures

    @property
    def captures(self) -> dict[int, list[dict[str, Any]]]:
        """Get captures (must call fetch() first).

        Raises
        ------
        RuntimeError
            If captures haven't been fetched yet.
        """
        if self._captures is None:
            raise RuntimeError(
                f"Captures not fetched yet for request {self.request_id}. "
                "Call: await handle.fetch()"
            )
        # Mark as accessed for finalizer tracking
        self._accessed = True
        self._accessed_container[0] = True  # Update mutable container
        return self._captures

    def get_message_activations(
        self,
        message_idx: int,
        layer_idx: int,
        *,
        include_generated: bool = False,
    ) -> torch.Tensor:
        """Get activations for a specific message from chat-style captures.

        Parameters
        ----------
        message_idx : int
            Index of the message in the original conversation.
        layer_idx : int
            Layer index to extract activations from.
        include_generated : bool, default False
            If True and message_idx is the last message, include generated tokens.
            Otherwise, only return activations for the message content itself.

        Returns
        -------
        torch.Tensor
            Activations for the specified message, shape [num_tokens, hidden_size].

        Raises
        ------
        RuntimeError
            If captures haven't been fetched yet or message boundaries aren't available.
        ValueError
            If message_idx or layer_idx is out of range.
        """
        if self._captures is None:
            raise RuntimeError(
                f"Captures not fetched yet for request {self.request_id}. "
                "Call: await handle.fetch() or await model.fetch_captures_batch([handle])"
            )

        if self.message_boundaries is None:
            raise RuntimeError(
                "Message boundaries not available for this capture. "
                "This handle was not created from a chat() call with message tracking."
            )

        if message_idx < 0 or message_idx >= len(self.message_boundaries):
            raise ValueError(
                f"message_idx {message_idx} out of range [0, {len(self.message_boundaries)})"
            )

        if layer_idx not in self._captures:
            raise ValueError(
                f"layer_idx {layer_idx} not in captured layers: {list(self._captures.keys())}"
            )

        full_hidden = self._captures[layer_idx][0]["hidden"]

        boundary = self.message_boundaries[message_idx]
        start = boundary.start_token
        end = boundary.end_token

        if include_generated and message_idx == len(self.message_boundaries) - 1:
            end = full_hidden.shape[0]

        return full_hidden[start:end]


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


def compute_message_boundaries(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    chat_kwargs: dict[str, Any],
) -> tuple[MessageBoundary, ...]:
    """Compute token boundaries for each message in a conversation.

    Uses incremental tokenization to determine where each message starts and
    ends in the formatted prompt. This allows slicing activations by message
    for interpretability analysis.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        Conversation messages in OpenAI format (role, content).
    tokenizer : Any
        Tokenizer with apply_chat_template() method.
    chat_kwargs : dict[str, Any]
        Keyword arguments passed to apply_chat_template().

    Returns
    -------
    tuple[MessageBoundary, ...]
        Boundary information for each message with token ranges.
    """
    boundaries: list[MessageBoundary] = []
    current_offset = 0

    for i, msg in enumerate(messages):
        partial_conv = messages[:i + 1]

        partial_text = tokenizer.apply_chat_template(
            partial_conv,
            tokenize=False,
            **chat_kwargs,
        )

        partial_tokens = tokenizer(partial_text, return_tensors="pt")
        total_len = partial_tokens.input_ids.shape[1]

        boundary = MessageBoundary(
            role=msg["role"],
            content=msg["content"],
            start_token=current_offset,
            end_token=total_len,
        )
        boundaries.append(boundary)
        current_offset = total_len

    return tuple(boundaries)


class VLLMSteerModel:
    """Steerable wrapper around ``vllm.LLM`` for Qwen and Llama models.

    Supports per-request steering by passing a ``SteeringSpec`` to the
    ``generate()`` method. The steering spec is serialized and broadcast
    to all workers for the duration of the request.

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
        use_shared_memory: bool | None = None,
        shm_threshold_kb: int | None = None,
        shm_ttl_seconds: int | None = None,
        shm_max_gb: float | None = None,
        decode_buffer_size: int | None = None,
        **vllm_kwargs,
    ) -> None:
        self.cfg = cfg

        # Shared memory configuration (default from env vars if not specified)
        import os
        self._use_shared_memory = (
            use_shared_memory if use_shared_memory is not None
            else bool(int(os.getenv("CHATSPACE_SHARED_MEMORY", "0")))
        )
        self._shm_threshold_kb = (
            shm_threshold_kb if shm_threshold_kb is not None
            else int(os.getenv("CHATSPACE_SHM_THRESHOLD_KB", "1024"))
        )
        self._shm_ttl_seconds = (
            shm_ttl_seconds if shm_ttl_seconds is not None
            else int(os.getenv("CHATSPACE_SHM_TTL", "600"))
        )
        self._shm_max_gb = (
            shm_max_gb if shm_max_gb is not None
            else float(os.getenv("CHATSPACE_MAX_SHM_GB", "128"))
        )
        self._decode_buffer_size = (
            decode_buffer_size if decode_buffer_size is not None
            else int(os.getenv("CHATSPACE_DECODE_BUFFER_SIZE", "128"))
        )

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
        llm_kwargs.setdefault(
            "worker_extension_cls", steering_runtime.STEERING_WORKER_EXTENSION
        )

        steering_runtime.ensure_layer_patch_installed()
        steering_runtime.ensure_collective_rpc_gateway_installed()

        # Use AsyncLLMEngine for async API
        from vllm import AsyncEngineArgs
        import asyncio

        engine_args = AsyncEngineArgs(
            model=cfg.model_name,
            **llm_kwargs
        )

        # Create engine (this needs to run in async context, so we'll defer)
        self._engine_args = engine_args
        self._engine = None  # Will be initialized on first use
        self._engine_client = None
        self._engine_init_lock = asyncio.Lock()

        if not enforce_eager:
            logger.warning(
                "vLLM steering currently requires enforce_eager=True to apply layer hooks."
            )

        self._init_layers: tuple[int, ...]
        if bootstrap_layers is not None:
            self._init_layers = tuple(int(idx) for idx in bootstrap_layers)
        else:
            self._init_layers = tuple(int(idx) for idx in cfg.bootstrap_layers)

        # Load model config to get dimensions before engine init
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.hidden_size: int = model_config.hidden_size
        self.layer_count: int = model_config.num_hidden_layers
        self._vector_dtype: torch.dtype | None = None

        # Track prompt token lengths for capture reconstruction
        self._last_prompt_lengths: list[int] | None = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load tokenizer for prompt length tracking."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        return self._tokenizer

    @property
    def llm(self):
        """Access the underlying AsyncLLMEngine for raw generation (tests only)."""
        return self._engine

    @llm.setter
    def llm(self, value):
        """Set the underlying AsyncLLMEngine (for test mocking only)."""
        self._engine = value

    async def _ensure_engine_initialized(self) -> None:
        """Initialize AsyncLLMEngine and workers on first use."""
        async with self._engine_init_lock:
            if self._engine is not None:
                return

            # Skip initialization if _engine_args doesn't exist (e.g., dummy test models)
            if not hasattr(self, "_engine_args"):
                return

            from vllm import AsyncLLMEngine

            # Create async engine
            self._engine = AsyncLLMEngine.from_engine_args(self._engine_args)
            self._engine_client = self._engine  # AsyncLLMEngine has collective_rpc directly

            # Initialize worker state with shared memory and capture config
            setup_info = await self._collective_rpc(
                "initialize_worker_state",
                self._init_layers,
                self._use_shared_memory,
                self._shm_threshold_kb,
                self._shm_ttl_seconds,
                self._shm_max_gb,
                self._decode_buffer_size,
            )
            if not setup_info:
                raise RuntimeError("Failed to initialize steering state on workers.")

            first = setup_info[0]
            # Verify dimensions match what we loaded from config
            worker_hidden_size = int(first["hidden_size"])
            worker_layer_count = int(first["layer_count"])
            if worker_hidden_size != self.hidden_size:
                raise RuntimeError(
                    f"Worker hidden_size {worker_hidden_size} doesn't match config {self.hidden_size}"
                )
            if worker_layer_count != self.layer_count:
                raise RuntimeError(
                    f"Worker layer_count {worker_layer_count} doesn't match config {self.layer_count}"
                )
            self._vector_dtype = _parse_dtype(first["dtype"])

    async def _collective_rpc(
        self,
        op: str,
        *args: Any,
        timeout: float | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        if self._engine_client is None:
            raise RuntimeError("Engine not initialized. Call an async method first.")
        rpc_kwargs = kwargs if kwargs else None
        return await self._engine_client.collective_rpc(
            steering_runtime.STEERING_RPC_METHOD,
            timeout=timeout,
            args=steering_runtime.rpc_args(op, *args),
            kwargs=rpc_kwargs,
        )

    async def _register_steering_spec(
        self, request_id: str, steering_spec: SteeringSpec
    ) -> None:
        """Register a per-request steering spec on all workers."""
        # Serialize the steering spec
        serialized_spec = self._serialize_steering_spec(steering_spec)
        await self._collective_rpc(
            "register_steering_spec", request_id, serialized_spec
        )

    async def _unregister_steering_spec(self, request_id: str) -> None:
        """Unregister a per-request steering spec from all workers."""
        await self._collective_rpc("unregister_steering_spec", request_id)

    @staticmethod
    def simple_steering(layer: int, vector: torch.Tensor, scale: float = 1.0) -> SteeringSpec:
        """Create a simple additive steering spec for a single layer.

        Parameters
        ----------
        layer : int
            Layer index to apply steering to.
        vector : torch.Tensor
            Steering vector (will be normalized to unit vector).
        scale : float, optional
            Magnitude to scale the unit vector by (default: 1.0).

        Returns
        -------
        SteeringSpec
            A steering spec with a single additive steering layer.

        Examples
        --------
        >>> model = VLLMSteerModel(cfg)
        >>> my_vector = torch.randn(model.hidden_size)
        >>> spec = VLLMSteerModel.simple_steering(5, my_vector, scale=2.0)
        >>> outputs = await model.generate(prompts, sampling_params, steering_spec=spec)
        """
        norm = float(vector.norm().item())
        unit = vector / norm if norm > 0 else vector
        return SteeringSpec(
            layers={layer: LayerSteeringSpec(add=AddSpec(vector=unit, scale=norm * scale))}
        )

    @staticmethod
    def simple_projection_cap(
        layer: int, vector: torch.Tensor, min: float | None = None, max: float | None = None
    ) -> SteeringSpec:
        """Create a projection cap steering spec for a single layer.

        Limits the projection of hidden states onto a direction vector.

        Parameters
        ----------
        layer : int
            Layer index to apply projection cap to.
        vector : torch.Tensor
            Direction vector to project onto (will be normalized).
        min : float | None
            Minimum allowed projection value.
        max : float | None
            Maximum allowed projection value.

        Returns
        -------
        SteeringSpec
            A steering spec with a single projection cap layer.

        Examples
        --------
        >>> spec = VLLMSteerModel.simple_projection_cap(5, direction_vec, min=-0.5, max=0.5)
        >>> outputs = await model.generate(prompts, sampling_params, steering_spec=spec)
        """
        if min is None and max is None:
            raise ValueError("Must specify at least one of min or max")
        norm = float(vector.norm().item())
        if norm == 0:
            raise ValueError("Projection cap vector must have nonzero norm")
        unit = vector / norm
        return SteeringSpec(
            layers={
                layer: LayerSteeringSpec(projection_cap=ProjectionCapSpec(vector=unit, min=min, max=max))
            }
        )

    @staticmethod
    def simple_ablation(layer: int, vector: torch.Tensor, scale: float = 1.0) -> SteeringSpec:
        """Create an ablation steering spec for a single layer.

        Removes the component of hidden states along a direction vector.

        Parameters
        ----------
        layer : int
            Layer index to apply ablation to.
        vector : torch.Tensor
            Direction vector to ablate (will be normalized).
        scale : float, optional
            Ablation strength, where 1.0 = full ablation (default: 1.0).

        Returns
        -------
        SteeringSpec
            A steering spec with a single ablation layer.

        Examples
        --------
        >>> spec = VLLMSteerModel.simple_ablation(5, direction_vec, scale=0.7)
        >>> outputs = await model.generate(prompts, sampling_params, steering_spec=spec)
        """
        norm = float(vector.norm().item())
        if norm == 0:
            raise ValueError("Ablation vector must have nonzero norm")
        unit = vector / norm
        return SteeringSpec(
            layers={layer: LayerSteeringSpec(ablation=AblationSpec(vector=unit, scale=scale))}
        )

    def _serialize_steering_spec(self, spec: SteeringSpec) -> dict[str, Any]:
        """Serialize a SteeringSpec for RPC transmission."""
        # Validate layer indices before serialization
        for layer_idx in spec.layers.keys():
            if layer_idx < 0 or layer_idx >= self.layer_count:
                raise ValueError(
                    f"Layer index {layer_idx} out of range [0, {self.layer_count}). "
                    f"Model {self.cfg.model_name} has {self.layer_count} layers."
                )

        serialized_layers = {}
        for layer_idx, layer_spec in spec.layers.items():
            serialized_layer = {}

            if layer_spec.add is not None:
                serialized_layer["add"] = {
                    "vector": steering_runtime.serialize_tensor(
                        layer_spec.add.materialize().to(dtype=self._vector_dtype)
                    ),
                }

            if layer_spec.projection_cap is not None:
                serialized_layer["projection_cap"] = {
                    "vector": steering_runtime.serialize_tensor(
                        layer_spec.projection_cap.vector.to(dtype=torch.float32)
                    ),
                    "min": layer_spec.projection_cap.min,
                    "max": layer_spec.projection_cap.max,
                }

            if layer_spec.ablation is not None:
                serialized_layer["ablation"] = {
                    "vector": steering_runtime.serialize_tensor(
                        layer_spec.ablation.vector.to(dtype=torch.float32)
                    ),
                    "scale": float(layer_spec.ablation.scale),
                }

            serialized_layers[int(layer_idx)] = serialized_layer

        return {"layers": serialized_layers}

    async def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams | None = None,
        *,
        capture_layers: int | Sequence[int] | None = None,
        steering_spec: SteeringSpec | None = None,
        raw_output: bool = False,
        **kwargs: Any,
    ) -> list[str] | tuple[list[str], list[CaptureHandle]] | list[Any] | tuple[list[Any], list[CaptureHandle]]:
        """Generate text with optional activation capture and per-request steering.

        Parameters
        ----------
        prompts : list[str] | str
            Prompt or list of prompts to generate from.
        sampling_params : SamplingParams | None
            Sampling parameters for generation.
        capture_layers : int | Sequence[int] | None
            Layer indices to capture activations from. If provided, returns
            (texts, handles) instead of just texts.
        steering_spec : SteeringSpec | None
            Per-request steering configuration. If provided, applies steering
            vectors, projection caps, and ablations to the specified layers.
        raw_output : bool
            If True, return full RequestOutput objects instead of text strings.
        **kwargs : Any
            Additional sampling parameters (used if sampling_params is None).

        Returns
        -------
        list[str] if capture_layers is None and raw_output is False
        tuple[list[str], list[CaptureHandle]] if capture_layers is not None and raw_output is False
        list[RequestOutput] if capture_layers is None and raw_output is True
        tuple[list[RequestOutput], list[CaptureHandle]] if capture_layers is not None and raw_output is True
        """
        import uuid
        await self._ensure_engine_initialized()

        # Note: No longer using read lock since steering is per-request
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)

        # Setup capture if requested
        handles: list[CaptureHandle] | None = None
        if capture_layers is not None:
            # Convert to tuple
            if isinstance(capture_layers, int):
                layers_tuple = (capture_layers,)
            else:
                layers_tuple = tuple(capture_layers)

            # Register captures for each prompt
            handles = []
            for i, prompt in enumerate(prompts):
                req_id = f"capture_{uuid.uuid4().hex}"
                await self._collective_rpc("register_capture_request", req_id, list(layers_tuple))
                handle = CaptureHandle(
                    request_id=req_id,
                    layer_indices=layers_tuple,
                    model=self,
                )
                handles.append(handle)

        # Register steering spec for each request if provided
        request_ids = []
        if handles:
            request_ids = [h.request_id for h in handles]
        else:
            request_ids = [f"gen_{uuid.uuid4().hex}" for _ in prompts]

        if steering_spec is not None:
            for req_id in request_ids:
                await self._register_steering_spec(req_id, steering_spec)

        try:
            # Generate each prompt
            results = []
            for i, prompt in enumerate(prompts):
                request_id = request_ids[i]

                final_output = None
                async for output in self._engine.generate(prompt, sampling_params, request_id=request_id):
                    final_output = output

                if final_output is None:
                    raise RuntimeError(f"No output for prompt: {prompt}")

                if raw_output:
                    results.append(final_output)
                else:
                    results.append(final_output.outputs[0].text)

            # Return with or without handles
            if handles is not None:
                return results, handles
            else:
                return results
        finally:
            # Clean up steering specs in parallel with timeout handling
            if steering_spec is not None:
                cleanup_tasks = [
                    self._unregister_steering_spec(req_id) for req_id in request_ids
                ]
                try:
                    # Run all cleanups in parallel with a single timeout for the entire batch
                    results = await asyncio.wait_for(
                        asyncio.gather(*cleanup_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                    # Check for failures
                    failures = []
                    for req_id, result in zip(request_ids, results):
                        if isinstance(result, Exception):
                            failures.append(f"{req_id} ({type(result).__name__}: {result})")
                    if failures:
                        logger.warning(f"Steering cleanup failed for {len(failures)} requests: {failures[:5]}")
                except asyncio.TimeoutError:
                    logger.warning(f"Steering cleanup timed out for batch of {len(request_ids)} requests")

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        capture_layers: None = None,
        raw_output: Literal[False] = False,
        **sampling_kwargs: Any,
    ) -> list[ChatResponse]: ...

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        capture_layers: None = None,
        raw_output: Literal[True] = True,
        **sampling_kwargs: Any,
    ) -> list[Any]: ...

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        capture_layers: int | Sequence[int],
        raw_output: Literal[False] = False,
        **sampling_kwargs: Any,
    ) -> tuple[list[ChatResponse], list[CaptureHandle]]: ...

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        capture_layers: int | Sequence[int],
        raw_output: Literal[True] = True,
        **sampling_kwargs: Any,
    ) -> tuple[list[Any], list[CaptureHandle]]: ...

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        capture_layers: int | Sequence[int] | None = None,
        raw_output: bool = False,
        **sampling_kwargs: Any,
    ) -> list[ChatResponse] | list[Any] | tuple[list[ChatResponse], list[CaptureHandle]] | tuple[list[Any], list[CaptureHandle]]:
        """Execute chat-style generation with optional sampling overrides and activation capture.

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
            Additional keyword arguments forwarded to tokenizer.apply_chat_template()
            (for example ``chat_template`` or ``add_generation_prompt``).
        capture_layers : int | Sequence[int] | None
            Layer indices to capture activations from. If provided, returns
            (texts, handles) instead of just texts.
        raw_output : bool
            If True, return full RequestOutput objects with token IDs and logprobs.
            If False (default), return ChatResponse objects with prefill/generated separation.
        **sampling_kwargs : Any
            Keyword arguments used to build a ``SamplingParams`` instance when
            ``sampling_params`` is not supplied.

        Returns
        -------
        list[ChatResponse] if capture_layers is None and raw_output is False
        tuple[list[ChatResponse], list[CaptureHandle]] if capture_layers is not None and raw_output is False
        list[RequestOutput] if capture_layers is None and raw_output is True
        tuple[list[RequestOutput], list[CaptureHandle]] if capture_layers is not None and raw_output is True
        """
        if sampling_params is None:
            sampling_params = SamplingParams(**sampling_kwargs)
        elif sampling_kwargs:
            raise ValueError(
                "Provide either sampling_params or sampling keyword overrides, not both."
            )

        # Normalize to batch format
        single_conversation = isinstance(messages, list) and (
            len(messages) == 0 or isinstance(messages[0], dict)
        )
        if single_conversation:
            batched_messages = [messages]
        else:
            batched_messages = messages

        # Setup chat template options
        chat_kwargs = dict(chat_options or {})
        chat_kwargs.setdefault("chat_template_content_format", "string")

        await self._ensure_engine_initialized()

        # Setup capture if requested
        handles: list[CaptureHandle] | None = None
        if capture_layers is not None:
            # Convert to tuple
            if isinstance(capture_layers, int):
                layers_tuple = (capture_layers,)
            else:
                layers_tuple = tuple(capture_layers)

            # Register captures for each conversation
            handles = []
            for i in range(len(batched_messages)):
                import uuid
                req_id = f"chat_capture_{uuid.uuid4().hex}"
                await self._collective_rpc("register_capture_request", req_id, list(layers_tuple))

                message_boundaries = compute_message_boundaries(
                    batched_messages[i],
                    self.tokenizer,
                    chat_kwargs,
                )

                handle = CaptureHandle(
                    request_id=req_id,
                    layer_indices=layers_tuple,
                    model=self,
                    message_boundaries=message_boundaries,
                )
                handles.append(handle)

        import uuid
        results: list[ChatResponse] | list[Any] = []

        for i, messages_conv in enumerate(batched_messages):
            has_prefill = False
            prefill_text = ""
            if (
                messages_conv
                and messages_conv[-1].get("role") == "assistant"
                and chat_kwargs.get("continue_final_message", False)
            ):
                has_prefill = True
                prefill_text = messages_conv[-1].get("content", "")

            prompt = self.tokenizer.apply_chat_template(
                messages_conv,
                tokenize=False,
                **chat_kwargs,
            )

            if handles:
                request_id = handles[i].request_id
            else:
                request_id = f"chat_{uuid.uuid4().hex}"

            final_output = None
            async for output in self._engine.generate(
                prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output

            if raw_output:
                results.append(final_output)
            else:
                generated_text = final_output.outputs[0].text
                response = ChatResponse(
                    prefill=prefill_text if has_prefill else "",
                    generated=generated_text,
                )
                results.append(response)

        if handles:
            return results, handles
        else:
            return results


    # ------------------------------------------------------------------
    # Sync wrappers for backward compatibility (deprecated)
    # ------------------------------------------------------------------

    def generate_sync(self, *args, **kwargs) -> list[str]:
        """Synchronous wrapper for generate(). DEPRECATED - use async generate()."""
        import asyncio
        import warnings

        warnings.warn(
            "generate_sync() is deprecated. Use async generate() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.generate(*args, **kwargs))

    def chat_sync(self, *args, **kwargs) -> list[str]:
        """Synchronous wrapper for chat(). DEPRECATED - use async chat()."""
        import asyncio
        import warnings

        warnings.warn(
            "chat_sync() is deprecated. Use async chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.chat(*args, **kwargs))

    def generate_with_activations_sync(
        self, *args, **kwargs
    ) -> tuple[str, dict[int, torch.Tensor]]:
        """Synchronous wrapper for generate_with_activations(). DEPRECATED."""
        import asyncio
        import warnings

        warnings.warn(
            "generate_with_activations_sync() is deprecated. Use async generate_with_activations() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.generate_with_activations(*args, **kwargs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_request_captures(
        self,
        request_id: str,
        shm_objects_list: list[SharedMemory] | None = None
    ) -> dict[int, list[dict[str, Any]]]:
        """Fetch and deserialize captures for a single request.

        Args:
            request_id: Request identifier
            shm_objects_list: Optional list to track SharedMemory objects to keep them alive
        """
        await self._ensure_engine_initialized()
        payloads = await self._collective_rpc("fetch_request_activations", request_id)

        decoded: dict[int, list[dict[str, Any]]] = {}
        for worker_payload in payloads:
            for layer_idx_str, tensor_data in worker_payload.items():
                layer_idx = int(layer_idx_str)
                tensor = steering_runtime.deserialize_tensor(
                    tensor_data,
                    device=torch.device("cpu"),
                    dtype=self._vector_dtype,
                    shm_objects_list=shm_objects_list
                )
                if layer_idx not in decoded:
                    decoded[layer_idx] = []
                decoded[layer_idx].append({"hidden": tensor})

        return decoded

    async def fetch_captures_batch(
        self,
        handles: Sequence[CaptureHandle]
    ) -> None:
        """Fetch captures for multiple handles in a single RPC call.

        Args:
            handles: Sequence of CaptureHandle objects to fetch captures for.

        Note:
            This mutates the handles in-place by populating their _captures field.
            Handles that already have captures fetched are skipped.
        """
        await self._ensure_engine_initialized()

        # Filter to handles that need fetching
        to_fetch = [h for h in handles if h._captures is None]
        if not to_fetch:
            return

        # Extract request IDs
        request_ids = [h.request_id for h in to_fetch]

        # Fetch all at once
        batch_payloads = await self._collective_rpc("fetch_batch_captures", request_ids)

        # Deserialize: batch_payloads is a list (one per worker) of
        # dict[str, dict[int, Any]] where outer key is request_id
        results_by_request: dict[str, dict[int, list[dict[str, Any]]]] = {}

        # Track SharedMemory objects per handle to keep them alive
        shm_tracking: dict[str, list[SharedMemory]] = {}  # request_id -> list of SharedMemory objects

        for worker_batch in batch_payloads:
            for request_id, layer_data in worker_batch.items():
                if request_id not in results_by_request:
                    results_by_request[request_id] = {}
                if request_id not in shm_tracking:
                    shm_tracking[request_id] = []

                for layer_idx_str, tensor_data in layer_data.items():
                    layer_idx = int(layer_idx_str)

                    # Check if this is shared memory or bytes encoding
                    if tensor_data.get("encoding") == "shm":
                        # Shared memory path: open shm and create tensor view
                        shm_name = tensor_data["shm_name"]
                        shape = tuple(tensor_data["shape"])
                        dtype_str = tensor_data["dtype"]

                        try:
                            shm = SharedMemory(name=shm_name)

                            # Handle bfloat16 specially
                            if dtype_str == "bfloat16":
                                import ml_dtypes
                                # Create numpy array view as bfloat16
                                np_array = np.ndarray(shape, dtype=ml_dtypes.bfloat16, buffer=shm.buf)
                                # Convert to torch: view as uint16, then reinterpret as bfloat16
                                tensor = torch.from_numpy(np_array.view(np.uint16)).view(torch.bfloat16)
                            else:
                                # Standard dtypes
                                np_dtype = getattr(np, dtype_str.replace("torch.", ""))
                                np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
                                tensor = torch.from_numpy(np_array)

                            # Convert to desired dtype
                            tensor = tensor.to(dtype=self._vector_dtype)

                            # Track SharedMemory object to keep it alive
                            shm_tracking[request_id].append(shm)
                        except Exception as e:
                            logger.error(f"Failed to open shared memory {shm_name}: {e}")
                            raise
                    else:
                        # Bytes encoding: use existing deserialization
                        tensor = steering_runtime.deserialize_tensor(
                            tensor_data,
                            device=torch.device("cpu"),
                            dtype=self._vector_dtype
                        )

                    if layer_idx not in results_by_request[request_id]:
                        results_by_request[request_id][layer_idx] = []
                    results_by_request[request_id][layer_idx].append({"hidden": tensor})

        # Populate handles with captures and shm tracking
        for handle in to_fetch:
            handle._captures = results_by_request.get(handle.request_id, {})
            # Store SharedMemory objects to keep them alive
            shm_objects = shm_tracking.get(handle.request_id, [])
            handle._shm_objects.clear()
            handle._shm_objects.extend(shm_objects)
            # Extract names for cleanup RPC (updates the list in-place that finalizer references)
            handle._shm_names.clear()
            handle._shm_names.extend([shm.name for shm in shm_objects])

    # ------------------------------------------------------------------
    # Internal debugging helpers (used in tests)
    # ------------------------------------------------------------------

    async def _fetch_worker_vectors(self) -> list[dict[int, torch.Tensor]]:
        """Retrieve current worker vectors for validation."""
        await self._ensure_engine_initialized()
        payloads = await self._collective_rpc("fetch_worker_vectors")
        worker_vectors: list[dict[int, torch.Tensor]] = []
        for payload in payloads:
            worker_map: dict[int, torch.Tensor] = {}
            for layer_idx, tensor_payload in payload.items():
                tensor = steering_runtime.deserialize_tensor(
                    tensor_payload,
                    device=torch.device("cpu"),
                    dtype=self._vector_dtype,
                ).clone()
                worker_map[int(layer_idx)] = tensor
            worker_vectors.append(worker_map)
        return worker_vectors

    async def fetch_last_profiler_summaries(self) -> list[dict[str, Any]]:
        """Retrieve the most recent torch profiler summaries from each worker."""
        await self._ensure_engine_initialized()
        payloads = await self._collective_rpc("fetch_last_profile")
        return payloads
