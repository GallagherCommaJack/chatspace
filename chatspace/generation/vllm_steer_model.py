"""Steering helpers for coordinating vLLM worker state with chatspace.

The helpers in this module wrap ``vllm.LLM`` so steering vectors can be injected
into Qwen and Llama decoder layers without breaking CUDA graph capture.  The primary
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
import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence, cast, overload
import logging

import torch
from vllm import LLM, SamplingParams

from chatspace.vllm_steering import runtime as steering_runtime
from .base import SteerableModel


logger = logging.getLogger(__name__)


class AsyncRWLock:
    """Readers-writer lock for async code.

    Allows multiple concurrent readers OR a single exclusive writer.
    Writers wait for all active readers to complete before acquiring.
    """

    def __init__(self):
        self._readers = 0
        self._writer_waiting = False
        self._lock = asyncio.Lock()
        self._read_ok = asyncio.Condition(self._lock)
        self._write_ok = asyncio.Condition(self._lock)

    @asynccontextmanager
    async def read_lock(self):
        """Acquire read lock (shared access)."""
        async with self._lock:
            # Wait if a writer is waiting or active
            while self._writer_waiting:
                await self._read_ok.wait()
            self._readers += 1

        try:
            yield
        finally:
            async with self._lock:
                self._readers -= 1
                if self._readers == 0:
                    # Notify waiting writers
                    self._write_ok.notify()

    @asynccontextmanager
    async def write_lock(self):
        """Acquire write lock (exclusive access)."""
        async with self._lock:
            # Signal that a writer is waiting
            self._writer_waiting = True
            # Wait until no readers are active
            while self._readers > 0:
                await self._write_ok.wait()

        try:
            yield
        finally:
            async with self._lock:
                self._writer_waiting = False
                # Notify all waiting readers
                self._read_ok.notify_all()


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
class CaptureHandle:
    """Handle for lazily fetching activation captures for a single request.

    Attributes
    ----------
    request_id : str
        Internal request identifier used for fetching captures.
    layer_indices : tuple[int, ...]
        Layer indices that were captured for this request.
    """

    request_id: str
    layer_indices: tuple[int, ...]
    _model: "VLLMSteerModel" = field(repr=False)
    _captures: dict[int, list[dict[str, Any]]] | None = field(default=None, repr=False, init=False)

    async def fetch(self) -> dict[int, list[dict[str, Any]]]:
        """Fetch captures from workers (idempotent).

        Returns
        -------
        dict[int, list[dict[str, Any]]]
            Mapping of layer indices to lists of capture entries. Each entry
            contains "before", "after", "meta" keys.
        """
        if self._captures is None:
            self._captures = await self._model._fetch_request_captures(self.request_id)
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
        return self._captures


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
    """Steerable wrapper around ``vllm.LLM`` for Qwen and Llama models.

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
        llm_kwargs.setdefault(
            "worker_extension_cls", steering_runtime.STEERING_WORKER_EXTENSION
        )

        steering_runtime.ensure_layer_patch_installed()
        steering_runtime.ensure_collective_rpc_gateway_installed()

        # Use AsyncLLMEngine for async API
        from vllm import AsyncEngineArgs, AsyncLLMEngine
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
        self._layer_specs: dict[int, LayerSteeringSpec] = {}
        self._steering_stack: list[SteeringSpec] = []
        self._active_layer: int | None = None

        # Track prompt token lengths for capture reconstruction
        self._last_prompt_lengths: list[int] | None = None
        self._tokenizer = None

        # RWLock for coordinating generation (readers) and steering changes (writers)
        self._steering_rwlock = AsyncRWLock()

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

            # Initialize worker state
            setup_info = await self._collective_rpc("initialize_worker_state", self._init_layers)
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
            layer_count = self.layer_count

            for idx in self._init_layers:
                if idx < 0 or idx >= layer_count:
                    raise ValueError(
                        f"bootstrap layer {idx} is out of range for model with {layer_count} layers"
                    )
                self._ensure_layer_spec(idx)

            if self._init_layers:
                self.set_target_layer(self._init_layers[0])
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

    async def _broadcast_add(self, layer_idx: int, add_spec: AddSpec) -> None:
        payload = steering_runtime.serialize_tensor(
            add_spec.materialize().to(dtype=self._vector_dtype)
        )
        await self._collective_rpc("set_worker_vector", int(layer_idx), payload)
        layer_spec = self._ensure_layer_spec(int(layer_idx))
        layer_spec.add = add_spec.clone()

    async def _broadcast_projection_cap(
        self, layer_idx: int, spec: ProjectionCapSpec
    ) -> None:
        payload = {
            "vector": steering_runtime.serialize_tensor(
                spec.vector.to(dtype=torch.float32)
            ),
            "min": spec.min,
            "max": spec.max,
        }
        await self._collective_rpc("set_worker_projection_cap", int(layer_idx), payload)
        layer_spec = self._ensure_layer_spec(int(layer_idx))
        layer_spec.projection_cap = spec.clone()

    async def _broadcast_ablation(self, layer_idx: int, spec: AblationSpec) -> None:
        payload = {
            "vector": steering_runtime.serialize_tensor(
                spec.vector.to(dtype=self._vector_dtype)
            ),
            "scale": float(spec.scale),
        }
        await self._collective_rpc("set_worker_ablation", int(layer_idx), payload)
        layer_spec = self._ensure_layer_spec(int(layer_idx))
        layer_spec.ablation = spec.clone()

    async def _broadcast_clear(self, layer_idx: int | None = None) -> None:
        if layer_idx is None:
            await self._collective_rpc("clear_worker_vector")
            await self._collective_rpc("clear_worker_projection_cap")
            await self._collective_rpc("clear_worker_ablation")
            self._layer_specs.clear()
        else:
            target = int(layer_idx)
            await self._collective_rpc("clear_worker_vector", target)
            await self._collective_rpc("clear_worker_projection_cap", target)
            await self._collective_rpc("clear_worker_ablation", target)
            layer_spec = self._layer_specs.get(target)
            if layer_spec is not None:
                layer_spec.add = None
                layer_spec.projection_cap = None
                layer_spec.ablation = None
            self._prune_layer_entry(target)

    async def set_layer_vector(self, layer_idx: int, vector: torch.Tensor | None) -> None:
        await self._ensure_engine_initialized()
        async with self._steering_rwlock.write_lock():
            target = int(layer_idx)
            if vector is None:
                await self._broadcast_clear(target)
                return

            prepared = self._prepare_layer_vector(
                vector, context="Steering vector update"
            )
            magnitude = float(prepared.norm().item())
            if not math.isfinite(magnitude) or magnitude <= 0.0:
                await self._broadcast_clear(target)
                return
            unit = (prepared / magnitude).contiguous()
            add_spec = AddSpec(vector=unit, scale=magnitude)
            await self._broadcast_add(target, add_spec)

    async def set_vector(self, vector: torch.Tensor | None) -> None:
        """Set the steering vector for the active layer."""
        if self._active_layer is None:
            raise ValueError("Target layer not set. Call set_target_layer first.")
        await self.set_layer_vector(self._active_layer, vector)

    def set_target_layer(self, layer_idx: int) -> None:
        """Select which layer receives ``set_vector`` updates."""
        idx = int(layer_idx)
        self._ensure_layer_spec(idx)
        self._active_layer = idx

    async def set_layer_projection_cap(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> None:
        await self._ensure_engine_initialized()
        async with self._steering_rwlock.write_lock():
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
            await self._broadcast_projection_cap(target, spec)

    async def clear_layer_projection_cap(self, layer_idx: int) -> None:
        await self._ensure_engine_initialized()
        async with self._steering_rwlock.write_lock():
            target = int(layer_idx)
            await self._collective_rpc("clear_worker_projection_cap", target)
            layer_spec = self._layer_specs.get(target)
            if layer_spec is not None:
                layer_spec.projection_cap = None
            self._prune_layer_entry(target)

    def current_projection_cap(self, layer_idx: int) -> ProjectionCapSpec | None:
        layer_spec = self._layer_specs.get(int(layer_idx))
        if layer_spec is None or layer_spec.projection_cap is None:
            return None
        return layer_spec.projection_cap.clone()

    async def set_projection_cap_precision(
        self, dtype: torch.dtype | str | None
    ) -> None:
        """Override the working precision used for projection cap math."""
        await self._ensure_engine_initialized()
        if dtype is None:
            dtype_name: str | None = None
        elif isinstance(dtype, torch.dtype):
            dtype_name = str(dtype).removeprefix("torch.")
        elif isinstance(dtype, str):
            dtype_name = dtype.removeprefix("torch.")
        else:
            raise TypeError("dtype must be a torch.dtype, string, or None.")
        await self._collective_rpc("set_projection_cap_precision", dtype_name)

    async def set_layer_ablation(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        scale: float,
    ) -> None:
        await self._ensure_engine_initialized()
        async with self._steering_rwlock.write_lock():
            target = int(layer_idx)
            self._ensure_layer_spec(target)
            if not isinstance(scale, (int, float)):
                raise TypeError("Ablation scale must be a numeric value.")
            scale_value = float(scale)
            if not math.isfinite(scale_value):
                raise ValueError("Ablation scale must be finite.")
            prepared = self._prepare_direction_vector(vector, context="Ablation vector")
            spec = AblationSpec(vector=prepared, scale=scale_value)
            await self._broadcast_ablation(target, spec)

    async def clear_layer_ablation(self, layer_idx: int) -> None:
        await self._ensure_engine_initialized()
        async with self._steering_rwlock.write_lock():
            target = int(layer_idx)
            await self._collective_rpc("clear_worker_ablation", target)
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

    async def apply_steering_spec(
        self, spec: SteeringSpec, *, clear_missing: bool = True
    ) -> None:
        """Apply a previously captured steering specification.

        Each layer entry replaces the additive vector, projection cap, and
        ablation settings. Layers omitted from the spec are cleared when
        ``clear_missing`` is ``True``. When a layer entry omits the additive
        vector the helper clears any existing steering state before applying
        projection caps or ablations present in the spec.
        """
        await self._ensure_engine_initialized()
        async with self._steering_rwlock.write_lock():
            target_layers = {int(idx) for idx in spec.layers.keys()}
            if clear_missing:
                existing_layers = set(self._layer_specs.keys())
                for layer_idx in sorted(existing_layers - target_layers):
                    # Clear operations: call _broadcast_clear directly to avoid nested locks
                    target = int(layer_idx)
                    await self._broadcast_clear(target)

            for layer_idx_raw, layer_spec in spec.layers.items():
                layer_idx = int(layer_idx_raw)
                add_spec = layer_spec.add
                cleared = add_spec is None
                if cleared:
                    # Call _broadcast_clear directly to avoid nested locks
                    await self._broadcast_clear(layer_idx)
                else:
                    await self._broadcast_add(layer_idx, add_spec)

                projection_spec = layer_spec.projection_cap
                if projection_spec is not None:
                    # Call _broadcast_projection_cap directly to avoid nested locks
                    target = int(layer_idx)
                    self._ensure_layer_spec(target)
                    await self._broadcast_projection_cap(target, projection_spec)
                elif not cleared:
                    # Call clear RPC directly to avoid nested locks
                    target = int(layer_idx)
                    await self._collective_rpc("clear_worker_projection_cap", target)
                    layer_spec_obj = self._layer_specs.get(target)
                    if layer_spec_obj is not None:
                        layer_spec_obj.projection_cap = None
                    self._prune_layer_entry(target)

                ablation_spec = layer_spec.ablation
                if ablation_spec is not None:
                    # Call _broadcast_ablation directly to avoid nested locks
                    target = int(layer_idx)
                    self._ensure_layer_spec(target)
                    await self._broadcast_ablation(target, ablation_spec)
                elif not cleared:
                    # Call clear RPC directly to avoid nested locks
                    target = int(layer_idx)
                    await self._collective_rpc("clear_worker_ablation", target)
                    layer_spec_obj = self._layer_specs.get(target)
                    if layer_spec_obj is not None:
                        layer_spec_obj.ablation = None
                    self._prune_layer_entry(target)

    async def push_steering_spec(
        self, spec: SteeringSpec, *, clear_missing: bool = True
    ) -> None:
        """Push current steering onto a stack and apply ``spec``."""
        baseline = self.export_steering_spec().clone()
        self._steering_stack.append(baseline)
        await self.apply_steering_spec(spec, clear_missing=clear_missing)

    async def pop_steering_spec(self) -> SteeringSpec:
        """Restore the most recently pushed steering spec."""
        if not self._steering_stack:
            raise RuntimeError("No steering spec to pop.")
        baseline = self._steering_stack.pop()
        await self.apply_steering_spec(baseline, clear_missing=True)
        return baseline

    @asynccontextmanager
    async def steering(
        self, spec: SteeringSpec, *, clear_missing: bool = True
    ):
        """Async context manager that reapplies previous steering on exit.

        Usage:
            async with model.steering(spec):
                await model.generate(...)
        """
        await self.push_steering_spec(spec, clear_missing=clear_missing)
        try:
            yield
        finally:
            await self.pop_steering_spec()

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

    async def clear_layer_vector(self, layer_idx: int) -> None:
        await self.set_layer_vector(layer_idx, None)

    async def clear_all_vectors(self) -> None:
        await self._ensure_engine_initialized()
        async with self._steering_rwlock.write_lock():
            await self._broadcast_clear()

    async def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams | None = None,
        *,
        capture_layers: int | Sequence[int] | None = None,
        raw_output: bool = False,
        **kwargs: Any,
    ) -> list[str] | tuple[list[str], list[CaptureHandle]] | list[Any] | tuple[list[Any], list[CaptureHandle]]:
        """Generate text with optional activation capture.

        Parameters
        ----------
        prompts : list[str] | str
            Prompt or list of prompts to generate from.
        sampling_params : SamplingParams | None
            Sampling parameters for generation.
        capture_layers : int | Sequence[int] | None
            Layer indices to capture activations from. If provided, returns
            (texts, handles) instead of just texts.
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

        # Acquire read lock for the duration of generation
        # This prevents steering changes while requests are in flight
        async with self._steering_rwlock.read_lock():
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
                        _model=self,
                    )
                    handles.append(handle)

            # Generate each prompt
            results = []
            for i, prompt in enumerate(prompts):
                # Use capture request_id if capturing, otherwise random
                if handles:
                    request_id = handles[i].request_id
                else:
                    request_id = f"gen_{uuid.uuid4().hex}"

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
            if handles:
                return results, handles
            else:
                return results

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        prefill_assistant: str | bool | None = None,
        raw_output: Literal[False] = False,
        **sampling_kwargs: Any,
    ) -> list[str]: ...

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        prefill_assistant: str | bool | None = None,
        raw_output: Literal[True] = True,
        **sampling_kwargs: Any,
    ) -> list[Any]: ...  # RequestOutput not imported, use Any

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        prefill_assistant: str | bool | None = None,
        raw_output: bool = False,
        **sampling_kwargs: Any,
    ) -> list[str] | list[Any]:
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
        raw_output : bool
            If True, return full RequestOutput objects with token IDs and logprobs.
            If False (default), return just the generated text strings.
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

        await self._ensure_engine_initialized()

        # Convert messages to prompts using tokenizer's chat template
        import uuid
        results: list[str] | list[Any] = []

        for messages_conv in prepared_messages:
            # Apply chat template to convert messages to prompt string
            prompt = self.tokenizer.apply_chat_template(
                messages_conv,
                tokenize=False,
                **chat_kwargs,
            )

            # Generate using the formatted prompt
            request_id = f"chat_{uuid.uuid4().hex}"
            final_output = None

            async for output in self._engine.generate(
                prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output

            if final_output is None:
                raise RuntimeError("No output from chat")

            if raw_output:
                results.append(final_output)
            else:
                text = final_output.outputs[0].text
                if trim_prefix and text.startswith(trim_prefix):
                    text = text[len(trim_prefix) :].lstrip()
                results.append(text)

        return results

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
        request_id: str
    ) -> dict[int, list[dict[str, Any]]]:
        """Fetch and deserialize captures for a single request."""
        await self._ensure_engine_initialized()
        payloads = await self._collective_rpc("fetch_request_activations", request_id)

        # Deserialize tensors to CPU for analysis
        decoded: dict[int, list[dict[str, Any]]] = {}
        for worker_payload in payloads:
            for layer_idx_str, tensor_data in worker_payload.items():
                layer_idx = int(layer_idx_str)
                tensor = steering_runtime.deserialize_tensor(
                    tensor_data,
                    device=torch.device("cpu"),
                    dtype=self._vector_dtype
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

        for worker_batch in batch_payloads:
            for request_id, layer_data in worker_batch.items():
                if request_id not in results_by_request:
                    results_by_request[request_id] = {}

                for layer_idx_str, tensor_data in layer_data.items():
                    layer_idx = int(layer_idx_str)
                    tensor = steering_runtime.deserialize_tensor(
                        tensor_data,
                        device=torch.device("cpu"),
                        dtype=self._vector_dtype
                    )
                    if layer_idx not in results_by_request[request_id]:
                        results_by_request[request_id][layer_idx] = []
                    results_by_request[request_id][layer_idx].append({"hidden": tensor})

        # Populate handles
        for handle in to_fetch:
            handle._captures = results_by_request.get(handle.request_id, {})

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

