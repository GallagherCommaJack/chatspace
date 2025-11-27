"""vLLM steering model implementation.

This module provides the VLLMSteeringModel class which wraps vLLM's AsyncLLMEngine
with steering vector injection and activation capture capabilities.

Note: This implementation currently delegates to chatspace's vLLM steering runtime.
A clean rewrite is planned for v0.2.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from steerllm.core.capture import CaptureHandle, ChatResponse, MessageBoundary
from steerllm.core.exceptions import BackendError
from steerllm.core.protocols import SyncWrapperMixin
from steerllm.core.specs import (
    AddSpec,
    AblationSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
)

logger = logging.getLogger(__name__)


def _convert_to_chatspace_spec(spec: SteeringSpec) -> Any:
    """Convert steerllm SteeringSpec to chatspace format.

    The chatspace specs have the same structure, so this is mostly
    a namespace conversion.
    """
    # Import chatspace specs
    from chatspace.generation.vllm_steer_model import (
        SteeringSpec as CSSteeringSpec,
        LayerSteeringSpec as CSLayerSteeringSpec,
        AddSpec as CSAddSpec,
        ProjectionCapSpec as CSProjectionCapSpec,
        AblationSpec as CSAblationSpec,
    )

    cs_layers = {}
    for layer_idx, layer_spec in spec.layers.items():
        cs_ops = []
        for op in layer_spec.operations:
            if isinstance(op, AddSpec):
                cs_ops.append(CSAddSpec(vector=op.vector, scale=op.scale))
            elif isinstance(op, ProjectionCapSpec):
                cs_ops.append(CSProjectionCapSpec(
                    vector=op.vector, min=op.min, max=op.max
                ))
            elif isinstance(op, AblationSpec):
                cs_ops.append(CSAblationSpec(vector=op.vector, scale=op.scale))
        cs_layers[layer_idx] = CSLayerSteeringSpec(operations=cs_ops)

    return CSSteeringSpec(layers=cs_layers)


class VLLMSteeringModel(SyncWrapperMixin):
    """vLLM steering backend with zero-copy shared memory capture.

    Provides per-request steering configuration, allowing different requests
    in the same batch to use different steering vectors (heterogeneous batching).

    Parameters
    ----------
    model_name :
        HuggingFace model identifier or path.
    tensor_parallel_size :
        Number of GPUs for tensor parallelism.
    gpu_memory_utilization :
        Fraction of GPU memory to use.
    max_model_len :
        Maximum sequence length. None for auto-detection.
    dtype :
        Model dtype ("auto", "float16", "bfloat16").
    bootstrap_layers :
        Layer indices to pre-warm for steering.
    shm_ttl_seconds :
        Worker-side TTL for shared memory segments.
    shm_max_gb :
        Maximum total shared memory usage.
    **vllm_kwargs :
        Additional arguments passed to vLLM engine.

    Example
    -------
    >>> model = VLLMSteeringModel("Qwen/Qwen3-0.6B")
    >>> steering = SteeringSpec.simple_add(layer=5, vector=v, scale=1.0)
    >>> texts, handles = await model.generate(prompts, steering_spec=steering)
    """

    def __init__(
        self,
        model_name: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        bootstrap_layers: tuple[int, ...] = (),
        shm_ttl_seconds: int = 600,
        shm_max_gb: float = 128.0,
        **vllm_kwargs: Any,
    ) -> None:
        # Lazy import to check for vLLM
        try:
            from vllm import SamplingParams
        except ImportError as e:
            raise BackendError(
                "vLLM backend requires vllm. "
                "Install with: pip install steerllm[vllm]"
            ) from e

        # Import chatspace's VLLMSteerModel
        try:
            from chatspace.generation.vllm_steer_model import (
                VLLMSteerModel as CSVLLMSteerModel,
                VLLMSteeringConfig,
            )
        except ImportError as e:
            raise BackendError(
                "VLLMSteeringModel currently requires chatspace. "
                "Install chatspace or wait for steerllm v0.2 with standalone vLLM support."
            ) from e

        # Create chatspace config
        config = VLLMSteeringConfig(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            bootstrap_layers=bootstrap_layers,
        )

        # Initialize chatspace model
        self._cs_model = CSVLLMSteerModel(
            config,
            shm_ttl_seconds=shm_ttl_seconds,
            shm_max_gb=shm_max_gb,
            **vllm_kwargs,
        )

        self._model_name = model_name

    @property
    def hidden_size(self) -> int:
        """Model's hidden dimension."""
        return self._cs_model.hidden_size

    @property
    def layer_count(self) -> int:
        """Number of transformer layers."""
        return self._cs_model.layer_count

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        """Tokenizer instance."""
        return self._cs_model.tokenizer

    async def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None]:
        """Generate text with optional steering and capture.

        Parameters
        ----------
        prompts :
            Input prompts for generation.
        max_tokens :
            Maximum tokens to generate per prompt.
        temperature :
            Sampling temperature.
        steering_spec :
            Optional steering configuration.
        capture_layers :
            Optional layer indices to capture activations from.
        **sampling_kwargs :
            Additional sampling parameters.

        Returns
        -------
        tuple[list[str], list[CaptureHandle] | None]
            Generated texts and capture handles.
        """
        from vllm import SamplingParams

        # Create sampling params
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **sampling_kwargs,
        )

        # Convert steering spec if provided
        cs_spec = None
        if steering_spec is not None and not steering_spec.is_empty():
            cs_spec = _convert_to_chatspace_spec(steering_spec)

        # Call chatspace model
        outputs, cs_handles = await self._cs_model.generate(
            prompts,
            params,
            steering_spec=cs_spec,
            capture_layers=capture_layers,
        )

        # Extract texts
        texts = [out.outputs[0].text for out in outputs]

        # Convert handles
        handles: list[CaptureHandle] | None = None
        if cs_handles is not None:
            handles = []
            for cs_handle in cs_handles:
                handle = self._wrap_capture_handle(cs_handle)
                handles.append(handle)

        return texts, handles

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[ChatResponse], list[CaptureHandle] | None]:
        """Chat-style generation with optional steering and capture.

        Parameters
        ----------
        messages :
            Single conversation or batch of conversations.
        max_tokens :
            Maximum tokens to generate.
        temperature :
            Sampling temperature.
        steering_spec :
            Optional steering configuration.
        capture_layers :
            Optional layer indices to capture.
        **sampling_kwargs :
            Additional sampling parameters.

        Returns
        -------
        tuple[list[ChatResponse], list[CaptureHandle] | None]
            Chat responses and capture handles.
        """
        from vllm import SamplingParams

        # Normalize to list of conversations
        if messages and isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = messages

        # Create sampling params
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **sampling_kwargs,
        )

        # Convert steering spec if provided
        cs_spec = None
        if steering_spec is not None and not steering_spec.is_empty():
            cs_spec = _convert_to_chatspace_spec(steering_spec)

        # Call chatspace model
        cs_responses, cs_handles = await self._cs_model.chat(
            conversations,
            params,
            steering_spec=cs_spec,
            capture_layers=capture_layers,
        )

        # Convert responses
        responses = []
        for cs_resp in cs_responses:
            if hasattr(cs_resp, 'prefill'):
                resp = ChatResponse(prefill=cs_resp.prefill, generated=cs_resp.generated)
            else:
                resp = ChatResponse(prefill="", generated=str(cs_resp))
            responses.append(resp)

        # Convert handles
        handles: list[CaptureHandle] | None = None
        if cs_handles is not None:
            handles = []
            for i, cs_handle in enumerate(cs_handles):
                # Convert message boundaries if available
                boundaries = None
                if cs_handle.message_boundaries:
                    boundaries = tuple(
                        MessageBoundary(
                            role=b.role,
                            content=b.content,
                            start_token=b.start_token,
                            end_token=b.end_token,
                        )
                        for b in cs_handle.message_boundaries
                    )
                handle = self._wrap_capture_handle(cs_handle, boundaries)
                handles.append(handle)

        return responses, handles

    def _wrap_capture_handle(
        self,
        cs_handle: Any,
        message_boundaries: tuple[MessageBoundary, ...] | None = None,
    ) -> CaptureHandle:
        """Wrap a chatspace CaptureHandle in a steerllm CaptureHandle."""

        async def fetch_fn():
            return await cs_handle.fetch()

        async def cleanup_fn():
            await cs_handle.close()

        handle = CaptureHandle(
            request_id=cs_handle.request_id,
            layer_indices=cs_handle.layer_indices,
            fetch_fn=fetch_fn,
            cleanup_fn=cleanup_fn,
            message_boundaries=message_boundaries,
        )
        return handle

    async def _collective_rpc(self, method: str, *args: Any) -> Any:
        """Proxy to chatspace's collective RPC for advanced usage."""
        return await self._cs_model._collective_rpc(method, *args)
