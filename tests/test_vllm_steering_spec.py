"""Unit tests for steering specification helpers on VLLMSteerModel."""

from __future__ import annotations

import math

import pytest
import torch

from chatspace.generation import (
    AddSpec,
    AblationSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
    VLLMSteerModel,
    VLLMSteeringConfig,
)
from chatspace.vllm_steering import runtime as steering_runtime


class _DummyEngineClient:
    """Record-only stub for the vLLM engine RPC interface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object] | None]] = []

    async def collective_rpc(
        self,
        method,
        timeout: float | None = None,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ):
        if method == steering_runtime.STEERING_RPC_METHOD:
            if not args:
                raise AssertionError("RPC opcode missing in stub call.")
            op = str(args[0])
            payload = args[1:]
            self.calls.append((op, payload, kwargs))
        else:
            name = getattr(method, "__name__", repr(method))
            self.calls.append((name, args, kwargs))
        return [{}]


def _make_dummy_model(hidden_size: int = 8, layer_count: int = 12) -> VLLMSteerModel:
    import asyncio
    model = object.__new__(VLLMSteerModel)
    model.cfg = VLLMSteeringConfig()
    model.hidden_size = hidden_size
    model._vector_dtype = torch.float32
    model.layer_count = layer_count
    model.llm = None  # type: ignore[assignment]
    model._engine_client = _DummyEngineClient()
    model._layer_specs = {}
    model._steering_stack = []
    model._engine_init_lock = asyncio.Lock()
    model._engine_initialized = True  # Skip lazy init for dummy model
    return model


def _unit(vector: torch.Tensor) -> torch.Tensor:
    norm = float(vector.norm().item())
    if not math.isfinite(norm) or norm <= 0:
        raise AssertionError("Vector must have positive finite norm for comparison")
    return vector / norm


def _assert_spec_eq(left: SteeringSpec, right: SteeringSpec) -> None:
    assert set(left.layers.keys()) == set(right.layers.keys())
    for layer_idx in left.layers:
        left_layer = left.layers[layer_idx]
        right_layer = right.layers[layer_idx]
        if left_layer.add is None:
            assert right_layer.add is None
        else:
            assert right_layer.add is not None
            assert torch.allclose(
                left_layer.add.materialize(), right_layer.add.materialize()
            )
            assert left_layer.add.scale == pytest.approx(right_layer.add.scale)
        if left_layer.projection_cap is None:
            assert right_layer.projection_cap is None
        else:
            assert right_layer.projection_cap is not None
            assert torch.allclose(
                _unit(left_layer.projection_cap.vector),
                _unit(right_layer.projection_cap.vector),
            )
            assert left_layer.projection_cap.min == pytest.approx(
                right_layer.projection_cap.min
            )
            assert left_layer.projection_cap.max == pytest.approx(
                right_layer.projection_cap.max
            )
        if left_layer.ablation is None:
            assert right_layer.ablation is None
        else:
            assert right_layer.ablation is not None
            assert torch.allclose(
                _unit(left_layer.ablation.vector),
                _unit(right_layer.ablation.vector),
            )
            assert left_layer.ablation.scale == pytest.approx(right_layer.ablation.scale)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Steering stack (push/pop) removed with global API migration - per-request steering doesn't use a stack")
async def test_pop_without_push_raises():
    model = _make_dummy_model()
    with pytest.raises(RuntimeError):
        await model.pop_steering_spec()
