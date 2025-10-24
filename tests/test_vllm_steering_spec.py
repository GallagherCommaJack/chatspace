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

    def collective_rpc(  # pragma: no cover - simple stub
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
    model = object.__new__(VLLMSteerModel)
    model.cfg = VLLMSteeringConfig()
    model.hidden_size = hidden_size
    model._vector_dtype = torch.float32
    model.layer_count = layer_count
    model.llm = None  # type: ignore[assignment]
    model._engine_client = _DummyEngineClient()
    model._layer_specs = {}
    model._steering_stack = []
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


def test_export_and_apply_steering_spec_roundtrip():
    model = _make_dummy_model(hidden_size=6)

    layer = 3
    vector = torch.linspace(0.0, 1.0, steps=model.hidden_size)
    cap_vector = torch.arange(model.hidden_size, dtype=torch.float32) + 1
    ablation_vector = torch.ones(model.hidden_size, dtype=torch.float32) * 2

    model.set_layer_vector(layer, vector)
    model.set_layer_projection_cap(layer, cap_vector, min=-0.5, max=0.75)
    model.set_layer_ablation(layer, ablation_vector, scale=0.4)

    exported = model.export_steering_spec()
    assert list(exported.layers.keys()) == [layer]
    layer_spec = exported.layers[layer]
    assert layer_spec.add is not None
    assert torch.allclose(layer_spec.add.materialize(), vector.to(dtype=torch.float32))
    assert layer_spec.projection_cap is not None
    cap_unit = cap_vector / cap_vector.norm()
    assert torch.allclose(layer_spec.projection_cap.vector, cap_unit)
    assert layer_spec.projection_cap.min == pytest.approx(-0.5)
    assert layer_spec.projection_cap.max == pytest.approx(0.75)
    assert layer_spec.ablation is not None
    ablation_unit = ablation_vector / ablation_vector.norm()
    assert torch.allclose(layer_spec.ablation.vector, ablation_unit)
    assert layer_spec.ablation.scale == pytest.approx(0.4)

    model.clear_all_vectors()
    assert not model._layer_specs

    model.apply_steering_spec(exported)

    restored_vector = model.current_vector(layer)
    assert torch.allclose(restored_vector, vector.to(dtype=torch.float32))
    restored_cap = model.current_projection_cap(layer)
    assert restored_cap is not None
    assert torch.allclose(restored_cap.vector, cap_unit)
    assert restored_cap.min == pytest.approx(-0.5)
    assert restored_cap.max == pytest.approx(0.75)
    restored_ablation = model.current_ablation(layer)
    assert restored_ablation is not None
    assert torch.allclose(restored_ablation.vector, ablation_unit)
    assert restored_ablation.scale == pytest.approx(0.4)


def test_steering_context_manager_restores_state():
    model = _make_dummy_model(hidden_size=4)

    base_vector = torch.tensor([1.0, -1.0, 0.5, -0.5])
    model.set_layer_vector(2, base_vector)
    baseline = model.export_steering_spec()

    new_spec = SteeringSpec(
        layers={
            2: LayerSteeringSpec(
                add=AddSpec(
                    vector=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    scale=0.0,
                ),
                projection_cap=ProjectionCapSpec(
                    vector=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
                    min=-1.0,
                    max=1.0,
                ),
            ),
            3: LayerSteeringSpec(
                add=AddSpec(
                    vector=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    scale=0.25,
                ),
                ablation=AblationSpec(
                    vector=torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32),
                    scale=0.2,
                ),
            ),
        }
    )

    with model.steering(new_spec):
        applied = model.export_steering_spec()
        _assert_spec_eq(applied, new_spec)
        assert len(model._steering_stack) == 1

    restored = model.export_steering_spec()
    _assert_spec_eq(restored, baseline)
    assert not model._steering_stack


def test_pop_without_push_raises():
    model = _make_dummy_model()
    with pytest.raises(RuntimeError):
        model.pop_steering_spec()
