"""Tests for legacy persona steering configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chatspace.generation import LegacyExperiment, load_legacy_role_trait_config


def _write_legacy_bundle(path: Path) -> None:
    payload = {
        "vectors": {
            "layer_0/foo": {
                "vector": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float16),
                "layer": 0,
            },
            "layer_1/bar": {
                "vector": torch.tensor([0.0, -3.0, 4.0], dtype=torch.float16),
                "layer": 1,
            },
        },
        "experiments": [
            {
                "id": "expA",
                "interventions": [
                    {"vector": "layer_0/foo", "cap": -0.5},
                    {"vector": "layer_1/bar"},
                ],
            },
            {
                "id": "expB",
                "interventions": [{"vector": "layer_1/bar", "cap": 1.25}],
            },
        ],
    }
    torch.save(payload, path)


def test_load_legacy_role_trait_config(tmp_path: Path) -> None:
    bundle_path = tmp_path / "legacy.pt"
    _write_legacy_bundle(bundle_path)

    experiments = load_legacy_role_trait_config(bundle_path)
    assert [exp.id for exp in experiments] == ["expA", "expB"]

    first = experiments[0]
    assert isinstance(first, LegacyExperiment)
    spec = first.spec
    assert set(spec.layers.keys()) == {0, 1}

    layer0 = spec.layers[0]
    assert layer0.add is not None
    assert torch.allclose(layer0.add.materialize(), torch.tensor([1.0, 0.0, 0.0]))
    assert layer0.projection_cap is not None
    assert layer0.projection_cap.cap_below == pytest.approx(-0.5)
    assert layer0.projection_cap.cap_above is None
    assert torch.allclose(layer0.projection_cap.vector, torch.tensor([1.0, 0.0, 0.0]))

    layer1 = spec.layers[1]
    assert layer1.add is not None
    assert torch.allclose(layer1.add.materialize(), torch.tensor([0.0, -3.0, 4.0]))
    assert layer1.projection_cap is None

    second = experiments[1].spec
    assert second.layers[1].projection_cap is not None
    assert second.layers[1].projection_cap.cap_below == pytest.approx(1.25)


def test_loader_rejects_unknown_vector(tmp_path: Path) -> None:
    bundle_path = tmp_path / "legacy.pt"
    payload = {
        "vectors": {"layer_0/foo": {"vector": torch.ones(2), "layer": 0}},
        "experiments": [
            {
                "id": "bad",
                "interventions": [{"vector": "missing"}],
            }
        ],
    }
    torch.save(payload, bundle_path)

    with pytest.raises(KeyError, match="unknown vector"):
        load_legacy_role_trait_config(bundle_path)
