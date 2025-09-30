"""Tests for chatspace.hf_embed.model module."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from chatspace.hf_embed.model import (
    _default_model_kwargs,
    _default_tokenizer_kwargs,
    _ModelRunner,
)
from chatspace.hf_embed.config import SentenceTransformerConfig


def test_default_model_kwargs():
    """Test model kwargs construction."""
    cfg = SentenceTransformerConfig(
        dataset="test",
        attention_impl="flash_attention_2",
        device="cuda",
        dtype="bfloat16",
    )

    kwargs = _default_model_kwargs(cfg)

    assert kwargs["attn_implementation"] == "flash_attention_2"
    assert kwargs["device_map"] == "cuda"
    assert kwargs["dtype"] == "bfloat16"


def test_default_model_kwargs_custom():
    """Test model kwargs with custom overrides."""
    cfg = SentenceTransformerConfig(
        dataset="test",
        model_kwargs={"custom_param": "value"},
    )

    kwargs = _default_model_kwargs(cfg)
    assert kwargs["custom_param"] == "value"


def test_default_tokenizer_kwargs():
    """Test tokenizer kwargs construction."""
    cfg = SentenceTransformerConfig(
        dataset="test",
        tokenizer_padding="left",
    )

    kwargs = _default_tokenizer_kwargs(cfg)

    assert kwargs["padding"] is True
    assert kwargs["truncation"] is True
    assert kwargs["padding_side"] == "left"


def test_default_tokenizer_kwargs_custom():
    """Test tokenizer kwargs with custom overrides."""
    cfg = SentenceTransformerConfig(
        dataset="test",
        tokenizer_kwargs={"custom_param": "value"},
    )

    kwargs = _default_tokenizer_kwargs(cfg)
    assert kwargs["custom_param"] == "value"


def create_mock_model():
    """Create a mock SentenceTransformer model."""
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.eval.return_value = mock_model
    mock_model.requires_grad_ = MagicMock()

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_model.tokenizer = mock_tokenizer

    return mock_model


def test_model_runner_initialization():
    """Test ModelRunner initialization."""
    mock_model = create_mock_model()
    runner = _ModelRunner(mock_model, compile_enabled=False, compile_mode=None)

    assert runner.model == mock_model
    assert runner.device == torch.device("cpu")
    assert runner.compile_enabled is False
    assert runner.pad_values["input_ids"] == 0
    assert runner.pad_values["attention_mask"] == 0


def test_model_runner_tokenize():
    """Test tokenization."""
    mock_model = create_mock_model()

    # Mock tokenize method
    mock_model.tokenize.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }

    runner = _ModelRunner(mock_model, compile_enabled=False, compile_mode=None)
    result = runner.tokenize("test text", max_length=128)

    assert "input_ids" in result
    assert "attention_mask" in result
    # Should squeeze batch dimension
    assert result["input_ids"].shape == (4,)
    assert result["attention_mask"].shape == (4,)


def test_model_runner_forward():
    """Test forward pass."""
    mock_model = create_mock_model()

    # Mock forward method
    expected_output = {"sentence_embedding": torch.randn(2, 768)}
    mock_model.forward.return_value = expected_output

    runner = _ModelRunner(mock_model, compile_enabled=False, compile_mode=None)

    features = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }

    output = runner.forward(features)

    assert "sentence_embedding" in output
    mock_model.forward.assert_called_once()


def test_model_runner_warmup_disabled():
    """Test warmup when compilation is disabled."""
    mock_model = create_mock_model()
    runner = _ModelRunner(mock_model, compile_enabled=False, compile_mode=None)

    timings = runner.warmup([128, 256, 512])

    # Should return empty dict when compilation is disabled
    assert timings == {}


def test_model_runner_dummy_features():
    """Test dummy feature generation."""
    mock_model = create_mock_model()
    runner = _ModelRunner(mock_model, compile_enabled=False, compile_mode=None)

    features = runner._dummy_features(seq_len=8)

    assert "input_ids" in features
    assert "attention_mask" in features
    assert features["input_ids"].shape == (1, 8)
    assert features["attention_mask"].shape == (1, 8)

    # Attention mask should be all ones
    assert torch.all(features["attention_mask"] == 1)

    # Input IDs should be pad tokens
    assert torch.all(features["input_ids"] == 0)