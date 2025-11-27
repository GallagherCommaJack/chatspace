"""Tests for hidden state extraction and output reconstruction.

Tests cover:
- _extract_hidden_from_output() with different output formats
- Qwen (delta, residual) tuple handling
- _reconstruct_output_with_hidden() correctness
- Malformed output error handling
- Fallback paths for unknown formats
- Edge cases (empty tuples, non-tensor elements, None)
"""

import pytest
import torch

from chatspace.vllm_steering import runtime


class MockOutput:
    """Mock object with last_hidden_state attribute."""

    def __init__(self, hidden):
        self.last_hidden_state = hidden


@pytest.fixture
def sample_hidden():
    """Create a sample hidden state tensor."""
    return torch.randn(4, 8, 512)  # [batch, seq_len, hidden_size]


@pytest.fixture
def qwen_output():
    """Create a Qwen-style (delta, residual) output."""
    # vLLM Qwen layers return (delta, residual)
    # where hidden_state = residual + delta
    residual = torch.randn(4, 8, 512)
    delta = torch.randn(4, 8, 512)
    return (delta, residual)


def test_extract_from_tensor(sample_hidden):
    """Test extraction from direct tensor output."""
    result = runtime._extract_hidden_from_output(sample_hidden)
    assert result is sample_hidden
    assert torch.equal(result, sample_hidden)


def test_extract_from_qwen_tuple(qwen_output):
    """Test extraction from Qwen (delta, residual) tuple.

    Should return residual + delta to match HuggingFace format.
    """
    delta, residual = qwen_output
    result = runtime._extract_hidden_from_output(qwen_output)

    assert result is not None
    assert isinstance(result, torch.Tensor)
    # Should equal residual + delta
    expected = residual + delta
    assert torch.allclose(result, expected, rtol=1e-5)


def test_extract_from_single_element_tuple(sample_hidden):
    """Test extraction from tuple with single tensor element."""
    output = (sample_hidden,)
    result = runtime._extract_hidden_from_output(output)

    assert result is sample_hidden
    assert torch.equal(result, sample_hidden)


def test_extract_from_list_qwen_format(qwen_output):
    """Test extraction from list in Qwen format."""
    delta, residual = qwen_output
    output = [delta, residual]  # List instead of tuple
    result = runtime._extract_hidden_from_output(output)

    assert result is not None
    expected = residual + delta
    assert torch.allclose(result, expected, rtol=1e-5)


def test_extract_from_single_element_list(sample_hidden):
    """Test extraction from list with single tensor."""
    output = [sample_hidden]
    result = runtime._extract_hidden_from_output(output)

    assert result is sample_hidden


def test_extract_from_dict_with_key(sample_hidden):
    """Test extraction from dict with 'last_hidden_state' key."""
    output = {"last_hidden_state": sample_hidden, "other_key": "other_value"}
    result = runtime._extract_hidden_from_output(output)

    assert result is sample_hidden
    assert torch.equal(result, sample_hidden)


def test_extract_from_object_with_attribute(sample_hidden):
    """Test extraction from object with last_hidden_state attribute."""
    output = MockOutput(sample_hidden)
    result = runtime._extract_hidden_from_output(output)

    assert result is sample_hidden
    assert torch.equal(result, sample_hidden)


def test_extract_from_empty_tuple():
    """Test extraction from empty tuple raises TypeError."""
    output = ()
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(output)


def test_extract_from_empty_list():
    """Test extraction from empty list raises TypeError."""
    output = []
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(output)


def test_extract_from_tuple_with_non_tensor():
    """Test extraction from tuple with non-tensor elements raises TypeError."""
    output = ("not a tensor", "also not a tensor")
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(output)


def test_extract_from_tuple_mixed_types(sample_hidden):
    """Test extraction from tuple with first element as tensor, second as non-tensor."""
    output = (sample_hidden, "not a tensor")
    result = runtime._extract_hidden_from_output(output)

    # Should fall back to first element
    assert result is sample_hidden


def test_extract_from_dict_without_key():
    """Test extraction from dict without 'last_hidden_state' key raises TypeError."""
    output = {"other_key": torch.randn(4, 8, 512)}
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(output)


def test_extract_from_object_without_attribute():
    """Test extraction from object without last_hidden_state attribute raises TypeError."""
    output = type("MockBadOutput", (), {})()
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(output)


def test_extract_from_none():
    """Test extraction from None raises TypeError."""
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(None)


def test_extract_from_scalar():
    """Test extraction from scalar value raises TypeError."""
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(42)


def test_extract_from_string():
    """Test extraction from string raises TypeError."""
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output("not a valid output")


def test_is_qwen_layer_output_positive(qwen_output):
    """Test _is_qwen_layer_output returns True for valid Qwen format."""
    assert runtime._is_qwen_layer_output(qwen_output) is True


def test_is_qwen_layer_output_negative_cases():
    """Test _is_qwen_layer_output returns False for non-Qwen formats."""
    # Single tensor
    assert runtime._is_qwen_layer_output(torch.randn(4, 8, 512)) is False

    # Single element tuple
    assert runtime._is_qwen_layer_output((torch.randn(4, 8, 512),)) is False

    # Tuple with non-tensors
    assert runtime._is_qwen_layer_output(("not", "tensors")) is False

    # List (not tuple)
    assert runtime._is_qwen_layer_output([torch.randn(4, 8), torch.randn(4, 8)]) is False

    # Empty tuple
    assert runtime._is_qwen_layer_output(()) is False

    # None
    assert runtime._is_qwen_layer_output(None) is False


def test_extract_from_three_element_tuple(sample_hidden):
    """Test extraction from tuple with 3+ elements (Qwen format with extras).

    Should still extract residual + delta from first two elements.
    """
    residual = torch.randn_like(sample_hidden)
    delta = sample_hidden - residual
    extra = torch.randn(10)  # Extra element

    output = (delta, residual, extra)
    result = runtime._extract_hidden_from_output(output)

    assert result is not None
    expected = residual + delta
    assert torch.allclose(result, expected, rtol=1e-5)


def test_extract_from_dict_with_non_tensor_value():
    """Test extraction from dict where 'last_hidden_state' is not a tensor."""
    output = {"last_hidden_state": "not a tensor"}
    result = runtime._extract_hidden_from_output(output)

    # Should return the value even if not a tensor (no type check in dict path)
    # Actually, looking at the code, it just returns output["last_hidden_state"]
    # without checking if it's a tensor
    assert result == "not a tensor"


def test_extract_from_object_with_non_tensor_attribute():
    """Test extraction from object where last_hidden_state is not a tensor raises TypeError."""
    output = MockOutput("not a tensor")

    # The code checks isinstance(hidden, torch.Tensor) so should raise TypeError
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(output)


def test_reconstruct_output_with_hidden_tensor():
    """Test _reconstruct_output_with_hidden with direct tensor."""
    original_hidden = torch.randn(4, 8, 512)
    new_hidden = torch.randn(4, 8, 512)

    result = runtime._reconstruct_output_with_hidden(original_hidden, original_hidden, new_hidden)
    assert result is new_hidden


def test_reconstruct_output_with_hidden_qwen_tuple():
    """Test _reconstruct_output_with_hidden with Qwen (delta, residual) tuple.

    Should reconstruct delta such that residual + new_delta = new_hidden.
    """
    original_residual = torch.randn(4, 8, 512)
    original_delta = torch.randn(4, 8, 512)
    original = (original_delta, original_residual)
    original_hidden = original_residual + original_delta

    new_hidden = torch.randn(4, 8, 512)

    result = runtime._reconstruct_output_with_hidden(original, original_hidden, new_hidden)

    assert isinstance(result, tuple)
    assert len(result) == 2

    new_delta, residual = result

    # Residual should be unchanged
    assert torch.equal(residual, original_residual)

    # new_delta should be computed such that residual + new_delta = new_hidden
    expected_delta = new_hidden - original_residual
    assert torch.allclose(new_delta, expected_delta, atol=1e-6, rtol=1e-5)

    # Verify reconstruction matches new_hidden (within numerical precision)
    reconstructed_hidden = residual + new_delta
    assert reconstructed_hidden.shape == new_hidden.shape
    # The reconstruction should be very close (exact in theory, but allow for FP errors)
    max_diff = torch.abs(reconstructed_hidden - new_hidden).max().item()
    assert max_diff < 1e-6, f"Max difference: {max_diff}"


def test_reconstruct_output_with_hidden_single_tuple():
    """Test _reconstruct_output_with_hidden with single-element tuple."""
    original_hidden = torch.randn(4, 8, 512)
    original = (original_hidden,)
    new_hidden = torch.randn(4, 8, 512)

    result = runtime._reconstruct_output_with_hidden(original, original_hidden, new_hidden)

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0] is new_hidden


def test_reconstruct_output_with_hidden_list():
    """Test _reconstruct_output_with_hidden with list."""
    original_hidden = torch.randn(4, 8, 512)
    original = [original_hidden, "extra"]
    new_hidden = torch.randn(4, 8, 512)

    result = runtime._reconstruct_output_with_hidden(original, original_hidden, new_hidden)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is new_hidden
    assert result[1] == "extra"


def test_reconstruct_output_with_hidden_dict():
    """Test _reconstruct_output_with_hidden with dict."""
    original_hidden = torch.randn(4, 8, 512)
    original = {"last_hidden_state": original_hidden, "other": "value"}
    new_hidden = torch.randn(4, 8, 512)

    result = runtime._reconstruct_output_with_hidden(original, original_hidden, new_hidden)

    assert isinstance(result, dict)
    assert result["last_hidden_state"] is new_hidden
    assert result["other"] == "value"


def test_reconstruct_output_with_hidden_object():
    """Test _reconstruct_output_with_hidden with object attribute."""
    original_hidden = torch.randn(4, 8, 512)
    original = MockOutput(original_hidden)
    new_hidden = torch.randn(4, 8, 512)

    result = runtime._reconstruct_output_with_hidden(original, original_hidden, new_hidden)

    assert result is original
    assert result.last_hidden_state is new_hidden


def test_reconstruct_output_with_hidden_unsupported_type():
    """Test _reconstruct_output_with_hidden with unsupported type returns transformed_hidden."""
    original = 42  # Unsupported type
    original_hidden = torch.randn(4, 8, 512)
    new_hidden = torch.randn(4, 8, 512)

    # Looking at the code, it returns transformed_hidden for unsupported types
    result = runtime._reconstruct_output_with_hidden(original, original_hidden, new_hidden)
    assert result is new_hidden


def test_extract_then_reconstruct_roundtrip(qwen_output):
    """Test extract → transform → reconstruct roundtrip preserves structure.

    This tests the full pipeline of extracting hidden state, modifying it,
    and reconstructing the original output format.
    """
    # Extract hidden state
    hidden = runtime._extract_hidden_from_output(qwen_output)
    assert hidden is not None

    # Transform it (add small perturbation)
    transformed = hidden + 0.1

    # Reconstruct output
    reconstructed = runtime._reconstruct_output_with_hidden(qwen_output, hidden, transformed)

    # Should maintain tuple structure
    assert isinstance(reconstructed, tuple)
    assert len(reconstructed) == 2

    # Extract from reconstructed should give transformed hidden
    extracted_after = runtime._extract_hidden_from_output(reconstructed)
    assert extracted_after.shape == transformed.shape
    # Check that values are close (allow for numerical precision)
    max_diff = torch.abs(extracted_after - transformed).max().item()
    assert max_diff < 1e-6, f"Max difference: {max_diff}"


def test_extract_from_tuple_with_three_tensors():
    """Test extraction from tuple with three tensor elements.

    This tests the case where output has more elements than expected.
    Should still extract residual + delta from first two.
    """
    delta = torch.randn(4, 8, 512)
    residual = torch.randn(4, 8, 512)
    extra = torch.randn(4, 8, 512)

    output = (delta, residual, extra)
    result = runtime._extract_hidden_from_output(output)

    assert result is not None
    expected = residual + delta
    assert torch.allclose(result, expected, rtol=1e-5)


def test_reconstruct_qwen_with_extra_elements():
    """Test reconstructing Qwen tuple with extra elements beyond delta/residual."""
    delta = torch.randn(4, 8, 512)
    residual = torch.randn(4, 8, 512)
    extra1 = "extra1"
    extra2 = 42

    original = (delta, residual, extra1, extra2)
    original_hidden = residual + delta
    new_hidden = torch.randn(4, 8, 512)

    result = runtime._reconstruct_output_with_hidden(original, original_hidden, new_hidden)

    assert isinstance(result, tuple)
    assert len(result) == 4

    new_delta, res, e1, e2 = result

    # Residual and extras should be unchanged
    assert torch.equal(res, residual)
    assert e1 == extra1
    assert e2 == extra2

    # New delta should reconstruct new_hidden
    reconstructed_hidden = res + new_delta
    assert reconstructed_hidden.shape == new_hidden.shape
    max_diff = torch.abs(reconstructed_hidden - new_hidden).max().item()
    assert max_diff < 1e-6, f"Max difference: {max_diff}"


def test_extract_with_mismatched_tensor_shapes():
    """Test extraction when delta and residual have different shapes.

    This is a malformed case - should still try to add them but may fail.
    """
    delta = torch.randn(4, 8, 512)
    residual = torch.randn(4, 10, 512)  # Different seq_len

    output = (delta, residual)

    # Extraction will try to add them, which will fail due to shape mismatch
    with pytest.raises(RuntimeError):
        runtime._extract_hidden_from_output(output)


def test_extract_from_nested_tuple():
    """Test extraction from nested tuple structure raises TypeError.

    The code doesn't recursively descend into nested tuples.
    """
    inner_hidden = torch.randn(4, 8, 512)
    output = ((inner_hidden, "extra"),)

    # output[0] is a tuple, not a tensor, so raises TypeError
    with pytest.raises(TypeError):
        runtime._extract_hidden_from_output(output)
