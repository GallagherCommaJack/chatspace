"""Project-wide interpreter customization for vLLM steering patches.

Any Python process launched with this repository on its PYTHONPATH will import
this module during interpreter startup, ensuring Qwen decoder layers are
patched before vLLM performs CUDA-graph capture.
"""

try:
    from chatspace.vllm_steering.runtime import (
        ensure_collective_rpc_gateway_installed,
        ensure_layer_patch_installed,
    )
except Exception:
    # vLLM may not be available in lightweight tooling contexts.
    ensure_layer_patch_installed = None  # type: ignore[assignment]
    ensure_collective_rpc_gateway_installed = None  # type: ignore[assignment]

if ensure_layer_patch_installed is not None:
    try:
        ensure_layer_patch_installed()
    except Exception:
        # Avoid breaking unrelated tooling if patch installation fails early.
        pass

if ensure_collective_rpc_gateway_installed is not None:
    try:
        ensure_collective_rpc_gateway_installed()
    except Exception:
        pass
