"""Pytest configuration for chatspace test suite."""

from __future__ import annotations

import warnings

# vLLM brings in SWIG-backed helper types that currently emit DeprecationWarnings
# under Python 3.11. Filter them here so GPU-enabled steering tests run cleanly.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy(?:Packed|Object) has no __module__ attribute",
    category=DeprecationWarning,
)
