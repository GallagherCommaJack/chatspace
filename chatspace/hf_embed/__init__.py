"""HuggingFace dataset embedding pipeline using SentenceTransformer models."""

from __future__ import annotations

from .config import SentenceTransformerConfig
from .pipeline import run_sentence_transformer

__all__ = ["SentenceTransformerConfig", "run_sentence_transformer"]