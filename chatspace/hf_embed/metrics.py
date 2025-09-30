"""Performance metrics and statistics tracking for the embedding pipeline."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineStats:
    """Statistics about processed rows and embeddings."""

    total_rows: int = 0
    skipped_rows: int = 0
    embedding_dim: Optional[int] = None
    min_norm: Optional[float] = None
    max_norm: Optional[float] = None

    def register_rows(self, count: int) -> None:
        """Register successfully processed rows."""
        self.total_rows += count

    def register_skipped(self, count: int = 1) -> None:
        """Register skipped rows."""
        self.skipped_rows += count

    def update_embedding_dim(self, dim: int) -> None:
        """Update embedding dimension, ensuring consistency."""
        if self.embedding_dim is None:
            self.embedding_dim = dim
        elif dim is not None and self.embedding_dim != dim:
            raise ValueError(f"Inconsistent embedding dimension: expected {self.embedding_dim}, received {dim}")

    def update_norm_bounds(self, min_norm: Optional[float], max_norm: Optional[float]) -> None:
        """Update min/max norm bounds."""
        if min_norm is None or max_norm is None:
            return
        self.min_norm = min_norm if self.min_norm is None else min(self.min_norm, min_norm)
        self.max_norm = max_norm if self.max_norm is None else max(self.max_norm, max_norm)


@dataclass
class StageTimings:
    """Process-safe timing statistics for a pipeline stage."""

    busy_seconds: float = 0.0
    idle_seconds: float = 0.0
    _lock: mp.Lock = field(default_factory=mp.Lock, init=False, repr=False)

    def add_busy(self, delta: float) -> None:
        """Add busy time (doing work)."""
        if delta <= 0:
            return
        with self._lock:
            self.busy_seconds += delta

    def add_idle(self, delta: float) -> None:
        """Add idle time (waiting)."""
        if delta <= 0:
            return
        with self._lock:
            self.idle_seconds += delta

    def to_dict(self, total_duration: float) -> dict[str, Any]:
        """Convert to dictionary with computed utilization metrics."""
        total_stage = self.busy_seconds + self.idle_seconds
        utilization = (self.busy_seconds / total_duration) if total_duration > 0 else None
        stage_busy_fraction = (self.busy_seconds / total_stage) if total_stage > 0 else None
        return {
            "busy_seconds": self.busy_seconds,
            "idle_seconds": self.idle_seconds,
            "busy_fraction_of_stage": stage_busy_fraction,
            "busy_fraction_of_run": utilization,
        }


@dataclass
class PipelineMetrics:
    """Aggregated metrics for all pipeline stages."""

    loader: StageTimings = field(default_factory=StageTimings)
    encoder: StageTimings = field(default_factory=StageTimings)
    writer: StageTimings = field(default_factory=StageTimings)
    _lock: mp.Lock = field(default_factory=mp.Lock, init=False, repr=False)
    encoder_encode_seconds: float = 0.0

    def add_encoder_encode(self, delta: float) -> None:
        """Add time spent in model.encode() calls."""
        if delta <= 0:
            return
        with self._lock:
            self.encoder_encode_seconds += delta

    def to_dict(self, total_duration: float) -> dict[str, Any]:
        """Convert to dictionary with all stage metrics."""
        return {
            "loader": self.loader.to_dict(total_duration),
            "encoder": {
                **self.encoder.to_dict(total_duration),
                "encode_call_seconds": self.encoder_encode_seconds,
            },
            "writer": self.writer.to_dict(total_duration),
        }