"""Tests for chatspace.hf_embed.metrics module."""

import threading

import pytest

from chatspace.hf_embed.metrics import PipelineStats, StageTimings, PipelineMetrics


def test_pipeline_stats_initialization():
    """Test PipelineStats initial state."""
    stats = PipelineStats()
    assert stats.total_rows == 0
    assert stats.skipped_rows == 0
    assert stats.embedding_dim is None
    assert stats.min_norm is None
    assert stats.max_norm is None


def test_pipeline_stats_register_rows():
    """Test row registration."""
    stats = PipelineStats()
    stats.register_rows(10)
    assert stats.total_rows == 10
    stats.register_rows(5)
    assert stats.total_rows == 15


def test_pipeline_stats_register_skipped():
    """Test skipped row registration."""
    stats = PipelineStats()
    stats.register_skipped()
    assert stats.skipped_rows == 1
    stats.register_skipped(5)
    assert stats.skipped_rows == 6


def test_pipeline_stats_update_embedding_dim():
    """Test embedding dimension updates."""
    stats = PipelineStats()

    # First update sets the dimension
    stats.update_embedding_dim(768)
    assert stats.embedding_dim == 768

    # Same dimension is ok
    stats.update_embedding_dim(768)
    assert stats.embedding_dim == 768

    # Different dimension raises error
    with pytest.raises(ValueError, match="Inconsistent embedding dimension"):
        stats.update_embedding_dim(1024)


def test_pipeline_stats_update_norm_bounds():
    """Test norm bound updates."""
    stats = PipelineStats()

    # First update
    stats.update_norm_bounds(0.5, 2.0)
    assert stats.min_norm == 0.5
    assert stats.max_norm == 2.0

    # Updates should track min/max
    stats.update_norm_bounds(0.3, 1.5)
    assert stats.min_norm == 0.3
    assert stats.max_norm == 2.0

    stats.update_norm_bounds(0.8, 3.0)
    assert stats.min_norm == 0.3
    assert stats.max_norm == 3.0

    # None values should be ignored
    stats.update_norm_bounds(None, None)
    assert stats.min_norm == 0.3
    assert stats.max_norm == 3.0


def test_stage_timings_initialization():
    """Test StageTimings initial state."""
    timings = StageTimings()
    assert timings.busy_seconds == 0.0
    assert timings.idle_seconds == 0.0


def test_stage_timings_add_busy():
    """Test busy time tracking."""
    timings = StageTimings()
    timings.add_busy(1.5)
    assert timings.busy_seconds == 1.5
    timings.add_busy(0.5)
    assert timings.busy_seconds == 2.0

    # Negative/zero values should be ignored
    timings.add_busy(0)
    timings.add_busy(-1.0)
    assert timings.busy_seconds == 2.0


def test_stage_timings_add_idle():
    """Test idle time tracking."""
    timings = StageTimings()
    timings.add_idle(0.5)
    assert timings.idle_seconds == 0.5
    timings.add_idle(1.0)
    assert timings.idle_seconds == 1.5

    # Negative/zero values should be ignored
    timings.add_idle(0)
    timings.add_idle(-0.5)
    assert timings.idle_seconds == 1.5


def test_stage_timings_to_dict():
    """Test conversion to dictionary."""
    timings = StageTimings()
    timings.add_busy(3.0)
    timings.add_idle(1.0)

    result = timings.to_dict(total_duration=5.0)

    assert result["busy_seconds"] == 3.0
    assert result["idle_seconds"] == 1.0
    assert result["busy_fraction_of_stage"] == 0.75  # 3/(3+1)
    assert result["busy_fraction_of_run"] == 0.6  # 3/5


def test_stage_timings_thread_safety():
    """Test that StageTimings is thread-safe."""
    timings = StageTimings()
    iterations = 100

    def add_busy():
        for _ in range(iterations):
            timings.add_busy(0.01)

    def add_idle():
        for _ in range(iterations):
            timings.add_idle(0.01)

    threads = [
        threading.Thread(target=add_busy),
        threading.Thread(target=add_busy),
        threading.Thread(target=add_idle),
        threading.Thread(target=add_idle),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have accumulated correctly despite concurrency
    assert timings.busy_seconds == pytest.approx(2.0, rel=1e-10)
    assert timings.idle_seconds == pytest.approx(2.0, rel=1e-10)


def test_pipeline_metrics_initialization():
    """Test PipelineMetrics initial state."""
    metrics = PipelineMetrics()
    assert isinstance(metrics.loader, StageTimings)
    assert isinstance(metrics.encoder, StageTimings)
    assert isinstance(metrics.writer, StageTimings)
    assert metrics.encoder_encode_seconds == 0.0


def test_pipeline_metrics_add_encoder_encode():
    """Test encoder timing tracking."""
    metrics = PipelineMetrics()
    metrics.add_encoder_encode(1.5)
    assert metrics.encoder_encode_seconds == 1.5
    metrics.add_encoder_encode(0.5)
    assert metrics.encoder_encode_seconds == 2.0

    # Negative/zero values should be ignored
    metrics.add_encoder_encode(0)
    metrics.add_encoder_encode(-1.0)
    assert metrics.encoder_encode_seconds == 2.0


def test_pipeline_metrics_to_dict():
    """Test conversion to dictionary."""
    metrics = PipelineMetrics()
    metrics.loader.add_busy(1.0)
    metrics.encoder.add_busy(2.0)
    metrics.add_encoder_encode(1.5)
    metrics.writer.add_busy(0.5)

    result = metrics.to_dict(total_duration=5.0)

    assert "loader" in result
    assert "encoder" in result
    assert "writer" in result
    assert result["encoder"]["encode_call_seconds"] == 1.5