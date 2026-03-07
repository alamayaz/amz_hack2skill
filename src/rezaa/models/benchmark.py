"""Benchmark / performance report models."""

from pydantic import BaseModel, Field


class StageTiming(BaseModel):
    """Timing for a single pipeline stage."""

    stage_name: str
    duration_seconds: float = Field(..., ge=0)


class InputSummary(BaseModel):
    """Summary of pipeline inputs."""

    audio_duration_seconds: float = 0.0
    num_clips: int = 0
    total_video_duration_seconds: float = 0.0


class AudioMetrics(BaseModel):
    """Key metrics from audio analysis."""

    bpm: float = 0.0
    beat_count: int = 0
    confidence: float = 0.0
    beat_strength_avg: float = 0.0
    drop_count: int = 0


class VideoClipMetrics(BaseModel):
    """Per-clip video metrics."""

    clip_id: str
    duration: float = 0.0
    motion_score: float = 0.0
    energy_score: float = 0.0
    temporal_consistency: float = 0.0
    recommended_usage: str = "general"


class AlignmentMetrics(BaseModel):
    """Metrics from beat-clip alignment."""

    coverage: float = 0.0
    average_energy_match: float = 0.0
    clips_used: int = 0
    total_placements: int = 0


class OrchestrationMetrics(BaseModel):
    """Metrics from orchestration/decision stage."""

    method: str = "unknown"
    clip_decisions_count: int = 0
    transition_types_used: list[str] = Field(default_factory=list)
    total_duration: float = 0.0


class RenderMetrics(BaseModel):
    """Metrics from rendering output."""

    output_duration: float = 0.0
    file_size_mb: float = 0.0
    video_codec: str = ""
    resolution: str = ""
    fps: float = 0.0


class BenchmarkReport(BaseModel):
    """Full pipeline benchmark report."""

    job_id: str
    total_pipeline_duration_seconds: float = 0.0
    stage_timings: list[StageTiming] = Field(default_factory=list)
    input_summary: InputSummary = Field(default_factory=InputSummary)
    audio_metrics: AudioMetrics = Field(default_factory=AudioMetrics)
    video_metrics: list[VideoClipMetrics] = Field(default_factory=list)
    alignment_metrics: AlignmentMetrics = Field(default_factory=AlignmentMetrics)
    orchestration_metrics: OrchestrationMetrics = Field(default_factory=OrchestrationMetrics)
    render_metrics: RenderMetrics = Field(default_factory=RenderMetrics)
