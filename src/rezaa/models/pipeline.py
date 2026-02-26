"""Pipeline state and stage models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class PipelineStage(StrEnum):
    """Stages of the processing pipeline."""

    UPLOAD = "upload"
    VALIDATION = "validation"
    AUDIO_EXTRACTION = "audio_extraction"
    VIDEO_EXTRACTION = "video_extraction"
    AUDIO_ANALYSIS = "audio_analysis"
    VIDEO_ANALYSIS = "video_analysis"
    ALIGNMENT = "alignment"
    ORCHESTRATION = "orchestration"
    RENDERING = "rendering"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineState(BaseModel):
    """Current state of a processing job."""

    job_id: str = Field(..., min_length=1)
    stage: PipelineStage = Field(default=PipelineStage.UPLOAD)
    progress: float = Field(default=0.0, ge=0, le=1)
    message: str = Field(default="")
    started_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    audio_path: str | None = None
    video_paths: list[str] = Field(default_factory=list)
    output_path: str | None = None
