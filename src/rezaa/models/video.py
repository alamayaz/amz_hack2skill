"""Video data models."""

from pydantic import BaseModel, Field, field_validator, model_validator


class Segment(BaseModel):
    """A usable segment within a video clip."""

    start: float = Field(..., ge=0, description="Segment start time in seconds")
    end: float = Field(..., ge=0, description="Segment end time in seconds")
    energy_score: float = Field(..., ge=0, le=1, description="Energy score for this segment")

    @model_validator(mode="after")
    def validate_start_before_end(self) -> "Segment":
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be < end ({self.end})")
        return self


class VideoFeatures(BaseModel):
    """Raw extracted video features for a single clip."""

    clip_id: str = Field(..., min_length=1)
    motion_score: float = Field(..., ge=0, le=1, description="Overall motion score")
    scene_changes: list[tuple[float, float]] = Field(
        default_factory=list, description="(timestamp, magnitude) pairs"
    )
    energy_score: float = Field(..., ge=0, le=1, description="Overall energy score")
    best_segments: list[Segment] = Field(default_factory=list)
    duration: float = Field(..., ge=0, description="Clip duration in seconds")
    fps: float = Field(default=30.0, gt=0)
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)

    @field_validator("scene_changes")
    @classmethod
    def validate_scene_changes(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for ts, mag in v:
            if ts < 0:
                raise ValueError(f"Scene change timestamp must be non-negative, got {ts}")
            if not 0 <= mag <= 1:
                raise ValueError(f"Scene change magnitude must be in [0, 1], got {mag}")
        return v

    @field_validator("best_segments")
    @classmethod
    def validate_non_overlapping_segments(cls, v: list[Segment]) -> list[Segment]:
        sorted_segs = sorted(v, key=lambda s: s.start)
        for i in range(1, len(sorted_segs)):
            if sorted_segs[i].start < sorted_segs[i - 1].end:
                raise ValueError("Segments must not overlap")
        return v


class VideoAnalysisOutput(BaseModel):
    """Output from the VideoUnderstandingAgent."""

    clip_id: str = Field(..., min_length=1)
    features: VideoFeatures
    temporal_consistency: float = Field(
        default=1.0, ge=0, le=1, description="How consistent the clip is over time"
    )
    recommended_usage: str = Field(default="general", description="Recommended usage context")
    analysis_metadata: dict = Field(default_factory=dict)
