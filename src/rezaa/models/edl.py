"""Edit Decision List data models."""

from pydantic import BaseModel, Field, field_validator, model_validator


class ClipDecision(BaseModel):
    """A single clip edit decision in the final EDL."""

    clip_id: str = Field(..., min_length=1)
    source_start: float = Field(..., ge=0, description="Start time in source clip")
    source_end: float = Field(..., ge=0, description="End time in source clip")
    timeline_start: float = Field(..., ge=0, description="Start time in output timeline")
    timeline_end: float = Field(..., ge=0, description="End time in output timeline")
    transition_type: str = Field(default="cut", description="Transition: cut, fade, crossfade")
    transition_duration: float = Field(default=0.0, ge=0, description="Transition duration")
    energy_match_score: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_time_ranges(self) -> "ClipDecision":
        if self.source_start >= self.source_end:
            raise ValueError(
                f"source_start ({self.source_start}) must be < source_end ({self.source_end})"
            )
        if self.timeline_start >= self.timeline_end:
            raise ValueError(
                f"timeline_start ({self.timeline_start}) must be < "
                f"timeline_end ({self.timeline_end})"
            )
        source_dur = self.source_end - self.source_start
        timeline_dur = self.timeline_end - self.timeline_start
        if abs(source_dur - timeline_dur) > 0.1:
            raise ValueError(
                f"Source duration ({source_dur:.3f}) and timeline duration ({timeline_dur:.3f}) "
                f"must match within 0.1s"
            )
        return self


class AudioDecision(BaseModel):
    """Audio track decision for the final output."""

    source_path: str = Field(default="", description="Path to audio source")
    trim_start: float = Field(default=0.0, ge=0)
    trim_end: float = Field(default=0.0, ge=0)
    fade_in: float = Field(default=0.0, ge=0)
    fade_out: float = Field(default=0.0, ge=0)
    volume: float = Field(default=1.0, ge=0, le=2.0)


class EditDecisionList(BaseModel):
    """Complete edit decision list for rendering."""

    clip_decisions: list[ClipDecision] = Field(default_factory=list)
    audio_decision: AudioDecision = Field(default_factory=AudioDecision)
    total_duration: float = Field(..., ge=0)
    target_fps: float = Field(default=30.0, gt=0)
    target_width: int = Field(default=1920, gt=0)
    target_height: int = Field(default=1080, gt=0)
    edl_metadata: dict = Field(default_factory=dict)

    @field_validator("clip_decisions")
    @classmethod
    def validate_no_overlaps(cls, v: list[ClipDecision]) -> list[ClipDecision]:
        sorted_clips = sorted(v, key=lambda c: c.timeline_start)
        for i in range(1, len(sorted_clips)):
            if sorted_clips[i].timeline_start < sorted_clips[i - 1].timeline_end - 0.01:
                raise ValueError(
                    f"Clip decisions must not overlap: clip at {sorted_clips[i].timeline_start} "
                    f"overlaps with previous ending at {sorted_clips[i - 1].timeline_end}"
                )
        return v

    @field_validator("clip_decisions")
    @classmethod
    def validate_no_consecutive_same_clip(cls, v: list[ClipDecision]) -> list[ClipDecision]:
        sorted_clips = sorted(v, key=lambda c: c.timeline_start)
        for i in range(1, len(sorted_clips)):
            if sorted_clips[i].clip_id == sorted_clips[i - 1].clip_id:
                raise ValueError(
                    f"Consecutive clips must not be the same: '{sorted_clips[i].clip_id}' "
                    f"appears at positions {i - 1} and {i}"
                )
        return v
