"""Alignment data models."""

from pydantic import BaseModel, Field, field_validator, model_validator


class ClipPlacement(BaseModel):
    """A single clip placement aligned to a beat."""

    clip_id: str = Field(..., min_length=1)
    align_to_beat: float = Field(..., ge=0, description="Beat timestamp to align to")
    trim_start: float = Field(..., ge=0, description="Start trim point within clip")
    trim_end: float = Field(..., ge=0, description="End trim point within clip")
    energy_match_score: float = Field(..., ge=0, le=1, description="Energy match quality")

    @model_validator(mode="after")
    def validate_trim_bounds(self) -> "ClipPlacement":
        if self.trim_start >= self.trim_end:
            raise ValueError(f"trim_start ({self.trim_start}) must be < trim_end ({self.trim_end})")
        return self


class AlignmentOutput(BaseModel):
    """Output from the BeatClipAlignmentAgent."""

    placements: list[ClipPlacement] = Field(default_factory=list)
    total_duration: float = Field(..., ge=0, description="Total aligned duration")
    coverage: float = Field(..., ge=0, le=1, description="Fraction of beats covered")
    average_energy_match: float = Field(..., ge=0, le=1)
    clips_used: list[str] = Field(default_factory=list, description="Unique clip IDs used")
    alignment_metadata: dict = Field(default_factory=dict)

    @field_validator("placements")
    @classmethod
    def validate_placements_sorted(cls, v: list[ClipPlacement]) -> list[ClipPlacement]:
        for i in range(1, len(v)):
            if v[i].align_to_beat < v[i - 1].align_to_beat:
                raise ValueError("Placements must be sorted by align_to_beat")
        return v
