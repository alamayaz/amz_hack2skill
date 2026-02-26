"""Audio data models."""

from pydantic import BaseModel, Field, field_validator


class AudioFeatures(BaseModel):
    """Raw extracted audio features."""

    bpm: float = Field(..., description="Beats per minute")
    beat_timestamps: list[float] = Field(default_factory=list, description="Beat times in seconds")
    energy_curve: list[tuple[float, float]] = Field(
        default_factory=list, description="(timestamp, energy) pairs normalized [0,1]"
    )
    drop_timestamps: list[float] = Field(
        default_factory=list, description="Energy drop times in seconds"
    )
    duration: float = Field(..., ge=0, description="Audio duration in seconds")
    sample_rate: int = Field(default=22050, gt=0)

    @field_validator("bpm")
    @classmethod
    def validate_bpm(cls, v: float) -> float:
        if not 30 <= v <= 300:
            raise ValueError(f"BPM must be in [30, 300], got {v}")
        return v

    @field_validator("beat_timestamps")
    @classmethod
    def validate_beat_timestamps(cls, v: list[float]) -> list[float]:
        for i in range(1, len(v)):
            if v[i] <= v[i - 1]:
                raise ValueError("Beat timestamps must be monotonically increasing")
        for t in v:
            if t < 0:
                raise ValueError(f"Beat timestamp must be non-negative, got {t}")
        return v

    @field_validator("energy_curve")
    @classmethod
    def validate_energy_curve(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for ts, energy in v:
            if not 0 <= energy <= 1:
                raise ValueError(f"Energy must be in [0, 1], got {energy}")
            if ts < 0:
                raise ValueError(f"Timestamp must be non-negative, got {ts}")
        return v

    @field_validator("drop_timestamps")
    @classmethod
    def validate_drop_timestamps(cls, v: list[float]) -> list[float]:
        for t in v:
            if t < 0:
                raise ValueError(f"Drop timestamp must be non-negative, got {t}")
        return v


class AudioAnalysisOutput(BaseModel):
    """Output from the AudioAnalysisAgent."""

    features: AudioFeatures
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence score")
    beat_strength: list[float] = Field(
        default_factory=list, description="Strength of each beat [0,1]"
    )
    analysis_metadata: dict = Field(default_factory=dict)

    @field_validator("beat_strength")
    @classmethod
    def validate_beat_strength(cls, v: list[float]) -> list[float]:
        for s in v:
            if not 0 <= s <= 1:
                raise ValueError(f"Beat strength must be in [0, 1], got {s}")
        return v
