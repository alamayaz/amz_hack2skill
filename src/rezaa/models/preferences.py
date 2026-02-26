"""User preferences data models."""

from typing import Literal

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """User-specified editing preferences."""

    pacing: Literal["fast", "medium", "slow"] = Field(default="medium", description="Editing pace")
    style: Literal["dramatic", "smooth", "energetic"] = Field(
        default="smooth", description="Editing style"
    )
    target_duration: float | None = Field(
        default=None, ge=5.0, le=600.0, description="Target output duration in seconds"
    )
    transition_type: Literal["cut", "fade", "crossfade"] = Field(
        default="cut", description="Preferred transition type"
    )
    max_clip_reuse: int = Field(
        default=3, ge=1, le=10, description="Max times a single clip can be reused"
    )
