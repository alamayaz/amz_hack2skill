"""User preferences data models."""

from typing import Literal

from pydantic import BaseModel, Field

# All concrete transition types (no meta-types like ai_mix)
TRANSITION_TYPES = (
    "cut",
    "fade",
    "crossfade",
    "wipeleft",
    "wiperight",
    "slideleft",
    "slideright",
    "fadeblack",
    "fadewhite",
    "dissolve",
    "zoomin",
    "circleopen",
    "radial",
)

# Transitions that use FFmpeg xfade (everything except cut)
XFADE_TRANSITIONS = tuple(t for t in TRANSITION_TYPES if t != "cut")


class UserPreferences(BaseModel):
    """User-specified editing preferences."""

    pacing: Literal["fast", "medium", "slow"] = Field(default="medium", description="Editing pace")
    style: Literal["dramatic", "smooth", "energetic"] = Field(
        default="smooth", description="Editing style"
    )
    target_duration: float | None = Field(
        default=None, ge=5.0, le=600.0, description="Target output duration in seconds"
    )
    transition_type: Literal[
        "cut", "fade", "crossfade",
        "wipeleft", "wiperight", "slideleft", "slideright",
        "fadeblack", "fadewhite", "dissolve", "zoomin",
        "circleopen", "radial", "ai_mix",
    ] = Field(default="cut", description="Preferred transition type")
    max_clip_reuse: int = Field(
        default=3, ge=1, le=10, description="Max times a single clip can be reused"
    )
    audio_start: float | None = Field(
        default=None, ge=0.0, description="Manual audio start offset in seconds"
    )
