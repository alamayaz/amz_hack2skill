"""Render result data models."""

from pathlib import Path

from pydantic import BaseModel, Field


class RenderResult(BaseModel):
    """Result of a rendering operation."""

    output_path: str = Field(..., description="Path to rendered output file")
    duration: float = Field(..., ge=0, description="Output duration in seconds")
    file_size_bytes: int = Field(..., ge=0)
    video_codec: str = Field(default="h264")
    audio_codec: str = Field(default="aac")
    container_format: str = Field(default="mp4")
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: float = Field(default=30.0, gt=0)

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)

    @property
    def output_file(self) -> Path:
        return Path(self.output_path)
