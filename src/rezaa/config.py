"""Application configuration using Pydantic BaseSettings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Rezaa AI configuration loaded from environment variables."""

    model_config = {"env_prefix": "REZAA_", "env_file": ".env", "extra": "ignore"}

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Upload constraints
    upload_max_size_mb: int = 500
    allowed_audio_formats: list[str] = ["mp3", "wav", "aac", "m4a"]
    allowed_video_formats: list[str] = ["mp4", "mov", "avi", "webm"]

    # Duration limits
    min_audio_duration: float = 5.0
    max_audio_duration: float = 600.0
    min_video_duration: float = 0.5
    max_video_duration: float = 300.0

    # Directories
    temp_dir: Path = Path("/tmp/rezaa/temp")
    output_dir: Path = Path("/tmp/rezaa/output")
    feature_store_dir: Path = Path("/tmp/rezaa/features")

    # Processing
    max_concurrent_jobs: int = 4
    temp_file_ttl_seconds: int = 60

    # Rendering
    output_video_codec: str = "libx264"
    output_audio_codec: str = "aac"
    output_format: str = "mp4"
    output_crf: int = 23
    output_preset: str = "medium"


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
