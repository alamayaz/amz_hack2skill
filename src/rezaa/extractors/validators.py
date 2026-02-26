"""File format, size, and integrity validation."""

import json
import subprocess
from pathlib import Path

from rezaa.config import get_settings
from rezaa.models.errors import ValidationError


def validate_file_format(file_path: Path, allowed_formats: list[str]) -> None:
    """Validate that file has an allowed extension."""
    ext = file_path.suffix.lstrip(".").lower()
    if ext not in allowed_formats:
        raise ValidationError(
            f"Unsupported format: .{ext}. Allowed: {allowed_formats}",
            details={"extension": ext, "allowed": allowed_formats},
        )


def validate_file_size(file_path: Path, max_size_mb: int | None = None) -> None:
    """Validate that file size is within limits."""
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    max_mb = max_size_mb or get_settings().upload_max_size_mb
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(
            f"File too large: {size_mb:.1f}MB exceeds {max_mb}MB limit",
            details={"size_mb": size_mb, "max_mb": max_mb},
        )


def validate_file_integrity(file_path: Path) -> dict:
    """Validate file integrity using ffprobe. Returns probe data."""
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ValidationError(
                "File appears to be corrupted or unreadable",
                details={"stderr": result.stderr[:500]},
            )
        probe_data = json.loads(result.stdout)
        if not probe_data.get("streams"):
            raise ValidationError(
                "No media streams found in file",
                details={"file": str(file_path)},
            )
        return probe_data
    except FileNotFoundError:
        raise ValidationError(
            "ffprobe not found. Please install FFmpeg.",
            details={"command": "ffprobe"},
        )
    except subprocess.TimeoutExpired:
        raise ValidationError(
            "File probe timed out — file may be corrupted",
            details={"file": str(file_path)},
        )
    except json.JSONDecodeError:
        raise ValidationError(
            "Failed to parse ffprobe output",
            details={"file": str(file_path)},
        )


def validate_audio_upload(file_path: Path) -> dict:
    """Full validation for audio file uploads."""
    settings = get_settings()
    validate_file_format(file_path, settings.allowed_audio_formats)
    validate_file_size(file_path)
    probe_data = validate_file_integrity(file_path)

    # Check for audio stream
    has_audio = any(s.get("codec_type") == "audio" for s in probe_data.get("streams", []))
    if not has_audio:
        raise ValidationError("No audio stream found in file")

    # Check duration
    duration = float(probe_data.get("format", {}).get("duration", 0))
    if duration < settings.min_audio_duration:
        raise ValidationError(
            f"Audio too short: {duration:.1f}s (min {settings.min_audio_duration}s)"
        )
    if duration > settings.max_audio_duration:
        raise ValidationError(
            f"Audio too long: {duration:.1f}s (max {settings.max_audio_duration}s)"
        )
    return probe_data


def validate_video_upload(file_path: Path) -> dict:
    """Full validation for video file uploads."""
    settings = get_settings()
    validate_file_format(file_path, settings.allowed_video_formats)
    validate_file_size(file_path)
    probe_data = validate_file_integrity(file_path)

    # Check for video stream
    has_video = any(s.get("codec_type") == "video" for s in probe_data.get("streams", []))
    if not has_video:
        raise ValidationError("No video stream found in file")

    # Check duration
    duration = float(probe_data.get("format", {}).get("duration", 0))
    if duration < settings.min_video_duration:
        raise ValidationError(
            f"Video too short: {duration:.1f}s (min {settings.min_video_duration}s)"
        )
    if duration > settings.max_video_duration:
        raise ValidationError(
            f"Video too long: {duration:.1f}s (max {settings.max_video_duration}s)"
        )
    return probe_data
