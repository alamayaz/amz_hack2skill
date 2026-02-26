"""Tests for file validators."""

import subprocess
from unittest.mock import patch

import pytest

from rezaa.extractors.validators import (
    validate_file_format,
    validate_file_integrity,
    validate_file_size,
)
from rezaa.models.errors import ValidationError


class TestValidateFileFormat:
    def test_valid_audio_format(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.touch()
        validate_file_format(f, ["mp3", "wav", "aac", "m4a"])

    def test_valid_video_format(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.touch()
        validate_file_format(f, ["mp4", "mov", "avi", "webm"])

    def test_invalid_format(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.touch()
        with pytest.raises(ValidationError, match="Unsupported format"):
            validate_file_format(f, ["mp4", "mov"])

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "test.MP4"
        f.touch()
        validate_file_format(f, ["mp4"])


class TestValidateFileSize:
    def test_within_limit(self, tmp_path):
        f = tmp_path / "small.mp4"
        f.write_bytes(b"x" * 1024)
        validate_file_size(f, max_size_mb=1)

    def test_exceeds_limit(self, tmp_path):
        f = tmp_path / "big.mp4"
        f.write_bytes(b"x" * (2 * 1024 * 1024))
        with pytest.raises(ValidationError, match="too large"):
            validate_file_size(f, max_size_mb=1)

    def test_file_not_found(self, tmp_path):
        f = tmp_path / "missing.mp4"
        with pytest.raises(ValidationError, match="not found"):
            validate_file_size(f)


class TestValidateFileIntegrity:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_file_integrity(tmp_path / "missing.mp4")

    def test_ffprobe_not_found(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fake")
        with patch("rezaa.extractors.validators.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ValidationError, match="ffprobe not found"):
                validate_file_integrity(f)

    def test_corrupt_file(self, tmp_path):
        f = tmp_path / "corrupt.mp4"
        f.write_bytes(b"not a real video")
        with patch("rezaa.extractors.validators.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="Invalid data"
            )
            with pytest.raises(ValidationError, match="corrupted"):
                validate_file_integrity(f)

    def test_valid_file_with_streams(self, tmp_path):
        f = tmp_path / "valid.mp4"
        f.write_bytes(b"data")
        import json

        probe_output = json.dumps(
            {
                "streams": [{"codec_type": "video"}],
                "format": {"duration": "10.0"},
            }
        )
        with patch("rezaa.extractors.validators.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=probe_output, stderr=""
            )
            result = validate_file_integrity(f)
            assert "streams" in result
