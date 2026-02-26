"""Property-based tests for validation (Properties 1-3, 34)."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rezaa.extractors.validators import validate_file_format
from rezaa.models.errors import ValidationError

pytestmark = pytest.mark.property


class TestValidationProperties:
    @given(ext=st.sampled_from(["mp4", "mov", "avi", "webm"]))
    @settings(max_examples=20)
    def test_property_1_valid_video_formats(self, ext):
        """Property 1: Valid video formats are accepted."""
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / f"test.{ext}"
            f.touch()
            validate_file_format(f, ["mp4", "mov", "avi", "webm"])

    @given(ext=st.sampled_from(["mp3", "wav", "aac", "m4a"]))
    @settings(max_examples=20)
    def test_property_1_valid_audio_formats(self, ext):
        """Property 1: Valid audio formats are accepted."""
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / f"test.{ext}"
            f.touch()
            validate_file_format(f, ["mp3", "wav", "aac", "m4a"])

    @given(
        ext=st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz").filter(
            lambda x: x not in ["mp4", "mov", "avi", "webm", "mp3", "wav", "aac", "m4a"]
        )
    )
    @settings(max_examples=50)
    def test_property_1_invalid_formats_rejected(self, ext):
        """Property 1: Invalid formats are rejected."""
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / f"test.{ext}"
            f.touch()
            with pytest.raises(ValidationError):
                validate_file_format(f, ["mp4", "mov", "avi", "webm"])
