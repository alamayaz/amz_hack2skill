"""Shared test fixtures and test media generators."""

import struct
import tempfile
import wave
from pathlib import Path

import pytest

from rezaa.models.audio import AudioAnalysisOutput, AudioFeatures
from rezaa.models.video import Segment, VideoAnalysisOutput, VideoFeatures


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_audio_features():
    """Create sample AudioFeatures for testing."""
    return AudioFeatures(
        bpm=120.0,
        beat_timestamps=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        energy_curve=[(0.0, 0.3), (0.5, 0.6), (1.0, 0.8), (1.5, 0.9), (2.0, 0.7), (2.5, 0.5)],
        drop_timestamps=[1.5],
        duration=3.0,
        sample_rate=22050,
    )


@pytest.fixture
def sample_audio_analysis(sample_audio_features):
    """Create sample AudioAnalysisOutput for testing."""
    return AudioAnalysisOutput(
        features=sample_audio_features,
        confidence=0.85,
        beat_strength=[0.7, 0.9, 0.6, 1.0, 0.5, 0.8],
    )


@pytest.fixture
def sample_video_features():
    """Create sample VideoFeatures for testing."""
    return VideoFeatures(
        clip_id="clip_001",
        motion_score=0.6,
        scene_changes=[(1.0, 0.8), (3.0, 0.5)],
        energy_score=0.7,
        best_segments=[
            Segment(start=0.0, end=1.5, energy_score=0.8),
            Segment(start=2.0, end=3.5, energy_score=0.6),
        ],
        duration=5.0,
        fps=30.0,
        width=1920,
        height=1080,
    )


@pytest.fixture
def sample_video_analysis(sample_video_features):
    """Create sample VideoAnalysisOutput for testing."""
    return VideoAnalysisOutput(
        clip_id="clip_001",
        features=sample_video_features,
        temporal_consistency=0.9,
        recommended_usage="high_energy",
    )


@pytest.fixture
def multiple_video_analyses():
    """Create multiple video analyses for alignment testing."""
    clips = []
    for i, (motion, energy) in enumerate([(0.8, 0.9), (0.3, 0.2), (0.6, 0.5)]):
        features = VideoFeatures(
            clip_id=f"clip_{i:03d}",
            motion_score=motion,
            scene_changes=[(1.0, 0.5)],
            energy_score=energy,
            best_segments=[Segment(start=0.0, end=2.0, energy_score=energy)],
            duration=5.0,
        )
        clips.append(
            VideoAnalysisOutput(
                clip_id=f"clip_{i:03d}",
                features=features,
                temporal_consistency=0.8,
            )
        )
    return clips


def generate_test_wav(
    path: Path, duration: float = 1.0, sample_rate: int = 22050, freq: float = 440.0
) -> Path:
    """Generate a simple test WAV file with a sine wave."""
    import math

    n_samples = int(duration * sample_rate)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            sample = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
            wav.writeframes(struct.pack("<h", sample))
    return path


def generate_click_track_wav(
    path: Path, bpm: float = 120.0, duration: float = 3.0, sample_rate: int = 22050
) -> Path:
    """Generate a click track WAV at given BPM."""
    import math

    n_samples = int(duration * sample_rate)
    beat_interval = 60.0 / bpm
    click_duration = 0.01
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            beat_pos = t % beat_interval
            if beat_pos < click_duration:
                sample = int(32767 * 0.8 * math.sin(2 * math.pi * 1000 * t))
            else:
                sample = 0
            wav.writeframes(struct.pack("<h", sample))
    return path


def generate_silent_wav(path: Path, duration: float = 1.0, sample_rate: int = 22050) -> Path:
    """Generate a silent WAV file."""
    n_samples = int(duration * sample_rate)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_samples)
    return path
