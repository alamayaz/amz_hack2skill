"""Property-based tests for audio models and analysis (Properties 4-8)."""

import pytest
from hypothesis import given, settings

from tests.property.conftest import generate_audio_analysis_output, generate_audio_features

pytestmark = pytest.mark.property


class TestAudioProperties:
    @given(features=generate_audio_features())
    @settings(max_examples=100)
    def test_property_4_beat_timestamps_precision(self, features):
        """Property 4: Beat timestamps are non-negative and monotonically increasing."""
        for i in range(1, len(features.beat_timestamps)):
            assert features.beat_timestamps[i] > features.beat_timestamps[i - 1]
        for t in features.beat_timestamps:
            assert t >= 0

    @given(features=generate_audio_features())
    @settings(max_examples=100)
    def test_property_5_bpm_range(self, features):
        """Property 5: BPM is in [30, 300]."""
        assert 30 <= features.bpm <= 300

    @given(features=generate_audio_features())
    @settings(max_examples=100)
    def test_property_6_energy_normalization(self, features):
        """Property 6: Energy values are in [0, 1]."""
        for ts, energy in features.energy_curve:
            assert 0 <= energy <= 1

    @given(features=generate_audio_features())
    @settings(max_examples=100)
    def test_property_7_drop_timestamps_valid(self, features):
        """Property 7: Drop timestamps are non-negative."""
        for t in features.drop_timestamps:
            assert t >= 0

    @given(output=generate_audio_analysis_output())
    @settings(max_examples=100)
    def test_property_8_confidence_range(self, output):
        """Property 8: Confidence score is in [0, 1]."""
        assert 0 <= output.confidence <= 1

    @given(output=generate_audio_analysis_output())
    @settings(max_examples=100)
    def test_beat_strength_count_matches(self, output):
        """Beat strength list length matches beat timestamps."""
        assert len(output.beat_strength) == len(output.features.beat_timestamps)

    @given(output=generate_audio_analysis_output())
    @settings(max_examples=100)
    def test_serialization_roundtrip(self, output):
        """Audio analysis survives JSON serialization roundtrip."""
        json_str = output.model_dump_json()
        from rezaa.models.audio import AudioAnalysisOutput

        restored = AudioAnalysisOutput.model_validate_json(json_str)
        assert restored.features.bpm == output.features.bpm
        assert len(restored.features.beat_timestamps) == len(output.features.beat_timestamps)
