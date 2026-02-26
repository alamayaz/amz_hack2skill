"""Property-based tests for video models (Properties 9-12)."""

import pytest
from hypothesis import given, settings

from tests.property.conftest import generate_video_analysis_output, generate_video_features

pytestmark = pytest.mark.property


class TestVideoProperties:
    @given(features=generate_video_features())
    @settings(max_examples=100)
    def test_property_9_motion_normalization(self, features):
        """Property 9: Motion score is normalized to [0, 1]."""
        assert 0 <= features.motion_score <= 1

    @given(features=generate_video_features())
    @settings(max_examples=100)
    def test_property_10_scene_change_validity(self, features):
        """Property 10: Scene changes have valid timestamps and magnitudes."""
        for ts, mag in features.scene_changes:
            assert ts >= 0
            assert 0 <= mag <= 1

    @given(features=generate_video_features())
    @settings(max_examples=100)
    def test_property_11_energy_normalization(self, features):
        """Property 11: Energy score is normalized to [0, 1]."""
        assert 0 <= features.energy_score <= 1

    @given(features=generate_video_features())
    @settings(max_examples=100)
    def test_property_12_segment_validity(self, features):
        """Property 12: Segments are valid and non-overlapping."""
        for seg in features.best_segments:
            assert seg.start >= 0
            assert seg.end > seg.start
            assert 0 <= seg.energy_score <= 1

        sorted_segs = sorted(features.best_segments, key=lambda s: s.start)
        for i in range(1, len(sorted_segs)):
            assert sorted_segs[i].start >= sorted_segs[i - 1].end

    @given(output=generate_video_analysis_output())
    @settings(max_examples=100)
    def test_temporal_consistency_range(self, output):
        """Temporal consistency is in [0, 1]."""
        assert 0 <= output.temporal_consistency <= 1

    @given(output=generate_video_analysis_output())
    @settings(max_examples=100)
    def test_serialization_roundtrip(self, output):
        """Video analysis survives JSON serialization roundtrip."""
        from rezaa.models.video import VideoAnalysisOutput

        json_str = output.model_dump_json()
        restored = VideoAnalysisOutput.model_validate_json(json_str)
        assert restored.clip_id == output.clip_id
        assert restored.features.motion_score == output.features.motion_score
