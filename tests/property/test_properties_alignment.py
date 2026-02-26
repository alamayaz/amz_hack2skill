"""Property-based tests for alignment (Properties 13-16)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from tests.property.conftest import generate_audio_analysis_output, generate_video_analysis_output

pytestmark = pytest.mark.property


@st.composite
def generate_alignment_inputs(draw):
    """Generate valid inputs for alignment."""
    audio = draw(generate_audio_analysis_output())
    n_clips = draw(st.integers(min_value=1, max_value=5))
    videos = [draw(generate_video_analysis_output()) for _ in range(n_clips)]
    # Ensure unique clip IDs
    for i, v in enumerate(videos):
        v.clip_id = f"clip_{i:03d}"
        v.features.clip_id = f"clip_{i:03d}"
    return audio, videos


class TestAlignmentProperties:
    @given(inputs=generate_alignment_inputs())
    @settings(max_examples=50, deadline=10000)
    def test_property_13_energy_correspondence(self, inputs):
        """Property 13: Energy match scores are in [0, 1]."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        result = agent.align(audio, videos)
        for p in result.placements:
            assert 0 <= p.energy_match_score <= 1

    @given(inputs=generate_alignment_inputs())
    @settings(max_examples=50, deadline=10000)
    def test_property_14_beat_alignment(self, inputs):
        """Property 14: Every placement align_to_beat matches a beat timestamp."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        result = agent.align(audio, videos)
        beats = set(audio.features.beat_timestamps)
        for p in result.placements:
            assert p.align_to_beat in beats

    @given(inputs=generate_alignment_inputs())
    @settings(max_examples=50, deadline=10000)
    def test_property_15_trim_bounds_validity(self, inputs):
        """Property 15: Trim bounds are non-negative and start < end."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        result = agent.align(audio, videos)
        for p in result.placements:
            assert p.trim_start >= 0
            assert p.trim_end > p.trim_start

    @given(inputs=generate_alignment_inputs())
    @settings(max_examples=50, deadline=10000)
    def test_property_16_timeline_coverage(self, inputs):
        """Property 16: Coverage is in [0, 1]."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        result = agent.align(audio, videos)
        assert 0 <= result.coverage <= 1

    @given(inputs=generate_alignment_inputs())
    @settings(max_examples=50, deadline=10000)
    def test_placements_sorted_by_beat(self, inputs):
        """Placements are sorted by align_to_beat."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        result = agent.align(audio, videos)
        for i in range(1, len(result.placements)):
            assert result.placements[i].align_to_beat >= result.placements[i - 1].align_to_beat

    @given(inputs=generate_alignment_inputs())
    @settings(max_examples=50, deadline=10000)
    def test_serialization_roundtrip(self, inputs):
        """Alignment output survives JSON roundtrip."""
        from rezaa.models.alignment import AlignmentOutput

        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        result = agent.align(audio, videos)
        json_str = result.model_dump_json()
        restored = AlignmentOutput.model_validate_json(json_str)
        assert len(restored.placements) == len(result.placements)
