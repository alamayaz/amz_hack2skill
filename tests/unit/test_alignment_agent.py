"""Tests for BeatClipAlignmentAgent."""

import pytest

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.models.alignment import AlignmentOutput


class TestBeatClipAlignmentAgent:
    @pytest.fixture
    def agent(self):
        return BeatClipAlignmentAgent(sigma=0.3, min_reuse_gap=5.0)

    def test_align_returns_output(self, agent, sample_audio_analysis, multiple_video_analyses):
        result = agent.align(sample_audio_analysis, multiple_video_analyses)
        assert isinstance(result, AlignmentOutput)
        assert result.total_duration > 0
        assert 0 <= result.coverage <= 1
        assert 0 <= result.average_energy_match <= 1

    def test_placements_sorted(self, agent, sample_audio_analysis, multiple_video_analyses):
        result = agent.align(sample_audio_analysis, multiple_video_analyses)
        for i in range(1, len(result.placements)):
            assert result.placements[i].align_to_beat >= result.placements[i - 1].align_to_beat

    def test_placements_match_beats(self, agent, sample_audio_analysis, multiple_video_analyses):
        result = agent.align(sample_audio_analysis, multiple_video_analyses)
        beats = set(sample_audio_analysis.features.beat_timestamps)
        for p in result.placements:
            assert p.align_to_beat in beats

    def test_energy_correspondence(self, agent, sample_audio_analysis, multiple_video_analyses):
        """High-energy beats should get high-energy clips (on average)."""
        result = agent.align(sample_audio_analysis, multiple_video_analyses)
        if not result.placements:
            return
        # At least some placements should have decent energy match
        avg_match = sum(p.energy_match_score for p in result.placements) / len(result.placements)
        assert avg_match > 0.3, f"Average energy match too low: {avg_match}"

    def test_trim_bounds_valid(self, agent, sample_audio_analysis, multiple_video_analyses):
        result = agent.align(sample_audio_analysis, multiple_video_analyses)
        for p in result.placements:
            assert p.trim_start >= 0
            assert p.trim_end > p.trim_start

    def test_no_empty_clips(self, agent, sample_audio_analysis, multiple_video_analyses):
        result = agent.align(sample_audio_analysis, multiple_video_analyses)
        assert len(result.clips_used) > 0

    def test_calculate_energy_match(self, agent):
        # Same energy = perfect match
        assert agent.calculate_energy_match(0.5, 0.5) == 1.0
        # Different energy = lower match
        score = agent.calculate_energy_match(0.9, 0.1)
        assert score < 0.5

    def test_optimize_clip_duration(self, agent):
        # High energy = shorter duration
        high = agent.optimize_clip_duration(0.9, 120.0)
        low = agent.optimize_clip_duration(0.1, 120.0)
        assert high < low

    def test_empty_video_analyses(self, agent, sample_audio_analysis):
        result = agent.align(sample_audio_analysis, [])
        assert result.coverage == 0.0
        assert len(result.placements) == 0

    def test_serialization(self, agent, sample_audio_analysis, multiple_video_analyses):
        result = agent.align(sample_audio_analysis, multiple_video_analyses)
        json_str = agent.to_json(result)
        restored = AlignmentOutput.model_validate_json(json_str)
        assert len(restored.placements) == len(result.placements)
