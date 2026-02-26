"""Tests for AudioAnalysisAgent."""

import pytest

from rezaa.agents.audio_agent import AudioAnalysisAgent
from rezaa.models.audio import AudioAnalysisOutput
from tests.conftest import generate_click_track_wav


class TestAudioAnalysisAgent:
    @pytest.fixture
    def agent(self):
        return AudioAnalysisAgent()

    @pytest.fixture
    def click_audio(self, tmp_path):
        return generate_click_track_wav(tmp_path / "click.wav", bpm=120.0, duration=5.0)

    def test_analyze_returns_output(self, agent, click_audio):
        result = agent.analyze(click_audio)
        assert isinstance(result, AudioAnalysisOutput)
        assert result.features.bpm >= 30
        assert result.features.bpm <= 300
        assert result.confidence > 0
        assert result.confidence <= 1

    def test_beat_strength_matches_beats(self, agent, click_audio):
        result = agent.analyze(click_audio)
        assert len(result.beat_strength) == len(result.features.beat_timestamps)

    def test_beat_strength_normalized(self, agent, click_audio):
        result = agent.analyze(click_audio)
        for s in result.beat_strength:
            assert 0 <= s <= 1

    def test_refine_beats_monotonic(self, agent, click_audio):
        result = agent.analyze(click_audio)
        for i in range(1, len(result.features.beat_timestamps)):
            assert result.features.beat_timestamps[i] > result.features.beat_timestamps[i - 1]

    def test_output_serialization(self, agent, click_audio):
        result = agent.analyze(click_audio)
        json_str = agent.to_json(result)
        assert json_str
        restored = AudioAnalysisOutput.model_validate_json(json_str)
        assert restored.features.bpm == result.features.bpm

    def test_validate_output(self, agent, click_audio):
        result = agent.analyze(click_audio)
        assert agent.validate_output(result) is True
