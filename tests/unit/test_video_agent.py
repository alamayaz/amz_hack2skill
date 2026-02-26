"""Tests for VideoUnderstandingAgent."""

import cv2
import numpy as np
import pytest

from rezaa.agents.video_agent import VideoUnderstandingAgent
from rezaa.models.video import VideoAnalysisOutput


def _make_video(path, frames=60, motion=False):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (160, 120))
    for i in range(frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if motion:
            x = int((i / frames) * 120)
            cv2.rectangle(frame, (x, 30), (x + 40, 90), (0, 255, 0), -1)
        else:
            cv2.rectangle(frame, (50, 30), (110, 90), (100, 100, 100), -1)
        writer.write(frame)
    writer.release()
    return path


class TestVideoUnderstandingAgent:
    @pytest.fixture
    def agent(self):
        return VideoUnderstandingAgent()

    @pytest.fixture
    def static_video(self, tmp_path):
        return _make_video(tmp_path / "static.mp4", motion=False)

    @pytest.fixture
    def motion_video(self, tmp_path):
        return _make_video(tmp_path / "motion.mp4", motion=True)

    def test_analyze_returns_output(self, agent, static_video):
        result = agent.analyze(static_video, "test_clip")
        assert isinstance(result, VideoAnalysisOutput)
        assert result.clip_id == "test_clip"
        assert 0 <= result.features.motion_score <= 1
        assert 0 <= result.features.energy_score <= 1

    def test_temporal_consistency(self, agent, static_video):
        result = agent.analyze(static_video, "test")
        assert 0 <= result.temporal_consistency <= 1

    def test_recommended_usage(self, agent, static_video):
        result = agent.analyze(static_video, "test")
        assert result.recommended_usage in [
            "high_energy",
            "medium_energy",
            "low_energy",
            "static_background",
        ]

    def test_output_serialization(self, agent, static_video):
        result = agent.analyze(static_video, "test")
        json_str = agent.to_json(result)
        restored = VideoAnalysisOutput.model_validate_json(json_str)
        assert restored.clip_id == result.clip_id

    def test_validate_output(self, agent, static_video):
        result = agent.analyze(static_video, "test")
        assert agent.validate_output(result) is True
