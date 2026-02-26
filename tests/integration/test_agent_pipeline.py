"""Integration tests for agent pipeline."""

import cv2
import numpy as np
import pytest

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.agents.audio_agent import AudioAnalysisAgent
from rezaa.agents.video_agent import VideoUnderstandingAgent
from rezaa.orchestrator.decision import DecisionOrchestrator
from tests.conftest import generate_click_track_wav

pytestmark = pytest.mark.integration


def _make_test_video(path, frames=90, motion=True):
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


class TestAgentPipeline:
    @pytest.fixture
    def audio_path(self, tmp_path):
        return generate_click_track_wav(tmp_path / "track.wav", bpm=120.0, duration=5.0)

    @pytest.fixture
    def video_paths(self, tmp_path):
        return [_make_test_video(tmp_path / f"clip_{i}.mp4", motion=(i % 2 == 0)) for i in range(3)]

    def test_full_agent_pipeline(self, audio_path, video_paths):
        """Test the full analysis pipeline: audio → video → alignment → orchestration."""
        # Audio analysis
        audio_agent = AudioAnalysisAgent()
        audio_analysis = audio_agent.analyze(audio_path)
        assert audio_analysis.confidence > 0

        # Video analysis
        video_agent = VideoUnderstandingAgent()
        video_analyses = []
        for i, vpath in enumerate(video_paths):
            va = video_agent.analyze(vpath, f"clip_{i:03d}")
            video_analyses.append(va)
        assert len(video_analyses) == 3

        # Alignment
        alignment_agent = BeatClipAlignmentAgent()
        alignment = alignment_agent.align(audio_analysis, video_analyses)
        assert alignment.coverage > 0 or len(audio_analysis.features.beat_timestamps) == 0

        # Orchestration (fallback mode)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(audio_analysis, video_analyses, alignment)
        assert edl.total_duration > 0 or len(alignment.placements) == 0

        # Verify EDL validity
        for i in range(1, len(edl.clip_decisions)):
            assert edl.clip_decisions[i].clip_id != edl.clip_decisions[i - 1].clip_id
            prev_end = edl.clip_decisions[i - 1].timeline_end
            assert edl.clip_decisions[i].timeline_start >= prev_end - 0.01
