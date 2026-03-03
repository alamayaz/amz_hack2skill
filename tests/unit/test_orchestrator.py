"""Tests for DecisionOrchestrator (mocked LLM)."""

import json

import pytest

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.models.alignment import AlignmentOutput
from rezaa.models.audio import AudioAnalysisOutput, AudioFeatures
from rezaa.models.edl import EditDecisionList
from rezaa.models.preferences import UserPreferences
from rezaa.models.video import Segment, VideoAnalysisOutput, VideoFeatures
from rezaa.orchestrator.decision import DecisionOrchestrator
from rezaa.orchestrator.parser import parse_llm_response, validate_edl


class TestParseResponse:
    def test_parse_plain_json(self):
        data = {"clip_decisions": [], "total_duration": 5.0}
        result = parse_llm_response(json.dumps(data))
        assert result == data

    def test_parse_markdown_wrapped(self):
        data = {"clip_decisions": [], "total_duration": 5.0}
        text = f"```json\n{json.dumps(data)}\n```"
        result = parse_llm_response(text)
        assert result == data

    def test_parse_with_extra_text(self):
        data = {"clip_decisions": [], "total_duration": 5.0}
        text = f"Here is the EDL:\n{json.dumps(data)}\nDone!"
        result = parse_llm_response(text)
        assert result == data


class TestValidateEdl:
    def test_valid_edl(self):
        data = {
            "clip_decisions": [
                {
                    "clip_id": "clip_001",
                    "source_start": 0.0,
                    "source_end": 1.0,
                    "timeline_start": 0.0,
                    "timeline_end": 1.0,
                    "transition_type": "cut",
                },
                {
                    "clip_id": "clip_002",
                    "source_start": 0.0,
                    "source_end": 1.0,
                    "timeline_start": 1.0,
                    "timeline_end": 2.0,
                    "transition_type": "cut",
                },
            ],
            "total_duration": 2.0,
        }
        edl = validate_edl(data, 2.0)
        assert isinstance(edl, EditDecisionList)
        assert len(edl.clip_decisions) == 2

    def test_skips_invalid_decisions(self):
        data = {
            "clip_decisions": [
                {
                    "clip_id": "c1",
                    "source_start": 5.0,
                    "source_end": 1.0,  # Invalid: start > end
                    "timeline_start": 0.0,
                    "timeline_end": 1.0,
                },
                {
                    "clip_id": "c2",
                    "source_start": 0.0,
                    "source_end": 1.0,
                    "timeline_start": 0.0,
                    "timeline_end": 1.0,
                    "transition_type": "cut",
                },
            ],
            "total_duration": 1.0,
        }
        edl = validate_edl(data, 1.0)
        assert len(edl.clip_decisions) == 1


class TestDecisionOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        return DecisionOrchestrator(client=None)

    @pytest.fixture
    def alignment(self, sample_audio_analysis, multiple_video_analyses):
        agent = BeatClipAlignmentAgent()
        return agent.align(sample_audio_analysis, multiple_video_analyses)

    def test_fallback_edl(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        """Test that fallback EDL works without LLM."""
        edl = orchestrator.orchestrate(sample_audio_analysis, multiple_video_analyses, alignment)
        assert isinstance(edl, EditDecisionList)
        assert edl.total_duration > 0

    def test_fallback_no_consecutive_same_clip(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        edl = orchestrator.orchestrate(sample_audio_analysis, multiple_video_analyses, alignment)
        for i in range(1, len(edl.clip_decisions)):
            assert edl.clip_decisions[i].clip_id != edl.clip_decisions[i - 1].clip_id

    def test_apply_fast_pacing(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        prefs = UserPreferences(pacing="fast")
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        for cd in edl.clip_decisions:
            dur = cd.timeline_end - cd.timeline_start
            assert dur <= 1.5 + 0.01, f"Fast pacing clip duration {dur} > 1.5"

    def test_apply_slow_pacing(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        prefs = UserPreferences(pacing="slow")
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        for cd in edl.clip_decisions:
            dur = cd.timeline_end - cd.timeline_start
            assert dur >= 2.0 - 0.01, f"Slow pacing clip duration {dur} < 2.0"

    def test_apply_transition_type(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        prefs = UserPreferences(transition_type="fade")
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        for cd in edl.clip_decisions:
            assert cd.transition_type == "fade"

    def test_apply_target_duration(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        prefs = UserPreferences(target_duration=10.0)
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        if edl.clip_decisions:
            last_end = max(cd.timeline_end for cd in edl.clip_decisions)
            assert last_end <= 10.0 + 0.01

    def test_edl_serialization(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        edl = orchestrator.orchestrate(sample_audio_analysis, multiple_video_analyses, alignment)
        json_str = edl.model_dump_json()
        restored = EditDecisionList.model_validate_json(json_str)
        assert len(restored.clip_decisions) == len(edl.clip_decisions)

    @pytest.fixture
    def long_audio_analysis(self):
        """30-second audio for testing manual offset."""
        features = AudioFeatures(
            bpm=120.0,
            beat_timestamps=[0.5 * i for i in range(1, 61)],
            energy_curve=[(i * 0.5, 0.5) for i in range(61)],
            duration=30.0,
        )
        return AudioAnalysisOutput(features=features, confidence=0.9)

    @pytest.fixture
    def long_audio_videos(self):
        clips = []
        for i in range(2):
            features = VideoFeatures(
                clip_id=f"clip_{i:03d}",
                motion_score=0.6,
                energy_score=0.6,
                best_segments=[Segment(start=0.0, end=5.0, energy_score=0.6)],
                duration=15.0,
            )
            clips.append(VideoAnalysisOutput(
                clip_id=f"clip_{i:03d}",
                features=features,
                temporal_consistency=0.8,
            ))
        return clips

    @pytest.fixture
    def long_audio_alignment(self, long_audio_analysis, long_audio_videos):
        agent = BeatClipAlignmentAgent()
        return agent.align(long_audio_analysis, long_audio_videos)

    def test_manual_audio_start(
        self, orchestrator, long_audio_analysis, long_audio_videos, long_audio_alignment
    ):
        """Manual audio_start should set trim_start to that value."""
        prefs = UserPreferences(audio_start=10.0, target_duration=15.0)
        edl = orchestrator.orchestrate(
            long_audio_analysis, long_audio_videos, long_audio_alignment, prefs
        )
        assert edl.audio_decision.trim_start == 10.0

    def test_manual_audio_start_clamped(
        self, orchestrator, long_audio_analysis, long_audio_videos, long_audio_alignment
    ):
        """audio_start near end should be clamped so the window fits."""
        prefs = UserPreferences(audio_start=100.0, target_duration=15.0)
        edl = orchestrator.orchestrate(
            long_audio_analysis, long_audio_videos, long_audio_alignment, prefs
        )
        # Max offset = 30.0 - 15.0 = 15.0
        assert edl.audio_decision.trim_start == 15.0

    def test_auto_audio_start_uses_energy(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        """Omitting audio_start should use the energy-based auto selection."""
        prefs = UserPreferences()
        assert prefs.audio_start is None
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        # Should still produce a valid EDL with energy-based start
        assert edl.audio_decision.trim_start >= 0.0
