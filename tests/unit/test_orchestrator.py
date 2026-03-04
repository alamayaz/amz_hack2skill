"""Tests for DecisionOrchestrator (mocked LLM)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.models.alignment import AlignmentOutput
from rezaa.models.audio import AudioAnalysisOutput, AudioFeatures
from rezaa.models.edl import AudioDecision, ClipDecision, EditDecisionList
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

    def test_ai_mix_fallback_cycles_transitions(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        """ai_mix fallback should cycle through XFADE_TRANSITIONS."""
        from rezaa.models.preferences import XFADE_TRANSITIONS

        prefs = UserPreferences(transition_type="ai_mix")
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        if len(edl.clip_decisions) >= 2:
            types = [cd.transition_type for cd in edl.clip_decisions]
            # ai_mix should never appear as a clip transition_type
            assert "ai_mix" not in types
            # All types should be from XFADE_TRANSITIONS
            for t in types:
                assert t in XFADE_TRANSITIONS

    def test_ai_mix_preserves_per_clip_types(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        """ai_mix should preserve diverse per-clip transitions, not override uniformly."""
        prefs = UserPreferences(transition_type="ai_mix")
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        if len(edl.clip_decisions) >= 2:
            types = set(cd.transition_type for cd in edl.clip_decisions)
            # With cycling, we expect more than one transition type
            assert len(types) > 1

    def test_non_ai_mix_overrides_uniformly(
        self, orchestrator, sample_audio_analysis, multiple_video_analyses, alignment
    ):
        """Non-ai_mix transition should be applied uniformly to all clips."""
        prefs = UserPreferences(transition_type="wipeleft")
        edl = orchestrator.orchestrate(
            sample_audio_analysis, multiple_video_analyses, alignment, prefs
        )
        for cd in edl.clip_decisions:
            assert cd.transition_type == "wipeleft"

    def test_fallback_fills_target_duration(
        self, orchestrator, long_audio_analysis, long_audio_videos, long_audio_alignment
    ):
        """Fallback EDL with 15s target and 30s audio should fill >= 90% of target."""
        prefs = UserPreferences(target_duration=15.0)
        edl = orchestrator.orchestrate(
            long_audio_analysis, long_audio_videos, long_audio_alignment, prefs
        )
        assert edl.total_duration >= 13.5, (
            f"Expected >= 13.5s (90% of 15s), got {edl.total_duration}s"
        )

    @pytest.fixture
    def short_clip_videos(self):
        """4 clips each 0.8s long — tests the early-exit bug with short clips."""
        clips = []
        for i in range(4):
            features = VideoFeatures(
                clip_id=f"short_{i:03d}",
                motion_score=0.7,
                energy_score=0.7,
                best_segments=[Segment(start=0.0, end=0.8, energy_score=0.7)],
                duration=0.8,
            )
            clips.append(VideoAnalysisOutput(
                clip_id=f"short_{i:03d}",
                features=features,
                temporal_consistency=0.8,
            ))
        return clips

    def test_short_clips_still_fill_target(self, orchestrator, short_clip_videos):
        """With 10s target and fast pacing, 4x0.8s clips should still fill >= 9s."""
        audio_features = AudioFeatures(
            bpm=140.0,
            beat_timestamps=[0.43 * i for i in range(1, 24)],
            energy_curve=[(i * 0.5, 0.6) for i in range(21)],
            duration=10.0,
        )
        audio = AudioAnalysisOutput(features=audio_features, confidence=0.9)
        agent = BeatClipAlignmentAgent()
        align = agent.align(audio, short_clip_videos)
        prefs = UserPreferences(target_duration=10.0, pacing="fast")
        edl = orchestrator.orchestrate(audio, short_clip_videos, align, prefs)
        assert edl.total_duration >= 9.0, (
            f"Expected >= 9.0s (90% of 10s), got {edl.total_duration}s"
        )

    def test_xfade_transitions_fill_target(
        self, orchestrator, long_audio_analysis, long_audio_videos, long_audio_alignment
    ):
        """Fast pacing + fade transition + 15s target should fill >= 13.5s despite xfade overlap."""
        prefs = UserPreferences(target_duration=15.0, pacing="fast", transition_type="fade")
        edl = orchestrator.orchestrate(
            long_audio_analysis, long_audio_videos, long_audio_alignment, prefs
        )
        assert edl.total_duration >= 13.5, (
            f"Expected >= 13.5s, got {edl.total_duration}s with {len(edl.clip_decisions)} clips"
        )
        # With xfade compensation, we need more clips than the naive 15
        assert len(edl.clip_decisions) > 15

    def test_xfade_medium_pacing_fills_target(
        self, orchestrator, long_audio_analysis, long_audio_videos, long_audio_alignment
    ):
        """Medium pacing + wipeleft + 15s target should fill >= 13.5s."""
        prefs = UserPreferences(target_duration=15.0, pacing="medium", transition_type="wipeleft")
        edl = orchestrator.orchestrate(
            long_audio_analysis, long_audio_videos, long_audio_alignment, prefs
        )
        assert edl.total_duration >= 13.5, (
            f"Expected >= 13.5s, got {edl.total_duration}s"
        )

    def test_ai_mix_xfade_fills_target(
        self, orchestrator, long_audio_analysis, long_audio_videos, long_audio_alignment
    ):
        """Fast pacing + ai_mix + 15s target should fill >= 13.5s."""
        prefs = UserPreferences(target_duration=15.0, pacing="fast", transition_type="ai_mix")
        edl = orchestrator.orchestrate(
            long_audio_analysis, long_audio_videos, long_audio_alignment, prefs
        )
        assert edl.total_duration >= 13.5, (
            f"Expected >= 13.5s, got {edl.total_duration}s"
        )

    def test_llm_shortfall_triggers_fallback(
        self, long_audio_analysis, long_audio_videos, long_audio_alignment
    ):
        """When LLM returns a short EDL (5s for 15s target), fallback should produce >= 13.5s."""
        # Build a short 5s EDL that the mocked LLM will return
        short_edl = EditDecisionList(
            clip_decisions=[
                ClipDecision(
                    clip_id="clip_000",
                    source_start=0.0,
                    source_end=5.0,
                    timeline_start=0.0,
                    timeline_end=5.0,
                    transition_type="cut",
                    transition_duration=0.0,
                    energy_match_score=0.8,
                ),
            ],
            audio_decision=AudioDecision(trim_start=0.0, trim_end=5.0),
            total_duration=5.0,
        )

        mock_client = MagicMock()
        orchestrator = DecisionOrchestrator(client=mock_client)

        with patch.object(orchestrator, "_call_llm", return_value=short_edl):
            prefs = UserPreferences(target_duration=15.0)
            edl = orchestrator.orchestrate(
                long_audio_analysis, long_audio_videos, long_audio_alignment, prefs
            )
            assert edl.total_duration >= 13.5, (
                f"Expected >= 13.5s (90% of 15s), got {edl.total_duration}s"
            )
