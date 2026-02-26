"""Tests for DecisionOrchestrator (mocked LLM)."""

import json

import pytest

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.models.edl import EditDecisionList
from rezaa.models.preferences import UserPreferences
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
