"""Property-based tests for EDL (Properties 17-20)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.orchestrator.decision import DecisionOrchestrator
from tests.property.conftest import generate_audio_analysis_output, generate_video_analysis_output

pytestmark = pytest.mark.property


@st.composite
def generate_orchestration_inputs(draw):
    """Generate valid inputs for orchestration."""
    audio = draw(generate_audio_analysis_output())
    n_clips = draw(st.integers(min_value=2, max_value=4))
    videos = [draw(generate_video_analysis_output()) for _ in range(n_clips)]
    for i, v in enumerate(videos):
        v.clip_id = f"clip_{i:03d}"
        v.features.clip_id = f"clip_{i:03d}"
    return audio, videos


class TestEdlProperties:
    @given(inputs=generate_orchestration_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_17_energy_match_prioritization(self, inputs):
        """Property 17: EDL clip decisions have valid energy match scores."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(audio, videos, alignment)
        for cd in edl.clip_decisions:
            assert 0 <= cd.energy_match_score <= 1

    @given(inputs=generate_orchestration_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_18_no_consecutive_same_clip(self, inputs):
        """Property 18: No two consecutive clip decisions use the same clip."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(audio, videos, alignment)
        for i in range(1, len(edl.clip_decisions)):
            assert edl.clip_decisions[i].clip_id != edl.clip_decisions[i - 1].clip_id

    @given(inputs=generate_orchestration_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_19_duration_constraint(self, inputs):
        """Property 19: Source and timeline durations match within 0.1s."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(audio, videos, alignment)
        for cd in edl.clip_decisions:
            source_dur = cd.source_end - cd.source_start
            timeline_dur = cd.timeline_end - cd.timeline_start
            assert abs(source_dur - timeline_dur) <= 0.1

    @given(inputs=generate_orchestration_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_20_edl_completeness(self, inputs):
        """Property 20: EDL has a valid total duration."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(audio, videos, alignment)
        assert edl.total_duration >= 0
