"""Property-based tests for user preferences (Properties 39-43)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rezaa.agents.alignment_agent import BeatClipAlignmentAgent
from rezaa.models.preferences import UserPreferences
from rezaa.orchestrator.decision import DecisionOrchestrator
from tests.property.conftest import generate_audio_analysis_output, generate_video_analysis_output

pytestmark = pytest.mark.property


@st.composite
def generate_pref_inputs(draw):
    """Generate inputs for preference testing."""
    audio = draw(generate_audio_analysis_output())
    n_clips = draw(st.integers(min_value=2, max_value=4))
    videos = [draw(generate_video_analysis_output()) for _ in range(n_clips)]
    for i, v in enumerate(videos):
        v.clip_id = f"clip_{i:03d}"
        v.features.clip_id = f"clip_{i:03d}"
    return audio, videos


class TestPreferenceProperties:
    @given(inputs=generate_pref_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_39_fast_pacing_short_clips(self, inputs):
        """Property 39: Fast pacing produces clips <= 1.5s."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(audio, videos, alignment, UserPreferences(pacing="fast"))
        for cd in edl.clip_decisions:
            dur = cd.timeline_end - cd.timeline_start
            assert dur <= 1.5 + 0.01

    @given(inputs=generate_pref_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_40_slow_pacing_long_clips(self, inputs):
        """Property 40: Slow pacing produces longer clips than fast pacing."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl_slow = orchestrator.orchestrate(audio, videos, alignment, UserPreferences(pacing="slow"))
        edl_fast = orchestrator.orchestrate(audio, videos, alignment, UserPreferences(pacing="fast"))
        # Slow pacing should use fewer or equal clips to cover the same duration
        # (each clip is longer), unless constrained by source material.
        if edl_slow.clip_decisions and edl_fast.clip_decisions:
            avg_slow = sum(cd.timeline_end - cd.timeline_start for cd in edl_slow.clip_decisions) / len(edl_slow.clip_decisions)
            avg_fast = sum(cd.timeline_end - cd.timeline_start for cd in edl_fast.clip_decisions) / len(edl_fast.clip_decisions)
            assert avg_slow >= avg_fast - 0.01

    @given(inputs=generate_pref_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_41_transition_type_applied(self, inputs):
        """Property 41: User-specified transition type is applied to all clips."""
        audio, videos = inputs
        for trans in ["cut", "fade", "crossfade"]:
            agent = BeatClipAlignmentAgent()
            alignment = agent.align(audio, videos)
            orchestrator = DecisionOrchestrator(client=None)
            edl = orchestrator.orchestrate(
                audio,
                videos,
                alignment,
                UserPreferences(transition_type=trans),
            )
            for cd in edl.clip_decisions:
                assert cd.transition_type == trans

    @given(inputs=generate_pref_inputs())
    @settings(max_examples=30, deadline=15000)
    def test_property_42_defaults_work(self, inputs):
        """Property 42: Default preferences produce valid EDL."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(audio, videos, alignment, UserPreferences())
        assert edl.total_duration >= 0

    @given(
        inputs=generate_pref_inputs(),
        target=st.floats(min_value=5.0, max_value=30.0),
    )
    @settings(max_examples=30, deadline=15000)
    def test_property_43_target_duration_respected(self, inputs, target):
        """Property 43: Target duration is respected."""
        audio, videos = inputs
        agent = BeatClipAlignmentAgent()
        alignment = agent.align(audio, videos)
        orchestrator = DecisionOrchestrator(client=None)
        edl = orchestrator.orchestrate(
            audio,
            videos,
            alignment,
            UserPreferences(target_duration=round(target, 1)),
        )
        if edl.clip_decisions:
            last_end = max(cd.timeline_end for cd in edl.clip_decisions)
            assert last_end <= target + 0.1
