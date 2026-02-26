"""Tests for all Pydantic data models."""

import pytest

from rezaa.models.alignment import AlignmentOutput, ClipPlacement
from rezaa.models.audio import AudioAnalysisOutput, AudioFeatures
from rezaa.models.edl import ClipDecision, EditDecisionList
from rezaa.models.errors import (
    ErrorResponse,
    ProcessingError,
    RenderingError,
    ResourceError,
    RezaaError,
    ValidationError,
)
from rezaa.models.pipeline import PipelineStage, PipelineState
from rezaa.models.preferences import UserPreferences
from rezaa.models.render import RenderResult
from rezaa.models.video import Segment, VideoAnalysisOutput, VideoFeatures

# --- AudioFeatures ---


class TestAudioFeatures:
    def test_valid_construction(self, sample_audio_features):
        assert sample_audio_features.bpm == 120.0
        assert len(sample_audio_features.beat_timestamps) == 6
        assert sample_audio_features.duration == 3.0

    def test_bpm_too_low(self):
        with pytest.raises(Exception):
            AudioFeatures(bpm=10.0, duration=3.0)

    def test_bpm_too_high(self):
        with pytest.raises(Exception):
            AudioFeatures(bpm=400.0, duration=3.0)

    def test_non_monotonic_beats(self):
        with pytest.raises(Exception):
            AudioFeatures(bpm=120.0, beat_timestamps=[1.0, 0.5, 2.0], duration=3.0)

    def test_negative_beat_timestamp(self):
        with pytest.raises(Exception):
            AudioFeatures(bpm=120.0, beat_timestamps=[-0.5, 0.5], duration=3.0)

    def test_energy_out_of_range(self):
        with pytest.raises(Exception):
            AudioFeatures(bpm=120.0, energy_curve=[(0.0, 1.5)], duration=3.0)

    def test_serialization_roundtrip(self, sample_audio_features):
        json_str = sample_audio_features.model_dump_json()
        restored = AudioFeatures.model_validate_json(json_str)
        assert restored == sample_audio_features


class TestAudioAnalysisOutput:
    def test_valid_construction(self, sample_audio_analysis):
        assert sample_audio_analysis.confidence == 0.85
        assert len(sample_audio_analysis.beat_strength) == 6

    def test_confidence_out_of_range(self, sample_audio_features):
        with pytest.raises(Exception):
            AudioAnalysisOutput(features=sample_audio_features, confidence=1.5)

    def test_beat_strength_out_of_range(self, sample_audio_features):
        with pytest.raises(Exception):
            AudioAnalysisOutput(features=sample_audio_features, confidence=0.8, beat_strength=[1.5])

    def test_serialization_roundtrip(self, sample_audio_analysis):
        json_str = sample_audio_analysis.model_dump_json()
        restored = AudioAnalysisOutput.model_validate_json(json_str)
        assert restored == sample_audio_analysis


# --- VideoFeatures ---


class TestSegment:
    def test_valid_segment(self):
        seg = Segment(start=0.0, end=1.0, energy_score=0.5)
        assert seg.start == 0.0
        assert seg.end == 1.0

    def test_start_after_end(self):
        with pytest.raises(Exception):
            Segment(start=2.0, end=1.0, energy_score=0.5)

    def test_start_equals_end(self):
        with pytest.raises(Exception):
            Segment(start=1.0, end=1.0, energy_score=0.5)

    def test_energy_out_of_range(self):
        with pytest.raises(Exception):
            Segment(start=0.0, end=1.0, energy_score=1.5)


class TestVideoFeatures:
    def test_valid_construction(self, sample_video_features):
        assert sample_video_features.clip_id == "clip_001"
        assert sample_video_features.motion_score == 0.6

    def test_motion_score_out_of_range(self):
        with pytest.raises(Exception):
            VideoFeatures(clip_id="test", motion_score=1.5, energy_score=0.5, duration=5.0)

    def test_overlapping_segments(self):
        with pytest.raises(Exception):
            VideoFeatures(
                clip_id="test",
                motion_score=0.5,
                energy_score=0.5,
                best_segments=[
                    Segment(start=0.0, end=2.0, energy_score=0.5),
                    Segment(start=1.0, end=3.0, energy_score=0.5),
                ],
                duration=5.0,
            )

    def test_scene_change_invalid_magnitude(self):
        with pytest.raises(Exception):
            VideoFeatures(
                clip_id="test",
                motion_score=0.5,
                scene_changes=[(1.0, 1.5)],
                energy_score=0.5,
                duration=5.0,
            )

    def test_serialization_roundtrip(self, sample_video_features):
        json_str = sample_video_features.model_dump_json()
        restored = VideoFeatures.model_validate_json(json_str)
        assert restored == sample_video_features


class TestVideoAnalysisOutput:
    def test_valid_construction(self, sample_video_analysis):
        assert sample_video_analysis.clip_id == "clip_001"
        assert sample_video_analysis.temporal_consistency == 0.9

    def test_serialization_roundtrip(self, sample_video_analysis):
        json_str = sample_video_analysis.model_dump_json()
        restored = VideoAnalysisOutput.model_validate_json(json_str)
        assert restored == sample_video_analysis


# --- Alignment ---


class TestClipPlacement:
    def test_valid_placement(self):
        cp = ClipPlacement(
            clip_id="clip_001",
            align_to_beat=1.0,
            trim_start=0.5,
            trim_end=1.5,
            energy_match_score=0.8,
        )
        assert cp.clip_id == "clip_001"

    def test_trim_start_after_end(self):
        with pytest.raises(Exception):
            ClipPlacement(
                clip_id="clip_001",
                align_to_beat=1.0,
                trim_start=2.0,
                trim_end=1.0,
                energy_match_score=0.8,
            )

    def test_energy_match_out_of_range(self):
        with pytest.raises(Exception):
            ClipPlacement(
                clip_id="clip_001",
                align_to_beat=1.0,
                trim_start=0.0,
                trim_end=1.0,
                energy_match_score=1.5,
            )


class TestAlignmentOutput:
    def test_valid_alignment(self):
        ao = AlignmentOutput(
            placements=[
                ClipPlacement(
                    clip_id="clip_001",
                    align_to_beat=0.5,
                    trim_start=0.0,
                    trim_end=0.5,
                    energy_match_score=0.8,
                ),
                ClipPlacement(
                    clip_id="clip_002",
                    align_to_beat=1.0,
                    trim_start=0.0,
                    trim_end=0.5,
                    energy_match_score=0.7,
                ),
            ],
            total_duration=1.5,
            coverage=0.9,
            average_energy_match=0.75,
            clips_used=["clip_001", "clip_002"],
        )
        assert len(ao.placements) == 2

    def test_unsorted_placements(self):
        with pytest.raises(Exception):
            AlignmentOutput(
                placements=[
                    ClipPlacement(
                        clip_id="clip_001",
                        align_to_beat=2.0,
                        trim_start=0.0,
                        trim_end=0.5,
                        energy_match_score=0.8,
                    ),
                    ClipPlacement(
                        clip_id="clip_002",
                        align_to_beat=1.0,
                        trim_start=0.0,
                        trim_end=0.5,
                        energy_match_score=0.7,
                    ),
                ],
                total_duration=2.5,
                coverage=0.9,
                average_energy_match=0.75,
                clips_used=["clip_001", "clip_002"],
            )

    def test_serialization_roundtrip(self):
        ao = AlignmentOutput(
            placements=[
                ClipPlacement(
                    clip_id="c1",
                    align_to_beat=0.5,
                    trim_start=0.0,
                    trim_end=0.5,
                    energy_match_score=0.8,
                )
            ],
            total_duration=1.0,
            coverage=0.5,
            average_energy_match=0.8,
            clips_used=["c1"],
        )
        json_str = ao.model_dump_json()
        restored = AlignmentOutput.model_validate_json(json_str)
        assert restored == ao


# --- EDL ---


class TestClipDecision:
    def test_valid_decision(self):
        cd = ClipDecision(
            clip_id="clip_001",
            source_start=0.0,
            source_end=1.0,
            timeline_start=0.0,
            timeline_end=1.0,
        )
        assert cd.transition_type == "cut"

    def test_source_start_after_end(self):
        with pytest.raises(Exception):
            ClipDecision(
                clip_id="clip_001",
                source_start=2.0,
                source_end=1.0,
                timeline_start=0.0,
                timeline_end=1.0,
            )

    def test_duration_mismatch(self):
        with pytest.raises(Exception):
            ClipDecision(
                clip_id="clip_001",
                source_start=0.0,
                source_end=2.0,
                timeline_start=0.0,
                timeline_end=1.0,
            )

    def test_no_consecutive_same_clip(self):
        with pytest.raises(Exception):
            EditDecisionList(
                clip_decisions=[
                    ClipDecision(
                        clip_id="clip_001",
                        source_start=0.0,
                        source_end=1.0,
                        timeline_start=0.0,
                        timeline_end=1.0,
                    ),
                    ClipDecision(
                        clip_id="clip_001",
                        source_start=1.0,
                        source_end=2.0,
                        timeline_start=1.0,
                        timeline_end=2.0,
                    ),
                ],
                total_duration=2.0,
            )


class TestEditDecisionList:
    def test_valid_edl(self):
        edl = EditDecisionList(
            clip_decisions=[
                ClipDecision(
                    clip_id="clip_001",
                    source_start=0.0,
                    source_end=1.0,
                    timeline_start=0.0,
                    timeline_end=1.0,
                ),
                ClipDecision(
                    clip_id="clip_002",
                    source_start=0.0,
                    source_end=1.0,
                    timeline_start=1.0,
                    timeline_end=2.0,
                ),
            ],
            total_duration=2.0,
        )
        assert len(edl.clip_decisions) == 2

    def test_overlapping_clips(self):
        with pytest.raises(Exception):
            EditDecisionList(
                clip_decisions=[
                    ClipDecision(
                        clip_id="clip_001",
                        source_start=0.0,
                        source_end=2.0,
                        timeline_start=0.0,
                        timeline_end=2.0,
                    ),
                    ClipDecision(
                        clip_id="clip_002",
                        source_start=0.0,
                        source_end=2.0,
                        timeline_start=1.0,
                        timeline_end=3.0,
                    ),
                ],
                total_duration=3.0,
            )

    def test_serialization_roundtrip(self):
        edl = EditDecisionList(
            clip_decisions=[
                ClipDecision(
                    clip_id="c1",
                    source_start=0.0,
                    source_end=1.0,
                    timeline_start=0.0,
                    timeline_end=1.0,
                )
            ],
            total_duration=1.0,
        )
        json_str = edl.model_dump_json()
        restored = EditDecisionList.model_validate_json(json_str)
        assert restored == edl


# --- Preferences ---


class TestUserPreferences:
    def test_defaults(self):
        prefs = UserPreferences()
        assert prefs.pacing == "medium"
        assert prefs.style == "smooth"
        assert prefs.transition_type == "cut"
        assert prefs.target_duration is None

    def test_fast_pacing(self):
        prefs = UserPreferences(pacing="fast", style="energetic")
        assert prefs.pacing == "fast"

    def test_invalid_pacing(self):
        with pytest.raises(Exception):
            UserPreferences(pacing="turbo")

    def test_target_duration_range(self):
        prefs = UserPreferences(target_duration=30.0)
        assert prefs.target_duration == 30.0

    def test_target_duration_too_short(self):
        with pytest.raises(Exception):
            UserPreferences(target_duration=2.0)


# --- Pipeline ---


class TestPipelineState:
    def test_default_state(self):
        state = PipelineState(job_id="test-job")
        assert state.stage == PipelineStage.UPLOAD
        assert state.progress == 0.0

    def test_all_stages(self):
        for stage in PipelineStage:
            state = PipelineState(job_id="test", stage=stage)
            assert state.stage == stage


# --- Errors ---


class TestErrors:
    def test_rezaa_error(self):
        err = RezaaError("something failed", component="test")
        assert str(err) == "something failed"
        assert err.component == "test"

    def test_validation_error(self):
        err = ValidationError("bad format", details={"file": "test.xyz"})
        assert err.component == "validation"

    def test_processing_error(self):
        err = ProcessingError("extraction failed")
        assert err.component == "processing"

    def test_resource_error(self):
        err = ResourceError("disk full")
        assert err.component == "resource"

    def test_rendering_error(self):
        err = RenderingError("ffmpeg crashed")
        assert err.component == "rendering"

    def test_error_response_from_exception(self):
        err = ValidationError("bad file", details={"ext": "xyz"})
        resp = ErrorResponse.from_exception(err, guidance="Use MP4", retry=False)
        assert resp.error_type == "ValidationError"
        assert resp.component == "validation"
        assert resp.actionable_guidance == "Use MP4"

    def test_error_response_serialization(self):
        resp = ErrorResponse(
            error_type="ValidationError",
            message="bad file",
            component="validation",
        )
        json_str = resp.model_dump_json()
        restored = ErrorResponse.model_validate_json(json_str)
        assert restored == resp


# --- Render ---


class TestRenderResult:
    def test_valid_result(self):
        rr = RenderResult(
            output_path="/tmp/output.mp4",
            duration=30.0,
            file_size_bytes=1024 * 1024 * 10,
        )
        assert rr.file_size_mb == pytest.approx(10.0)
        assert rr.output_file.name == "output.mp4"

    def test_serialization_roundtrip(self):
        rr = RenderResult(
            output_path="/tmp/out.mp4",
            duration=5.0,
            file_size_bytes=5000,
        )
        json_str = rr.model_dump_json()
        restored = RenderResult.model_validate_json(json_str)
        assert restored.output_path == rr.output_path
