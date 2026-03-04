"""Tests for rendering engine."""

import pytest

from rezaa.rendering.ffmpeg_builder import FFmpegFilterGraphBuilder
from rezaa.rendering.progress import FFmpegProgressMonitor


class TestFFmpegFilterGraphBuilder:
    @pytest.fixture
    def builder(self):
        return FFmpegFilterGraphBuilder()

    def test_build_empty_graph(self, builder):
        inputs, fc, dur = builder.build_filter_graph([])
        assert inputs == []
        assert fc == ""
        assert dur == 0.0

    def test_build_single_clip(self, builder):
        clips = [
            {
                "clip_path": "/tmp/clip1.mp4",
                "source_start": 0.0,
                "source_end": 2.0,
                "transition_type": "cut",
            }
        ]
        inputs, fc, dur = builder.build_filter_graph(clips)
        assert "-i" in inputs
        assert "/tmp/clip1.mp4" in inputs
        assert "trim" in fc
        assert "concat" in fc
        assert dur == pytest.approx(2.0)

    def test_build_multiple_clips(self, builder):
        clips = [
            {
                "clip_path": f"/tmp/clip{i}.mp4",
                "source_start": 0.0,
                "source_end": 1.0,
                "transition_type": "cut",
            }
            for i in range(3)
        ]
        inputs, fc, dur = builder.build_filter_graph(clips)
        assert inputs.count("-i") == 3
        assert "concat=n=3" in fc
        assert dur == pytest.approx(3.0)

    def test_build_audio_filter(self, builder):
        af = builder.build_audio_filter(
            "/tmp/audio.mp3",
            trim_start=0.0,
            trim_end=10.0,
            fade_in=0.5,
            fade_out=0.5,
            volume=0.8,
        )
        assert "atrim" in af
        assert "afade" in af
        assert "volume" in af

    def test_build_audio_filter_no_ops(self, builder):
        af = builder.build_audio_filter("/tmp/audio.mp3")
        assert af == ""

    def test_xfade_two_clips(self, builder):
        clips = [
            {"clip_path": "/tmp/a.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "cut"},
            {"clip_path": "/tmp/b.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "fade", "transition_duration": 0.5},
        ]
        inputs, fc, dur = builder.build_filter_graph(clips)
        assert inputs.count("-i") == 2
        assert "xfade" in fc
        assert "transition=fade" in fc
        assert "[outv]" in fc
        # Should NOT use concat
        assert "concat" not in fc
        # 2 + 2 - 0.5 = 3.5
        assert dur == pytest.approx(3.5)

    def test_xfade_three_clips(self, builder):
        clips = [
            {"clip_path": "/tmp/a.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "cut"},
            {"clip_path": "/tmp/b.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "wipeleft", "transition_duration": 0.5},
            {"clip_path": "/tmp/c.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "dissolve", "transition_duration": 0.5},
        ]
        inputs, fc, dur = builder.build_filter_graph(clips)
        assert inputs.count("-i") == 3
        assert "xfade" in fc
        assert "transition=wipeleft" in fc
        assert "transition=dissolve" in fc
        assert "[outv]" in fc
        # 3 clips * 2s - 2 * 0.5s = 5.0s
        assert dur == pytest.approx(5.0)

    def test_xfade_offset_calculation(self, builder):
        """Offset = accumulated_duration - transition_duration."""
        clips = [
            {"clip_path": "/tmp/a.mp4", "source_start": 0.0, "source_end": 3.0, "transition_type": "cut"},
            {"clip_path": "/tmp/b.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "fadeblack", "transition_duration": 0.5},
        ]
        _, fc, dur = builder.build_filter_graph(clips)
        # First clip is 3s, trans_dur is 0.5, so offset = 3.0 - 0.5 = 2.5
        assert "offset=2.5000" in fc
        # actual duration: offset(2.5) + clip2_dur(2.0) = 4.5
        assert dur == pytest.approx(4.5)

    def test_all_cuts_uses_concat(self, builder):
        clips = [
            {"clip_path": f"/tmp/c{i}.mp4", "source_start": 0.0, "source_end": 1.0, "transition_type": "cut"}
            for i in range(3)
        ]
        _, fc, dur = builder.build_filter_graph(clips)
        assert "concat=n=3" in fc
        assert "xfade" not in fc
        assert dur == pytest.approx(3.0)

    def test_single_clip_xfade_passthrough(self, builder):
        clips = [
            {"clip_path": "/tmp/a.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "fade", "transition_duration": 0.5},
        ]
        _, fc, dur = builder.build_filter_graph(clips)
        assert "copy" in fc
        assert "[outv]" in fc
        assert "xfade" not in fc
        assert dur == pytest.approx(2.0)

    def test_map_transition_to_xfade(self, builder):
        assert builder._map_transition_to_xfade("cut") is None
        assert builder._map_transition_to_xfade("fade") == "fade"
        assert builder._map_transition_to_xfade("crossfade") == "fade"
        assert builder._map_transition_to_xfade("wipeleft") == "wipeleft"
        assert builder._map_transition_to_xfade("dissolve") == "dissolve"
        assert builder._map_transition_to_xfade("radial") == "radial"

    def test_xfade_actual_duration_3_clips_2s(self, builder):
        """3 clips of 2s each with 0.5s transitions → 3*2 - 2*0.5 = 5.0."""
        clips = [
            {"clip_path": "/tmp/a.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "cut"},
            {"clip_path": "/tmp/b.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "fade", "transition_duration": 0.5},
            {"clip_path": "/tmp/c.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "fade", "transition_duration": 0.5},
        ]
        _, _, dur = builder.build_filter_graph(clips)
        assert dur == pytest.approx(5.0)

    def test_concat_actual_duration_3_clips_2s(self, builder):
        """3 clips of 2s each, all cuts → 3*2 = 6.0."""
        clips = [
            {"clip_path": f"/tmp/c{i}.mp4", "source_start": 0.0, "source_end": 2.0, "transition_type": "cut"}
            for i in range(3)
        ]
        _, _, dur = builder.build_filter_graph(clips)
        assert dur == pytest.approx(6.0)


class TestRenderingEngine:
    def test_no_shortest_flag(self):
        """Verify -shortest is absent from the generated ffmpeg command."""
        from pathlib import Path
        from rezaa.models.edl import AudioDecision, ClipDecision, EditDecisionList
        from rezaa.rendering.engine import RenderingEngine

        edl = EditDecisionList(
            clip_decisions=[
                ClipDecision(
                    clip_id="clip_000",
                    source_start=0.0,
                    source_end=2.0,
                    timeline_start=0.0,
                    timeline_end=2.0,
                    transition_type="cut",
                    transition_duration=0.0,
                    energy_match_score=0.8,
                ),
            ],
            audio_decision=AudioDecision(trim_start=0.0, trim_end=2.0),
            total_duration=2.0,
        )
        engine = RenderingEngine()
        cmd = engine.build_ffmpeg_command(
            edl,
            video_clips={"clip_000": Path("/tmp/clip.mp4")},
            audio_path=Path("/tmp/audio.mp3"),
            output_path=Path("/tmp/output.mp4"),
        )
        assert "-shortest" not in cmd


class TestFFmpegProgressMonitor:
    def test_parse_time(self):
        callback_values = []
        monitor = FFmpegProgressMonitor(10.0, callback=callback_values.append)
        ffmpeg_line = (
            "frame= 120 fps= 30 q=28.0 size=    256kB time=00:00:05.00 bitrate= 419.4kbits/s"
        )
        progress = monitor.parse_line(ffmpeg_line)
        assert progress is not None
        assert abs(progress - 0.5) < 0.01
        assert len(callback_values) == 1

    def test_parse_no_time(self):
        monitor = FFmpegProgressMonitor(10.0)
        progress = monitor.parse_line("Some other ffmpeg output")
        assert progress is None

    def test_progress_property(self):
        monitor = FFmpegProgressMonitor(10.0)
        monitor.parse_line("time=00:00:07.50")
        assert abs(monitor.progress - 0.75) < 0.01

    def test_zero_duration(self):
        monitor = FFmpegProgressMonitor(0.0)
        assert monitor.progress == 0.0
