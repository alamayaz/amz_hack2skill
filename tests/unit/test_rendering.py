"""Tests for rendering engine."""

import pytest

from rezaa.rendering.ffmpeg_builder import FFmpegFilterGraphBuilder
from rezaa.rendering.progress import FFmpegProgressMonitor


class TestFFmpegFilterGraphBuilder:
    @pytest.fixture
    def builder(self):
        return FFmpegFilterGraphBuilder()

    def test_build_empty_graph(self, builder):
        inputs, fc = builder.build_filter_graph([])
        assert inputs == []
        assert fc == ""

    def test_build_single_clip(self, builder):
        clips = [
            {
                "clip_path": "/tmp/clip1.mp4",
                "source_start": 0.0,
                "source_end": 2.0,
                "transition_type": "cut",
            }
        ]
        inputs, fc = builder.build_filter_graph(clips)
        assert "-i" in inputs
        assert "/tmp/clip1.mp4" in inputs
        assert "trim" in fc
        assert "concat" in fc

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
        inputs, fc = builder.build_filter_graph(clips)
        assert inputs.count("-i") == 3
        assert "concat=n=3" in fc

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
