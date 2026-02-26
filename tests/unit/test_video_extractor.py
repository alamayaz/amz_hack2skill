"""Tests for VideoFeatureExtractor."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from rezaa.extractors.video_extractor import VideoFeatureExtractor


def generate_test_video(
    path: Path,
    frames: int = 90,
    fps: float = 30.0,
    width: int = 160,
    height: int = 120,
    motion: bool = False,
) -> Path:
    """Generate a small test MP4 video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if motion:
            # Moving rectangle to create motion
            x = int((i / frames) * (width - 40))
            cv2.rectangle(frame, (x, 30), (x + 40, 90), (0, 255, 0), -1)
        else:
            # Static frame
            cv2.rectangle(frame, (50, 30), (110, 90), (100, 100, 100), -1)
        writer.write(frame)
    writer.release()
    return path


def generate_scene_change_video(path: Path, fps: float = 30.0) -> Path:
    """Generate a video with distinct scene changes."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height = 160, 120
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for color in colors:
        for _ in range(30):  # 1 second per color at 30fps
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    writer.release()
    return path


class TestVideoFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return VideoFeatureExtractor(sample_interval=5)

    @pytest.fixture
    def static_video(self, tmp_path):
        return generate_test_video(tmp_path / "static.mp4", motion=False)

    @pytest.fixture
    def motion_video(self, tmp_path):
        return generate_test_video(tmp_path / "motion.mp4", motion=True)

    @pytest.fixture
    def scene_video(self, tmp_path):
        return generate_scene_change_video(tmp_path / "scenes.mp4")

    def test_extract_features_returns_video_features(self, extractor, static_video):
        features = extractor.extract_features(static_video, "test_clip")
        assert features.clip_id == "test_clip"
        assert features.duration > 0
        assert features.fps > 0
        assert features.width > 0
        assert features.height > 0

    def test_static_video_low_motion(self, extractor, static_video):
        features = extractor.extract_features(static_video, "static")
        assert features.motion_score < 0.15, f"Static video has motion {features.motion_score}"

    def test_motion_video_higher_motion(self, extractor, motion_video):
        features = extractor.extract_features(motion_video, "motion")
        assert features.motion_score > 0.01, f"Motion video has motion {features.motion_score}"

    def test_motion_score_normalized(self, extractor, motion_video):
        features = extractor.extract_features(motion_video, "motion")
        assert 0 <= features.motion_score <= 1

    def test_energy_score_normalized(self, extractor, motion_video):
        features = extractor.extract_features(motion_video, "motion")
        assert 0 <= features.energy_score <= 1

    def test_scene_changes_valid(self, extractor, scene_video):
        features = extractor.extract_features(scene_video, "scenes")
        for ts, mag in features.scene_changes:
            assert ts >= 0
            assert 0 <= mag <= 1

    def test_scene_change_detection(self, extractor, scene_video):
        features = extractor.extract_features(scene_video, "scenes")
        # Should detect at least 1 scene change in the 3-color video
        assert len(features.scene_changes) >= 1

    def test_segments_non_overlapping(self, extractor, motion_video):
        features = extractor.extract_features(motion_video, "motion")
        sorted_segs = sorted(features.best_segments, key=lambda s: s.start)
        for i in range(1, len(sorted_segs)):
            assert sorted_segs[i].start >= sorted_segs[i - 1].end

    def test_segment_energy_scores_valid(self, extractor, motion_video):
        features = extractor.extract_features(motion_video, "motion")
        for seg in features.best_segments:
            assert 0 <= seg.energy_score <= 1
            assert seg.start < seg.end

    def test_features_serialization(self, extractor, static_video):
        features = extractor.extract_features(static_video, "test")
        json_str = features.model_dump_json()
        from rezaa.models.video import VideoFeatures

        restored = VideoFeatures.model_validate_json(json_str)
        assert restored.clip_id == features.clip_id
        assert restored.motion_score == features.motion_score

    def test_calculate_motion_score_single_frame(self, extractor):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert extractor.calculate_motion_score([frame]) == 0.0

    def test_calculate_motion_score_empty(self, extractor):
        assert extractor.calculate_motion_score([]) == 0.0
