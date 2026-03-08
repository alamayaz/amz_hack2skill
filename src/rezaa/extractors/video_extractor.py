"""Video feature extraction using OpenCV."""

from pathlib import Path

import cv2
import numpy as np

from rezaa.models.video import Segment, VideoFeatures


class VideoFeatureExtractor:
    """Extracts video features from video files using OpenCV."""

    def __init__(self, sample_interval: int = 5, segment_window_sec: float | None = None):
        self.sample_interval = sample_interval  # Process every Nth frame
        self.segment_window_sec = segment_window_sec  # Beat-synced window (set by pipeline)

    def extract_features(self, video_path: Path, clip_id: str) -> VideoFeatures:
        """Extract all video features from a file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0.0

            # Read frames at sample interval
            frames = []
            frame_indices = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.sample_interval == 0:
                    frames.append(frame)
                    frame_indices.append(idx)
                idx += 1

            if not frames:
                return VideoFeatures(
                    clip_id=clip_id,
                    motion_score=0.0,
                    energy_score=0.0,
                    duration=duration,
                    fps=fps,
                    width=width,
                    height=height,
                )

            per_frame_flow = self._compute_per_frame_flow(frames)
            motion_score = self.calculate_motion_score(frames, per_frame_flow)
            scene_changes = self.detect_scene_changes(frames, frame_indices, fps)
            energy_score = self.calculate_energy_score(frames, motion_score)
            window_sec = self.segment_window_sec or min(2.0, max(0.5, duration / 10))
            best_segments = self.identify_best_segments(
                frames, frame_indices, fps, duration, per_frame_flow, window_sec
            )

            return VideoFeatures(
                clip_id=clip_id,
                motion_score=motion_score,
                scene_changes=scene_changes,
                energy_score=energy_score,
                best_segments=best_segments,
                duration=duration,
                fps=fps,
                width=max(width, 1),
                height=max(height, 1),
            )
        finally:
            cap.release()

    @staticmethod
    def _compute_per_frame_flow(frames: list[np.ndarray]) -> list[float]:
        """Return normalized [0,1] optical flow magnitude per frame (0.0 for frame 0)."""
        if len(frames) < 2:
            return [0.0] * len(frames)

        flow_values = [0.0]  # first frame has no flow
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            # Normalize: flow > 10 pixels/frame is very high motion
            flow_values.append(float(min(1.0, np.mean(magnitude) / 10.0)))
            prev_gray = curr_gray

        return flow_values

    def calculate_motion_score(
        self, frames: list[np.ndarray], per_frame_flow: list[float] | None = None,
    ) -> float:
        """Calculate motion score using pre-computed or freshly computed optical flow."""
        if len(frames) < 2:
            return 0.0

        if per_frame_flow is None:
            per_frame_flow = self._compute_per_frame_flow(frames)

        # Skip frame 0 (always 0.0)
        flow_values = per_frame_flow[1:] if len(per_frame_flow) > 1 else per_frame_flow
        if not flow_values:
            return 0.0

        score = float(min(1.0, np.mean(flow_values)))
        return round(score, 4)

    def detect_scene_changes(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        fps: float,
        threshold: float = 0.3,
    ) -> list[tuple[float, float]]:
        """Detect scene changes using histogram chi-squared difference."""
        if len(frames) < 2:
            return []

        changes = []
        prev_hist = self._compute_histogram(frames[0])

        for i in range(1, len(frames)):
            curr_hist = self._compute_histogram(frames[i])
            diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
            # Normalize diff to [0,1] using sigmoid-like mapping
            normalized = float(min(1.0, diff / (diff + 1.0)))
            if normalized > threshold:
                timestamp = round(frame_indices[i] / fps, 4)
                changes.append((timestamp, round(normalized, 4)))
            prev_hist = curr_hist

        return changes

    def calculate_energy_score(self, frames: list[np.ndarray], motion_score: float) -> float:
        """Calculate energy score: 0.4*motion + 0.3*saturation + 0.2*brightness_var + 0.1*edges."""
        if not frames:
            return 0.0

        saturations = []
        brightness_vars = []
        edge_densities = []

        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Saturation (normalized to [0,1])
            saturations.append(float(np.mean(hsv[:, :, 1])) / 255.0)
            # Brightness variance (normalized)
            brightness_vars.append(float(np.std(hsv[:, :, 2])) / 128.0)
            # Edge density
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_densities.append(float(np.mean(edges > 0)))

        avg_sat = min(1.0, np.mean(saturations))
        avg_bvar = min(1.0, np.mean(brightness_vars))
        avg_edges = min(1.0, np.mean(edge_densities))

        score = 0.4 * motion_score + 0.3 * avg_sat + 0.2 * avg_bvar + 0.1 * avg_edges
        return round(float(min(1.0, max(0.0, score))), 4)

    def identify_best_segments(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        fps: float,
        duration: float,
        per_frame_flow: list[float] | None = None,
        window_sec: float = 2.0,
    ) -> list[Segment]:
        """Identify best segments using sliding window + non-max suppression.

        Per-segment energy now includes motion (matching whole-clip energy weights):
        0.4 * motion + 0.3 * saturation + 0.2 * edges + 0.1 * brightness_var
        """
        if len(frames) < 2 or duration < window_sec:
            if frames and duration > 0:
                energy = self.calculate_energy_score(frames, 0.5)
                return [Segment(start=0.0, end=round(duration, 4), energy_score=energy)]
            return []

        if per_frame_flow is None:
            per_frame_flow = self._compute_per_frame_flow(frames)

        # Calculate per-frame energy scores (matching calculate_energy_score weights)
        frame_times = [idx / fps for idx in frame_indices]
        frame_energies = []
        for i, frame in enumerate(frames):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1])) / 255.0
            bvar = min(1.0, float(np.std(hsv[:, :, 2])) / 128.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
            motion = per_frame_flow[i] if i < len(per_frame_flow) else 0.0
            frame_energies.append(0.4 * motion + 0.3 * sat + 0.2 * edges + 0.1 * bvar)

        # Sliding window
        candidates = []
        for i in range(len(frame_times)):
            start_t = frame_times[i]
            end_t = start_t + window_sec
            if end_t > duration:
                break
            # Average energy in window
            window_energies = [
                frame_energies[j] for j in range(i, len(frame_times)) if frame_times[j] < end_t
            ]
            if window_energies:
                avg_energy = float(np.mean(window_energies))
                candidates.append((start_t, end_t, avg_energy))

        if not candidates:
            return []

        # Sort by energy descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Adaptive segment cap
        max_segments = max(5, min(15, int(duration / window_sec)))

        # Non-max suppression
        selected = []
        for start, end, energy in candidates:
            overlaps = any(not (end <= s_start or start >= s_end) for s_start, s_end, _ in selected)
            if not overlaps:
                selected.append((start, end, energy))
            if len(selected) >= max_segments:
                break

        # Sort by start time
        selected.sort(key=lambda x: x[0])
        return [
            Segment(
                start=round(s, 4),
                end=round(e, 4),
                energy_score=round(min(1.0, sc), 4),
            )
            for s, e, sc in selected
        ]

    @staticmethod
    def _compute_histogram(frame: np.ndarray) -> np.ndarray:
        """Compute normalized HSV histogram for a frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
