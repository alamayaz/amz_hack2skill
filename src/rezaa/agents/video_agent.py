"""Video understanding agent."""

from pathlib import Path

import numpy as np

from rezaa.agents.base import BaseAgent
from rezaa.extractors.video_extractor import VideoFeatureExtractor
from rezaa.models.video import VideoAnalysisOutput, VideoFeatures


class VideoUnderstandingAgent(BaseAgent):
    """Agent that analyzes video clips and produces structured analysis output."""

    def __init__(self, extractor: VideoFeatureExtractor | None = None):
        self.extractor = extractor or VideoFeatureExtractor()

    def analyze(self, video_path: Path, clip_id: str) -> VideoAnalysisOutput:
        """Analyze a video file and return structured output."""
        features = self.extractor.extract_features(video_path, clip_id)
        temporal_consistency = self.analyze_temporal_consistency(features)
        recommended_usage = self._determine_usage(features)

        return VideoAnalysisOutput(
            clip_id=clip_id,
            features=features,
            temporal_consistency=temporal_consistency,
            recommended_usage=recommended_usage,
            analysis_metadata={
                "extractor": "opencv",
                "sample_interval": self.extractor.sample_interval,
            },
        )

    def analyze_temporal_consistency(self, features: VideoFeatures) -> float:
        """Analyze how consistent the clip's energy is over time."""
        if not features.best_segments:
            return 1.0

        energies = [seg.energy_score for seg in features.best_segments]
        if len(energies) < 2:
            return 1.0

        # Low standard deviation = high consistency
        std = float(np.std(energies))
        consistency = max(0.0, 1.0 - std)
        return round(consistency, 4)

    def _determine_usage(self, features: VideoFeatures) -> str:
        """Determine recommended usage based on features."""
        if features.energy_score > 0.7:
            return "high_energy"
        elif features.energy_score > 0.4:
            return "medium_energy"
        elif features.motion_score < 0.1:
            return "static_background"
        else:
            return "low_energy"
