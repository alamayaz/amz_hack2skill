"""Audio analysis agent."""

from pathlib import Path

import librosa
import numpy as np

from rezaa.agents.base import BaseAgent
from rezaa.extractors.audio_extractor import AudioFeatureExtractor
from rezaa.models.audio import AudioAnalysisOutput, AudioFeatures


class AudioAnalysisAgent(BaseAgent):
    """Agent that analyzes audio files and produces structured analysis output."""

    def __init__(self, extractor: AudioFeatureExtractor | None = None):
        self.extractor = extractor or AudioFeatureExtractor()

    def analyze(self, audio_path: Path) -> AudioAnalysisOutput:
        """Analyze an audio file and return structured output."""
        features = self.extractor.extract_features(audio_path)
        beat_strength = self.classify_beat_strength(audio_path, features)
        refined_beats = self.refine_beat_detection(features)

        # Update features with refined beats
        features = features.model_copy(update={"beat_timestamps": refined_beats})

        confidence = self._calculate_confidence(features)

        return AudioAnalysisOutput(
            features=features,
            confidence=confidence,
            beat_strength=beat_strength,
            analysis_metadata={
                "extractor": "librosa",
                "refinement": "bpm_grid_snap",
            },
        )

    def refine_beat_detection(self, features: AudioFeatures) -> list[float]:
        """Refine beats by snapping to BPM grid."""
        if not features.beat_timestamps or features.bpm <= 0:
            return features.beat_timestamps

        beat_interval = 60.0 / features.bpm
        if not features.beat_timestamps:
            return []

        # Snap each beat to nearest grid position
        first_beat = features.beat_timestamps[0]
        refined = []
        for beat in features.beat_timestamps:
            # Find nearest grid position
            grid_pos = round((beat - first_beat) / beat_interval)
            snapped = first_beat + grid_pos * beat_interval
            # Only snap if within 10% of beat interval
            if abs(snapped - beat) < beat_interval * 0.1:
                refined.append(round(snapped, 4))
            else:
                refined.append(round(beat, 4))

        # Ensure monotonically increasing
        result = [refined[0]]
        for i in range(1, len(refined)):
            if refined[i] > result[-1]:
                result.append(refined[i])

        return result

    def classify_beat_strength(self, audio_path: Path, features: AudioFeatures) -> list[float]:
        """Classify the strength of each beat using onset strength."""
        if not features.beat_timestamps:
            return []

        y, sr = librosa.load(str(audio_path), sr=features.sample_rate)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=512)

        strengths = []
        for beat_time in features.beat_timestamps:
            # Find nearest onset strength value
            idx = np.argmin(np.abs(times - beat_time))
            strength = float(onset_env[idx])
            strengths.append(strength)

        # Normalize to [0, 1]
        if strengths:
            max_s = max(strengths)
            if max_s > 0:
                strengths = [round(s / max_s, 4) for s in strengths]
            else:
                strengths = [0.0] * len(strengths)

        return strengths

    def _calculate_confidence(self, features: AudioFeatures) -> float:
        """Calculate confidence based on feature quality indicators."""
        score = 0.5  # Base confidence

        # More beats = higher confidence
        if len(features.beat_timestamps) > 10:
            score += 0.2
        elif len(features.beat_timestamps) > 4:
            score += 0.1

        # Energy curve with variation = higher confidence
        if features.energy_curve:
            energies = [e for _, e in features.energy_curve]
            if len(energies) > 1:
                std = float(np.std(energies))
                score += min(0.2, std)

        # BPM in typical range
        if 60 <= features.bpm <= 180:
            score += 0.1

        return round(min(1.0, score), 4)
