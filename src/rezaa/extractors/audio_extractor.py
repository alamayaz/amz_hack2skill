"""Audio feature extraction using librosa."""

from pathlib import Path

import librosa
import numpy as np

from rezaa.models.audio import AudioFeatures


class AudioFeatureExtractor:
    """Extracts audio features from audio files using librosa."""

    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def extract_features(self, audio_path: Path) -> AudioFeatures:
        """Extract all audio features from a file."""
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)

        bpm = self.extract_bpm(y, sr)
        beat_timestamps = self.extract_beats(y, sr)
        energy_curve = self.extract_energy_curve(y, sr)
        drop_timestamps = self.detect_drops(energy_curve)

        return AudioFeatures(
            bpm=bpm,
            beat_timestamps=beat_timestamps,
            energy_curve=energy_curve,
            drop_timestamps=drop_timestamps,
            duration=duration,
            sample_rate=sr,
        )

    def extract_bpm(
        self, y: np.ndarray | None = None, sr: int | None = None, audio_path: Path | None = None
    ) -> float:
        """Extract BPM from audio signal or file."""
        if y is None:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        sr = sr or self.sample_rate
        tempo = librosa.beat.tempo(y=y, sr=sr, hop_length=self.hop_length)
        bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
        return max(30.0, min(300.0, bpm))

    def extract_beats(
        self, y: np.ndarray | None = None, sr: int | None = None, audio_path: Path | None = None
    ) -> list[float]:
        """Extract beat timestamps from audio signal or file."""
        if y is None:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        sr = sr or self.sample_rate
        _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
        return [round(float(t), 4) for t in beat_times]

    def extract_energy_curve(
        self,
        y: np.ndarray | None = None,
        sr: int | None = None,
        audio_path: Path | None = None,
        target_rate: float = 10.0,
    ) -> list[tuple[float, float]]:
        """Extract energy curve normalized to [0,1] at ~10Hz."""
        if y is None:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        sr = sr or self.sample_rate

        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Normalize to [0, 1]
        if rms.max() > 0:
            rms_norm = rms / rms.max()
        else:
            rms_norm = rms

        # Get timestamps
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)

        # Downsample to target_rate
        step = max(1, int(sr / (self.hop_length * target_rate)))
        curve = [
            (round(float(times[i]), 4), round(float(rms_norm[i]), 4))
            for i in range(0, len(times), step)
        ]
        return curve

    def detect_drops(
        self,
        energy_curve: list[tuple[float, float]] | None = None,
        y: np.ndarray | None = None,
        sr: int | None = None,
        threshold: float = 0.3,
    ) -> list[float]:
        """Detect energy drops (sudden energy increases after quiet)."""
        if energy_curve is None:
            if y is None:
                return []
            energy_curve = self.extract_energy_curve(y, sr)

        if len(energy_curve) < 3:
            return []

        drops = []
        energies = [e for _, e in energy_curve]
        times = [t for t, _ in energy_curve]

        for i in range(2, len(energies)):
            # Look for significant energy increase (derivative spike)
            diff = energies[i] - energies[i - 1]
            prev_diff = energies[i - 1] - energies[i - 2]
            if diff > threshold and prev_diff <= 0:
                drops.append(round(times[i], 4))

        return drops
