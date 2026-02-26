"""Tests for AudioFeatureExtractor."""

import pytest

from rezaa.extractors.audio_extractor import AudioFeatureExtractor
from tests.conftest import generate_click_track_wav, generate_silent_wav, generate_test_wav


class TestAudioFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return AudioFeatureExtractor(sample_rate=22050)

    @pytest.fixture
    def click_track(self, tmp_path):
        return generate_click_track_wav(tmp_path / "click_120bpm.wav", bpm=120.0, duration=5.0)

    @pytest.fixture
    def tone_file(self, tmp_path):
        return generate_test_wav(tmp_path / "tone.wav", duration=3.0, freq=440.0)

    @pytest.fixture
    def silent_file(self, tmp_path):
        return generate_silent_wav(tmp_path / "silent.wav", duration=2.0)

    def test_extract_features_returns_audio_features(self, extractor, click_track):
        features = extractor.extract_features(click_track)
        assert features.bpm >= 30.0
        assert features.bpm <= 300.0
        assert features.duration > 0
        assert features.sample_rate == 22050

    def test_bpm_detection_accuracy(self, extractor, click_track):
        features = extractor.extract_features(click_track)
        # Should detect BPM within +/-5 of 120
        assert abs(features.bpm - 120.0) <= 5.0, f"BPM {features.bpm} not within 5 of 120"

    def test_beat_timestamps_monotonic(self, extractor, click_track):
        features = extractor.extract_features(click_track)
        for i in range(1, len(features.beat_timestamps)):
            assert features.beat_timestamps[i] > features.beat_timestamps[i - 1]

    def test_energy_curve_normalized(self, extractor, click_track):
        features = extractor.extract_features(click_track)
        for ts, energy in features.energy_curve:
            assert 0 <= energy <= 1, f"Energy {energy} not in [0,1]"
            assert ts >= 0

    def test_energy_curve_has_data(self, extractor, click_track):
        features = extractor.extract_features(click_track)
        assert len(features.energy_curve) > 0

    def test_silent_audio_low_energy(self, extractor, silent_file):
        features = extractor.extract_features(silent_file)
        if features.energy_curve:
            avg_energy = sum(e for _, e in features.energy_curve) / len(features.energy_curve)
            assert avg_energy < 0.1, f"Silent audio has avg energy {avg_energy}"

    def test_extract_bpm_standalone(self, extractor, click_track):
        import librosa

        y, sr = librosa.load(str(click_track), sr=22050)
        bpm = extractor.extract_bpm(y, sr)
        assert 30 <= bpm <= 300

    def test_extract_beats_standalone(self, extractor, click_track):
        import librosa

        y, sr = librosa.load(str(click_track), sr=22050)
        beats = extractor.extract_beats(y, sr)
        assert isinstance(beats, list)
        assert all(isinstance(t, float) for t in beats)

    def test_detect_drops(self, extractor):
        # Create an energy curve with a drop
        curve = [
            (0.0, 0.2),
            (0.1, 0.2),
            (0.2, 0.1),
            (0.3, 0.1),
            (0.4, 0.0),
            (0.5, 0.8),
            (0.6, 0.9),
        ]
        drops = extractor.detect_drops(energy_curve=curve, threshold=0.3)
        assert len(drops) > 0
        # Drop should be around 0.5s where energy jumps from 0.0 to 0.8
        assert any(abs(d - 0.5) < 0.2 for d in drops)

    def test_detect_drops_no_drops(self, extractor):
        # Flat energy curve — no drops
        curve = [(i * 0.1, 0.5) for i in range(20)]
        drops = extractor.detect_drops(energy_curve=curve)
        assert drops == []

    def test_features_serialization(self, extractor, click_track):
        features = extractor.extract_features(click_track)
        json_str = features.model_dump_json()
        from rezaa.models.audio import AudioFeatures

        restored = AudioFeatures.model_validate_json(json_str)
        assert restored.bpm == features.bpm
        assert len(restored.beat_timestamps) == len(features.beat_timestamps)
