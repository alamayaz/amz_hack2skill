"""Hypothesis strategies for property-based testing."""

from hypothesis import strategies as st

from rezaa.models.audio import AudioAnalysisOutput, AudioFeatures
from rezaa.models.video import Segment, VideoAnalysisOutput, VideoFeatures


@st.composite
def generate_audio_features(draw):
    """Generate random valid AudioFeatures."""
    bpm = draw(st.floats(min_value=30.0, max_value=300.0))
    duration = draw(st.floats(min_value=1.0, max_value=60.0))
    n_beats = draw(st.integers(min_value=0, max_value=50))

    # Generate monotonically increasing beat timestamps
    beat_timestamps = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.01, max_value=duration - 0.01),
                min_size=n_beats,
                max_size=n_beats,
            )
        )
    )
    # Ensure strict monotonicity
    unique_beats = []
    for b in beat_timestamps:
        if not unique_beats or b > unique_beats[-1] + 0.001:
            unique_beats.append(round(b, 4))
    beat_timestamps = unique_beats

    # Generate energy curve
    n_energy = draw(st.integers(min_value=2, max_value=30))
    energy_times = sorted([round(i * duration / n_energy, 4) for i in range(n_energy)])
    energy_values = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=n_energy,
            max_size=n_energy,
        )
    )
    energy_curve = list(zip(energy_times, [round(e, 4) for e in energy_values]))

    return AudioFeatures(
        bpm=round(bpm, 2),
        beat_timestamps=beat_timestamps,
        energy_curve=energy_curve,
        drop_timestamps=[],
        duration=round(duration, 4),
        sample_rate=22050,
    )


@st.composite
def generate_audio_analysis_output(draw):
    """Generate random valid AudioAnalysisOutput."""
    features = draw(generate_audio_features())
    n_beats = len(features.beat_timestamps)
    beat_strength = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=n_beats,
            max_size=n_beats,
        )
    )
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))

    return AudioAnalysisOutput(
        features=features,
        confidence=round(confidence, 4),
        beat_strength=[round(s, 4) for s in beat_strength],
    )


@st.composite
def generate_segment(draw, max_duration=10.0):
    """Generate a random valid Segment."""
    start = draw(st.floats(min_value=0.0, max_value=max_duration - 0.5))
    end = draw(st.floats(min_value=start + 0.1, max_value=min(start + 5.0, max_duration)))
    energy = draw(st.floats(min_value=0.0, max_value=1.0))
    return Segment(start=round(start, 4), end=round(end, 4), energy_score=round(energy, 4))


@st.composite
def generate_video_features(draw):
    """Generate random valid VideoFeatures."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_"
    clip_id = draw(st.text(min_size=1, max_size=20, alphabet=alphabet))
    duration = draw(st.floats(min_value=1.0, max_value=30.0))
    motion = draw(st.floats(min_value=0.0, max_value=1.0))
    energy = draw(st.floats(min_value=0.0, max_value=1.0))

    # Generate non-overlapping segments
    segments = []
    pos = 0.0
    n_segments = draw(st.integers(min_value=0, max_value=3))
    for _ in range(n_segments):
        if pos + 0.5 >= duration:
            break
        seg_start = pos
        seg_end = min(pos + draw(st.floats(min_value=0.5, max_value=2.0)), duration - 0.01)
        if seg_end <= seg_start:
            break
        seg_energy = draw(st.floats(min_value=0.0, max_value=1.0))
        segments.append(
            Segment(
                start=round(seg_start, 4),
                end=round(seg_end, 4),
                energy_score=round(seg_energy, 4),
            )
        )
        pos = seg_end + 0.1

    return VideoFeatures(
        clip_id=clip_id,
        motion_score=round(motion, 4),
        energy_score=round(energy, 4),
        best_segments=segments,
        duration=round(duration, 4),
    )


@st.composite
def generate_video_analysis_output(draw):
    """Generate random valid VideoAnalysisOutput."""
    features = draw(generate_video_features())
    consistency = draw(st.floats(min_value=0.0, max_value=1.0))
    return VideoAnalysisOutput(
        clip_id=features.clip_id,
        features=features,
        temporal_consistency=round(consistency, 4),
    )
