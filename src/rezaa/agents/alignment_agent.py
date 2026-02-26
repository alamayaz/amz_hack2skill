"""Beat-clip alignment agent."""

import math
from collections import deque

from rezaa.agents.base import BaseAgent
from rezaa.models.alignment import AlignmentOutput, ClipPlacement
from rezaa.models.audio import AudioAnalysisOutput
from rezaa.models.video import VideoAnalysisOutput


class BeatClipAlignmentAgent(BaseAgent):
    """Agent that aligns video clips to audio beats based on energy matching."""

    def __init__(self, sigma: float = 0.3, min_reuse_gap: float = 5.0):
        self.sigma = sigma  # Gaussian kernel width for energy matching
        self.min_reuse_gap = min_reuse_gap  # Minimum seconds between reusing same clip

    def analyze(self, *args, **kwargs) -> AlignmentOutput:
        """Alias for align()."""
        return self.align(*args, **kwargs)

    def align(
        self,
        audio_analysis: AudioAnalysisOutput,
        video_analyses: list[VideoAnalysisOutput],
        target_duration: float | None = None,
    ) -> AlignmentOutput:
        """Align video clips to audio beats."""
        if not video_analyses:
            return AlignmentOutput(total_duration=0.0, coverage=0.0, average_energy_match=0.0)

        beats = audio_analysis.features.beat_timestamps
        beat_strengths = audio_analysis.beat_strength or [0.5] * len(beats)
        energy_curve = audio_analysis.features.energy_curve
        duration = target_duration or audio_analysis.features.duration

        if not beats:
            return AlignmentOutput(total_duration=duration, coverage=0.0, average_energy_match=0.0)

        # Filter beats within target duration
        beats = [b for b in beats if b < duration]
        beat_strengths = beat_strengths[: len(beats)]

        placements = self._build_timeline(
            beats, beat_strengths, energy_curve, video_analyses, duration
        )

        clips_used = list({p.clip_id for p in placements})
        coverage = len(placements) / len(beats) if beats else 0.0
        avg_match = (
            sum(p.energy_match_score for p in placements) / len(placements) if placements else 0.0
        )

        return AlignmentOutput(
            placements=placements,
            total_duration=duration,
            coverage=round(min(1.0, coverage), 4),
            average_energy_match=round(avg_match, 4),
            clips_used=clips_used,
            alignment_metadata={
                "sigma": self.sigma,
                "min_reuse_gap": self.min_reuse_gap,
                "total_beats": len(beats),
                "total_clips": len(video_analyses),
            },
        )

    def calculate_energy_match(self, beat_energy: float, clip_energy: float) -> float:
        """Gaussian kernel energy matching: exp(-diff^2 / (2*sigma^2))."""
        diff = beat_energy - clip_energy
        score = math.exp(-(diff**2) / (2 * self.sigma**2))
        return round(score, 4)

    def optimize_clip_duration(
        self, beat_energy: float, bpm: float, energy_trajectory: str = "stable"
    ) -> float:
        """Determine optimal clip duration based on beat intervals and energy."""
        beat_interval = 60.0 / bpm if bpm > 0 else 0.5

        if beat_energy > 0.7:
            # High energy: shorter clips (1 beat)
            duration = beat_interval
        elif beat_energy > 0.4:
            # Medium energy: 2 beats
            duration = beat_interval * 2
        else:
            # Low energy: 4 beats for more sustained shots
            duration = beat_interval * 4

        # Adjust for trajectory
        if energy_trajectory == "rising":
            duration *= 0.8
        elif energy_trajectory == "falling":
            duration *= 1.2

        return round(max(0.2, min(duration, 8.0)), 4)

    def handle_clip_reuse(self, clip_id: str, recent_clips: deque, current_time: float) -> bool:
        """Check if a clip can be reused (LRU with minimum separation)."""
        for used_id, used_time in recent_clips:
            if used_id == clip_id and (current_time - used_time) < self.min_reuse_gap:
                return False
        return True

    def _build_timeline(
        self,
        beats: list[float],
        beat_strengths: list[float],
        energy_curve: list[tuple[float, float]],
        video_analyses: list[VideoAnalysisOutput],
        duration: float,
    ) -> list[ClipPlacement]:
        """Build chronological timeline by iterating beats."""
        placements: list[ClipPlacement] = []
        recent_clips: deque = deque(maxlen=20)
        last_clip_id = None

        bpm = 120.0  # Default
        if len(beats) >= 2:
            intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
            avg_interval = sum(intervals) / len(intervals)
            bpm = 60.0 / avg_interval if avg_interval > 0 else 120.0

        for i, beat_time in enumerate(beats):
            beat_energy = self._get_energy_at_time(energy_curve, beat_time)
            beat_strength = beat_strengths[i] if i < len(beat_strengths) else 0.5

            # Combined energy for matching
            combined_energy = 0.6 * beat_energy + 0.4 * beat_strength

            # Find best matching clip
            best_clip = None
            best_score = -1.0

            for va in video_analyses:
                # Skip if same as last clip (variety enforcement)
                if va.clip_id == last_clip_id:
                    continue

                # Check reuse policy
                if not self.handle_clip_reuse(va.clip_id, recent_clips, beat_time):
                    continue

                score = self.calculate_energy_match(combined_energy, va.features.energy_score)
                if score > best_score:
                    best_score = score
                    best_clip = va

            # Fallback: if no clip available (all filtered), use best overall
            if best_clip is None:
                scores = [
                    (
                        va,
                        self.calculate_energy_match(combined_energy, va.features.energy_score),
                    )
                    for va in video_analyses
                    if va.clip_id != last_clip_id
                ]
                if not scores:
                    # Last resort: allow same as last
                    scores = [
                        (
                            va,
                            self.calculate_energy_match(combined_energy, va.features.energy_score),
                        )
                        for va in video_analyses
                    ]
                if scores:
                    best_clip, best_score = max(scores, key=lambda x: x[1])

            if best_clip is None:
                continue

            # Calculate clip duration and trim points
            clip_dur = self.optimize_clip_duration(combined_energy, bpm)
            clip_dur = min(clip_dur, best_clip.features.duration)

            # Find best segment within clip
            trim_start, trim_end = self._find_best_trim(best_clip, clip_dur, combined_energy)

            placement = ClipPlacement(
                clip_id=best_clip.clip_id,
                align_to_beat=beat_time,
                trim_start=trim_start,
                trim_end=trim_end,
                energy_match_score=best_score,
            )
            placements.append(placement)
            recent_clips.append((best_clip.clip_id, beat_time))
            last_clip_id = best_clip.clip_id

        return placements

    def _get_energy_at_time(self, energy_curve: list[tuple[float, float]], time: float) -> float:
        """Get interpolated energy at a given time."""
        if not energy_curve:
            return 0.5

        # Find bracketing points
        prev_t, prev_e = energy_curve[0]
        for t, e in energy_curve:
            if t >= time:
                if t == prev_t:
                    return e
                # Linear interpolation
                alpha = (time - prev_t) / (t - prev_t)
                return prev_e + alpha * (e - prev_e)
            prev_t, prev_e = t, e

        return energy_curve[-1][1]

    def _find_best_trim(
        self, clip: VideoAnalysisOutput, target_dur: float, target_energy: float
    ) -> tuple[float, float]:
        """Find the best trim points within a clip."""
        max_dur = clip.features.duration
        target_dur = min(target_dur, max_dur)

        if target_dur <= 0:
            target_dur = min(0.5, max_dur)

        # Try to find a segment that matches target energy
        if clip.features.best_segments:
            best_seg = None
            best_match = -1.0
            for seg in clip.features.best_segments:
                seg_dur = seg.end - seg.start
                if seg_dur >= target_dur:
                    match = self.calculate_energy_match(target_energy, seg.energy_score)
                    if match > best_match:
                        best_match = match
                        best_seg = seg

            if best_seg is not None:
                start = best_seg.start
                end = min(start + target_dur, best_seg.end)
                if end - start < target_dur * 0.5:
                    start = max(0, end - target_dur)
                return round(start, 4), round(end, 4)

        # Default: start from beginning
        return 0.0, round(target_dur, 4)
