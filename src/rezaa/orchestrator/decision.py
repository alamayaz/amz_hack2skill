"""Decision orchestrator — LLM-based editing decision maker."""

import logging
import math

from openai import OpenAI

from rezaa.config import get_settings
from rezaa.models.alignment import AlignmentOutput, ClipPlacement
from rezaa.models.audio import AudioAnalysisOutput
from rezaa.models.edl import AudioDecision, ClipDecision, EditDecisionList
from rezaa.models.preferences import XFADE_TRANSITIONS, UserPreferences
from rezaa.models.video import Segment, VideoAnalysisOutput
from rezaa.orchestrator.parser import parse_llm_response, validate_edl
from rezaa.orchestrator.prompts import SYSTEM_PROMPT, build_orchestration_prompt

logger = logging.getLogger(__name__)

_DEFAULT_TRANSITION_DURATION = 0.5


class DecisionOrchestrator:
    """Orchestrates editing decisions using LLM or fallback rules."""

    def __init__(self, client: OpenAI | None = None, model: str | None = None):
        settings = get_settings()
        self.model = model or settings.openai_model
        self.client = client
        if self.client is None and settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)

    def orchestrate(
        self,
        audio_analysis: AudioAnalysisOutput,
        video_analyses: list[VideoAnalysisOutput],
        alignment: AlignmentOutput,
        user_preferences: UserPreferences | None = None,
    ) -> EditDecisionList:
        """Create an EDL by calling LLM or falling back to rules."""
        prefs = user_preferences or UserPreferences()

        target = prefs.target_duration or audio_analysis.features.duration

        # Try LLM
        if self.client:
            try:
                edl = self._call_llm(audio_analysis, video_analyses, alignment, prefs)
                edl = self.apply_user_preferences(edl, prefs)
                if edl.total_duration >= target * 0.9:
                    edl.edl_metadata["orchestration_method"] = "llm"
                    return edl
                logger.warning(
                    "LLM EDL too short (%.1fs vs %.1fs target), using fallback",
                    edl.total_duration, target,
                )
            except Exception as e:
                logger.warning(f"LLM orchestration failed, using fallback: {e}")

        # Fallback
        edl = self._create_fallback_edl(audio_analysis, video_analyses, alignment, prefs)
        edl = self.apply_user_preferences(edl, prefs)
        edl.edl_metadata["orchestration_method"] = "fallback"
        return edl

    def _call_llm(
        self,
        audio_analysis: AudioAnalysisOutput,
        video_analyses: list[VideoAnalysisOutput],
        alignment: AlignmentOutput,
        prefs: UserPreferences,
    ) -> EditDecisionList:
        """Call LLM to generate EDL."""
        prompt = build_orchestration_prompt(
            audio_analysis=audio_analysis.model_dump(),
            video_analyses=[va.model_dump() for va in video_analyses],
            alignment=alignment.model_dump(),
            user_preferences=prefs.model_dump(),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )

        response_text = response.choices[0].message.content
        edl_data = parse_llm_response(response_text)
        return validate_edl(edl_data, audio_analysis.features.duration)

    @staticmethod
    def _find_best_audio_window(
        audio_analysis: AudioAnalysisOutput, window: float
    ) -> float:
        """Find the start time of the most energetic window in the audio.

        Slides a *window*-second frame across the energy curve and returns the
        offset with the highest average energy.  Falls back to a drop timestamp
        if one exists, or 0.0 if energy data is unavailable.
        """
        audio_dur = audio_analysis.features.duration
        if window >= audio_dur:
            return 0.0

        curve = audio_analysis.features.energy_curve
        if not curve:
            # No energy data — prefer first drop if available
            for dt in audio_analysis.features.drop_timestamps:
                start = max(0.0, dt - 1.0)
                if start + window <= audio_dur:
                    return round(start, 4)
            return 0.0

        best_start = 0.0
        best_avg = -1.0

        # Sample candidate start positions every 0.5 s
        step = 0.5
        t = 0.0
        while t + window <= audio_dur + 0.01:
            pts = [(ts, e) for ts, e in curve if t <= ts <= t + window]
            if pts:
                avg = sum(e for _, e in pts) / len(pts)
                if avg > best_avg:
                    best_avg = avg
                    best_start = t
            t += step

        return round(best_start, 4)

    @staticmethod
    def _get_audio_energy_at(
        energy_curve: list[tuple[float, float]], timestamp: float,
    ) -> float:
        """Return interpolated audio energy at *timestamp*."""
        if not energy_curve:
            return 0.5
        # Find nearest point
        best_e = energy_curve[0][1]
        best_dist = abs(energy_curve[0][0] - timestamp)
        for ts, e in energy_curve:
            d = abs(ts - timestamp)
            if d < best_dist:
                best_dist = d
                best_e = e
        return best_e

    @staticmethod
    def _find_best_segment_for_energy(
        segments: list[Segment],
        target_energy: float,
        min_dur: float,
        clip_duration: float,
        used_ranges: list[tuple[float, float]],
    ) -> tuple[float, float, float] | None:
        """Pick the best segment matching *target_energy*.

        Returns ``(start, end, energy_match_score)`` or ``None``.
        Prefers segments that don't overlap previously used ranges.
        """
        if not segments:
            return None

        sigma = 0.3
        candidates: list[tuple[float, float, float, bool]] = []
        for seg in segments:
            seg_len = seg.end - seg.start
            if seg_len < min_dur:
                continue
            diff = abs(seg.energy_score - target_energy)
            score = math.exp(-(diff ** 2) / (2 * sigma ** 2))
            # Check overlap with already-used ranges
            overlaps = any(
                seg.start < ue and seg.end > us for us, ue in used_ranges
            )
            candidates.append((seg.start, min(seg.start + min_dur, seg.end), score, overlaps))

        if not candidates:
            return None

        # Prefer non-overlapping, then highest score
        candidates.sort(key=lambda c: (c[3], -c[2]))
        best = candidates[0]
        return (best[0], best[1], best[2])

    def _create_fallback_edl(
        self,
        audio_analysis: AudioAnalysisOutput,
        video_analyses: list[VideoAnalysisOutput],
        alignment: AlignmentOutput,
        prefs: UserPreferences,
    ) -> EditDecisionList:
        """Beat-aligned fallback EDL using alignment placements and best segments.

        Advances along the timeline, snapping cut boundaries to the nearest
        beat when possible.  For each segment it picks the best clip using:
        1. Alignment placement data (beat-matched, energy-scored)
        2. Best-segment energy matching from video analysis
        3. Round-robin with source stride as a last resort
        """
        audio_dur = audio_analysis.features.duration
        target = prefs.target_duration or audio_dur
        if not video_analyses:
            return EditDecisionList(
                total_duration=target,
                audio_decision=AudioDecision(trim_end=target),
            )

        # Use manual audio_start if provided, otherwise auto-select
        if prefs.audio_start is not None:
            audio_start = min(prefs.audio_start, max(0.0, audio_dur - target))
        else:
            audio_start = self._find_best_audio_window(audio_analysis, target)
        audio_end = round(min(audio_start + target, audio_dur), 4)
        actual_target = round(audio_end - audio_start, 4)

        # Pacing → per-segment duration
        pacing_defaults = {"fast": 1.0, "medium": 2.0, "slow": 4.0}
        clip_dur = pacing_defaults.get(prefs.pacing, 2.0)

        # Lookup maps
        clip_durations = {va.clip_id: va.features.duration for va in video_analyses}
        clip_segments: dict[str, list[Segment]] = {
            va.clip_id: sorted(va.features.best_segments, key=lambda s: -s.energy_score)
            for va in video_analyses
        }
        energy_curve = audio_analysis.features.energy_curve

        # Single-clip fallback: auto-upgrade "cut" to "ai_mix" for visible edits
        if len(video_analyses) == 1 and prefs.transition_type == "cut":
            is_ai_mix = True
        else:
            is_ai_mix = prefs.transition_type == "ai_mix"

        # ── Alignment placements shifted into the window, sorted by beat ──
        window_placements: list[tuple[float, ClipPlacement]] = []
        for p in alignment.placements:
            rel = round(p.align_to_beat - audio_start, 4)
            if 0 <= rel < actual_target:
                window_placements.append((rel, p))
        window_placements.sort(key=lambda x: x[0])

        all_clip_ids = [va.clip_id for va in video_analyses]
        max_reuse = prefs.max_clip_reuse

        # ── Helpers ──────────────────────────────────────────────────────
        def _reuse_count(cid: str) -> int:
            return sum(1 for cd in clip_decisions if cd.clip_id == cid)

        def _is_available(cid: str) -> bool:
            return _reuse_count(cid) < max_reuse

        # ── Thin placements: skip any < clip_dur*0.5 from previous ───────
        thinned: list[tuple[float, ClipPlacement]] = []
        last_accepted = -float("inf")
        for rel, p in window_placements:
            if rel - last_accepted >= clip_dur * 0.5:
                thinned.append((rel, p))
                last_accepted = rel

        # ── Convert thinned placements to ClipDecisions ──────────────────
        clip_decisions: list[ClipDecision] = []
        timeline_pos = 0.0
        xfade_overlap = 0.0
        last_clip_id: str | None = None
        used_src_ranges: dict[str, list[tuple[float, float]]] = {}
        robin_idx = 0

        # Build a queue of placement clip choices indexed by timeline proximity
        placement_queue = list(thinned)  # [(rel_beat, ClipPlacement), ...]
        pq_idx = 0

        safety = 0
        while (timeline_pos - xfade_overlap) < actual_target - 0.05:
            safety += 1
            if safety > len(all_clip_ids) * 200:
                break

            seg_dur = clip_dur

            # Determine transition
            if is_ai_mix:
                clip_index = len(clip_decisions)
                trans_type = XFADE_TRANSITIONS[clip_index % len(XFADE_TRANSITIONS)]
                trans_dur = _DEFAULT_TRANSITION_DURATION
            else:
                trans_type = prefs.transition_type
                trans_dur = _DEFAULT_TRANSITION_DURATION if trans_type != "cut" else 0.0

            remaining = actual_target - (timeline_pos - xfade_overlap)
            overlap_adj = trans_dur if clip_decisions and trans_dur > 0 else 0.0
            seg_dur = min(seg_dur, remaining + overlap_adj)
            if seg_dur < 0.1:
                break

            # Audio energy at this position for matching
            abs_time = timeline_pos + audio_start
            beat_energy = self._get_audio_energy_at(energy_curve, abs_time)

            # ── Tier-1: Consume next alignment placement near timeline_pos ─
            clip_id: str | None = None
            src_start: float = 0.0
            energy_score: float = 0.5

            # Advance past stale placements
            while pq_idx < len(placement_queue) and placement_queue[pq_idx][0] < timeline_pos - clip_dur:
                pq_idx += 1

            if pq_idx < len(placement_queue):
                rel_beat, placement = placement_queue[pq_idx]
                if abs(rel_beat - timeline_pos) < clip_dur:
                    cid = placement.clip_id
                    skip = cid == last_clip_id and len(all_clip_ids) > 1
                    if not skip and _is_available(cid):
                        clip_id = cid
                        max_src = clip_durations.get(cid, seg_dur)
                        src_start = placement.trim_start
                        seg_dur = min(seg_dur, max_src)
                        if src_start + seg_dur > max_src:
                            src_start = max(0.0, max_src - seg_dur)
                        energy_score = placement.energy_match_score
                    pq_idx += 1

            # ── Tier-2: Best-segment energy matching ─────────────────
            if clip_id is None:
                best_match: tuple[str, float, float, float] | None = None
                best_score = -1.0
                for cid in all_clip_ids:
                    if cid == last_clip_id and len(all_clip_ids) > 1:
                        continue
                    if not _is_available(cid):
                        continue
                    max_src = clip_durations.get(cid, seg_dur)
                    if max_src < 0.1:
                        continue
                    result = self._find_best_segment_for_energy(
                        clip_segments.get(cid, []),
                        beat_energy,
                        min(seg_dur, max_src),
                        max_src,
                        used_src_ranges.get(cid, []),
                    )
                    if result and result[2] > best_score:
                        best_match = (cid, result[0], result[1], result[2])
                        best_score = result[2]

                if best_match:
                    clip_id = best_match[0]
                    src_start = best_match[1]
                    max_src = clip_durations.get(clip_id, seg_dur)
                    seg_dur = min(seg_dur, max_src)
                    if src_start + seg_dur > max_src:
                        src_start = max(0.0, max_src - seg_dur)
                    energy_score = best_match[3]

            # ── Tier-3: Round-robin with stride ──────────────────────
            if clip_id is None:
                for _ in range(len(all_clip_ids)):
                    cid = all_clip_ids[robin_idx % len(all_clip_ids)]
                    robin_idx += 1
                    if cid == last_clip_id and len(all_clip_ids) > 1:
                        continue
                    if _is_available(cid):
                        clip_id = cid
                        break
                if clip_id is None:
                    counts = {cid: _reuse_count(cid) for cid in all_clip_ids}
                    for cid in sorted(counts, key=counts.get):
                        if cid != last_clip_id or len(all_clip_ids) == 1:
                            clip_id = cid
                            break
                    if clip_id is None:
                        clip_id = min(counts, key=counts.get)

                max_src = clip_durations.get(clip_id, seg_dur)
                seg_dur = min(seg_dur, max_src)
                reuse_count = _reuse_count(clip_id)
                stride = max(seg_dur, max_src / max(1, int(actual_target / clip_dur)))
                src_start = (reuse_count * stride) % max_src
                if src_start + seg_dur > max_src:
                    src_start = max(0.0, max_src - seg_dur)
                energy_score = 0.5

            if seg_dur < 0.1:
                continue

            try:
                cd = ClipDecision(
                    clip_id=clip_id,
                    source_start=round(src_start, 4),
                    source_end=round(src_start + seg_dur, 4),
                    timeline_start=round(timeline_pos, 4),
                    timeline_end=round(timeline_pos + seg_dur, 4),
                    transition_type=trans_type,
                    transition_duration=trans_dur,
                    energy_match_score=round(energy_score, 4),
                )
                clip_decisions.append(cd)
                if len(clip_decisions) > 1 and trans_dur > 0:
                    xfade_overlap += trans_dur
                timeline_pos += seg_dur
                last_clip_id = clip_id
                used_src_ranges.setdefault(clip_id, []).append(
                    (src_start, src_start + seg_dur)
                )
            except Exception:
                robin_idx += 1
                if safety > len(all_clip_ids) * 200:
                    break
                continue

        total_duration = round(timeline_pos - xfade_overlap, 4) if clip_decisions else actual_target

        return EditDecisionList(
            clip_decisions=clip_decisions,
            audio_decision=AudioDecision(
                trim_start=audio_start,
                trim_end=audio_end,
                fade_in=min(0.3, total_duration * 0.03),
                fade_out=min(0.5, total_duration * 0.05),
            ),
            total_duration=total_duration,
        )

    def apply_user_preferences(
        self, edl: EditDecisionList, prefs: UserPreferences
    ) -> EditDecisionList:
        """Apply transition type preference.

        Duration/pacing are already handled by _create_fallback_edl, so this
        method only enforces transition type and trims to target if needed.

        For ai_mix, per-clip transition types are preserved from the
        LLM/fallback instead of being overridden.
        """
        if not edl.clip_decisions:
            return edl

        is_ai_mix = prefs.transition_type == "ai_mix"
        adjusted = []
        timeline_pos = 0.0
        xfade_overlap = 0.0
        target = prefs.target_duration or edl.total_duration

        for cd in edl.clip_decisions:
            if (timeline_pos - xfade_overlap) >= target - 0.05:
                break

            # For ai_mix, preserve the per-clip transition type;
            # otherwise override with the user preference
            if is_ai_mix:
                trans_type = cd.transition_type
                trans_dur = cd.transition_duration if cd.transition_duration > 0 else (
                    _DEFAULT_TRANSITION_DURATION if trans_type != "cut" else 0.0
                )
            else:
                trans_type = prefs.transition_type
                trans_dur = _DEFAULT_TRANSITION_DURATION if trans_type != "cut" else 0.0

            dur = cd.source_end - cd.source_start
            remaining = target - (timeline_pos - xfade_overlap)
            # Account for xfade overlap when trimming the last clip
            overlap_adj = trans_dur if adjusted and trans_dur > 0 else 0.0
            if dur > remaining + overlap_adj + 0.05:
                dur = round(remaining + overlap_adj, 4)

            try:
                new_cd = ClipDecision(
                    clip_id=cd.clip_id,
                    source_start=cd.source_start,
                    source_end=round(cd.source_start + dur, 4),
                    timeline_start=round(timeline_pos, 4),
                    timeline_end=round(timeline_pos + dur, 4),
                    transition_type=trans_type,
                    transition_duration=trans_dur,
                    energy_match_score=cd.energy_match_score,
                )
                adjusted.append(new_cd)
                if len(adjusted) > 1 and trans_dur > 0:
                    xfade_overlap += trans_dur
                timeline_pos += dur
            except Exception:
                continue

        total = round(timeline_pos - xfade_overlap, 4) if adjusted else edl.total_duration
        # trim_end must be absolute audio position, not just the timeline length
        audio_trim_end = round(edl.audio_decision.trim_start + total, 4)
        return EditDecisionList(
            clip_decisions=adjusted,
            audio_decision=edl.audio_decision.model_copy(update={"trim_end": audio_trim_end}),
            total_duration=total,
            target_fps=edl.target_fps,
            target_width=edl.target_width,
            target_height=edl.target_height,
        )
