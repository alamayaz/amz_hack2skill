"""Decision orchestrator — LLM-based editing decision maker."""

import logging

from openai import OpenAI

from rezaa.config import get_settings
from rezaa.models.alignment import AlignmentOutput
from rezaa.models.audio import AudioAnalysisOutput
from rezaa.models.edl import AudioDecision, ClipDecision, EditDecisionList
from rezaa.models.preferences import XFADE_TRANSITIONS, UserPreferences
from rezaa.models.video import VideoAnalysisOutput
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

    def _create_fallback_edl(
        self,
        audio_analysis: AudioAnalysisOutput,
        video_analyses: list[VideoAnalysisOutput],
        alignment: AlignmentOutput,
        prefs: UserPreferences,
    ) -> EditDecisionList:
        """Rule-based fallback EDL that fills the full target duration.

        Cycles through video clips to produce back-to-back clip decisions
        covering the entire audio (or user-specified target) duration.
        Picks the most energetic section of the audio rather than always
        starting from the beginning.
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
        # Actual reel length may be shorter if audio isn't long enough
        actual_target = round(audio_end - audio_start, 4)

        # Determine per-clip duration from pacing
        pacing_defaults = {"fast": 1.0, "medium": 2.0, "slow": 4.0}
        clip_dur = pacing_defaults.get(prefs.pacing, 2.0)

        # Build a round-robin order; prefer alignment order if available
        if alignment.placements:
            ordered_ids = []
            for p in sorted(alignment.placements, key=lambda p: p.align_to_beat):
                if not ordered_ids or ordered_ids[-1] != p.clip_id:
                    ordered_ids.append(p.clip_id)
            if not ordered_ids:
                ordered_ids = [va.clip_id for va in video_analyses]
        else:
            ordered_ids = [va.clip_id for va in video_analyses]

        # Map clip_id → max source duration
        clip_durations = {va.clip_id: va.features.duration for va in video_analyses}

        # Determine transition type per clip for ai_mix
        is_ai_mix = prefs.transition_type == "ai_mix"

        clip_decisions = []
        timeline_pos = 0.0
        xfade_overlap = 0.0
        idx = 0
        last_clip_id = None

        while (timeline_pos - xfade_overlap) < actual_target - 0.05:
            clip_id = ordered_ids[idx % len(ordered_ids)]
            idx += 1

            # Skip consecutive same clip when alternatives exist
            if clip_id == last_clip_id and len(ordered_ids) > 1:
                clip_id = ordered_ids[idx % len(ordered_ids)]
                idx += 1

            # Determine transition type for this clip (needed before seg_dur calc)
            if is_ai_mix:
                clip_index = len(clip_decisions)
                trans_type = XFADE_TRANSITIONS[clip_index % len(XFADE_TRANSITIONS)]
                trans_dur = _DEFAULT_TRANSITION_DURATION
            else:
                trans_type = prefs.transition_type
                trans_dur = _DEFAULT_TRANSITION_DURATION if trans_type != "cut" else 0.0

            remaining = actual_target - (timeline_pos - xfade_overlap)
            # For non-first clips with xfade, include the upcoming overlap
            # so seg_dur is always large enough to make net rendered progress
            overlap_adj = trans_dur if clip_decisions and trans_dur > 0 else 0.0
            seg_dur = min(clip_dur, remaining + overlap_adj)
            max_src = clip_durations.get(clip_id, seg_dur)
            seg_dur = min(seg_dur, max_src)
            if seg_dur < 0.1:
                # Clip too short to use, skip it
                if idx > len(ordered_ids) * 10:
                    break
                continue

            # Cycle source offset so repeated uses show different parts
            reuse_count = sum(1 for cd in clip_decisions if cd.clip_id == clip_id)
            src_start = (reuse_count * clip_dur) % max_src
            if src_start + seg_dur > max_src:
                src_start = max(0.0, max_src - seg_dur)

            try:
                cd = ClipDecision(
                    clip_id=clip_id,
                    source_start=round(src_start, 4),
                    source_end=round(src_start + seg_dur, 4),
                    timeline_start=round(timeline_pos, 4),
                    timeline_end=round(timeline_pos + seg_dur, 4),
                    transition_type=trans_type,
                    transition_duration=trans_dur,
                    energy_match_score=0.8,
                )
                clip_decisions.append(cd)
                if len(clip_decisions) > 1 and trans_dur > 0:
                    xfade_overlap += trans_dur
                timeline_pos += seg_dur
                last_clip_id = clip_id
            except Exception:
                if idx > len(ordered_ids) * 10:
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
        # Apply manual audio_start if provided; otherwise keep the LLM/fallback value
        audio_trim_start = edl.audio_decision.trim_start
        if prefs.audio_start is not None:
            audio_trim_start = prefs.audio_start
        audio_trim_end = round(audio_trim_start + total, 4)
        return EditDecisionList(
            clip_decisions=adjusted,
            audio_decision=edl.audio_decision.model_copy(
                update={"trim_start": audio_trim_start, "trim_end": audio_trim_end}
            ),
            total_duration=total,
            target_fps=edl.target_fps,
            target_width=edl.target_width,
            target_height=edl.target_height,
        )
