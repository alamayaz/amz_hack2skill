"""Decision orchestrator — LLM-based editing decision maker."""

import logging

from openai import OpenAI

from rezaa.config import get_settings
from rezaa.models.alignment import AlignmentOutput
from rezaa.models.audio import AudioAnalysisOutput
from rezaa.models.edl import AudioDecision, ClipDecision, EditDecisionList
from rezaa.models.preferences import UserPreferences
from rezaa.models.video import VideoAnalysisOutput
from rezaa.orchestrator.parser import parse_llm_response, validate_edl
from rezaa.orchestrator.prompts import SYSTEM_PROMPT, build_orchestration_prompt

logger = logging.getLogger(__name__)


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

        # Try LLM
        if self.client:
            try:
                edl = self._call_llm(audio_analysis, video_analyses, alignment, prefs)
                edl = self.apply_user_preferences(edl, prefs)
                return edl
            except Exception as e:
                logger.warning(f"LLM orchestration failed, using fallback: {e}")

        # Fallback
        edl = self._create_fallback_edl(audio_analysis, video_analyses, alignment, prefs)
        edl = self.apply_user_preferences(edl, prefs)
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
            temperature=0.3,
        )

        response_text = response.choices[0].message.content
        edl_data = parse_llm_response(response_text)
        return validate_edl(edl_data, audio_analysis.features.duration)

    def _create_fallback_edl(
        self,
        audio_analysis: AudioAnalysisOutput,
        video_analyses: list[VideoAnalysisOutput],
        alignment: AlignmentOutput,
        prefs: UserPreferences,
    ) -> EditDecisionList:
        """Rule-based fallback EDL from alignment placements."""
        if not alignment.placements:
            return EditDecisionList(
                total_duration=audio_analysis.features.duration,
                audio_decision=AudioDecision(trim_end=audio_analysis.features.duration),
            )

        clip_decisions = []
        timeline_pos = 0.0
        last_clip_id = None

        for placement in alignment.placements:
            clip_id = placement.clip_id

            # Skip if same as last (variety enforcement)
            if clip_id == last_clip_id and len(video_analyses) > 1:
                # Try to find an alternative
                alt = None
                for va in video_analyses:
                    if va.clip_id != last_clip_id:
                        alt = va
                        break
                if alt:
                    clip_id = alt.clip_id

            source_dur = placement.trim_end - placement.trim_start
            if source_dur <= 0:
                continue

            try:
                cd = ClipDecision(
                    clip_id=clip_id,
                    source_start=placement.trim_start,
                    source_end=placement.trim_end,
                    timeline_start=round(timeline_pos, 4),
                    timeline_end=round(timeline_pos + source_dur, 4),
                    transition_type=prefs.transition_type,
                    energy_match_score=placement.energy_match_score,
                )
                clip_decisions.append(cd)
                timeline_pos += source_dur
                last_clip_id = clip_id
            except Exception:
                continue

        total_duration = timeline_pos if clip_decisions else audio_analysis.features.duration

        return EditDecisionList(
            clip_decisions=clip_decisions,
            audio_decision=AudioDecision(
                trim_end=total_duration,
                fade_out=min(0.5, total_duration * 0.05),
            ),
            total_duration=round(total_duration, 4),
        )

    def apply_user_preferences(
        self, edl: EditDecisionList, prefs: UserPreferences
    ) -> EditDecisionList:
        """Apply user preferences to EDL."""
        if not edl.clip_decisions:
            return edl

        # Pacing adjustment
        pacing_limits = {
            "fast": (0.3, 1.5),
            "medium": (1.0, 3.0),
            "slow": (2.0, 8.0),
        }
        min_dur, max_dur = pacing_limits.get(prefs.pacing, (1.0, 3.0))

        adjusted_decisions = []
        timeline_pos = 0.0
        last_clip_id = None

        for cd in edl.clip_decisions:
            source_dur = cd.source_end - cd.source_start

            # Clamp duration to pacing range
            clamped_dur = max(min_dur, min(max_dur, source_dur))

            # Adjust source end if needed
            new_source_end = cd.source_start + clamped_dur

            new_clip_id = cd.clip_id
            # Ensure no consecutive same clip
            if new_clip_id == last_clip_id:
                continue

            try:
                new_cd = ClipDecision(
                    clip_id=new_clip_id,
                    source_start=cd.source_start,
                    source_end=round(new_source_end, 4),
                    timeline_start=round(timeline_pos, 4),
                    timeline_end=round(timeline_pos + clamped_dur, 4),
                    transition_type=prefs.transition_type,
                    transition_duration=cd.transition_duration,
                    energy_match_score=cd.energy_match_score,
                )
                adjusted_decisions.append(new_cd)
                timeline_pos += clamped_dur
                last_clip_id = new_clip_id
            except Exception:
                continue

        # Apply target duration if specified
        if prefs.target_duration and adjusted_decisions:
            # Trim to target duration
            trimmed = []
            for cd in adjusted_decisions:
                if cd.timeline_start >= prefs.target_duration:
                    break
                if cd.timeline_end > prefs.target_duration:
                    excess = cd.timeline_end - prefs.target_duration
                    try:
                        new_cd = ClipDecision(
                            clip_id=cd.clip_id,
                            source_start=cd.source_start,
                            source_end=round(cd.source_end - excess, 4),
                            timeline_start=cd.timeline_start,
                            timeline_end=round(prefs.target_duration, 4),
                            transition_type=cd.transition_type,
                            energy_match_score=cd.energy_match_score,
                        )
                        trimmed.append(new_cd)
                    except Exception:
                        trimmed.append(cd)
                    break
                trimmed.append(cd)
            adjusted_decisions = trimmed
            timeline_pos = prefs.target_duration

        total = round(timeline_pos, 4)
        return EditDecisionList(
            clip_decisions=adjusted_decisions,
            audio_decision=edl.audio_decision.model_copy(update={"trim_end": total}),
            total_duration=total,
            target_fps=edl.target_fps,
            target_width=edl.target_width,
            target_height=edl.target_height,
        )
