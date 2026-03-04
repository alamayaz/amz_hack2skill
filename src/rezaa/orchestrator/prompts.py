"""Prompt templates for LLM orchestration."""

import json

from rezaa.models.preferences import TRANSITION_TYPES

SYSTEM_PROMPT = (
    "You are an expert video editor AI. You receive structured "
    "analysis data about audio beats, video clips, and an alignment "
    "suggestion. Your job is to create an optimal Edit Decision List "
    "(EDL) that synchronizes video clips to music beats.\n"
    "\n"
    "Rules:\n"
    "1. Each clip decision must have non-overlapping timeline positions\n"
    "2. No two consecutive clips should use the same source clip\n"
    "3. Source duration must match timeline duration within 0.1s tolerance\n"
    "4. High-energy beats should use high-energy clips\n"
    "5. Apply user preferences for pacing, style, and transitions\n"
    "6. You MUST fill the entire target_duration with clips — no gaps, no short EDLs. "
    "Add more clips until the timeline reaches the target.\n"
    "7. When choosing transitions, use a variety of types for visual interest. "
    "Match transition style to the mood: use fadeblack/fadewhite for calm moments, "
    "wipeleft/wiperight for energetic sections, dissolve for emotional transitions, "
    "and cuts for fast-paced beats.\n"
    "\n"
    "Respond with ONLY valid JSON matching the provided schema."
)


def build_json_schema() -> dict:
    """Build the JSON schema for expected LLM output."""
    return {
        "type": "object",
        "required": ["clip_decisions", "total_duration"],
        "properties": {
            "clip_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "clip_id",
                        "source_start",
                        "source_end",
                        "timeline_start",
                        "timeline_end",
                        "transition_type",
                    ],
                    "properties": {
                        "clip_id": {"type": "string"},
                        "source_start": {"type": "number", "minimum": 0},
                        "source_end": {"type": "number", "minimum": 0},
                        "timeline_start": {"type": "number", "minimum": 0},
                        "timeline_end": {"type": "number", "minimum": 0},
                        "transition_type": {
                            "type": "string",
                            "enum": list(TRANSITION_TYPES),
                        },
                        "transition_duration": {"type": "number", "minimum": 0},
                        "energy_match_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
            },
            "total_duration": {"type": "number", "minimum": 0},
        },
    }


def build_orchestration_prompt(
    audio_analysis: dict,
    video_analyses: list[dict],
    alignment: dict,
    user_preferences: dict | None = None,
) -> str:
    """Build the orchestration prompt for the LLM."""
    prompt = f"""Create an Edit Decision List (EDL) from the following analysis data.

## Audio Analysis
```json
{json.dumps(audio_analysis, indent=2)}
```

## Video Clips ({len(video_analyses)} clips)
```json
{json.dumps(video_analyses, indent=2)}
```

## Alignment Suggestion
```json
{json.dumps(alignment, indent=2)}
```
"""
    if user_preferences:
        prompt += f"""
## User Preferences
```json
{json.dumps(user_preferences, indent=2)}
```
"""
        td = user_preferences.get("target_duration")
        if td:
            prompt += (
                f"\n**CRITICAL: The reel MUST be exactly {td} seconds long. "
                f"Keep adding clips until the timeline reaches {td}s. "
                f"Do NOT stop early.**\n"
            )

    prompt += f"""
## Output Schema
```json
{json.dumps(build_json_schema(), indent=2)}
```

Generate a valid EDL JSON that synchronizes the video clips to the music beats. Ensure:
- Timeline positions are chronological and non-overlapping
- No consecutive clips use the same clip_id
- Source durations match timeline durations within 0.1s
- Apply user preferences for pacing and transitions"""

    # When ai_mix is requested, tell the LLM to pick diverse transitions
    if user_preferences and user_preferences.get("transition_type") == "ai_mix":
        prompt += (
            "\n- IMPORTANT: Pick the best transition type for each clip from the "
            "available types. Use a diverse mix of transitions that match the mood "
            "and energy of each clip transition point."
        )

    return prompt
