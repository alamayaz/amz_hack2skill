"""LLM response parser for EDL extraction."""

import json
import re

from rezaa.models.edl import ClipDecision, EditDecisionList
from rezaa.models.errors import ProcessingError


def parse_llm_response(response_text: str) -> dict:
    """Parse LLM response, handling markdown-wrapped JSON."""
    text = response_text.strip()

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON object in text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        raise ProcessingError(
            f"Failed to parse LLM response as JSON: {e}",
            component="orchestrator",
            details={"response_preview": text[:200]},
        )


def validate_edl(edl_data: dict, expected_duration: float) -> EditDecisionList:
    """Validate and construct an EDL from parsed data."""
    clip_decisions = []
    for cd_data in edl_data.get("clip_decisions", []):
        try:
            cd = ClipDecision(
                clip_id=cd_data["clip_id"],
                source_start=cd_data["source_start"],
                source_end=cd_data["source_end"],
                timeline_start=cd_data["timeline_start"],
                timeline_end=cd_data["timeline_end"],
                transition_type=cd_data.get("transition_type", "cut"),
                transition_duration=cd_data.get("transition_duration", 0.0),
                energy_match_score=cd_data.get("energy_match_score", 0.0),
            )
            clip_decisions.append(cd)
        except Exception:
            continue  # Skip invalid decisions

    total_duration = edl_data.get("total_duration", expected_duration)

    return EditDecisionList(
        clip_decisions=clip_decisions,
        total_duration=total_duration,
    )
