"""Turns a raw patient message into a validated ExtractedRequest via Claude tool-use."""

from typing import Optional

import anthropic

from .config import ANTHROPIC_API_KEY, MODEL_NAME
from .schemas import ExtractedRequest

_TOOL_NAME = "extract_patient_request"

_SYSTEM_PROMPT = (
    "You are the intake assistant for a medical clinic's messaging channel. "
    f"Read the patient's message and extract structured information using only the "
    f"'{_TOOL_NAME}' tool. Never invent details the patient did not state — leave a "
    "field empty instead of guessing. "
    "Set explicit_confirmation to true only if the patient clearly wants the action "
    "carried out right now (e.g. 'book it', 'cancel it', 'yes please'). Hedging "
    "language such as 'don't book anything yet', 'I might want to', or 'just "
    "checking' means explicit_confirmation must stay false. "
    "Set ambiguous to true whenever the message could map to more than one intent, "
    "is vague (e.g. 'sometime next week'), or is missing information needed to act "
    "on it, and list what's missing in missing_fields."
)


def _tool_schema() -> dict:
    """Derive the Claude tool input schema from the ExtractedRequest model."""
    schema = ExtractedRequest.model_json_schema()
    schema.pop("title", None)
    return {
        "name": _TOOL_NAME,
        "description": "Structured representation of a patient's message.",
        "input_schema": schema,
    }


class ClaudeExtractor:
    """Wraps the Anthropic client so it can be swapped for a fake in tests."""

    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self._client = client or anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def extract(self, message: str) -> ExtractedRequest:
        response = self._client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_tool_schema()],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": message}],
        )
        tool_use = next(block for block in response.content if block.type == "tool_use")
        return ExtractedRequest.model_validate(tool_use.input)
