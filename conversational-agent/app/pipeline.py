"""Orchestrates one end-to-end turn: extract -> decide -> execute -> reply.

Shared by the FastAPI app and the CLI so both entry points stay thin.
"""

from typing import Optional

from . import actions
from .claude_client import ClaudeExtractor
from .dispatcher import decide_action
from .responses import generate_reply
from .schemas import ActionDecision, ActionType, ExtractedRequest, PipelineResult

_ACTIONS_WITHOUT_ARGS = {
    ActionType.HANDOFF_TO_HUMAN: lambda extracted: actions.handoff_to_human(),
    ActionType.ANSWER_FAQ: lambda extracted: actions.answer_faq(),
    ActionType.ASK_FOR_MORE_INFORMATION: lambda extracted: actions.ask_for_more_information(
        extracted.missing_fields
    ),
    ActionType.CHECK_AVAILABILITY: lambda extracted: actions.check_availability(
        extracted.doctor, extracted.preferred_date, extracted.preferred_time
    ),
    ActionType.CREATE_APPOINTMENT: lambda extracted: actions.create_appointment(
        extracted.doctor, extracted.preferred_date, extracted.preferred_time
    ),
    ActionType.RESCHEDULE_APPOINTMENT: lambda extracted: actions.reschedule_appointment(
        extracted.doctor, extracted.original_date, extracted.preferred_date, extracted.preferred_time
    ),
    ActionType.CANCEL_APPOINTMENT: lambda extracted: actions.cancel_appointment(
        extracted.doctor, extracted.original_date or extracted.preferred_date
    ),
}


def execute_action(decision: ActionDecision, extracted: ExtractedRequest) -> dict:
    return _ACTIONS_WITHOUT_ARGS[decision.action](extracted)


def process_message(message: str, extractor: Optional[ClaudeExtractor] = None) -> PipelineResult:
    extractor = extractor or ClaudeExtractor()
    extracted = extractor.extract(message)
    decision = decide_action(extracted)
    result = execute_action(decision, extracted)
    reply = generate_reply(extracted, decision, result)
    return PipelineResult(
        message=message, extracted=extracted, decision=decision, action_result=result, reply=reply
    )
