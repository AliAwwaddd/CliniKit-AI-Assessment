"""Deterministic guardrail: decides the next action from extracted slots.

Kept separate from the LLM on purpose — the model proposes structured data,
this code decides what's safe to execute. A mutating action (create/reschedule/
cancel) only fires when required fields are present AND the patient explicitly
confirmed. This is what stops "don't book anything yet" from booking anything.
"""

from .schemas import ActionDecision, ActionType, ExtractedRequest, Intent

_MUTATING_INTENTS = {
    Intent.BOOK_APPOINTMENT,
    Intent.RESCHEDULE_APPOINTMENT,
    Intent.CANCEL_APPOINTMENT,
}


def _required_fields_present(extracted: ExtractedRequest) -> bool:
    """Code-level check, independent of what the model self-reports as missing."""
    if extracted.intent == Intent.BOOK_APPOINTMENT:
        return bool(extracted.preferred_date and extracted.preferred_time)
    if extracted.intent == Intent.RESCHEDULE_APPOINTMENT:
        has_target = bool(extracted.doctor or extracted.original_date)
        return bool(has_target and extracted.preferred_date)
    if extracted.intent == Intent.CANCEL_APPOINTMENT:
        return bool(extracted.doctor or extracted.original_date or extracted.preferred_date)
    return True


def decide_action(extracted: ExtractedRequest) -> ActionDecision:
    intent = extracted.intent

    if intent == Intent.REQUEST_HUMAN:
        return ActionDecision(
            action=ActionType.HANDOFF_TO_HUMAN, reason="Patient explicitly asked for a human."
        )

    if intent == Intent.ASK_OPENING_HOURS:
        return ActionDecision(
            action=ActionType.ANSWER_FAQ, reason="Opening-hours question answered directly."
        )

    if intent == Intent.ASK_DOCTOR_AVAILABILITY:
        return ActionDecision(
            action=ActionType.CHECK_AVAILABILITY,
            reason="Patient is asking about availability, not booking.",
        )

    if intent == Intent.UNCLEAR:
        return ActionDecision(
            action=ActionType.ASK_FOR_MORE_INFORMATION, reason="Intent could not be determined."
        )

    if intent in _MUTATING_INTENTS:
        if not _required_fields_present(extracted):
            return ActionDecision(
                action=ActionType.ASK_FOR_MORE_INFORMATION,
                reason="Missing information required to act safely.",
            )

        if intent == Intent.CANCEL_APPOINTMENT:
            if extracted.ambiguous and not extracted.explicit_confirmation:
                return ActionDecision(
                    action=ActionType.ASK_FOR_MORE_INFORMATION,
                    reason="Cancellation not clearly confirmed.",
                )
            return ActionDecision(
                action=ActionType.CANCEL_APPOINTMENT, reason="Enough detail to cancel."
            )

        # book_appointment / reschedule_appointment
        if not extracted.explicit_confirmation:
            return ActionDecision(
                action=ActionType.CHECK_AVAILABILITY,
                reason="Patient has not confirmed yet — checking availability only.",
            )

        final_action = (
            ActionType.CREATE_APPOINTMENT
            if intent == Intent.BOOK_APPOINTMENT
            else ActionType.RESCHEDULE_APPOINTMENT
        )
        return ActionDecision(
            action=final_action, reason="Patient confirmed and all required details are present."
        )

    return ActionDecision(
        action=ActionType.ASK_FOR_MORE_INFORMATION,
        reason="Fallback: could not confidently determine next step.",
    )
