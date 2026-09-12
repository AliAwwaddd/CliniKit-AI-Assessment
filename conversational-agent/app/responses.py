"""Turns an action decision + its result into a patient-facing reply."""

from .schemas import ActionDecision, ActionType, ExtractedRequest


def generate_reply(extracted: ExtractedRequest, decision: ActionDecision, result: dict) -> str:
    doctor_part = f" with {extracted.doctor}" if extracted.doctor else ""

    if decision.action == ActionType.ANSWER_FAQ:
        return f"Our opening hours are: {result['opening_hours']}."

    if decision.action == ActionType.CHECK_AVAILABILITY:
        slots = ", ".join(result["available_slots"])
        if extracted.intent.value == "book_appointment":
            return (
                f"Before I book anything{doctor_part}, could you confirm you'd like me to go "
                f"ahead? Open slots around your request: {slots}."
            )
        return f"Available slots{doctor_part}: {slots}."

    if decision.action == ActionType.CREATE_APPOINTMENT:
        return f"You're booked{doctor_part} on {result['date']} at {result['time']}."

    if decision.action == ActionType.RESCHEDULE_APPOINTMENT:
        return (
            f"Your appointment{doctor_part} has been moved from {result['from_date']} to "
            f"{result['to_date']} {result['to_time'] or ''}.".strip()
        )

    if decision.action == ActionType.CANCEL_APPOINTMENT:
        return f"Your appointment{doctor_part} on {result['date']} has been cancelled."

    if decision.action == ActionType.HANDOFF_TO_HUMAN:
        return "I'm connecting you with a member of our staff who will reach out shortly."

    if decision.action == ActionType.ASK_FOR_MORE_INFORMATION:
        if extracted.missing_fields:
            missing = ", ".join(extracted.missing_fields)
            return f"Could you share a bit more detail? Specifically: {missing}."
        return "Could you clarify your request so I can help with it?"

    return "Sorry, I couldn't process that request."
