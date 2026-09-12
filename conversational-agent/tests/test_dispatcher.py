"""Unit tests for the guardrail logic. No API key needed — ExtractedRequest is built by hand."""

from app.dispatcher import decide_action
from app.schemas import ActionType, ExtractedRequest, Intent


def _request(**overrides) -> ExtractedRequest:
    defaults = dict(intent=Intent.UNCLEAR)
    defaults.update(overrides)
    return ExtractedRequest(**defaults)


def test_confirmed_booking_with_all_fields_creates_appointment():
    extracted = _request(
        intent=Intent.BOOK_APPOINTMENT,
        doctor="Dr. George",
        preferred_date="tomorrow",
        preferred_time="afternoon",
        explicit_confirmation=True,
    )
    assert decide_action(extracted).action == ActionType.CREATE_APPOINTMENT


def test_hedged_booking_does_not_create_appointment():
    """The assessment's ambiguous case: full details, but no confirmation."""
    extracted = _request(
        intent=Intent.BOOK_APPOINTMENT,
        doctor="Dr. George",
        preferred_date="tomorrow",
        preferred_time="4 PM",
        explicit_confirmation=False,
    )
    decision = decide_action(extracted)
    assert decision.action == ActionType.CHECK_AVAILABILITY


def test_booking_missing_fields_asks_for_more_information():
    extracted = _request(
        intent=Intent.BOOK_APPOINTMENT,
        explicit_confirmation=True,
        missing_fields=["preferred_date", "preferred_time"],
    )
    assert decide_action(extracted).action == ActionType.ASK_FOR_MORE_INFORMATION


def test_confirmed_reschedule_with_all_fields_reschedules():
    extracted = _request(
        intent=Intent.RESCHEDULE_APPOINTMENT,
        original_date="Monday",
        preferred_date="Wednesday",
        explicit_confirmation=True,
    )
    assert decide_action(extracted).action == ActionType.RESCHEDULE_APPOINTMENT


def test_reschedule_missing_target_asks_for_more_information():
    extracted = _request(intent=Intent.RESCHEDULE_APPOINTMENT, preferred_date="Wednesday")
    assert decide_action(extracted).action == ActionType.ASK_FOR_MORE_INFORMATION


def test_cancel_with_doctor_cancels():
    extracted = _request(intent=Intent.CANCEL_APPOINTMENT, doctor="Dr. Karim")
    assert decide_action(extracted).action == ActionType.CANCEL_APPOINTMENT


def test_ambiguous_unconfirmed_cancel_asks_for_more_information():
    extracted = _request(
        intent=Intent.CANCEL_APPOINTMENT,
        doctor="Dr. Karim",
        ambiguous=True,
        explicit_confirmation=False,
    )
    assert decide_action(extracted).action == ActionType.ASK_FOR_MORE_INFORMATION


def test_ask_opening_hours_answers_faq():
    extracted = _request(intent=Intent.ASK_OPENING_HOURS)
    assert decide_action(extracted).action == ActionType.ANSWER_FAQ


def test_ask_doctor_availability_checks_availability():
    extracted = _request(intent=Intent.ASK_DOCTOR_AVAILABILITY, preferred_time="after 5 tomorrow")
    assert decide_action(extracted).action == ActionType.CHECK_AVAILABILITY


def test_request_human_hands_off():
    extracted = _request(intent=Intent.REQUEST_HUMAN)
    assert decide_action(extracted).action == ActionType.HANDOFF_TO_HUMAN


def test_unclear_intent_asks_for_more_information():
    extracted = _request(intent=Intent.UNCLEAR)
    assert decide_action(extracted).action == ActionType.ASK_FOR_MORE_INFORMATION
