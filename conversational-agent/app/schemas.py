"""Data contracts shared across the pipeline."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    BOOK_APPOINTMENT = "book_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    ASK_OPENING_HOURS = "ask_opening_hours"
    ASK_DOCTOR_AVAILABILITY = "ask_doctor_availability"
    REQUEST_HUMAN = "request_human"
    UNCLEAR = "unclear"


class ExtractedRequest(BaseModel):
    """Structured slots extracted from a raw patient message."""

    intent: Intent
    doctor: Optional[str] = Field(default=None, description="Doctor name if mentioned.")
    preferred_date: Optional[str] = Field(
        default=None, description="New/requested date, as the patient phrased it."
    )
    preferred_time: Optional[str] = Field(
        default=None, description="New/requested time, as the patient phrased it."
    )
    original_date: Optional[str] = Field(
        default=None,
        description="For reschedule/cancel: the existing appointment's date, if stated.",
    )
    reason: Optional[str] = Field(default=None, description="Reason for the visit, if stated.")
    explicit_confirmation: bool = Field(
        default=False,
        description="True only if the patient clearly wants the action executed now.",
    )
    ambiguous: bool = Field(
        default=False,
        description="True if the message maps to more than one intent or is missing key info.",
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Fields still needed to act on this request."
    )


class ActionType(str, Enum):
    CHECK_AVAILABILITY = "check_availability"
    CREATE_APPOINTMENT = "create_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    HANDOFF_TO_HUMAN = "handoff_to_human"
    ASK_FOR_MORE_INFORMATION = "ask_for_more_information"
    ANSWER_FAQ = "answer_faq"


class ActionDecision(BaseModel):
    action: ActionType
    reason: str  # short internal note, useful for debugging/audit


class PipelineResult(BaseModel):
    message: str
    extracted: ExtractedRequest
    decision: ActionDecision
    action_result: dict
    reply: str
