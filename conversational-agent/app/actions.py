"""Mocked clinic-system actions. No real scheduling backend is wired up."""

from typing import Optional


def check_availability(doctor: Optional[str], date: Optional[str], time: Optional[str]) -> dict:
    return {
        "action": "check_availability",
        "doctor": doctor,
        "date": date,
        "time": time,
        "available_slots": ["10:00", "14:30", "16:00"],
    }


def create_appointment(doctor: Optional[str], date: Optional[str], time: Optional[str]) -> dict:
    return {
        "action": "create_appointment",
        "status": "created",
        "doctor": doctor,
        "date": date,
        "time": time,
    }


def reschedule_appointment(
    doctor: Optional[str],
    original_date: Optional[str],
    new_date: Optional[str],
    new_time: Optional[str],
) -> dict:
    return {
        "action": "reschedule_appointment",
        "status": "rescheduled",
        "doctor": doctor,
        "from_date": original_date,
        "to_date": new_date,
        "to_time": new_time,
    }


def cancel_appointment(doctor: Optional[str], date: Optional[str]) -> dict:
    return {
        "action": "cancel_appointment",
        "status": "cancelled",
        "doctor": doctor,
        "date": date,
    }


def handoff_to_human() -> dict:
    return {"action": "handoff_to_human", "status": "queued_for_staff"}


def ask_for_more_information(missing_fields: list) -> dict:
    return {"action": "ask_for_more_information", "missing_fields": missing_fields}


def answer_faq() -> dict:
    return {"action": "answer_faq", "opening_hours": "Mon-Fri 9:00-18:00, Sat 9:00-13:00"}
