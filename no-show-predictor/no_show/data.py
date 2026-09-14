"""Load real appointment dataset from CliniKit."""

from pathlib import Path

import numpy as np
import pandas as pd

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "CliniKit_NoShow_Dataset.csv"

_GENDERS = ["F", "M"]
_APPOINTMENT_TYPES = ["checkup", "follow_up", "consultation", "procedure"]
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
_APPOINTMENT_TIMES = ["morning", "afternoon", "evening"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(1, 90, size=n_rows)
    gender = rng.choice(_GENDERS, size=n_rows)
    appointment_type = rng.choice(
        _APPOINTMENT_TYPES, size=n_rows, p=[0.35, 0.30, 0.25, 0.10]
    )
    days_before_appointment = rng.integers(0, 60, size=n_rows)
    previous_appointments = rng.integers(0, 30, size=n_rows)
    # a patient can't have missed more visits than they had
    previous_no_shows = rng.binomial(previous_appointments, p=0.2)
    weekday = rng.choice(_WEEKDAYS, size=n_rows, p=[0.2, 0.18, 0.18, 0.18, 0.18, 0.08])
    appointment_time = rng.choice(_APPOINTMENT_TIMES, size=n_rows, p=[0.4, 0.4, 0.2])
    reminder_sent = rng.binomial(1, p=0.6, size=n_rows)
    new_patient = (previous_appointments == 0).astype(int)

    no_show_rate = previous_no_shows / np.maximum(previous_appointments, 1)
    is_procedure = (appointment_type == "procedure").astype(int)
    is_edge_of_week = np.isin(weekday, ["Mon", "Fri"]).astype(int)
    is_evening = (appointment_time == "evening").astype(int)

    # Weighted combination of effects, each documented in the README, plus noise
    # so the target isn't a deterministic function of the features.
    logit = (
        -1.3
        + 0.03 * days_before_appointment
        + 1.6 * no_show_rate
        - 0.8 * reminder_sent
        + 0.5 * new_patient
        + 0.3 * is_procedure
        + 0.2 * is_edge_of_week
        + 0.2 * is_evening
        - 0.01 * age
        + rng.normal(0, 0.5, size=n_rows)
    )
    no_show = rng.binomial(1, _sigmoid(logit))

    return pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "appointment_type": appointment_type,
            "days_before_appointment": days_before_appointment,
            "previous_appointments": previous_appointments,
            "previous_no_shows": previous_no_shows,
            "weekday": weekday,
            "appointment_time": appointment_time,
            "reminder_sent": reminder_sent,
            "new_patient": new_patient,
            "no_show": no_show,
        }
    )


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the real dataset from disk."""
    return pd.read_csv(path)
