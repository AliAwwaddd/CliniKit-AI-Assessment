"""Example predictions on a few hand-crafted patient records."""

import pandas as pd
from sklearn.pipeline import Pipeline

EXAMPLE_PATIENTS = pd.DataFrame(
    [
        {
            "age": 34,
            "gender": "F",
            "appointment_type": "checkup",
            "days_before_appointment": 2,
            "previous_appointments": 10,
            "previous_no_shows": 0,
            "weekday": "Wed",
            "appointment_time": "morning",
            "reminder_sent": 1,
            "new_patient": 0,
        },
        {
            "age": 22,
            "gender": "M",
            "appointment_type": "procedure",
            "days_before_appointment": 45,
            "previous_appointments": 4,
            "previous_no_shows": 3,
            "weekday": "Fri",
            "appointment_time": "evening",
            "reminder_sent": 0,
            "new_patient": 0,
        },
        {
            "age": 70,
            "gender": "F",
            "appointment_type": "follow_up",
            "days_before_appointment": 5,
            "previous_appointments": 20,
            "previous_no_shows": 1,
            "weekday": "Tue",
            "appointment_time": "afternoon",
            "reminder_sent": 1,
            "new_patient": 0,
        },
        {
            "age": 29,
            "gender": "M",
            "appointment_type": "consultation",
            "days_before_appointment": 30,
            "previous_appointments": 0,
            "previous_no_shows": 0,
            "weekday": "Mon",
            "appointment_time": "evening",
            "reminder_sent": 0,
            "new_patient": 1,
        },
    ]
)


def predict_examples(pipeline: Pipeline, patients: pd.DataFrame = EXAMPLE_PATIENTS) -> pd.DataFrame:
    probabilities = pipeline.predict_proba(patients)[:, 1]
    predictions = pipeline.predict(patients)
    result = patients.copy()
    result["no_show_probability"] = probabilities.round(3)
    result["predicted_no_show"] = predictions
    return result
