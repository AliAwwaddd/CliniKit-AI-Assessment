"""Feature/target split and the column transformer shared by every model."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "no_show"
CATEGORICAL_COLUMNS = ["gender", "appointment_type", "weekday", "appointment_time"]
NUMERIC_COLUMNS = [
    "age",
    "days_before_appointment",
    "previous_appointments",
    "previous_no_shows",
]
PASSTHROUGH_COLUMNS = ["reminder_sent", "new_patient"]


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics, pass booleans through as-is."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("passthrough", "passthrough", PASSTHROUGH_COLUMNS),
        ]
    )
