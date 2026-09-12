"""Trains the baseline and main models and persists them to disk."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, split_features_and_target

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# class_weight="balanced" matters here: no-shows are the minority class, and
# missing a real no-show (false negative) is costlier to a clinic than a
# false alarm, so we don't want the model to just default to "always attends".
_MODEL_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    ),
}


def make_pipeline(model_name: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", _MODEL_FACTORIES[model_name]()),
        ]
    )


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_features_and_target(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    pipelines = {}
    for model_name in _MODEL_FACTORIES:
        pipeline = make_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        pipelines[model_name] = pipeline
    return pipelines


def save_models(pipelines: Dict[str, Pipeline]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, pipeline in pipelines.items():
        joblib.dump(pipeline, MODELS_DIR / f"{model_name}.joblib")


def load_model(model_name: str) -> Pipeline:
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")
