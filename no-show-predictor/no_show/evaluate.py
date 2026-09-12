"""Metrics, confusion matrix, ROC curve, and feature importance reporting."""

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # write plots to disk, no display server needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "name": name,
        "roc_auc": auc,
        "precision_no_show": report["1"]["precision"],
        "recall_no_show": report["1"]["recall"],
        "f1_no_show": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
    }


def print_comparison(results: list) -> None:
    print(pd.DataFrame(results).set_index("name").round(3).to_string())


def save_confusion_matrix(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, display_labels=["attended", "no_show"])
    plt.title(f"Confusion matrix — {name}")
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name}.png", bbox_inches="tight")
    plt.close()


def save_roc_curve(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title(f"ROC curve — {name}")
    plt.savefig(OUTPUT_DIR / f"roc_curve_{name}.png", bbox_inches="tight")
    plt.close()


def save_feature_importance(name: str, pipeline: Pipeline) -> None:
    """Tree models expose feature_importances_; linear models use |coefficient| instead."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        scores = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        scores = abs(classifier.coef_[0])
    else:
        return

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(scores, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(15)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importances.plot(kind="barh", figsize=(8, 6), title=f"Feature importance — {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"feature_importance_{name}.png", bbox_inches="tight")
    plt.close()
