# CliniKit — Machine Learning (Part 2)

Predicts whether a patient will miss an upcoming appointment (`no_show`).

## Dataset

The pipeline loads the real appointment data from `CliniKit_NoShow_Dataset.csv`
(3,000 historical appointment records with a 15.9% no-show rate). The data is
loaded by `no_show/data.py` and contains all required fields: `age`, `gender`,
`appointment_type`, `days_before_appointment`, `previous_appointments`,
`previous_no_shows`, `weekday`, `appointment_time`, `reminder_sent`, `new_patient`,
and `no_show`.

## Setup

```bash
cd no-show-predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

This loads the real appointment dataset from `dataset/CliniKit_NoShow_Dataset.csv`,
runs basic exploration, trains both models, evaluates them, saves plots to `output/`,
and prints example predictions from the best-performing model.

Tests:

```bash
pytest
```

## Architecture

```
data.py            -> loads the real appointment dataset from CSV
preprocessing.py   -> ColumnTransformer: one-hot categoricals, scale numerics
train.py           -> builds & fits Pipeline(preprocessor, classifier), saves to models/
evaluate.py        -> metrics, confusion matrix, ROC curve, feature importance -> output/
predict.py         -> example predictions on hand-crafted patient records
run.py             -> orchestrates all of the above end-to-end
```

Each model is a single `sklearn.Pipeline` (preprocessing + classifier), so
saving/loading/predicting never has to worry about applying preprocessing
steps in the right order separately — `joblib.dump`/`load` captures the
whole thing.

## Approach

**Data preparation.** Categorical columns (`gender`, `appointment_type`,
`weekday`, `appointment_time`) are one-hot encoded; numeric columns (`age`,
`days_before_appointment`, `previous_appointments`, `previous_no_shows`) are
standardized; the two already-binary columns (`reminder_sent`, `new_patient`)
pass through unchanged. All of this lives in one `ColumnTransformer`
(`preprocessing.py`) shared by every model, so there's a single place that
defines "how a raw row becomes model input."

**Model selection.** Two models, both plain scikit-learn:
- **Logistic Regression** — an interpretable baseline.
- **Random Forest** — to check whether a nonlinear model captures anything
  the baseline misses.

Both use `class_weight="balanced"`: no-shows are the minority class (~33%
here), and for a clinic a missed true no-show (a wasted slot) is costlier
than a false alarm (an unnecessary reminder), so recall on the no-show class
matters more than raw accuracy.

**Evaluation metrics.** Accuracy is misleading on an imbalanced target — a
model that always predicts "attends" would score high on it. So the primary
metrics are **ROC-AUC** (ranking quality independent of a threshold) and
**precision/recall/F1 on the no-show class specifically**, not the overall
average.

**Results** (3,000 real appointment records, 20% held-out test set):

| model | roc_auc | precision (no_show) | recall (no_show) | f1 (no_show) | accuracy |
|---|---|---|---|---|---|
| logistic_regression | 0.702 | 0.292 | 0.583 | 0.389 | 0.707 |
| random_forest | 0.689 | 0.384 | 0.292 | 0.331 | 0.812 |

Logistic Regression wins on ROC-AUC (0.702 vs 0.689) and achieves higher
recall on the no-show class (0.583 vs 0.292) — it catches more than half of
the actual no-shows, which is critical for a clinic's operational planning.
Random Forest prioritizes precision (fewer false alarms) at the cost of missing
more true no-shows. `run.py` selects logistic regression based on ROC-AUC for
the example predictions.

**Feature importance.** For Random Forest this is `feature_importances_`;
for Logistic Regression there's no such attribute, so `evaluate.py` uses
`|coefficient|` instead — both are saved as bar charts in `output/`.

**Example predictions.** `predict.py` runs four hand-crafted patients
through the winning model — a reliable long-time patient with a reminder
(low risk), a young patient with a history of no-shows and no reminder for
an evening procedure booked far out (high risk), and two in between. Run
`python run.py` to see the actual probabilities.

## Integrating this into a real software product

- **Serve it behind an API**, not embedded in application code, so the
  clinic's booking system calls it (e.g. `POST /predict` with the
  appointment's features) instead of duplicating model logic.
- **Score at booking time**, not just before the appointment — a
  high-risk score could trigger an extra reminder, a confirmation call, or
  deliberate double-booking of that slot.
- **Retrain on a schedule** as new appointment data accumulates — monitor model
  performance and refresh when drift is detected (e.g. after a policy change
  that affects the baseline no-show rate).
- **Track calibration over time**, not just accuracy at deploy time — a
  model that's well-calibrated at launch can drift as patient mix or
  scheduling policy changes.

**Cloud architecture (AWS):** this would fit AWS's ML stack well —
**S3** as the data lake for historical appointment records, **SageMaker**
for training, hyperparameter tuning, and hosting a versioned model registry,
a lightweight **SageMaker (or Lambda) endpoint** the booking flow calls
synchronously at appointment-creation time to get a risk score, and
**SageMaker Model Monitor** watching for data/prediction drift so retraining
is triggered by evidence rather than a fixed calendar schedule. This mirrors
Part 1's production architecture — same AWS account, same event-driven shape
— so a no-show risk score could realistically feed back into Part 1's agent
(e.g. flagging a high-risk booking for an extra confirmation step).

This is intentionally not built out — per the brief, this stays a
non-production exercise: a scriptable CLI pipeline with saved plots, not a
deployed service.
