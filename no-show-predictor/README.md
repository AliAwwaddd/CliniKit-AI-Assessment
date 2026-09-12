# CliniKit — Machine Learning (Part 2)

Predicts whether a patient will miss an upcoming appointment (`no_show`).

## A note on the data

**No dataset was provided with the assessment brief** — only the field list
(`age, gender, appointment_type, days_before_appointment, previous_appointments,
previous_no_shows, weekday, appointment_time, reminder_sent, new_patient,
no_show`). `no_show/data.py` generates a synthetic dataset using exactly
these fields, with realistic (and disclosed) relationships baked into the
target so the full pipeline — exploration, prep, training, evaluation,
feature importance — has real signal to find rather than pure noise:

- More `days_before_appointment` (longer lead time) → higher no-show risk.
- A higher historical `previous_no_shows / previous_appointments` ratio →
  higher risk (past behavior is the strongest predictor of future behavior).
- `reminder_sent` → lower risk.
- `new_patient` and `appointment_type == "procedure"` → slightly higher risk.
- Monday/Friday and evening slots → slightly higher risk.
- Gaussian noise is added on top so the target isn't a deterministic function
  of the inputs — otherwise the classification task would be trivial.

Swapping in a real dataset later only means pointing `load_dataset()` at a
different CSV with the same columns — nothing else in the pipeline changes.

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

This generates (and caches) the dataset in `dataset/`, runs basic
exploration, trains both models, evaluates them, saves plots to `output/`,
and prints example predictions from the best-performing model.

Tests:

```bash
pytest
```

## Architecture

```
data.py            -> generates/loads the dataset
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

**Results** (5,000 synthetic rows, 20% held-out test set):

| model | roc_auc | precision (no_show) | recall (no_show) | f1 (no_show) | accuracy |
|---|---|---|---|---|---|
| logistic_regression | 0.684 | 0.467 | 0.649 | 0.543 | 0.642 |
| random_forest | 0.658 | 0.473 | 0.454 | 0.463 | 0.655 |

Logistic Regression wins on ROC-AUC and, more importantly, on recall for the
no-show class — it catches noticeably more real no-shows (0.649 vs 0.454)
at a similar precision. This isn't surprising: the synthetic generator
computes `no_show` as a linear combination of the features in log-odds
space, which is exactly the relationship logistic regression is built to
fit, while the random forest's extra flexibility doesn't pay off — and can
overfit — when the true relationship is linear. `run.py` picks the model
with the higher ROC-AUC automatically rather than assuming the more complex
model wins, which is why it selects logistic regression for the example
predictions. On a real dataset, where feature/outcome relationships are
rarely this clean, I'd expect the tree ensemble to pull ahead, especially
if there are interaction effects (e.g. "new patients booking evening
procedure slots" being disproportionately risky) that a linear model can't
represent without manual feature crosses.

**Feature importance.** For Random Forest this is `feature_importances_`;
for Logistic Regression there's no such attribute, so `evaluate.py` uses
`|coefficient|` instead — both are saved as bar charts in `output/`. Both
models agree on the top drivers: `reminder_sent`, `days_before_appointment`,
`previous_no_shows`, and `age` — which lines up with how the synthetic
target was constructed, confirming the models are learning the intended
signal rather than noise.

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
- **Retrain on a schedule** as real appointment data accumulates — the
  synthetic relationships here are a stand-in; a production model needs to
  learn the clinic's actual patient behavior and be refreshed as it drifts
  (e.g. after a new reminder policy changes the baseline no-show rate).
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
