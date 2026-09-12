# CliniKit AI Trainee Assessment — Submission

Two independent exercises, each in its own folder with its own README, setup
instructions, and tests.

## [Part 1 — Conversational / Agentic AI](conversational-agent/)

An intake assistant for a medical clinic's messaging channel: reads a patient
message, extracts structured intent + slots via the Claude API, and decides
the next action through a deterministic guardrail — mutating actions
(book/reschedule/cancel) only fire when required fields are present **and**
the patient explicitly confirmed. Ships as both a CLI and a FastAPI app.

See [conversational-agent/README.md](conversational-agent/README.md) for
setup, the approach write-up, and the production/AWS discussion.

## [Part 2 — Machine Learning](no-show-predictor/)

Predicts whether a patient will miss an appointment. No dataset came with
the assessment brief, so this generates a synthetic one matching the exact
fields described (disclosed and documented in its README), then trains and
compares a Logistic Regression baseline against a Random Forest, with
ROC-AUC/precision/recall evaluation and feature importance.

See [no-show-predictor/README.md](no-show-predictor/README.md) for setup,
real evaluation numbers, and the production/AWS discussion.

## Notes on approach

Both parts were built with AI assistance (Claude), as explicitly permitted
by the assessment brief, and I can walk through and explain every part of
the implementation.
