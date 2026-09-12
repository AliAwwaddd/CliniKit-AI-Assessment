# CliniKit — Conversational / Agentic AI (Part 1)

An intake assistant for a medical clinic's messaging channel. It reads a
patient message, extracts structured intent + slots via Claude, and decides
the next action (mocked — no real scheduling backend).

## Setup

```bash
cd conversational-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
```

## Run

CLI — single message:

```bash
python run_cli.py "Cancel my appointment with Dr. Karim."
```

CLI — replay all example messages from the assessment brief:

```bash
python run_cli.py --examples
```

API:

```bash
uvicorn app.main:app --reload
curl -X POST localhost:8000/message -H "Content-Type: application/json" \
  -d '{"message": "Can I see Dr. George tomorrow afternoon?"}'
```

Tests:

```bash
pytest                     # dispatcher unit tests — no API key needed
ANTHROPIC_API_KEY=sk-... pytest tests/test_examples.py   # live integration check
```

## Architecture

```
message -> ClaudeExtractor -> ExtractedRequest -> decide_action() -> execute_action() -> generate_reply()
              (app/claude_client.py)   (app/schemas.py)  (app/dispatcher.py)  (app/actions.py)  (app/responses.py)
```

Each stage is a single-responsibility module, wired together in `app/pipeline.py`
and reused by both `app/main.py` (FastAPI) and `run_cli.py`.

## Approach

**Intent identification & extraction.** One Claude tool-use call per message,
forced to call a single tool (`tool_choice`) whose input schema is generated
directly from the `ExtractedRequest` Pydantic model — so the model's output is
schema-validated automatically instead of parsing free text. It's a single
call, not a multi-step agent loop: the task is single-turn classification +
slot extraction, so an agent graph would add framework overhead without
buying anything here.

Extracted slots: `intent`, `doctor`, `preferred_date`, `preferred_time`,
`original_date` (for reschedule/cancel), `reason`, plus two guardrail signals
the model self-reports: `explicit_confirmation` and `ambiguous` (with
`missing_fields`).

**Handling missing/ambiguous information.** Two layers:
1. The model is prompted to flag vague or incomplete messages itself
   (`ambiguous`, `missing_fields`) rather than guessing at unstated details.
2. `dispatcher.py` re-checks required fields per intent in code
   (`_required_fields_present`), independent of what the model self-reports.
   If required info is missing, the action is always `ask_for_more_information`.

**Avoiding incorrect automated actions.** This is the part the assessment's
ambiguous example is testing for ("I might want to see Dr. George tomorrow at
4, but don't book anything yet.") — all slots are present here, so a naive
"do I have enough info" check would book it. The dispatcher instead treats
`explicit_confirmation` as a second, independent gate: `create_appointment`
and `reschedule_appointment` only fire when the fields are complete **and**
the patient explicitly confirmed. Without confirmation, the system falls back
to `check_availability` — useful to the patient, but non-committal. The LLM
never calls a mutating action directly; it only produces structured data,
and a plain Python function decides what's safe to execute. This keeps the
guardrail testable and auditable independent of prompt behavior — see
`tests/test_dispatcher.py`, which drives every branch (confirmed booking,
hedged booking, missing fields, ambiguous cancel, etc.) with hand-built
`ExtractedRequest` objects and no API key.

**Response generation.** Deterministic templates keyed by the chosen action
(`app/responses.py`), not a second LLM call — keeps replies predictable and
easy to test. In production this could be handed back to Claude for more
natural phrasing once the guardrail has already decided what's allowed to
happen.

## Improving this for production

- **Multi-turn state.** Today each message is handled independently. A real
  chat needs conversation memory (e.g. "yes, confirm that" as a follow-up to
  a prior `check_availability` reply) — this would call for a lightweight
  state machine (or LangGraph, which is otherwise unnecessary for this
  single-turn scope) plus a session store.
- **Real integrations.** Swap the mocked functions in `app/actions.py` for
  calls into the clinic's actual scheduling system.
- **Observability & evaluation.** Log every (message, extraction, decision)
  triple for auditing and to build a regression test set from real traffic.
- **Cloud architecture (AWS):**
  - **API Gateway + Lambda** to receive messages from the clinic's messaging
    channel and run the pipeline, instead of a long-lived FastAPI process.
  - **Bedrock (Claude)** as the managed inference layer for the extraction
    step, instead of calling the Anthropic API directly — keeps everything
    inside the AWS account/VPC and simplifies IAM-based access control.
  - **DynamoDB** to persist per-patient conversation state between messages
    (needed once multi-turn confirmation flows are supported).
  - **SQS** as the queue behind `handoff_to_human()`, so escalations are
    durably delivered to clinic staff instead of being a fire-and-forget call.

  I deliberately didn't reach for **Amazon Lex** here: Lex is itself an NLU/
  slot-filling engine, so it would compete with — not complement — Claude's
  extraction step, and its rigid slot types handle the messy, ambiguous
  phrasing in this brief (typos, hedging, vague dates) worse than an LLM does.
  Claude stays the single NLU layer; the AWS services above are just
  infrastructure around it.

This is intentionally not built out — the brief scopes this as a non-production
exercise, so the code stays a simple FastAPI/CLI app with mocked actions.
