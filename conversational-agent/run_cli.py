"""CLI demo: run one message, or replay the assessment's example messages.

Usage:
    python run_cli.py "Cancel my appointment with Dr. Karim."
    python run_cli.py --examples
"""

import argparse
import json

from app.pipeline import process_message

EXAMPLE_MESSAGES = [
    "Can I see Dr. George tomorrow afternoon?",
    "Move my appointment from Monday to Wednesday.",
    "Cancel my appointment with Dr. Karim.",
    "What time does the clinic close?",
    "Do you have anything available after 5 tomorrow?",
    "I want to see my doctor again for the same problem.",
    "Book me Friday at 4 but don't confirm anything yet.",
    "I need an appointment sometime next week.",
    "Can somebody from the clinic call me?",
    "I might want to see Dr. George tomorrow at 4, but don't book anything yet.",
]


def _print_result(message: str) -> None:
    result = process_message(message)
    print(f"\nPatient: {message}")
    print(f"Extracted: {json.dumps(result.extracted.model_dump(), indent=2)}")
    print(f"Action: {result.decision.action.value} ({result.decision.reason})")
    print(f"Reply: {result.reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", help="A single patient message to process.")
    parser.add_argument(
        "--examples", action="store_true", help="Run all example messages from the assessment."
    )
    args = parser.parse_args()

    if args.examples or not args.message:
        for message in EXAMPLE_MESSAGES:
            _print_result(message)
    else:
        _print_result(args.message)


if __name__ == "__main__":
    main()
