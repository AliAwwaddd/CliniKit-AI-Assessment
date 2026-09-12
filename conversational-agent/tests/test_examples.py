"""Integration test: runs every example message through the real Claude extractor.

Skipped automatically when ANTHROPIC_API_KEY isn't set, since it makes live API calls.
"""

import os

import pytest

from app.pipeline import process_message
from run_cli import EXAMPLE_MESSAGES

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY for a live API call"
)


@pytest.mark.parametrize("message", EXAMPLE_MESSAGES)
def test_example_message_produces_a_reply(message):
    result = process_message(message)
    assert result.reply
