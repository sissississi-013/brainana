"""LLM interface for the autoresearch agent.

Wraps Claude API for stimulus generation, hypothesis reasoning,
and result interpretation.
"""

from __future__ import annotations

import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ANTHROPIC_API_KEY environment variable. "
                "Get one at https://console.anthropic.com/"
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_llm(
    system: str,
    user_message: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> str:
    """Call Claude and return the text response."""
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def call_llm_json(
    system: str,
    user_message: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> dict:
    """Call Claude and parse the response as JSON."""
    raw = call_llm(system, user_message, model, max_tokens, temperature)
    # Extract JSON from markdown code block if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return json.loads(raw.strip())
