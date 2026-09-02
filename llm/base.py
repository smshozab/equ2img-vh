"""LLM provider abstraction. Providers turn (system, user) messages into text."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Raised when an LLM call fails or returns unusable output."""


class LLMProvider(ABC):
    """Minimal chat-completion interface all providers implement."""

    name: str = "base"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the assistant's text for a system+user message pair."""

    def is_configured(self) -> bool:
        return True


def extract_json(text: str) -> dict:
    """Extract a single JSON object from LLM output.

    Tolerates markdown fences and leading/trailing prose, but requires a
    parseable object in the end.
    """
    if not text or not text.strip():
        raise LLMError("LLM returned an empty response")

    candidate = text.strip()

    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    # Direct parse first
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} block
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(candidate[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM output is not valid JSON: {exc}") from exc

    raise LLMError("LLM output does not contain a JSON object")
