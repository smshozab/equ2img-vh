"""Google Gemini provider (REST generateContent API)."""
from __future__ import annotations

import httpx

from llm.base import LLMError, LLMProvider

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> str:
        if not self.is_configured():
            raise LLMError("GEMINI_API_KEY is not set")
        try:
            response = httpx.post(
                f"{_API_ROOT}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 8000,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"Gemini returned HTTP {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from exc
