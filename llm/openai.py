"""OpenAI provider (chat completions REST API)."""
from __future__ import annotations

import httpx

from llm.base import LLMError, LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> str:
        if not self.is_configured():
            raise LLMError("OPENAI_API_KEY is not set")
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 8000,
                    "response_format": {"type": "json_object"},
                },
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"OpenAI returned HTTP {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response shape: {data}") from exc


class OpenRouterProvider(OpenAIProvider):
    name = "openrouter"

    def complete(self, system: str, user: str) -> str:
        if not self.is_configured():
            raise LLMError("OPENROUTER_API_KEY is not set")
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": "http://localhost:3000", "X-Title": "Math-to-Image"},
                json={"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.4, "max_tokens": 8000, "response_format": {"type": "json_object"}},
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenRouter request failed: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenRouter response shape: {data}") from exc
