"""Vercel AI Gateway provider (OpenAI-compatible chat completions API)."""
from __future__ import annotations

import httpx

from llm.base import LLMError, LLMProvider


class GatewayProvider(LLMProvider):
    name = "gateway"

    def __init__(self, api_key: str | None, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> str:
        if not self.is_configured():
            raise LLMError(
                "AI_GATEWAY_API_KEY is not set. Add it to your environment or choose "
                "another provider via LLM_PROVIDER."
            )
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 8000,
                },
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"AI Gateway request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"AI Gateway returned HTTP {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected AI Gateway response shape: {data}") from exc
