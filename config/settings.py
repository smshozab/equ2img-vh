"""Environment-based configuration (python-dotenv)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

# Load .env then any local overrides; existing env always wins.
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.development.local")


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    gateway_api_key: str | None
    gateway_base_url: str
    gateway_model: str
    gemini_api_key: str | None
    gemini_model: str
    openai_api_key: str | None
    openai_model: str
    openrouter_api_key: str | None
    openrouter_model: str
    max_repair_attempts: int


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openrouter" if os.getenv("OPENROUTER_API_KEY") else ("gemini" if os.getenv("GEMINI_API_KEY") else "gateway")).lower(),
        gateway_api_key=os.getenv("AI_GATEWAY_API_KEY"),
        gateway_base_url=os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1"),
        gateway_model=os.getenv("AI_GATEWAY_MODEL", "openai/gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"),
        max_repair_attempts=int(os.getenv("MAX_REPAIR_ATTEMPTS", "3")),
    )
