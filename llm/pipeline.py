"""Generation pipeline: prompt -> LLM -> validate -> render -> repair loop."""
from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.figure import Figure

from config.settings import Settings, get_settings
from export.exporter import scene_to_json
from llm.base import LLMError, LLMProvider, extract_json
from llm.gateway import GatewayProvider
from llm.gemini import GeminiProvider
from llm.openai import OpenAIProvider
from llm.prompts import SYSTEM_PROMPT, build_repair_prompt
from math_engine.validator import validate_scene, validate_scene_dict
from renderer.base import RenderReport, render_scene
from schema.scene import Scene


def make_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.llm_provider == "gemini":
        return GeminiProvider(s.gemini_api_key, s.gemini_model)
    if s.llm_provider == "openai":
        return OpenAIProvider(s.openai_api_key, s.openai_model)
    return GatewayProvider(s.gateway_api_key, s.gateway_base_url, s.gateway_model)


@dataclass
class AttemptLog:
    attempt: int
    stage: str  # "llm" | "schema" | "expressions" | "render" | "success"
    detail: str


@dataclass
class PipelineResult:
    success: bool
    scene: Scene | None = None
    figure: Figure | None = None
    render_report: RenderReport | None = None
    attempts: list[AttemptLog] = field(default_factory=list)
    error: str | None = None


def render_validated(scene: Scene) -> tuple[Figure | None, RenderReport | None, str | None]:
    """Validate expressions and render a Scene. No LLM involved.

    Returns (figure, report, error_text). error_text is None on success.
    """
    validation = validate_scene(scene)
    if not validation.valid:
        return None, None, validation.error_text()
    fig, report = render_scene(scene)
    if not report.success:
        err = report.error_text() or "; ".join(report.warnings) or "Rendering produced no output"
        return fig, report, err
    return fig, report, None


def generate(prompt: str, provider: LLMProvider | None = None,
             max_attempts: int | None = None) -> PipelineResult:
    """Full pipeline with automatic repair loop (max 3 retries by default)."""
    settings = get_settings()
    provider = provider or make_provider(settings)
    max_attempts = max_attempts or settings.max_repair_attempts

    result = PipelineResult(success=False)
    user_message = prompt
    last_scene_json = ""

    for attempt in range(1, max_attempts + 1):
        # --- 1. LLM call -----------------------------------------------------
        try:
            raw = provider.complete(SYSTEM_PROMPT, user_message)
            data = extract_json(raw)
        except LLMError as exc:
            result.attempts.append(AttemptLog(attempt, "llm", str(exc)))
            result.error = str(exc)
            # LLM/transport errors are unlikely to be fixed by a repair prompt
            if "not set" in str(exc):
                return result
            user_message = prompt  # plain retry
            continue

        # --- 2. Schema validation -------------------------------------------
        scene, schema_error = validate_scene_dict(data)
        if scene is None:
            result.attempts.append(AttemptLog(attempt, "schema", schema_error or "unknown"))
            import json as _json

            last_scene_json = _json.dumps(data, indent=2)[:6000]
            user_message = build_repair_prompt(prompt, last_scene_json, schema_error or "")
            result.error = schema_error
            continue

        last_scene_json = scene_to_json(scene)[:6000]

        # --- 3. Expression validation + render -------------------------------
        fig, report, error_text = render_validated(scene)
        if error_text is None:
            result.success = True
            result.scene = scene
            result.figure = fig
            result.render_report = report
            result.error = None
            result.attempts.append(AttemptLog(attempt, "success", "Scene validated and rendered"))
            return result

        stage = "render" if report is not None else "expressions"
        result.attempts.append(AttemptLog(attempt, stage, error_text))
        result.error = error_text
        # keep last (possibly partial) artifacts so the UI can show context
        result.scene = scene
        result.figure = fig
        result.render_report = report
        user_message = build_repair_prompt(prompt, last_scene_json, error_text)

    result.success = False
    if result.error is None:
        result.error = "Unable to generate a valid mathematical representation"
    return result
