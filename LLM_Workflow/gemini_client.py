from __future__ import annotations

import re
from typing import Any

from config.settings import Settings, get_settings


class GeminiResponseClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = self._build_client()

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def _build_client(self) -> Any | None:
        if not self.settings.gemini_api_key:
            return None
        try:
            from google import genai
        except ImportError:
            return None
        return genai.Client(api_key=self.settings.gemini_api_key)

    def generate_text(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self._client:
            return {
                "status": "unconfigured",
                "text": "",
                "model": None,
                "error_type": "ConfigurationError",
                "error_message": "Gemini no esta configurado o falta la libreria google-genai.",
            }

        try:
            from google.genai import types
        except ImportError:
            return {
                "status": "unconfigured",
                "text": "",
                "model": None,
                "error_type": "ImportError",
                "error_message": "No se pudo importar google.genai.types. Instala google-genai.",
            }

        prompt = _messages_to_prompt(messages)

        try:
            response = self._client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.settings.gemini_temperature,
                ),
            )
        except Exception as exc:
            error_message = str(exc)
            error_type = type(exc).__name__
            if _is_quota_exhausted(error_message):
                return {
                    "status": "quota_exhausted",
                    "text": "",
                    "model": self.settings.gemini_model,
                    "error_type": error_type,
                    "error_message": error_message,
                    "retry_delay_seconds": _extract_retry_delay_seconds(error_message),
                }
            return {
                "status": "unavailable",
                "text": "",
                "model": self.settings.gemini_model,
                "error_type": error_type,
                "error_message": error_message,
                "retry_delay_seconds": None,
            }

        text = getattr(response, "text", "") or ""
        return {
            "status": "ok" if text.strip() else "empty",
            "text": text,
            "model": self.settings.gemini_model,
            "error_type": None,
            "error_message": None,
            "retry_delay_seconds": None,
        }


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    blocks = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks)


def _is_quota_exhausted(error_message: str) -> bool:
    normalized = error_message.lower()
    return (
        "resource_exhausted" in normalized
        or "quota exceeded" in normalized
        or "429" in normalized and "quota" in normalized
    )


def _extract_retry_delay_seconds(error_message: str) -> int | None:
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_message, flags=re.IGNORECASE)
    if match:
        return int(float(match.group(1)))
    match = re.search(r"'retrydelay':\s*'([0-9]+)s'", error_message, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None
