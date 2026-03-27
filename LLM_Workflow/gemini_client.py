from __future__ import annotations

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
            return {
                "status": "unavailable",
                "text": "",
                "model": self.settings.gemini_model,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        text = getattr(response, "text", "") or ""
        return {
            "status": "ok" if text.strip() else "empty",
            "text": text,
            "model": self.settings.gemini_model,
            "error_type": None,
            "error_message": None,
        }


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    blocks = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks)
