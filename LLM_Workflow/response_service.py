from __future__ import annotations

import unicodedata
from typing import Any

from analytics_core.query_service import QueryService
from LLM_Workflow.gemini_client import GeminiResponseClient
from LLM_Workflow.intent_router import route
from LLM_Workflow.prompt_builder import (
    build_general_chat_messages,
    build_messages,
)


class ResponseService:
    def __init__(
        self,
        query_service: QueryService | None = None,
        llm_client: GeminiResponseClient | None = None,
    ) -> None:
        self.query_service = query_service or QueryService.from_default_paths()
        self.llm_client = llm_client or GeminiResponseClient(self.query_service.settings)

    def generate_answer(
        self, question: str, analytics_result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        route_result = route(question, self.query_service.bundle)
        general_chat_mode = _should_use_general_chat(question, route_result)

        if general_chat_mode:
            structured_result = analytics_result or {
                "status": "general_chat",
                "metric_name": "Conversacion general",
                "filters": {},
                "values": {},
                "answer": "",
                "business_interpretation": "",
                "support_table": None,
                "table_preview": [],
                "confidence": route_result.get("confidence", 0.0),
            }
            llm_output = self.llm_client.generate_text(
                build_general_chat_messages(
                    question,
                    _build_app_context(self.query_service),
                )
            )
            final_answer = (
                llm_output["text"].strip()
                if llm_output["status"] == "ok"
                else _general_chat_fallback(llm_output)
            )
        else:
            structured_result = analytics_result or self.query_service.answer_structured_query(
                question=question,
                routed_intent=route_result,
            )
            llm_output = self.llm_client.generate_text(
                build_messages(question, structured_result)
            )
            final_answer = (
                llm_output["text"].strip()
                if llm_output["status"] == "ok"
                else _fallback_text(structured_result)
            )

        return {
            "question": question,
            "route": route_result,
            "analytics_result": structured_result,
            "final_answer": final_answer,
            "llm_status": llm_output["status"],
            "model": llm_output["model"],
            "llm_error_type": llm_output.get("error_type"),
            "llm_error_message": llm_output.get("error_message"),
            "llm_retry_delay_seconds": llm_output.get("retry_delay_seconds"),
            "used_fallback": llm_output["status"] != "ok",
            "chat_mode": "general" if general_chat_mode else "analytics",
        }


def _fallback_text(structured_result: dict[str, Any]) -> str:
    return (
        f"{structured_result.get('answer', 'No pude generar una respuesta.')}\n\n"
        f"Interpretacion: {structured_result.get('business_interpretation', 'Sin interpretacion disponible.')}"
    )


def _general_chat_fallback(llm_output: dict[str, Any]) -> str:
    if llm_output.get("status") == "quota_exhausted":
        return (
            "Ahora mismo Gemini no puede responder porque la cuota del proyecto esta agotada. "
            "Puedes seguir usando preguntas analiticas estructuradas mientras se restablece la cuota."
        )
    return (
        "Ahora mismo no puedo responder de forma conversacional porque Gemini no esta disponible. "
        "Si quieres, puedo seguir respondiendo preguntas analiticas estructuradas sobre el dataset."
    )


def _build_app_context(query_service: QueryService) -> dict[str, Any]:
    sessions = query_service.bundle.canonical_sessions
    return {
        "product_name": "Copiloto conversacional de analitica web",
        "available_capabilities": [
            "explicar que hace el copiloto",
            "explicar como usarlo",
            "responder preguntas sobre paginas mas visitadas",
            "responder sobre abandono, salidas, interaccion, registros e interes por servicios",
        ],
        "dataset_summary": {
            "total_sessions": int(sessions["session_id"].nunique()),
            "min_date": str(sessions["timestamp"].dt.date.min()),
            "max_date": str(sessions["timestamp"].dt.date.max()),
        },
        "token_policy": (
            "No se envia la base completa al modelo. "
            "Solo se envia contexto pequeño y resultados estructurados resumidos."
        ),
    }


def _should_use_general_chat(question: str, route_result: dict[str, Any]) -> bool:
    normalized = _normalize_text(question)
    general_terms = [
        "que es",
        "como se usa",
        "como usar",
        "para que sirve",
        "ayuda",
        "explicame",
        "explica",
        "hola",
        "buenas",
        "quien eres",
    ]
    if any(term in normalized for term in general_terms):
        return True
    return route_result.get("intent") == "unknown_or_unsupported"


def _normalize_text(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    ).strip()
