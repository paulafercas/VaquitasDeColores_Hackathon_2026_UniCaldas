from __future__ import annotations

from typing import Any

from analytics_core.query_service import QueryService
from LLM_Workflow.gemini_client import GeminiResponseClient
from LLM_Workflow.intent_router import route
from LLM_Workflow.prompt_builder import build_messages


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
            "used_fallback": llm_output["status"] != "ok",
        }


def _fallback_text(structured_result: dict[str, Any]) -> str:
    return (
        f"{structured_result.get('answer', 'No pude generar una respuesta.')}\n\n"
        f"Interpretacion: {structured_result.get('business_interpretation', 'Sin interpretacion disponible.')}"
    )
