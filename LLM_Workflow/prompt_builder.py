from __future__ import annotations

import json
from typing import Any


def build_messages(question: str, analytics_result: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "Eres un copiloto analitico para negocio. Responde en espanol, con tono ejecutivo y claro. "
        "Debes usar unicamente la evidencia estructurada entregada; no inventes metricas ni supongas datos faltantes. "
        "Si el resultado es unsupported o no_data, dilo con honestidad y explica la limitacion en una frase."
    )
    user_prompt = (
        f"Pregunta del usuario:\n{question}\n\n"
        "Resultado analitico estructurado:\n"
        f"{json.dumps(_compact_analytics_result(analytics_result), ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Devuelve una respuesta breve con dos partes: 1) respuesta principal, 2) interpretacion de negocio."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_general_chat_messages(
    question: str,
    app_context: dict[str, Any],
) -> list[dict[str, str]]:
    system_prompt = (
        "Eres un copiloto conversacional de analitica web. Responde en espanol, de forma natural, util y cercana. "
        "Puedes explicar que hace el copiloto, como usarlo, que preguntas soporta y como interpretar resultados. "
        "No inventes hechos sobre datos que no te fueron dados."
    )
    user_prompt = (
        f"Pregunta: {question}\n"
        f"Contexto breve: {_compact_app_context(app_context)}\n"
        "Responde de forma natural y practica."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
            if key != "support_table"
        }
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def _compact_analytics_result(analytics_result: dict[str, Any]) -> dict[str, Any]:
    safe = _json_safe(analytics_result)
    return {
        "status": safe.get("status"),
        "metric_name": safe.get("metric_name"),
        "filters": safe.get("filters"),
        "values": safe.get("values"),
        "answer": safe.get("answer"),
        "business_interpretation": safe.get("business_interpretation"),
        "table_preview": safe.get("table_preview", [])[:5],
        "confidence": safe.get("confidence"),
    }


def _compact_app_context(app_context: dict[str, Any]) -> str:
    capabilities = ", ".join(app_context.get("available_capabilities", []))
    dataset = app_context.get("dataset_summary", {})
    return (
        f"{app_context.get('product_name', 'Copiloto')}. "
        f"Soporta: {capabilities}. "
        f"Dataset: {dataset.get('total_sessions', 'N/A')} sesiones entre "
        f"{dataset.get('min_date', 'N/A')} y {dataset.get('max_date', 'N/A')}. "
        f"{app_context.get('token_policy', '')}"
    ).strip()
