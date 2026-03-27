from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from analytics_core.metrics_catalog import SUPPORTED_METRICS


def render_sidebar(
    dataset_info: dict[str, Any],
    api_configured: bool,
    llm_runtime: dict[str, Any] | None = None,
) -> None:
    with st.sidebar:
        st.title("Copiloto Analitico")
        st.caption("Consultas en lenguaje natural sobre el motor de analisis.")
        st.metric("API Gemini", "Configurada" if api_configured else "No configurada")
        if llm_runtime:
            st.write("Estado actual del LLM:")
            st.write(f"- Estado: {llm_runtime.get('llm_status', 'unknown')}")
            st.write(f"- Modelo: {llm_runtime.get('model') or 'N/A'}")
            st.write(f"- Fallback local: {'Si' if llm_runtime.get('used_fallback') else 'No'}")
            if llm_runtime.get("llm_error_type") or llm_runtime.get("llm_error_message"):
                st.warning(
                    f"{llm_runtime.get('llm_error_type') or 'LLMError'}: "
                    f"{llm_runtime.get('llm_error_message') or 'Sin detalle'}"
                )
        st.write("Rango de datos:")
        st.write(
            f"{dataset_info['min_date']} a {dataset_info['max_date']} "
            f"({dataset_info['total_sessions']} sesiones)"
        )
        st.write("Preguntas sugeridas:")
        st.write("- Cual fue la pagina mas vista hoy?")
        st.write("- Cual fue la principal pagina de salida?")
        st.write("- Cual fue el flujo de navegacion mas frecuente?")
        st.write("- Compara el comportamiento por pais.")
        st.write("Metricas soportadas:")
        for item in SUPPORTED_METRICS:
            st.write(f"- {item['label']}")


def render_chat_message(message: dict[str, Any]) -> None:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])
        if role == "assistant" and message.get("used_fallback"):
            st.info(
                "Respuesta generada con fallback local del motor analitico porque Gemini no respondio correctamente."
            )
        if role == "assistant" and message.get("support_table") is not None:
            with st.expander("Como se calculo", expanded=False):
                analytics_result = message.get("analytics_result", {})
                st.json(
                    {
                        "intent": message.get("route", {}).get("intent"),
                        "metric_name": analytics_result.get("metric_name"),
                        "filters": analytics_result.get("filters"),
                        "status": analytics_result.get("status"),
                        "confidence": analytics_result.get("confidence"),
                        "llm_status": message.get("llm_status"),
                        "llm_error_type": message.get("llm_error_type"),
                        "llm_error_message": message.get("llm_error_message"),
                    }
                )
                support_table = message["support_table"]
                if isinstance(support_table, pd.DataFrame) and not support_table.empty:
                    st.dataframe(support_table, use_container_width=True)
