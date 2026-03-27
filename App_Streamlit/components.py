from __future__ import annotations

from typing import Any

import altair as alt
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
            if llm_runtime.get("llm_status") == "quota_exhausted":
                retry_delay = llm_runtime.get("llm_retry_delay_seconds")
                retry_text = (
                    f" Reintenta en aproximadamente {retry_delay} segundos."
                    if retry_delay is not None
                    else ""
                )
                st.error("Gemini agotó la cuota disponible para este proyecto." + retry_text)
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
        st.write("- Cuales son las 5 paginas mas visitadas?")
        st.write("- Cuales son las 5 paginas con mayor sesiones de salida?")
        st.write("- Cual fue el flujo de navegacion mas frecuente?")
        st.write("- En que paginas se gasto mas tiempo?")
        st.write("Metricas soportadas:")
        for item in SUPPORTED_METRICS:
            st.write(f"- {item['label']}")


def render_chat_message(message: dict[str, Any]) -> None:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])
        if role == "assistant" and message.get("llm_status") == "quota_exhausted":
            retry_delay = message.get("llm_retry_delay_seconds")
            retry_text = (
                f" Gemini reportó cuota agotada. Reintenta en aproximadamente {retry_delay} segundos."
                if retry_delay is not None
                else " Gemini reportó cuota agotada para este proyecto."
            )
            st.error(retry_text)
        elif role == "assistant" and message.get("used_fallback"):
            st.info(
                "Respuesta generada con fallback local del motor analitico porque Gemini no respondio correctamente."
            )

        if role == "assistant":
            analytics_result = message.get("analytics_result", {})
            support_table = message.get("support_table")
            if isinstance(support_table, pd.DataFrame) and not support_table.empty:
                _render_insight_chart(message.get("route", {}).get("intent"), analytics_result, support_table)
                with st.expander("Como se calculo", expanded=False):
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
                            "llm_retry_delay_seconds": message.get("llm_retry_delay_seconds"),
                        }
                    )
                    st.dataframe(support_table, use_container_width=True)


def _render_insight_chart(
    intent: str | None,
    analytics_result: dict[str, Any],
    support_table: pd.DataFrame,
) -> None:
    st.caption("Visualizacion del insight")
    chart = _build_chart(intent, analytics_result, support_table)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)


def _build_chart(
    intent: str | None,
    analytics_result: dict[str, Any],
    support_table: pd.DataFrame,
) -> alt.Chart | None:
    if support_table.empty:
        return None

    if intent in {"top_product_or_page", "page_metric_ranking", "top_exit_page", "page_abandonment_ranking"}:
        return _build_rank_chart(support_table)
    if intent == "top_navigation_flow":
        return _build_flow_chart(support_table)
    if intent == "interaction_overview":
        return _build_summary_chart(support_table)
    if intent in {"service_interest", "registration_interest"}:
        return _build_interest_chart(support_table)
    if intent == "hackathon_objective_summary":
        return _build_objective_chart(support_table)
    return _build_generic_chart(support_table, analytics_result)


def _build_rank_chart(df: pd.DataFrame) -> alt.Chart | None:
    label_col = _first_existing_column(df, ["page", "entry_page", "exit_page"])
    value_col = _first_numeric_column(df, exclude={"share_of_sessions"})
    if label_col is None or value_col is None:
        return None
    chart_df = df[[label_col, value_col]].copy().head(10)
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{value_col}:Q", title=value_col),
            y=alt.Y(f"{label_col}:N", sort="-x", title="pagina"),
            tooltip=[label_col, value_col],
        )
    )


def _build_flow_chart(df: pd.DataFrame) -> alt.Chart | None:
    label_col = _first_existing_column(df, ["flow_path"])
    value_col = _first_existing_column(df, ["sessions"])
    if label_col is None or value_col is None:
        return None
    chart_df = df[[label_col, value_col]].copy().head(10)
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{value_col}:Q", title="sesiones"),
            y=alt.Y(f"{label_col}:N", sort="-x", title="flujo"),
            tooltip=[label_col, value_col],
        )
    )


def _build_summary_chart(df: pd.DataFrame) -> alt.Chart | None:
    if not {"metric", "value"}.issubset(df.columns):
        return None
    chart_df = df.copy()
    chart_df["value"] = pd.to_numeric(chart_df["value"], errors="coerce")
    chart_df = chart_df.dropna(subset=["value"])
    if chart_df.empty:
        return None
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("value:Q", title="valor"),
            y=alt.Y("metric:N", sort="-x", title="metrica"),
            tooltip=["metric", "value"],
        )
    )


def _build_interest_chart(df: pd.DataFrame) -> alt.Chart | None:
    label_col = _first_existing_column(df, ["page"])
    value_col = _first_existing_column(df, ["sessions"])
    if label_col is None or value_col is None:
        return None
    base = alt.Chart(df.head(12))
    if "interest_type" in df.columns:
        return (
            base.mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X(f"{value_col}:Q", title="sesiones"),
                y=alt.Y(f"{label_col}:N", sort="-x", title="pagina"),
                color=alt.Color("interest_type:N", title="tipo de interes"),
                tooltip=[label_col, value_col, "interest_type"],
            )
        )
    return (
        base.mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{value_col}:Q", title="sesiones"),
            y=alt.Y(f"{label_col}:N", sort="-x", title="pagina"),
            tooltip=[label_col, value_col],
        )
    )


def _build_objective_chart(df: pd.DataFrame) -> alt.Chart | None:
    if not {"objective", "value"}.issubset(df.columns):
        return None
    chart_df = df.copy()
    chart_df["value_numeric"] = pd.to_numeric(chart_df["value"], errors="coerce")
    chart_df = chart_df.dropna(subset=["value_numeric"])
    if chart_df.empty:
        return None
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("value_numeric:Q", title="valor"),
            y=alt.Y("objective:N", sort="-x", title="insight"),
            color=alt.Color("insight_type:N", title="tipo"),
            tooltip=["objective", "value_numeric", "insight_type", "finding"],
        )
    )


def _build_generic_chart(df: pd.DataFrame, analytics_result: dict[str, Any]) -> alt.Chart | None:
    numeric_col = _first_numeric_column(df)
    label_col = _first_categorical_column(df, exclude={numeric_col} if numeric_col else set())
    if numeric_col is None or label_col is None:
        return None
    chart_df = df[[label_col, numeric_col]].copy().head(10)
    title = analytics_result.get("metric_name") or "Insight"
    return (
        alt.Chart(chart_df, title=title)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{numeric_col}:Q", title=numeric_col),
            y=alt.Y(f"{label_col}:N", sort="-x", title=label_col),
            tooltip=[label_col, numeric_col],
        )
    )


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _first_numeric_column(df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    excluded = exclude or set()
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            return column
    return None


def _first_categorical_column(df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    excluded = exclude or set()
    for column in df.columns:
        if column in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            return column
    return None
