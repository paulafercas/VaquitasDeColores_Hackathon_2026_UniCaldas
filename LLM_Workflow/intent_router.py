from __future__ import annotations

import re
import unicodedata
from datetime import timedelta
from typing import Any

import pandas as pd

from analytics_core.data_loader import AnalyticsBundle


def route(question: str, bundle: AnalyticsBundle) -> dict[str, Any]:
    normalized = _normalize_text(question)
    filters = _extract_filters(normalized, bundle)

    intent = "unknown_or_unsupported"
    confidence = 0.35
    interaction_terms = [
        "interaccion",
        "clic",
        "click",
        "paginas mas visitadas",
        "scroll",
        "tiempo de navegacion",
        "tiempo",
        "registros",
        "pagos",
        "precios",
        "analisis",
    ]
    registration_terms = ["registro", "registrado", "registrados", "registrarse", "registration", "register", "signup", "sign up"]
    service_terms = ["servicios", "servicio", "pricing", "price", "precio", "precios", "contact", "contacto", "demo", "request-demo", "interes", "registration", "register"]

    if (
        any(keyword in normalized for keyword in ["abandono", "rebote"])
        and any(keyword in normalized for keyword in ["pagina", "paginas", "donde", "mayor", "top"])
    ):
        intent = "page_abandonment_ranking"
        confidence = 0.9
    elif "servicio" in normalized or "servicios" in normalized:
        intent = "service_interest"
        confidence = 0.9
    elif any(keyword in normalized for keyword in service_terms) and any(
        keyword in normalized for keyword in ["pricing", "price", "precio", "precios", "contact", "contacto", "demo", "request-demo", "interes"]
    ):
        intent = "service_interest"
        confidence = 0.87
    elif any(keyword in normalized for keyword in registration_terms):
        intent = "registration_interest"
        confidence = 0.9
    elif sum(keyword in normalized for keyword in interaction_terms) >= 2:
        intent = "interaction_overview"
        confidence = 0.88
    elif any(keyword in normalized for keyword in ["flujo", "naveg", "ruta", "camino", "recorrido"]):
        intent = "top_navigation_flow"
        confidence = 0.88
    elif any(keyword in normalized for keyword in ["salida", "exit"]):
        intent = "top_exit_page"
        confidence = 0.9
    elif any(keyword in normalized for keyword in ["rebote", "abandono", "bounce", "frustr"]):
        intent = "bounce_or_abandonment"
        confidence = 0.87
    elif any(
        keyword in normalized
        for keyword in ["compar", "segment", "pais", "country", "dispositivo", "device", "browser", "navegador"]
    ):
        intent = "segment_comparison"
        confidence = 0.82
        if "dispositivo" in normalized or "device" in normalized or "mobile" in normalized or "pc" in normalized:
            filters["segment_dimension"] = "device"
        elif "browser" in normalized or "navegador" in normalized or "chrome" in normalized:
            filters["segment_dimension"] = "browser"
        else:
            filters["segment_dimension"] = "country"
    elif any(keyword in normalized for keyword in ["tendencia", "evolu", "diario", "por dia"]):
        intent = "trend_over_time"
        confidence = 0.86
    elif any(keyword in normalized for keyword in ["porcentaje", "por ciento", "%"]):
        intent = "share_of_sessions"
        confidence = 0.8
    elif any(
        keyword in normalized
        for keyword in [
            "mas visto",
            "mas vista",
            "mas visitado",
            "mas visitada",
            "top producto",
            "top pagina",
            "pagina mas vista",
            "producto mas visto",
        ]
    ):
        intent = "top_product_or_page"
        confidence = 0.84

    return {
        "intent": intent,
        "filters": filters,
        "confidence": confidence,
        "question": question,
    }


def _extract_filters(question: str, bundle: AnalyticsBundle) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    sessions = bundle.canonical_sessions
    max_date = sessions["timestamp"].dt.date.max()

    if "hoy" in question:
        filters["date_from"] = max_date
        filters["date_to"] = max_date
        filters["date_label"] = f"hoy ({max_date})"
    elif "ayer" in question:
        yesterday = max_date - timedelta(days=1)
        filters["date_from"] = yesterday
        filters["date_to"] = yesterday
        filters["date_label"] = f"ayer ({yesterday})"
    elif any(
        phrase in question
        for phrase in ["ultimos 7 dias", "ultima semana", "ultimos siete dias"]
    ):
        start_date = max_date - timedelta(days=6)
        filters["date_from"] = start_date
        filters["date_to"] = max_date
        filters["date_label"] = f"ultimos 7 dias ({start_date} a {max_date})"

    for country in _top_values(sessions["country"]):
        if _contains_value(question, country):
            filters["country"] = country
            break

    for device in _top_values(sessions["device"]):
        if _contains_value(question, device):
            filters["device"] = device
            break

    for browser in _top_values(sessions["browser"]):
        if _contains_value(question, browser):
            filters["browser"] = browser
            break

    explicit_date = re.search(r"(\d{4}-\d{2}-\d{2})", question)
    if explicit_date:
        parsed_date = pd.to_datetime(explicit_date.group(1), errors="coerce")
        if pd.notna(parsed_date):
            filters["date_from"] = parsed_date.date()
            filters["date_to"] = parsed_date.date()
            filters["date_label"] = f"fecha especifica ({parsed_date.date()})"

    return filters


def _top_values(series: pd.Series, limit: int = 20) -> list[str]:
    return [str(value) for value in series.dropna().astype(str).value_counts().head(limit).index]


def _normalize_text(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(char)
    ).strip()


def _contains_value(question: str, candidate: str) -> bool:
    normalized_candidate = _normalize_text(str(candidate))
    if not normalized_candidate:
        return False
    if len(normalized_candidate) <= 2:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_candidate)}(?!\w)"
    return re.search(pattern, question) is not None
