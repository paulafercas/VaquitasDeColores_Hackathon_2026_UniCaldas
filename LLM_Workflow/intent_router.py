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
    if _is_hackathon_objective_request(normalized):
        intent = "hackathon_objective_summary"
        confidence = 0.95
    elif any(phrase in normalized for phrase in ["interaccion promedio por pagina", "interaccion por pagina", "promedio por pagina"]) and any(
        keyword in normalized for keyword in ["interaccion", "clic", "click", "scroll", "tiempo"]
    ):
        intent = "page_interaction_profile"
        confidence = 0.92
    elif (
        any(keyword in normalized for keyword in ["abandono", "rebote"])
        and any(keyword in normalized for keyword in ["pagina", "paginas", "donde", "mayor", "top", "criticos", "criticas", "puntos"])
    ):
        intent = "page_abandonment_ranking"
        confidence = 0.9
    elif any(phrase in normalized for phrase in ["paginas mas buscadas", "pagina mas buscada", "paginas mas consultadas"]):
        intent = "page_metric_ranking"
        confidence = 0.92
        filters["page_metric"] = "entry_sessions"
    elif _extract_page_metric(normalized) and any(
        keyword in normalized for keyword in ["pagina", "paginas", "top", "ranking", "mas", "donde", "cuales son", "cuales fueron"]
    ):
        intent = "page_metric_ranking"
        confidence = 0.93
        filters["page_metric"] = _extract_page_metric(normalized)
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
    elif any(phrase in normalized for phrase in ["patrones basicos de conversion", "patrones basicos de intencion"]):
        intent = "service_interest"
        confidence = 0.9
    elif any(keyword in normalized for keyword in ["conversion", "intencion"]) and any(
        keyword in normalized for keyword in ["pricing", "contacto", "contact", "producto", "productos", "demo", "registro", "register", "servicio", "servicios"]
    ):
        intent = "service_interest"
        confidence = 0.9
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
    top_n = _extract_top_n(question)
    if top_n is not None:
        filters["top_n"] = top_n
    if any(phrase in question for phrase in ["menos visitas", "menos visitadas", "menos vistas", "menor trafico", "menos sesiones"]):
        filters["sort_order"] = "asc"

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


def _extract_top_n(question: str) -> int | None:
    patterns = [
        r"\btop\s+(\d{1,2})\b",
        r"\b(\d{1,2})\s+paginas\b",
        r"\bprimer[oa]s?\s+(\d{1,2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 50:
                return value
    return None


def _extract_page_metric(question: str) -> str | None:
    metric_aliases = [
        ("exit_sessions", ["sesiones de salida", "paginas de salida", "salidas", "salida", "exits"]),
        ("entry_sessions", ["sesiones de entrada", "visitadas", "visitas", "mas vistas", "mas visitadas", "mas buscadas", "mas consultadas", "trafico", "sesiones"]),
        ("avg_duration_seconds", ["tiempo", "duracion", "duracion promedio", "mas tiempo"]),
        ("avg_clicks", ["clics", "clicks", "mas clics", "mas clicks"]),
        ("avg_pages", ["paginas por sesion", "profundidad", "mas paginas"]),
        ("avg_scroll_depth", ["scroll", "profundidad de scroll"]),
        ("bounce_proxy_rate", ["rebote"]),
        ("abandono_rapido_rate", ["abandono rapido"]),
        ("frustration_rate", ["frustracion"]),
        ("engagement_score_avg", ["engagement", "interaccion", "score"]),
        ("registration_rate", ["registros", "registro"]),
        ("payment_rate", ["pagos", "pago"]),
    ]
    for metric, aliases in metric_aliases:
        if any(alias in question for alias in aliases):
            return metric
    return None


def _is_hackathon_objective_request(question: str) -> bool:
    meta_terms = [
        "objetivo especifico",
        "solucion end-to-end",
        "end-to-end",
        "insights adicionales",
        "equipo de marketing",
    ]
    scope_terms = [
        "paginas y productos top",
        "puntos criticos de abandono",
        "flujo de navegacion comun",
        "interaccion promedio por pagina",
        "patrones basicos de conversion",
    ]
    if any(term in question for term in meta_terms):
        return True
    return sum(term in question for term in scope_terms) >= 2
