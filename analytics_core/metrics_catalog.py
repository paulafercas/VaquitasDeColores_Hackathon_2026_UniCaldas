from __future__ import annotations


SUPPORTED_METRICS = [
    {
        "intent": "top_product_or_page",
        "label": "Producto o pagina mas visitada",
        "description": "Identifica la pagina o producto con mas sesiones estimadas en el periodo.",
    },
    {
        "intent": "share_of_sessions",
        "label": "Porcentaje de sesiones",
        "description": "Calcula que porcentaje de sesiones visito una pagina o producto.",
    },
    {
        "intent": "top_exit_page",
        "label": "Pagina de salida principal",
        "description": "Encuentra la pagina que mas veces cerro una sesion.",
    },
    {
        "intent": "top_navigation_flow",
        "label": "Flujo principal de navegacion",
        "description": "Resume el camino mas frecuente entre fuente, entrada y salida.",
    },
    {
        "intent": "bounce_or_abandonment",
        "label": "Rebote o abandono rapido",
        "description": "Mide sesiones con abandono rapido o entrada y salida en la misma pagina.",
    },
    {
        "intent": "page_abandonment_ranking",
        "label": "Paginas con mayor abandono",
        "description": "Rankea paginas de entrada por abandono rapido y rebote proxy.",
    },
    {
        "intent": "segment_comparison",
        "label": "Comparacion por segmento",
        "description": "Compara desempeno por pais, dispositivo o navegador.",
    },
    {
        "intent": "trend_over_time",
        "label": "Tendencia temporal",
        "description": "Muestra evolucion diaria de sesiones y engagement.",
    },
    {
        "intent": "interaction_overview",
        "label": "Resumen de interaccion",
        "description": "Resume clics, paginas visitadas, tiempo, scroll y otras metricas disponibles.",
    },
    {
        "intent": "registration_interest",
        "label": "Interes en registro",
        "description": "Estima usuarios o sesiones que llegaron a URLs relacionadas con registro.",
    },
    {
        "intent": "service_interest",
        "label": "Interes por servicios",
        "description": "Estima usuarios o sesiones que visitaron URLs como pricing, contact o request-demo.",
    },
]


def metric_labels() -> dict[str, str]:
    return {item["intent"]: item["label"] for item in SUPPORTED_METRICS}
