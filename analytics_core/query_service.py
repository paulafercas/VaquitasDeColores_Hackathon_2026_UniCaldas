from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_core.data_loader import AnalyticsBundle, load_analytics_bundle
from analytics_core.metrics_catalog import metric_labels
from config.settings import Settings, get_settings


@dataclass
class QueryService:
    bundle: AnalyticsBundle
    settings: Settings

    @classmethod
    def from_default_paths(cls) -> "QueryService":
        settings = get_settings()
        bundle = load_analytics_bundle(settings)
        return cls(bundle=bundle, settings=settings)

    def answer_structured_query(
        self,
        question: str,
        routed_intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        route = routed_intent or {
            "intent": "unknown_or_unsupported",
            "filters": {},
            "confidence": 0.0,
        }
        sessions = self._filter_sessions(
            self.bundle.canonical_sessions, route.get("filters", {})
        )
        metric_name = metric_labels().get(route["intent"], route["intent"])

        if sessions.empty:
            return {
                "status": "no_data",
                "metric_name": metric_name,
                "filters": route.get("filters", {}),
                "values": {},
                "answer": "No encontre sesiones para ese filtro.",
                "business_interpretation": "No hay base suficiente para concluir comportamiento del usuario en ese periodo.",
                "support_table": pd.DataFrame(),
                "table_preview": [],
                "confidence": route.get("confidence", 0.0),
                "unsupported_reason": None,
            }

        handlers = {
            "top_product_or_page": self._answer_top_product_or_page,
            "share_of_sessions": self._answer_share_of_sessions,
            "top_exit_page": self._answer_top_exit_page,
            "top_navigation_flow": self._answer_top_navigation_flow,
            "bounce_or_abandonment": self._answer_bounce_or_abandonment,
            "page_abandonment_ranking": self._answer_page_abandonment_ranking,
            "segment_comparison": self._answer_segment_comparison,
            "trend_over_time": self._answer_trend_over_time,
            "interaction_overview": self._answer_interaction_overview,
            "registration_interest": self._answer_registration_interest,
            "service_interest": self._answer_service_interest,
        }
        handler = handlers.get(route["intent"])

        if handler is None:
            return {
                "status": "unsupported",
                "metric_name": metric_name,
                "filters": route.get("filters", {}),
                "values": {},
                "answer": "Todavia no se responder esa pregunta con el motor actual.",
                "business_interpretation": "El copiloto necesita una metrica o intencion adicional para responder con trazabilidad.",
                "support_table": pd.DataFrame(),
                "table_preview": [],
                "confidence": route.get("confidence", 0.0),
                "unsupported_reason": "Intent no soportado por la capa analitica.",
            }

        result = handler(question, sessions, route)
        result["metric_name"] = metric_name
        result["filters"] = route.get("filters", {})
        result["confidence"] = route.get("confidence", 0.0)
        result["table_preview"] = _table_preview(
            result.get("support_table"), self.settings.max_table_rows
        )
        return result

    def _filter_sessions(
        self, sessions: pd.DataFrame, filters: dict[str, Any]
    ) -> pd.DataFrame:
        filtered = sessions.copy()

        if "date_from" in filters:
            filtered = filtered[filtered["timestamp"].dt.date >= filters["date_from"]]
        if "date_to" in filters:
            filtered = filtered[filtered["timestamp"].dt.date <= filters["date_to"]]

        for filter_column, canonical_column in {
            "country": "country",
            "device": "device",
            "browser": "browser",
        }.items():
            if filters.get(filter_column):
                filtered = filtered[
                    filtered[canonical_column].astype(str).str.lower()
                    == str(filters[filter_column]).lower()
                ]

        return filtered

    def _answer_top_product_or_page(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        page_visits = _estimate_page_visits(sessions)
        if page_visits.empty:
            return _no_data_result("No pude estimar paginas visitadas para ese filtro.")

        top_row = page_visits.iloc[0]
        return {
            "status": "ok",
            "values": {
                "top_page": top_row["page"],
                "sessions": int(top_row["sessions"]),
                "share_of_sessions": float(top_row["share_of_sessions"]),
                "total_sessions": int(sessions["session_id"].nunique()),
            },
            "answer": (
                f"La pagina o producto mas visto fue {top_row['page']} con {int(top_row['sessions'])} "
                f"sesiones estimadas, equivalente a {top_row['share_of_sessions']:.1%} del total filtrado."
            ),
            "business_interpretation": (
                "Esto sugiere que esa experiencia concentra la atencion inicial o final del recorrido y conviene revisarla como punto de entrada o conversion."
            ),
            "support_table": page_visits.head(10),
        }

    def _answer_share_of_sessions(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        target = _extract_page_candidate(question)
        page_visits = _estimate_page_visits(sessions)
        if page_visits.empty:
            return _no_data_result("No pude calcular el porcentaje de sesiones.")

        filtered = (
            page_visits[page_visits["page"].astype(str).str.contains(target, case=False, na=False)]
            if target
            else page_visits.head(1)
        )

        if filtered.empty:
            return {
                "status": "no_data",
                "values": {},
                "answer": f"No encontre una pagina o producto que coincida con '{target}'.",
                "business_interpretation": "Puede que el nombre no exista en el dataset o use otra nomenclatura.",
                "support_table": page_visits.head(10),
            }

        row = filtered.iloc[0]
        return {
            "status": "ok",
            "values": {
                "page": row["page"],
                "sessions": int(row["sessions"]),
                "share_of_sessions": float(row["share_of_sessions"]),
                "total_sessions": int(sessions["session_id"].nunique()),
            },
            "answer": (
                f"{row['page']} aparecio en {int(row['sessions'])} sesiones estimadas, "
                f"lo que representa {row['share_of_sessions']:.1%} del total analizado."
            ),
            "business_interpretation": (
                "Ese porcentaje ayuda a dimensionar el alcance real de la pagina dentro del funnel y a priorizar optimizaciones o campanas."
            ),
            "support_table": filtered.head(10),
        }

    def _answer_top_exit_page(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        exit_table = (
            sessions.groupby("exit_page", dropna=False)["session_id"]
            .nunique()
            .reset_index(name="sessions")
            .sort_values("sessions", ascending=False)
            .reset_index(drop=True)
        )
        top_row = exit_table.iloc[0]
        share = top_row["sessions"] / sessions["session_id"].nunique()
        return {
            "status": "ok",
            "values": {
                "top_exit_page": top_row["exit_page"],
                "sessions": int(top_row["sessions"]),
                "share_of_sessions": float(share),
            },
            "answer": (
                f"La principal pagina de salida fue {top_row['exit_page']} con {int(top_row['sessions'])} "
                f"sesiones, equivalente a {share:.1%} del total filtrado."
            ),
            "business_interpretation": (
                "Esa pagina merece revision como punto de fuga: puede ser una salida natural o una oportunidad de retencion."
            ),
            "support_table": exit_table.head(10),
        }

    def _answer_top_navigation_flow(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        flow_table = (
            sessions.groupby("flow_path", dropna=False)["session_id"]
            .nunique()
            .reset_index(name="sessions")
            .sort_values("sessions", ascending=False)
            .reset_index(drop=True)
        )
        top_row = flow_table.iloc[0]
        share = top_row["sessions"] / sessions["session_id"].nunique()
        return {
            "status": "ok",
            "values": {
                "top_flow": top_row["flow_path"],
                "sessions": int(top_row["sessions"]),
                "share_of_sessions": float(share),
            },
            "answer": (
                f"El flujo mas frecuente fue {top_row['flow_path']} con {int(top_row['sessions'])} sesiones, "
                f"que representan {share:.1%} del total filtrado."
            ),
            "business_interpretation": (
                "Este patron resume el recorrido dominante y sirve como referencia para detectar fricciones o rutas secundarias."
            ),
            "support_table": flow_table.head(10),
        }

    def _answer_bounce_or_abandonment(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        bounce_rate = float((sessions["entry_page"] == sessions["exit_page"]).mean())
        quick_abandonment_rate = float(sessions["quick_abandonment"].mean())
        table = (
            sessions.assign(same_page=(sessions["entry_page"] == sessions["exit_page"]).astype(int))
            .groupby("entry_page", dropna=False)
            .agg(
                sessions=("session_id", "nunique"),
                bounce_proxy_rate=("same_page", "mean"),
                quick_abandonment_rate=("quick_abandonment", "mean"),
            )
            .reset_index()
            .sort_values("quick_abandonment_rate", ascending=False)
            .reset_index(drop=True)
        )
        return {
            "status": "ok",
            "values": {
                "bounce_proxy_rate": bounce_rate,
                "quick_abandonment_rate": quick_abandonment_rate,
            },
            "answer": (
                f"La tasa proxy de rebote fue {bounce_rate:.1%} y el abandono rapido promedio fue {quick_abandonment_rate:.1%}."
            ),
            "business_interpretation": (
                "Si ambas metricas se mantienen altas, conviene revisar la propuesta de valor y los llamados a la accion de las paginas de entrada."
            ),
            "support_table": table.head(10),
        }

    def _answer_page_abandonment_ranking(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        table = _build_abandonment_table(sessions)
        top_row = table.iloc[0]
        return {
            "status": "ok",
            "values": {
                "top_abandonment_page": top_row["entry_page"],
                "quick_abandonment_rate": float(top_row["quick_abandonment_rate"]),
                "bounce_proxy_rate": float(top_row["bounce_proxy_rate"]),
                "sessions": int(top_row["sessions"]),
            },
            "answer": (
                f"La pagina con mayor abandono fue {top_row['entry_page']} con {top_row['quick_abandonment_rate']:.1%} "
                f"de abandono rapido sobre {int(top_row['sessions'])} sesiones."
            ),
            "business_interpretation": (
                "Estas paginas son candidatas prioritarias para revisar propuesta de valor, claridad del contenido y llamados a la accion."
            ),
            "support_table": table.head(10),
        }

    def _answer_segment_comparison(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        dimension = route.get("filters", {}).get("segment_dimension", "country")
        table = (
            sessions.groupby(dimension, dropna=False)
            .agg(
                sessions=("session_id", "nunique"),
                avg_duration_seconds=("session_duration_seconds", "mean"),
                quick_abandonment_rate=("quick_abandonment", "mean"),
                avg_engagement_score=("engagement_score", "mean"),
            )
            .reset_index()
            .sort_values("sessions", ascending=False)
            .reset_index(drop=True)
        )
        top_row = table.iloc[0]
        return {
            "status": "ok",
            "values": {
                "segment_dimension": dimension,
                "top_segment": top_row[dimension],
                "sessions": int(top_row["sessions"]),
            },
            "answer": (
                f"El segmento con mas sesiones fue {top_row[dimension]} dentro de {dimension}, "
                f"con {int(top_row['sessions'])} sesiones."
            ),
            "business_interpretation": (
                "Este corte ayuda a priorizar campanas, UX y mensajes segun el segmento con mas peso o peor calidad."
            ),
            "support_table": table.head(10),
        }

    def _answer_trend_over_time(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        trend = (
            sessions.assign(date=sessions["timestamp"].dt.date)
            .groupby("date", dropna=False)
            .agg(
                sessions=("session_id", "nunique"),
                avg_engagement_score=("engagement_score", "mean"),
                avg_duration_seconds=("session_duration_seconds", "mean"),
            )
            .reset_index()
            .sort_values("date")
            .reset_index(drop=True)
        )
        latest = trend.iloc[-1]
        return {
            "status": "ok",
            "values": {
                "latest_date": str(latest["date"]),
                "latest_sessions": int(latest["sessions"]),
            },
            "answer": (
                f"La tendencia diaria muestra {len(trend)} puntos; la fecha mas reciente es {latest['date']} "
                f"con {int(latest['sessions'])} sesiones."
            ),
            "business_interpretation": (
                "La serie temporal sirve para detectar picos, caidas y validar el impacto de cambios en contenido o adquisicion."
            ),
            "support_table": trend.tail(10),
        }

    def _answer_interaction_overview(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        page_visits = _estimate_page_visits(sessions)
        abandonment_table = _build_abandonment_table(sessions)
        total_sessions = int(sessions["session_id"].nunique())
        available_metrics = _available_metric_names(sessions)
        missing_metrics = _missing_metric_names(sessions)

        summary_table = pd.DataFrame(
            [
                {
                    "metric": "total_sessions",
                    "value": total_sessions,
                },
                {
                    "metric": "avg_clicks_per_session",
                    "value": round(float(sessions["clicks"].mean()), 2),
                },
                {
                    "metric": "avg_pages_per_session",
                    "value": round(float(sessions["page_count"].mean()), 2),
                },
                {
                    "metric": "avg_session_duration_seconds",
                    "value": round(float(sessions["session_duration_seconds"].mean()), 2),
                },
                {
                    "metric": "quick_abandonment_rate",
                    "value": round(float(sessions["quick_abandonment"].mean()), 4),
                },
                {
                    "metric": "top_page",
                    "value": page_visits.iloc[0]["page"] if not page_visits.empty else "N/A",
                },
                {
                    "metric": "top_abandonment_page",
                    "value": abandonment_table.iloc[0]["entry_page"] if not abandonment_table.empty else "N/A",
                },
            ]
        )

        top_page = page_visits.iloc[0] if not page_visits.empty else None
        answer = (
            f"En el periodo analizado hubo {total_sessions} sesiones, con un promedio de {sessions['clicks'].mean():.1f} clics, "
            f"{sessions['page_count'].mean():.1f} paginas y {sessions['session_duration_seconds'].mean():.1f} segundos por sesion. "
            f"La pagina mas visitada fue {top_page['page']} ({top_page['share_of_sessions']:.1%} de las sesiones estimadas)."
            if top_page is not None
            else "No pude construir el resumen de interaccion."
        )
        business = (
            "Este resumen permite ver de forma integrada atraccion, profundidad de navegacion y friccion. "
            + (
                f"Hoy puedo calcular: {', '.join(available_metrics)}. "
                if available_metrics
                else ""
            )
            + (
                f"Aun no puedo responder con datos reales sobre: {', '.join(missing_metrics)}."
                if missing_metrics
                else ""
            )
        )

        return {
            "status": "ok",
            "values": {
                "total_sessions": total_sessions,
                "avg_clicks_per_session": float(sessions["clicks"].mean()),
                "avg_pages_per_session": float(sessions["page_count"].mean()),
                "avg_session_duration_seconds": float(sessions["session_duration_seconds"].mean()),
                "quick_abandonment_rate": float(sessions["quick_abandonment"].mean()),
                "available_metrics": available_metrics,
                "missing_metrics": missing_metrics,
            },
            "answer": answer,
            "business_interpretation": business,
            "support_table": summary_table,
        }

    def _answer_registration_interest(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        keyword_map = {
            "registro": ["register", "registration", "signup", "sign-up", "sign_up"],
        }
        matched = _build_url_interest_table(sessions, keyword_map)
        if matched.empty:
            return _no_data_result(
                "No encontre sesiones asociadas a URLs de registro en el dataset filtrado."
            )

        total_sessions = int(sessions["session_id"].nunique())
        matched_sessions = int(matched["sessions"].sum())
        share = matched_sessions / max(total_sessions, 1)
        top_row = matched.iloc[0]

        return {
            "status": "ok",
            "values": {
                "registration_sessions": matched_sessions,
                "registration_share": float(share),
                "top_registration_page": top_row["page"],
                "top_registration_page_sessions": int(top_row["sessions"]),
            },
            "answer": (
                f"Estimo {matched_sessions} sesiones con interes en registro, equivalentes a {share:.1%} del total analizado. "
                f"La URL de registro mas frecuente fue {top_row['page']}."
            ),
            "business_interpretation": (
                "Esto no prueba un registro completado, pero si muestra cuantos usuarios llegaron a puntos claros del journey de registro."
            ),
            "support_table": matched.head(10),
        }

    def _answer_service_interest(
        self, question: str, sessions: pd.DataFrame, route: dict[str, Any]
    ) -> dict[str, Any]:
        keyword_map = {
            "registration": ["register", "registration", "signup", "sign-up", "sign_up"],
            "pricing": ["pricing", "price", "precio", "precios"],
            "contact": ["contact", "contacto"],
            "demo": ["demo", "request-demo", "request_demo"],
        }
        matched = _build_url_interest_table(sessions, keyword_map)
        if matched.empty:
            return _no_data_result(
                "No encontre sesiones asociadas a URLs de pricing, contacto o demo en el dataset filtrado."
            )

        total_sessions = int(sessions["session_id"].nunique())
        matched_sessions = int(matched["sessions"].sum())
        share = matched_sessions / max(total_sessions, 1)
        matched_by_group = (
            matched.groupby("interest_type", dropna=False)["sessions"]
            .sum()
            .reset_index()
            .sort_values("sessions", ascending=False)
            .reset_index(drop=True)
        )
        top_group = matched_by_group.iloc[0]
        top_page = matched.iloc[0]

        return {
            "status": "ok",
            "values": {
                "service_interest_sessions": matched_sessions,
                "service_interest_share": float(share),
                "top_interest_type": top_group["interest_type"],
                "top_interest_type_sessions": int(top_group["sessions"]),
                "top_interest_page": top_page["page"],
            },
            "answer": (
                f"Estimo {matched_sessions} sesiones con interes por servicios, equivalentes a {share:.1%} del total. "
                f"La categoria mas frecuente fue {top_group['interest_type']} y la URL mas visitada fue {top_page['page']}."
            ),
            "business_interpretation": (
                "Estas URLs suelen representar intencion comercial mas fuerte, asi que sirven como proxy de interes por la oferta."
            ),
            "support_table": matched.head(12),
        }


def _estimate_page_visits(sessions: pd.DataFrame) -> pd.DataFrame:
    total_sessions = sessions["session_id"].nunique()
    entry = sessions[["session_id", "entry_page"]].rename(columns={"entry_page": "page"})
    exit_ = sessions[["session_id", "exit_page"]].rename(columns={"exit_page": "page"})
    page_visits = (
        pd.concat([entry, exit_], ignore_index=True)
        .drop_duplicates()
        .groupby("page", dropna=False)["session_id"]
        .nunique()
        .reset_index(name="sessions")
        .sort_values("sessions", ascending=False)
        .reset_index(drop=True)
    )
    page_visits["share_of_sessions"] = page_visits["sessions"] / max(total_sessions, 1)
    return page_visits


def _build_url_interest_table(
    sessions: pd.DataFrame, keyword_map: dict[str, list[str]]
) -> pd.DataFrame:
    expanded = pd.concat(
        [
            sessions[["session_id", "entry_page"]].rename(columns={"entry_page": "page"}),
            sessions[["session_id", "exit_page"]].rename(columns={"exit_page": "page"}),
        ],
        ignore_index=True,
    ).drop_duplicates()

    matched_frames: list[pd.DataFrame] = []
    for interest_type, keywords in keyword_map.items():
        pattern = "|".join(re.escape(keyword) for keyword in keywords)
        matched = expanded[expanded["page"].astype(str).str.contains(pattern, case=False, na=False)].copy()
        if matched.empty:
            continue
        matched["interest_type"] = interest_type
        matched_frames.append(matched)

    if not matched_frames:
        return pd.DataFrame(columns=["interest_type", "page", "sessions", "share_within_interest"])

    combined = pd.concat(matched_frames, ignore_index=True)
    interest_table = (
        combined.groupby(["interest_type", "page"], dropna=False)["session_id"]
        .nunique()
        .reset_index(name="sessions")
        .sort_values(["sessions", "interest_type"], ascending=[False, True])
        .reset_index(drop=True)
    )
    group_totals = interest_table.groupby("interest_type")["sessions"].transform("sum")
    interest_table["share_within_interest"] = interest_table["sessions"] / group_totals
    return interest_table


def _build_abandonment_table(sessions: pd.DataFrame) -> pd.DataFrame:
    return (
        sessions.assign(
            same_page=(sessions["entry_page"] == sessions["exit_page"]).astype(int)
        )
        .groupby("entry_page", dropna=False)
        .agg(
            sessions=("session_id", "nunique"),
            quick_abandonment_rate=("quick_abandonment", "mean"),
            bounce_proxy_rate=("same_page", "mean"),
            avg_duration_seconds=("session_duration_seconds", "mean"),
            avg_clicks=("clicks", "mean"),
            avg_pages=("page_count", "mean"),
        )
        .reset_index()
        .sort_values(
            ["quick_abandonment_rate", "bounce_proxy_rate", "sessions"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def _available_metric_names(sessions: pd.DataFrame) -> list[str]:
    metric_labels = {
        "clicks": "clics",
        "page_count": "paginas visitadas",
        "session_duration_seconds": "tiempo de navegacion",
        "scroll_depth": "scroll",
        "registration_count": "registros",
        "payment_count": "pagos",
        "price_value": "precios",
    }
    available: list[str] = []
    for column, label in metric_labels.items():
        if column not in sessions.columns:
            continue
        series = pd.to_numeric(sessions[column], errors="coerce")
        if series.notna().any() and not series.fillna(0).eq(0).all():
            available.append(label)
    return available


def _missing_metric_names(sessions: pd.DataFrame) -> list[str]:
    metric_labels = {
        "scroll_depth": "scroll",
        "registration_count": "registros",
        "payment_count": "pagos",
        "price_value": "precios",
    }
    missing: list[str] = []
    for column, label in metric_labels.items():
        if column not in sessions.columns:
            missing.append(label)
            continue
        series = pd.to_numeric(sessions[column], errors="coerce")
        if not series.notna().any() or series.fillna(0).eq(0).all():
            missing.append(label)
    return missing


def _table_preview(df: pd.DataFrame | None, max_rows: int) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    preview = df.head(max_rows).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].astype(str)
    return preview.to_dict(orient="records")


def _extract_page_candidate(question: str) -> str | None:
    match = re.search(
        r"(?:producto|pagina|page)\s+([^\s,;:]+)",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def _no_data_result(message: str) -> dict[str, Any]:
    return {
        "status": "no_data",
        "values": {},
        "answer": message,
        "business_interpretation": "No hay suficiente informacion para responder con confianza.",
        "support_table": pd.DataFrame(),
    }
