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
        route = routed_intent or {"intent": "unknown_or_unsupported", "filters": {}, "confidence": 0.0}
        sessions = self._filter_sessions(self.bundle.canonical_sessions, route.get("filters", {}))
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
            "page_metric_ranking": self._answer_page_metric_ranking,
            "page_interaction_profile": self._answer_page_interaction_profile,
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
            "hackathon_objective_summary": self._answer_hackathon_objective_summary,
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
        result["table_preview"] = _table_preview(result.get("support_table"), self.settings.max_table_rows)
        return result

    def _filter_sessions(self, sessions: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
        filtered = sessions.copy()
        if "date_from" in filters:
            filtered = filtered[filtered["timestamp"].dt.date >= filters["date_from"]]
        if "date_to" in filters:
            filtered = filtered[filtered["timestamp"].dt.date <= filters["date_to"]]
        for filter_column, canonical_column in {"country": "country", "device": "device", "browser": "browser"}.items():
            if filters.get(filter_column):
                filtered = filtered[filtered[canonical_column].astype(str).str.lower() == str(filters[filter_column]).lower()]
        return filtered

    def _answer_top_product_or_page(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
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
            "answer": f"La pagina o producto mas visto fue {top_row['page']} con {int(top_row['sessions'])} sesiones estimadas, equivalente a {top_row['share_of_sessions']:.1%} del total filtrado.",
            "business_interpretation": "Esto sugiere que esa experiencia concentra la atencion inicial o final del recorrido y conviene revisarla como punto de entrada o conversion.",
            "support_table": page_visits.head(10),
        }

    def _answer_page_metric_ranking(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        metric_key = route.get("filters", {}).get("page_metric", "entry_sessions")
        top_n = int(route.get("filters", {}).get("top_n", 10))
        sort_order = route.get("filters", {}).get("sort_order", "desc")
        page_metrics = _build_page_metric_table(sessions)
        if page_metrics.empty or metric_key not in page_metrics.columns:
            return _no_data_result("No pude construir el ranking de paginas para esa metrica.")
        filtered = page_metrics[page_metrics[metric_key].notna()].copy()
        if filtered.empty:
            return _no_data_result("La metrica solicitada no tiene datos disponibles para construir el ranking.")
        ascending = sort_order == "asc"
        ranking = filtered.assign(metric_value=filtered[metric_key]).sort_values(["metric_value", "page"], ascending=[ascending, True]).head(top_n).reset_index(drop=True)
        metric_label = _page_metric_label(metric_key)
        top_row = ranking.iloc[0]
        ranking_lines = [
            f"{index + 1}. {row['page']} - {_format_metric_value(row['metric_value'], metric_key)}"
            for index, (_, row) in enumerate(ranking.iterrows())
        ]
        return {
            "status": "ok",
            "values": {
                "page_metric": metric_key,
                "metric_label": metric_label,
                "top_n": top_n,
                "top_page": str(top_row["page"]),
                "top_value": _coerce_scalar(top_row["metric_value"]),
                "sort_order": sort_order,
            },
            "answer": (
                f"Este es el ranking de las {min(top_n, len(ranking))} paginas con {'menor' if ascending else 'mayor'} {metric_label}:\n"
                + "\n".join(ranking_lines)
            ),
            "business_interpretation": f"Este ranking ayuda a priorizar paginas segun la metrica {metric_label} para optimizacion, testing y decisiones de marketing.",
            "support_table": ranking[["page", metric_key]].rename(columns={metric_key: metric_label}),
        }

    def _answer_share_of_sessions(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        target = _extract_page_candidate(question)
        page_visits = _estimate_page_visits(sessions)
        if page_visits.empty:
            return _no_data_result("No pude calcular el porcentaje de sesiones.")
        filtered = page_visits[page_visits["page"].astype(str).str.contains(target, case=False, na=False)] if target else page_visits.head(1)
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
            "answer": f"{row['page']} aparecio en {int(row['sessions'])} sesiones estimadas, lo que representa {row['share_of_sessions']:.1%} del total analizado.",
            "business_interpretation": "Ese porcentaje ayuda a dimensionar el alcance real de la pagina dentro del funnel y a priorizar optimizaciones o campanas.",
            "support_table": filtered.head(10),
        }

    def _answer_top_exit_page(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        exit_table = sessions.groupby("exit_page", dropna=False)["session_id"].nunique().reset_index(name="sessions").sort_values("sessions", ascending=False).reset_index(drop=True)
        top_row = exit_table.iloc[0]
        share = top_row["sessions"] / sessions["session_id"].nunique()
        return {
            "status": "ok",
            "values": {"top_exit_page": top_row["exit_page"], "sessions": int(top_row["sessions"]), "share_of_sessions": float(share)},
            "answer": f"La principal pagina de salida fue {top_row['exit_page']} con {int(top_row['sessions'])} sesiones, equivalente a {share:.1%} del total filtrado.",
            "business_interpretation": "Esa pagina merece revision como punto de fuga: puede ser una salida natural o una oportunidad de retencion.",
            "support_table": exit_table.head(10),
        }

    def _answer_top_navigation_flow(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        flow_table = sessions.groupby("flow_path", dropna=False)["session_id"].nunique().reset_index(name="sessions").sort_values("sessions", ascending=False).reset_index(drop=True)
        top_n = int(route.get("filters", {}).get("top_n", 5))
        ranking = flow_table.head(top_n).copy()
        total_sessions = max(sessions["session_id"].nunique(), 1)
        ranking["share_of_sessions"] = ranking["sessions"] / total_sessions
        lines = [
            f"{index + 1}. {row['flow_path']} - {_format_metric_value(row['sessions'], 'entry_sessions')} sesiones ({row['share_of_sessions']:.1%})"
            for index, (_, row) in enumerate(ranking.iterrows())
        ]
        return {
            "status": "ok",
            "values": {"top_flow": ranking.iloc[0]["flow_path"], "sessions": int(ranking.iloc[0]["sessions"]), "share_of_sessions": float(ranking.iloc[0]["share_of_sessions"]), "top_n": top_n},
            "answer": "Estos son los flujos de navegacion mas comunes:\n" + "\n".join(lines),
            "business_interpretation": "Los primeros flujos muestran los recorridos dominantes y ayudan a detectar rutas a proteger, simplificar o redirigir.",
            "support_table": ranking,
        }

    def _answer_bounce_or_abandonment(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        bounce_rate = float((sessions["entry_page"] == sessions["exit_page"]).mean())
        quick_abandonment_rate = float(sessions["quick_abandonment"].mean())
        table = sessions.assign(same_page=(sessions["entry_page"] == sessions["exit_page"]).astype(int)).groupby("entry_page", dropna=False).agg(sessions=("session_id", "nunique"), bounce_proxy_rate=("same_page", "mean"), quick_abandonment_rate=("quick_abandonment", "mean")).reset_index().sort_values("quick_abandonment_rate", ascending=False).reset_index(drop=True)
        return {
            "status": "ok",
            "values": {"bounce_proxy_rate": bounce_rate, "quick_abandonment_rate": quick_abandonment_rate},
            "answer": f"La tasa proxy de rebote fue {bounce_rate:.1%} y el abandono rapido promedio fue {quick_abandonment_rate:.1%}.",
            "business_interpretation": "Si ambas metricas se mantienen altas, conviene revisar la propuesta de valor y los llamados a la accion de las paginas de entrada.",
            "support_table": table.head(10),
        }

    def _answer_page_abandonment_ranking(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        table = _build_abandonment_table(sessions)
        top_n = int(route.get("filters", {}).get("top_n", 5))
        ranking = table.head(top_n).copy()
        lines = [
            f"{index + 1}. {row['entry_page']} - abandono rapido {_format_metric_value(row['quick_abandonment_rate'], 'abandono_rapido_rate')}, frustracion {_format_metric_value(row['frustration_rate'], 'frustration_rate')}, rebote {_format_metric_value(row['bounce_proxy_rate'], 'bounce_proxy_rate')}"
            for index, (_, row) in enumerate(ranking.iterrows())
        ]
        top_row = ranking.iloc[0]
        return {
            "status": "ok",
            "values": {
                "top_abandonment_page": top_row["entry_page"],
                "quick_abandonment_rate": float(top_row["quick_abandonment_rate"]),
                "bounce_proxy_rate": float(top_row["bounce_proxy_rate"]),
                "frustration_rate": float(top_row["frustration_rate"]),
                "sessions": int(top_row["sessions"]),
            },
            "answer": "Estos son los puntos criticos de abandono:\n" + "\n".join(lines),
            "business_interpretation": "Son las paginas donde conviene revisar propuesta de valor, friccion de UX y llamados a la accion.",
            "support_table": ranking,
        }

    def _answer_page_interaction_profile(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        table = _build_page_metric_table(sessions)[["page", "avg_clicks", "avg_scroll_depth", "avg_duration_seconds", "engagement_score_avg"]].copy()
        table = table.sort_values(["engagement_score_avg", "avg_duration_seconds", "avg_clicks"], ascending=[False, False, False]).head(int(route.get("filters", {}).get("top_n", 10))).reset_index(drop=True)
        if table.empty:
            return _no_data_result("No pude calcular la interaccion promedio por pagina.")
        lines = [
            f"{index + 1}. {row['page']} - clics {_format_metric_value(row['avg_clicks'], 'avg_clicks')}, scroll {_format_metric_value(row['avg_scroll_depth'], 'avg_scroll_depth')}, tiempo {_format_metric_value(row['avg_duration_seconds'], 'avg_duration_seconds')}"
            for index, (_, row) in enumerate(table.iterrows())
        ]
        return {
            "status": "ok",
            "values": {
                "top_page": str(table.iloc[0]["page"]),
                "avg_clicks": _coerce_scalar(table.iloc[0]["avg_clicks"]),
                "avg_scroll_depth": _coerce_scalar(table.iloc[0]["avg_scroll_depth"]),
                "avg_duration_seconds": _coerce_scalar(table.iloc[0]["avg_duration_seconds"]),
            },
            "answer": "Esta es la interaccion promedio por pagina:\n" + "\n".join(lines),
            "business_interpretation": "La combinacion de clics, scroll y tiempo ayuda a detectar paginas con interaccion profunda frente a paginas con consumo superficial.",
            "support_table": table,
        }

    def _answer_segment_comparison(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        dimension = route.get("filters", {}).get("segment_dimension", "country")
        table = sessions.groupby(dimension, dropna=False).agg(sessions=("session_id", "nunique"), avg_duration_seconds=("session_duration_seconds", "mean"), quick_abandonment_rate=("quick_abandonment", "mean"), avg_engagement_score=("engagement_score", "mean")).reset_index().sort_values("sessions", ascending=False).reset_index(drop=True)
        top_row = table.iloc[0]
        return {
            "status": "ok",
            "values": {"segment_dimension": dimension, "top_segment": top_row[dimension], "sessions": int(top_row["sessions"])},
            "answer": f"El segmento con mas sesiones fue {top_row[dimension]} dentro de {dimension}, con {int(top_row['sessions'])} sesiones.",
            "business_interpretation": "Este corte ayuda a priorizar campanas, UX y mensajes segun el segmento con mas peso o peor calidad.",
            "support_table": table.head(10),
        }

    def _answer_trend_over_time(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        trend = sessions.assign(date=sessions["timestamp"].dt.date).groupby("date", dropna=False).agg(sessions=("session_id", "nunique"), avg_engagement_score=("engagement_score", "mean"), avg_duration_seconds=("session_duration_seconds", "mean")).reset_index().sort_values("date").reset_index(drop=True)
        latest = trend.iloc[-1]
        return {
            "status": "ok",
            "values": {"latest_date": str(latest["date"]), "latest_sessions": int(latest["sessions"])},
            "answer": f"La tendencia diaria muestra {len(trend)} puntos; la fecha mas reciente es {latest['date']} con {int(latest['sessions'])} sesiones.",
            "business_interpretation": "La serie temporal sirve para detectar picos, caidas y validar el impacto de cambios en contenido o adquisicion.",
            "support_table": trend.tail(10),
        }

    def _answer_interaction_overview(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        page_visits = _estimate_page_visits(sessions)
        abandonment_table = _build_abandonment_table(sessions)
        total_sessions = int(sessions["session_id"].nunique())
        available_metrics = _available_metric_names(sessions)
        missing_metrics = _missing_metric_names(sessions)
        summary_table = pd.DataFrame([
            {"metric": "total_sessions", "value": total_sessions},
            {"metric": "avg_clicks_per_session", "value": round(float(sessions["clicks"].mean()), 2)},
            {"metric": "avg_pages_per_session", "value": round(float(sessions["page_count"].mean()), 2)},
            {"metric": "avg_session_duration_seconds", "value": round(float(sessions["session_duration_seconds"].mean()), 2)},
            {"metric": "quick_abandonment_rate", "value": round(float(sessions["quick_abandonment"].mean()), 4)},
            {"metric": "top_page", "value": page_visits.iloc[0]["page"] if not page_visits.empty else "N/A"},
            {"metric": "top_abandonment_page", "value": abandonment_table.iloc[0]["entry_page"] if not abandonment_table.empty else "N/A"},
        ])
        top_page = page_visits.iloc[0] if not page_visits.empty else None
        answer = (
            f"En el periodo analizado hubo {total_sessions} sesiones, con un promedio de {sessions['clicks'].mean():.1f} clics, {sessions['page_count'].mean():.1f} paginas y {sessions['session_duration_seconds'].mean():.1f} segundos por sesion. La pagina mas visitada fue {top_page['page']} ({top_page['share_of_sessions']:.1%} de las sesiones estimadas)."
            if top_page is not None else "No pude construir el resumen de interaccion."
        )
        business = "Este resumen permite ver de forma integrada atraccion, profundidad de navegacion y friccion. " + (f"Hoy puedo calcular: {', '.join(available_metrics)}. " if available_metrics else "") + (f"Aun no puedo responder con datos reales sobre: {', '.join(missing_metrics)}." if missing_metrics else "")
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

    def _answer_registration_interest(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        matched = _build_url_interest_table(sessions, {"registro": ["register", "registration", "signup", "sign-up", "sign_up"]})
        if matched.empty:
            return _no_data_result("No encontre sesiones asociadas a URLs de registro en el dataset filtrado.")
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
            "answer": f"Estimo {matched_sessions} sesiones con interes en registro, equivalentes a {share:.1%} del total analizado. La URL de registro mas frecuente fue {top_row['page']}.",
            "business_interpretation": "Esto no prueba un registro completado, pero si muestra cuantos usuarios llegaron a puntos claros del journey de registro.",
            "support_table": matched.head(10),
        }

    def _answer_service_interest(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        matched = _build_url_interest_table(sessions, _commercial_interest_keywords())
        if matched.empty:
            return _no_data_result("No encontre sesiones asociadas a URLs de pricing, contacto o demo en el dataset filtrado.")
        total_sessions = int(sessions["session_id"].nunique())
        matched_sessions = int(matched["sessions"].sum())
        share = matched_sessions / max(total_sessions, 1)
        matched_by_group = matched.groupby("interest_type", dropna=False)["sessions"].sum().reset_index().sort_values("sessions", ascending=False).reset_index(drop=True)
        top_group = matched_by_group.iloc[0]
        top_page = matched.iloc[0]
        lines = [
            f"{index + 1}. {row['interest_type']} - {row['page']} - {_format_metric_value(row['sessions'], 'entry_sessions')} sesiones"
            for index, (_, row) in enumerate(matched.head(8).iterrows())
        ]
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
                f"Estos son los patrones basicos de conversion o intencion detectados en paginas comerciales. "
                f"En total aparecen en {share:.1%} de las sesiones analizadas:\n" + "\n".join(lines)
            ),
            "business_interpretation": "Las visitas a pricing, contacto, demo o registro funcionan como proxy de intencion comercial.",
            "support_table": matched.head(12),
        }

    def _answer_hackathon_objective_summary(self, question: str, sessions: pd.DataFrame, route: dict[str, Any]) -> dict[str, Any]:
        page_visits = _estimate_page_visits(sessions)
        exit_table = sessions.groupby("exit_page", dropna=False)["session_id"].nunique().reset_index(name="sessions").sort_values("sessions", ascending=False).reset_index(drop=True)
        flow_table = sessions.groupby("flow_path", dropna=False)["session_id"].nunique().reset_index(name="sessions").sort_values("sessions", ascending=False).reset_index(drop=True)
        abandonment_table = _build_abandonment_table(sessions)
        service_interest = _build_url_interest_table(sessions, _commercial_interest_keywords())
        total_sessions = int(sessions["session_id"].nunique())
        top_page = page_visits.iloc[0]
        top_exit = exit_table.iloc[0]
        top_flow = flow_table.iloc[0]
        top_abandonment = abandonment_table.iloc[0]
        top_interest = service_interest.groupby("interest_type", dropna=False)["sessions"].sum().reset_index().sort_values("sessions", ascending=False).reset_index(drop=True) if not service_interest.empty else pd.DataFrame()
        top_interest_row = top_interest.iloc[0] if not top_interest.empty else None
        source_table = _build_source_summary(sessions)
        top_source = source_table.iloc[0] if not source_table.empty else None
        country_table = _build_country_opportunity_table(sessions)
        top_country = country_table.iloc[0] if not country_table.empty else None
        friction_table = _build_friction_table(sessions)
        top_friction = friction_table.iloc[0] if not friction_table.empty else None
        support_rows = [
            {"insight_type": "requerido", "objective": "Paginas y productos top", "finding": str(top_page["page"]), "metric": "share_of_sessions", "value": round(float(top_page["share_of_sessions"]), 4), "why_it_matters": "Prioriza las experiencias con mayor alcance para optimizacion y experimentacion."},
            {"insight_type": "requerido", "objective": "Puntos criticos de abandono", "finding": str(top_abandonment["entry_page"]), "metric": "quick_abandonment_rate", "value": round(float(top_abandonment["quick_abandonment_rate"]), 4), "why_it_matters": "Ubica paginas de entrada donde se esta perdiendo demanda."},
            {"insight_type": "requerido", "objective": "Flujo de navegacion comun", "finding": str(top_flow["flow_path"]), "metric": "share_of_sessions", "value": round(float(top_flow["sessions"] / max(total_sessions, 1)), 4), "why_it_matters": "Define el recorrido dominante que conviene proteger y optimizar."},
            {"insight_type": "requerido", "objective": "Interaccion promedio por pagina", "finding": "Promedio global de interaccion", "metric": "avg_clicks|avg_pages|avg_duration_seconds", "value": f"{sessions['clicks'].mean():.1f} clics, {sessions['page_count'].mean():.1f} paginas, {sessions['session_duration_seconds'].mean():.1f} segundos", "why_it_matters": "Resume profundidad y calidad de navegacion para evaluar engagement."},
            {"insight_type": "requerido", "objective": "Patrones basicos de conversion o intencion", "finding": str(top_interest_row['interest_type']) if top_interest_row is not None else "Sin evidencia", "metric": "share_of_sessions", "value": round(float(service_interest['sessions'].sum() / max(total_sessions, 1)), 4) if not service_interest.empty else 0.0, "why_it_matters": "Usa URLs comerciales como proxy de intencion para el equipo de Marketing."},
        ]
        if top_source is not None:
            support_rows.append({"insight_type": "adicional", "objective": "Adquisicion por fuente", "finding": str(top_source["source"]), "metric": "sessions_share", "value": round(float(top_source["share_of_sessions"]), 4), "why_it_matters": "Ayuda a identificar el canal con mayor volumen para reasignar presupuesto."})
        if top_country is not None:
            support_rows.append({"insight_type": "adicional", "objective": "Mercados con mejor engagement", "finding": str(top_country["country"]), "metric": "avg_engagement_score", "value": round(float(top_country["avg_engagement_score"]), 4), "why_it_matters": "Permite priorizar geografias con mejor respuesta al contenido y oferta."})
        if top_friction is not None:
            support_rows.append({"insight_type": "adicional", "objective": "Paginas con friccion", "finding": str(top_friction["entry_page"]), "metric": "frustration_or_abandonment_rate", "value": round(float(top_friction["friction_rate"]), 4), "why_it_matters": "Senala experiencias donde conviene ajustar copy, UX o CTA para no perder demanda."})
        support_table = pd.DataFrame(support_rows)
        answer_lines = [
            "La solucion ya cubre los cinco insights obligatorios del objetivo especifico con calculos locales sobre el dataset.",
            f"Top paginas/productos: {top_page['page']} lidera con {top_page['share_of_sessions']:.1%} de las sesiones estimadas.",
            f"Abandono: la principal pagina critica es {top_abandonment['entry_page']} con {top_abandonment['quick_abandonment_rate']:.1%} de abandono rapido.",
            f"Flujo comun: el recorrido dominante es {top_flow['flow_path']}.",
            f"Interaccion promedio: {sessions['clicks'].mean():.1f} clics, {sessions['page_count'].mean():.1f} paginas y {sessions['session_duration_seconds'].mean():.1f} segundos por sesion.",
            f"Intencion comercial: {int(service_interest['sessions'].sum())} sesiones tocaron pricing/contact/demo/registro." if not service_interest.empty else "Intencion comercial: no se encontraron URLs comerciales en el filtro actual.",
        ]
        if top_source is not None and top_country is not None and top_friction is not None:
            answer_lines.append("Adicionalmente, propongo tres insights de valor para Marketing: fuente de adquisicion dominante, mercados con mejor engagement y paginas con mayor friccion comercial.")
        business_lines = [
            "Esto convierte el copiloto en una solucion end-to-end porque el motor analitico resuelve el calculo, el router interpreta la intencion y la interfaz devuelve una respuesta trazable sin enviar la base completa al LLM.",
            "Los tres insights adicionales recomendados para Marketing son:",
        ]
        if top_source is not None:
            business_lines.append(f"1. Fuente dominante: {top_source['source']} concentra {top_source['share_of_sessions']:.1%} de las sesiones.")
        if top_country is not None:
            business_lines.append(f"2. Mercado con mejor engagement: {top_country['country']} destaca con score promedio {top_country['avg_engagement_score']:.2f}.")
        if top_friction is not None:
            business_lines.append(f"3. Friccion comercial: {top_friction['entry_page']} combina abandono/frustracion por {top_friction['friction_rate']:.1%}.")
        business_lines.append("Cada uno ayuda a decidir inversion de canales, segmentacion geografica y optimizaciones de landing o CTA.")
        return {
            "status": "ok",
            "values": {
                "total_sessions": total_sessions,
                "top_page": str(top_page["page"]),
                "top_exit_page": str(top_exit["exit_page"]),
                "top_flow": str(top_flow["flow_path"]),
                "top_abandonment_page": str(top_abandonment["entry_page"]),
                "commercial_interest_sessions": int(service_interest["sessions"].sum()) if not service_interest.empty else 0,
                "additional_insights_count": int((support_table["insight_type"] == "adicional").sum()),
            },
            "answer": "\n".join(answer_lines),
            "business_interpretation": "\n".join(business_lines),
            "support_table": support_table,
        }


def _estimate_page_visits(sessions: pd.DataFrame) -> pd.DataFrame:
    total_sessions = sessions["session_id"].nunique()
    entry = sessions[["session_id", "entry_page"]].rename(columns={"entry_page": "page"})
    exit_ = sessions[["session_id", "exit_page"]].rename(columns={"exit_page": "page"})
    page_visits = pd.concat([entry, exit_], ignore_index=True).drop_duplicates().groupby("page", dropna=False)["session_id"].nunique().reset_index(name="sessions").sort_values("sessions", ascending=False).reset_index(drop=True)
    page_visits["share_of_sessions"] = page_visits["sessions"] / max(total_sessions, 1)
    return page_visits


def _build_page_metric_table(sessions: pd.DataFrame) -> pd.DataFrame:
    frustration_series = pd.to_numeric(sessions.get("frustration_flag", 0), errors="coerce").fillna(0)
    scroll_series = pd.to_numeric(sessions.get("scroll_depth", pd.NA), errors="coerce")
    registration_series = pd.to_numeric(sessions.get("registration_count", pd.NA), errors="coerce")
    payment_series = pd.to_numeric(sessions.get("payment_count", pd.NA), errors="coerce")
    table = (
        sessions.assign(
            same_page=(sessions["entry_page"] == sessions["exit_page"]).astype(int),
            frustration_flag_numeric=frustration_series,
            scroll_depth_numeric=scroll_series,
            registration_count_numeric=registration_series,
            payment_count_numeric=payment_series,
        )
        .groupby("entry_page", dropna=False)
        .agg(
            entry_sessions=("session_id", "nunique"),
            avg_duration_seconds=("session_duration_seconds", "mean"),
            avg_clicks=("clicks", "mean"),
            avg_pages=("page_count", "mean"),
            avg_scroll_depth=("scroll_depth_numeric", "mean"),
            bounce_proxy_rate=("same_page", "mean"),
            abandono_rapido_rate=("quick_abandonment", "mean"),
            frustration_rate=("frustration_flag_numeric", "mean"),
            engagement_score_avg=("engagement_score", "mean"),
            registration_rate=("registration_count_numeric", "mean"),
            payment_rate=("payment_count_numeric", "mean"),
        )
        .reset_index()
        .rename(columns={"entry_page": "page"})
    )
    exit_sessions = sessions.groupby("exit_page", dropna=False)["session_id"].nunique().reset_index(name="exit_sessions").rename(columns={"exit_page": "page"})
    table = table.merge(exit_sessions, on="page", how="left")
    table["exit_sessions"] = table["exit_sessions"].fillna(0)
    return table


def _build_url_interest_table(sessions: pd.DataFrame, keyword_map: dict[str, list[str]]) -> pd.DataFrame:
    expanded = pd.concat([
        sessions[["session_id", "entry_page"]].rename(columns={"entry_page": "page"}),
        sessions[["session_id", "exit_page"]].rename(columns={"exit_page": "page"}),
    ], ignore_index=True).drop_duplicates()
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
    interest_table = combined.groupby(["interest_type", "page"], dropna=False)["session_id"].nunique().reset_index(name="sessions").sort_values(["sessions", "interest_type"], ascending=[False, True]).reset_index(drop=True)
    group_totals = interest_table.groupby("interest_type")["sessions"].transform("sum")
    interest_table["share_within_interest"] = interest_table["sessions"] / group_totals
    return interest_table


def _build_abandonment_table(sessions: pd.DataFrame) -> pd.DataFrame:
    frustration_series = pd.to_numeric(sessions.get("frustration_flag", 0), errors="coerce").fillna(0)
    return sessions.assign(same_page=(sessions["entry_page"] == sessions["exit_page"]).astype(int), frustration_flag_numeric=frustration_series).groupby("entry_page", dropna=False).agg(sessions=("session_id", "nunique"), quick_abandonment_rate=("quick_abandonment", "mean"), bounce_proxy_rate=("same_page", "mean"), frustration_rate=("frustration_flag_numeric", "mean"), avg_duration_seconds=("session_duration_seconds", "mean"), avg_clicks=("clicks", "mean"), avg_pages=("page_count", "mean")).reset_index().sort_values(["quick_abandonment_rate", "frustration_rate", "bounce_proxy_rate", "sessions"], ascending=[False, False, False, False]).reset_index(drop=True)


def _build_source_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    total_sessions = max(sessions["session_id"].nunique(), 1)
    table = sessions.groupby("source", dropna=False)["session_id"].nunique().reset_index(name="sessions").sort_values("sessions", ascending=False).reset_index(drop=True)
    if not table.empty:
        table["share_of_sessions"] = table["sessions"] / total_sessions
    return table


def _build_country_opportunity_table(sessions: pd.DataFrame) -> pd.DataFrame:
    return sessions.groupby("country", dropna=False).agg(sessions=("session_id", "nunique"), avg_engagement_score=("engagement_score", "mean"), quick_abandonment_rate=("quick_abandonment", "mean")).reset_index().sort_values(["avg_engagement_score", "sessions"], ascending=[False, False]).reset_index(drop=True)


def _build_friction_table(sessions: pd.DataFrame) -> pd.DataFrame:
    frustration_series = pd.to_numeric(sessions.get("frustration_flag", 0), errors="coerce").fillna(0)
    table = sessions.assign(frustration_flag_numeric=frustration_series).groupby("entry_page", dropna=False).agg(sessions=("session_id", "nunique"), quick_abandonment_rate=("quick_abandonment", "mean"), frustration_rate=("frustration_flag_numeric", "mean")).reset_index()
    if table.empty:
        return table
    table["friction_rate"] = table[["quick_abandonment_rate", "frustration_rate"]].max(axis=1)
    return table.sort_values(["friction_rate", "sessions"], ascending=[False, False]).reset_index(drop=True)


def _commercial_interest_keywords() -> dict[str, list[str]]:
    return {
        "registration": ["register", "registration", "signup", "sign-up", "sign_up"],
        "pricing": ["pricing", "price", "precio", "precios"],
        "contact": ["contact", "contacto"],
        "demo": ["demo", "request-demo", "request_demo"],
    }


def _page_metric_label(metric_key: str) -> str:
    labels = {
        "entry_sessions": "sesiones de entrada",
        "avg_duration_seconds": "tiempo promedio",
        "avg_clicks": "clics promedio",
        "avg_pages": "paginas promedio por sesion",
        "avg_scroll_depth": "scroll promedio",
        "bounce_proxy_rate": "tasa proxy de rebote",
        "abandono_rapido_rate": "tasa de abandono rapido",
        "frustration_rate": "tasa de frustracion",
        "engagement_score_avg": "engagement promedio",
        "exit_sessions": "sesiones de salida",
        "registration_rate": "registros promedio",
        "payment_rate": "pagos promedio",
    }
    return labels.get(metric_key, metric_key)


def _coerce_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _format_metric_value(value: Any, metric_key: str) -> str:
    scalar = _coerce_scalar(value)
    if scalar is None or pd.isna(scalar):
        return "N/A"
    if metric_key in {"entry_sessions", "exit_sessions"}:
        return f"{int(round(float(scalar))):,}".replace(",", ".")
    if metric_key in {"bounce_proxy_rate", "abandono_rapido_rate", "frustration_rate"}:
        return f"{float(scalar):.1%}"
    if metric_key in {"avg_duration_seconds", "avg_clicks", "avg_pages", "avg_scroll_depth", "engagement_score_avg", "registration_rate", "payment_rate"}:
        return f"{float(scalar):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return str(scalar)


def _available_metric_names(sessions: pd.DataFrame) -> list[str]:
    metric_map = {"clicks": "clics", "page_count": "paginas visitadas", "session_duration_seconds": "tiempo de navegacion", "scroll_depth": "scroll", "registration_count": "registros", "payment_count": "pagos", "price_value": "precios"}
    available: list[str] = []
    for column, label in metric_map.items():
        if column not in sessions.columns:
            continue
        series = pd.to_numeric(sessions[column], errors="coerce")
        if series.notna().any() and not series.fillna(0).eq(0).all():
            available.append(label)
    return available


def _missing_metric_names(sessions: pd.DataFrame) -> list[str]:
    metric_map = {"scroll_depth": "scroll", "registration_count": "registros", "payment_count": "pagos", "price_value": "precios"}
    missing: list[str] = []
    for column, label in metric_map.items():
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
    match = re.search(r"(?:producto|pagina|page)\s+([^\s,;:]+)", question, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _no_data_result(message: str) -> dict[str, Any]:
    return {
        "status": "no_data",
        "values": {},
        "answer": message,
        "business_interpretation": "No hay suficiente informacion para responder con confianza.",
        "support_table": pd.DataFrame(),
    }
