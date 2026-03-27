from __future__ import annotations

import unittest

import pandas as pd

from analytics_core.data_loader import AnalyticsBundle
from analytics_core.data_loader import _exclude_internal_traffic
from analytics_core.query_service import QueryService
from analytics_core.schema_mapper import SchemaMetadata, map_to_canonical_sessions
from LLM_Workflow.intent_router import route
from LLM_Workflow.response_service import ResponseService
from config.settings import get_settings


class QueryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        sessions = pd.DataFrame(
            {
                "session_id": ["1", "2", "3"],
                "timestamp": pd.to_datetime(["2026-03-10", "2026-03-10", "2026-03-11"]),
                "entry_page": ["home", "home", "product-a"],
                "exit_page": ["product-a/register", "pricing", "checkout/request-demo"],
                "source": ["External", "Direct/Unknown", "External"],
                "country": ["Colombia", "Mexico", "Colombia"],
                "device": ["PC", "Mobile", "PC"],
                "browser": ["Chrome", "Chrome", "Edge"],
                "session_duration_seconds": [100, 50, 120],
                "quick_abandonment": [0, 1, 0],
                "frustration_flag": [0, 0, 0],
                "engagement_score": [1.0, 0.3, 1.5],
                "page_count": [2, 1, 3],
                "clicks": [4, 1, 5],
                "clicks_per_page": [2.0, 1.0, 1.67],
                "time_per_page": [50.0, 50.0, 40.0],
                "scroll_depth": [80, 20, 90],
                "registration_count": [0, 0, 1],
                "payment_count": [0, 0, 0],
                "price_value": [pd.NA, pd.NA, pd.NA],
                "product": ["home", "home", "product-a"],
                "page_or_product": ["home", "home", "product-a"],
                "flow_path": [
                    "External -> home -> product-a",
                    "Direct/Unknown -> home -> home",
                    "External -> product-a -> checkout",
                ],
            }
        )
        self.service = QueryService(
            bundle=AnalyticsBundle(
                canonical_sessions=sessions,
                page_metrics=pd.DataFrame(),
                top_paths=pd.DataFrame(),
                source_to_entry=pd.DataFrame(),
                entry_to_exit=pd.DataFrame(),
                schema_metadata=SchemaMetadata(dataset_type="session_summary", source_columns={}),
            ),
            settings=get_settings(),
        )

    def test_top_product_or_page(self) -> None:
        result = self.service.answer_structured_query(
            "cual fue la pagina mas vista",
            {"intent": "top_product_or_page", "filters": {}, "confidence": 0.9},
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("share_of_sessions", result["values"])

    def test_page_metric_ranking_top_n(self) -> None:
        route_result = route("cuales son las 2 paginas mas visitadas", self.service.bundle)
        result = self.service.answer_structured_query(
            "cuales son las 2 paginas mas visitadas",
            route_result,
        )
        self.assertEqual(route_result["intent"], "page_metric_ranking")
        self.assertEqual(route_result["filters"]["page_metric"], "entry_sessions")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["values"]["top_n"], 2)
        self.assertEqual(len(result["support_table"]), 2)

    def test_page_metric_ranking_least_visited(self) -> None:
        route_result = route("cuales son las paginas con menos visitas", self.service.bundle)
        result = self.service.answer_structured_query(
            "cuales son las paginas con menos visitas",
            route_result,
        )
        self.assertEqual(route_result["intent"], "page_metric_ranking")
        self.assertEqual(route_result["filters"]["sort_order"], "asc")
        self.assertEqual(result["status"], "ok")
        self.assertIn("menor sesiones de entrada", result["answer"])

    def test_page_metric_ranking_time_spent(self) -> None:
        route_result = route("en que paginas se gastaron mas tiempo los usuarios", self.service.bundle)
        result = self.service.answer_structured_query(
            "en que paginas se gastaron mas tiempo los usuarios",
            route_result,
        )
        self.assertEqual(route_result["intent"], "page_metric_ranking")
        self.assertEqual(route_result["filters"]["page_metric"], "avg_duration_seconds")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["values"]["top_page"], "product-a")

    def test_page_metric_ranking_exit_sessions(self) -> None:
        route_result = route("cuales son las 5 paginas con mayor sesiones de salida", self.service.bundle)
        result = self.service.answer_structured_query(
            "cuales son las 5 paginas con mayor sesiones de salida",
            route_result,
        )
        self.assertEqual(route_result["intent"], "page_metric_ranking")
        self.assertEqual(route_result["filters"]["page_metric"], "exit_sessions")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["values"]["top_page"], "home")

    def test_segment_comparison(self) -> None:
        result = self.service.answer_structured_query(
            "compara por pais",
            {
                "intent": "segment_comparison",
                "filters": {"segment_dimension": "country"},
                "confidence": 0.9,
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["values"]["segment_dimension"], "country")

    def test_page_abandonment_ranking(self) -> None:
        result = self.service.answer_structured_query(
            "cuales son las paginas donde hay mayor abandono",
            {"intent": "page_abandonment_ranking", "filters": {}, "confidence": 0.9},
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("top_abandonment_page", result["values"])

    def test_interaction_overview(self) -> None:
        result = self.service.answer_structured_query(
            "analisis de interaccion usuarios clics paginas scroll tiempo registros pagos precios",
            {"intent": "interaction_overview", "filters": {}, "confidence": 0.9},
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("available_metrics", result["values"])
        self.assertIn("missing_metrics", result["values"])

    def test_registration_interest(self) -> None:
        result = self.service.answer_structured_query(
            "que tantos usuarios se han registrado",
            {"intent": "registration_interest", "filters": {}, "confidence": 0.9},
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["values"]["registration_sessions"], 0)

    def test_service_interest(self) -> None:
        result = self.service.answer_structured_query(
            "que tantos usuarios se han interesado por los servicios",
            {"intent": "service_interest", "filters": {}, "confidence": 0.9},
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["values"]["service_interest_sessions"], 0)

    def test_hackathon_objective_summary(self) -> None:
        route_result = route(
            "Resume el objetivo especifico, la solucion end-to-end y los insights adicionales para marketing",
            self.service.bundle,
        )
        result = self.service.answer_structured_query(
            "Resume el objetivo especifico, la solucion end-to-end y los insights adicionales para marketing",
            route_result,
        )
        self.assertEqual(route_result["intent"], "hackathon_objective_summary")
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["values"]["additional_insights_count"], 3)

    def test_response_service_skips_llm_for_analytics(self) -> None:
        response_service = ResponseService(query_service=self.service)
        result = response_service.generate_answer("cual fue la pagina mas vista")
        self.assertEqual(result["chat_mode"], "analytics")
        self.assertEqual(result["llm_status"], "skipped_analytics")
        self.assertIn("Interpretacion:", result["final_answer"])

    def test_points_criticos_de_abandono_routes_to_abandonment(self) -> None:
        route_result = route("cuales son los puntos criticos de abandono", self.service.bundle)
        result = self.service.answer_structured_query(
            "cuales son los puntos criticos de abandono",
            route_result,
        )
        self.assertEqual(route_result["intent"], "page_abandonment_ranking")
        self.assertEqual(result["status"], "ok")
        self.assertIn("frustracion", result["answer"])

    def test_interaccion_promedio_por_pagina(self) -> None:
        route_result = route("cual es la interaccion promedio por pagina", self.service.bundle)
        result = self.service.answer_structured_query(
            "cual es la interaccion promedio por pagina",
            route_result,
        )
        self.assertEqual(route_result["intent"], "page_interaction_profile")
        self.assertEqual(result["status"], "ok")
        self.assertIn("scroll", result["answer"])

    def test_patrones_basicos_de_conversion(self) -> None:
        route_result = route(
            "cuales son los patrones basicos de conversion o intencion en pricing, contacto o producto",
            self.service.bundle,
        )
        result = self.service.answer_structured_query(
            "cuales son los patrones basicos de conversion o intencion en pricing, contacto o producto",
            route_result,
        )
        self.assertEqual(route_result["intent"], "service_interest")
        self.assertEqual(result["status"], "ok")
        self.assertIn("patrones basicos de conversion", result["answer"])

    def test_patrones_basicos_de_conversion_without_keywords(self) -> None:
        route_result = route(
            "cuales son los patrones basicos de conversion o intencion",
            self.service.bundle,
        )
        self.assertEqual(route_result["intent"], "service_interest")

    def test_flujos_mas_comunes(self) -> None:
        route_result = route("cuales son los flujos de navegacion mas comunes", self.service.bundle)
        result = self.service.answer_structured_query(
            "cuales son los flujos de navegacion mas comunes",
            route_result,
        )
        self.assertEqual(route_result["intent"], "top_navigation_flow")
        self.assertEqual(result["status"], "ok")
        self.assertIn("1.", result["answer"])

    def test_url_normalization_in_schema_mapper(self) -> None:
        raw = pd.DataFrame(
            {
                "id_usuario_clarity": ["abc"],
                "fecha": ["03/10/2026"],
                "hora": ["10:00"],
                "direccion_url_entrada": ["https://CloudLabsLearning.com/request-demo/?err=SUBSCRIPTION_NOT_FOUND#login"],
                "direccion_url_salida": ["https://cloudlabslearning.com/elementary-school?%2Fregistration="],
                "referente": ["https://www.google.com/search?q=cloudlabs"],
            }
        )
        canonical, _ = map_to_canonical_sessions(raw)
        self.assertEqual(canonical.iloc[0]["entry_page"], "https://cloudlabslearning.com/request-demo")
        self.assertEqual(canonical.iloc[0]["exit_page"], "https://cloudlabslearning.com/elementary-school")
        self.assertEqual(canonical.iloc[0]["source"], "https://www.google.com/search")

    def test_semantic_path_normalization(self) -> None:
        raw = pd.DataFrame(
            {
                "id_usuario_clarity": ["abc", "def"],
                "fecha": ["03/10/2026", "03/10/2026"],
                "hora": ["10:00", "10:05"],
                "direccion_url_entrada": [
                    "https://cloudlabslearning.com/register/ACDC632DA9",
                    "https://cloudlabslearning.com/auth/session/62c33b3e-f59e-4959-9268-4561bf24c8e1",
                ],
                "direccion_url_salida": [
                    "https://cloudlabslearning.com/register/BF83F8C319 BF83F8C319 BF83F8C319",
                    "https://cloudlabslearning.com/verify/F9472E",
                ],
                "referente": ["Direct/Unknown", "Direct/Unknown"],
            }
        )
        canonical, _ = map_to_canonical_sessions(raw)
        self.assertEqual(canonical.iloc[0]["entry_page"], "https://cloudlabslearning.com/register")
        self.assertEqual(canonical.iloc[0]["exit_page"], "https://cloudlabslearning.com/register")
        self.assertEqual(canonical.iloc[1]["entry_page"], "https://cloudlabslearning.com/auth/session")
        self.assertEqual(canonical.iloc[1]["exit_page"], "https://cloudlabslearning.com/verify")

    def test_internal_localhost_traffic_is_excluded(self) -> None:
        sessions = pd.DataFrame(
            {
                "session_id": ["1", "2", "3", "4"],
                "entry_page": ["http://localhost:3000/", "https://cloudlabslearning.com/", "https://cloudlabslearning.com/auth/session", "https://cloudlabslearning.com/verify"],
                "exit_page": ["https://cloudlabslearning.com/request-demo", "https://cloudlabslearning.com/request-demo", "https://cloudlabslearning.com/request-demo", "https://cloudlabslearning.com/request-demo"],
                "source": ["Direct/Unknown", "https://www.google.com/search", "Direct/Unknown", "Direct/Unknown"],
            }
        )
        filtered = _exclude_internal_traffic(sessions)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["entry_page"], "https://cloudlabslearning.com/")

    def test_paginas_mas_buscadas_por_pais(self) -> None:
        route_result = route("cuales son las paginas mas buscadas en colombia", self.service.bundle)
        result = self.service.answer_structured_query(
            "cuales son las paginas mas buscadas en colombia",
            route_result,
        )
        self.assertEqual(route_result["intent"], "page_metric_ranking")
        self.assertEqual(route_result["filters"]["page_metric"], "entry_sessions")
        self.assertEqual(route_result["filters"]["country"], "Colombia")
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
