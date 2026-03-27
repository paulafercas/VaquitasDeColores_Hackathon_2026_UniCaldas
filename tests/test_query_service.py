from __future__ import annotations

import unittest

import pandas as pd

from analytics_core.data_loader import AnalyticsBundle
from analytics_core.query_service import QueryService
from analytics_core.schema_mapper import SchemaMetadata
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


if __name__ == "__main__":
    unittest.main()
