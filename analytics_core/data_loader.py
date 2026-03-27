from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics_core.schema_mapper import SchemaMetadata, map_to_canonical_sessions
from config.settings import Settings, get_settings


@dataclass(frozen=True)
class AnalyticsBundle:
    canonical_sessions: pd.DataFrame
    page_metrics: pd.DataFrame
    top_paths: pd.DataFrame
    source_to_entry: pd.DataFrame
    entry_to_exit: pd.DataFrame
    schema_metadata: SchemaMetadata


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_analytics_bundle(settings: Settings | None = None) -> AnalyticsBundle:
    settings = settings or get_settings()
    sessions_df = pd.read_csv(settings.cleaned_sessions_path)
    canonical_sessions, schema_metadata = map_to_canonical_sessions(sessions_df)
    canonical_sessions["timestamp"] = pd.to_datetime(
        canonical_sessions["timestamp"], errors="coerce"
    )

    return AnalyticsBundle(
        canonical_sessions=canonical_sessions,
        page_metrics=_read_csv_if_exists(settings.page_metrics_path),
        top_paths=_read_csv_if_exists(settings.top_paths_path),
        source_to_entry=_read_csv_if_exists(settings.source_to_entry_path),
        entry_to_exit=_read_csv_if_exists(settings.entry_to_exit_path),
        schema_metadata=schema_metadata,
    )
