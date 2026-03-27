from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

@dataclass(frozen=True)
class Settings:
    root_dir: Path
    cleaned_sessions_path: Path
    page_metrics_path: Path
    top_paths_path: Path
    source_to_entry_path: Path
    entry_to_exit_path: Path
    gemini_api_key: str | None
    gemini_model: str
    gemini_temperature: float
    max_table_rows: int
    llm_analytics_enabled: bool


def _load_local_secrets(root_dir: Path) -> dict[str, Any]:
    secrets_path = root_dir / "secrets.toml"
    if not secrets_path.exists():
        return {}

    with secrets_path.open("rb") as secrets_file:
        return tomllib.load(secrets_file)


def _get_secret(root_dir: Path, key: str, default: str | None = None) -> str | None:
    env_value = os.getenv(key)
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(key)
    except Exception:
        secret_value = None
    if secret_value:
        return str(secret_value)

    return _load_local_secrets(root_dir).get(key, default)


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[1]
    return Settings(
        root_dir=root_dir,
        cleaned_sessions_path=root_dir
        / "Motor_Analitico"
        / "outputs"
        / "data_cleaning"
        / "1_Data_Recordings_clean.csv",
        page_metrics_path=root_dir
        / "Motor_Analitico"
        / "outputs"
        / "flujo_navegacion"
        / "page_metrics.csv",
        top_paths_path=root_dir
        / "Motor_Analitico"
        / "outputs"
        / "flujo_navegacion"
        / "top_paths.csv",
        source_to_entry_path=root_dir
        / "Motor_Analitico"
        / "outputs"
        / "flujo_navegacion"
        / "source_to_entry_transitions.csv",
        entry_to_exit_path=root_dir
        / "Motor_Analitico"
        / "outputs"
        / "flujo_navegacion"
        / "entry_to_exit_transitions.csv",
        gemini_api_key=_get_secret(root_dir, "GEMINI_API_KEY"),
        gemini_model=_get_secret(root_dir, "GEMINI_MODEL", "gemini-2.5-flash-lite") or "gemini-2.5-flash-lite",
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
        max_table_rows=int(os.getenv("MAX_TABLE_ROWS", "8")),
        llm_analytics_enabled=os.getenv("LLM_ANALYTICS_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
    )
