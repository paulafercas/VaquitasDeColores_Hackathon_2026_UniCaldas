from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import streamlit as st

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
        gemini_api_key=st.secrets["GEMINI_API_KEY"],
        gemini_model= st.secrets["GEMINI_MODEL"],
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
        max_table_rows=int(os.getenv("MAX_TABLE_ROWS", "8")),
    )
