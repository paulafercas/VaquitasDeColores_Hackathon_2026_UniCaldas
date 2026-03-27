from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SESSION_SCHEMA_ALIASES = {
    "session_id": ["session_id", "id_usuario_clarity", "session", "sessionid"],
    "timestamp": ["timestamp", "fecha_hora", "datetime"],
    "entry_page": ["direccion_url_entrada", "entry_page", "landing_page", "page"],
    "exit_page": ["direccion_url_salida", "exit_page", "salida", "last_page"],
    "source": ["referente", "source", "referrer", "referer"],
    "country": ["pais", "country"],
    "device": ["dispositivo", "device"],
    "browser": ["explorador", "browser"],
    "session_duration_seconds": ["duracion_sesion_segundos", "session_duration_seconds"],
    "quick_abandonment": ["abandono_rapido", "quick_abandonment", "bounce"],
    "frustration_flag": ["posible_frustracion", "frustration_flag"],
    "engagement_score": ["standarized_engagement_score", "engagement_score"],
    "page_count": ["recuento_paginas", "page_count"],
    "clicks": ["clics_sesion", "clicks", "session_clicks"],
    "clicks_per_page": ["clicks_por_pagina", "clicks_per_page"],
    "time_per_page": ["tiempo_por_pagina", "time_per_page"],
    "product": ["product", "producto"],
    "scroll_depth": ["scroll", "scroll_pct", "scroll_percent"],
    "registration_count": ["registros", "registro", "registrations", "registration_count"],
    "payment_count": ["pagos", "payment_count", "payments"],
    "price_value": ["precio", "price", "price_value"],
}

EVENT_SCHEMA_ALIASES = {
    "session_id": ["session_id", "id_usuario_clarity", "session", "sessionid"],
    "timestamp": ["timestamp", "fecha_hora", "datetime"],
    "page": ["page", "url", "direccion_url", "page_url"],
    "event": ["event", "interaction_type", "tipo_evento"],
    "product": ["product", "producto"],
    "exit_page": ["exit_page", "is_exit", "salida"],
    "scroll": ["scroll", "scroll_pct", "scroll_percent"],
    "price_value": ["precio", "price", "price_value"],
    "country": ["pais", "country"],
    "device": ["dispositivo", "device"],
    "browser": ["explorador", "browser"],
}


@dataclass(frozen=True)
class SchemaMetadata:
    dataset_type: str
    source_columns: dict[str, str]


def _first_existing_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def detect_dataset_type(df: pd.DataFrame) -> str:
    if {"direccion_url_entrada", "direccion_url_salida"}.issubset(df.columns):
        return "session_summary"
    if {"session_id", "page", "event"}.issubset(df.columns):
        return "event_log"
    if {"id_usuario_clarity", "timestamp", "page"}.issubset(df.columns):
        return "event_log"
    return "unknown"


def map_to_canonical_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, SchemaMetadata]:
    dataset_type = detect_dataset_type(df)
    if dataset_type == "session_summary":
        return _map_session_summary(df)
    if dataset_type == "event_log":
        return _map_event_log(df)
    raise ValueError(
        "Unsupported dataset structure. Expected session summary columns or event-log columns."
    )


def _map_session_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, SchemaMetadata]:
    mapped_columns: dict[str, str] = {}
    canonical = pd.DataFrame(index=df.index)

    for target, aliases in SESSION_SCHEMA_ALIASES.items():
        source_column = _first_existing_column(df, aliases)
        if source_column:
            mapped_columns[target] = source_column
            canonical[target] = df[source_column]

    if "timestamp" not in canonical.columns and {"fecha", "hora"}.issubset(df.columns):
        canonical["timestamp"] = pd.to_datetime(
            df["fecha"].astype(str).str.strip() + " " + df["hora"].astype(str).str.strip(),
            format="%m/%d/%Y %H:%M",
            errors="coerce",
        )
        mapped_columns["timestamp"] = "fecha+hora"
    else:
        canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], errors="coerce")

    canonical["session_id"] = canonical.get(
        "session_id", pd.Series(range(len(df)), index=df.index)
    ).astype(str)
    canonical["entry_page"] = canonical.get(
        "entry_page", pd.Series("Unknown", index=df.index)
    ).fillna("Unknown")
    canonical["exit_page"] = canonical.get("exit_page", canonical["entry_page"]).fillna(
        canonical["entry_page"]
    )
    canonical["source"] = canonical.get(
        "source", pd.Series("Direct/Unknown", index=df.index)
    ).fillna("Direct/Unknown")
    canonical["country"] = canonical.get(
        "country", pd.Series("Unknown", index=df.index)
    ).fillna("Unknown")
    canonical["device"] = canonical.get(
        "device", pd.Series("Unknown", index=df.index)
    ).fillna("Unknown")
    canonical["browser"] = canonical.get(
        "browser", pd.Series("Unknown", index=df.index)
    ).fillna("Unknown")
    canonical["session_duration_seconds"] = pd.to_numeric(
        canonical.get("session_duration_seconds", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    canonical["quick_abandonment"] = pd.to_numeric(
        canonical.get("quick_abandonment", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    canonical["frustration_flag"] = pd.to_numeric(
        canonical.get("frustration_flag", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    canonical["engagement_score"] = pd.to_numeric(
        canonical.get("engagement_score", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    canonical["page_count"] = pd.to_numeric(
        canonical.get("page_count", pd.Series(1, index=df.index)),
        errors="coerce",
    ).fillna(1)
    canonical["clicks"] = pd.to_numeric(
        canonical.get("clicks", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    canonical["clicks_per_page"] = pd.to_numeric(
        canonical.get("clicks_per_page", pd.Series(pd.NA, index=df.index)),
        errors="coerce",
    )
    canonical["time_per_page"] = pd.to_numeric(
        canonical.get("time_per_page", pd.Series(pd.NA, index=df.index)),
        errors="coerce",
    )
    canonical["scroll_depth"] = pd.to_numeric(
        canonical.get("scroll_depth", pd.Series(pd.NA, index=df.index)),
        errors="coerce",
    )
    canonical["registration_count"] = pd.to_numeric(
        canonical.get("registration_count", pd.Series(pd.NA, index=df.index)),
        errors="coerce",
    )
    canonical["payment_count"] = pd.to_numeric(
        canonical.get("payment_count", pd.Series(pd.NA, index=df.index)),
        errors="coerce",
    )
    canonical["price_value"] = pd.to_numeric(
        canonical.get("price_value", pd.Series(pd.NA, index=df.index)),
        errors="coerce",
    )
    canonical["product"] = canonical.get("product", canonical["entry_page"]).fillna(
        canonical["entry_page"]
    )
    canonical["page_or_product"] = canonical["product"].where(
        canonical["product"].astype(str).str.strip().ne(""),
        canonical["entry_page"],
    )
    canonical["flow_path"] = (
        canonical["source"].astype(str)
        + " -> "
        + canonical["entry_page"].astype(str)
        + " -> "
        + canonical["exit_page"].astype(str)
    )

    metadata = SchemaMetadata(
        dataset_type="session_summary",
        source_columns=mapped_columns,
    )
    return canonical, metadata


def _map_event_log(df: pd.DataFrame) -> tuple[pd.DataFrame, SchemaMetadata]:
    mapped_columns: dict[str, str] = {}
    event_df = pd.DataFrame(index=df.index)

    for target, aliases in EVENT_SCHEMA_ALIASES.items():
        source_column = _first_existing_column(df, aliases)
        if source_column:
            mapped_columns[target] = source_column
            event_df[target] = df[source_column]

    required = {"session_id", "timestamp", "page"}
    missing = [name for name in required if name not in event_df.columns]
    if missing:
        raise ValueError(
            f"Event dataset is missing required columns after mapping: {', '.join(missing)}"
        )

    event_df["timestamp"] = pd.to_datetime(event_df["timestamp"], errors="coerce")
    event_df = event_df.sort_values(["session_id", "timestamp"], kind="stable")
    grouped = event_df.groupby("session_id", dropna=False)

    canonical = grouped.agg(
        timestamp=("timestamp", "min"),
        entry_page=("page", "first"),
        exit_page=("page", "last"),
    ).reset_index()

    for source_column, target_column in {
        "country": "country",
        "device": "device",
        "browser": "browser",
        "product": "product",
    }.items():
        if source_column in event_df.columns:
            canonical[target_column] = grouped[source_column].first().values
        else:
            canonical[target_column] = "Unknown" if target_column != "product" else canonical["entry_page"]

    canonical["source"] = "Direct/Unknown"
    canonical["session_duration_seconds"] = (
        grouped["timestamp"].max() - grouped["timestamp"].min()
    ).dt.total_seconds().fillna(0).values
    canonical["page_count"] = grouped["page"].nunique().values
    if "event" in event_df.columns:
        canonical["clicks"] = grouped["event"].apply(
            lambda series: series.astype(str).str.contains("click", case=False, na=False).sum()
        ).values
    else:
        canonical["clicks"] = 0
    canonical["clicks_per_page"] = canonical["clicks"] / canonical["page_count"].replace(0, pd.NA)
    canonical["time_per_page"] = canonical["session_duration_seconds"] / canonical["page_count"].replace(0, pd.NA)
    if "scroll" in event_df.columns:
        canonical["scroll_depth"] = pd.to_numeric(grouped["scroll"].max(), errors="coerce").values
    else:
        canonical["scroll_depth"] = pd.NA
    if "event" in event_df.columns:
        event_text = grouped["event"].apply(lambda series: " ".join(series.astype(str).tolist()))
        canonical["registration_count"] = event_text.str.count(r"register|registro|sign[\s_-]?up", flags=0).values
        canonical["payment_count"] = event_text.str.count(r"payment|pago|checkout|purchase", flags=0).values
    else:
        canonical["registration_count"] = pd.NA
        canonical["payment_count"] = pd.NA
    if "price_value" in event_df.columns:
        canonical["price_value"] = pd.to_numeric(grouped["price_value"].max(), errors="coerce").values
    else:
        canonical["price_value"] = pd.NA
    canonical["quick_abandonment"] = (canonical["page_count"] <= 1).astype(int)
    canonical["frustration_flag"] = 0
    canonical["engagement_score"] = 0
    canonical["page_or_product"] = canonical["product"].fillna(canonical["entry_page"])
    canonical["flow_path"] = (
        canonical["source"].astype(str)
        + " -> "
        + canonical["entry_page"].astype(str)
        + " -> "
        + canonical["exit_page"].astype(str)
    )

    metadata = SchemaMetadata(
        dataset_type="event_log",
        source_columns=mapped_columns,
    )
    return canonical, metadata
