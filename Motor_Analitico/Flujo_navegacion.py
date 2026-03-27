from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "1_Data_Recordings.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "flujo_navegacion"
DIRECT_UNKNOWN = "Direct/Unknown"

REQUIRED_COLUMNS = {
    "fecha",
    "hora",
    "direccion_url_entrada",
    "direccion_url_salida",
    "referente",
    "id_usuario_clarity",
    "pais",
    "dispositivo",
    "explorador",
    "recuento_paginas",
    "clics_sesion",
    "duracion_sesion_segundos",
    "abandono_rapido",
    "standarized_engagement_score",
    "posible_frustracion",
}

NUMERIC_COLUMNS = [
    "recuento_paginas",
    "clics_sesion",
    "duracion_sesion_segundos",
    "abandono_rapido",
    "standarized_engagement_score",
    "posible_frustracion",
]

def load_data(input_path: Path) -> pd.DataFrame:
    """Load the source dataset and validate the expected schema."""
    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")

    df["timestamp"] = pd.to_datetime(
        df["fecha"].astype(str).str.strip() + " " + df["hora"].astype(str).str.strip(),
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        invalid_rows = int(df["timestamp"].isna().sum())
        raise ValueError(
            f"Unable to parse timestamp for {invalid_rows} rows using format %m/%d/%Y %H:%M."
        )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def build_session_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Add source, entry, exit, and flow helper columns without URL normalization."""
    enriched = df.copy()
    enriched["ref_source"] = (
        enriched["referente"].fillna("").astype(str).str.strip().replace("", DIRECT_UNKNOWN)
    )
    enriched["url_entrada"] = enriched["direccion_url_entrada"].fillna("").astype(str).str.strip()
    enriched["url_salida"] = enriched["direccion_url_salida"].fillna("").astype(str).str.strip()
    enriched["same_page_session"] = (enriched["url_entrada"] == enriched["url_salida"]).astype(int)
    enriched["flow_path"] = (
        enriched["ref_source"]
        + " -> "
        + enriched["url_entrada"]
        + " -> "
        + enriched["url_salida"]
    )
    return enriched


def compute_transitions(
    df: pd.DataFrame,
    source_column: str,
    target_column: str,
    value_name: str = "sessions",
) -> pd.DataFrame:
    transition = (
        df.groupby([source_column, target_column], dropna=False)
        .size()
        .reset_index(name=value_name)
        .sort_values(value_name, ascending=False)
        .reset_index(drop=True)
    )
    return transition


def compute_combined_paths(df: pd.DataFrame) -> pd.DataFrame:
    path_metrics = (
        df.groupby(
            ["ref_source", "url_entrada", "url_salida"],
            dropna=False,
        )
        .agg(
            sessions=("id_usuario_clarity", "size"),
            avg_duration_seconds=("duracion_sesion_segundos", "mean"),
            avg_clicks=("clics_sesion", "mean"),
            avg_pages=("recuento_paginas", "mean"),
            bounce_proxy_rate=("same_page_session", "mean"),
            abandono_rapido_rate=("abandono_rapido", "mean"),
            frustration_rate=("posible_frustracion", "mean"),
            engagement_score_avg=("standarized_engagement_score", "mean"),
        )
        .reset_index()
        .sort_values("sessions", ascending=False)
        .reset_index(drop=True)
    )
    path_metrics["flow_path"] = (
        path_metrics["ref_source"]
        + " -> "
        + path_metrics["url_entrada"]
        + " -> "
        + path_metrics["url_salida"]
    )
    return path_metrics


def compute_page_metrics(df: pd.DataFrame) -> pd.DataFrame:
    entry_metrics = (
        df.groupby("url_entrada", dropna=False)
        .agg(
            entry_sessions=("id_usuario_clarity", "size"),
            avg_duration_seconds=("duracion_sesion_segundos", "mean"),
            avg_clicks=("clics_sesion", "mean"),
            avg_pages=("recuento_paginas", "mean"),
            bounce_proxy_rate=("same_page_session", "mean"),
            abandono_rapido_rate=("abandono_rapido", "mean"),
            frustration_rate=("posible_frustracion", "mean"),
            engagement_score_avg=("standarized_engagement_score", "mean"),
        )
        .reset_index()
        .rename(columns={"url_entrada": "page"})
    )

    exit_counts = (
        df.groupby("url_salida", dropna=False)
        .size()
        .reset_index(name="exit_sessions")
        .rename(columns={"url_salida": "page"})
    )

    page_metrics = (
        entry_metrics.merge(exit_counts, on="page", how="outer")
        .fillna(
            {
                "entry_sessions": 0,
                "exit_sessions": 0,
                "avg_duration_seconds": 0,
                "avg_clicks": 0,
                "avg_pages": 0,
                "bounce_proxy_rate": 0,
                "abandono_rapido_rate": 0,
                "frustration_rate": 0,
                "engagement_score_avg": 0,
            }
        )
        .sort_values("entry_sessions", ascending=False)
        .reset_index(drop=True)
    )

    page_metrics["entry_sessions"] = page_metrics["entry_sessions"].astype(int)
    page_metrics["exit_sessions"] = page_metrics["exit_sessions"].astype(int)

    return page_metrics


def compute_segment_summary(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    summary = (
        df.groupby(segment_column, dropna=False)
        .agg(
            sessions=("id_usuario_clarity", "size"),
            unique_users=("id_usuario_clarity", "nunique"),
            avg_duration_seconds=("duracion_sesion_segundos", "mean"),
            avg_clicks=("clics_sesion", "mean"),
            avg_pages=("recuento_paginas", "mean"),
            bounce_proxy_rate=("same_page_session", "mean"),
            abandono_rapido_rate=("abandono_rapido", "mean"),
            frustration_rate=("posible_frustracion", "mean"),
            engagement_score_avg=("standarized_engagement_score", "mean"),
        )
        .reset_index()
        .sort_values("sessions", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def compute_segment_transitions(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    summary = (
        df.groupby([segment_column, "url_entrada", "url_salida"], dropna=False)
        .size()
        .reset_index(name="sessions")
        .sort_values([segment_column, "sessions"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def build_sankey_edges(
    source_to_entry: pd.DataFrame, entry_to_exit: pd.DataFrame
) -> pd.DataFrame:
    sankey_edges = pd.concat(
        [
            source_to_entry.rename(
                columns={
                    "ref_source": "source",
                    "url_entrada": "target",
                    "sessions": "value",
                }
            )[["source", "target", "value"]],
            entry_to_exit.rename(
                columns={
                    "url_entrada": "source",
                    "url_salida": "target",
                    "sessions": "value",
                }
            )[["source", "target", "value"]],
        ],
        ignore_index=True,
    )
    return sankey_edges


def build_transition_matrix(
    transitions: pd.DataFrame, row_col: str, column_col: str
) -> pd.DataFrame:
    matrix = transitions.pivot_table(
        index=row_col,
        columns=column_col,
        values="sessions",
        aggfunc="sum",
        fill_value=0,
    )
    return matrix.reset_index()


def select_top_rows(df: pd.DataFrame, column_name: str, top_n: int = 20) -> pd.DataFrame:
    return df.sort_values(column_name, ascending=False).head(top_n).reset_index(drop=True)


def create_charts(
    page_metrics: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> list[Path]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    generated_paths: list[Path] = []

    chart_specs = [
        (
            select_top_rows(page_metrics, "entry_sessions", top_n),
            "entry_sessions",
            "Top Landing Pages",
            chart_dir / "top_landing_pages.png",
        ),
        (
            select_top_rows(page_metrics, "exit_sessions", top_n),
            "exit_sessions",
            "Top Exit Pages",
            chart_dir / "top_exit_pages.png",
        ),
        (
            select_top_rows(page_metrics, "abandono_rapido_rate", top_n),
            "abandono_rapido_rate",
            "Highest Quick-Abandonment Entry Pages",
            chart_dir / "quick_abandonment_by_entry.png",
        ),
    ]

    for chart_df, value_column, title, output_path in chart_specs:
        if chart_df.empty:
            continue

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=chart_df,
            x=value_column,
            y="page",
            hue="page",
            dodge=False,
            palette="Blues_r",
            legend=False,
        )
        ax.set_title(title)
        ax.set_xlabel(value_column.replace("_", " ").title())
        ax.set_ylabel("Page")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        generated_paths.append(output_path)

    return generated_paths


def validate_outputs(
    sessions: pd.DataFrame,
    source_to_entry: pd.DataFrame,
    entry_to_exit: pd.DataFrame,
) -> pd.DataFrame:
    validations = [
        {
            "check_name": "session_count_matches_source_to_entry",
            "passed": int(source_to_entry["sessions"].sum() == len(sessions)),
            "details": f"{int(source_to_entry['sessions'].sum())} grouped vs {len(sessions)} sessions",
        },
        {
            "check_name": "session_count_matches_entry_to_exit",
            "passed": int(entry_to_exit["sessions"].sum() == len(sessions)),
            "details": f"{int(entry_to_exit['sessions'].sum())} grouped vs {len(sessions)} sessions",
        },
        {
            "check_name": "same_page_session_binary",
            "passed": int(sessions["same_page_session"].isin([0, 1]).all()),
            "details": "All same_page_session values are binary.",
        },
        {
            "check_name": "timestamp_not_null",
            "passed": int(sessions["timestamp"].notna().all()),
            "details": f"{int(sessions['timestamp'].notna().sum())} non-null timestamps",
        },
    ]
    return pd.DataFrame(validations)

def export_outputs(output_dir: Path, artifacts: dict[str, pd.DataFrame]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files: list[Path] = []

    for filename, dataframe in artifacts.items():
        output_path = output_dir / filename
        dataframe.to_csv(output_path, index=False)
        exported_files.append(output_path)

    return exported_files


def print_summary(
    page_metrics: pd.DataFrame,
    paths: pd.DataFrame,
    source_to_entry: pd.DataFrame,
    entry_to_exit: pd.DataFrame,
) -> None:
    print("\nNavigation workflow completed.\n")
    print("Top landing pages:")
    print(select_top_rows(page_metrics, "entry_sessions", 10)[["page", "entry_sessions"]].to_string(index=False))

    print("\nTop exit pages:")
    print(select_top_rows(page_metrics, "exit_sessions", 10)[["page", "exit_sessions"]].to_string(index=False))

    print("\nTop source -> entry transitions:")
    print(
        select_top_rows(source_to_entry, "sessions", 10)[
            ["ref_source", "url_entrada", "sessions"]
        ].to_string(index=False)
    )

    print("\nTop entry -> exit transitions:")
    print(
        select_top_rows(entry_to_exit, "sessions", 10)[
            ["url_entrada", "url_salida", "sessions"]
        ].to_string(index=False)
    )

    print("\nTop complete paths:")
    print(select_top_rows(paths, "sessions", 10)[["flow_path", "sessions"]].to_string(index=False))


def run_workflow(input_path: Path, output_dir: Path, generate_charts: bool) -> dict[str, Iterable[Path]]:
    raw_df = load_data(input_path)
    sessions = build_session_nodes(raw_df)

    source_to_entry = compute_transitions(sessions, "ref_source", "url_entrada")
    entry_to_exit = compute_transitions(sessions, "url_entrada", "url_salida")
    combined_paths = compute_combined_paths(sessions)
    page_metrics = compute_page_metrics(sessions)
    sankey_edges = build_sankey_edges(source_to_entry, entry_to_exit)
    segment_pais = compute_segment_summary(sessions, "pais")
    segment_dispositivo = compute_segment_summary(sessions, "dispositivo")
    segment_explorador = compute_segment_summary(sessions, "explorador")
    segment_transition_pais = compute_segment_transitions(sessions, "pais")
    segment_transition_dispositivo = compute_segment_transitions(sessions, "dispositivo")
    validations = validate_outputs(sessions, source_to_entry, entry_to_exit)

    source_entry_matrix = build_transition_matrix(source_to_entry, "ref_source", "url_entrada")
    entry_exit_matrix = build_transition_matrix(entry_to_exit, "url_entrada", "url_salida")

    artifacts = {
        "cleaned_sessions.csv": sessions,
        "source_to_entry_transitions.csv": source_to_entry,
        "entry_to_exit_transitions.csv": entry_to_exit,
        "top_paths.csv": combined_paths,
        "page_metrics.csv": page_metrics,
        "sankey_edges.csv": sankey_edges,
        "segment_summary_pais.csv": segment_pais,
        "segment_summary_dispositivo.csv": segment_dispositivo,
        "segment_summary_explorador.csv": segment_explorador,
        "segment_transitions_pais.csv": segment_transition_pais,
        "segment_transitions_dispositivo.csv": segment_transition_dispositivo,
        "transition_matrix_source_entry.csv": source_entry_matrix,
        "transition_matrix_entry_exit.csv": entry_exit_matrix,
        "validation_checks.csv": validations,
    }
    exported_files = export_outputs(output_dir, artifacts)

    chart_paths: list[Path] = []
    if generate_charts:
        chart_paths = create_charts(page_metrics, output_dir)

    print_summary(page_metrics, combined_paths, source_to_entry, entry_to_exit)

    return {
        "exports": exported_files,
        "charts": chart_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a session-level navigation flow workflow from 1_Data_Recordings.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where workflow outputs will be exported.",
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Skip PNG chart generation and export only CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_workflow(
        input_path=args.input,
        output_dir=args.output_dir,
        generate_charts=not args.skip_charts,
    )

    print("\nExported files:")
    for path in results["exports"]:
        print(f" - {path}")

    if results["charts"]:
        print("\nGenerated charts:")
        for path in results["charts"]:
            print(f" - {path}")


if __name__ == "__main__":
    main()
