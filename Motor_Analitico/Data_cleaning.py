from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "1_Data_Recordings.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "data_cleaning"

NUMERIC_COLUMNS = [
    "recuento_paginas",
    "clics_sesion",
    "duracion_sesion_segundos",
    "abandono_rapido",
    "clicks_por_pagina",
    "tiempo_por_pagina",
    "interaccion_total",
    "posible_frustracion",
    "standarized_engagement_score",
]

TIME_COLUMNS = [
    "duracion_sesion_segundos",
    "tiempo_por_pagina",
]

COUNT_COLUMNS = [
    "recuento_paginas",
    "clics_sesion",
    "interaccion_total",
]

RATE_COLUMNS = [
    "abandono_rapido",
    "posible_frustracion",
]

DERIVED_RULES = {
    "clicks_por_pagina": ("clics_sesion", "recuento_paginas"),
    "tiempo_por_pagina": ("duracion_sesion_segundos", "recuento_paginas"),
}


def load_data(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def clean_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if {"fecha", "hora"}.issubset(cleaned.columns):
        cleaned["timestamp"] = pd.to_datetime(
            cleaned["fecha"].astype(str).str.strip() + " " + cleaned["hora"].astype(str).str.strip(),
            format="%m/%d/%Y %H:%M",
            errors="coerce",
        )
    return cleaned


def apply_basic_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cleaned = df.copy()
    audit_rows: list[dict[str, object]] = []

    for column in TIME_COLUMNS:
        if column not in cleaned.columns:
            continue

        invalid_mask = cleaned[column] < 0
        if invalid_mask.any():
            audit_rows.append(
                build_audit_row(column, "negative_to_null", invalid_mask, cleaned)
            )
            cleaned.loc[invalid_mask, column] = pd.NA

    for column in COUNT_COLUMNS:
        if column not in cleaned.columns:
            continue

        invalid_mask = cleaned[column] < 0
        if invalid_mask.any():
            audit_rows.append(
                build_audit_row(column, "negative_to_null", invalid_mask, cleaned)
            )
            cleaned.loc[invalid_mask, column] = pd.NA

    for column in RATE_COLUMNS:
        if column not in cleaned.columns:
            continue

        invalid_mask = ~cleaned[column].isin([0, 1]) & cleaned[column].notna()
        if invalid_mask.any():
            audit_rows.append(
                build_audit_row(column, "non_binary_to_null", invalid_mask, cleaned)
            )
            cleaned.loc[invalid_mask, column] = pd.NA

    if {"duracion_sesion_segundos", "tiempo_por_pagina"}.issubset(cleaned.columns):
        invalid_mask = cleaned["tiempo_por_pagina"] > cleaned["duracion_sesion_segundos"]
        if invalid_mask.any():
            audit_rows.append(
                build_audit_row(
                    "tiempo_por_pagina",
                    "greater_than_session_duration_to_null",
                    invalid_mask,
                    cleaned,
                )
            )
            cleaned.loc[invalid_mask, "tiempo_por_pagina"] = pd.NA

    return cleaned, audit_rows


def apply_outlier_caps(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cleaned = df.copy()
    audit_rows: list[dict[str, object]] = []

    for column in TIME_COLUMNS:
        if column not in cleaned.columns:
            continue

        hard_limit = cleaned[column].quantile(0.999)
        hard_limit = min(hard_limit, 86_400)
        invalid_mask = cleaned[column] > hard_limit

        if invalid_mask.any():
            audit_rows.append(
                build_audit_row(
                    column,
                    f"cap_above_{hard_limit:.2f}",
                    invalid_mask,
                    cleaned,
                )
            )
            cleaned.loc[invalid_mask, column] = hard_limit

    if "standarized_engagement_score" in cleaned.columns:
        lower_limit = cleaned["standarized_engagement_score"].quantile(0.001)
        upper_limit = cleaned["standarized_engagement_score"].quantile(0.999)
        low_mask = cleaned["standarized_engagement_score"] < lower_limit
        high_mask = cleaned["standarized_engagement_score"] > upper_limit

        if low_mask.any():
            audit_rows.append(
                build_audit_row(
                    "standarized_engagement_score",
                    f"cap_below_{lower_limit:.4f}",
                    low_mask,
                    cleaned,
                )
            )
            cleaned.loc[low_mask, "standarized_engagement_score"] = lower_limit

        if high_mask.any():
            audit_rows.append(
                build_audit_row(
                    "standarized_engagement_score",
                    f"cap_above_{upper_limit:.4f}",
                    high_mask,
                    cleaned,
                )
            )
            cleaned.loc[high_mask, "standarized_engagement_score"] = upper_limit

    return cleaned, audit_rows


def recompute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for target_column, (numerator, denominator) in DERIVED_RULES.items():
        if {target_column, numerator, denominator}.issubset(cleaned.columns):
            cleaned[target_column] = cleaned[target_column].where(cleaned[target_column].notna())
            valid_denominator = cleaned[denominator].gt(0)
            cleaned.loc[valid_denominator, target_column] = (
                cleaned.loc[valid_denominator, numerator]
                / cleaned.loc[valid_denominator, denominator]
            )
            cleaned.loc[~valid_denominator, target_column] = pd.NA

    return cleaned


def build_audit_row(
    column: str, rule: str, mask: pd.Series, df: pd.DataFrame
) -> dict[str, object]:
    return {
        "column": column,
        "rule": rule,
        "affected_rows": int(mask.sum()),
        "min_original": df.loc[mask, column].min(),
        "max_original": df.loc[mask, column].max(),
    }


def build_quality_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in NUMERIC_COLUMNS:
        if column not in cleaned_df.columns:
            continue

        raw_series = pd.to_numeric(raw_df[column], errors="coerce")
        cleaned_series = pd.to_numeric(cleaned_df[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "raw_nulls": int(raw_series.isna().sum()),
                "cleaned_nulls": int(cleaned_series.isna().sum()),
                "raw_min": raw_series.min(),
                "cleaned_min": cleaned_series.min(),
                "raw_p99": raw_series.quantile(0.99),
                "cleaned_p99": cleaned_series.quantile(0.99),
                "raw_max": raw_series.max(),
                "cleaned_max": cleaned_series.max(),
            }
        )

    return pd.DataFrame(rows)


def export_outputs(
    output_dir: Path,
    cleaned_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_dir / "1_Data_Recordings_clean.csv", index=False)
    audit_df.to_csv(output_dir / "cleaning_audit.csv", index=False)
    summary_df.to_csv(output_dir / "quality_summary.csv", index=False)


def print_summary(audit_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\nData cleaning completed.\n")
    if audit_df.empty:
        print("No invalid rows were detected by the active cleaning rules.")
    else:
        print("Applied cleaning rules:")
        print(audit_df.to_string(index=False))

    print("\nQuality summary:")
    print(summary_df.to_string(index=False))


def run_cleaning(input_path: Path, output_dir: Path) -> None:
    raw_df = load_data(input_path)
    cleaned_df = coerce_numeric_columns(raw_df)
    cleaned_df = clean_timestamps(cleaned_df)
    cleaned_df, basic_audit = apply_basic_rules(cleaned_df)
    cleaned_df = recompute_derived_metrics(cleaned_df)
    cleaned_df, outlier_audit = apply_outlier_caps(cleaned_df)

    audit_df = pd.DataFrame(basic_audit + outlier_audit)
    summary_df = build_quality_summary(raw_df, cleaned_df)
    export_outputs(output_dir, cleaned_df, audit_df, summary_df)
    print_summary(audit_df, summary_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean 1_Data_Recordings.csv before downstream analytics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where cleaned outputs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cleaning(args.input, args.output_dir)


if __name__ == "__main__":
    main()
