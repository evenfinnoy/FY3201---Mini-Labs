"""
Generic reader/discovery for MET Norway (seklima.met.no) "Frie meteorologiske
data" CSV exports, of the kind saved into Mini-Lab2/<Place>/.

Export format (observed across temp_*.csv, presip_*.csv, wind_pressure_*.csv):
  - ';'-delimited, UTF-8 with a BOM
  - first column "Navn" (station name), always the same station in a file
  - a "Tid(norsk normaltid)" column with dates as DD.MM.YYYY
  - decimals use a comma ("4,9"), missing values are written as "-"
  - a trailing source-note row: "Data er gyldig per ..., Meteorologisk
    institutt (MET)" (no valid date, so it's dropped rather than parsed)

Files are matched to a physical quantity (temperature / precipitation /
pressure / wind) by their Norwegian *column* headers, not by filename, so a
folder for another place works unchanged even if its files are split or
named differently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TIME_COLUMN_PREFIX = "Tid"
NON_VALUE_COLUMNS = {"Navn", "Stasjon"}
DATE_RE = r"^\d{2}\.\d{2}\.\d{4}$"


def discover_place_files(folder: Path) -> list[Path]:
    """All CSVs directly inside a place's data folder, e.g. Mini-Lab2/Oslo/."""
    folder = Path(folder)
    return sorted(folder.glob("*.csv"))


def load_seklima_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read one seklima.met.no export into a clean DataFrame.

    Returns (df, value_columns) where df has a "date"/"year"/"month" column
    plus one numeric column per original measurement column (missing values
    as NaN), and value_columns lists those original measurement column
    names so callers can find the one(s) they need by keyword.
    """
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)

    time_col = next(c for c in raw.columns if c.startswith(TIME_COLUMN_PREFIX))
    raw = raw[raw[time_col].str.match(DATE_RE, na=False)].copy()

    df = pd.DataFrame({"date": pd.to_datetime(raw[time_col], format="%d.%m.%Y")})

    value_columns = [c for c in raw.columns if c not in NON_VALUE_COLUMNS | {time_col}]
    for col in value_columns:
        cleaned = raw[col].str.replace(",", ".", regex=False).replace("-", pd.NA)
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    # Fold Dec 31 of leap years (day 366) onto day 365 so every year lines
    # up on a fixed 1-365 axis for day-of-year climatology plots.
    df["doy"] = doy.where(doy <= 365, 365)

    return df, value_columns


def find_column(columns: list[str], keyword: str) -> str | None:
    """First column whose header contains keyword (case-insensitive)."""
    keyword = keyword.lower()
    for col in columns:
        if keyword in col.lower():
            return col
    return None
