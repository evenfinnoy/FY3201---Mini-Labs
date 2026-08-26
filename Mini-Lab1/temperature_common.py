"""
Shared helpers for reading and plotting the Berkeley Earth Land/Ocean
global temperature anomaly record.

Data source (record it for the report):
  Rohde, R. A. & Hausfather, Z. (2020), Earth Syst. Sci. Data, 12, 3469-3479,
  https://doi.org/10.5194/essd-12-3469-2020
  Raw file: https://berkeley-earth-temperature.s3.us-west1.amazonaws.com/Global/Land_and_Ocean_complete.txt
  Downloaded: 2026-08-26

The raw file is never edited by hand. It contains two whitespace-delimited
data blocks, one after the other:
  1) "... Sea Ice Temperature Inferred from Air Temperatures"   -> airTemps.txt
  2) "... Sea Ice Temperature Inferred from Water Temperatures" -> waterTemps.txt
Both blocks share the same 12 columns and use the string "NaN" for missing
values. Temperatures are anomalies in degC relative to the Jan 1951-Dec 1980
mean, and the file states the uncertainty columns are 95% confidence
intervals (not 1-sigma).

Columns (0-indexed):
  0 year
  1 month
  2 monthly anomaly (degC)
  3 monthly anomaly uncertainty, 95% CI (degC)
  4-11 annual / 5-yr / 10-yr / 20-yr moving averages + their 95% CI
       (as already computed by Berkeley Earth; not used here since we
       compute our own 12-month moving average below)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

COLUMNS = [
    "year", "month",
    "monthly_anomaly", "monthly_unc_95",
    "annual_anomaly", "annual_unc_95",
    "five_yr_anomaly", "five_yr_unc_95",
    "ten_yr_anomaly", "ten_yr_unc_95",
    "twenty_yr_anomaly", "twenty_yr_unc_95",
]


def load_series(path: Path) -> pd.DataFrame:
    """Read one Berkeley Earth-style anomaly block: whitespace-delimited,
    no header row, missing values written as the literal string 'NaN'."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        na_values="NaN",
    )
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df = df.sort_values("date").reset_index(drop=True)

    # 12-month centered moving average of the monthly anomaly, computed
    # ourselves (rather than relying on Berkeley Earth's own "annual" column)
    df["rolling_12mo"] = (
        df["monthly_anomaly"].rolling(window=12, center=True, min_periods=12).mean()
    )
    return df


def plot_anomaly(df: pd.DataFrame, title: str, caption: str, color: str):
    """Plot the monthly anomaly as a line with its 95% CI as a shaded band
    around it, plus a 12-month moving average, in its own figure/window."""
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    lower = df.monthly_anomaly - df.monthly_unc_95
    upper = df.monthly_anomaly + df.monthly_unc_95

    # Shaded uncertainty band, with a faint outline on its edges so the
    # band reads clearly as a band (not just noise around the line).
    ax.fill_between(
        df.date, lower, upper,
        color=color, alpha=0.30, linewidth=0,
        label="95% confidence interval",
        zorder=1,
    )
    ax.plot(df.date, lower, color=color, alpha=0.35, linewidth=0.4, zorder=2)
    ax.plot(df.date, upper, color=color, alpha=0.35, linewidth=0.4, zorder=2)

    ax.plot(
        df.date, df.monthly_anomaly,
        color=color, alpha=0.9, linewidth=0.7,
        label="Monthly anomaly",
        zorder=3,
    )
    ax.plot(
        df.date, df.rolling_12mo,
        color="black", linewidth=1.6,
        label="12-month moving average",
        zorder=4,
    )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly (°C, relative to 1951–1980)")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)

    fig.text(0.01, -0.05, caption, fontsize=8, ha="left", va="top", wrap=True)

    return fig, ax
