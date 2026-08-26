"""
Part 2, Q1 - What is the linear rate of warming?

Fits a linear trend (degC/decade, with its 95% CI) to the annual-mean
air-temperature-above-sea-ice anomaly (the standard series, per Part 1),
for three start years: 1970, 1980 (primary), and 1990. All three windows
run to the same end year - the last full calendar year in the record.

Monthly values are averaged into annual means before fitting, because
adjacent months are not statistically independent (a warm month tends to
be followed by another warm one). Fitting the raw monthly series would
treat each month as an independent observation and understate the
trend's uncertainty.

Uses Part1/temperature_common.py to read the data, so the raw file is
never edited by hand here either.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PART1_DIR = Path(__file__).resolve().parent.parent / "Part1"
sys.path.insert(0, str(PART1_DIR))
from temperature_common import load_series  # noqa: E402

# Start years to compare. 1980 is the assignment's primary trend period.
START_YEARS = [1970, 1980, 1990]
PRIMARY_START_YEAR = 1980
REQUESTED_END_YEAR = 2025  # brief asks for "1980-2025"; clipped to available data below


def annual_means(df: pd.DataFrame) -> pd.Series:
    """Collapse the monthly anomaly into one mean per full calendar year
    (a year is kept only if all 12 months are present)."""
    by_year = df.groupby(df.date.dt.year)["monthly_anomaly"]
    full_years = by_year.count()[by_year.count() == 12].index
    return by_year.mean().loc[full_years]


def fit_trend(annual: pd.Series, start_year: int, end_year: int) -> dict:
    """OLS fit of annual-mean anomaly vs. year over [start_year, end_year].
    Returns the slope and its 95% CI in degC/decade (using the t-distribution
    with n-2 degrees of freedom, appropriate for a small/finite sample)."""
    window = annual.loc[start_year:end_year]
    years = window.index.to_numpy(dtype=float)
    values = window.to_numpy(dtype=float)

    fit = stats.linregress(years, values)
    n = len(years)
    t_crit = stats.t.ppf(0.975, df=n - 2)

    return {
        "start_year": int(years.min()),
        "end_year": int(years.max()),
        "n_years": n,
        "slope_per_year": fit.slope,
        "intercept": fit.intercept,
        "slope_degC_per_decade": fit.slope * 10,
        "ci95_degC_per_decade": t_crit * fit.stderr * 10,
        "r_squared": fit.rvalue ** 2,
    }


def main():
    air = load_series(PART1_DIR / "airTemps.txt")
    annual = annual_means(air)

    last_full_year = int(annual.index.max())
    end_year = min(REQUESTED_END_YEAR, last_full_year)
    if end_year < REQUESTED_END_YEAR:
        print(
            f"Note: the record's last full calendar year is {last_full_year}; "
            f"using {end_year} as the fit window's end year instead of "
            f"{REQUESTED_END_YEAR}.\n"
        )

    print("Linear trend, annual-mean air-temperature-above-sea-ice anomaly")
    print(f"(all windows end in {end_year}; fitted to annual means, not monthly)\n")
    header = f"{'Start year':>10} | {'n (yrs)':>7} | {'Trend (degC/decade)':>20} | {'95% CI (+/-)':>13} | {'R^2':>6}"
    print(header)
    print("-" * len(header))

    results = {}
    for start in START_YEARS:
        r = fit_trend(annual, start, end_year)
        results[start] = r
        marker = "  <- primary" if start == PRIMARY_START_YEAR else ""
        print(
            f"{start:>10} | {r['n_years']:>7} | {r['slope_degC_per_decade']:>20.3f} | "
            f"±{r['ci95_degC_per_decade']:<12.3f} | {r['r_squared']:>6.3f}{marker}"
        )

    # --- Plot: annual means with the three fitted trend lines overlaid ---
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(
        annual.index, annual.values, "o-",
        color="tab:red", markersize=3, linewidth=0.8, alpha=0.6,
        label="Annual mean anomaly",
    )

    line_style = {1970: "tab:blue", 1980: "black", 1990: "tab:green"}
    for start in START_YEARS:
        r = results[start]
        years_fit = np.array([r["start_year"], r["end_year"]])
        fitted = r["intercept"] + r["slope_per_year"] * years_fit
        ax.plot(
            years_fit, fitted,
            color=line_style[start],
            linewidth=2.4 if start == PRIMARY_START_YEAR else 1.8,
            label=(f"{start}–{end_year} fit: {r['slope_degC_per_decade']:.3f} "
                   f"± {r['ci95_degC_per_decade']:.3f} °C/decade"),
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mean anomaly (°C, relative to 1951–1980)")
    ax.set_title(
        "Linear warming trend by fit window — air temperature above sea ice",
        loc="left", fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)

    fig.text(
        0.01, -0.06,
        f"Figure. Annual-mean global temperature anomaly (red) with ordinary least-squares linear trends fitted\n"
        f"over three start years, all ending {end_year}. Slopes are reported in °C/decade with a 95% confidence\n"
        "interval (t-distribution, n−2 degrees of freedom) from the annual-mean regression, which avoids treating\n"
        "autocorrelated monthly values as independent observations. Source: Berkeley Earth Land/Ocean\n"
        "Temperature Record (Rohde & Hausfather, 2020), https://doi.org/10.5194/essd-12-3469-2020.",
        fontsize=8, ha="left", va="top",
    )

    plt.show()


if __name__ == "__main__":
    main()
