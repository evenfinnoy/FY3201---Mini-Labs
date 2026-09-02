"""
Shared trend-fitting helpers for Part 2 (Q1: linear rate of warming,
Q2: is the warming accelerating).

Both questions start from the same annual-mean air-temperature-above-sea-ice
anomaly (the standard series, per Part 1). See Part1/temperature_common.py
for how the raw file is read - it is never edited by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PART1_DIR = Path(__file__).resolve().parent.parent / "Part1"
sys.path.insert(0, str(PART1_DIR))
from temperature_common import load_series  # noqa: E402


def annual_means(df: pd.DataFrame) -> pd.Series:
    """Collapse the monthly anomaly into one mean per full calendar year
    (a year is kept only if all 12 months are present). Averaging first
    matters here: adjacent months are not statistically independent (a warm
    month tends to be followed by another warm one), so fitting the raw
    monthly series would treat autocorrelated values as independent
    observations and understate the trend's uncertainty."""
    by_year = df.groupby(df.date.dt.year)["monthly_anomaly"]
    full_years = by_year.count()[by_year.count() == 12].index
    return by_year.mean().loc[full_years]


def load_annual_air_temperature() -> pd.Series:
    """Convenience: read airTemps.txt and return its annual-mean series."""
    air = load_series(PART1_DIR / "airTemps.txt")
    return annual_means(air)


def clip_end_year(annual: pd.Series, requested_end_year: int) -> int:
    """The assignment's fit windows run to 2025, but the record's newest
    full calendar year may be earlier. Clip to what's actually available
    and print a note when that happens, rather than hardcoding a year."""
    last_full_year = int(annual.index.max())
    end_year = min(requested_end_year, last_full_year)
    if end_year < requested_end_year:
        print(
            f"Note: the record's last full calendar year is {last_full_year}; "
            f"using {end_year} as the fit window's end year instead of "
            f"{requested_end_year}.\n"
        )
    return end_year


def fit_linear_trend(annual: pd.Series, start_year: int, end_year: int) -> dict:
    """OLS fit of annual-mean anomaly vs. year over [start_year, end_year].
    Returns the slope and its 95% CI in degC/decade (t-distribution,
    n-2 degrees of freedom - appropriate for a small/finite sample)."""
    window = annual.loc[start_year:end_year]
    years = window.index.to_numpy(dtype=float)
    values = window.to_numpy(dtype=float)
    n = len(years)

    fit = stats.linregress(years, values)
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


def fit_quadratic_trend(annual: pd.Series, start_year: int, end_year: int) -> dict:
    """Fit anomaly(t) = a*t^2 + b*t + c by OLS, where t = decades since
    start_year (just to keep the coefficients numerically well-scaled - a
    raw calendar year like 1970, squared, gives an ill-conditioned fit).

    Returns the three coefficients with their standard errors and two-sided
    p-values (H0: coefficient = 0, Student's t, n-3 degrees of freedom),
    plus the acceleration d^2(anomaly)/dt^2 = 2a with its 95% CI, in
    degC/decade^2 (its p-value is identical to a's, since acceleration is
    just 2a - doubling a coefficient doesn't change its significance).

    This is a purely empirical curvature test, per the assignment brief -
    it does not assume temperature is physically expected to be quadratic.
    """
    window = annual.loc[start_year:end_year]
    t = (window.index.to_numpy(dtype=float) - start_year) / 10.0
    y = window.to_numpy(dtype=float)
    n = len(t)
    dof = n - 3

    # np.polyfit(..., cov=True) gives both the coefficients (highest power
    # first: a, b, c for a*t^2 + b*t + c) and their covariance matrix from
    # the same OLS fit - the coefficient variances are its diagonal.
    (a, b, c), cov = np.polyfit(t, y, deg=2, cov=True)
    a_stderr, b_stderr, c_stderr = np.sqrt(np.diag(cov))

    def p_value(coef: float, stderr: float) -> float:
        return 2 * stats.t.sf(np.abs(coef / stderr), df=dof)

    t_crit = stats.t.ppf(0.975, df=dof)

    fitted = a * t ** 2 + b * t + c
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    def predict(years):
        tt = (np.asarray(years, dtype=float) - start_year) / 10.0
        return a * tt ** 2 + b * tt + c

    return {
        "start_year": int(window.index.min()),
        "end_year": int(window.index.max()),
        "n_years": n,
        "dof": dof,
        # anomaly(t) = a*t^2 + b*t + c, t in decades since start_year
        "a": a, "a_stderr": a_stderr, "a_pvalue": p_value(a, a_stderr),
        "b": b, "b_stderr": b_stderr, "b_pvalue": p_value(b, b_stderr),
        "c": c, "c_stderr": c_stderr, "c_pvalue": p_value(c, c_stderr),
        "acceleration_degC_per_decade2": 2 * a,
        "acceleration_ci95_degC_per_decade2": 2 * t_crit * a_stderr,
        "acceleration_pvalue": p_value(a, a_stderr),
        "r_squared": 1 - ss_res / ss_tot,
        "predict": predict,
    }
