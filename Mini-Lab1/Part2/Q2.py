"""
Part 2, Q2 - Is the warming linear, or accelerating?

Two complementary tests, both on the annual-mean air-temperature-above-sea-ice
anomaly (see trend_utils.py, shared with Q1):

1. Split-window comparison: fit separate linear trends to an early window
   (1970-1997) and a late window (1998-2025, clipped to the data), and
   compare the two slopes with their combined uncertainty.

2. Quadratic fit: fit anomaly = a + b*t + c*t^2 as a simple empirical test
   for curvature over the same combined span (1970-2025, clipped), and
   report the acceleration (=2c) with its 95% CI. This does NOT assume
   global temperature is physically expected to follow a quadratic function
   - it is just a curvature test, per the assignment brief.

Whether either result implies a genuine change in the underlying forced
warming rate (vs. internal variability / the chosen window edges) is a
judgment call for the report, not something this script decides.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from trend_utils import clip_end_year, fit_linear_trend, fit_quadratic_trend, load_annual_air_temperature

# Early/late split. The brief suggests 1970-1997 vs 1998-2025; edit these if
# your team justifies a different split instead.
EARLY_START, EARLY_END = 1970, 1997
LATE_START, REQUESTED_LATE_END = 1998, 2025


def compare_windows(early: dict, late: dict) -> dict:
    """Difference in slope (late - early), with an approximate 95% CI found
    by combining the two (independent, non-overlapping) windows' standard
    errors in quadrature. Uses each fit's own t-critical value averaged by
    degrees of freedom as a simple, slightly conservative combination."""
    stderr_early = early["ci95_degC_per_decade"] / stats.t.ppf(0.975, df=early["n_years"] - 2)
    stderr_late = late["ci95_degC_per_decade"] / stats.t.ppf(0.975, df=late["n_years"] - 2)
    stderr_diff = np.sqrt(stderr_early ** 2 + stderr_late ** 2)

    dof_diff = early["n_years"] + late["n_years"] - 4
    t_crit = stats.t.ppf(0.975, df=dof_diff)

    return {
        "diff_degC_per_decade": late["slope_degC_per_decade"] - early["slope_degC_per_decade"],
        "diff_ci95_degC_per_decade": t_crit * stderr_diff,
    }


def main():
    annual = load_annual_air_temperature()
    late_end = clip_end_year(annual, REQUESTED_LATE_END)

    early = fit_linear_trend(annual, EARLY_START, EARLY_END)
    late = fit_linear_trend(annual, LATE_START, late_end)
    diff = compare_windows(early, late)

    print("Part 1: split-window linear trend comparison")
    print(f"  Early  {early['start_year']}-{early['end_year']} (n={early['n_years']:>2}): "
          f"{early['slope_degC_per_decade']:.3f} ± {early['ci95_degC_per_decade']:.3f} °C/decade "
          f"(R²={early['r_squared']:.3f})")
    print(f"  Late   {late['start_year']}-{late['end_year']} (n={late['n_years']:>2}): "
          f"{late['slope_degC_per_decade']:.3f} ± {late['ci95_degC_per_decade']:.3f} °C/decade "
          f"(R²={late['r_squared']:.3f})")
    print(f"  Late minus early: {diff['diff_degC_per_decade']:+.3f} ± {diff['diff_ci95_degC_per_decade']:.3f} °C/decade "
          "(95% CI, quadrature-combined)\n")

    # --- Quadratic fit over the same combined span ---
    quad_start, quad_end = EARLY_START, late_end
    quad = fit_quadratic_trend(annual, quad_start, quad_end)
    # For comparison: how much does allowing curvature actually improve on
    # a single straight line over the same span?
    linear_full = fit_linear_trend(annual, quad_start, quad_end)

    print(f"Part 2: quadratic fit, {quad['start_year']}-{quad['end_year']} (n={quad['n_years']})")
    print(f"  Acceleration: {quad['acceleration_degC_per_decade2']:+.4f} "
          f"± {quad['acceleration_ci95_degC_per_decade2']:.4f} °C/decade² (95% CI)")
    print(f"  R² (quadratic fit):        {quad['r_squared']:.3f}")
    print(f"  R² (single linear fit):    {linear_full['r_squared']:.3f}  "
          f"({linear_full['slope_degC_per_decade']:.3f} °C/decade over the same span)")

    # --- Plot: annual means, the two linear fits, and the quadratic fit ---
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(
        annual.index, annual.values, "o-",
        color="lightcoral", markersize=3, linewidth=0.8, alpha=0.6,
        label="Annual mean anomaly",
    )

    for r, color, name in [(early, "tab:blue", "Early"), (late, "tab:orange", "Late")]:
        years_fit = np.array([r["start_year"], r["end_year"]])
        fitted = r["intercept"] + r["slope_per_year"] * years_fit
        ax.plot(
            years_fit, fitted, color=color, linewidth=2.4,
            label=(f"{name} {r['start_year']}–{r['end_year']}: "
                   f"{r['slope_degC_per_decade']:.3f} ± {r['ci95_degC_per_decade']:.3f} °C/decade"),
        )

    years_smooth = np.linspace(quad_start, quad_end, 200)
    ax.plot(
        years_smooth, quad["predict"](years_smooth),
        color="black", linewidth=2.0, linestyle="--",
        label=(f"Quadratic fit {quad_start}–{quad_end}: acceleration "
               f"{quad['acceleration_degC_per_decade2']:+.4f} ± "
               f"{quad['acceleration_ci95_degC_per_decade2']:.4f} °C/decade²"),
    )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mean anomaly (°C, relative to 1951–1980)")
    ax.set_title(
        "Linear vs. accelerating warming — air temperature above sea ice",
        loc="left", fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.2)
    ax.set_xlim(quad_start - 3, quad_end + 3)

    fig.text(
        0.01, -0.07,
        f"Figure. Annual-mean global temperature anomaly, {quad_start}–{quad_end} (red), with independent linear\n"
        "trends fitted to an early and a late window (blue, orange) and a single quadratic curve fitted over the\n"
        "whole span (black dashed). The quadratic's curvature term gives an empirical acceleration estimate in\n"
        "°C/decade²; it is a curvature test, not a claim that warming is physically expected to be quadratic.\n"
        "Source: Berkeley Earth Land/Ocean Temperature Record (Rohde & Hausfather, 2020),\n"
        "https://doi.org/10.5194/essd-12-3469-2020.",
        fontsize=8, ha="left", va="top",
    )

    plt.show()


if __name__ == "__main__":
    main()
