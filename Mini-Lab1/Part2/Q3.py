"""
Part 2, Q3 - What co-varies with the warming?

Correlates the annual-mean temperature anomaly (air temperature above sea
ice, per Part 1) with atmospheric CO2, CH4, and world population - each
downloaded and cached by external_data.py. The series have different
resolutions and coverage (temperature: monthly since 1850; CO2: monthly
since 1979; CH4: monthly since 1983; population: annual since 1950), so
everything is first reduced to annual means/values and then restricted to
the years common to all four before any correlation is computed.

Reports Pearson r and its p-value for each pair. Which of these
relationships could plausibly be causal, in which direction, through what
physical mechanism, and what confounds them, is for your report - the
numbers here are the evidence, not the argument.
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from external_data import load_ch4_annual, load_co2_annual, load_population_annual
from trend_utils import load_annual_air_temperature

VARIABLES = {
    "CO2": ("CO2 (ppm)", "tab:green"),
    "CH4": ("CH4 (ppb)", "tab:purple"),
    "Population": ("World population (billions)", "tab:orange"),
}


def harmonize(series_dict: dict) -> pd.DataFrame:
    """Inner-join all series on their year index: the common period."""
    return pd.DataFrame(series_dict).dropna()


def main():
    temperature = load_annual_air_temperature()
    co2 = load_co2_annual()
    ch4 = load_ch4_annual()
    population = load_population_annual() / 1e9  # persons -> billions, for readable axes

    harmonized = harmonize({
        "Temperature": temperature,
        "CO2": co2,
        "CH4": ch4,
        "Population": population,
    })
    start_year, end_year = harmonized.index.min(), harmonized.index.max()

    print("Q3: correlation with the annual-mean temperature anomaly")
    print(f"Common period across all four series: {start_year}-{end_year} (n={len(harmonized)} years)\n")
    print(f"{'Variable':<12} | {'Pearson r':>9} | {'p-value':>10}")
    print("-" * 38)

    correlations = {}
    for var in VARIABLES:
        r, p = stats.pearsonr(harmonized["Temperature"], harmonized[var])
        correlations[var] = (r, p)
        print(f"{var:<12} | {r:>9.3f} | {p:>10.2e}")

    # --- Figure: standardized time series overlay + pairwise scatter plots ---
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_ts = fig.add_subplot(gs[0, :])
    z = (harmonized - harmonized.mean()) / harmonized.std()
    ax_ts.plot(z.index, z["Temperature"], color="tab:red", linewidth=2, label="Temperature anomaly", zorder=5)
    for var in VARIABLES:
        label, color = VARIABLES[var]
        ax_ts.plot(z.index, z[var], color=color, linewidth=1.4, alpha=0.85, label=label)
    ax_ts.set_ylabel("Standardized value (z-score)")
    ax_ts.set_xlabel("Year")
    ax_ts.set_title(
        f"Standardized annual series, {start_year}–{end_year} (common period)",
        loc="left", fontsize=12, fontweight="bold",
    )
    ax_ts.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax_ts.grid(alpha=0.2)

    for i, var in enumerate(VARIABLES):
        ax = fig.add_subplot(gs[1, i])
        label, color = VARIABLES[var]
        x = harmonized[var]
        y = harmonized["Temperature"]
        ax.scatter(x, y, color=color, s=14, alpha=0.7)

        slope, intercept, r, p, se = stats.linregress(x, y)
        x_fit = [x.min(), x.max()]
        y_fit = [intercept + slope * xv for xv in x_fit]
        ax.plot(x_fit, y_fit, color="black", linewidth=1.5)

        ax.set_xlabel(label)
        if i == 0:
            ax.set_ylabel("Temperature anomaly (°C)")
        ax.set_title(f"r = {r:.3f}  (p = {p:.1e})", fontsize=10)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Temperature anomaly vs. CO₂, CH₄, and world population", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.01, -0.04,
        f"Figure. Top: annual series over {start_year}–{end_year}, each standardized (zero mean, unit variance) so\n"
        "they can be compared on one axis regardless of units. Bottom: pairwise scatter plots of the same data with\n"
        "an ordinary least-squares fit and Pearson correlation. Sources: Berkeley Earth (temperature), NOAA GML\n"
        "(CO2, CH4), UN World Population Prospects 2024 via Our World in Data (population). All series harmonized\n"
        "to annual means/values over their common period before computing correlations.",
        fontsize=8, ha="left", va="top",
    )

    plt.show()


if __name__ == "__main__":
    main()
