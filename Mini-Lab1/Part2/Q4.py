"""
Part 2, Q4 - The chocolate test.

Correlates the annual-mean temperature anomaly with world cocoa bean
production (FAO, via Our World in Data) - a real, publicly downloadable,
steadily-growing quantity with no plausible physical mechanism linking it
to the climate. It is a stand-in for the brief's "world chocolate
consumption": production is what is actually available as a long, clean
annual series, and is a reasonable global proxy for it.

You will very likely find an impressively high correlation. That is the
point: a correlation coefficient alone can never establish causation. What
actually separates this from Q3 is physical reasoning - CO2 and CH4 have a
known radiative-forcing mechanism connecting them to temperature; cocoa
production does not. Both series simply share a common driver (a growing,
industrializing world economy) without either causing the other. Write
that argument up in your report - this script only produces the number
that makes the argument worth making.
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from external_data import load_cocoa_production_annual, print_citations
from trend_utils import load_annual_air_temperature


def main():
    temperature = load_annual_air_temperature()
    cocoa = load_cocoa_production_annual() / 1e6  # tonnes -> million tonnes

    harmonized = pd.DataFrame({"Temperature": temperature, "Cocoa": cocoa}).dropna()
    start_year, end_year = harmonized.index.min(), harmonized.index.max()

    print_citations(["Cocoa"])
    r, p = stats.pearsonr(harmonized["Temperature"], harmonized["Cocoa"])
    print("Q4: the chocolate test")
    print(f"Common period: {start_year}-{end_year} (n={len(harmonized)} years)")
    print(f"Temperature anomaly vs. world cocoa bean production: r = {r:.3f}  (p = {p:.2e})")
    print(
        "\nA high r here does not mean cocoa farming drives climate, or vice versa - see the\n"
        "module docstring / your report for why (no radiative-forcing-style mechanism, and a\n"
        "shared confound: both track a growing, industrializing 20th/21st-century world)."
    )

    # --- Figure: dual time-series (classic "spurious correlation" style) + scatter ---
    fig, (ax_ts, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax_ts2 = ax_ts.twinx()
    line_temp, = ax_ts.plot(
        harmonized.index, harmonized["Temperature"],
        color="tab:red", linewidth=2, label="Temperature anomaly",
    )
    line_cocoa, = ax_ts2.plot(
        harmonized.index, harmonized["Cocoa"],
        color="saddlebrown", linewidth=2, label="World cocoa bean production",
    )
    ax_ts.set_ylabel("Temperature anomaly (°C)", color="tab:red")
    ax_ts2.set_ylabel("World cocoa bean production (million tonnes)", color="saddlebrown")
    ax_ts.tick_params(axis="y", labelcolor="tab:red")
    ax_ts2.tick_params(axis="y", labelcolor="saddlebrown")
    ax_ts.set_xlabel("Year")
    ax_ts.set_title(f"Two rising lines, {start_year}–{end_year}", loc="left", fontsize=11, fontweight="bold")
    ax_ts.grid(alpha=0.2)
    # Two y-axes means two separate legends by default - combine both lines'
    # handles onto one legend instead.
    ax_ts.legend(
        [line_temp, line_cocoa], [line_temp.get_label(), line_cocoa.get_label()],
        loc="upper left", fontsize=8.5, frameon=False,
    )

    ax_scatter.scatter(
        harmonized["Cocoa"], harmonized["Temperature"],
        color="saddlebrown", s=16, alpha=0.7, label="Annual data point",
    )
    slope, intercept, r_fit, p_fit, se = stats.linregress(harmonized["Cocoa"], harmonized["Temperature"])
    x_fit = [harmonized["Cocoa"].min(), harmonized["Cocoa"].max()]
    y_fit = [intercept + slope * xv for xv in x_fit]
    ax_scatter.plot(x_fit, y_fit, color="black", linewidth=1.5, label="OLS fit")
    ax_scatter.set_xlabel("World cocoa bean production (million tonnes)")
    ax_scatter.set_ylabel("Temperature anomaly (°C)")
    ax_scatter.set_title(f"r = {r:.3f}  (p = {p:.1e})", loc="left", fontsize=11, fontweight="bold")
    ax_scatter.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax_scatter.grid(alpha=0.2)

    fig.suptitle("The chocolate test: temperature vs. world cocoa bean production", fontsize=13, fontweight="bold")
    fig.text(
        0.01, -0.06,
        f"Figure. Left: temperature anomaly (red) and world cocoa bean production (brown), {start_year}–{end_year} -\n"
        "two unrelated quantities that both happen to rise steadily. Right: the same data as a scatter plot with an\n"
        "OLS fit; the high r illustrates that correlation alone cannot establish causation. Sources: Berkeley Earth\n"
        "(temperature); FAO cocoa bean production via Our World in Data (production used as a global proxy for\n"
        "consumption, since a long clean consumption series is not readily available) - exact URL and download\n"
        "date printed above and recorded in Part2/data_cache/*.source.txt.",
        fontsize=8, ha="left", va="top",
    )

    plt.show()


if __name__ == "__main__":
    main()
