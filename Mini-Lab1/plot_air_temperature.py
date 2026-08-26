"""
Part 1 - Data acquisition and first plot: AIR temperature above sea ice.

This is the Berkeley Earth "standard" series (the version normally quoted
for global warming). See temperature_common.py for how the file is read.
"""

import matplotlib.pyplot as plt

from temperature_common import DATA_DIR, load_series, plot_anomaly

CAPTION = (
    "Figure. Monthly global mean surface temperature anomaly (thin line) relative to the Jan 1951–Dec 1980\n"
    "baseline, with its quoted measurement uncertainty shown as a shaded 95% confidence interval band.\n"
    "The black line is a 12-month centered moving average, computed here from the monthly series. This is\n"
    "the “air temperature above sea ice” version, the standard series normally quoted for global warming.\n"
    "Source: Berkeley Earth Land/Ocean Temperature Record (Rohde & Hausfather, 2020),\n"
    "https://doi.org/10.5194/essd-12-3469-2020, downloaded 2026-08-26."
)


def main():
    air = load_series(DATA_DIR / "airTemps.txt")
    plot_anomaly(
        air,
        title="Berkeley Earth global temperature anomaly — air temperature above sea ice",
        caption=CAPTION,
        color="tab:red",
    )
    plt.show()


if __name__ == "__main__":
    main()
