"""
Part 1 - Data acquisition and first plot: WATER temperature below sea ice.

This is Berkeley Earth's alternative sea-ice treatment (uses sea-surface
water temperature instead of air temperature above the ice). See
temperature_common.py for how the file is read.
"""

import matplotlib.pyplot as plt

from temperature_common import DATA_DIR, load_series, plot_anomaly

CAPTION = (
    "Figure. Monthly global mean surface temperature anomaly (thin line) relative to the Jan 1951–Dec 1980\n"
    "baseline, with its quoted measurement uncertainty shown as a shaded 95% confidence interval band.\n"
    "The black line is a 12-month centered moving average, computed here from the monthly series. This is\n"
    "the alternative “water temperature below sea ice” version (sea-surface temperature is used instead of\n"
    "air temperature over sea ice). Source: Berkeley Earth Land/Ocean Temperature Record (Rohde & Hausfather,\n"
    "2020), https://doi.org/10.5194/essd-12-3469-2020, downloaded 2026-08-26."
)


def main():
    water = load_series(DATA_DIR / "waterTemps.txt")
    plot_anomaly(
        water,
        title="Berkeley Earth global temperature anomaly — water temperature below sea ice",
        caption=CAPTION,
        color="tab:blue",
    )
    plt.show()


if __name__ == "__main__":
    main()
