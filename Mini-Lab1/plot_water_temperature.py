"""
Part 1 - Data acquisition and first plot: WATER temperature below sea ice.

This is Berkeley Earth's alternative sea-ice treatment (uses sea-surface
water temperature instead of air temperature above the ice). See
temperature_common.py for how the file is read.
"""

import matplotlib.pyplot as plt

from temperature_common import DATA_DIR, load_series, plot_anomaly

# Which column to plot: "monthly", "annual", "5yr", "10yr", or "20yr".
RESOLUTION = "monthly"

SOURCE_NOTE = (
    "Source: Berkeley Earth Land/Ocean Temperature Record (Rohde & Hausfather, 2020),\n"
    "https://doi.org/10.5194/essd-12-3469-2020, downloaded 2026-08-26."
)


def main():
    water = load_series(DATA_DIR / "waterTemps.txt")
    plot_anomaly(
        water,
        series_label="water temperature below sea ice",
        source_note=SOURCE_NOTE,
        color="tab:blue",
        resolution=RESOLUTION,
    )
    plt.show()


if __name__ == "__main__":
    main()
