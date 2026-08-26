"""
Part 1 - Data acquisition and first plot: AIR temperature above sea ice.

This is the Berkeley Earth "standard" series (the version normally quoted
for global warming). See temperature_common.py for how the file is read.
"""

import matplotlib.pyplot as plt

from temperature_common import DATA_DIR, load_series, plot_anomaly

# Which column to plot: "monthly", "annual", "5yr", "10yr", or "20yr".
RESOLUTION = "annual"

SOURCE_NOTE = (
    "Source: Berkeley Earth Land/Ocean Temperature Record (Rohde & Hausfather, 2020),\n"
    "https://doi.org/10.5194/essd-12-3469-2020, downloaded 2026-08-26."
)


def main():
    air = load_series(DATA_DIR / "airTemps.txt")
    plot_anomaly(
        air,
        series_label="air temperature above sea ice",
        source_note=SOURCE_NOTE,
        color="tab:red",
        resolution=RESOLUTION,
    )
    plt.show()


if __name__ == "__main__":
    main()
