"""
Generic climate plotter for Mini-project 2.

Point FOLDER at a place's data folder (e.g. Oslo/, holding the CSVs
exported from seklima.met.no) and run this file. It discovers every CSV in
that folder, works out which physical quantities each one contains by
looking at the Norwegian column headers (not the filename), and draws
whichever of the temperature / precipitation / pressure / wind plots apply.

To plot a different place: change FOLDER below to that place's folder
(e.g. FOLDER = Path(__file__).resolve().parent / "Bergen") - no other code
needs to change, as long as the folder holds the same kind of seklima.met.no
export. Figures are saved as PNGs into <FOLDER>/plots/.

To ignore older data (e.g. only look at the last 25 years): set START_YEAR
below to that first year to include; every plot is then built only from
START_YEAR onward. Leave it as None to use the full record.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import climate_plots as cp
from climate_common import discover_place_files, find_column, load_seklima_csv
from climate_stats import compare_year_to_normal

FOLDER = Path(__file__).resolve().parent / "Oslo"
REFERENCE_YEARS = (1991, 2020)
CURRENT_YEAR = 2025
# Drop everything before this year, e.g. START_YEAR = 2000, so every plot
# (climate normals, box plots, gamma fits, trend lines, ...) is built only
# from more recent data - useful for judging how "newer changes" compare
# without the full historical record diluting them. None keeps all years.
START_YEAR = 1991
SHOW = True
SAVE = True


def main():
    place = FOLDER.name
    files = discover_place_files(FOLDER)
    if not files:
        raise SystemExit(f"No CSV files found in {FOLDER}")

    figures = []
    for path in files:
        df, cols = load_seklima_csv(path)
        if START_YEAR is not None:
            df = df[df["year"] >= START_YEAR].reset_index(drop=True)
        source_note = f"Source: MET Norway (seklima.met.no), {path.name}. CC BY 4.0."

        max_col = find_column(cols, "maksimumstemperatur")
        min_col = find_column(cols, "minimumstemperatur")
        if max_col and min_col:
            fig, _ = cp.plot_temperature_vs_normal(
                df, place, max_col, min_col,
                reference_years=REFERENCE_YEARS, current_year=CURRENT_YEAR,
                source_note=source_note,
            )
            figures.append(("temperature_vs_normal", fig))

        mean_col = find_column(cols, "middeltemperatur")
        if mean_col:
            highlight_years = [CURRENT_YEAR, CURRENT_YEAR - 1, CURRENT_YEAR - 2]
            fig, _ = cp.plot_temperature_daily_spaghetti(
                df, mean_col, place,
                baseline_years=REFERENCE_YEARS, highlight_years=highlight_years,
                source_note=source_note,
            )
            figures.append(("temperature_daily_spaghetti", fig))

        precip_col = find_column(cols, "nedbør")
        if precip_col:
            fig, _ = cp.plot_precip_monthly_climatology(
                df, precip_col, place,
                reference_years=REFERENCE_YEARS,
                highlight_years=[CURRENT_YEAR, CURRENT_YEAR + 1],
                source_note=source_note,
            )
            figures.append(("precip_monthly_climatology", fig))
            fig, _ = cp.plot_precip_annual_trend(df, precip_col, place, source_note=source_note)
            figures.append(("precip_annual_trend", fig))
            fig, _ = cp.plot_precip_cumulative_trend(df, precip_col, place, source_note=source_note)
            figures.append(("precip_cumulative_trend", fig))
            fig, _ = cp.plot_precip_cumulative_spaghetti(
                df, precip_col, place,
                highlight_years=[CURRENT_YEAR, CURRENT_YEAR + 1],
                source_note=source_note,
            )
            figures.append(("precip_cumulative_spaghetti", fig))
            fig, _ = cp.plot_precip_monthly_trend_grid(df, precip_col, place, source_note=source_note)
            figures.append(("precip_monthly_trend_grid", fig))
            fig, _ = cp.plot_precip_daily_gamma_fit(df, precip_col, place, source_note=source_note)
            figures.append(("precip_daily_gamma_fit", fig))
            fig, _ = cp.plot_precip_monthly_gamma_fit(df, precip_col, place, source_note=source_note)
            figures.append(("precip_monthly_gamma_fit", fig))

            compare_year_to_normal(
                df, precip_col, reference_years=REFERENCE_YEARS, test_year=CURRENT_YEAR,
                label=f"{place} daily precipitation", unit=" mm",
            )
            fig, _ = cp.plot_precip_ecdf_2025_vs_normal(
                df, precip_col, place,
                reference_years=REFERENCE_YEARS, test_year=CURRENT_YEAR,
                source_note=source_note,
            )
            figures.append(("precip_ecdf_2025_vs_normal", fig))

        pressure_col = find_column(cols, "lufttrykk")
        if pressure_col:
            fig, _ = cp.plot_pressure_monthly_boxplot(
                df, pressure_col, place,
                reference_years=REFERENCE_YEARS,
                highlight_years=[CURRENT_YEAR, CURRENT_YEAR + 1],
                source_note=source_note,
            )
            figures.append(("pressure_monthly_boxplot", fig))
            fig, _ = cp.plot_pressure_annual_trend(df, pressure_col, place, source_note=source_note)
            figures.append(("pressure_annual_trend", fig))

        wind_col = find_column(cols, "middelvind") or find_column(cols, "vind")
        if wind_col:
            fig, _ = cp.plot_wind_monthly_climatology(df, wind_col, place, source_note=source_note)
            figures.append(("wind_monthly_climatology", fig))

    if not figures:
        raise SystemExit(f"Found {len(files)} CSV(s) in {FOLDER} but none matched a known column type.")

    if SAVE:
        out_dir = FOLDER / "plots"
        out_dir.mkdir(exist_ok=True)
        for name, fig in figures:
            fig.savefig(out_dir / f"{place}_{name}.png", dpi=150)
        print(f"Saved {len(figures)} figure(s) to {out_dir}")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
