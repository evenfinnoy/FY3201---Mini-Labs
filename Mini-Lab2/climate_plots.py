"""
Plot functions for one place's climate data (see climate_common.py for how
the data is loaded). Each function takes a cleaned DataFrame plus the name
of the value column(s) to use, and returns (fig, ax) in its own figure -
none of them call plt.show(), that's left to the driver script.
"""

import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Day-of-year each month starts on, in a non-leap reference year (the "doy"
# column already folds Dec 31 of leap years onto day 365, so this lines up).
_MONTH_STARTS_DOY = [pd.Timestamp(2001, m, 1).dayofyear for m in range(1, 13)] + [366]


def _shade_months(ax, color="0.90"):
    """Shade Jan, Mar, May, ... with a light gray band and leave Feb, Apr,
    Jun, ... white, so the calendar structure of a day-of-year x-axis reads
    at a glance."""
    for i in range(0, 12, 2):
        ax.axvspan(_MONTH_STARTS_DOY[i], _MONTH_STARTS_DOY[i + 1],
                   color=color, zorder=0, linewidth=0)


def _caption(fig, source_note: str):
    if source_note:
        fig.text(0.01, -0.04, source_note, fontsize=8, ha="left", va="top", wrap=True)


def _reference_period(df, reference_years):
    """Rows within reference_years, or - if the record doesn't reach that
    far back - every year on record. Also returns the years actually used,
    so the title/legend can say what was really plotted."""
    ref = df[df["year"].between(*reference_years)]
    if ref.empty:
        return df, (int(df["year"].min()), int(df["year"].max()))
    return ref, reference_years


def _smooth_doy(series, window=15):
    """Centered rolling mean over a 1..365 day-of-year index, wrapping
    across the year boundary (climate normals have no hard edge on Jan 1).
    Turns a noisy single-day statistic into a smooth climatological curve."""
    full = series.reindex(range(1, 366))
    padded = pd.concat([full.iloc[-window:], full, full.iloc[:window]])
    smoothed = padded.rolling(window, center=True, min_periods=1).mean()
    return smoothed.iloc[window:-window]


def plot_temperature_vs_normal(
    df, place, max_col, min_col,
    reference_years=(1991, 2020), current_year=2025, source_note="",
):
    """Part 1: climate profile of daily max/min temperature - a smoothed
    10th-90th percentile "typical" band plus the mean, over the reference
    period - with the given year's daily max/min overlaid, to answer "is
    this temperature normal for X?". The all-time record extremes are
    shown too, as thin dotted lines, for context on how unusual a spike is."""
    ref, used_years = _reference_period(df, reference_years)
    cur = df[df["year"] == current_year]

    def profile(col):
        g = ref.groupby("doy")[col]
        mean = _smooth_doy(g.mean())
        p10 = _smooth_doy(g.quantile(0.10))
        p90 = _smooth_doy(g.quantile(0.90))
        rec_min = _smooth_doy(g.min(), window=5)
        rec_max = _smooth_doy(g.max(), window=5)
        return mean, p10, p90, rec_min, rec_max

    max_mean, max_p10, max_p90, max_rec_min, max_rec_max = profile(max_col)
    min_mean, min_p10, min_p90, min_rec_min, min_rec_max = profile(min_col)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    _shade_months(ax)

    ax.fill_between(max_p10.index, max_p10, max_p90, color="tab:red", alpha=0.20,
                     label="Max temp: typical range (10th–90th pct.)")
    ax.plot(max_mean.index, max_mean, color="tab:red", alpha=0.8, linewidth=1.2,
            linestyle="--", label="Average daily max")
    ax.plot(max_rec_max.index, max_rec_max, color="tab:red", alpha=0.35, linewidth=0.6,
            linestyle=":", label="Record max")

    ax.fill_between(min_p10.index, min_p10, min_p90, color="tab:blue", alpha=0.20,
                     label="Min temp: typical range (10th–90th pct.)")
    ax.plot(min_mean.index, min_mean, color="tab:blue", alpha=0.8, linewidth=1.2,
            linestyle="--", label="Average daily min")
    ax.plot(min_rec_min.index, min_rec_min, color="tab:blue", alpha=0.35, linewidth=0.6,
            linestyle=":", label="Record min")

    if not cur.empty:
        ax.plot(cur["doy"], cur[max_col], color="tab:red", linewidth=1.4,
                label=f"{current_year} daily max")
        ax.plot(cur["doy"], cur[min_col], color="tab:blue", linewidth=1.4,
                label=f"{current_year} daily min")

    ax.set_xlim(1, 365)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"{place}: {current_year} temperature vs. climate normal ({used_years[0]}–{used_years[1]})",
        loc="left", fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    _caption(fig, source_note)
    return fig, ax


def plot_temperature_daily_spaghetti(
    df, mean_col, place, baseline_years=(1991, 2020), highlight_years=None,
    smooth_window=7, source_note="",
):
    """Part 1 (extra): one line per year of daily mean-temperature anomaly
    against a smoothed day-of-year climatology, coloured by decade and with
    the most recent years picked out - in the style of Carbon Brief's
    "Monthly global temperature" chart, but at daily instead of monthly
    resolution. Each year's daily series is itself lightly smoothed
    (smooth_window days) since raw day-to-day weather noise at a single
    station is far larger than at the monthly/global scale and would
    otherwise swamp the seasonal signal."""
    d = df.dropna(subset=[mean_col])
    ref, used_years = _reference_period(d, baseline_years)
    clim = _smooth_doy(ref.groupby("doy")[mean_col].mean())

    daily = d[["year", "doy", mean_col]].copy()
    daily["anomaly"] = daily[mean_col] - daily["doy"].map(clim)

    years = sorted(daily["year"].unique())
    if highlight_years is None:
        highlight_years = years[-1:]
    highlight_years = [y for y in highlight_years if y in years]

    decades = sorted({(y // 10) * 10 for y in years})
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=decades[0], vmax=decades[-1] + 9)

    def smoothed_year(year):
        # groupby (not set_index) because leap-year day 366 was folded onto
        # day 365 in "doy", so that day can have two rows for one year.
        series = daily.loc[daily["year"] == year].groupby("doy")["anomaly"].mean()
        series = series.reindex(range(1, 366))
        return series.rolling(smooth_window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    _shade_months(ax)

    for year in years:
        if year in highlight_years:
            continue
        ys = smoothed_year(year)
        ax.plot(ys.index, ys.values, color=cmap(norm((year // 10) * 10)),
                alpha=0.5, linewidth=0.7, zorder=2)

    highlight_colors = plt.cm.Reds(np.linspace(0.55, 0.95, len(highlight_years)))
    for color, year in zip(highlight_colors, sorted(highlight_years)):
        ys = smoothed_year(year)
        ax.plot(ys.index, ys.values, color=color, linewidth=2.0, zorder=5)
        valid = ys.dropna()
        if not valid.empty:
            ax.annotate(str(year), xy=(valid.index[-1], valid.iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", color=color, fontsize=9,
                        fontweight="bold", va="center")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(1, 380)
    ax.set_xlabel("Day of year")
    ax.set_ylabel(
        f"Temperature anomaly (°C, {smooth_window}-day smoothed, "
        f"vs. {used_years[0]}–{used_years[1]} normal)"
    )
    ax.set_title(
        f"{place}: daily temperature anomaly, one line per year ({years[0]}–{years[-1]})",
        loc="left", fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, aspect=30, label="Decade")

    _caption(fig, source_note)
    return fig, ax


def plot_precip_monthly_climatology(
    df, precip_col, place, reference_years=None, highlight_years=None, source_note="",
):
    """Part 2: normal monthly precipitation - the gray box is the climate
    envelope for each month (box = 25th-75th percentile/IQR, whiskers =
    10th-90th percentile), with every individual year's total scattered on
    top and coloured by year, so a wetter/drier trend over time is visible
    directly on the climatology (Part 2 also asks whether precipitation has
    increased). Mean +/- std bars hide all of this: precipitation totals
    are right-skewed and can't go below zero, so a handful of exceptionally
    wet years can dominate a month's average without a symmetric error bar
    showing it.
    highlight_years (e.g. [2025, 2026]) are drawn as distinct outlined
    markers on top of the gradient so a specific year can be read off; a
    year's most recent month is checked for completeness (>=90% of its
    days observed) and drawn hollow if it's still in progress."""
    d = df if reference_years is None else _reference_period(df, reference_years)[0]
    grouped = d.groupby(["year", "month"])[precip_col]
    # min_count=1 so a month with zero *observed* days (all NaN) sums to NaN,
    # not a spurious 0 mm - pandas' default sum() of an all-NaN group is 0.
    monthly = pd.DataFrame({"total": grouped.sum(min_count=1), "n_days": grouped.count()}).reset_index()
    monthly = monthly.dropna(subset=["total"])
    data_by_month = [monthly.loc[monthly["month"] == m, "total"].values for m in range(1, 13)]

    highlight_years = [y for y in (highlight_years or []) if y in monthly["year"].unique()]

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.boxplot(data_by_month, tick_labels=MONTH_LABELS, whis=(10, 90), showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="0.8", alpha=0.5),
               medianprops=dict(color="black", linewidth=1.8),
               whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"),
               showmeans=True,
               meanprops=dict(marker="X", markerfacecolor="white", markeredgecolor="black",
                               markersize=7, markeredgewidth=1.3),
               zorder=3)

    background = monthly[~monthly["year"].isin(highlight_years)]
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=monthly["year"].min(), vmax=monthly["year"].max())
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, size=len(background))
    ax.scatter(background["month"] + jitter, background["total"],
               c=background["year"], cmap=cmap, norm=norm,
               alpha=0.6, s=16, linewidths=0, zorder=2)

    marker_styles = ["D", "^", "s", "o"]
    highlight_colors = ["black", "tab:green", "tab:orange", "tab:purple"]
    for color, marker, year in zip(highlight_colors, marker_styles, sorted(highlight_years)):
        sub = monthly[monthly["year"] == year].copy()
        sub["complete"] = sub.apply(
            lambda r: r["n_days"] >= 0.9 * calendar.monthrange(year, int(r["month"]))[1], axis=1)
        full, partial = sub[sub["complete"]], sub[~sub["complete"]]
        ax.scatter(full["month"], full["total"], color=color, marker=marker, s=65,
                   edgecolor="white", linewidths=0.8, zorder=5, label=str(year))
        if not partial.empty:
            ax.scatter(partial["month"], partial["total"], facecolor="none", edgecolor=color,
                       marker=marker, s=65, linewidths=1.6, zorder=5,
                       label=f"{year} (month in progress)")

    ax.set_ylabel("Precipitation (mm)")
    ax.set_title(f"{place}: normal monthly precipitation", loc="left", fontsize=13, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    box_handles = [
        Patch(facecolor="0.8", alpha=0.5, label="25th–75th percentile (IQR)"),
        Line2D([0], [0], color="0.3", linewidth=1.2, label="10th–90th percentile (whiskers)"),
        Line2D([0], [0], color="black", linewidth=1.8, label="Median"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=7, markeredgewidth=1.3, label="Mean"),
    ]
    ax.legend(handles=box_handles + handles, loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.2, axis="y")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, aspect=30, label="Year")

    _caption(fig, source_note + (" × marks the mean (vs. the median line) to show how much "
                                  "the right-skewed wet years pull it above typical."))
    return fig, ax


def plot_precip_annual_trend(df, precip_col, place, source_note=""):
    """Part 2: annual total precipitation over time, with a linear trend,
    to show whether precipitation has increased."""
    counts = df.groupby("year")[precip_col].count()
    complete_years = counts[counts >= 300].index
    annual = df[df["year"].isin(complete_years)].groupby("year")[precip_col].sum()

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.bar(annual.index, annual.values, color="tab:blue", alpha=0.6, label="Annual total")

    slope, intercept = np.polyfit(annual.index, annual.values, 1)
    trend = slope * annual.index + intercept
    ax.plot(annual.index, trend, color="black", linewidth=1.6, linestyle="--",
            label=f"Trend: {slope * 10:+.1f} mm/decade")

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual precipitation (mm)")
    ax.set_title(f"{place}: annual precipitation over time", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2, axis="y")
    _caption(fig, source_note + (" Years with fewer than 300 observed days are excluded."
                                  if source_note else ""))
    return fig, ax


def plot_precip_daily_gamma_fit(df, precip_col, place, wet_threshold=1.0, source_note=""):
    """Part 2 (extra, teamwork task): "explore whether the measurements can
    be fitted to a statistical distribution" - the gamma distribution is
    the standard textbook model for daily rainfall amounts (right-skewed,
    non-negative, no natural upper bound). Fits scipy.stats.gamma (location
    fixed at 0) to daily precipitation on wet days (> wet_threshold mm, to
    exclude trace/measurement noise on nominally dry days) and shows both a
    histogram-vs-PDF overlay and a Q-Q plot, which is the more honest way
    to eyeball goodness-of-fit here: a formal KS-test p-value would be
    invalid since the same data both fits the distribution and tests it."""
    wet = df.loc[df[precip_col] > wet_threshold, precip_col].dropna()
    shape, loc, scale = stats.gamma.fit(wet, floc=0)

    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax_hist.hist(wet, bins=40, density=True, color="tab:blue", alpha=0.55,
                 label=f"Observed wet days (n={len(wet):,})")
    x = np.linspace(0, np.quantile(wet, 0.999), 400)
    ax_hist.plot(x, stats.gamma.pdf(x, shape, loc, scale), color="black", linewidth=2,
                 label=f"Gamma fit: k={shape:.2f}, θ={scale:.2f} mm\n(fitted mean = k·θ = {shape * scale:.1f} mm)")
    ax_hist.set_xlabel(f"Daily precipitation on wet days (> {wet_threshold:g} mm)")
    ax_hist.set_ylabel("Probability density")
    ax_hist.set_title(f"{place}: daily rainfall amount vs. gamma fit", loc="left",
                       fontsize=12, fontweight="bold")
    ax_hist.legend(loc="upper right", fontsize=8, frameon=False)
    ax_hist.grid(alpha=0.2)

    stats.probplot(wet, dist=stats.gamma, sparams=(shape, loc, scale), plot=ax_qq)
    ax_qq.get_lines()[0].set(markerfacecolor="tab:blue", markeredgecolor="none",
                              markersize=4, alpha=0.5)
    ax_qq.get_lines()[1].set(color="black", linewidth=2)
    ax_qq.set_xlabel("Gamma quantile")
    ax_qq.set_ylabel("Observed quantile (mm)")
    ax_qq.set_title("")  # clear scipy's own centered "Probability Plot" title
    ax_qq.set_title("Q–Q plot (points on the line = good fit)", loc="left",
                     fontsize=12, fontweight="bold")
    ax_qq.grid(alpha=0.2)

    _caption(fig, source_note + " Gamma fit via scipy.stats.gamma, location fixed at 0.")
    return fig, (ax_hist, ax_qq)


def plot_precip_monthly_gamma_fit(df, precip_col, place, reference_years=None, source_note=""):
    """Part 2 (extra, teamwork task): fit a gamma distribution to each
    calendar month's year-to-year total precipitation - the same "amount of
    precipitation" already summarised as a box in
    plot_precip_monthly_climatology - so the shape of that spread (and how
    well gamma describes it) can be compared month to month."""
    d = df if reference_years is None else _reference_period(df, reference_years)[0]
    # min_count=1 so a month with zero *observed* days (all NaN) sums to NaN,
    # not a spurious 0 mm - pandas' default sum() of an all-NaN group is 0.
    monthly = d.groupby(["year", "month"])[precip_col].sum(min_count=1).reset_index()

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    for m, ax in zip(range(1, 13), axes.flat):
        vals = monthly.loc[monthly["month"] == m, precip_col].dropna().values
        # A continuous gamma has zero probability of landing on exactly 0,
        # so an entirely dry month (rare, but real - e.g. Apr 1974 here)
        # can't enter its own MLE fit.
        vals = vals[vals > 0]
        shape, loc, scale = stats.gamma.fit(vals, floc=0)

        ax.hist(vals, bins=14, density=True, color="tab:blue", alpha=0.55)
        x = np.linspace(0, vals.max() * 1.05, 200)
        ax.plot(x, stats.gamma.pdf(x, shape, loc, scale), color="black", linewidth=1.8)

        ax.set_title(f"{MONTH_LABELS[m - 1]}  (k={shape:.1f}, θ={scale:.0f})",
                     loc="left", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)

    fig.supxlabel("Monthly total precipitation (mm)", fontsize=10)
    fig.supylabel("Probability density", fontsize=10)
    fig.suptitle(f"{place}: monthly precipitation totals vs. gamma fit, by calendar month",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")

    _caption(fig, source_note + (" Gamma fit via scipy.stats.gamma, location fixed at 0, per month; "
                                  "any entirely dry month (total = 0) is excluded from its fit since a "
                                  "continuous gamma distribution assigns it zero probability."))
    return fig, axes


def plot_pressure_monthly_boxplot(df, pressure_col, place, source_note=""):
    """Part 3a: within-month pressure variation, across all years on
    record. Box = 25th-75th percentile (IQR); whiskers extend to the
    furthest daily value within 1.5x IQR of the box (Tukey's rule); more
    extreme days exist but are hidden as outlier points to keep the chart
    readable. The box/whiskers are built from every daily value so they
    show real day-to-day spread; on top, each year's mean for that month is
    plotted as a dot coloured by year, so a long-term drift (Part 3a also
    asks whether pressure has changed over time) is visible at a glance
    without changing what the box itself represents."""
    d = df.dropna(subset=[pressure_col])
    data_by_month = [d.loc[d["month"] == m, pressure_col].values for m in range(1, 13)]
    monthly_means = d.groupby(["year", "month"])[pressure_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.boxplot(data_by_month, tick_labels=MONTH_LABELS, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="tab:purple", alpha=0.4),
               medianprops=dict(color="black", linewidth=1.8),
               whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"),
               showmeans=True,
               meanprops=dict(marker="X", markerfacecolor="white", markeredgecolor="black",
                               markersize=7, markeredgewidth=1.3),
               zorder=3)

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=monthly_means["year"].min(), vmax=monthly_means["year"].max())
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, size=len(monthly_means))
    ax.scatter(monthly_means["month"] + jitter, monthly_means[pressure_col],
               c=monthly_means["year"], cmap=cmap, norm=norm,
               alpha=0.6, s=14, linewidths=0, zorder=2)

    ax.set_ylabel("Mean sea-level pressure (hPa)")
    ax.set_title(f"{place}: pressure variation within a month", loc="left", fontsize=13, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="tab:purple", alpha=0.4, label="25th–75th percentile (IQR)"),
            Line2D([0], [0], color="0.3", linewidth=1.2, label="Whiskers: within 1.5×IQR"),
            Line2D([0], [0], color="black", linewidth=1.8, label="Median"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="white",
                   markeredgecolor="black", markersize=7, markeredgewidth=1.3, label="Mean"),
        ],
        loc="upper left", fontsize=8, frameon=False,
    )
    ax.grid(alpha=0.2, axis="y")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, aspect=30, label="Year")

    _caption(fig, source_note + " Dots = each year's mean pressure for that month, coloured by year.")
    return fig, ax


def plot_pressure_annual_trend(df, pressure_col, place, source_note=""):
    """Part 3a: whether mean pressure has changed over time."""
    annual = df.groupby("year")[pressure_col].mean().dropna()

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(annual.index, annual.values, marker="o", ms=3, linewidth=1,
            color="tab:purple", label="Annual mean pressure")

    slope, intercept = np.polyfit(annual.index, annual.values, 1)
    trend = slope * annual.index + intercept
    ax.plot(annual.index, trend, color="black", linewidth=1.6, linestyle="--",
            label=f"Trend: {slope * 10:+.2f} hPa/decade")

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean sea-level pressure (hPa)")
    ax.set_title(f"{place}: annual mean pressure over time", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    _caption(fig, source_note)
    return fig, ax


def plot_wind_monthly_climatology(df, wind_col, place, source_note=""):
    """Part 3b: normal monthly wind, to show which months are windier/calmer."""
    d = df.dropna(subset=[wind_col])
    stats = d.groupby("month")[wind_col].agg(mean="mean", std="std").reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.bar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), capsize=3,
           color="tab:green", alpha=0.75, label="Mean daily wind ± 1 std (inter-annual)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title(f"{place}: normal monthly wind", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2, axis="y")
    _caption(fig, source_note)
    return fig, ax
