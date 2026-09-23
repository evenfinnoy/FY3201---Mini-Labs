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


def _add_top_margin(ax, frac=0.12):
    """Push the y-axis top up by frac of the current data range, so an
    "upper left"/"upper right" legend has clear air above the plotted
    lines/points instead of sitting on top of them."""
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + frac * (top - bottom))


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
    _add_top_margin(ax, frac=0.06)
    _caption(fig, source_note)
    return fig, ax


def plot_precip_monthly_climatology(
    df, precip_col, place, reference_years=(1991, 2020), highlight_years=None, source_note="",
):
    """Part 2: normal monthly precipitation - the gray box is the climate
    envelope for each month, built only from the reference period (box =
    25th-75th percentile/IQR, whiskers = 10th-90th percentile of those
    reference years), with every year's total scattered on top (not just
    the reference period) and coloured by year, so a wetter/drier trend
    over time is visible directly on the climatology (Part 2 also asks
    whether precipitation has increased). Mean +/- std bars hide all of
    this: precipitation totals are right-skewed and can't go below zero,
    so a handful of exceptionally wet years can dominate a month's average
    without a symmetric error bar showing it.
    highlight_years (e.g. [2025, 2026]) are drawn as distinct outlined
    markers on top of the gradient so a specific year can be read off,
    including years outside the reference period itself; a year's most
    recent month is checked for completeness (>=90% of its days observed)
    and drawn hollow if it's still in progress."""
    def monthly_totals(d):
        grouped = d.groupby(["year", "month"])[precip_col]
        # min_count=1 so a month with zero *observed* days (all NaN) sums to
        # NaN, not a spurious 0 mm - pandas' default sum() of an all-NaN
        # group is 0.
        m = pd.DataFrame({"total": grouped.sum(min_count=1), "n_days": grouped.count()}).reset_index()
        return m.dropna(subset=["total"])

    ref, used_years = _reference_period(df, reference_years)
    box_monthly = monthly_totals(ref)
    data_by_month = [box_monthly.loc[box_monthly["month"] == m, "total"].values for m in range(1, 13)]

    monthly = monthly_totals(df)
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

    ref_label = f"{used_years[0]}–{used_years[1]}"
    handles, labels = ax.get_legend_handles_labels()
    box_handles = [
        Patch(facecolor="0.8", alpha=0.5, label=f"25th–75th percentile ({ref_label})"),
        Line2D([0], [0], color="0.3", linewidth=1.2, label=f"10th–90th percentile ({ref_label})"),
        Line2D([0], [0], color="black", linewidth=1.8, label=f"Median ({ref_label})"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=7, markeredgewidth=1.3, label=f"Mean ({ref_label})"),
    ]
    ax.legend(handles=box_handles + handles, loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.2, axis="y")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, aspect=30, label="Year")

    _caption(fig, source_note + (
        f" Box/whiskers built from the {ref_label} reference period only; dots = every year's "
        "total, coloured by year. × marks the mean (vs. the median line) to show how much the "
        "right-skewed wet years pull it above typical."
    ))
    return fig, ax


def plot_precip_cumulative_trend(df, precip_col, place, smooth_years=10, poly_degree=2, source_note=""):
    """Part 2: has precipitation increased over time? A single straight
    trend line through noisy annual bars is easy to eyeball as "barely
    there" even when a real multi-year trend exists, because year-to-year
    variability is so much larger than the trend itself. Two more robust,
    standard views overlaid on the annual totals instead: a
    smooth_years-year centered moving average (a data-driven local
    smoother) and a degree-poly_degree polynomial fit via
    np.polyfit/np.polyval (a single smooth curve over the whole period, so
    it doesn't depend on choosing a window length) - both make a real
    multi-year rise or fall easier to see than the bars alone."""
    counts = df.groupby("year")[precip_col].count()
    complete_years = counts[counts >= 300].index
    annual = df[df["year"].isin(complete_years)].groupby("year")[precip_col].sum(min_count=1).dropna()

    mean_annual = annual.mean()
    rolling = annual.rolling(smooth_years, center=True, min_periods=max(3, smooth_years // 2)).mean()

    slope, _ = np.polyfit(annual.index, annual.values, 1)
    poly_coeffs = np.polyfit(annual.index, annual.values, poly_degree)
    x_smooth = np.linspace(annual.index.min(), annual.index.max(), 200)
    poly_curve = np.polyval(poly_coeffs, x_smooth)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)

    ax.bar(annual.index, annual.values, color="tab:blue", alpha=0.45, label="Annual total")
    ax.plot(rolling.index, rolling.values, color="black", linewidth=2.2,
            label=f"{smooth_years}-year moving average")
    ax.plot(x_smooth, poly_curve, color="tab:orange", linewidth=2.4, linestyle="-.",
            label=f"Degree-{poly_degree} polynomial fit")
    ax.axhline(mean_annual, color="0.4", linewidth=1, linestyle=":",
               label=f"Long-term mean ({mean_annual:.0f} mm)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual precipitation (mm)")
    ax.set_title(f"{place}: has annual precipitation increased over time?", loc="left",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2, axis="y")

    _caption(fig, source_note + (
        f" Only years with ≥300 observed days are included (n={len(annual)}). Overall linear "
        f"trend across the full period: {slope * 10:+.1f} mm/decade (the polynomial fit is for "
        "shape, not a trend rate - its curvature isn't a constant mm/decade)."
    ))
    return fig, ax


def plot_precip_cumulative_spaghetti(df, precip_col, place, highlight_years=None, source_note=""):
    """Part 2 (extra): one line per year of cumulative precipitation
    through the year (day-of-year on the x-axis), coloured by decade, in
    the style of Carbon Brief's "Monthly global temperature" chart. Raw
    daily precipitation is too spiky (individual rain events, mostly zero)
    to compare year to year as a spaghetti plot directly; its running
    cumulative total is instead a smooth, monotonically increasing curve,
    so "is this year running wetter or drier than usual, and by how much"
    is readable at a glance at any point in the year, and each
    line's endpoint is that year's annual total."""
    d = df.dropna(subset=[precip_col])
    years = sorted(d["year"].unique())
    if highlight_years is None:
        highlight_years = years[-1:]
    highlight_years = [y for y in highlight_years if y in years]

    decades = sorted({(y // 10) * 10 for y in years})
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=decades[0], vmax=decades[-1] + 9)

    def cumulative_year(year):
        # groupby (not set_index) because leap-year day 366 was folded onto
        # day 365 in "doy", so that day can have two rows for one year;
        # cumsum's default skipna=True leaves any day beyond the last one
        # actually on record as NaN, so an in-progress year's line simply
        # stops there rather than continuing flat.
        series = d.loc[d["year"] == year].groupby("doy")[precip_col].sum(min_count=1)
        return series.reindex(range(1, 366)).cumsum()

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    _shade_months(ax)

    for year in years:
        if year in highlight_years:
            continue
        ys = cumulative_year(year)
        ax.plot(ys.index, ys.values, color=cmap(norm((year // 10) * 10)),
                alpha=0.5, linewidth=0.8, zorder=2)

    highlight_colors = plt.cm.Reds(np.linspace(0.55, 0.95, len(highlight_years)))
    for color, year in zip(highlight_colors, sorted(highlight_years)):
        ys = cumulative_year(year)
        ax.plot(ys.index, ys.values, color=color, linewidth=2.2, zorder=5)
        valid = ys.dropna()
        if not valid.empty:
            ax.annotate(str(year), xy=(valid.index[-1], valid.iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", color=color, fontsize=9,
                        fontweight="bold", va="center")

    ax.set_xlim(1, 380)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Cumulative precipitation (mm)")
    ax.set_title(
        f"{place}: cumulative precipitation through the year, one line per year ({years[0]}–{years[-1]})",
        loc="left", fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, aspect=30, label="Decade")

    _caption(fig, source_note)
    return fig, ax


def plot_precip_monthly_trend_grid(
    df, precip_col, place, min_complete_frac=0.9, poly_degree=2, source_note="",
):
    """Part 2: has precipitation increased over time, broken down by
    calendar month? Summing all twelve months into one annual total
    (plot_precip_cumulative_trend) can hide a change concentrated in one
    season - e.g. wetter autumns masked by flat summers. One panel per
    calendar month, each showing that month's total across every year on
    record with its own linear trend line (np.polyfit degree 1) and a
    degree-poly_degree polynomial fit (np.polyfit/np.polyval) for any
    curvature the straight line misses, so a monthly increase or decrease
    is visible even when the annual total looks flat.
    A year's month is excluded from its panel if fewer than
    min_complete_frac of that month's calendar days were actually
    observed (mainly matters for the current, still-in-progress month)."""
    grouped = df.groupby(["year", "month"])[precip_col]
    monthly = pd.DataFrame({"total": grouped.sum(min_count=1), "n_days": grouped.count()}).reset_index()
    monthly = monthly.dropna(subset=["total"])

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True, sharex=True)
    for m, ax in zip(range(1, 13), axes.flat):
        d = monthly[monthly["month"] == m]
        expected_days = d.apply(lambda r: calendar.monthrange(int(r["year"]), m)[1], axis=1)
        complete = d[d["n_days"] >= min_complete_frac * expected_days]

        ax.bar(complete["year"], complete["total"], color="tab:blue", alpha=0.55, width=0.8)

        if len(complete) >= 2:
            slope, intercept = np.polyfit(complete["year"], complete["total"], 1)
            trend = slope * complete["year"] + intercept
            trend_color = "tab:red" if slope > 0 else "tab:blue"
            ax.plot(complete["year"], trend, color=trend_color, linewidth=1.8, linestyle="--", zorder=4)
            trend_label = f"{slope * 10:+.1f} mm/decade"
        else:
            trend_label = "n/a"

        if len(complete) > poly_degree:
            poly_coeffs = np.polyfit(complete["year"], complete["total"], poly_degree)
            x_smooth = np.linspace(complete["year"].min(), complete["year"].max(), 100)
            ax.plot(x_smooth, np.polyval(poly_coeffs, x_smooth),
                    color="tab:orange", linewidth=1.8, linestyle="-.", zorder=5)

        ax.set_title(f"{MONTH_LABELS[m - 1]}  ({trend_label})", loc="left", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2, axis="y")

    fig.supxlabel("Year", fontsize=10)
    fig.supylabel("Monthly total precipitation (mm)", fontsize=10)
    fig.suptitle(f"{place}: has each calendar month's precipitation increased over time?",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")

    _caption(fig, source_note + (
        " Dashed line = linear trend (red = increasing, blue = decreasing), dash-dot orange = "
        f"polynomial fit, both via np.polyfit/np.polyval on that month's yearly totals; a year is "
        f"excluded from its month's panel if fewer than {min_complete_frac:.0%} of that month's "
        "days were observed."
    ))
    return fig, axes


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


def plot_pressure_vs_normal(
    df, place, pressure_col, reference_years=(1991, 2020), current_year=2025, source_note="",
):
    """Part 3a: the section's actual framing question is "is this pressure
    normal for X?" - plot_pressure_monthly_boxplot answers "explore
    variations during a month" and plot_pressure_annual_trend answers "has
    it changed over time", but neither directly shows a specific day's
    pressure against what's typical for that time of year. This mirrors
    plot_temperature_vs_normal for Part 1: a smoothed 10th-90th percentile
    "typical" day-of-year band (plus the mean) over the reference period,
    with the given year's actual daily pressure overlaid."""
    ref, used_years = _reference_period(df, reference_years)
    cur = df[df["year"] == current_year]

    g = ref.groupby("doy")[pressure_col]
    mean = _smooth_doy(g.mean())
    p10 = _smooth_doy(g.quantile(0.10))
    p90 = _smooth_doy(g.quantile(0.90))

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    _shade_months(ax)

    ax.fill_between(p10.index, p10, p90, color="tab:purple", alpha=0.20,
                     label=f"Typical range (10th–90th pct.), {used_years[0]}–{used_years[1]}")
    ax.plot(mean.index, mean, color="tab:purple", alpha=0.8, linewidth=1.4,
            linestyle="--", label="Normal daily pressure")

    if not cur.empty:
        ax.plot(cur["doy"], cur[pressure_col], color="black", linewidth=1.2,
                label=f"{current_year} daily pressure")

    ax.set_xlim(1, 365)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Mean sea-level pressure (hPa)")
    ax.set_title(
        f"{place}: is {current_year} pressure normal? (vs. {used_years[0]}–{used_years[1]} normal)",
        loc="left", fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    _add_top_margin(ax, frac=0.14)
    _caption(fig, source_note)
    return fig, ax


def plot_pressure_monthly_boxplot(df, pressure_col, place, source_note=""):
    """Part 3a: within-month pressure variation. Box = 25th-75th
    percentile, whiskers = true min-max, both computed from each year's
    mean pressure for that month - the same points shown as the coloured
    scatter (one dot per year, coloured by year) - rather than from daily
    values. A box built from daily values would have whisker tips no
    scattered point could ever reach, since a month's mean is always less
    extreme than its most extreme day (averaging narrows the range); here
    the box/whiskers and the dots are the same data, so the whisker tips
    are always an actual plotted year. Day-to-day variation is instead
    covered by plot_pressure_vs_normal, which shows actual daily pressure."""
    d = df.dropna(subset=[pressure_col])
    monthly_means = d.groupby(["year", "month"])[pressure_col].mean().reset_index()
    data_by_month = [monthly_means.loc[monthly_means["month"] == m, pressure_col].values
                      for m in range(1, 13)]

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.boxplot(data_by_month, tick_labels=MONTH_LABELS, whis=(0, 100), patch_artist=True,
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
               alpha=0.7, s=16, linewidths=0, zorder=2)

    ax.set_ylabel("Mean sea-level pressure (hPa)")
    ax.set_title(f"{place}: pressure variation within a month", loc="left", fontsize=13, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="tab:purple", alpha=0.4, label="25th–75th percentile"),
            Line2D([0], [0], color="0.3", linewidth=1.2, label="Min–max"),
            Line2D([0], [0], color="black", linewidth=1.8, label="Median"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="white",
                   markeredgecolor="black", markersize=7, markeredgewidth=1.3, label="Mean"),
        ],
        loc="upper left", fontsize=8, frameon=False,
    )
    ax.grid(alpha=0.2, axis="y")
    _add_top_margin(ax, frac=0.20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, aspect=30, label="Year")

    _caption(fig, source_note + (
        " Each dot is one year's mean pressure for that month, coloured by year; "
        "box/whiskers summarise that same set of points."
    ))
    return fig, ax


def plot_pressure_annual_trend(df, pressure_col, place, smooth_years=10, poly_degree=2, source_note=""):
    """Part 3a: has pressure changed over time? A single straight trend
    line through noisy annual means is easy to eyeball as "no trend" even
    when a real one exists, since year-to-year variability swamps a slow
    drift - the same issue precipitation had (see
    plot_precip_cumulative_trend). Two more robust, standard views
    overlaid on the annual means instead: a smooth_years-year centered
    moving average (a data-driven local smoother) and a degree-poly_degree
    polynomial fit via np.polyfit/np.polyval (a single smooth curve over
    the whole period, not tied to a window length)."""
    annual = df.groupby("year")[pressure_col].mean().dropna()

    mean_all = annual.mean()
    rolling = annual.rolling(smooth_years, center=True, min_periods=max(3, smooth_years // 2)).mean()

    slope, _ = np.polyfit(annual.index, annual.values, 1)
    poly_coeffs = np.polyfit(annual.index, annual.values, poly_degree)
    x_smooth = np.linspace(annual.index.min(), annual.index.max(), 200)
    poly_curve = np.polyval(poly_coeffs, x_smooth)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.plot(annual.index, annual.values, marker="o", ms=3, linewidth=1,
            color="tab:purple", alpha=0.5, label="Annual mean pressure")
    ax.plot(rolling.index, rolling.values, color="black", linewidth=2.2,
            label=f"{smooth_years}-year moving average")
    ax.plot(x_smooth, poly_curve, color="tab:orange", linewidth=2.2, linestyle="-.",
            label=f"Degree-{poly_degree} polynomial fit")
    ax.axhline(mean_all, color="0.4", linewidth=1, linestyle=":",
               label=f"Long-term mean ({mean_all:.1f} hPa)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean sea-level pressure (hPa)")
    ax.set_title(f"{place}: has pressure changed over time?", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    _add_top_margin(ax, frac=0.20)
    _caption(fig, source_note + (
        f" Overall linear trend across the full period: {slope * 10:+.2f} hPa/decade "
        "(the polynomial fit is for shape, not a trend rate)."
    ))
    return fig, ax
