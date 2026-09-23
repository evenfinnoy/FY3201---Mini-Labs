"""
Statistical tests comparing one year's daily values against the reference
period's ("climate normal") daily values for the same variable - answering
"is {test_year} normal?" at the level of the whole distribution, not just
its mean.

Test choice: a two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp) is
used as the primary test, not a t-test. A t-test assumes the underlying
values are approximately normal (or that the sample is large enough for the
CLT to cover for it) and only tests for a difference in means - it can miss
a real change in spread or shape that leaves the mean almost unchanged, and
daily precipitation in particular is right-skewed (see the gamma-fit plots
in climate_plots.py), not normal. The KS test instead compares the two
empirical distributions directly, with no assumption about their shape, and
its test statistic D (the largest vertical gap between the two empirical
CDFs) has a direct, plottable meaning - see
climate_plots.plot_precip_ecdf_2025_vs_normal for the visual companion to
this test.

A Mann-Whitney U test is reported alongside it as a secondary, more
targeted test: KS can detect any kind of distributional difference
(location, spread, shape) but is comparatively weak at detecting any one of
them; Mann-Whitney specifically tests whether one sample's values tend to
be systematically higher or lower than the other's (a shift), which is
usually the more practically interesting question ("was the year wetter or
drier than normal?").
"""

from scipy import stats


def compare_year_to_normal(
    df, value_col, reference_years=(1991, 2020), test_year=2025,
    label=None, unit="", min_value=None, alpha=0.05,
):
    """Compare test_year's daily values for value_col against the
    reference-period daily values for the same column.

    min_value: if set (e.g. 1.0 for a "wet day" precipitation threshold),
    both samples are filtered to values strictly greater than this before
    testing - i.e. this is where you'd encode a particular definition of
    "amount of precipitation" (all days vs. wet days only; see the Part 2
    teamwork task on defining that). Left as None, every day (including
    dry days at 0 mm) is used, which folds both "how often" and "how much"
    together into a single test.

    Returns a dict of the sample sizes, summary stats, and both test
    results, and also prints a short human-readable report.
    """
    label = label or value_col
    ref = df.loc[df["year"].between(*reference_years), value_col].dropna()
    year = df.loc[df["year"] == test_year, value_col].dropna()
    if min_value is not None:
        ref = ref[ref > min_value]
        year = year[year > min_value]

    ks_stat, ks_p = stats.ks_2samp(ref, year)
    mwu_stat, mwu_p = stats.mannwhitneyu(ref, year, alternative="two-sided")

    result = {
        "label": label, "unit": unit,
        "reference_years": reference_years, "test_year": test_year,
        "min_value": min_value,
        "n_reference": len(ref), "n_test_year": len(year),
        "reference_mean": ref.mean(), "test_year_mean": year.mean(),
        "reference_median": ref.median(), "test_year_median": year.median(),
        "ks_statistic": ks_stat, "ks_pvalue": ks_p,
        "mannwhitney_statistic": mwu_stat, "mannwhitney_pvalue": mwu_p,
        "alpha": alpha,
    }

    def verdict(p):
        return "REJECT H0 (different)" if p < alpha else "fail to reject H0 (consistent with normal)"

    filt = f" (values > {min_value}{unit} only)" if min_value is not None else " (all days)"
    print(f"--- {label}: {test_year} vs. {reference_years[0]}–{reference_years[1]} normal{filt} ---")
    print(f"  n: {test_year}={result['n_test_year']}, reference={result['n_reference']}")
    print(f"  mean:   {test_year}={result['test_year_mean']:.2f}{unit}  vs  "
          f"reference={result['reference_mean']:.2f}{unit}")
    print(f"  median: {test_year}={result['test_year_median']:.2f}{unit}  vs  "
          f"reference={result['reference_median']:.2f}{unit}")
    print(f"  Two-sample Kolmogorov–Smirnov (distribution shape): "
          f"D={ks_stat:.3f}, p={ks_p:.4g} -> {verdict(ks_p)}")
    print(f"  Mann-Whitney U (systematic shift):                    "
          f"p={mwu_p:.4g} -> {verdict(mwu_p)}")

    return result
