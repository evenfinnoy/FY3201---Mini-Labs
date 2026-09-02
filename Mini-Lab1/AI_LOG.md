# AI Log — Mini-Project 1: Global Temperature

**Tool used:** Claude Code (Claude Sonnet 5, Anthropic), used interactively in a chat/CLI
session with direct access to the repository files, a Python environment, and (for Part 2,
Q3–Q4) live internet access to download data.

**Scope of use:** AI wrote essentially all of the Python code in `Part1/` and `Part2/` in
this repository, across a single ongoing conversation. Every script was produced in
response to a specific request from the team (usually a photo of one bullet point or
section from the assignment PDF), reviewed by the team by actually running it and looking
at the output, and revised again based on team feedback. Two bugs the team introduced
themselves while editing the code directly were also diagnosed and fixed by the AI (see
entries 8 and 9 below) — flagged here explicitly since those are places the team wrote code
that didn't end up in the final files.

This log is a factual record of what was asked and done. The "You understand and can
explain your results" requirement from the assignment is not something the AI can certify
on the team's behalf — before submitting, the team should be able to explain, without this
log open, what each script computes and why (see **Open items for the team**, at the end).

---

## Log of prompts and actions

Each entry: what was asked → what the AI did → what happened to it (accepted / modified /
rejected) and why.

**1. "Can you help me plot the data from airTemps.txt and waterTemps?"** (Part 1 bullet
points: download/read with own code, plot monthly anomaly + 95% CI shaded band, add
12-month moving average)
→ Read the two data files and the assignment PDF; wrote a script that parses the
whitespace-delimited Berkeley Earth format, plots the monthly anomaly with a shaded 95% CI
band and a 12-month centered moving average, saved as a PNG.
→ **Accepted**, but immediately revised (next entry).

**2. "Can you use matplotlib instead? I do not want it to save as a png, just open another
window"**
→ Replaced `fig.savefig(...)` with `plt.show()`.
→ **Accepted.**

**3. VS Code showing `Import "pandas" could not be resolved`**
→ Diagnosed as VS Code/Pylance pointing at a different Python interpreter (Homebrew Python
3.14, no packages installed) than the one the code actually ran with (Python 3.9 via
Command Line Tools, which has pandas/matplotlib). Explained how to switch VS Code's
selected interpreter; no code changed.

**4. "Can you split it into two files, so one shows the air and the other shows the water?
And can you also make it a shaded band as the exercise wants?"**
→ Refactored the single script into a shared `temperature_common.py` (file reading +
plotting logic) plus `plot_air_temperature.py` and `plot_water_temperature.py`. Slightly
increased the band's opacity and added faint edge lines, and confirmed with a zoomed-in
render that the band was genuinely present — it was just visually subtle at full
1850–present zoom because Berkeley Earth's measurement uncertainty is much narrower than
real month-to-month climate variability.
→ **Accepted**; the old combined script (the AI's own earlier file) was deleted.

**5. "Can you add a switchcase to the plotters that makes it so the user can choose which of
the columns to plot?"** (Monthly / Annual / 5-yr / 10-yr / 20-yr columns)
→ Added a `COLUMN_CHOICES` dict (Python's idiomatic switch/case) and a `choose_resolution()`
function that prompted interactively via `input()`.
→ **Rejected** (see next entry) — the interactive prompt wasn't what the team wanted.

**6. "Ah sorry, can you just make this a variable I can change? I don't need the user
input"**
→ Removed `choose_resolution()`/`input()`; replaced it with a plain `RESOLUTION = "monthly"`
variable at the top of each script. The `COLUMN_CHOICES` dict itself (the actual dispatch
table) was kept.
→ **Accepted.** This is the one clear case of an AI suggestion being explicitly reversed by
the team in favor of a simpler approach.

**7. "Can you help us with this as well?"** (Q1: linear rate of warming, primary window
1980–2025, repeat with 1970/1990 start years, annual means before fitting, °C/decade with
uncertainty)
→ Wrote `Part2/Q1.py`: collapses monthly data to annual means, fits OLS trends for three
start years via `scipy.stats.linregress`, reports slope ± 95% CI (Student's t, not a
normal-approximation doubling of the standard error). Judgment call: the brief says
"1980–2025" but the actual downloaded file's last full year is 2024, so the end year is
detected from the data and clipped automatically (with a printed note) rather than
hardcoded to 2025.
→ **Accepted.**

**8. "I tried adding `plt.xlim(1960, 2022)` ... it seems not to do what it's supposed to?"**
(a bug in code the team had edited themselves)
→ Diagnosed: the plot's x-axis holds real dates (`pd.to_datetime`), so bare integers passed
to `xlim` are read as *days since 1970-01-01*, not years — 1960 and 2022 landed the view in
a ~2-month window around mid-1975. Fixed by using `pd.Timestamp(...)` and capturing
`ax = plot_anomaly(...)`'s return value to call `ax.set_xlim(...)` on it directly.
→ **Accepted.**

**9. "Hmm I tried to do the same thing in Q1?"** (the team had copied the entry-8 fix into
`Q1.py`, introducing `from turtle import pd` and wrapping years in `pd.Timestamp(...)`
there too)
→ Diagnosed two separate problems: `from turtle import pd` doesn't work at all (Python's
`turtle` module has no `pd` — likely a bad editor auto-import) and, separately, `Q1.py`'s
x-axis is plain integer calendar years (from `groupby(df.date.dt.year)`), not real dates —
so wrapping in `pd.Timestamp` was the *previous* fix applied to a different kind of axis.
Removed the bad import and reverted to plain `ax.set_xlim(1960, 2022)`.
→ **Accepted.** Notable because it's a second, related-but-distinct bug from entry 8 — the
same surface-level fix doesn't generalize between the two plotting styles used in this
project.

**10. "Can you do this as well?"** (Q2: is the warming linear or accelerating — early/late
split 1970–1997 vs 1998–2025, quadratic fit, report acceleration ± uncertainty)
→ Factored the annual-mean/linear-fit code shared between Q1 and Q2 out into a new
`trend_utils.py` (disclosed as a proactive refactor, not asked for outright), then wrote
`Part2/Q2.py`: independent linear fits on an early and late window with a
quadrature-combined uncertainty on their difference, plus a quadratic fit
(`anomaly = a·t² + b·t + c`) via `np.polyfit(..., cov=True)`, reporting the acceleration
(2·a) with its 95% CI. Re-ran `Q1.py` after the refactor and confirmed byte-identical
numbers to before.
→ **Accepted.**

**11. "Can you do these two as well?"** (Q3: correlate temperature with CO₂, CH₄, world
population; Q4: "the chocolate test" — correlate with an absurd steadily-growing variable)
→ Verified live internet access, located the exact NOAA GML CO₂/CH₄ monthly CSV URLs and
Our World in Data population CSV URL, and built `Part2/external_data.py` as a
download-once/cache-locally layer. Wrote `Q3.py` (harmonizes all four series to their
common annual period via an inner join, then Pearson r + p-value for each pair) and `Q4.py`
(same idea against a single "absurd" variable).
→ **Judgment call flagged explicitly to the team:** a genuine long public "world chocolate
consumption" series wasn't available to download, so **world cocoa bean production** (FAO,
via Our World in Data) was substituted as a labeled, disclosed proxy — the team was told
this is an easy one-line swap in `external_data.py` if they'd rather use something else.
→ Also added a `.gitignore` for the new `data_cache/` folder (proactive, disclosed) since
it's a reproducible download cache rather than original work.
→ **Accepted**, cocoa substitution not revisited.

**12. "Now, in Q2, ... larger STD than the actual data, can you find the p-value, as well as
the actual formula for the quadratic fit? ax² + bx + c?"**
→ Extended `fit_quadratic_trend()` to expose the fitted `a`, `b`, `c` coefficients (matching
the team's naming) with standard errors and two-sided p-values (Student's t, n−3 degrees of
freedom); updated `Q2.py`'s printed output and plot legend. Explained that the "large STD"
the team had run into was expected, correct behavior once the time variable is properly
rescaled to decades-since-start (fitting with raw calendar years like 1970² is numerically
unstable and was the likely original cause) — i.e., not a bug, but the team correctly
noticing that the acceleration term is not significant at 95% (p≈0.08).
→ **Accepted.**

**13. "Can you add a legend to the Q4 plot as well?"**
→ Added a combined legend to Q4's dual-y-axis time-series panel (matplotlib gives twinned
axes separate legends by default, so the two lines' handles were combined manually) and a
legend to its scatter panel.
→ **Accepted.**

**14. "Does all the files respect these pitfalls?"** (the assignment's "Common pitfalls"
list)
→ Audited all 9 scripts against the six listed pitfalls directly (grep + reading the actual
label/citation strings, not from memory). Five were already satisfied. Found one real gap:
Q3/Q4's figure captions cited dataset *names* but not the actual URLs/download dates the
way Part 1/Q1/Q2 did. Fixed by having `fetch()` write a `<filename>.source.txt` sidecar
recording the exact URL and download date next to every cached file, plus a
`print_citations()` helper wired into `Q3.py`/`Q4.py`.
→ While re-verifying this end-to-end (clearing and re-downloading the cache), found and
fixed an **unrelated live bug**: Our World in Data's server (behind Cloudflare) returns
`403 Forbidden` for Python's default `urllib` User-Agent string, even though the same URL
works fine via `curl` or a browser — this had been silently masked earlier because the
files were already cached from prior testing. Fixed by sending a normal `User-Agent`
header.
→ **Accepted**; two fixes delivered beyond the literal question asked, both disclosed.

**15. "Can you create a log as well?"**
→ This document.

---

## Open items for the team

A few places where the AI made a call that the team should actively confirm, not just
inherit silently:

- **Q4's variable is cocoa production, not chocolate consumption.** Read as a stand-in
  because a real consumption series wasn't downloadable; if a better "absurd" variable is
  available, swap it in `Part2/external_data.py`.
- **Q1/Q2's "2025" end year is silently clipped to 2024** (the record's last full calendar
  year) — printed as a note each run, but worth stating explicitly in the report rather
  than leaving it as a script detail.
- **The quadratic fit's acceleration term is not significant at 95%** (p≈0.08 in the last
  run) — a real, defensible result, but one the team should be ready to explain rather than
  just report.
- Before submitting: run every script once as a team, from a clean `Part2/data_cache/`
  deletion, to confirm the whole pipeline (including the live downloads in Q3/Q4)
  reproduces the numbers in the report on a different machine.
