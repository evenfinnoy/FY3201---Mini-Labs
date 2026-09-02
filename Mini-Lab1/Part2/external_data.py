"""
Downloads and reads the supporting datasets for Q3 (CO2, CH4, population)
and Q4 (world cocoa bean production, our "steadily growing but absurd"
variable) - all as annual global means/totals, ready to harmonize against
the temperature record's annual mean.

Each source is downloaded once into data_cache/ and read from that local
copy afterwards, so re-runs are fast and offline-safe and the analysis
doesn't silently change if the upstream file is revised later. The cached
files are the untouched originals - parsing happens entirely in code.

Sources (record these in the report too):
  CO2, monthly global mean, ppm - NOAA GML
    https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv
  CH4, monthly global mean, ppb - NOAA GML
    https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv
  World population, annual, UN World Population Prospects (2024 rev.)
  via Our World in Data
    https://ourworldindata.org/grapher/population-unwpp.csv
  World cocoa bean production, annual, tonnes - FAO, via Our World in Data
  (a real, downloadable stand-in for "world chocolate consumption")
    https://ourworldindata.org/grapher/cocoa-bean-production.csv
"""

import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

DATA_CACHE_DIR = Path(__file__).resolve().parent / "data_cache"

CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv"
CH4_URL = "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv"
POPULATION_URL = "https://ourworldindata.org/grapher/population-unwpp.csv"
COCOA_URL = "https://ourworldindata.org/grapher/cocoa-bean-production.csv"

# name -> cache filename, for looking citations back up after loading.
CACHE_FILENAMES = {
    "CO2": "co2_mm_gl.csv",
    "CH4": "ch4_mm_gl.csv",
    "Population": "population-unwpp.csv",
    "Cocoa": "cocoa-bean-production.csv",
}


USER_AGENT = "Mozilla/5.0 (compatible; mini-project-1-data-fetch/1.0)"


def fetch(url: str, filename: str) -> Path:
    """Download `url` to data_cache/`filename` if not already cached, and
    return the local path. The source URL and the actual download date are
    written to a `<filename>.source.txt` sidecar the first time - so that
    record survives even if this only ever prints to a console no one saved
    (see citation() below to read it back).

    Sends a browser-like User-Agent: some hosts (e.g. Our World in Data,
    behind Cloudflare) return 403 Forbidden for Python's default
    "Python-urllib/x.y" string, even though the URL works fine in a
    browser or with curl."""
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    path = DATA_CACHE_DIR / filename
    meta_path = DATA_CACHE_DIR / f"{filename}.source.txt"
    if not path.exists():
        download_date = date.today().isoformat()
        print(f"Downloading {filename}\n  from {url}\n  on {download_date} ...")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
        meta_path.write_text(f"url: {url}\ndownloaded: {download_date}\n")
    return path


def citation(filename: str) -> str:
    """Read back a cached file's source URL and download date, e.g. for
    printing in a report or a figure caption: 'downloaded 2026-08-27 from
    https://...'."""
    meta_path = DATA_CACHE_DIR / f"{filename}.source.txt"
    if not meta_path.exists():
        return f"(no local download record for {filename} - run the loader once first)"
    info = dict(line.split(": ", 1) for line in meta_path.read_text().strip().splitlines())
    return f"downloaded {info['downloaded']} from {info['url']}"


def _annual_mean_from_noaa(url: str, filename: str) -> pd.Series:
    """Read a NOAA GML global-mean CSV (comment lines start with '#',
    columns year,month,decimal,average,...) and return its annual mean,
    keeping only calendar years with all 12 months present."""
    path = fetch(url, filename)
    df = pd.read_csv(path, comment="#")
    by_year = df.groupby("year")["average"]
    full_years = by_year.count()[by_year.count() == 12].index
    return by_year.mean().loc[full_years]


def _world_series_from_owid(url: str, filename: str) -> pd.Series:
    """Read an Our World in Data grapher CSV (columns Entity, Code, Year,
    <value>) and return the World row as a Year-indexed series."""
    path = fetch(url, filename)
    df = pd.read_csv(path)
    value_col = df.columns[-1]
    world = df.loc[df["Entity"] == "World", ["Year", value_col]]
    return world.set_index("Year")[value_col].sort_index()


def load_co2_annual() -> pd.Series:
    """Global mean atmospheric CO2, ppm, annual mean of the monthly series."""
    return _annual_mean_from_noaa(CO2_URL, "co2_mm_gl.csv")


def load_ch4_annual() -> pd.Series:
    """Global mean atmospheric CH4, ppb, annual mean of the monthly series."""
    return _annual_mean_from_noaa(CH4_URL, "ch4_mm_gl.csv")


def load_population_annual() -> pd.Series:
    """World population, annual (UN World Population Prospects 2024)."""
    return _world_series_from_owid(POPULATION_URL, "population-unwpp.csv")


def load_cocoa_production_annual() -> pd.Series:
    """World cocoa bean production, annual, tonnes (FAO via OWID) - our
    real, downloadable stand-in for "world chocolate consumption"."""
    return _world_series_from_owid(COCOA_URL, "cocoa-bean-production.csv")


def print_citations(names) -> None:
    """Print 'downloaded <date> from <url>' for each dataset name in
    `names` (keys of CACHE_FILENAMES) - call after loading, so the record
    reflects files that actually exist on disk."""
    print("Data sources (record these in your report too):")
    for name in names:
        print(f"  {name:<12} {citation(CACHE_FILENAMES[name])}")
    print()
