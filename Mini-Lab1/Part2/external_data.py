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


def fetch(url: str, filename: str) -> Path:
    """Download `url` to data_cache/`filename` if not already cached, and
    return the local path."""
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    path = DATA_CACHE_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}\n  from {url}\n  on {date.today().isoformat()} ...")
        urllib.request.urlretrieve(url, path)
    return path


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
