"""
Open-Meteo historical weather fetcher.
Fetches hourly data for a given lat/lon and caches to CSV.
No API key required.
"""

import time
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL       = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL      = "https://api.open-meteo.com/v1/forecast"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Surface variables — available from archive API for the full 2014+ history.
HOURLY_VARS = [
    "temperature_2m",
    "dewpoint_2m",
    "snowfall",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
]

# Pressure-level variables — only available from the historical-forecast API,
# which covers 2021-03-23 onward. Pre-2021 rows will be NaN (imputed in dataset.py).
PRESSURE_LEVEL_VARS = [
    "temperature_850hPa",   # snow/rain line indicator (~1500m asl)
]

PRESSURE_LEVEL_START = "2021-03-23"  # earliest date with non-null data

RAW_DIR = Path("data/raw/weather")


def fetch_hourly(
    lat: float,
    lon: float,
    start: str,
    end: str,
    forecast: bool = False,
) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo. Returns DataFrame indexed by UTC time.

    URL + variable selection:
    - forecast=True                         → FORECAST_URL   + 850hPa
    - forecast=False, start ≥ 2021-03-23   → HIST_FORECAST_URL + 850hPa
    - forecast=False, start <  2021-03-23  → ARCHIVE_URL, surface only
    """
    if forecast or pd.Timestamp(start) >= pd.Timestamp(PRESSURE_LEVEL_START):
        url         = FORECAST_URL if forecast else HIST_FORECAST_URL
        hourly_vars = HOURLY_VARS + PRESSURE_LEVEL_VARS
    else:
        url         = ARCHIVE_URL
        hourly_vars = HOURLY_VARS

    resp = requests.get(
        url,
        params={
            "latitude":           lat,
            "longitude":          lon,
            "start_date":         start,
            "end_date":           end,
            "hourly":             ",".join(hourly_vars),
            "timezone":           "auto",
            "wind_speed_unit":    "kmh",
            "precipitation_unit": "mm",
            "cell_selection":     "nearest",
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def fetch_and_cache(
    resort_id: str,
    lat: float,
    lon: float,
    start: str = "2001-10-01",
    end: str   = "2025-05-31",
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly weather for a resort and cache to CSV.
    Skips the network call if the cache file already exists (unless force=True).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{resort_id}_{start}_{end}.csv"

    if cache_path.exists() and not force:
        print(f"  [{resort_id}] loading from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print(f"  [{resort_id}] fetching {start} → {end} ({lat}, {lon}) …")
    df = fetch_hourly(lat, lon, start, end)
    df.to_csv(cache_path)
    print(f"  [{resort_id}] cached {len(df):,} hourly rows → {cache_path}")
    time.sleep(1)  # polite delay
    return df
