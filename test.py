"""
Snow Weather Feature Engineering Pipeline
Uses Open-Meteo historical archive API (free, no key required).
OpenWeather key kept for reference / supplemental forecasting.
"""

import requests
import numpy as np
import pandas as pd
from typing import Optional

# ── API credentials ──────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = "98ae72981ee4afea2053fde33092f6b6"  # OpenWeatherMap

# Open-Meteo endpoints — no key needed
OPEN_METEO_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "snowfall",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "cloud_cover",
]


# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Return hourly DataFrame from Open-Meteo archive API."""
    resp = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude":           lat,
            "longitude":          lon,
            "start_date":         start,
            "end_date":           end,
            "hourly":             ",".join(HOURLY_VARS),
            "timezone":           "auto",
            "wind_speed_unit":    "kmh",
            "precipitation_unit": "mm",
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


# ── Daily aggregation ─────────────────────────────────────────────────────────
def _agg_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Core per-day aggregations from hourly data."""
    h = hourly.copy()
    h["date"]    = h.index.normalize()          # midnight timestamps for groupby
    h["is_snow"] = h["snowfall"] > 0

    # ── Base aggregates ───────────────────────────────────────────────────
    base = h.groupby("date").agg(
        snowfall_24h         =("snowfall",             "sum"),
        precipitation        =("precipitation",        "sum"),
        hours_with_snow      =("is_snow",              "sum"),
        max_snowfall_1h      =("snowfall",             "max"),
        temp_min             =("temperature_2m",       "min"),
        temp_max             =("temperature_2m",       "max"),
        temp_mean            =("temperature_2m",       "mean"),
        wind_max             =("wind_speed_10m",       "max"),
        wind_mean            =("wind_speed_10m",       "mean"),
        humidity_mean        =("relative_humidity_2m", "mean"),
        radiation_mean       =("shortwave_radiation",  "mean"),
        radiation_peak       =("shortwave_radiation",  "max"),
        cloud_cover_mean     =("cloud_cover",          "mean"),
    )

    # ── Snow-conditional aggregates (snow hours only) ─────────────────────
    # Days with no snowfall get 0, not NaN — keeps data consistent.
    snow_only = h[h["is_snow"]]
    if not snow_only.empty:
        cond = snow_only.groupby("date").agg(
            snow_temp_mean        =("temperature_2m",       "mean"),
            wind_during_snow_mean =("wind_speed_10m",       "mean"),
            wind_during_snow_max  =("wind_speed_10m",       "max"),
            humidity_during_snow  =("relative_humidity_2m", "mean"),
        )
        base = base.join(cond)
    # Fill any days with zero snowfall hours with 0 (not NaN)
    snow_cond_cols = ["snow_temp_mean", "wind_during_snow_mean",
                      "wind_during_snow_max", "humidity_during_snow"]
    for col in snow_cond_cols:
        if col not in base.columns:
            base[col] = 0.0
    no_snow_mask = base["hours_with_snow"] == 0
    base.loc[no_snow_mask, snow_cond_cols] = 0.0

    # ── Clear-sky fraction ────────────────────────────────────────────────
    base["clear_sky_fraction"] = (
        h.groupby("date")["cloud_cover"].apply(lambda x: (x < 20).mean())
    )

    # ── Freeze-thaw event (crossed 0°C both ways within the day) ─────────
    base["freeze_thaw_event"] = h.groupby("date")["temperature_2m"].apply(
        lambda t: int(
            ((t.shift(1) < 0) & (t >= 0)).any()
            and ((t.shift(1) >= 0) & (t < 0)).any()
        )
    )

    # ── Longest continuous snowfall block per day ─────────────────────────
    def _max_run(s):
        best = cur = 0
        for v in s > 0:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    base["storm_duration_hours"] = h.groupby("date")["snowfall"].apply(_max_run)

    return base


# ── Full feature engineering ──────────────────────────────────────────────────
def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Compute all Tier 1–4 and critical derived features on top of daily aggs."""
    d = _agg_daily(hourly)

    # ── TIER 1: Rolling snow windows ──────────────────────────────────────
    d["snowfall_48h"]      = d["snowfall_24h"].rolling(2,  min_periods=1).sum()
    d["snowfall_72h"]      = d["snowfall_24h"].rolling(3,  min_periods=1).sum()
    d["snowfall_7d"]       = d["snowfall_24h"].rolling(7,  min_periods=1).sum()
    d["snowfall_14d"]      = d["snowfall_24h"].rolling(14, min_periods=1).sum()
    d["snowfall_last_30d"] = d["snowfall_24h"].rolling(30, min_periods=1).sum()

    # ── TIER 2: Freeze-thaw rolling ───────────────────────────────────────
    d["freeze_thaw_count_7d"] = (
        d["freeze_thaw_event"].rolling(7, min_periods=1).sum()
    )

    # ── TIER 3: Seasonal cumulative (season resets Oct 1) ─────────────────
    d["_season"] = d.index.map(
        lambda dt: dt.year if dt.month >= 10 else dt.year - 1
    )
    d["seasonal_snowfall_total"] = (
        d.groupby("_season")["snowfall_24h"].cumsum()
    )
    d.drop(columns=["_season"], inplace=True)

    # ── Days since last snow / major snow (≥10 cm) ────────────────────────
    snow_idx  = d.index[d["snowfall_24h"] > 0]
    major_idx = d.index[d["snowfall_24h"] >= 10]

    def _days_since(idx_arr, dt):
        prev = idx_arr[idx_arr <= dt]
        return (dt - prev[-1]).days if len(prev) else np.nan

    d["days_since_last_snow"]  = [_days_since(snow_idx,  dt) for dt in d.index]
    d["days_since_major_snow"] = [_days_since(major_idx, dt) for dt in d.index]

    # ── Snowfall decay ratios ─────────────────────────────────────────────
    d["snowfall_ratio_1d_3d"] = (
        d["snowfall_24h"] / d["snowfall_72h"].replace(0, np.nan)
    )
    d["snowfall_ratio_3d_7d"] = (
        d["snowfall_72h"] / d["snowfall_7d"].replace(0, np.nan)
    )

    # ── Wind loading index (wind × snowfall proxy) — capped at 100 ─────────
    d["wind_loading_index"] = (d["wind_max"] * d["snowfall_24h"]).clip(upper=100)

    # ── Time since storm peak (trailing 7-day window) ─────────────────────
    tsp = []
    for i, dt in enumerate(d.index):
        window = d.iloc[max(0, i - 6): i + 1]["snowfall_24h"]
        tsp.append((dt - window.idxmax()).days)
    d["time_since_storm_peak"] = tsp

    # ── Cap extreme raw values ────────────────────────────────────────────
    d["radiation_peak"] = d["radiation_peak"].clip(upper=600)

    # ── CRITICAL DERIVED FEATURES ─────────────────────────────────────────
    # Normalised helpers (clipped to [0, 1])
    temp_norm  = ((d["snow_temp_mean"] - (-30)) / (10 - (-30))).clip(0, 1)
    wind_norm  = (d["wind_during_snow_mean"] / 80).clip(0, 1)   # 80 km/h ref

    # 1. Snow Quality Index
    #    snowfall × (1 − temp_norm) × (1 − wind_norm) — cold & calm → high
    d["snow_quality_index"] = (
        d["snowfall_24h"] * (1 - temp_norm) * (1 - wind_norm)
    ).fillna(0)

    # 2. Powder Freshness — recent dump relative to days elapsed
    d["powder_freshness"] = (
        d["snowfall_72h"] / (d["days_since_last_snow"].fillna(999) + 1)
    )

    # 3. Melt Risk — high temp + solar radiation + humidity
    d["melt_risk"] = (
        ((d["temp_max"]    - 0) / 15 ).clip(0, 1)   # >15 °C → full risk
        + (d["radiation_mean"] / 500).clip(0, 1)   # 500 W/m² reference
        + (d["humidity_mean"]  / 100).clip(0, 1)
    ) / 3

    # 4. Wind Damage Score — wind during snowfall × daily snowfall
    d["wind_damage_score"] = (
        d["wind_during_snow_mean"] * d["snowfall_24h"]
    )

    # ── SNOWPACK STATE INDICATORS ─────────────────────────────────────────
    # has_base_snow: distinguishes mid-season dry spell from pre-season dust
    BASE_SNOW_THRESHOLD = 5  # cm
    d["has_base_snow"] = (d["seasonal_snowfall_total"] > BASE_SNOW_THRESHOLD).astype(int)

    # ── TEMPORAL / TREND FEATURES ─────────────────────────────────────────
    # Snowfall intensity ratio: burst storm vs steady snow
    d["intensity_ratio"] = (
        d["max_snowfall_1h"] / d["snowfall_24h"].replace(0, np.nan)
    ).fillna(0)

    # Post-storm degradation proxy: wind exposure × days since peak
    d["wind_after_storm"] = d["wind_mean"] * d["time_since_storm_peak"]

    # Temperature trend: warming/cooling relative to previous day
    d["temp_trend"] = d["temp_mean"].diff().fillna(0)

    # ── EARLY-SEASON NOISE FILTER ─────────────────────────────────────────
    # Drop rows where there is no meaningful snowpack yet:
    # keep if seasonal_snowfall_total > 5 cm OR snowfall_7d > 2 cm
    d = d[
        (d["seasonal_snowfall_total"] > 5) | (d["snowfall_7d"] > 2)
    ].copy()

    return d


# ── Public API ────────────────────────────────────────────────────────────────
def get_snow_features(
    lat: float,
    lon: float,
    start: str,
    end: str,
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo and return engineered snow features.

    Parameters
    ----------
    lat, lon   : decimal degrees
    start, end : 'YYYY-MM-DD'
    save_csv   : optional path to write CSV output
    """
    print(f"Fetching {start} → {end}  ({lat}, {lon}) …")
    hourly = fetch_hourly(lat, lon, start, end)
    print(f"  {len(hourly):,} hourly rows retrieved.")

    print("Engineering features …")
    features = build_features(hourly)
    print(f"  {len(features)} days × {features.shape[1]} features.")

    if save_csv:
        features.to_csv(save_csv)
        print(f"  Saved → {save_csv}")

    return features


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: Park City Mountain, Utah — 2024-25 ski season
    df = get_snow_features(
        lat=40.6461,
        lon=-111.4980,
        start="2024-11-01",
        end="2025-04-30",
        save_csv="snow_features.csv",
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n─── First 5 days with snowfall ───")
    print(df[df["snowfall_24h"] > 0].head(5).T)

    print("\n─── All features ───")
    for col in df.columns:
        print(f"  {col}")