"""
Daily feature engineering from hourly Open-Meteo data.
All rolling windows use season_year (not calendar year) so Northern Hemisphere
winters that cross Jan 1 are handled correctly.
"""

import numpy as np
import pandas as pd


def _season_year(dt) -> int:
    """Return the year the ski season started: Oct-Dec → that year, Jan-Sep → year-1."""
    return dt.year if dt.month >= 10 else dt.year - 1


def _agg_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather to per-day rows."""
    h = hourly.copy()
    h["date"]    = h.index.normalize()
    h["is_snow"] = h["snowfall"] > 0

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

    # Snow-conditional aggregates — 0 (not NaN) on no-snow days
    snow_only = h[h["is_snow"]]
    if not snow_only.empty:
        cond = snow_only.groupby("date").agg(
            snow_temp_mean        =("temperature_2m",       "mean"),
            wind_during_snow_mean =("wind_speed_10m",       "mean"),
            wind_during_snow_max  =("wind_speed_10m",       "max"),
            humidity_during_snow  =("relative_humidity_2m", "mean"),
        )
        base = base.join(cond)

    snow_cond_cols = [
        "snow_temp_mean", "wind_during_snow_mean",
        "wind_during_snow_max", "humidity_during_snow",
    ]
    for col in snow_cond_cols:
        if col not in base.columns:
            base[col] = 0.0
    base.loc[base["hours_with_snow"] == 0, snow_cond_cols] = 0.0

    base["clear_sky_fraction"] = (
        h.groupby("date")["cloud_cover"].apply(lambda x: (x < 20).mean())
    )

    base["freeze_thaw_event"] = h.groupby("date")["temperature_2m"].apply(
        lambda t: int(
            ((t.shift(1) < 0) & (t >= 0)).any()
            and ((t.shift(1) >= 0) & (t < 0)).any()
        )
    )

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


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Tier 1-4 and critical derived features on top of daily aggregates.
    Input: hourly DataFrame from Open-Meteo (indexed by datetime).
    Output: daily DataFrame with ~40 engineered features.
    """
    d = _agg_daily(hourly)

    # ── Rolling snow windows ───────────────────────────────────────────────────
    d["snowfall_48h"]      = d["snowfall_24h"].rolling(2,  min_periods=1).sum()
    d["snowfall_72h"]      = d["snowfall_24h"].rolling(3,  min_periods=1).sum()
    d["snowfall_7d"]       = d["snowfall_24h"].rolling(7,  min_periods=1).sum()
    d["snowfall_14d"]      = d["snowfall_24h"].rolling(14, min_periods=1).sum()
    d["snowfall_last_30d"] = d["snowfall_24h"].rolling(30, min_periods=1).sum()

    # ── Freeze-thaw rolling ────────────────────────────────────────────────────
    d["freeze_thaw_count_7d"] = d["freeze_thaw_event"].rolling(7, min_periods=1).sum()

    # ── Seasonal cumulative (resets at season start, Oct 1 for N. hemisphere) ──
    # Uses season_year so the window never crosses a calendar year boundary.
    d["_season_year"] = d.index.map(_season_year)
    d["seasonal_snowfall_total"] = (
        d.groupby("_season_year")["snowfall_24h"].cumsum()
    )
    d.drop(columns=["_season_year"], inplace=True)

    # ── Days since last snow / major snow (>= 10 cm) ──────────────────────────
    snow_idx  = d.index[d["snowfall_24h"] > 0]
    major_idx = d.index[d["snowfall_24h"] >= 10]

    def _days_since(idx_arr, dt):
        prev = idx_arr[idx_arr <= dt]
        return (dt - prev[-1]).days if len(prev) else np.nan

    d["days_since_last_snow"]  = [_days_since(snow_idx,  dt) for dt in d.index]
    d["days_since_major_snow"] = [_days_since(major_idx, dt) for dt in d.index]

    # ── Snowfall decay ratios ──────────────────────────────────────────────────
    d["snowfall_ratio_1d_3d"] = d["snowfall_24h"] / d["snowfall_72h"].replace(0, np.nan)
    d["snowfall_ratio_3d_7d"] = d["snowfall_72h"] / d["snowfall_7d"].replace(0, np.nan)

    # ── Wind loading index (capped) ────────────────────────────────────────────
    d["wind_loading_index"] = (d["wind_max"] * d["snowfall_24h"]).clip(upper=100)

    # ── Time since storm peak (trailing 7-day window) ─────────────────────────
    tsp = []
    for i, dt in enumerate(d.index):
        window = d.iloc[max(0, i - 6): i + 1]["snowfall_24h"]
        tsp.append((dt - window.idxmax()).days)
    d["time_since_storm_peak"] = tsp

    d["radiation_peak"] = d["radiation_peak"].clip(upper=600)

    # ── Critical derived features ──────────────────────────────────────────────
    temp_norm = ((d["snow_temp_mean"] - (-30)) / (10 - (-30))).clip(0, 1)
    wind_norm = (d["wind_during_snow_mean"] / 80).clip(0, 1)

    d["snow_quality_index"] = (
        d["snowfall_24h"] * (1 - temp_norm) * (1 - wind_norm)
    ).fillna(0)

    d["powder_freshness"] = (
        d["snowfall_72h"] / (d["days_since_last_snow"].fillna(999) + 1)
    )

    d["melt_risk"] = (
        ((d["temp_max"] - 0) / 15).clip(0, 1)
        + (d["radiation_mean"] / 500).clip(0, 1)
        + (d["humidity_mean"] / 100).clip(0, 1)
    ) / 3

    d["wind_damage_score"] = d["wind_during_snow_mean"] * d["snowfall_24h"]

    # ── Snowpack state ─────────────────────────────────────────────────────────
    d["has_base_snow"] = (d["seasonal_snowfall_total"] > 5).astype(int)

    # ── Temporal / trend features ──────────────────────────────────────────────
    d["intensity_ratio"] = (
        d["max_snowfall_1h"] / d["snowfall_24h"].replace(0, np.nan)
    ).fillna(0)

    d["wind_after_storm"] = d["wind_mean"] * d["time_since_storm_peak"]
    d["temp_trend"]       = d["temp_mean"].diff().fillna(0)

    # ── Calendar features ──────────────────────────────────────────────────────
    d["day_of_year"]   = d.index.day_of_year
    d["month"]         = d.index.month
    d["season_year"]   = d.index.map(_season_year)
    d["days_in_season"] = (
        d.groupby("season_year").cumcount()
    )

    return d
