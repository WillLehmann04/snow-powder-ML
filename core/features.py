"""
Daily feature engineering from hourly Open-Meteo data.

New vs original:
  - Wind direction: decomposed to u/v components (circular mean), plus
    mean vector direction during snowfall hours and directional consistency.
    Raw degrees are useless (359deg and 1deg average to 180deg, not 0deg).
  - Pressure: mean sea-level pressure, 24h tendency (falling = incoming storm),
    and minimum daily pressure (deep lows = heavy snow events).
  - Dewpoint depression (temp - dewpoint): low depression = air near saturation
    = precipitation likely. High depression = dry air = no snow even if cold.
  - Pressure tendency rolling: how fast pressure has been falling over 3 days.
  - Season-year rollup is hemisphere-aware (NH: Oct, SH: Apr).
"""

import numpy as np
import pandas as pd


def _season_year_north(dt) -> int:
    """NH: Oct-Dec -> that year, Jan-Sep -> year-1."""
    return dt.year if dt.month >= 10 else dt.year - 1


def _season_year_south(dt) -> int:
    """SH: Apr-Dec -> that year, Jan-Mar -> year-1."""
    return dt.year if dt.month >= 4 else dt.year - 1


def _season_year(dt) -> int:
    """Legacy alias — northern hemisphere."""
    return _season_year_north(dt)


def _vector_mean_direction(directions_deg: pd.Series) -> float:
    """Circular mean of wind directions. Returns degrees [0, 360)."""
    rad = np.deg2rad(directions_deg.dropna())
    if len(rad) == 0:
        return np.nan
    u = np.sin(rad).mean()
    v = np.cos(rad).mean()
    deg = np.rad2deg(np.arctan2(u, v)) % 360
    return float(deg)


def _directional_consistency(directions_deg: pd.Series) -> float:
    """
    Ratio of vector-mean wind speed to scalar-mean wind speed.
    1.0 = all wind from same direction. 0.0 = random / cancelling directions.
    """
    rad = np.deg2rad(directions_deg.dropna())
    if len(rad) == 0:
        return np.nan
    r = np.sqrt(np.sin(rad).mean() ** 2 + np.cos(rad).mean() ** 2)
    return float(r)


def _agg_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather to per-day rows."""
    h = hourly.copy()
    h["date"]    = h.index.normalize()
    h["is_snow"] = h["snowfall"] > 0

    # ── Wind u/v components (handle circular nature of direction) ─────────────
    if "wind_direction_10m" in h.columns:
        dir_rad = np.deg2rad(h["wind_direction_10m"].fillna(0))
        h["wind_u"] = h["wind_speed_10m"] * np.sin(dir_rad)   # eastward
        h["wind_v"] = h["wind_speed_10m"] * np.cos(dir_rad)   # northward
    else:
        h["wind_u"] = 0.0
        h["wind_v"] = 0.0

    agg_dict = dict(
        snowfall_24h         =("snowfall",             "sum"),
        precipitation        =("precipitation",        "sum"),
        hours_with_snow      =("is_snow",              "sum"),
        max_snowfall_1h      =("snowfall",             "max"),
        temp_min             =("temperature_2m",       "min"),
        temp_max             =("temperature_2m",       "max"),
        temp_mean            =("temperature_2m",       "mean"),
        wind_max             =("wind_speed_10m",       "max"),
        wind_mean            =("wind_speed_10m",       "mean"),
        wind_u_mean          =("wind_u",               "mean"),
        wind_v_mean          =("wind_v",               "mean"),
        humidity_mean        =("relative_humidity_2m", "mean"),
        radiation_mean       =("shortwave_radiation",  "mean"),
        radiation_peak       =("shortwave_radiation",  "max"),
        cloud_cover_mean     =("cloud_cover",          "mean"),
    )

    if "pressure_msl" in h.columns:
        agg_dict["pressure_mean"] = ("pressure_msl", "mean")
        agg_dict["pressure_min"]  = ("pressure_msl", "min")

    if "dewpoint_2m" in h.columns:
        agg_dict["dewpoint_mean"] = ("dewpoint_2m", "mean")
        agg_dict["dewpoint_min"]  = ("dewpoint_2m", "min")

    if "temperature_850hPa" in h.columns:
        agg_dict["temp_850_mean"] = ("temperature_850hPa", "mean")
        agg_dict["temp_850_min"]  = ("temperature_850hPa", "min")
        agg_dict["temp_850_max"]  = ("temperature_850hPa", "max")

    base = h.groupby("date").agg(**agg_dict)

    # ── Wind direction features (need apply, can't go in agg dict) ────────────
    if "wind_direction_10m" in h.columns:
        base["wind_dir_mean"] = h.groupby("date")["wind_direction_10m"].apply(
            _vector_mean_direction
        )
        base["wind_dir_consistency"] = h.groupby("date")["wind_direction_10m"].apply(
            _directional_consistency
        )
    else:
        base["wind_dir_mean"] = np.nan
        base["wind_dir_consistency"] = np.nan

    # ── Snow-conditional aggregates ───────────────────────────────────────────
    snow_only = h[h["is_snow"]]
    if not snow_only.empty:
        snow_agg = dict(
            snow_temp_mean        =("temperature_2m",       "mean"),
            wind_during_snow_mean =("wind_speed_10m",       "mean"),
            wind_during_snow_max  =("wind_speed_10m",       "max"),
            humidity_during_snow  =("relative_humidity_2m", "mean"),
            wind_u_during_snow    =("wind_u",               "mean"),
            wind_v_during_snow    =("wind_v",               "mean"),
        )
        cond = snow_only.groupby("date").agg(**snow_agg)
        base = base.join(cond)

        if "wind_direction_10m" in snow_only.columns:
            base["wind_dir_during_snow"] = snow_only.groupby("date")["wind_direction_10m"].apply(
                _vector_mean_direction
            )

    snow_cond_cols = [
        "snow_temp_mean", "wind_during_snow_mean", "wind_during_snow_max",
        "humidity_during_snow", "wind_u_during_snow", "wind_v_during_snow",
    ]
    if "wind_dir_during_snow" not in base.columns:
        base["wind_dir_during_snow"] = np.nan

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


def build_features(hourly: pd.DataFrame, hemisphere: str = "north") -> pd.DataFrame:
    """
    Compute all features on top of daily aggregates.
    hemisphere: "north" (default) or "south" — controls season-year rollup.
    """
    _sy = _season_year_south if hemisphere == "south" else _season_year_north
    d = _agg_daily(hourly)

    # ── Season year — computed once, kept through all season-grouped ops ─────────
    d["_sy_tmp"] = d.index.map(_sy)

    # ── Rolling snow windows (reset per season to avoid cross-season bleed) ──────
    for _col, _win in [
        ("snowfall_48h", 2), ("snowfall_72h", 3), ("snowfall_7d", 7),
        ("snowfall_14d", 14), ("snowfall_last_30d", 30),
    ]:
        d[_col] = (
            d.groupby("_sy_tmp")["snowfall_24h"]
            .transform(lambda s, w=_win: s.rolling(w, min_periods=1).sum())
        )

    # ── Freeze-thaw rolling (also season-bounded) ─────────────────────────────
    d["freeze_thaw_count_7d"] = (
        d.groupby("_sy_tmp")["freeze_thaw_event"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )

    # ── Seasonal cumulative ────────────────────────────────────────────────────
    d["seasonal_snowfall_total"] = d.groupby("_sy_tmp")["snowfall_24h"].cumsum()

    # ── Pressure features (season-bounded to avoid season-start contamination) ──
    if "pressure_mean" in d.columns:
        # 24h tendency: falling pressure = incoming storm (most important sign)
        d["pressure_tendency_24h"] = d.groupby("_sy_tmp")["pressure_mean"].transform(
            lambda s: s.diff(1).fillna(0)
        )
        # 3-day tendency: sustained pattern
        d["pressure_tendency_3d"] = d.groupby("_sy_tmp")["pressure_mean"].transform(
            lambda s: s.diff(3).fillna(0)
        )
        # Anomaly from 7-day rolling mean: deviations from local baseline
        d["pressure_anomaly"] = d.groupby("_sy_tmp")["pressure_mean"].transform(
            lambda s: (s - s.rolling(7, min_periods=1).mean()).fillna(0)
        )
    else:
        d["pressure_tendency_24h"] = 0.0
        d["pressure_tendency_3d"]  = 0.0
        d["pressure_anomaly"]      = 0.0

    # ── 850 hPa temperature ───────────────────────────────────────────────────
    # 850hPa sits at ~1500m asl — close to most resort summit elevations.
    # When temp_850 > 0°C, precipitation is likely rain not snow at resort level.
    # The trend (how fast it's cooling/warming aloft) is a better storm indicator
    # than surface temp because it's unaffected by local terrain effects.
    if "temp_850_mean" in d.columns:
        d["temp_850_trend"]   = d["temp_850_mean"].diff(1).fillna(0)
        d["rain_risk_850"]    = d["temp_850_mean"].clip(-20, 10) / 10
        d["freeze_depth_850"] = (-d["temp_850_mean"]).clip(0)
    else:
        # Use NaN, not 0. 0°C at 850hPa is a real weather condition (marginal
        # snow level); NaN signals genuinely missing data and gets imputed with
        # monthly means in core/dataset.py before training.
        d["temp_850_mean"]    = np.nan
        d["temp_850_min"]     = np.nan
        d["temp_850_max"]     = np.nan
        d["temp_850_trend"]   = np.nan
        d["rain_risk_850"]    = np.nan
        d["freeze_depth_850"] = np.nan

    # ── Dewpoint depression ───────────────────────────────────────────────────
    if "dewpoint_mean" in d.columns:
        # Low depression (temp near dewpoint) = air near saturation = snow likely
        d["dewpoint_depression"] = d["temp_mean"] - d["dewpoint_mean"]
        d["dewpoint_depression_min"] = d["temp_min"] - d["dewpoint_min"]
    else:
        d["dewpoint_depression"]     = np.nan
        d["dewpoint_depression_min"] = np.nan

    # ── Wind direction features ───────────────────────────────────────────────
    # Wind direction sin/cos encode the circular feature for tree models
    if "wind_dir_mean" in d.columns and d["wind_dir_mean"].notna().any():
        d["wind_dir_sin"] = np.sin(np.deg2rad(d["wind_dir_mean"].fillna(0)))
        d["wind_dir_cos"] = np.cos(np.deg2rad(d["wind_dir_mean"].fillna(0)))
        # On no-snow days, fill wind_dir_during_snow with the day's overall wind
        # direction (not 0°=North, which would encode a spurious Siberian-outbreak
        # signal on clear dry days).
        wind_dir_snow_filled = d["wind_dir_during_snow"].fillna(d["wind_dir_mean"].fillna(0))
        d["wind_dir_snow_sin"] = np.sin(np.deg2rad(wind_dir_snow_filled))
        d["wind_dir_snow_cos"] = np.cos(np.deg2rad(wind_dir_snow_filled))
    else:
        d["wind_dir_sin"] = d["wind_dir_cos"] = 0.0
        d["wind_dir_snow_sin"] = d["wind_dir_snow_cos"] = 0.0

    # Moisture flux: u/v wind * humidity — directional moisture transport
    d["moisture_flux_u"] = d["wind_u_mean"] * d["humidity_mean"] / 100
    d["moisture_flux_v"] = d["wind_v_mean"] * d["humidity_mean"] / 100

    # ── Days since last / major snow ──────────────────────────────────────────
    snow_idx  = d.index[d["snowfall_24h"] > 0]
    major_idx = d.index[d["snowfall_24h"] >= 10]

    def _days_since(idx_arr, dt):
        prev = idx_arr[idx_arr <= dt]
        return (dt - prev[-1]).days if len(prev) else np.nan

    d["days_since_last_snow"]  = [_days_since(snow_idx,  dt) for dt in d.index]
    d["days_since_major_snow"] = [_days_since(major_idx, dt) for dt in d.index]

    # ── Snowfall ratios ───────────────────────────────────────────────────────
    d["snowfall_ratio_1d_3d"] = d["snowfall_24h"] / d["snowfall_72h"].replace(0, np.nan)
    d["snowfall_ratio_3d_7d"] = d["snowfall_72h"] / d["snowfall_7d"].replace(0, np.nan)

    # ── Wind loading (capped) ─────────────────────────────────────────────────
    d["wind_loading_index"] = (d["wind_max"] * d["snowfall_24h"]).clip(upper=100)

    # ── Time since storm peak (season-bounded) ────────────────────────────────
    # Look back up to 6 days but stop at season boundary so the previous
    # season's big storm doesn't bleed into opening week of the new season.
    _sy_arr = d["_sy_tmp"].values
    tsp = []
    for i, dt in enumerate(d.index):
        cur_sy = _sy_arr[i]
        lookback_start = i
        for j in range(i - 1, max(-1, i - 7), -1):
            if _sy_arr[j] == cur_sy:
                lookback_start = j
            else:
                break
        window = d.iloc[lookback_start: i + 1]["snowfall_24h"]
        tsp.append((dt - window.idxmax()).days)
    d["time_since_storm_peak"] = tsp

    d["radiation_peak"] = d["radiation_peak"].clip(upper=600)

    # ── Critical derived features ─────────────────────────────────────────────
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

    # ── Cold air advection index (hemisphere-aware) ───────────────────────────
    # NH (Japan): Siberian outbreak = rising pressure + cold temps + NW wind.
    # SH (AU/NZ/Andes): Antarctic front = rising pressure + cold temps + S/SW wind.
    pressure_rise = d["pressure_tendency_24h"].clip(lower=0)
    cold_temp     = (-d["temp_min"]).clip(lower=0)
    if hemisphere == "north":
        # NW component: negative wind_u means flow from the west (Siberian outbreak)
        cold_wind = (-d["wind_u_during_snow"]).clip(lower=0)
    else:
        # Southerly component: negative wind_v means flow from the south (Antarctic front)
        cold_wind = (-d["wind_v_during_snow"]).clip(lower=0)
    d["cold_air_advection"] = pressure_rise * cold_temp * cold_wind

    # ── Temporal / trend ──────────────────────────────────────────────────────
    d["wind_after_storm"] = d["wind_mean"] * d["time_since_storm_peak"]
    d["temp_trend"]       = d["temp_mean"].diff().fillna(0)

    # ── Calendar ──────────────────────────────────────────────────────────────
    d["day_of_year"]    = d.index.day_of_year
    d["month"]          = d.index.month
    d["season_year"]    = d["_sy_tmp"]
    d["days_in_season"] = d.groupby("_sy_tmp").cumcount()

    d.drop(columns=["_sy_tmp"], inplace=True)
    return d
