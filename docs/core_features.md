# core/features.py

**Purpose:** Converts raw hourly Open-Meteo weather data into a daily feature DataFrame suitable for model training and inference.

**Inputs:**
- `hourly` — `pd.DataFrame` with hourly rows indexed by datetime, containing columns: `temperature_2m`, `dewpoint_2m`, `snowfall`, `precipitation`, `wind_speed_10m`, `wind_direction_10m`, `relative_humidity_2m`, `shortwave_radiation`, `cloud_cover`, `pressure_msl`, `surface_pressure`, `temperature_850hPa`
- `hemisphere` — `"north"` or `"south"` (controls season_year boundary and cold_air_advection direction)

**Outputs:**
- Daily `pd.DataFrame` indexed by date with ~30 engineered feature columns including rolling snowfall windows, thermodynamic indices, and season metadata

**Key parameters / constants:**
- Rolling windows: `snowfall_48h` (2d), `snowfall_72h` (3d), `snowfall_7d` (7d), `snowfall_14d` (14d), `snowfall_last_30d` (30d), `freeze_thaw_count_7d`
- `snow_quality_index` = mean temp × wind speed during snowing hours (proxy for powder quality)
- `dewpoint_depression` = temp_2m − dewpoint_2m (lower → drier air → lighter snow)
- `cold_air_advection` = pressure_rise × cold_temperature_anomaly × cold_wind_component
  - NH: uses `-wind_u` (westerly/NW Siberian flow); SH: uses `-wind_v` (southerly Antarctic flow)

**Notes:**
- `_sy_tmp` is computed once at the start of `build_features` and reused for all season-bounded operations: rolling windows, pressure features, seasonal cumulative, `time_since_storm_peak`. It is dropped before returning.
- **Pressure features** (`pressure_tendency_24h/3d`, `pressure_anomaly`) are now season-grouped via `groupby("_sy_tmp")`. Previously these used global `diff()`/`rolling()` which bled the previous season's pressure trend into the first few days of a new season.
- **`wind_dir_snow_sin/cos`** on no-snow days now fills `wind_dir_during_snow` with `wind_dir_mean` (the day's overall wind direction) instead of 0° (North). 0° was encoding a spurious Siberian-outbreak signal on dry clear days.
- **`time_since_storm_peak`** now respects season boundaries — the 6-day lookback window stops at the start of the current season, so last season's storm peak doesn't bleed into early-season rows.
- `season_year` NH: `year if month >= 10 else year - 1` (Oct–Sep window); SH: `year if month >= 4 else year - 1` (Apr–Mar window)
- 850hPa features (`temperature_850hPa`) are NaN before 2021-03-23 (historical-forecast API limit). XGBoost handles these natively via its NaN routing — do not impute.
- Lag features (`overnight_snow_lag1`, etc.) are **not produced here** — they are added in `core/dataset.py` and intentionally excluded from model `FEATURE_COLS` (unavailable at inference).

**Last updated:** 2026-06-07
