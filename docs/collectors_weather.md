# collectors/weather.py

**Purpose:** Downloads and caches historical hourly weather data from the Open-Meteo archive API for a given resort's lat/lon and date range.

**Inputs:**
- Resort config dict with `lat`, `lon`, `elevation`
- Date range: `start` / `end` strings (`YYYY-MM-DD`)
- Open-Meteo archive API (live HTTP) or historical-forecast API for 850hPa fields (available from 2021-03-23 only)
- Local cache: `data/raw/weather/{resort_id}_{start}_{end}.csv`

**Outputs:**
- Hourly `pd.DataFrame` cached to `data/raw/weather/{resort_id}_{start}_{end}.csv`
- Returns the DataFrame from cache on subsequent calls (no re-fetch unless cache is absent or stale)

**Key parameters / constants:**
- `RAW_DIR = Path("data/raw/weather")`
- `PRESSURE_LEVEL_START = "2021-03-23"` — earliest date with 850hPa data from the historical-forecast API
- `HOURLY_VARS` — surface-level variables, available from the archive API for the full history
- `PRESSURE_LEVEL_VARS = ["temperature_850hPa"]` — fetched only when start ≥ 2021-03-23 or `forecast=True`

**Notes:**
- **Auto-selecting URL and variables**: `fetch_hourly()` picks the correct API endpoint based on the date range. `forecast=True` → `FORECAST_URL` + 850hPa. `start ≥ 2021-03-23` → `HIST_FORECAST_URL` + 850hPa. Earlier → `ARCHIVE_URL`, surface only. This means the 35-day warmup fetch in `forecast.py` (always post-2021) automatically includes 850hPa.
- 850hPa data is only available from 2021-03-23; pre-2021 rows have NaN, which XGBoost handles natively.
- `scripts/refresh_weather.py` can detect and re-download stale CSVs that predate the 850hPa / dewpoint feature additions.

**Last updated:** 2026-06-07
