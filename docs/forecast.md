# forecast.py

**Purpose:** Generates 7-day (or N-day) powder forecasts for all configured resorts by fetching live Open-Meteo NWP data, running the trained model, and emitting JSON results.

**Inputs:**
- `data/models/xgb_overnight_snow.pkl` — trained model payload (model, feature_cols, amp_per_resort, japan_correction_iso, powder_pred_threshold)
- `data/models/transfer_calibration.json` — SH isotonic calibration parameters
- `data/models/nwp_direct_models.pkl` — multi-feature XGBoost models for AU/NZ NWP-direct resorts
- `regions.yaml` — resort config (lat, lon, elevation, hemisphere, region)
- Open-Meteo forecast API (live HTTP) — 7-day hourly weather for each resort's lat/lon

**Outputs:**
- JSON array written to stdout (or captured by caller), one record per resort per day:
  `{resort_id, date, predicted_snow_cm, powder_score, condition, temp_min, wind_max, ...}`
- `condition` field: one of `powder`, `wind_affected`, `natural`, `ice`, `slush`, `spring`, `packed`, `groomed`, `variable` — derived by `core/conditions.py`

**Key parameters / constants:**
- `POWDER_CM = 15.0` — Japan/NH powder threshold (observed cm); used for powder_score scaling
- `powder_score(snow_cm, temp, wind)` — 0–100 score combining predicted snowfall, temperature quality, and wind penalty
- Physical gate: `predicted_snow_cm` zeroed when `temp_min > 2°C`
- `amp_per_resort` — loaded from model payload; applied to `snowfall_24h`/`snowfall_48h` before feature computation for all resorts (not just SH-calibrated ones)

**Notes:**
- **Rolling window warmup**: `forecast_resort()` now fetches 35 days of historical weather (archive API) before the forecast start date, concatenates with the 7-day forward forecast, builds features on the combined data, then slices to the forecast period for output. This ensures `snowfall_72h`, `snowfall_7d`, `snowfall_14d`, `snowfall_last_30d`, `days_since_last_snow` have valid history on day 1 of the forecast. Without warmup, all rolling windows were garbage (only 7 days of data).
- **850hPa NaN handling** now matches `train.py`'s `_prep_X`: 850hPa columns are left as NaN so XGBoost routes them correctly. Previously `fillna(0)` encoded "0°C at 850hPa" (marginal snow level) for all forecast days.
- `nwp_amplification` is stored in the model payload (training-only values) and applied unconditionally at inference.
- **Three inference paths:**
  1. **AU/NZ NWP-direct**: `nwp_direct_models[resort_id].predict(weather_features)` → local ERA5 cm directly. No isotonic step — XGBoost IS the calibration. Falls back to 1D isotonic on `snowfall_48h` if pkl missing.
  2. **Andes japan_model**: Japan model → classifier gate → Japan post-hoc correction → Andes isotonic calibration → local ERA5 cm.
  3. **Japan NH**: Japan model → classifier gate → Japan post-hoc correction → local Japan predicted cm.
- **Japan post-hoc correction**: `japan_correction_iso` (loaded from model payload) is applied after the classifier gate on both Japan NH and Andes paths. Monotonically stretches the compressed model output range to improve big-event detection.
- **Powder thresholds**: SH resorts use `powder_threshold_local_cm` from calibration JSON (e.g. 4cm ERA5); Japan NH resorts use `powder_pred_threshold` from model payload (F1-optimal on corrected predictions, ~8.8cm).
- **Classifier gate at inference**: `load_model()` returns an 8-tuple including `japan_correction_iso` and `powder_pred_threshold`. When `snow_classifier` is present, regressor predictions are multiplied by `(P(snow) >= snow_gate_threshold)` before the Japan correction.
- The NWP-direct path for AU/NZ bypasses the classifier gate — only the Japan model path uses it.
- **NWP trace floor**: for the NWP-direct 1D fallback path, raw `snowfall_48h` values below 0.3cm are zeroed.
- **Hidden resorts** (`hidden: true` in `regions.yaml`): excluded from default forecast runs. Currently only `niseko_annupuri`. Requesting `--resort niseko_annupuri` explicitly still works.
- **Season gating** (`opens_month`/`closes_month` in `regions.yaml`): terminal output for off-season resorts is replaced with a brief "Off-season (opens <month>)" line. Forecasts are still computed and included in `--json` output. Uses month-level granularity; early-in-month edge cases may still show for the opening month.
- `condition` column: rule-based surface condition estimate via `core/conditions.estimate_condition()`. Will be replaced by ML classifier once labeled snapshot data accumulates.

**Last updated:** 2026-06-08 (Japan correction + AU/NZ multi-feature XGBoost + powder threshold fix)
