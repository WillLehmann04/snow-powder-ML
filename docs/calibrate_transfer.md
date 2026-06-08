# calibrate_transfer.py

**Purpose:** Fits per-resort calibrations mapping Japan model output (or NWP snowfall) to local SH snowfall, and computes the powder detection threshold for each SH resort.

**Inputs:**
- `data/models/xgb_overnight_snow.pkl` — trained Japan model (also provides `japan_correction_iso`)
- `data/raw/weather/{resort_id}_*.csv` — historical hourly weather per SH resort
- `data/processed/sh_labels.csv` — ERA5 snow_depth-change labels (proxy observed snowfall)
- `regions.yaml` — resort config, including region (determines calibration method) and powder threshold

**Outputs:**
- `data/models/transfer_calibration.json` — calibration dict keyed by resort_id (isotonic knots + metadata)
- `data/models/nwp_direct_models.pkl` — dict of `{resort_id: XGBRegressor}` for AU/NZ multi-feature calibration
- `data/plots/calibration_{resort_id}.png` — scatter plot of raw vs calibrated predictions for each resort

**Key parameters / constants:**
- Two calibration methods:
  - `nwp_direct` (AU/NZ): fits a multi-feature XGBRegressor on `[snowfall_48h, snowfall_72h, temp_min, hours_with_snow, precipitation, pressure_tendency_24h]` → local ERA5 snow_cm. Delivers r=0.90–0.97 vs 0.35–0.74 for 1D isotonic. A 1D isotonic on `snowfall_48h` is also stored in the JSON as a fallback if the pkl is missing.
  - `japan_model` (Andes): applies the Japan correction from the payload first, then fits isotonic on corrected model output → ERA5 proxy.
- `NWP_DIRECT_FEATURES` — the 6 features used by the NWP-direct XGBoost model
- Powder threshold in `RESORTS_POWDER_THRESHOLD` dict (per-resort, in local ERA5 cm) is stored in the JSON

**Notes:**
- **Japan post-hoc correction**: `run_japan_model()` loads `japan_correction_iso` from the model payload and applies it to raw model predictions before storing `japan_model_raw`. This ensures the SH calibration is fitted on the same corrected prediction scale that `forecast.py` produces at inference time.
- **Multi-feature XGBoost for NWP-direct**: `fit_calibration(nwp_direct=True)` fits both the XGBoost (stored in pkl) and a 1D isotonic (stored in JSON as fallback). The XGBoost captures rain/snow discrimination via `temp_min` and event character via `hours_with_snow`.
- **Calibration zero anchor**: `fit_calibration()` prepends `(0, 0)` before the isotonic fit to prevent non-zero dry-day output.
- **LivePass label priority**: `run_nwp_direct()` first checks `data/raw/snow_reports/{resort_id}.csv` for observed `snow_24h_cm`. If ≥30 valid rows exist, those replace ERA5. Once a full season accumulates, calibration quality will improve significantly.
- South American resorts use `japan_model` method; AU/NZ use `nwp_direct`. Whistler is NH and does not go through this pipeline.

**Last updated:** 2026-06-08 (multi-feature XGBoost AU/NZ + Japan correction)
