# hindcast_australia.py

**Purpose:** Runs a retrospective forecast evaluation for Australian SH resorts over historical weather data, producing day-by-day predictions comparable to observed snowfall.

**Inputs:**
- `data/models/xgb_overnight_snow.pkl` — trained model payload
- `data/models/transfer_calibration.json` — SH calibration parameters
- `data/raw/weather/{resort_id}_*.csv` — historical weather CSVs for AU resorts
- `data/processed/sh_labels.csv` — ERA5 snow_depth-change labels (for comparison)

**Outputs:**
- Printed hindcast table: date, resort, predicted cm, observed cm (ERA5 proxy), powder flag
- Optional: `data/plots/hindcast_australia.png` if `--plot` flag used

**Key parameters / constants:**
- Australian resorts: thredbo, perisher, falls_creek (filtered from regions.yaml by `region == "nsw"` or `region == "victoria"`)

**Notes:**
- Useful for manual QA after retraining — if the model is drastically over- or under-predicting for known big powder days in Australia, this is the first check.
- ERA5 snow_depth labels are an imperfect proxy; large events are often underrepresented due to ERA5's coarse grid resolution vs actual resort elevation.

**Last updated:** 2026-06-07
