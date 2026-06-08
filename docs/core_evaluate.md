# core/evaluate.py

**Purpose:** Runs evaluation on the trained model — season holdout metrics, hemisphere-stratified metrics, and leave-one-resort-out (LORO) generalization.

**Inputs:**
- `data/processed/training_dataset.parquet` — training dataset
- `data/models/xgb_overnight_snow.pkl` — trained model payload
- CLI args: `--holdout` (season boundary), `--dataset`, `--loro` (flag to run LORO)

**Outputs:**
- Printed metric tables: overall test metrics, per-hemisphere breakdown, LORO resort table
- No files written (results are stdout only)

**Key parameters / constants:**
- `POWDER_CM = 15.0` — Japan powder threshold in observed cm
- `POWDER_THRESHOLDS_BY_HEMI = {"north": 15.0, "south": 4.0}` — hemisphere-specific thresholds applied during per-hemisphere evaluation
- Physical gate: predictions zeroed when `temp_min > 2°C` (applied consistently with `forecast.py`)

**Notes:**
- **Japan correction applied**: `_predict()` now accepts `japan_correction` (dict with `iso_x`/`iso_y`). If present, the isotonic correction is applied after the classifier gate. Loaded from `payload["japan_correction_iso"]`. This matches the exact inference path used in `forecast.py`.
- **Classifier gate applied**: `_predict()` accepts `snow_classifier` and `snow_gate_threshold` from the model payload. If present, predictions are multiplied by `(P(snow) >= threshold)` before the Japan correction.  This matches the exact inference path used in `forecast.py`.
- **`_prep_X` matches `train.py`'s NaN handling**: 850hPa columns are left as NaN so XGBoost routes pre-2021 rows via its internal missing-value mechanism.
- **Per-resort table uses hemisphere-appropriate powder threshold**: NH resorts use 15cm, SH resorts use 4cm. `_metrics_row()` accepts `actual_powder_cm`.
- **LORO threshold is derived from training data only** via `_find_best_threshold(y_train, p_train)` on Japan rows. The old approach peeked at test labels.
- LORO uses the same XGBoost Tweedie objective as production.
- Hemisphere-split evaluation is important because SH powder threshold differs (4cm vs 15cm).
- `_metrics_row()` returns an empty dict for <5 rows.

**Last updated:** 2026-06-08 (Japan correction applied in _predict)
