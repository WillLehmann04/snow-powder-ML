# core/train.py

**Purpose:** Trains the XGBoost Tweedie regression model that predicts overnight snowfall in cm, and saves a versioned model payload to disk.

**Inputs:**
- `data/processed/training_dataset.parquet` — joined label + feature dataset built by `core/dataset.py`
- CLI args: `--holdout` (season string, e.g. `2022-2023`), `--dataset` (path override)

**Outputs:**
- `data/models/xgb_overnight_snow_{timestamp}.pkl` — timestamped versioned artifact
- `data/models/xgb_overnight_snow.pkl` — symlink/copy pointing to the latest artifact

Payload keys:
- `model` — fitted XGBRegressor
- `snow_classifier` — fitted XGBClassifier for binary snow/no-snow gate
- `snow_gate_threshold` — probability threshold for the gate (default 0.20)
- `feature_cols` — list of features the model expects (no lag features)
- `holdout_season` — which season was held out for test evaluation
- `train_metrics` / `test_metrics` — dicts with MAE, r, powder F1/precision/recall
- `log_transform` — always `False` (Tweedie handles right-skew natively)
- `powder_pred_threshold` — F1-optimal prediction threshold in **corrected** cm (Japan powder days, post-correction scale)
- `nwp_amplification_per_resort` — dict of `{resort_id: amplification_factor}` for inference-time use
- `japan_correction_iso` — `{"iso_x": [...], "iso_y": [...]}` — monotonic post-hoc correction knots for stretching compressed Japan model output
- `trained_at` — ISO-8601 UTC timestamp of training run

**Key parameters / constants:**
- `HOLDOUT_SEASON = "2022-2023"` — default test boundary (train on earlier, test on later)
- `FEATURE_COLS` — curated feature list; includes `maritime_influence = 100 / (dist_coast_km + 10)`; excludes all lag features (unavailable at inference)
- XGBoost regressor: `objective="reg:tweedie"`, `tweedie_variance_power=1.2`, `n_estimators=800` (max), `learning_rate=0.04`, `max_depth=5`, `min_child_weight=7`, `early_stopping_rounds=30`
- XGBoost classifier: `n_estimators=300`, `max_depth=4`, `scale_pos_weight` auto-computed from class imbalance, `objective="binary:logistic"`
- `SNOW_GATE_THRESHOLD = 0.20` — regressor predictions are zeroed when `P(snow) < 0.20`
- Physical gate: predictions zeroed when `temp_min > 2°C` (applied in forecast.py after calibration)

**Notes:**
- **Two-stage inference**: A binary XGBClassifier is trained first to predict P(snow today). At inference, regressor outputs are multiplied by `(p_snow >= 0.20)`. This reduces false-positive powder alerts on clear-but-cold days without needing to retrain the regressor.
- **Post-hoc Japan isotonic correction (`japan_correction_iso`)**: Fitted on Japan NH training predictions vs actual observed labels after the classifier gate is applied. Monotonically stretches the compressed output range so big events (>20cm actual) are better captured. Applied to all Japan model predictions at inference, including Andes resorts before SH calibration. The SH Andes calibration is re-fitted on corrected predictions to keep the pipeline consistent.
- **Sample weighting (2.0× for ≥15cm days)**: AU/NZ label-scale dilution is the primary fix (those 9 resorts excluded from training). The 2× weight additionally reduces regression-to-mean on extreme Japan events. Previous 6×/3× attempts degraded precision 13pp; 2× with cleaner training data is the calibrated compromise.
- **Test set is 4 seasons** (all seasons ≥ 2022-2023) — more reliable than the prior single-season holdout.
- Season holdout (never random split) is critical — adjacent days are nearly identical and would leak information across a random split.
- **Early stopping uses an internal validation split** (last training season, e.g. "2021-2022"), not the blind test holdout. After finding `best_iteration`, the final model is retrained on ALL training data with `n_estimators = best_iteration + 1`. This prevents test-set influence on model complexity.
- **NWP amplification is recomputed from training rows only** — `(mean observed / mean NWP) per resort on snowy days` — overriding the full-dataset values from dataset.py to avoid test-label contamination.
- Lag features were removed from `FEATURE_COLS` (they remain in the parquet). At inference, snowpack history is unknown — filling with 0 creates a systematic distribution mismatch.
- `nwp_amplification_per_resort` stored in payload = training-only amplification factors.
- Versioned saves prevent accidental overwrites during iterative retraining.

**Last updated:** 2026-06-08 (post-hoc correction added)
