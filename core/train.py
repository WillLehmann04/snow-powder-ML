"""
Train the XGBoost overnight-snowfall regression model.

Key design decisions:
  - Target: log1p(overnight_snow_cm) — reduces right-skew, forces the model to
    pay attention to big powder days instead of regressing to the mean.
    Predictions are expm1'd back to cm at inference time.
  - Split: season holdout — train on all seasons before HOLDOUT_SEASON, test on the rest.
    Never random split (adjacent days are nearly identical → data leakage).
  - Complexity: deliberately modest (n_estimators=300, max_depth=4) to close the
    train/test gap observed with the previous 600-tree depth-6 model.
  - Physical gate at inference: zero out predictions when temp_min > 2C (can't snow
    if it's not cold enough). Applied in evaluate.py and forecast.py.

Output: data/models/xgb_overnight_snow.pkl
  Contains: {"model", "feature_cols", "train_metrics", "test_metrics", "holdout_season"}

Usage:
  python -m core.train
  python -m core.train --holdout 2023-2024   # use different holdout boundary
  python -m core.train --dataset data/processed/training_dataset.parquet
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier, XGBRegressor

DATASET_PATH = Path("data/processed/training_dataset.parquet")
MODEL_PATH   = Path("data/models/xgb_overnight_snow.pkl")

# Seasons >= this go into the test set
HOLDOUT_SEASON = "2022-2023"

# Features: all engineered weather columns + static resort columns.
# Excludes identifiers (resort_id, date, season) and both target columns.
FEATURE_COLS = [
    # ── Open-Meteo daily aggregates ───────────────────────────────────────
    "snowfall_24h", "precipitation", "hours_with_snow", "max_snowfall_1h",
    "temp_min", "temp_max", "temp_mean",
    "wind_max", "wind_mean", "wind_u_mean", "wind_v_mean",
    "humidity_mean", "radiation_mean", "radiation_peak", "cloud_cover_mean",
    # ── Wind direction (encoded as sin/cos — circular quantity) ───────────
    "wind_dir_sin", "wind_dir_cos",
    "wind_dir_snow_sin", "wind_dir_snow_cos",
    "wind_dir_consistency",
    # ── Moisture flux (wind × humidity) ───────────────────────────────────
    "moisture_flux_u", "moisture_flux_v",
    # ── Pressure ──────────────────────────────────────────────────────────
    "pressure_mean",
    "pressure_tendency_24h", "pressure_tendency_3d", "pressure_anomaly",
    # ── 850 hPa temperature (real data from 2021-03-23, XGBoost NaN routing pre-2021)
    "temp_850_mean", "temp_850_min", "temp_850_max",
    "temp_850_trend", "rain_risk_850", "freeze_depth_850",
    # ── Dewpoint depression ────────────────────────────────────────────────
    "dewpoint_depression", "dewpoint_depression_min",
    # ── Snow-conditional aggregates ────────────────────────────────────────
    "snow_temp_mean", "wind_during_snow_mean", "wind_during_snow_max",
    "humidity_during_snow", "wind_u_during_snow", "wind_v_during_snow",
    # ── Derived daily ──────────────────────────────────────────────────────
    "clear_sky_fraction", "freeze_thaw_event", "storm_duration_hours",
    # ── Rolling windows ────────────────────────────────────────────────────
    "snowfall_48h", "snowfall_72h", "snowfall_7d", "snowfall_14d",
    "snowfall_last_30d", "seasonal_snowfall_total",
    # ── Time-since features ────────────────────────────────────────────────
    "days_since_last_snow", "days_since_major_snow",
    # ── Ratio / index features ─────────────────────────────────────────────
    "snowfall_ratio_1d_3d", "snowfall_ratio_3d_7d",
    "wind_loading_index", "time_since_storm_peak",
    # ── Critical derived ───────────────────────────────────────────────────
    "snow_quality_index", "powder_freshness", "melt_risk", "wind_damage_score",
    "wind_after_storm", "temp_trend",
    # ── Calendar ───────────────────────────────────────────────────────────
    "day_of_year", "month", "days_in_season",
    # ── NWP bias correction ───────────────────────────────────────────────
    "nwp_amplification",
    "amplified_snowfall_24h",  # snowfall_24h × nwp_amplification
    "amplified_snowfall_48h",  # snowfall_48h × nwp_amplification
    # ── Static resort ──────────────────────────────────────────────────────
    "elevation", "lat", "lon", "region_code",
    # Maritime influence = 100 / (dist_coast_km + 10); higher = more ocean moisture
    "maritime_influence",
]

TARGET = "overnight_snow_cm"


_HPA_COLS = {"temp_850_mean", "temp_850_min", "temp_850_max",
             "temp_850_trend", "rain_risk_850", "freeze_depth_850"}


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.reindex(columns=FEATURE_COLS, fill_value=0.0)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Non-850hPa NaN → 0; 850hPa NaN stays NaN so XGBoost routes them natively.
    non_hpa = [c for c in X.columns if c not in _HPA_COLS]
    X[non_hpa] = X[non_hpa].fillna(0.0)
    return X


def _find_best_threshold(y_true: np.ndarray, y_pred: np.ndarray,
                          powder_cm: float = 15.0) -> float:
    """
    Sweep prediction thresholds and return the one that maximises F1 for
    powder-day detection (actual >= powder_cm). Evaluated on whatever data
    is passed in — caller is responsible for using training data only.
    """
    actual_powder = y_true >= powder_cm
    if actual_powder.sum() < 5:
        return powder_cm * 0.5

    candidates = np.percentile(y_pred, np.arange(50, 99, 2))
    best_f1, best_thr = 0.0, candidates[0]
    for thr in candidates:
        pred_p = y_pred >= thr
        tp = (actual_powder & pred_p).sum()
        fp = (~actual_powder & pred_p).sum()
        fn = (actual_powder & ~pred_p).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray,
             pred_threshold: float, actual_threshold: float = 15.0) -> dict:
    """
    Compute MAE, RMSE, Pearson r, and powder-day detection stats.
    pred_threshold must be computed from the training set (not the split being evaluated).
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r, _ = pearsonr(y_pred, y_true)

    snow_mask = y_true >= 1.0
    if snow_mask.sum() > 10:
        r_snow, _ = pearsonr(y_pred[snow_mask], y_true[snow_mask])
        mae_snow  = mean_absolute_error(y_true[snow_mask], y_pred[snow_mask])
    else:
        r_snow = mae_snow = float("nan")

    actual_powder = y_true >= actual_threshold
    pred_powder   = y_pred >= pred_threshold
    tp = int((actual_powder & pred_powder).sum())
    fp = int((~actual_powder & pred_powder).sum())
    fn = int((actual_powder & ~pred_powder).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "n":               int(len(y_true)),
        "mae":             round(mae, 3),
        "rmse":            round(rmse, 3),
        "r":               round(r, 4),
        "n_snow_days":     int(snow_mask.sum()),
        "r_snow_days":     round(r_snow, 4),
        "mae_snow_days":   round(mae_snow, 3),
        "powder_threshold_actual":    actual_threshold,
        "powder_threshold_predicted": round(pred_threshold, 2),
        "powder_precision": round(prec, 4),
        "powder_recall":    round(rec, 4),
        "powder_f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def train(
    dataset_path: Path = DATASET_PATH,
    model_path:   Path = MODEL_PATH,
    holdout_season: str = HOLDOUT_SEASON,
) -> dict:

    print("Loading dataset ...")
    df = pd.read_parquet(dataset_path)
    print(f"  {len(df):,} rows, {df['resort_id'].nunique()} resorts")

    # ── Split ────────────────────────────────────────────────────────────────
    train_df = df[df["season"] <  holdout_season].copy()
    test_df  = df[df["season"] >= holdout_season].copy()
    print(f"  Train: {len(train_df):,} rows ({train_df['season'].nunique()} seasons: "
          f"{sorted(train_df['season'].unique())[0]} – {sorted(train_df['season'].unique())[-1]})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['season'].nunique()} seasons: "
          f"{sorted(test_df['season'].unique())[0]} – {sorted(test_df['season'].unique())[-1]})")

    # ── Recompute NWP amplification from training rows only ──────────────────
    # dataset.py computes amplification on all rows; here we recompute on
    # training rows only so test-set labels don't contaminate the feature.
    train_df = train_df.copy()
    test_df  = test_df.copy()
    snowy_train = train_df[(train_df[TARGET] > 0) & (train_df["snowfall_24h"] > 0)]
    amp_train = (
        snowy_train.groupby("resort_id")
        .apply(lambda g: g[TARGET].mean() / g["snowfall_24h"].mean(), include_groups=False)
        .clip(upper=6.0)
        .fillna(1.0)
    )
    amp_per_resort = amp_train.to_dict()
    for part in (train_df, test_df):
        part["nwp_amplification"]       = part["resort_id"].map(amp_per_resort).fillna(1.0)
        part["amplified_snowfall_24h"]  = part["snowfall_24h"] * part["nwp_amplification"]
        part["amplified_snowfall_48h"]  = part["snowfall_48h"] * part["nwp_amplification"]
    print("  NWP amplification recomputed from training rows only")

    X_train = _prep_X(train_df)
    X_test  = _prep_X(test_df)
    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    # ── Stage 1: Binary snow classifier ─────────────────────────────────────
    # Independently learns P(any snow today). Used as a gate at inference:
    # predictions are zeroed when P(snow) < SNOW_GATE_THRESHOLD, reducing
    # false-positive powder alerts on clear-but-cold days.
    SNOW_GATE_THRESHOLD = 0.20
    snow_labels = (y_train >= 0.5).astype(int)
    pos_weight  = float((snow_labels == 0).sum()) / max(1, (snow_labels == 1).sum())
    snow_classifier = XGBClassifier(
        n_estimators       = 300,
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = pos_weight,
        objective          = "binary:logistic",
        eval_metric        = "logloss",
        random_state       = 42,
        n_jobs             = -1,
    )
    snow_classifier.fit(X_train, snow_labels, verbose=False)
    p_snow_train = snow_classifier.predict_proba(X_train)[:, 1]
    gate_accuracy = ((p_snow_train >= SNOW_GATE_THRESHOLD) == snow_labels.astype(bool)).mean()
    print(f"  Snow classifier gate accuracy (P≥{SNOW_GATE_THRESHOLD}): {gate_accuracy:.3f}")

    # ── Model — Tweedie distribution ─────────────────────────────────────────
    # variance_power=1.2: close to Poisson, well-calibrated for Japan's skewed
    # snowfall distribution. VP=1.5 early-stopped at only 279 trees (-0.11 Japan r).
    #
    # 2.0x sample weight for actual powder days (>=15cm).
    # Audit finding: model predicts only 10-18cm on actual 20-50cm events (regression
    # to mean). AU/NZ label scale mismatch (3x smaller) is addressed by excluding
    # those resorts from training; this gentle upweighting further corrects for
    # the large mass of 0-5cm days drowning out the powder-day gradient signal.
    # Previous attempt at 6x/3x degraded precision — 2x is the calibrated compromise.
    sample_weight = np.where(y_train >= 15.0, 2.0, 1.0).astype(float)

    VARIANCE_POWER = 1.2
    _xgb_kwargs = dict(
        max_depth             = 5,
        learning_rate         = 0.04,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 7,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        objective             = "reg:tweedie",
        tweedie_variance_power= VARIANCE_POWER,
        eval_metric           = f"tweedie-nloglik@{VARIANCE_POWER}",
        random_state          = 42,
        n_jobs                = -1,
    )

    # ── Early stopping on internal validation split ───────────────────────────
    # Use the last training season as the validation set to find the optimal
    # n_estimators. The blind test holdout is never seen during this search.
    val_season = sorted(train_df["season"].unique())[-1]
    val_mask   = train_df["season"] == val_season
    X_tr2 = _prep_X(train_df[~val_mask])
    y_tr2 = y_train[~val_mask.values]
    X_val = _prep_X(train_df[val_mask])
    y_val = y_train[val_mask.values]
    print(f"\nEarly stopping validation: season {val_season} ({val_mask.sum()} rows)")

    sw_tr2 = sample_weight[~val_mask.values]
    early_model = XGBRegressor(n_estimators=800, early_stopping_rounds=30, **_xgb_kwargs)
    early_model.fit(X_tr2, y_tr2, sample_weight=sw_tr2, eval_set=[(X_val, y_val)], verbose=50)
    best_n = early_model.best_iteration + 1
    print(f"  Best iteration: {best_n}")

    # Retrain on ALL training data with the fixed n_estimators found above.
    # This uses every training row for the final model weights.
    print(f"\nRetraining on full training set with n_estimators={best_n} ...")
    model = XGBRegressor(n_estimators=best_n, **_xgb_kwargs)
    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

    # Tweedie outputs are already in cm (positive, via log-link internally).
    # Apply classifier gate: zero predictions when P(snow) < threshold.
    p_train_raw = model.predict(X_train).clip(0)
    p_test_raw  = model.predict(X_test).clip(0)

    p_snow_train = snow_classifier.predict_proba(X_train)[:, 1]
    p_snow_test  = snow_classifier.predict_proba(X_test)[:, 1]
    p_train = p_train_raw * (p_snow_train >= SNOW_GATE_THRESHOLD)
    p_test  = p_test_raw  * (p_snow_test  >= SNOW_GATE_THRESHOLD)

    japan_mask = (train_df["hemisphere"] == "north").values if "hemisphere" in train_df.columns \
                 else np.ones(len(y_train), dtype=bool)

    # ── Post-hoc Japan isotonic correction ────────────────────────────────────
    # XGBoost Tweedie regresses toward the mean — rare extreme events (>20cm) are
    # systematically underpredicted because the loss is dominated by the 0-5cm mass.
    # This monotonic correction is fitted on Japan NH training rows and stretches
    # the compressed output range back to the real observed distribution.
    # SH Andes calibration is re-fitted on the corrected output so inference is consistent.
    jp_iso_corr = IsotonicRegression(out_of_bounds="clip", increasing=True)
    jp_iso_corr.fit(p_train[japan_mask], y_train[japan_mask])
    japan_correction = {
        "iso_x": jp_iso_corr.X_thresholds_.tolist(),
        "iso_y": jp_iso_corr.y_thresholds_.tolist(),
    }
    p_train = np.interp(p_train, jp_iso_corr.X_thresholds_, jp_iso_corr.y_thresholds_).clip(0)
    p_test  = np.interp(p_test,  jp_iso_corr.X_thresholds_, jp_iso_corr.y_thresholds_).clip(0)
    print(f"\n  Japan post-hoc correction fitted on {int(japan_mask.sum())} NH training rows")

    # Threshold calibrated on Japan training rows only (post-correction).
    # SH resorts use ERA5 snow_depth labels which rarely exceed 15cm (their actual
    # powder threshold is 4-5cm, not 15cm), so including them in threshold
    # optimisation pulls the cut-off too high and hurts Japan recall.
    powder_pred_thresh = _find_best_threshold(y_train[japan_mask], p_train[japan_mask])
    print(f"\n  Powder detection threshold (post-correction, F1-optimal on Japan train): {powder_pred_thresh:.2f}cm")

    print("\n--- TRAIN metrics ---")
    train_metrics = _metrics(y_train, p_train, powder_pred_thresh)
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- TEST metrics (season holdout) ---")
    test_metrics = _metrics(y_test, p_test, powder_pred_thresh)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # ── Per-resort NWP amplification for inference ───────────────────────────
    # Use the training-only values computed above (not the full-dataset version
    # from dataset.py, which was overridden for correctness earlier in this function).
    # amp_per_resort is already set above.

    # ── Save ─────────────────────────────────────────────────────────────────
    import shutil
    from datetime import datetime

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":                        model,
        "snow_classifier":              snow_classifier,
        "snow_gate_threshold":          SNOW_GATE_THRESHOLD,
        "feature_cols":                 FEATURE_COLS,
        "holdout_season":               holdout_season,
        "train_metrics":                train_metrics,
        "test_metrics":                 test_metrics,
        "log_transform":                False,
        "powder_pred_threshold":        powder_pred_thresh,
        "nwp_amplification_per_resort": amp_per_resort,
        "japan_correction_iso":         japan_correction,
        "trained_at":                   datetime.utcnow().isoformat(),
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = model_path.parent / f"xgb_overnight_snow_{ts}.pkl"
    with open(versioned_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModel saved -> {versioned_path}")
    shutil.copy2(versioned_path, model_path)
    print(f"Latest    -> {model_path}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default=str(DATASET_PATH))
    parser.add_argument("--output",   default=str(MODEL_PATH))
    parser.add_argument("--holdout",  default=HOLDOUT_SEASON,
                        help="First season to go into the test set (default %(default)s)")
    args = parser.parse_args()
    train(Path(args.dataset), Path(args.output), args.holdout)


if __name__ == "__main__":
    main()
