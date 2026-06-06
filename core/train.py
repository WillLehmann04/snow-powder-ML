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
from xgboost import XGBRegressor

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
    "pressure_mean", "pressure_min",
    "pressure_tendency_24h", "pressure_tendency_3d", "pressure_anomaly",
    # ── 850 hPa temperature (real data from 2021-03-23, imputed earlier) ────────
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
    "snowfall_last_30d", "freeze_thaw_count_7d", "seasonal_snowfall_total",
    # ── Time-since features ────────────────────────────────────────────────
    "days_since_last_snow", "days_since_major_snow",
    # ── Ratio / index features ─────────────────────────────────────────────
    "snowfall_ratio_1d_3d", "snowfall_ratio_3d_7d",
    "wind_loading_index", "time_since_storm_peak",
    # ── Critical derived ───────────────────────────────────────────────────
    "snow_quality_index", "powder_freshness", "melt_risk", "wind_damage_score",
    "has_base_snow", "intensity_ratio", "wind_after_storm", "temp_trend",
    # ── Calendar ───────────────────────────────────────────────────────────
    "day_of_year", "month", "days_in_season",
    # ── Snowpack state (observed) ──────────────────────────────────────────────
    "snow_depth_lag1",
    # ── NWP bias correction (resort-level climatological ratio) ───────────────
    "nwp_amplification",
    # ── Static resort ──────────────────────────────────────────────────────
    "elevation", "lat", "lon", "region_code",
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

    X_train = _prep_X(train_df)
    X_test  = _prep_X(test_df)
    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    # ── Model — Tweedie distribution ─────────────────────────────────────────
    # Tweedie (variance_power between 1 and 2) is designed for zero-inflated,
    # right-skewed positive data — exactly our target distribution (76% zeros,
    # long tail to 80cm). It doesn't need a log-transform: the log-link and
    # compound Poisson-Gamma structure handle the zeros natively.
    # variance_power=1.2 (tuned): closer to Poisson than Gamma, which fits the
    # overnight snowfall distribution (many zeros, discrete-ish counts) better.
    model = XGBRegressor(
        n_estimators          = 600,
        max_depth             = 4,
        learning_rate         = 0.04,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        objective             = "reg:tweedie",
        tweedie_variance_power= 1.2,
        eval_metric           = "tweedie-nloglik@1.2",
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
    )

    print("\nTraining (Tweedie distribution) ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    print(f"  Best iteration: {model.best_iteration}")

    # Tweedie outputs are already in cm (positive, via log-link internally)
    p_train = model.predict(X_train).clip(0)
    p_test  = model.predict(X_test).clip(0)

    # Threshold calibrated on training data only, then applied to both splits
    powder_pred_thresh = _find_best_threshold(y_train, p_train)
    print(f"\n  Powder detection threshold (F1-optimal on train): {powder_pred_thresh:.2f}cm")

    print("\n--- TRAIN metrics ---")
    train_metrics = _metrics(y_train, p_train, powder_pred_thresh)
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- TEST metrics (season holdout) ---")
    test_metrics = _metrics(y_test, p_test, powder_pred_thresh)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":                 model,
        "feature_cols":          FEATURE_COLS,
        "holdout_season":        holdout_season,
        "train_metrics":         train_metrics,
        "test_metrics":          test_metrics,
        "log_transform":         False,   # Tweedie handles zeros natively; no expm1 needed
        "powder_pred_threshold": powder_pred_thresh,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModel saved -> {model_path}")
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
