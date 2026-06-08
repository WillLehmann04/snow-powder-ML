"""
Evaluate the trained model: season holdout + leave-one-resort-out.

Loads the model from data/models/xgb_overnight_snow.pkl and the dataset
from data/processed/training_dataset.parquet, then prints:
  - Overall test-set metrics
  - Per-resort breakdown on the test set
  - Leave-one-resort-out summary (train on all except one, predict that one)

Usage:
  python -m core.evaluate
  python -m core.evaluate --loro      # also run leave-one-resort-out (slow)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATASET_PATH = Path("data/processed/training_dataset.parquet")
MODEL_PATH   = Path("data/models/xgb_overnight_snow.pkl")
TARGET       = "overnight_snow_cm"
POWDER_CM    = 15.0

_HPA_COLS = {"temp_850_mean", "temp_850_min", "temp_850_max",
             "temp_850_trend", "rain_risk_850", "freeze_depth_850"}


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload


def _prep_X(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    X = df.reindex(columns=feat_cols, fill_value=0.0)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Keep 850hPa NaN so XGBoost routes pre-2021 rows correctly (0°C at 850hPa
    # is a real weather state — filling with 0 would be wrong).
    non_hpa = [c for c in X.columns if c not in _HPA_COLS]
    X[non_hpa] = X[non_hpa].fillna(0.0)
    return X


def _predict(model, feat_cols: list, df: pd.DataFrame, log_transform: bool,
             snow_classifier=None, snow_gate_threshold: float = 0.20,
             japan_correction: dict = None) -> np.ndarray:
    X = _prep_X(df, feat_cols)
    raw = model.predict(X)
    preds = np.expm1(raw).clip(0) if log_transform else raw.clip(0)
    if snow_classifier is not None:
        p_snow = snow_classifier.predict_proba(X)[:, 1]
        preds = preds * (p_snow >= snow_gate_threshold)
    if japan_correction is not None:
        iso_x = np.array(japan_correction["iso_x"])
        iso_y = np.array(japan_correction["iso_y"])
        preds = np.interp(preds, iso_x, iso_y).clip(0)
    return preds


def _metrics_row(y_true: np.ndarray, y_pred: np.ndarray, powder_pred_thresh: float,
                 actual_powder_cm: float = POWDER_CM) -> dict:
    if len(y_true) < 5:
        return {}
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r    = pearsonr(y_pred, y_true)[0] if y_true.std() > 0 else float("nan")

    snow  = y_true >= 1.0
    r_sn  = pearsonr(y_pred[snow], y_true[snow])[0] if snow.sum() > 5 and y_true[snow].std() > 0 else float("nan")

    actual_p = y_true >= actual_powder_cm
    pred_p   = y_pred >= powder_pred_thresh
    tp = (actual_p & pred_p).sum()
    fp = (~actual_p & pred_p).sum()
    fn = (actual_p & ~pred_p).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return dict(n=len(y_true), mae=mae, rmse=rmse, r=r, r_snow=r_sn,
                prec=prec, rec=rec, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn))


def evaluate(run_loro: bool = False) -> None:
    print("Loading model and dataset ...")
    payload      = load_model()
    model        = payload["model"]
    feat_cols    = payload["feature_cols"]
    log_transform = payload.get("log_transform", False)
    holdout      = payload.get("holdout_season", "2022-2023")
    train_m      = payload.get("train_metrics", {})
    test_m_saved = payload.get("test_metrics", {})
    snow_classifier     = payload.get("snow_classifier")
    snow_gate_threshold = payload.get("snow_gate_threshold", 0.20)
    japan_correction    = payload.get("japan_correction_iso")

    df = pd.read_parquet(DATASET_PATH)
    train_df = df[df["season"] <  holdout]
    test_df  = df[df["season"] >= holdout]

    # Use the threshold calibrated at training time (stored in the payload)
    powder_pred_thresh = payload.get("powder_pred_threshold", POWDER_CM * 0.5)
    print(f"  Powder detection threshold (predicted cm): {powder_pred_thresh:.2f}")
    if japan_correction:
        print(f"  Japan post-hoc correction loaded ({len(japan_correction['iso_x'])} knots)")

    # ── Overall metrics ───────────────────────────────────────────────────────
    y_te = test_df[TARGET].values
    p_te = _predict(model, feat_cols, test_df, log_transform,
                    snow_classifier, snow_gate_threshold, japan_correction)

    m = _metrics_row(y_te, p_te, powder_pred_thresh)
    stored_r = test_m_saved.get("r", "N/A")

    print(f"\n{'=' * 60}")
    print(f"  SEASON HOLDOUT  (test = seasons >= {holdout})")
    print(f"{'=' * 60}")
    print(f"  n={m['n']:,}  MAE={m['mae']:.2f}cm  RMSE={m['rmse']:.2f}cm  r={m['r']:.3f}")
    print(f"  Snow days:  r={m['r_snow']:.3f}")
    print(f"  Powder (>=15cm actual, >={powder_pred_thresh:.1f}cm pred):")
    print(f"    Precision={m['prec']:.3f}  Recall={m['rec']:.3f}  F1={m['f1']:.3f}")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    if train_m:
        print(f"\n  Train r={train_m.get('r','?')}  ->  Test r={m['r']:.3f}  "
              f"(gap = {m['r'] - train_m.get('r', m['r']):.3f})")

    # ── Hemisphere-split metrics ───────────────────────────────────────────────
    POWDER_THRESHOLDS_BY_HEMI = {"north": 15.0, "south": 4.0}
    test_df_copy = test_df.copy()
    test_df_copy["pred"] = p_te

    if "hemisphere" in test_df_copy.columns:
        print(f"\n{'=' * 60}")
        print(f"  HEMISPHERE-SPLIT METRICS (test set)")
        print(f"{'=' * 60}")
        for hemi, hemi_df in sorted(test_df_copy.groupby("hemisphere")):
            y_h   = hemi_df[TARGET].values
            p_h   = hemi_df["pred"].values
            thr_h = POWDER_THRESHOLDS_BY_HEMI.get(hemi, POWDER_CM)
            from core.train import _find_best_threshold
            hemi_thr = _find_best_threshold(y_h, p_h, powder_cm=thr_h)
            mh = _metrics_row(y_h, p_h, hemi_thr)
            if not mh:
                continue
            resorts_h = sorted(hemi_df["resort_id"].unique())
            print(f"\n  {hemi.upper()} ({len(resorts_h)} resorts)")
            print(f"  Resorts: {', '.join(resorts_h)}")
            print(f"  n={mh['n']:,}  MAE={mh['mae']:.2f}cm  r={mh['r']:.3f}")
            print(f"  Powder (actual>={thr_h}cm, pred>={hemi_thr:.1f}cm):")
            print(f"    Precision={mh['prec']:.3f}  Recall={mh['rec']:.3f}  F1={mh['f1']:.3f}")
            print(f"    TP={mh['tp']}  FP={mh['fp']}  FN={mh['fn']}")

    # ── Per-resort breakdown ──────────────────────────────────────────────────
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Per-resort (test set)")
    print(f"{sep}")
    print(f"  {'Resort':<25} {'n':>5} {'r':>6} {'r_snow':>7} {'MAE':>7} {'F1':>6} {'powder_days':>12}")
    print(f"  {'':-<25} {'':->5} {'':->6} {'':->7} {'':->7} {'':->6} {'':->12}")

    for resort, grp in sorted(test_df_copy.groupby("resort_id")):
        hemi_grp = grp["hemisphere"].iloc[0] if "hemisphere" in grp.columns else "north"
        resort_powder_cm = POWDER_THRESHOLDS_BY_HEMI.get(hemi_grp, POWDER_CM)
        mr = _metrics_row(grp[TARGET].values, grp["pred"].values, powder_pred_thresh,
                          actual_powder_cm=resort_powder_cm)
        if not mr:
            continue
        print(f"  {resort:<25} {mr['n']:>5} {mr['r']:>6.3f} {mr['r_snow']:>7.3f} "
              f"{mr['mae']:>6.2f}cm {mr['f1']:>6.3f} "
              f"{mr['tp']:>5}/{(grp[TARGET]>=resort_powder_cm).sum():<5}")

    # ── Leave-one-resort-out ──────────────────────────────────────────────────
    if not run_loro:
        print("\n  (Run with --loro to see leave-one-resort-out results)")
        return

    print(f"\n{'=' * 60}")
    print(f"  LEAVE-ONE-RESORT-OUT  (trains on all but one, predicts held-out)")
    print(f"{'=' * 60}")

    resorts = sorted(df["resort_id"].unique())
    print(f"  {'Resort':<25} {'r':>6} {'MAE':>7} {'F1':>6} {'n_test':>7}")
    print(f"  {'':-<25} {'':->6} {'':->7} {'':->6} {'':->7}")

    from core.train import FEATURE_COLS as FEAT, _find_best_threshold

    for held_out in resorts:
        tr = df[df["resort_id"] != held_out]
        te = df[df["resort_id"] == held_out]

        X_tr = _prep_X(tr, FEAT)
        X_te = _prep_X(te, FEAT)
        y_tr = tr[TARGET].values
        y_te = te[TARGET].values

        m_loro = XGBRegressor(
            n_estimators          = 800,
            max_depth             = 5,
            learning_rate         = 0.04,
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            min_child_weight      = 7,
            reg_alpha             = 0.1,
            reg_lambda            = 1.0,
            objective             = "reg:tweedie",
            tweedie_variance_power= 1.2,
            random_state          = 42,
            n_jobs                = -1,
        )
        m_loro.fit(X_tr, y_tr, verbose=False)
        p_tr = m_loro.predict(X_tr).clip(0)
        p_te = m_loro.predict(X_te).clip(0)

        # Threshold derived from training data only (not from the held-out resort)
        japan_mask = (tr["hemisphere"] == "north").values if "hemisphere" in tr.columns \
                     else np.ones(len(y_tr), dtype=bool)
        thr = _find_best_threshold(y_tr[japan_mask], p_tr[japan_mask], powder_cm=POWDER_CM)
        mr = _metrics_row(y_te, p_te, thr)
        if not mr:
            continue
        print(f"  {held_out:<25} {mr['r']:>6.3f} {mr['mae']:>6.2f}cm "
              f"{mr['f1']:>6.3f} {mr['n']:>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loro", action="store_true",
                        help="Run leave-one-resort-out validation (slow)")
    args = parser.parse_args()
    evaluate(run_loro=args.loro)


if __name__ == "__main__":
    main()
