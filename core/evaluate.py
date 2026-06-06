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


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload


def _prep_X(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    X = df.reindex(columns=feat_cols, fill_value=0.0)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)
    return X


def _predict(model, feat_cols: list, df: pd.DataFrame, log_transform: bool) -> np.ndarray:
    X = _prep_X(df, feat_cols)
    raw = model.predict(X)
    preds = np.expm1(raw).clip(0) if log_transform else raw.clip(0)
    return preds


def _metrics_row(y_true: np.ndarray, y_pred: np.ndarray, powder_pred_thresh: float) -> dict:
    if len(y_true) < 5:
        return {}
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r    = pearsonr(y_pred, y_true)[0] if y_true.std() > 0 else float("nan")

    snow  = y_true >= 1.0
    r_sn  = pearsonr(y_pred[snow], y_true[snow])[0] if snow.sum() > 5 and y_true[snow].std() > 0 else float("nan")

    actual_p = y_true >= POWDER_CM
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

    df = pd.read_parquet(DATASET_PATH)
    train_df = df[df["season"] <  holdout]
    test_df  = df[df["season"] >= holdout]

    # Use the threshold calibrated at training time (stored in the payload)
    powder_pred_thresh = payload.get("powder_pred_threshold", POWDER_CM * 0.5)
    print(f"  Powder detection threshold (predicted cm): {powder_pred_thresh:.2f}")

    # ── Overall metrics ───────────────────────────────────────────────────────
    y_te = test_df[TARGET].values
    p_te = _predict(model, feat_cols, test_df, log_transform)

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

    # ── Per-resort breakdown ──────────────────────────────────────────────────
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Per-resort (test set)")
    print(f"{sep}")
    print(f"  {'Resort':<25} {'n':>5} {'r':>6} {'r_snow':>7} {'MAE':>7} {'F1':>6} {'powder_days':>12}")
    print(f"  {'':-<25} {'':->5} {'':->6} {'':->7} {'':->7} {'':->6} {'':->12}")

    test_df = test_df.copy()
    test_df["pred"] = p_te

    for resort, grp in sorted(test_df.groupby("resort_id")):
        mr = _metrics_row(grp[TARGET].values, grp["pred"].values, powder_pred_thresh)
        if not mr:
            continue
        print(f"  {resort:<25} {mr['n']:>5} {mr['r']:>6.3f} {mr['r_snow']:>7.3f} "
              f"{mr['mae']:>6.2f}cm {mr['f1']:>6.3f} "
              f"{mr['tp']:>5}/{(grp[TARGET]>=POWDER_CM).sum():<5}")

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

    from core.train import FEATURE_COLS as FEAT

    for held_out in resorts:
        tr = df[df["resort_id"] != held_out]
        te = df[df["resort_id"] == held_out]

        X_tr = _prep_X(tr, FEAT)
        X_te = _prep_X(te, FEAT)
        y_tr = tr[TARGET].values
        y_te = te[TARGET].values

        m_loro = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, random_state=42, n_jobs=-1,
        )
        m_loro.fit(X_tr, np.log1p(y_tr), verbose=False)
        p = np.expm1(m_loro.predict(X_te)).clip(0)

        ap = y_te >= POWDER_CM
        thr = float(np.percentile(p[ap], 25)) if ap.sum() >= 5 else POWDER_CM * 0.5
        mr = _metrics_row(y_te, p, thr)
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
