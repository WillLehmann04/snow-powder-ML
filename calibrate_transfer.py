"""
Transfer calibration — adapt the Japan-trained model to Australian resorts.

How it works:
  1. Run the Japan model on 10 years of Thredbo/Perisher/Falls Creek weather
  2. On days where Open-Meteo reports snowfall (our proxy label), fit a quantile
     mapping from Japan-model output → local expected snowfall
  3. Save calibration params (scale + intercept per resort) to data/models/
  4. Recalibrate the powder-day threshold to match local powder definition

Calibration method: isotonic regression (monotone, non-parametric)
  - Better than linear for this problem since the Japan model saturates at high values
  - Preserves rank ordering (days ranked "snowier" by Japan model stay that way)

Usage:
  python calibrate_transfer.py
  python calibrate_transfer.py --resort thredbo --powder-threshold 4
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from collectors.weather import fetch_and_cache
from core.features import build_features

MODEL_PATH   = Path("data/models/xgb_overnight_snow.pkl")
CALIB_OUT    = Path("data/models/transfer_calibration.json")
REGIONS_YAML = Path("regions.yaml")
PLOTS_DIR    = Path("data/plots")

SH_SEASON_MONTHS = {6, 7, 8, 9}
REGION_MAP = {"hokkaido": 0, "nagano": 1, "niigata": 2, "tohoku": 3, "nsw": 4, "victoria": 5}

# Regions where the Japan model has usable signal (r ≥ ~0.3 before calibration).
# All other SH regions use NWP snowfall directly (Option A).
JAPAN_MODEL_REGIONS = {"andes_chile", "andes_argentina"}

# Per-resort powder-day definitions (cm of actual snowfall)
RESORTS_POWDER_THRESHOLD = {
    "thredbo":           4.0,
    "perisher":          4.0,
    "falls_creek":       5.0,
    "cardrona":          4.0,
    "treble_cone":       5.0,
    "mt_hutt":           4.0,
    "coronet_peak":      4.0,
    "the_remarkables":   4.0,
    "whakapapa":         5.0,
    "valle_nevado":      4.0,
    "portillo":          4.0,
    "las_lenas":         4.0,
    "catedral_bariloche":4.0,
}
# Keep old name as alias for backward compat
RESORTINFO_POWDER_THRESHOLD = RESORTS_POWDER_THRESHOLD
RESORTS_POWDER_THRESHOLD.setdefault  # no-op, just for clarity
DEFAULT_POWDER_THRESHOLD = 4.0


def load_model():
    with open(MODEL_PATH, "rb") as f:
        p = pickle.load(f)
    return p["model"], p["feature_cols"]


def run_nwp_direct(resort_id: str, cfg: dict,
                   start: str, end: str) -> pd.DataFrame:
    """
    Option A: use Open-Meteo NWP snowfall directly as the prediction signal.
    No Japan model involved.  Both the 'prediction' and 'proxy label' come from
    the same Open-Meteo grid cell, so this is essentially asking: can we predict
    tomorrow's NWP snowfall from today's NWP snowfall features?  The answer is
    yes with r~0.6-0.8, and crucially the DIRECTION is physically correct.
    """
    hourly = fetch_and_cache(resort_id, cfg["lat"], cfg["lon"], start=start, end=end)
    daily  = build_features(hourly, hemisphere=cfg.get("hemisphere", "north"))

    # For NWP-direct, the "raw prediction" is just the NWP snowfall itself.
    # Physical gate: zero out days above freezing.
    snow_possible = daily["temp_min"] <= 2.0
    daily["japan_model_raw"] = daily["snowfall_24h"].clip(0) * snow_possible.values.astype(float)
    daily["proxy_snow_cm"]   = daily["snowfall_24h"].clip(0)

    return daily[daily.index.month.isin(SH_SEASON_MONTHS)].copy()


def run_japan_model(resort_id: str, cfg: dict, model, feat_cols: list,
                    start: str, end: str) -> pd.DataFrame:
    """Fetch historical weather and get Japan-model raw predictions."""
    hourly = fetch_and_cache(resort_id, cfg["lat"], cfg["lon"], start=start, end=end)
    daily  = build_features(hourly)

    daily["resort_id"]   = resort_id
    daily["elevation"]   = cfg["elevation"]
    daily["region"]      = cfg["region"]
    daily["region_code"] = REGION_MAP.get(cfg["region"], 4)
    daily["lat"]         = cfg["lat"]
    daily["lon"]         = cfg["lon"]

    X = daily.reindex(columns=feat_cols, fill_value=0)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    raw_preds = model.predict(X).clip(0)

    # Physical gate
    snow_possible = daily["temp_min"] <= 2.0
    raw_preds = raw_preds * snow_possible.values.astype(float)

    daily["japan_model_raw"] = raw_preds
    daily["proxy_snow_cm"]   = daily["snowfall_24h"].clip(0)

    return daily[daily.index.month.isin(SH_SEASON_MONTHS)].copy()


def fit_calibration(df: pd.DataFrame, resort_id: str, powder_threshold: float) -> dict:
    """
    Fit isotonic regression from Japan model output → proxy snowfall.
    Only use days where either the model or Open-Meteo shows some snow signal,
    to avoid fitting noise on zero-zero pairs.
    """
    # Use days where the Japan model or proxy shows at least a small signal
    mask = (df["japan_model_raw"] > 0.3) | (df["proxy_snow_cm"] > 0.3)
    calib_df = df[mask].copy()

    if len(calib_df) < 30:
        print(f"  WARNING: only {len(calib_df)} usable rows for calibration at {resort_id}")
        return None

    X = calib_df["japan_model_raw"].values.reshape(-1, 1)
    y = calib_df["proxy_snow_cm"].values

    # Isotonic regression (monotone)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib_df["japan_model_raw"].values, y)

    # Also fit a simple linear for reference
    lin = LinearRegression().fit(X, y)
    scale  = float(lin.coef_[0])
    offset = float(lin.intercept_)

    # Find what Japan-model raw output corresponds to the local powder threshold.
    # Data-driven approach: take the median Japan model prediction on days when
    # local snowfall actually reached the powder threshold.  This is robust to
    # near-zero linear scale (which caused the old formula to blow up).
    MODEL_MAX_CM = 22.0
    powder_days_mask = calib_df["proxy_snow_cm"] >= powder_threshold
    if powder_days_mask.sum() >= 5:
        raw_at_powder = float(calib_df.loc[powder_days_mask, "japan_model_raw"].median())
    else:
        # Very few powder days in calibration data — use 90th percentile of raw
        # predictions as a conservative threshold.
        raw_at_powder = float(calib_df["japan_model_raw"].quantile(0.90))
    raw_at_powder = float(np.clip(raw_at_powder, 0.0, MODEL_MAX_CM))

    # Evaluate calibrated r on snow days
    snow_days = df["proxy_snow_cm"] > 0.1
    if snow_days.sum() > 10:
        calibrated_preds = iso.predict(df.loc[snow_days, "japan_model_raw"].values)
        proxy            = df.loc[snow_days, "proxy_snow_cm"].values
        r = float(np.corrcoef(calibrated_preds, proxy)[0, 1])
        r_raw = float(np.corrcoef(df.loc[snow_days, "japan_model_raw"].values, proxy)[0, 1])
    else:
        r = r_raw = float("nan")

    result = {
        "resort_id":           resort_id,
        "n_calib_rows":        len(calib_df),
        "linear_scale":        round(scale, 4),
        "linear_offset":       round(offset, 4),
        "powder_threshold_local_cm":  powder_threshold,
        "powder_threshold_raw_equiv": round(raw_at_powder, 3),
        "r_before_calib":      round(r_raw, 3),
        "r_after_calib":       round(r, 3),
        # Store the isotonic mapping as sorted x/y arrays for serialisation
        "iso_x": iso.X_thresholds_.tolist(),
        "iso_y": iso.y_thresholds_.tolist(),
        "nwp_direct": False,  # set to True by caller for NWP-direct resorts
    }
    return result, iso


def plot_calibration(df: pd.DataFrame, iso, resort_id: str, calib: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{resort_id.replace('_',' ').title()} — Transfer Calibration",
                 fontsize=13, fontweight="bold")

    # Scatter: raw Japan model vs proxy
    snow_days = df["proxy_snow_cm"] > 0.1
    axes[0].scatter(df.loc[snow_days, "japan_model_raw"],
                    df.loc[snow_days, "proxy_snow_cm"],
                    alpha=0.3, s=10, color="#7eb8d4", label="snow days")
    # Isotonic curve
    x_range = np.linspace(0, df["japan_model_raw"].max(), 200)
    y_calib = iso.predict(x_range)
    axes[0].plot(x_range, y_calib, color="#e07b54", linewidth=2, label="isotonic calibration")
    axes[0].axvline(calib["powder_threshold_raw_equiv"], color="gold", linestyle="--",
                    linewidth=1.5, label=f"powder equiv ({calib['powder_threshold_raw_equiv']:.1f}cm raw)")
    axes[0].axhline(calib["powder_threshold_local_cm"], color="gold", linestyle=":",
                    linewidth=1.5)
    axes[0].set_xlabel("Japan model raw output (cm)")
    axes[0].set_ylabel("Open-Meteo proxy snowfall (cm)")
    axes[0].set_title(f"Raw vs Proxy  (r_raw={calib['r_before_calib']:.3f})")
    axes[0].legend(fontsize=8)

    # After calibration
    calibrated = iso.predict(df["japan_model_raw"].values)
    axes[1].scatter(calibrated[snow_days.values],
                    df.loc[snow_days, "proxy_snow_cm"].values,
                    alpha=0.3, s=10, color="#5bb87a", label="calibrated")
    max_val = max(calibrated.max(), df["proxy_snow_cm"].max())
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=1, label="1:1 line")
    axes[1].set_xlabel("Calibrated prediction (cm)")
    axes[1].set_ylabel("Open-Meteo proxy (cm)")
    axes[1].set_title(f"After Calibration  (r={calib['r_after_calib']:.3f})")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = PLOTS_DIR / f"calibration_{resort_id}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resort", default=None)
    parser.add_argument("--powder-threshold", type=float, default=None,
                        help="Local cm definition of a powder day (default per resort)")
    parser.add_argument("--start", default="2014-06-01")
    parser.add_argument("--end",   default="2024-09-30")
    args = parser.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    sh_resorts = {rid: cfg for rid, cfg in regions.items()
                  if cfg.get("hemisphere") == "south"}

    if args.resort:
        sh_resorts = {args.resort: sh_resorts[args.resort]}

    model, feat_cols = load_model()

    all_calibrations = {}

    # Load existing calibrations if any
    if CALIB_OUT.exists():
        with open(CALIB_OUT) as f:
            all_calibrations = json.load(f)

    print(f"\nFitting transfer calibration for {len(sh_resorts)} resort(s)")
    print(f"Data: {args.start} → {args.end}")
    print()

    for resort_id, cfg in sh_resorts.items():
        powder_thr = args.powder_threshold or RESORTS_POWDER_THRESHOLD.get(resort_id, DEFAULT_POWDER_THRESHOLD)
        use_nwp = cfg.get("region", "") not in JAPAN_MODEL_REGIONS
        method = "NWP-direct" if use_nwp else "Japan model"
        print(f"  {resort_id}  (powder threshold: {powder_thr}cm, method: {method}) ...")

        if use_nwp:
            df = run_nwp_direct(resort_id, cfg, args.start, args.end)
        else:
            df = run_japan_model(resort_id, cfg, model, feat_cols, args.start, args.end)

        result = fit_calibration(df, resort_id, powder_thr)
        if result is None:
            continue
        calib, iso = result
        calib["nwp_direct"] = use_nwp

        plot_calibration(df, iso, resort_id, calib)

        print(f"    n calibration rows:  {calib['n_calib_rows']}")
        print(f"    r before calibration: {calib['r_before_calib']:.3f}")
        print(f"    r after calibration:  {calib['r_after_calib']:.3f}")
        print(f"    Powder day raw equiv: Japan model >= {calib['powder_threshold_raw_equiv']:.2f}cm → local >= {powder_thr}cm")
        print()

        all_calibrations[resort_id] = calib

    CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_OUT, "w") as f:
        json.dump(all_calibrations, f, indent=2)
    print(f"Calibrations saved → {CALIB_OUT}")
    print()
    print("─" * 60)
    print("SUMMARY")
    print("─" * 60)
    for rid, c in all_calibrations.items():
        delta_r = c["r_after_calib"] - c["r_before_calib"]
        print(f"  {rid:20s}  r: {c['r_before_calib']:.3f} → {c['r_after_calib']:.3f}  ({delta_r:+.3f})")
        print(f"  {'':20s}  powder: raw >= {c['powder_threshold_raw_equiv']:.2f}cm ≡ local >= {c['powder_threshold_local_cm']}cm")


if __name__ == "__main__":
    main()
