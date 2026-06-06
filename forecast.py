"""
Powder forecast — fetch 7-day weather and predict snowfall + powder score.

Usage:
  python forecast.py                              # all resorts
  python forecast.py --resort niseko_grand_hirafu
  python forecast.py --resort hakuba_happo --days 3
  python forecast.py --all --json                 # machine-readable output
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from collectors.weather import fetch_hourly
from core.features import build_features

MODEL_PATH   = Path("data/models/xgb_overnight_snow.pkl")
CALIB_PATH   = Path("data/models/transfer_calibration.json")
REGIONS_YAML = Path("regions.yaml")

REGION_MAP = {
    "hokkaido": 0, "nagano": 1, "niigata": 2, "tohoku": 3,  # Japan (trained)
    "nsw": 4, "victoria": 5,                                  # Australia
    "nz_south": 6, "nz_north": 7,                            # New Zealand
    "andes_chile": 8, "andes_argentina": 9,                   # South America
}

# The model (XGBoost regression) systematically undershoots big snowfall events
# because it regresses toward the mean. Diagnostic on test set shows:
#   - p90 of predictions  = 9.5cm  (vs actual p90 = 15cm)
#   - p99 of predictions  = 22.9cm (vs actual p99 = 45cm)
#   - On real ≥15cm powder days, median model output = 8.6cm
# So we calibrate the score against the model's own output range, not raw cm.

POWDER_SCORE_THRESHOLD = 7.0  # predicted cm → flag as powder day
                               # ~equivalent to ~15cm actual given model's underprediction bias

# Model output scale anchors (from test-set diagnostics):
_SNOW_SCALE_50 =  5.0   # model output at which score hits ~50 (good day)
_SNOW_SCALE_100 = 15.0  # model output at which score saturates at 100 (exceptional)


def powder_score(predicted_snow_cm: float, snow_temp_mean: float, wind_during_snow_mean: float) -> int:
    """Derive a 0–100 powder quality score from model outputs.

    Calibrated against the model's prediction range (not raw snowfall cm).
    A model output of 15cm → snow_component=1.0 (exceptional).
    A model output of  7cm → snow_component=0.47 (solid powder day).
    Zero snow = zero score regardless of temp/wind.
    """
    if predicted_snow_cm < 0.5:
        return 0
    snow_component = min(predicted_snow_cm / _SNOW_SCALE_100, 1.0)
    temp_quality   = min(1.0, max(0.0, 1.0 - (snow_temp_mean + 5) / 10))  # peaks at -5C, 0 above 5C
    wind_penalty   = max(0.0, 1.0 - wind_during_snow_mean / 40)
    return round(100 * snow_component * 0.6 + 100 * temp_quality * wind_penalty * 0.4)


def load_model():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run: python -m core.train")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_cols"], payload.get("log_transform", False)


def load_calibrations() -> dict:
    """Load transfer calibration params for southern hemisphere resorts."""
    if not CALIB_PATH.exists():
        return {}
    with open(CALIB_PATH) as f:
        return json.load(f)


def _apply_calibration(raw_preds: np.ndarray, calib: dict) -> np.ndarray:
    """Apply isotonic calibration mapping via linear interpolation of stored knots."""
    iso_x = np.array(calib["iso_x"])
    iso_y = np.array(calib["iso_y"])
    return np.interp(raw_preds, iso_x, iso_y).clip(0)


def forecast_resort(resort_id: str, cfg: dict, model, feat_cols: list,
                    calibrations: dict, days: int = 7,
                    log_transform: bool = False) -> list[dict]:
    """Fetch forecast, run model, return list of daily dicts."""
    from datetime import date, timedelta
    today = date.today()
    start = today.strftime("%Y-%m-%d")
    end   = (today + timedelta(days=days - 1)).strftime("%Y-%m-%d")

    is_southern = cfg.get("hemisphere") == "south"
    calib       = calibrations.get(resort_id) if is_southern else None

    # Per-resort powder threshold: lower for AU resorts
    powder_threshold = calib["powder_threshold_raw_equiv"] if calib else POWDER_SCORE_THRESHOLD

    try:
        hourly = fetch_hourly(cfg["lat"], cfg["lon"], start, end, forecast=True)
    except Exception as e:
        return [{"error": str(e)}]

    daily = build_features(hourly, hemisphere=cfg.get("hemisphere", "north"))

    # Add static resort features
    daily["resort_id"]   = resort_id
    daily["elevation"]   = cfg["elevation"]
    daily["region"]      = cfg["region"]
    daily["region_code"] = REGION_MAP.get(cfg["region"], -1)
    daily["lat"]         = cfg["lat"]
    daily["lon"]         = cfg["lon"]

    preds = None

    # ── NWP-direct path (AU/NZ): use 48h NWP accumulation as the snowfall signal.
    # snowfall_24h is what we're predicting; snowfall_48h is the lead indicator
    # (includes yesterday's snowfall, which is known at forecast time).
    if calib and calib.get("nwp_direct"):
        snow_possible = daily["temp_min"] <= 2.0
        preds = daily["snowfall_48h"].clip(0).values * snow_possible.values.astype(float)
    else:
        # ── Japan model path (Japan + Andes resorts) ──────────────────────────
        # Align to model feature columns, fill any missing with 0
        X = daily.reindex(columns=feat_cols, fill_value=0)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        raw = model.predict(X)
        preds = np.expm1(raw).clip(0) if log_transform else raw.clip(0)

        # Physical gate: no snow above freezing
        snow_possible = daily["temp_min"] <= 2.0
        preds = preds * snow_possible.values.astype(float)

    # ── Transfer calibration (both paths) ────────────────────────────────────
    if calib:
        preds = _apply_calibration(preds, calib)

    results = []
    for i, (date_idx, row) in enumerate(daily.iterrows()):
        pred_cm = float(preds[i])
        score   = powder_score(
            pred_cm,
            float(row.get("snow_temp_mean",        0)),
            float(row.get("wind_during_snow_mean", 0)),
        )
        local_threshold = calib["powder_threshold_local_cm"] if calib else None
        results.append({
            "date":              str(date_idx.date()),
            "predicted_snow_cm": round(pred_cm, 1),
            "powder_score":      score,
            "is_powder_day":     pred_cm >= powder_threshold,
            "powder_threshold":  local_threshold or POWDER_SCORE_THRESHOLD,
            "temp_min_c":        round(float(row.get("temp_min",  0)), 1),
            "temp_max_c":        round(float(row.get("temp_max",  0)), 1),
            "wind_max_kmh":      round(float(row.get("wind_max",  0)), 1),
            "wind_mean_kmh":     round(float(row.get("wind_mean", 0)), 1),
            "snow_temp_c":       round(float(row.get("snow_temp_mean",        0)), 1),
            "wind_during_snow":  round(float(row.get("wind_during_snow_mean", 0)), 1),
            "humidity_pct":      round(float(row.get("humidity_mean", 0)), 1),
            "snowfall_48h_mm":   round(float(row.get("snowfall_48h", 0)), 1),
        })
    return results


def render_resort(resort_id: str, days: list[dict]) -> None:
    """Print a clean terminal output for one resort."""
    resort_label = resort_id.replace("_", " ").title()

    if days and "error" in days[0]:
        print(f"\n  {resort_label}: ERROR — {days[0]['error']}")
        return

    powder_days = [d for d in days if d["is_powder_day"]]
    best        = max(days, key=lambda d: d["powder_score"])

    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {resort_label}")
    print(f"{sep}")
    print(f"  {'Date':<12} {'Snow':>6} {'Score':>6}  {'Temp':>14}  {'Wind':>10}  Conditions")
    print(f"  {'':-<12} {'':->6} {'':->6}  {'':->14}  {'':->10}  {'':->12}")

    for d in days:
        snow  = f"{d['predicted_snow_cm']}cm"
        score = f"{d['powder_score']}/100"
        temp  = f"{d['temp_min_c']} to {d['temp_max_c']}C"
        wind  = f"{d['wind_mean_kmh']}kmh"
        tag   = "*** POWDER ***" if d["is_powder_day"] else ""
        print(f"  {d['date']:<12} {snow:>6} {score:>6}  {temp:>10}  {wind:>10}  {tag}")

    print()
    if powder_days:
        print(f"  Powder days: {len(powder_days)}/7  |  Best day: {best['date']} "
              f"(score {best['powder_score']}/100, ~{best['predicted_snow_cm']}cm)")
    else:
        print(f"  No powder days forecast.  Best day: {best['date']} "
              f"(score {best['powder_score']}/100, ~{best['predicted_snow_cm']}cm)")


def main():
    parser = argparse.ArgumentParser(description="7-day powder forecast for Japanese ski resorts.")
    parser.add_argument("--resort", default=None,  help="Single resort ID (e.g. niseko_grand_hirafu)")
    parser.add_argument("--days",   type=int, default=7, help="Forecast horizon in days (default 7)")
    parser.add_argument("--json",   action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    if args.resort:
        if args.resort not in regions:
            sys.exit(f"Resort '{args.resort}' not in regions.yaml. Available: {', '.join(regions)}")
        targets = {args.resort: regions[args.resort]}
    else:
        targets = regions

    model, feat_cols, log_transform = load_model()
    calibrations = load_calibrations()
    if calibrations:
        print(f"  Loaded transfer calibration for: {', '.join(calibrations)}")

    all_results = {}
    for resort_id, cfg in targets.items():
        print(f"  Fetching forecast for {resort_id} ...", end="\r")
        all_results[resort_id] = forecast_resort(
            resort_id, cfg, model, feat_cols, calibrations,
            days=args.days, log_transform=log_transform,
        )

    if args.json:
        print(json.dumps(all_results, indent=2))
        return

    # ── Formatted output ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  POWDER FORECAST")
    print("=" * 62)

    # Sort resorts: powder days first, then by best score
    def sort_key(item):
        days_list = item[1]
        if days_list and "error" not in days_list[0]:
            powder = sum(1 for d in days_list if d["is_powder_day"])
            best   = max(d["powder_score"] for d in days_list)
            return (-powder, -best)
        return (0, 0)

    for resort_id, days_list in sorted(all_results.items(), key=sort_key):
        render_resort(resort_id, days_list)

    print("\n" + "=" * 62)
    thresholds = {r: d[0]["powder_threshold"] for r, d in all_results.items()
                  if d and "error" not in d[0] and "powder_threshold" in d[0]}
    unique_thr = sorted(set(thresholds.values()))
    for thr in unique_thr:
        resorts = [r for r, t in thresholds.items() if t == thr]
        print(f"  Powder threshold >={thr}cm: {', '.join(r.replace('_',' ') for r in resorts)}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
