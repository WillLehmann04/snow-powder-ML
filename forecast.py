"""
Powder forecast — fetch 7-day weather and predict snowfall + powder score.

Usage:
  python forecast.py                              # all resorts
  python forecast.py --resort niseko_grand_hirafu
  python forecast.py --resort hakuba_happo --days 3
  python forecast.py --all --json                 # machine-readable output
"""

import argparse
import calendar
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from collectors.weather import fetch_hourly
from core.conditions import estimate_condition
from core.features import build_features

MODEL_PATH       = Path("data/models/xgb_overnight_snow.pkl")
CALIB_PATH       = Path("data/models/transfer_calibration.json")
NWP_MODELS_PATH  = Path("data/models/nwp_direct_models.pkl")
REGIONS_YAML     = Path("regions.yaml")

# Features used by the NWP-direct multi-feature XGBoost calibration model.
NWP_DIRECT_FEATURES = [
    "snowfall_48h", "snowfall_72h", "temp_min",
    "hours_with_snow", "precipitation", "pressure_tendency_24h",
]

_HPA_COLS_SET = frozenset({
    "temp_850_mean", "temp_850_min", "temp_850_max",
    "temp_850_trend", "rain_risk_850", "freeze_depth_850",
})

REGION_MAP = {
    "hokkaido": 0, "nagano": 1, "niigata": 2, "tohoku": 3,  # Japan (trained)
    "nsw": 4, "victoria": 5,                                  # Australia
    "nz_south": 6, "nz_north": 7,                            # New Zealand
    "andes_chile": 8, "andes_argentina": 9,                   # South America
    "bc_canada": 10,                                          # British Columbia
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


def _in_season(cfg: dict) -> bool:
    """Return True if today falls within the resort's declared operating season."""
    opens  = cfg.get("opens_month")
    closes = cfg.get("closes_month")
    if opens is None or closes is None:
        return True
    m = pd.Timestamp.today().month
    if opens <= closes:      # SH/simple range e.g. Jun–Oct
        return opens <= m <= closes
    else:                    # NH wrap e.g. Nov–Apr
        return m >= opens or m <= closes


def _reopen_str(cfg: dict) -> str:
    opens = cfg.get("opens_month")
    return calendar.month_name[opens] if opens else "unknown"


def load_model():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run: python -m core.train")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return (
        payload["model"],
        payload["feature_cols"],
        payload.get("log_transform", False),
        payload.get("nwp_amplification_per_resort", {}),
        payload.get("snow_classifier"),
        payload.get("snow_gate_threshold", 0.20),
        payload.get("japan_correction_iso"),
        payload.get("powder_pred_threshold", 7.0),
    )


def load_calibrations() -> dict:
    """Load transfer calibration params for southern hemisphere resorts."""
    if not CALIB_PATH.exists():
        return {}
    with open(CALIB_PATH) as f:
        return json.load(f)


def load_nwp_direct_models() -> dict:
    """Load per-resort NWP-direct XGBoost calibration models if available."""
    if not NWP_MODELS_PATH.exists():
        return {}
    with open(NWP_MODELS_PATH, "rb") as f:
        return pickle.load(f)


def _apply_calibration(raw_preds: np.ndarray, calib: dict) -> np.ndarray:
    """Apply isotonic calibration mapping via linear interpolation of stored knots."""
    iso_x = np.array(calib["iso_x"])
    iso_y = np.array(calib["iso_y"])
    return np.interp(raw_preds, iso_x, iso_y).clip(0)


def forecast_resort(resort_id: str, cfg: dict, model, feat_cols: list,
                    calibrations: dict, amp_per_resort: dict, days: int = 7,
                    log_transform: bool = False, snow_classifier=None,
                    snow_gate_threshold: float = 0.20,
                    nwp_direct_models: dict = None,
                    japan_correction: dict = None,
                    japan_powder_threshold: float = 7.0) -> list[dict]:
    """Fetch forecast, run model, return list of daily dicts."""
    from datetime import date, timedelta
    today = date.today()
    start = today.strftime("%Y-%m-%d")
    end   = (today + timedelta(days=days - 1)).strftime("%Y-%m-%d")

    is_southern = cfg.get("hemisphere") == "south"
    calib       = calibrations.get(resort_id) if is_southern else None

    # Powder threshold: SH resorts use their local ERA5 cm definition (predictions
    # are in local cm after calibration). Japan NH resorts use the F1-optimal
    # threshold from training (post-correction, in Japan model output cm).
    powder_threshold = calib["powder_threshold_local_cm"] if calib else japan_powder_threshold

    # Fetch 35 days of historical data as a rolling-window warmup so that
    # snowfall_72h, snowfall_7d, snowfall_14d, snowfall_last_30d, and
    # days_since_last_snow all have valid history on day 1 of the forecast.
    warmup_start = (today - timedelta(days=35)).strftime("%Y-%m-%d")
    yesterday    = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        hourly_hist  = fetch_hourly(cfg["lat"], cfg["lon"], warmup_start, yesterday)
        hourly_fcast = fetch_hourly(cfg["lat"], cfg["lon"], start, end, forecast=True)
        hourly = pd.concat([hourly_hist, hourly_fcast])
        hourly = hourly[~hourly.index.duplicated(keep="last")]
    except Exception as e:
        return [{"error": str(e)}]

    daily = build_features(hourly, hemisphere=cfg.get("hemisphere", "north"))
    # Keep only the forward-looking forecast rows for output
    daily = daily[daily.index.normalize() >= pd.Timestamp(start)]

    # Add static resort features
    daily["resort_id"]   = resort_id
    daily["elevation"]   = cfg["elevation"]
    daily["region"]      = cfg["region"]
    daily["region_code"] = REGION_MAP.get(cfg["region"], -1)
    daily["lat"]         = cfg["lat"]
    daily["lon"]         = cfg["lon"]

    # ── Always set NWP amplification before calling the model ────────────────
    # For SH Andes resorts (Japan-model path + calibration): use the calibration's
    # stored factor (fitted on the same data). For all other resorts (Japan NH +
    # NWP-direct SH): use the per-resort factor stored in the model payload.
    if calib and not calib.get("nwp_direct"):
        amp = calib.get("nwp_amplification", 1.0)
    else:
        amp = amp_per_resort.get(resort_id, 1.0)

    daily["nwp_amplification"]      = amp
    daily["amplified_snowfall_24h"] = daily["snowfall_24h"] * amp
    daily["amplified_snowfall_48h"] = daily["snowfall_48h"] * amp

    if calib and calib.get("nwp_direct"):
        # ── NWP-direct path (AU/NZ) ───────────────────────────────────────────
        # Multi-feature XGBoost if available — directly predicts local ERA5 snow cm
        # from 6 weather features. Falls back to 1D isotonic on snowfall_48h.
        nwp_model = (nwp_direct_models or {}).get(resort_id)
        if nwp_model is not None:
            feats = calib.get("nwp_features", NWP_DIRECT_FEATURES)
            X_nwp = daily.reindex(columns=feats, fill_value=0.0)
            X_nwp.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_nwp.fillna(0.0, inplace=True)
            preds = nwp_model.predict(X_nwp.values).clip(0)
            # XGBoost IS the calibration — no isotonic step needed
        else:
            raw_nwp = daily["snowfall_48h"].clip(0).values
            preds = np.where(raw_nwp < 0.3, 0.0, raw_nwp)
            preds = _apply_calibration(preds, calib)  # 1D isotonic fallback
    else:
        # ── Japan model path (NH + Andes) ─────────────────────────────────────
        X = daily.reindex(columns=feat_cols, fill_value=0)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        non_hpa = [c for c in X.columns if c not in _HPA_COLS_SET]
        X[non_hpa] = X[non_hpa].fillna(0.0)
        raw = model.predict(X)
        preds = np.expm1(raw).clip(0) if log_transform else raw.clip(0)
        if snow_classifier is not None:
            p_snow = snow_classifier.predict_proba(X)[:, 1]
            preds = preds * (p_snow >= snow_gate_threshold)
        # Post-hoc Japan correction: stretches the compressed prediction range so
        # big events (>20cm) are no longer systematically underpredicted.
        if japan_correction is not None:
            iso_x = np.array(japan_correction["iso_x"])
            iso_y = np.array(japan_correction["iso_y"])
            preds = np.interp(preds, iso_x, iso_y).clip(0)
        # Andes SH isotonic calibration (maps corrected Japan output → local ERA5 cm)
        if calib:
            preds = _apply_calibration(preds, calib)

    # ── Physical gate: zero out predictions when too warm to snow ─────────────
    # Applied AFTER calibration so the gate's zeros aren't remapped by the
    # isotonic curve.
    snow_possible = daily["temp_min"] <= 2.0
    preds = preds * snow_possible.values.astype(float)

    results = []
    for i, (date_idx, row) in enumerate(daily.iterrows()):
        pred_cm = float(preds[i])
        score   = powder_score(
            pred_cm,
            float(row.get("snow_temp_mean",        0)),
            float(row.get("wind_during_snow_mean", 0)),
        )
        condition = estimate_condition(
            pred_snow_cm=pred_cm,
            snowfall_72h=float(row.get("snowfall_72h", 0)),
            temp_min=float(row.get("temp_min", 0)),
            temp_max=float(row.get("temp_max", 0)),
            wind_max=float(row.get("wind_max", 0)),
            month=date_idx.month,
            hemisphere=cfg.get("hemisphere", "north"),
        )
        results.append({
            "date":              str(date_idx.date()),
            "predicted_snow_cm": round(pred_cm, 1),
            "powder_score":      score,
            "is_powder_day":     pred_cm >= powder_threshold,
            "powder_threshold":  powder_threshold,
            "condition":         condition,
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

    _CONDITION_STAR = {"powder", "wind_affected"}

    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  {resort_label}")
    print(f"{sep}")
    print(f"  {'Date':<12} {'Snow':>6} {'Score':>6}  {'Temp':>14}  {'Wind':>10}  Condition")
    print(f"  {'':-<12} {'':->6} {'':->6}  {'':->14}  {'':->10}  {'':->14}")

    for d in days:
        snow      = f"{d['predicted_snow_cm']}cm"
        score     = f"{d['powder_score']}/100"
        temp      = f"{d['temp_min_c']} to {d['temp_max_c']}C"
        wind      = f"{d['wind_mean_kmh']}kmh"
        cond      = d.get("condition", "variable")
        star      = " ***" if d["is_powder_day"] else ""
        cond_col  = f"[{cond}]{star}"
        print(f"  {d['date']:<12} {snow:>6} {score:>6}  {temp:>10}  {wind:>10}  {cond_col}")

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
        targets = {rid: cfg for rid, cfg in regions.items() if not cfg.get("hidden")}

    model, feat_cols, log_transform, amp_per_resort, snow_classifier, \
        snow_gate_threshold, japan_correction, japan_powder_threshold = load_model()
    calibrations      = load_calibrations()
    nwp_direct_models = load_nwp_direct_models()
    if calibrations and not args.json:
        print(f"  Loaded transfer calibration for: {', '.join(calibrations)}")
    if nwp_direct_models and not args.json:
        print(f"  Loaded NWP-direct XGBoost models for: {', '.join(nwp_direct_models)}")

    all_results = {}
    for resort_id, cfg in targets.items():
        if not args.json:
            print(f"  Fetching forecast for {resort_id} ...", end="\r")
        all_results[resort_id] = forecast_resort(
            resort_id, cfg, model, feat_cols, calibrations, amp_per_resort,
            days=args.days, log_transform=log_transform,
            snow_classifier=snow_classifier, snow_gate_threshold=snow_gate_threshold,
            nwp_direct_models=nwp_direct_models, japan_correction=japan_correction,
            japan_powder_threshold=japan_powder_threshold,
        )

    if args.json:
        print(json.dumps(all_results, indent=2))
        return

    # ── Formatted output ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  POWDER FORECAST")
    print("=" * 70)

    # Sort resorts: powder days first, then by best score
    def sort_key(item):
        days_list = item[1]
        if days_list and "error" not in days_list[0]:
            powder = sum(1 for d in days_list if d["is_powder_day"])
            best   = max(d["powder_score"] for d in days_list)
            return (-powder, -best)
        return (0, 0)

    for resort_id, days_list in sorted(all_results.items(), key=sort_key):
        cfg = targets[resort_id]
        if not _in_season(cfg):
            resort_label = resort_id.replace("_", " ").title()
            print(f"\n  {resort_label}: Off-season  (season opens {_reopen_str(cfg)})")
            continue
        render_resort(resort_id, days_list)

    print("\n" + "=" * 70)
    thresholds = {r: d[0]["powder_threshold"] for r, d in all_results.items()
                  if d and "error" not in d[0] and "powder_threshold" in d[0]}
    unique_thr = sorted(set(thresholds.values()))
    for thr in unique_thr:
        resorts = [r for r, t in thresholds.items() if t == thr]
        print(f"  Powder threshold >={thr}cm: {', '.join(r.replace('_',' ') for r in resorts)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
