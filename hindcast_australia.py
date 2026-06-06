"""
Southern hemisphere hindcast — run the Japan-trained model on Australian ski resorts
and evaluate against Open-Meteo's own snowfall variable as a proxy label.

NOTE on the proxy label:
  Open-Meteo `snowfall_24h` is also used as a *feature* in the model, so correlation
  against it is not a true independent evaluation. It gives a rough sanity check only.
  For a real r value you would need on-mountain observations (e.g. Thredbo daily reports).

Usage:
  python hindcast_australia.py
  python hindcast_australia.py --resort thredbo
  python hindcast_australia.py --seasons 2022 2023 2024
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from collectors.weather import fetch_and_cache
from core.features import build_features

MODEL_PATH   = Path("data/models/xgb_overnight_snow.pkl")
REGIONS_YAML = Path("regions.yaml")
PLOTS_DIR    = Path("data/plots")

# Southern hemisphere ski season
SH_SEASON_MONTHS = {6, 7, 8, 9}
REGION_MAP       = {"hokkaido": 0, "nagano": 1, "niigata": 2, "tohoku": 3,
                    "nsw": 4, "victoria": 5}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        p = pickle.load(f)
    return p["model"], p["feature_cols"], p.get("log_transform", False)


def forecast_resort_historical(resort_id: str, cfg: dict, model, feat_cols: list,
                                start: str, end: str, log_transform: bool = False) -> pd.DataFrame:
    """Fetch historical weather, run model, return daily DataFrame."""
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

    raw = model.predict(X)
    preds = np.expm1(raw).clip(0) if log_transform else raw.clip(0)

    # Physical gate: no snow if temp_min > 2C
    snow_possible = daily["temp_min"] <= 2.0
    preds = preds * snow_possible.values.astype(float)

    daily["predicted_snow_cm"] = preds
    # Open-Meteo's own snowfall in mm → convert to cm
    daily["ometo_snow_cm"] = daily["snowfall_24h"]   # already in cm (mm converted in features)

    # Filter to ski season months only
    daily = daily[daily.index.month.isin(SH_SEASON_MONTHS)].copy()
    return daily


def evaluate(df: pd.DataFrame, resort_id: str) -> dict:
    """Compute stats vs Open-Meteo proxy snowfall."""
    pred  = df["predicted_snow_cm"].values
    proxy = df["ometo_snow_cm"].values

    # Only on days where Open-Meteo says there was at least some snowfall (> 0.1cm)
    snow_days_mask = proxy > 0.1
    n_snow = snow_days_mask.sum()

    if n_snow < 10:
        return {"resort_id": resort_id, "n_total": len(df), "n_snow_days": int(n_snow),
                "note": "too few snow days for evaluation"}

    r, pval = pearsonr(pred[snow_days_mask], proxy[snow_days_mask])
    mae     = mean_absolute_error(proxy[snow_days_mask], pred[snow_days_mask])

    return {
        "resort_id":        resort_id,
        "n_total":          len(df),
        "n_snow_days":      int(n_snow),
        "r_vs_ometo_proxy": round(r, 3),
        "p_value":          round(pval, 4),
        "mae_vs_proxy":     round(mae, 2),
        "mean_pred_cm":     round(float(pred.mean()), 2),
        "mean_proxy_cm":    round(float(proxy.mean()), 2),
        "powder_days_pred": int((pred >= 7).sum()),
        "powder_days_proxy": int((proxy >= 7).sum()),
    }


def plot_season(df: pd.DataFrame, resort_id: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{resort_id.replace('_',' ').title()} — Predicted vs Open-Meteo Snowfall",
                 fontsize=13, fontweight="bold")

    axes[0].bar(df.index, df["ometo_snow_cm"],  color="#7eb8d4", label="Open-Meteo snowfall (proxy)", alpha=0.7)
    axes[0].bar(df.index, df["predicted_snow_cm"], color="#e07b54", label="Model prediction", alpha=0.7)
    axes[0].set_ylabel("Snowfall (cm)")
    axes[0].set_title("Daily Snowfall — Ski Season Months Only")
    axes[0].legend(fontsize=9)

    # Scatter: predicted vs proxy on snow days
    mask = df["ometo_snow_cm"] > 0.1
    if mask.sum() > 5:
        axes[1].scatter(df.loc[mask, "ometo_snow_cm"], df.loc[mask, "predicted_snow_cm"],
                        alpha=0.4, s=12, color="#4c8cbf")
        max_val = max(df["ometo_snow_cm"].max(), df["predicted_snow_cm"].max())
        axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=1, label="1:1 line")
        axes[1].set_xlabel("Open-Meteo snowfall (proxy label, cm)")
        axes[1].set_ylabel("Model predicted (cm)")
        axes[1].set_title("Predicted vs Proxy (snow days only)")
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / f"australia_{resort_id}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resort",  default=None, help="Single resort ID")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024],
                        help="Southern hemisphere seasons to evaluate (year of season)")
    args = parser.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    # Filter to southern hemisphere resorts
    sh_resorts = {rid: cfg for rid, cfg in regions.items()
                  if cfg.get("hemisphere") == "south"}

    if args.resort:
        if args.resort not in sh_resorts:
            print(f"Resort '{args.resort}' not found or not southern hemisphere.")
            print(f"Available: {', '.join(sh_resorts)}")
            return
        sh_resorts = {args.resort: sh_resorts[args.resort]}

    model, feat_cols, log_transform = load_model()

    # Build date range covering requested seasons
    # Southern hemisphere: season is Jun–Sep, so season 2023 = Jun–Sep 2023
    years = sorted(set(args.seasons))
    start = f"{min(years)}-05-01"
    end   = f"{max(years)}-10-31"

    print(f"\nRunning hindcast for {len(sh_resorts)} Australian resort(s)")
    print(f"Date range: {start} → {end}")
    print(f"Seasons:    {years}\n")
    print("NOTE: correlation is vs Open-Meteo's own snowfall — not a true independent label.")
    print("      r values will be somewhat inflated due to shared data source.\n")

    results = []
    for resort_id, cfg in sh_resorts.items():
        print(f"  Processing {resort_id} …")
        df = forecast_resort_historical(resort_id, cfg, model, feat_cols, start, end, log_transform)
        plot_season(df, resort_id)
        result = evaluate(df, resort_id)
        results.append(result)

        if "note" in result:
            print(f"  {resort_id}: {result['note']}")
            continue

        print(f"  {resort_id.replace('_',' ').title()}:")
        print(f"    Season days (Jun–Sep):   {result['n_total']}")
        print(f"    Days with snow (proxy):  {result['n_snow_days']}")
        print(f"    r vs Open-Meteo proxy:   {result['r_vs_ometo_proxy']}  (p={result['p_value']})")
        print(f"    MAE vs proxy:            {result['mae_vs_proxy']} cm")
        print(f"    Mean predicted:          {result['mean_pred_cm']} cm/day")
        print(f"    Mean proxy:              {result['mean_proxy_cm']} cm/day")
        print(f"    Powder days predicted:   {result['powder_days_pred']}")
        print(f"    Powder days (proxy):     {result['powder_days_proxy']}")
        print()

    print("─" * 60)
    print("SUMMARY")
    print("─" * 60)
    for r in results:
        if "note" in r:
            print(f"  {r['resort_id']:20s}  {r['note']}")
        else:
            print(f"  {r['resort_id']:20s}  r={r['r_vs_ometo_proxy']:.3f}  "
                  f"MAE={r['mae_vs_proxy']:.1f}cm  "
                  f"powder days: {r['powder_days_pred']}/{r['powder_days_proxy']} (pred/proxy)")
    print()
    print("Plots saved to data/plots/australia_*.png")
    print()
    print("For a genuine r value, you need actual on-mountain observations.")
    print("Sources to consider:")
    print("  - Snowy Hydro snowfall records (Thredbo/Perisher)")
    print("  - Bureau of Meteorology (BOM) station data")
    print("  - Resort daily snow reports (thredbo.com.au/snow-report)")


if __name__ == "__main__":
    main()
