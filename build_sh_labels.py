"""
Build daily snowfall labels for Southern Hemisphere resorts from ERA5 snow_depth.

Why snow_depth change instead of NWP snowfall:
  - snowfall_24h (our current SH proxy) is NWP forecast output — the same model
    we're trying to calibrate against. Using it as a label is circular.
  - snow_depth is a state variable in ERA5 reanalysis. ERA5 assimilates billions of
    actual surface observations, so snow_depth is observationally constrained — much
    closer to ground truth than a pure NWP forecast.
  - Daily snow_depth increase (floored at 0) is exactly how SnowJapan labels work:
    new snow = max(0, today_depth - yesterday_depth).

Usage:
  python build_sh_labels.py
  python build_sh_labels.py --start 2016-05-01 --end 2024-10-31
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
REGIONS_YAML = Path("regions.yaml")
RAW_DIR      = Path("data/raw/snow_depth")
OUT_PATH     = Path("data/processed/sh_labels.csv")

# Southern hemisphere ski season months
SH_MONTHS = {5, 6, 7, 8, 9, 10}

# Physical plausibility cap (same as Japan labels)
LABEL_CAP_CM = 80.0


def _season_label(dt: pd.Timestamp) -> str:
    """SH convention: Apr-Dec → that year, Jan-Mar → year-1. Returns 'YYYY-YYYY+1'."""
    yr = dt.year if dt.month >= 4 else dt.year - 1
    return f"{yr}-{yr + 1}"


def fetch_snow_depth(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly snow_depth (m) from Open-Meteo archive API."""
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude":       lat,
            "longitude":      lon,
            "start_date":     start,
            "end_date":       end,
            "hourly":         "snow_depth",
            "timezone":       "auto",
            "cell_selection": "nearest",
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()
    df  = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def build_resort_labels(resort_id: str, cfg: dict,
                        start: str, end: str) -> pd.DataFrame:
    """Return a daily label DataFrame for one SH resort."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{resort_id}.csv"

    if cache.exists():
        print(f"  [{resort_id}] loading snow_depth from cache")
        hourly = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        print(f"  [{resort_id}] fetching ERA5 snow_depth …")
        hourly = fetch_snow_depth(cfg["lat"], cfg["lon"], start, end)
        hourly.to_csv(cache)
        time.sleep(1.2)

    # Depth in metres → cm, daily maximum (end-of-day reading)
    depth_cm  = (hourly["snow_depth"].clip(lower=0) * 100).resample("D").max()

    # New snow = positive daily change in snow depth (melt/compaction → 0)
    new_snow  = depth_cm.diff().clip(lower=0).fillna(0)

    # Cap implausible values
    new_snow  = new_snow.clip(upper=LABEL_CAP_CM)

    labels = pd.DataFrame({
        "date":         depth_cm.index,
        "resort_id":    resort_id,
        "season":       [_season_label(d) for d in depth_cm.index],
        "snow_depth_cm": depth_cm.values.round(1),
        "new_snow_cm":   new_snow.values.round(1),
    })

    # Keep only ski-season months
    labels = labels[labels["date"].dt.month.isin(SH_MONTHS)].copy()
    labels = labels.dropna(subset=["snow_depth_cm"])
    return labels.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-05-01")
    parser.add_argument("--end",   default="2024-10-31")
    parser.add_argument("--resort", default=None, help="Single resort ID (optional)")
    args = parser.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    sh_resorts = {
        rid: cfg for rid, cfg in regions.items()
        if cfg.get("hemisphere") == "south"
    }
    if args.resort:
        sh_resorts = {args.resort: sh_resorts[args.resort]}

    print(f"Building ERA5 snow_depth labels for {len(sh_resorts)} SH resort(s)")
    print(f"Period: {args.start} to {args.end}\n")

    frames = []
    for resort_id, cfg in sh_resorts.items():
        df = build_resort_labels(resort_id, cfg, args.start, args.end)
        frames.append(df)

        snow_days  = (df["new_snow_cm"] > 0).sum()
        powder     = (df["new_snow_cm"] >= 4).sum()
        mean_new   = df.loc[df["new_snow_cm"] > 0, "new_snow_cm"].mean()
        print(f"    {len(df)} days | snow_days={snow_days} | "
              f"powder(>=4cm)={powder} | mean_new_snow={mean_new:.1f}cm")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["resort_id", "date"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_PATH, index=False)

    print(f"\nSummary:")
    print(f"  Total rows:  {len(df_all):,}")
    print(f"  Resorts:     {sorted(df_all['resort_id'].unique())}")
    print(f"  Seasons:     {sorted(df_all['season'].unique())}")
    print(f"  Powder days: {(df_all['new_snow_cm'] >= 4).sum():,} "
          f"({100*(df_all['new_snow_cm'] >= 4).mean():.1f}%)")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
