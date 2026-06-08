"""
Build daily snowfall labels from ERA5 snow_depth for resorts without SnowJapan data.

Handles both Southern Hemisphere resorts (the original use case) and any NH resort
configured with era5_labels: true in regions.yaml (e.g. whistler_blackcomb).

Why snow_depth change instead of NWP snowfall:
  - snowfall_24h is NWP forecast output — the same model we're calibrating against.
    Using it as a label is circular.
  - snow_depth is a state variable in ERA5 reanalysis, observationally constrained.
  - Daily snow_depth increase (floored at 0) mirrors the SnowJapan methodology.

Usage:
  python build_sh_labels.py
  python build_sh_labels.py --start 2014-05-01 --end 2025-10-31
  python build_sh_labels.py --resort whistler_blackcomb   # single resort
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

# Southern hemisphere ski season months (May–Oct)
SH_MONTHS = {5, 6, 7, 8, 9, 10}
# Northern hemisphere ski season months (Oct–May)
NH_MONTHS = {10, 11, 12, 1, 2, 3, 4, 5}

# Physical plausibility cap (same as Japan labels)
LABEL_CAP_CM = 80.0


def _season_label(dt: pd.Timestamp, hemisphere: str = "south") -> str:
    """Return 'YYYY-YYYY+1' season string, hemisphere-aware."""
    if hemisphere == "south":
        yr = dt.year if dt.month >= 4 else dt.year - 1
    else:
        yr = dt.year if dt.month >= 10 else dt.year - 1
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
    """Return a daily label DataFrame for one resort (SH or NH with era5_labels)."""
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

    hemisphere = cfg.get("hemisphere", "south")
    season_months = NH_MONTHS if hemisphere == "north" else SH_MONTHS

    # Depth in metres → cm, daily maximum (end-of-day reading)
    depth_cm  = (hourly["snow_depth"].clip(lower=0) * 100).resample("D").max()

    # New snow = positive daily change in snow depth (melt/compaction → 0)
    new_snow  = depth_cm.diff().clip(lower=0).fillna(0)
    new_snow  = new_snow.clip(upper=LABEL_CAP_CM)

    labels = pd.DataFrame({
        "date":          depth_cm.index,
        "resort_id":     resort_id,
        "season":        [_season_label(d, hemisphere) for d in depth_cm.index],
        "snow_depth_cm": depth_cm.values.round(1),
        "new_snow_cm":   new_snow.values.round(1),
    })

    labels = labels[labels["date"].dt.month.isin(season_months)].copy()
    labels = labels.dropna(subset=["snow_depth_cm"])
    return labels.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2014-05-01")
    parser.add_argument("--end",    default="2025-10-31")
    parser.add_argument("--resort", default=None,
                        help="Single resort ID (any hemisphere, must be in regions.yaml)")
    args = parser.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    # Include SH resorts + any NH resort flagged with era5_labels: true
    era5_resorts = {
        rid: cfg for rid, cfg in regions.items()
        if cfg.get("hemisphere") == "south" or cfg.get("era5_labels") is True
    }
    if args.resort:
        if args.resort not in regions:
            raise SystemExit(f"Resort '{args.resort}' not in regions.yaml")
        era5_resorts = {args.resort: regions[args.resort]}

    print(f"Building ERA5 snow_depth labels for {len(era5_resorts)} resort(s)")
    print(f"Period: {args.start} to {args.end}\n")
    sh_resorts = era5_resorts  # keep variable name for rest of function

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
