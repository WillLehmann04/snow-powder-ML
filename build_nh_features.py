"""
Build feature CSVs for all Northern Hemisphere (Japan) resorts.

Mirrors build_sh_features.py: loads the raw hourly weather CSV,
calls build_features(hemisphere="north"), and saves the result to
data/processed/features_{resort_id}.csv.

Usage:
  python build_nh_features.py
  python build_nh_features.py --resort niseko_grand_hirafu
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from core.features import build_features

REGIONS_YAML = Path("regions.yaml")
RAW_DIR      = Path("data/raw/weather")
OUT_DIR      = Path("data/processed")


def build_nh_resort(resort_id: str) -> bool:
    # Prefer the largest CSV (longest date range) if multiple exist
    candidates = sorted(RAW_DIR.glob(f"{resort_id}_*.csv"),
                        key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        print(f"  [{resort_id}] no raw weather CSV found, skipping")
        return False

    raw_path = candidates[0]
    print(f"  [{resort_id}] loading {raw_path.name} ...")

    hourly = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if hourly.empty:
        print(f"  [{resort_id}] empty CSV, skipping")
        return False

    daily = build_features(hourly, hemisphere="north")

    out_path = OUT_DIR / f"features_{resort_id}.csv"
    daily.to_csv(out_path)
    print(f"  [{resort_id}] {len(daily)} days -> {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resort", default=None)
    args = parser.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    nh_resorts = {
        rid: cfg for rid, cfg in regions.items()
        if cfg.get("hemisphere", "north") == "north"
    }
    if args.resort:
        nh_resorts = {args.resort: nh_resorts[args.resort]}

    print(f"Building features for {len(nh_resorts)} NH resort(s) ...\n")
    ok = 0
    for resort_id in nh_resorts:
        if build_nh_resort(resort_id):
            ok += 1

    print(f"\nDone: {ok}/{len(nh_resorts)} resorts built.")


if __name__ == "__main__":
    main()
