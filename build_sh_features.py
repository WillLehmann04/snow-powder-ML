"""
Build feature CSVs for all Southern Hemisphere resorts.

Mirrors what is done for Japan resorts: loads the raw hourly weather CSV,
calls build_features(hemisphere="south"), and saves the result to
data/processed/features_{resort_id}.csv.

Usage:
  python build_sh_features.py
  python build_sh_features.py --resort coronet_peak
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from core.features import build_features

REGIONS_YAML = Path("regions.yaml")
RAW_DIR      = Path("data/raw/weather")
OUT_DIR      = Path("data/processed")


def build_sh_resort(resort_id: str, cfg: dict) -> bool:
    # Find the raw weather CSV — prefer longest date range
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

    daily = build_features(hourly, hemisphere="south")

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

    sh_resorts = {
        rid: cfg for rid, cfg in regions.items()
        if cfg.get("hemisphere") == "south"
    }
    if args.resort:
        sh_resorts = {args.resort: sh_resorts[args.resort]}

    print(f"Building features for {len(sh_resorts)} SH resort(s) ...\n")
    ok = 0
    for resort_id, cfg in sh_resorts.items():
        if build_sh_resort(resort_id, cfg):
            ok += 1

    print(f"\nDone: {ok}/{len(sh_resorts)} resorts built.")


if __name__ == "__main__":
    main()
