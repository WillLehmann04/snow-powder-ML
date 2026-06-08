"""
Detect and re-fetch weather CSVs that are missing variables added to HOURLY_VARS
after the original cache was written (dewpoint_2m, pressure_msl, wind_direction_10m).

Usage:
  python scripts/refresh_weather.py           # dry-run: list stale CSVs
  python scripts/refresh_weather.py --fetch   # actually re-fetch
  python scripts/refresh_weather.py --resort niseko_grand_hirafu --fetch
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from collectors.weather import fetch_hourly, RAW_DIR

REQUIRED_COLS = {"dewpoint_2m", "pressure_msl", "wind_direction_10m"}
REGIONS_YAML  = Path("regions.yaml")


def find_stale(resort_id: str | None = None) -> list[tuple[Path, set]]:
    """Return list of (path, missing_columns) for CSVs missing required columns."""
    candidates = sorted(RAW_DIR.glob("*.csv"))
    if resort_id:
        candidates = [p for p in candidates if p.name.startswith(resort_id + "_")]

    stale = []
    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=1)
            missing = REQUIRED_COLS - set(header.columns)
            if missing:
                stale.append((path, missing))
        except Exception as e:
            print(f"  WARNING: could not read {path.name}: {e}")
    return stale


def main():
    ap = argparse.ArgumentParser(
        description="Detect and re-fetch stale Open-Meteo weather CSVs"
    )
    ap.add_argument("--fetch",  action="store_true",
                    help="Actually re-fetch (default: dry-run, print only)")
    ap.add_argument("--resort", default=None,
                    help="Single resort_id to check/refresh")
    args = ap.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    stale = find_stale(resort_id=args.resort)

    if not stale:
        print("All weather CSVs have current columns. Nothing to refresh.")
        return

    label = "DRY RUN — " if not args.fetch else ""
    print(f"{label}Found {len(stale)} stale CSV(s):\n")
    for path, missing in stale:
        print(f"  {path.name}  (missing: {', '.join(sorted(missing))})")

    if not args.fetch:
        print("\nRe-run with --fetch to download updated CSVs.")
        return

    print()
    for path, _ in stale:
        # Filename pattern: {resort_id}_{YYYY-MM-DD}_{YYYY-MM-DD}.csv
        # Date strings are always 10 chars (YYYY-MM-DD), separated by _
        stem_parts = path.stem.split("_")
        end   = stem_parts[-1]
        start = stem_parts[-2]
        resort_id = "_".join(stem_parts[:-2])

        cfg = regions.get(resort_id)
        if cfg is None:
            print(f"  [{resort_id}] not in regions.yaml, skipping")
            continue

        print(f"  [{resort_id}] re-fetching {start} -> {end} ...")
        path.unlink()

        try:
            df = fetch_hourly(cfg["lat"], cfg["lon"], start, end)
            out = RAW_DIR / f"{resort_id}_{start}_{end}.csv"
            df.to_csv(out)
            print(f"  [{resort_id}] {len(df):,} rows -> {out.name}")
            time.sleep(1.5)
        except Exception as e:
            print(f"  [{resort_id}] ERROR: {e}")

    print("\nDone. Re-run build_sh_features.py / core.dataset / core.train to rebuild the pipeline.")


if __name__ == "__main__":
    main()
