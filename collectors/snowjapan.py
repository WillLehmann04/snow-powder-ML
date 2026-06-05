"""
SnowJapan snow depth collector.

SnowJapan is a React SPA backed by a JSON REST API at:
  https://www.snowjapan.com/rest-api

The key endpoint is:
  POST /skiarea/snowfall/{snowjapan_id}

It returns a list of daily snow-depth records going back to Dec 2014 for each
resort. Records with SnowDepth == 999 mean "no data / resort closed".

Fields returned per day:
  Season                    e.g. "2023-2024"
  WeatherYear/Month/Day     date components
  SnowDepth                 total base snow depth at resort (cm)
  SnowDepthCompareToYesterday  daily change in depth (positive = new snow)
  SnowDepthFormat           human-readable string ("160cm")

Derived label:
  new_snow_cm = max(0, SnowDepthCompareToYesterday)
  This is a proxy for overnight snowfall. Negative values (compaction/melt)
  are floored to 0. It is noisier than a precipitation gauge but is an
  independent on-mountain measurement, not a weather model output.

Usage (CLI):
  python -m collectors.snowjapan                 # fetch + parse all resorts
  python -m collectors.snowjapan --resort niseko_grand_hirafu
  python -m collectors.snowjapan --fetch-only    # download and cache only
  python -m collectors.snowjapan --parse-only    # re-parse from cache
  python -m collectors.snowjapan --force         # re-fetch even if cached
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

BASE_API  = "https://www.snowjapan.com/rest-api"
HEADERS   = {
    "User-Agent":   "powder-ml-research/1.0 (will.lehmann4@gmail.com)",
    "Content-Type": "application/json",
    "Referer":      "https://www.snowjapan.com/",
}
DELAY_SEC = 1.0

RAW_DIR       = Path("data/raw/snowjapan")
PROCESSED_DIR = Path("data/processed")


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_resort(resort_id: str, snowjapan_id: int, force: bool = False) -> Path:
    """
    POST /skiarea/snowfall/{snowjapan_id} and cache the JSON response.
    Cache path: data/raw/snowjapan/{resort_id}.json
    Skips the request if cache exists unless force=True.
    Returns the cache path.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{resort_id}.json"

    if cache_path.exists() and not force:
        print(f"  [{resort_id}] loaded from cache ({cache_path.stat().st_size // 1024} KB)")
        return cache_path

    print(f"  [{resort_id}] fetching (snowjapan_id={snowjapan_id}) ...")
    resp = requests.post(
        f"{BASE_API}/skiarea/snowfall/{snowjapan_id}",
        headers=HEADERS,
        json={},
        timeout=30,
    )
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8")
    n = len(resp.json())
    print(f"  [{resort_id}] {n} records -> {cache_path}")
    time.sleep(DELAY_SEC)
    return cache_path


def fetch_all(resorts: dict, force: bool = False) -> None:
    """Fetch and cache snowfall history for all resorts. Resumable."""
    for resort_id, cfg in resorts.items():
        sj_id = cfg.get("snowjapan_id")
        if sj_id is None:
            print(f"  [{resort_id}] no snowjapan_id configured, skipping")
            continue
        fetch_resort(resort_id, sj_id, force=force)


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_resort(resort_id: str) -> pd.DataFrame:
    """
    Parse a cached JSON file for one resort.
    Returns a DataFrame with columns:
      date, resort_id, snow_depth_cm, new_snow_cm
    Empty DataFrame if no cache or no valid records.
    """
    cache_path = RAW_DIR / f"{resort_id}.json"
    if not cache_path.exists():
        return pd.DataFrame()

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    records = []
    for r in raw:
        depth = r.get("SnowDepth")
        if depth is None or depth == 999:
            continue  # resort closed / no report

        change = r.get("SnowDepthCompareToYesterday", 0) or 0
        new_snow = max(0.0, float(change))  # floor negative (melt/compaction) to 0

        # Build date from components
        try:
            date = pd.Timestamp(
                year=int(r["WeatherYear"]),
                month=int(r["WeatherMonth"]),
                day=int(r["WeatherDay"]),
            )
        except (KeyError, ValueError):
            continue

        records.append({
            "date":          date,
            "resort_id":     resort_id,
            "season":        r.get("Season", ""),
            "snow_depth_cm": float(depth),
            "new_snow_cm":   new_snow,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def parse_all(resorts: dict) -> pd.DataFrame:
    """Parse all cached resort files into a single label DataFrame."""
    frames = []
    for resort_id in resorts:
        df = parse_resort(resort_id)
        if not df.empty:
            frames.append(df)
            print(f"  [{resort_id}] {len(df)} valid days, "
                  f"{df['date'].min().date()} to {df['date'].max().date()}")
        else:
            print(f"  [{resort_id}] no data")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


# ── Validate ──────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """Print a quick sanity check on the parsed label DataFrame."""
    print(f"\nLabel dataset: {len(df):,} rows, {df['resort_id'].nunique()} resorts")
    print(f"Date range:    {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Seasons:       {sorted(df['season'].unique())}")
    print()
    print("Per-resort summary:")
    summary = df.groupby("resort_id").agg(
        days=("date", "count"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        avg_depth=("snow_depth_cm", "mean"),
        avg_new_snow=("new_snow_cm", "mean"),
        powder_days=("new_snow_cm", lambda x: (x >= 15).sum()),
    )
    summary["avg_depth"]    = summary["avg_depth"].round(1)
    summary["avg_new_snow"] = summary["avg_new_snow"].round(2)
    print(summary.to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_regions(path: str = "regions.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SnowJapan JSON API collector")
    parser.add_argument("--resort",     help="Single resort_id key from regions.yaml")
    parser.add_argument("--fetch-only", action="store_true", help="Download only, skip parse")
    parser.add_argument("--parse-only", action="store_true", help="Parse from cache, skip fetch")
    parser.add_argument("--force",      action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    regions = _load_regions()
    resorts = {args.resort: regions[args.resort]} if args.resort else regions

    # ── Fetch ─────────────────────────────────────────────────────────────────
    if not args.parse_only:
        print(f"Fetching {len(resorts)} resort(s) ...")
        fetch_all(resorts, force=args.force)

    # ── Parse ─────────────────────────────────────────────────────────────────
    if not args.fetch_only:
        print(f"\nParsing {len(resorts)} resort(s) ...")
        df = parse_all(resorts)

        if df.empty:
            print("No records parsed.")
            return

        validate(df)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "snowjapan_labels.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
