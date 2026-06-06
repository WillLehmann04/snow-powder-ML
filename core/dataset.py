"""
Build the training dataset by joining SnowJapan labels with Open-Meteo features.

Steps:
  1. Load snowjapan_labels.csv (date, resort_id, season, snow_depth_cm, new_snow_cm)
  2. Load per-resort feature CSVs from data/processed/features_{resort_id}.csv
  3. Merge on (resort_id, date)
  4. Add static resort features from regions.yaml (elevation, lat, lon, region, region_code)
  5. Clean labels: drop physically implausible overnight snowfall values
  6. Save to data/processed/training_dataset.parquet

Usage:
  python -m core.dataset
  python -m core.dataset --labels data/processed/snowjapan_labels.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LABELS_PATH   = Path("data/processed/snowjapan_labels.csv")
FEATURES_DIR  = Path("data/processed")
REGIONS_YAML  = Path("regions.yaml")
OUTPUT_PATH   = Path("data/processed/training_dataset.parquet")

REGION_MAP = {
    "hokkaido":        0,
    "nagano":          1,
    "niigata":         2,
    "tohoku":          3,
    "nsw":             4,
    "victoria":        5,
    "nz_south":        6,
    "nz_north":        7,
    "andes_chile":     8,
    "andes_argentina": 9,
}

# Physical plausibility cap on overnight snowfall (cm).
# Japan resort records are ~60-80cm for a single overnight period.
# Values above this are measurement artifacts or reporting errors.
LABEL_CAP_CM = 80.0


def load_regions(path: Path = REGIONS_YAML) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build(
    labels_path: Path = LABELS_PATH,
    features_dir: Path = FEATURES_DIR,
    output_path: Path = OUTPUT_PATH,
    label_cap: float = LABEL_CAP_CM,
) -> pd.DataFrame:

    print("Loading labels ...")
    labels = pd.read_csv(labels_path, parse_dates=["date"])
    # Rename to canonical target column name
    labels = labels.rename(columns={"new_snow_cm": "overnight_snow_cm"})
    print(f"  {len(labels):,} label rows, {labels['resort_id'].nunique()} resorts")

    # ── Clean labels ──────────────────────────────────────────────────────────
    before = len(labels)
    labels = labels[labels["overnight_snow_cm"] <= label_cap].copy()
    dropped = before - len(labels)
    if dropped:
        print(f"  Dropped {dropped} rows with overnight_snow_cm > {label_cap}cm (artifacts)")

    print("\nJoining with weather features ...")
    regions = load_regions()
    frames  = []

    for resort_id in labels["resort_id"].unique():
        feat_path = features_dir / f"features_{resort_id}.csv"
        if not feat_path.exists():
            print(f"  [{resort_id}] no features CSV, skipping")
            continue

        feat = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        feat.index.name = "date"
        feat = feat.reset_index()
        feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()

        lab = labels[labels["resort_id"] == resort_id].copy()
        lab["date"] = lab["date"].dt.normalize()

        merged = lab.merge(feat, on="date", how="inner")
        if merged.empty:
            print(f"  [{resort_id}] no matching dates after join, skipping")
            continue

        # Add static resort features
        cfg = regions.get(resort_id, {})
        merged["elevation"]   = cfg.get("elevation", np.nan)
        merged["lat"]         = cfg.get("lat",       np.nan)
        merged["lon"]         = cfg.get("lon",       np.nan)
        merged["region"]      = cfg.get("region",    "unknown")
        merged["hemisphere"]  = cfg.get("hemisphere","north")
        merged["region_code"] = REGION_MAP.get(cfg.get("region", ""), -1)

        frames.append(merged)
        print(f"  [{resort_id}] {len(merged):,} rows")

    df = pd.concat(frames, ignore_index=True)

    # Drop duplicate season_year columns from the merge (labels had none, features had one)
    if "season_year" in df.columns:
        df = df.rename(columns={"season_year": "season_year_feat"})

    # 850hPa features are only available from 2021-03-23 (historical-forecast API
    # limit). Pre-2021 rows stay as NaN — XGBoost's native NaN routing handles this
    # by learning which branch to follow for missing values, which is more honest
    # than forcing imputed means onto 64% of the training data.
    hpa_cols = [c for c in df.columns if "850" in c]
    if hpa_cols:
        n_missing = df[hpa_cols].isna().any(axis=1).sum()
        if n_missing > 0:
            print(f"  850hPa features: {n_missing:,} rows with NaN (pre-2021, handled by XGBoost)")

    # ── Lagged snowpack depth + lagged actual new snow ────────────────────────
    df = df.sort_values(["resort_id", "date"]).reset_index(drop=True)
    df["snow_depth_lag1"]       = df.groupby("resort_id")["snow_depth_cm"].shift(1)
    df["overnight_snow_lag1"]   = df.groupby("resort_id")["overnight_snow_cm"].shift(1)
    df["overnight_snow_lag2"]   = df.groupby("resort_id")["overnight_snow_cm"].shift(2)
    df["overnight_snow_3d_sum"] = (
        df["overnight_snow_lag1"].fillna(0) + df["overnight_snow_lag2"].fillna(0)
    )
    print(f"  Added snow lags (depth_lag1, overnight_lag1, overnight_lag2, 3d_sum)")

    # ── NWP amplification factor (per resort, computed from all available rows) ──
    # Open-Meteo NWP snowfall systematically underestimates orographic events,
    # especially in Hokkaido (observed up to 70cm vs NWP max ~20cm).
    # Time-invariant climatological property → not leakage.
    snowy = df[(df["overnight_snow_cm"] > 0) & (df["snowfall_24h"] > 0)].copy()
    amp   = (
        snowy.groupby("resort_id")
        .apply(lambda g: g["overnight_snow_cm"].mean() / g["snowfall_24h"].mean(), include_groups=False)
        .rename("nwp_amplification")
    )
    df["nwp_amplification"] = df["resort_id"].map(amp).fillna(1.0)

    # Explicit amplified snowfall features — multiplicative interactions that
    # XGBoost won't reliably discover on its own from separate features.
    df["amplified_snowfall_24h"] = df["snowfall_24h"] * df["nwp_amplification"]
    df["amplified_snowfall_48h"] = df["snowfall_48h"] * df["nwp_amplification"]

    print(f"  Added nwp_amplification + amplified_snowfall_24h/48h (per resort)")
    for rid, val in amp.sort_values(ascending=False).items():
        print(f"    {rid:<28} {val:.2f}x")

    # Sort
    df = df.sort_values(["resort_id", "date"]).reset_index(drop=True)

    print(f"\nFinal dataset: {len(df):,} rows, {df.shape[1]} columns")
    print(f"Date range:    {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Resorts:       {sorted(df['resort_id'].unique())}")
    print(f"Seasons:       {sorted(df['season'].unique())}")

    target = "overnight_snow_cm"
    print(f"\nTarget ({target}):")
    print(f"  mean={df[target].mean():.2f}  std={df[target].std():.2f}  max={df[target].max():.1f}")
    print(f"  zero days:     {(df[target] == 0).sum():,} ({100*(df[target]==0).mean():.1f}%)")
    print(f"  powder (>=15): {(df[target] >= 15).sum():,} ({100*(df[target]>=15).mean():.1f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",  default=str(LABELS_PATH))
    parser.add_argument("--output",  default=str(OUTPUT_PATH))
    parser.add_argument("--cap",     type=float, default=LABEL_CAP_CM,
                        help=f"Drop rows with overnight_snow_cm above this (default {LABEL_CAP_CM})")
    args = parser.parse_args()
    build(Path(args.labels), FEATURES_DIR, Path(args.output), args.cap)


if __name__ == "__main__":
    main()
