# build_sh_labels.py

**Purpose:** Fetches ERA5 snow_depth reanalysis data and converts daily depth change into a proxy overnight snowfall label. Covers all SH resorts plus any NH resort with `era5_labels: true` in `regions.yaml` (e.g. `whistler_blackcomb`).

**Inputs:**
- `regions.yaml` — resort coordinates; all `hemisphere: south` resorts + any `era5_labels: true` resorts are included
- Open-Meteo ERA5 historical API (live HTTP) — hourly snow_depth cached to `data/raw/snow_depth/{resort_id}.csv`
- CLI args: `--resort`, `--start` (default `2014-05-01`), `--end` (default `2025-10-31`)

**Outputs:**
- `data/processed/sh_labels.csv` — combined daily labels for all covered resorts (name is legacy; covers ERA5-sourced labels regardless of hemisphere)

**Key parameters / constants:**
- `SH_MONTHS = {5,6,7,8,9,10}` — months kept for SH resorts
- `NH_MONTHS = {10,11,12,1,2,3,4,5}` — months kept for NH resorts (e.g. Whistler)
- `_season_label()` is hemisphere-aware: NH uses Oct as season start, SH uses Apr
- `LABEL_CAP_CM = 80.0` — implausible values dropped

**Notes:**
- ERA5 is used (not NWP snowfall) because it is an independent reanalysis — using NWP snowfall as both feature and label is circular.
- To add a new NH resort without SnowJapan coverage, add `era5_labels: true` to its entry in `regions.yaml` and rerun this script.
- Per-resort cache files in `data/raw/snow_depth/` must be deleted to force a re-fetch with a new date range.

**Last updated:** 2026-06-07
