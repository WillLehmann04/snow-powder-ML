# regions.yaml

**Purpose:** Single source of truth for all resort configurations — coordinates, elevation, hemisphere, region, and calibration settings.

**Inputs:** None (static config file).

**Outputs:** Read by `core/dataset.py`, `forecast.py`, `build_sh_features.py`, `build_nh_features.py`, `calibrate_transfer.py`, `build_sh_labels.py`.

**Key parameters / constants:**
Per-resort fields:
- `lat`, `lon` — decimal degrees; used for Open-Meteo API fetches
- `elevation` — metres above sea level; passed to Open-Meteo for lapse-rate correction
- `hemisphere` — `"north"` or `"south"`; controls season_year boundary and cold_air_advection direction in `core/features.py`
- `region` — string identifier (`hokkaido`, `nagano`, `niigata`, `tohoku`, `nsw`, `victoria`, `nz_south`, `nz_north`, `andes_chile`, `andes_argentina`); encoded as integer `region_code` in training data
- `dist_coast_km` — approximate great-circle distance to the nearest ocean coastline (km); used to compute `maritime_influence = 100 / (dist + 10)` in `core/dataset.py`. Values researched geographically for all 31 resorts (Hokkaido coast ~18–28km; inland Hokkaido 110–190km; Niigata/Nagano 58–145km; AU 195–310km; NZ 55–95km; Andes 120–420km).
- `powder_threshold_cm` — local observed powder threshold in cm (NH: 15, SH: 4–5)
- `calibration_method` — (SH only) `"nwp_direct"` (AU/NZ) or `"japan_model"` (Andes)
- `opens_month` / `closes_month` — 1-indexed month integers declaring the resort's operating season. When both are set, `forecast.py` replaces terminal output with an "Off-season" line for months outside the window. Wrap-around seasons (e.g. Nov–Apr for NH resorts) are handled by `opens_month > closes_month`. Japan resorts omit these fields since the physical temp gate already silences summer output.
- `hidden: true` — excludes a resort from the default forecast run. It remains in `regions.yaml` for explicit `--resort` lookups. Currently set for `niseko_annupuri` (label-quality issue).

**Notes:**
- `era5_labels: true` — optional flag for NH resorts that have no SnowJapan equivalent. Causes `build_sh_labels.py` to fetch ERA5 snow_depth for the resort and include it in `sh_labels.csv`. Currently set for `whistler_blackcomb`.
- Adding a new resort: (1) add entry with `dist_coast_km`, (2) if NH without SnowJapan add `era5_labels: true`, (3) fetch weather CSV, (4) run `build_nh_features.py` or `build_sh_features.py`, (5) re-run `build_sh_labels.py` if `era5_labels: true`, (6) rebuild dataset, (7) retrain, (8) recalibrate if SH.
- Do not change `hemisphere` for an existing resort without re-building its feature CSV.
- `dist_coast_km` defaults to 100 in `core/dataset.py` if absent.

**Last updated:** 2026-06-07

