# build_sh_features.py

**Purpose:** Generates daily weather feature CSVs for all Southern Hemisphere resorts by applying `build_features(hemisphere="south")` to raw hourly weather data.

**Inputs:**
- `data/raw/weather/{resort_id}_*.csv` — raw hourly Open-Meteo weather, one file per SH resort
- `regions.yaml` — used to enumerate SH resorts (`hemisphere == "south"`)
- CLI args: `--resort` (single resort override)

**Outputs:**
- `data/processed/features_{resort_id}.csv` — one daily feature CSV per SH resort (overwrites existing)

**Key parameters / constants:**
- Picks the largest (longest date range) weather CSV when multiple files exist for a resort

**Notes:**
- Must be re-run whenever `core/features.py` changes — the feature CSVs are a cache, not a primary artifact.
- The NH equivalent is `build_nh_features.py` (created 2026-06-07 — previously missing from the repo).
- `core/dataset.py` reads these CSVs directly; if a resort's CSV is absent, that resort is silently skipped during dataset build.

**Last updated:** 2026-06-07
