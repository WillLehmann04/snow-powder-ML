# collectors/snowjapan.py

**Purpose:** Fetches and caches daily snow-depth history from the SnowJapan REST API for all Japan resorts, producing the primary training labels for the NH model.

**Inputs:**
- `regions.yaml` — resorts with `snowjapan_id` and `snowjapan_slug` fields are fetched
- SnowJapan REST API: `POST /rest-api/skiarea/snowfall/{snowjapan_id}` — returns daily depth records back to Dec 2014
- CLI: `--resort`, `--fetch-only`, `--parse-only`, `--force`

**Outputs:**
- `data/raw/snowjapan/{resort_id}.json` — cached raw API response (used by `--parse-only` to re-derive labels without re-fetching)
- `data/processed/snowjapan_labels.csv` — columns: `date, resort_id, snow_depth_cm, new_snow_cm`

**Key parameters / constants:**
- `DELAY_SEC = 1.0` — polite delay between requests
- `new_snow_cm = max(0, SnowDepthCompareToYesterday)` — daily depth change, floored at 0; used as the training label
- Records with `SnowDepth == 999` are dropped (= resort closed / no sensor data)

**Notes:**
- `new_snow_cm` is derived from on-mountain depth sensors, not a weather model — it is independent of Open-Meteo NWP features, satisfying the core ML rule that labels must be independent of features.
- Negative depth changes (compaction, settlement, melt) are floored to 0; they represent the base settling, not real negative snowfall.
- Re-run with `--force` after SnowJapan updates historical records for a season (they sometimes backfill corrections after the season ends).

**Last updated:** 2026-06-07
