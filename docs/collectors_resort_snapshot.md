# collectors/resort_snapshot.py

**Purpose:** Fetches daily snow-report snapshots from AU/NZ resort websites and appends them to per-resort CSVs, building up labeled training data for the condition predictor over time.

**Inputs:**
- `regions.yaml` — resort config; resorts with `snow_report_url` and `snow_report_format` are scraped
- Live HTTP requests to each resort's snow report endpoint (HTML, XML, or JSON depending on format)
- CLI: `--resort`, `--dry-run`, `--list`, `--show`

**Outputs:**
- `data/raw/snow_reports/{resort_id}.csv` — append-only daily rows with 21 columns:
  `date, resort_id, snow_24h_cm, snow_48h_cm, snow_72h_cm, snow_7d_cm, base_cm, season_cm, temp_now_c, temp_min_c, temp_max_c, wind_speed_kmh, wind_direction, visibility, precipitation, conditions_text, weather_top, weather_village, bureau_forecast, groomer_comments, condition_label`

**Supported `snow_report_format` values:**

| Format | Parser | Used by |
|---|---|---|
| `livepass_xml` | `_parse_livepass_xml` | Thredbo (Vail Resorts LivePass XML schema) |
| `falls_creek_json` | `_parse_falls_creek_json` | Falls Creek (WordPress JSON endpoint) |
| `perisher_html` | `_parse_perisher_html` | Perisher (Joomla server-rendered HTML) |
| `whakapapa_lit` | `_parse_whakapapa_lit` | Whakapapa (Lit SSR web components HTML) |

**Key parameters / constants:**
- `CSV_COLS` — 21-column schema; rows are always reindexed to this schema before appending
- `_CONDITION_RULES` — priority-ordered list of `(label, [keywords])` tuples used by `classify_condition()`
- `TIMEOUT = 30` seconds for HTTP requests
- Falls Creek JSON URL is year-specific: `FCSnowReport_{YYYY}.json` — update in `regions.yaml` each season

**Notes:**
- Run once daily during ski season via cron (June–October SH, November–March NH)
- `save_snapshot()` deduplicates by date — safe to re-run on the same day
- NZSki resorts (Coronet Peak, Remarkables, Mt Hutt) and Cardrona use JS-rendered pages with no public API; they are intentionally excluded from `regions.yaml` scraping
- `classify_condition(text)` is also used by `scripts/backfill_wayback.py` for historical archives
- Condition labels match the taxonomy in `core/conditions.py` for consistency

**Last updated:** 2026-06-07
