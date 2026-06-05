# Powder Forecasting ML — Implementation Plan

## The Core Idea

SnowJapan has published daily on-mountain observations for 50+ Japanese resorts since the
2001-02 season: **overnight snowfall, 24h snowfall, base depth top, base depth bottom**.
This is the label source. Open-Meteo weather reanalysis (temp, wind, humidity, precip) is
the feature source. These are independent measurements of the same physical events, so using
one as X and one as Y is **not circular** — the model learns to predict what actually fell on
the mountain from atmospheric conditions aloft.

25 seasons × ~50 resorts × ~120 open days ≈ 150,000 labeled training rows.

**Target variable:** `overnight_snow_cm` (regression).
**Powder Score:** f(predicted snowfall, predicted temp, predicted wind) — derived, not scraped.

---

## Phase 0 — Project Scaffold

Create repo structure and dependencies. Promote `test.py` weather pipeline into proper modules.

**Repo layout:**
```
snow-powder-ML/
├── regions.yaml                    # resort configs: lat, lon, snowjapan_slug, elevation
├── requirements.txt                # add xgboost, scikit-learn, pyyaml, tqdm, lxml
├── collectors/
│   ├── snowjapan.py                # scrape + parse SnowJapan daily reports → label CSV
│   └── weather.py                  # Open-Meteo fetch (promoted from test.py)
├── core/
│   ├── features.py                 # daily feature engineering (promoted from test.py)
│   ├── dataset.py                  # join labels + features → training rows
│   ├── train.py                    # XGBoost fit + season holdout eval
│   └── evaluate.py                 # per-resort metrics, leave-one-resort-out
└── data/
    ├── raw/snowjapan/              # cached HTML per resort per season
    ├── raw/weather/                # cached Open-Meteo hourly CSVs per resort
    └── processed/                  # joined feature-label DataFrames
```

`regions.yaml` entry shape:
```yaml
niseko_annupuri:
  lat: 42.8989
  lon: 140.6989
  elevation: 1308
  snowjapan_slug: niseko-annupuri
  region: hokkaido
  hemisphere: north
```

---

## Phase 1 — SnowJapan Scraper

**Goal:** Cache raw HTML for all resorts × all seasons, parse to a canonical label CSV.
Separate fetch from parse (resumable).

### Step 1a — Discover resorts
Scrape the SnowJapan resort list to get all slugs and build `regions.yaml`.
Target ~50-80 resorts with complete records.

### Step 1b — Fetch historical daily reports
Each resort season URL pattern:
```
https://www.snowjapan.com/japan-ski-resorts/{slug}/daily-snow-summary/{YYYY-YY}
```
e.g. `.../niseko-annupuri/daily-snow-summary/2023-24`

Fetch seasons 2001-02 through 2024-25 for each resort. Cache each page as
`data/raw/snowjapan/{slug}/{season}.html`. Skip if file exists (resumable).
Add 1-2s delay between requests.

### Step 1c — Parse cached HTML
Parse each cached file to extract the daily table. Fields:
- `date` (YYYY-MM-DD)
- `overnight_snow_cm` (new snow since last report)
- `snow_24h_cm` (24h total)
- `base_top_cm` (depth at summit)
- `base_bottom_cm` (depth at base)

Output: `data/processed/snowjapan_labels.csv`
Columns: `date, resort_id, overnight_snow_cm, snow_24h_cm, base_top_cm, base_bottom_cm`

Drop rows where resort reports "closed" or all fields are null.

**Validation:** Histogram of overnight snowfall by resort. Niseko should show regular
10-30cm events. Flag any resort with suspicious flat/zero data.

---

## Phase 2 — Open-Meteo Historical Weather

**Goal:** Fetch hourly weather per resort lat/lon from 2001-present, aggregate to daily features.

Pipeline already exists in `test.py` — promote to `collectors/weather.py` + `core/features.py`
with two fixes:

1. **Season-year fix:** Rolling windows use `season_year = year if month >= 10 else year - 1`,
   not calendar year.
2. **No month filter:** Keep all months in cache; filter to open-season rows during the join.

Cache: `data/raw/weather/{resort_id}_{start}_{end}.csv` (hourly)
Output: `data/processed/features_{resort_id}.csv` (daily aggregated)

---

## Phase 3 — Feature-Label Join

**Goal:** `core/dataset.py` — produce the training DataFrame.

Join on `(resort_id, date)`. Keep only rows where the resort was open (non-null SnowJapan
observation). Drop dates outside Nov 1 – May 31. Add resort-level statics from `regions.yaml`:
`elevation`, `region` (hokkaido/honshu/etc.).

**Target column:** `overnight_snow_cm`. Keep `snow_24h_cm` as alternative.

**Sanity check:** Correlation between `snowfall_24h` (Open-Meteo) and `overnight_snow_cm`
(SnowJapan) should be positive but not > 0.9. If it is, the pipeline is leaking.

Output: `data/processed/training_dataset.parquet` (~150k rows)

---

## Phase 4 — Model

**Goal:** `core/train.py` — XGBoost regression baseline with season holdout.

```
Train: seasons 2001-02 through 2021-22
Test:  seasons 2022-23 through 2024-25
```

Feature set: all columns from `core/features.py` + `elevation`, `region` (encoded),
`day_of_year`.

Metrics:
- MAE (cm) — primary: "off by X cm on average"
- RMSE
- Pearson r (predicted vs actual snowfall)
- Precision/recall for powder days (overnight >= 15cm)

Also run **leave-one-resort-out**: train on all resorts except Niseko, predict Niseko.
If it generalises, the model is learning physics, not resort-specific patterns.

---

## Phase 5 — Powder Score

Derived from model outputs, not trained against:

```python
def powder_score(predicted_snow_cm, snow_temp_mean, wind_during_snow_mean):
    snow_component = min(predicted_snow_cm / 30, 1.0)
    temp_quality   = max(0, 1 - (snow_temp_mean + 5) / 10)
    wind_penalty   = max(0, 1 - wind_during_snow_mean / 40)
    return round(100 * snow_component * 0.6 + 40 * temp_quality * wind_penalty * 0.4)
```

Calibration target: Niseko on a 20cm overnight day → score ~75-80.

---

## Phase 6 — Inference Pipeline

Wire up the 7-day forecast path:
1. Fetch Open-Meteo forecast for each resort
2. Run through same `core/features.py` pipeline
3. Model predicts `overnight_snow_cm` per day
4. Output Powder Score + 7-day outlook per resort

---

## Key Risks

| Risk | Mitigation |
|---|---|
| SnowJapan URL structure differs pre-2010 | Test 2001-05 manually before bulk fetch |
| Resorts report closed days as 0 vs null | Detect "---" / blank; treat as closed, not zero snow |
| Open-Meteo grid vs actual resort elevation | Use `&cell_selection=nearest` + elevation param |
| Label leakage (Open-Meteo snowfall ≈ SnowJapan) | Check correlation; drop `snowfall_24h` from features if > 0.9 |
| SnowJapan scraping ToS | Check ToS; add polite delays; manual fallback if needed |

---

## Build Order

1. **Now:** Phase 0 scaffold + `regions.yaml` seed resorts
2. **Next:** Phase 1 scraper — validate one resort (Niseko) end-to-end before bulk fetch
3. **Then:** Phase 2 weather fetch for seeded resorts
4. **Then:** Phase 3 join → Phase 4 first model → Phase 5 score
5. **Later:** Phase 6 inference, add more resorts/regions (Utah UAC as second region)
