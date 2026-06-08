# CLAUDE.md — Powder Forecasting ML System

## What this project is

An ML system that predicts daily overnight snowfall (cm) for ski resorts worldwide and derives a 0–100 Powder Score. The model is trained on Japan (SnowJapan observed snow depths) and Southern Hemisphere (ERA5 snow_depth change) labels, using Open-Meteo weather reanalysis as features. A transfer calibration step adapts the Japan-trained model to Southern Hemisphere resorts where no labelled training data exists.

**Core ML rules (never violate these):**
- Train on past weather → predict future conditions. Never train on forecast data.
- Labels must be independent of features. SnowJapan depth-change / ERA5 depth-change are independent of Open-Meteo NWP. Never use NWP snowfall as both a feature and a label.
- Season-based holdout only. Adjacent days are nearly identical — random splits cause data leakage.
- Separate fetch from parse. Cache raw responses; re-parse locally as needed.

---

## File structure

```
snow-powder-ML/
├── CLAUDE.md                          # Project guide (this file)
├── README.md                          # Vision, architecture, data pillars, roadmap
├── implementation.md                  # Phase-by-phase build plan
├── regions.yaml                       # Resort configs: lat, lon, elevation, hemisphere, slugs
├── requirements.txt                   # Python dependencies
│
├── collectors/                        # Data-fetching modules
│   ├── __init__.py
│   ├── weather.py                     # Open-Meteo hourly fetch + CSV cache
│   ├── snowjapan.py                   # SnowJapan JSON API → daily depth/new-snow labels
│   └── resort_snapshot.py             # Daily LivePass XML / Alterra JSON snapshot collector
│
├── core/                              # ML pipeline
│   ├── __init__.py
│   ├── features.py                    # Hourly → daily feature engineering (hemisphere-aware)
│   ├── dataset.py                     # Join Japan + SH labels with features → parquet
│   ├── train.py                       # XGBoost Tweedie regression, season holdout
│   ├── evaluate.py                    # Season holdout + leave-one-resort-out metrics
│   └── conditions.py                  # Rule-based surface condition estimator (→ ML once labels accumulate)
│
├── build_nh_features.py               # Build features_{resort_id}.csv for NH/Japan resorts
├── build_sh_features.py               # Build features_{resort_id}.csv for SH resorts
├── build_sh_labels.py                 # Fetch ERA5 snow_depth → sh_labels.csv
├── calibrate_transfer.py              # Isotonic calibration: Japan model → SH resorts
├── forecast.py                        # 7-day inference: fetch forecast → predict → powder score
├── explore.py                         # EDA plots from training_dataset.parquet
├── hindcast_australia.py              # Historical hindcast for AU resorts (sanity check)
├── probe_nz.py                        # One-off URL probe for NZ resort snow report APIs
├── test.py                            # Original prototype pipeline (pre-refactor, kept for ref)
│
└── data/
    ├── models/
    │   ├── xgb_overnight_snow.pkl         # Trained model + feature_cols + metrics + threshold
    │   └── transfer_calibration.json      # Isotonic calibration params per SH resort
    ├── plots/
    │   ├── 01_target_distribution.png     # overnight_snow_cm histogram (explore.py)
    │   ├── 02_feature_correlations.png    # Pearson r vs target (explore.py)
    │   ├── 03_scatter_top_features.png    # Top-8 feature scatters (explore.py)
    │   ├── 04_powder_by_resort.png        # Powder day % per resort (explore.py)
    │   ├── 05_snowfall_by_month.png       # Monthly violin (explore.py)
    │   ├── 06_region_comparison.png       # CDF by region (explore.py)
    │   ├── australia_{resort_id}.png      # AU hindcast bar + scatter (hindcast_australia.py)
    │   └── calibration_{resort_id}.png    # Calibration scatter (calibrate_transfer.py)
    ├── processed/
    │   ├── snowjapan_labels.csv           # Japan labels: date, resort_id, snow_depth_cm, new_snow_cm
    │   ├── sh_labels.csv                  # SH labels: ERA5 snow_depth change per day
    │   ├── training_dataset.parquet       # Final joined dataset (~150k rows)
    │   └── features_{resort_id}.csv       # Per-resort daily feature tables
    └── raw/
        ├── snowjapan/
        │   └── {resort_id}.json           # Cached SnowJapan API responses
        ├── weather/
        │   └── {resort_id}_{start}_{end}.csv  # Hourly Open-Meteo weather cache
        ├── snow_depth/
        │   └── {resort_id}.csv            # Hourly ERA5 snow_depth (SH label source)
        └── snow_reports/
            └── {resort_id}.csv            # Daily resort snapshot CSVs (resort_snapshot.py)
```

---

## Key design decisions

| Decision | Why |
|---|---|
| XGBoost Tweedie (variance_power=1.2) | Target is zero-inflated, right-skewed, positive — Tweedie handles zeros natively without a log transform |
| Season holdout (not random split) | Adjacent days are correlated — random split leaks future into training |
| NWP amplification factor | Open-Meteo systematically underestimates orographic snowfall (Hokkaido: up to 3–4x). Computed per-resort from observed/NWP ratio on snowy days |
| Isotonic calibration for SH resorts | Japan model has little signal in AU/NZ (flat terrain, different climate). Isotonic regression maps raw model output → local expected snowfall |
| NWP-direct path for AU/NZ | Andes resorts have enough elevation/orographic signal for the Japan model; AU/NZ use 48h NWP accumulation directly |
| 850 hPa temperature | Snow/rain line indicator; available from historical-forecast API from 2021-03-23 only — pre-2021 rows get NaN, handled by XGBoost's native NaN routing |
| Physical gate (temp_min > 2°C → zero) | Applied after calibration in forecast.py so the gate's zeros are not remapped by the isotonic curve |

---

## Common workflows

### Full retrain pipeline
```bash
# 1. Fetch Japan labels (cached, resumable)
python -m collectors.snowjapan

# 2. Fetch weather features for Japan resorts (in collectors/weather.py)
# (run per-resort; weather CSVs are auto-cached)

# 3. Build NH and SH features (re-run whenever core/features.py changes)
python build_nh_features.py
python build_sh_labels.py
python build_sh_features.py

# 4. Assemble training dataset
python -m core.dataset

# 5. Train
python -m core.train

# 6. Evaluate
python -m core.evaluate
python -m core.evaluate --loro   # slow: leave-one-resort-out

# 7. Recalibrate SH transfer
python calibrate_transfer.py
```

### Forecast (inference)
```bash
python forecast.py                              # all resorts, 7-day
python forecast.py --resort niseko_grand_hirafu
python forecast.py --json                       # machine-readable JSON
```

### EDA
```bash
python explore.py                  # 6 plots to data/plots/
python hindcast_australia.py       # AU historical sanity check
```

---

## Adding a new resort

1. Add entry to `regions.yaml` with `lat`, `lon`, `elevation`, `region`, `hemisphere`.
2. For Japan resorts: add `snowjapan_id` and `snowjapan_slug`.
3. For Southern Hemisphere resorts: add `snow_report_url` and `snow_report_format` if a LivePass/Alterra endpoint exists.
4. Fetch weather: `collectors/weather.py fetch_and_cache(resort_id, lat, lon, ...)`.
5. If SH: run `build_sh_labels.py --resort {id}` and `build_sh_features.py --resort {id}`.
6. Rebuild dataset (`core.dataset`) and retrain.
7. If SH: rerun `calibrate_transfer.py --resort {id}`.

---

## /docs update requirement

**Every agent that modifies a source file MUST update the corresponding doc in `/docs`.**

### Rule

When you change any file listed below, locate or create its doc file at `docs/{filename}.md` and update it to reflect what the file does, its inputs/outputs, key parameters, and anything non-obvious about how it works. Do this as the final step of every task.

### Doc file format

```markdown
# {filename}

**Purpose:** One sentence on what this file does.

**Inputs:** What it reads (files, APIs, arguments).

**Outputs:** What it writes or returns.

**Key parameters / constants:** Any important tuneable values.

**Notes:** Non-obvious design decisions, known limitations, or gotchas.

**Last updated:** YYYY-MM-DD
```

### Files that require a doc

| Source file | Doc path |
|---|---|
| `collectors/weather.py` | `docs/collectors_weather.md` |
| `collectors/snowjapan.py` | `docs/collectors_snowjapan.md` |
| `collectors/resort_snapshot.py` | `docs/collectors_resort_snapshot.md` |
| `core/features.py` | `docs/core_features.md` |
| `core/dataset.py` | `docs/core_dataset.md` |
| `core/train.py` | `docs/core_train.md` |
| `core/evaluate.py` | `docs/core_evaluate.md` |
| `core/conditions.py` | `docs/core_conditions.md` |
| `build_nh_features.py` | `docs/build_nh_features.md` |
| `build_sh_features.py` | `docs/build_sh_features.md` |
| `build_sh_labels.py` | `docs/build_sh_labels.md` |
| `calibrate_transfer.py` | `docs/calibrate_transfer.md` |
| `forecast.py` | `docs/forecast.md` |
| `explore.py` | `docs/explore.md` |
| `hindcast_australia.py` | `docs/hindcast_australia.md` |
| `regions.yaml` | `docs/regions_yaml.md` |

If you create a new source file, create a corresponding doc in `docs/` and add it to the table above.
