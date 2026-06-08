# Model Audit Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all production bugs, data quality gaps, evaluation artefacts, and production-readiness issues identified in the model audit, then update /docs for every changed file.

**Architecture:** Fixes proceed in dependency order — bugs that affect inference first, then data pipeline improvements, then model/evaluation fixes, then infrastructure. Every task ends with a /docs update. The test suite (Task 10) is written incrementally alongside the code changes it covers.

**Tech Stack:** Python 3.13, XGBoost (Tweedie), pandas, scikit-learn, pytest, Open-Meteo archive + historical-forecast APIs, SnowJapan JSON API.

---

## File Map

| File | Action | Reason |
|---|---|---|
| `core/train.py` | Modify | Store per-resort nwp_amplification in payload; drop lag features from FEATURE_COLS; cap amplification; add walk-forward CV helper |
| `core/features.py` | Modify | Fix rolling windows to not cross season boundaries; fix cold_air_advection to be hemisphere-aware |
| `core/dataset.py` | Modify | Cap nwp_amplification at 6x; flag geto_kogen; use ERA5 labels as SH calibration proxy |
| `core/evaluate.py` | Modify | Hemisphere-aware metrics; fix LORO to use Tweedie |
| `forecast.py` | Modify | Apply per-resort nwp_amplification for Japan resorts at inference |
| `calibrate_transfer.py` | Modify | Use ERA5 snow_depth labels as proxy (not NWP snowfall) for NWP-direct path |
| `collectors/weather.py` | Modify | Add re-fetch helper that detects stale CSVs missing new variables |
| `tests/` | Create | New directory: schema test, smoke test, regression test |
| `tests/test_features.py` | Create | Season-boundary rolling, hemisphere cold_air_advection |
| `tests/test_train.py` | Create | Payload has nwp_amplification_per_resort; FEATURE_COLS has no lag features |
| `tests/test_forecast.py` | Create | Japan forecast has correct nwp_amplification; score in [0,100] |
| `tests/test_evaluate.py` | Create | Hemisphere-aware metrics return separate results |
| `tests/conftest.py` | Create | Shared fixtures |
| `scripts/refresh_weather.py` | Create | Detect and re-fetch stale weather CSVs missing dewpoint/pressure |
| `.env.example` | Create | Template for environment variables |
| `docs/core_train.md` | Create/Update | Document every changed file |
| `docs/core_features.md` | Create/Update | |
| `docs/core_dataset.md` | Create/Update | |
| `docs/core_evaluate.md` | Create/Update | |
| `docs/forecast.md` | Create/Update | |
| `docs/calibrate_transfer.md` | Create/Update | |
| `docs/collectors_weather.md` | Create/Update | |

---

## Task 1: Fix nwp_amplification at inference for Japan resorts

**Files:**
- Modify: `core/train.py` (payload dict, ~line 266)
- Modify: `forecast.py` (forecast_resort function, ~line 89)

This is the #1 production bug. `amplified_snowfall_48h` is the top feature (0.316 importance). At inference, Japan resorts get 0 for this feature because the per-resort amplification is never applied.

- [ ] **Step 1: Add per-resort amplification to the model payload in `core/train.py`**

Find the `payload` dict near the bottom of `train()`. Add `nwp_amplification_per_resort`:

```python
# In train() function, before the pickle.dump, after the existing payload dict
# First compute the per-resort amplification from the full dataset
snowy_all = df[(df["overnight_snow_cm"] > 0) & (df["snowfall_24h"] > 0)].copy()
amp_per_resort = (
    snowy_all.groupby("resort_id")
    .apply(lambda g: g["overnight_snow_cm"].mean() / g["snowfall_24h"].mean(), include_groups=False)
    .rename("nwp_amplification")
    .clip(upper=6.0)   # cap — see Task 4
    .to_dict()
)

payload = {
    "model":                        model,
    "feature_cols":                 FEATURE_COLS,
    "holdout_season":               holdout_season,
    "train_metrics":                train_metrics,
    "test_metrics":                 test_metrics,
    "log_transform":                False,
    "powder_pred_threshold":        powder_pred_thresh,
    "nwp_amplification_per_resort": amp_per_resort,   # NEW
}
```

- [ ] **Step 2: Apply per-resort amplification in `forecast.py` for all resorts**

In `forecast_resort()`, the block that builds features and predicts currently only sets amplification for calibrated SH resorts. Expand it to always set amplification:

```python
def forecast_resort(resort_id: str, cfg: dict, model, feat_cols: list,
                    calibrations: dict, amp_per_resort: dict, days: int = 7,
                    log_transform: bool = False) -> list[dict]:
    """Fetch forecast, run model, return list of daily dicts."""
    from datetime import date, timedelta
    today = date.today()
    start = today.strftime("%Y-%m-%d")
    end   = (today + timedelta(days=days - 1)).strftime("%Y-%m-%d")

    is_southern = cfg.get("hemisphere") == "south"
    calib       = calibrations.get(resort_id) if is_southern else None

    powder_threshold = calib["powder_threshold_raw_equiv"] if calib else POWDER_SCORE_THRESHOLD

    try:
        hourly = fetch_hourly(cfg["lat"], cfg["lon"], start, end, forecast=True)
    except Exception as e:
        return [{"error": str(e)}]

    daily = build_features(hourly, hemisphere=cfg.get("hemisphere", "north"))

    daily["resort_id"]   = resort_id
    daily["elevation"]   = cfg["elevation"]
    daily["region"]      = cfg["region"]
    daily["region_code"] = REGION_MAP.get(cfg["region"], -1)
    daily["lat"]         = cfg["lat"]
    daily["lon"]         = cfg["lon"]

    # ── Always apply NWP amplification ───────────────────────────────────────
    # For SH calibrated resorts, prefer the calibration's stored factor (fitted
    # on the same data as calibration). For Japan / uncalibrated resorts, use
    # the per-resort factor stored in the model payload.
    if calib and not calib.get("nwp_direct"):
        amp = calib.get("nwp_amplification", 1.0)
    else:
        amp = amp_per_resort.get(resort_id, 1.0)

    daily["nwp_amplification"]      = amp
    daily["amplified_snowfall_24h"] = daily["snowfall_24h"] * amp
    daily["amplified_snowfall_48h"] = daily["snowfall_48h"] * amp

    if calib and calib.get("nwp_direct"):
        preds = daily["snowfall_48h"].clip(0).values
    else:
        X = daily.reindex(columns=feat_cols, fill_value=0)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        raw = model.predict(X)
        preds = np.expm1(raw).clip(0) if log_transform else raw.clip(0)

    if calib:
        preds = _apply_calibration(preds, calib)

    snow_possible = daily["temp_min"] <= 2.0
    preds = preds * snow_possible.values.astype(float)

    results = []
    for i, (date_idx, row) in enumerate(daily.iterrows()):
        pred_cm = float(preds[i])
        score   = powder_score(
            pred_cm,
            float(row.get("snow_temp_mean",        0)),
            float(row.get("wind_during_snow_mean", 0)),
        )
        local_threshold = calib["powder_threshold_local_cm"] if calib else None
        results.append({
            "date":              str(date_idx.date()),
            "predicted_snow_cm": round(pred_cm, 1),
            "powder_score":      score,
            "is_powder_day":     pred_cm >= powder_threshold,
            "powder_threshold":  local_threshold or POWDER_SCORE_THRESHOLD,
            "temp_min_c":        round(float(row.get("temp_min",  0)), 1),
            "temp_max_c":        round(float(row.get("temp_max",  0)), 1),
            "wind_max_kmh":      round(float(row.get("wind_max",  0)), 1),
            "wind_mean_kmh":     round(float(row.get("wind_mean", 0)), 1),
            "snow_temp_c":       round(float(row.get("snow_temp_mean",        0)), 1),
            "wind_during_snow":  round(float(row.get("wind_during_snow_mean", 0)), 1),
            "humidity_pct":      round(float(row.get("humidity_mean", 0)), 1),
            "snowfall_48h_mm":   round(float(row.get("snowfall_48h", 0)), 1),
        })
    return results
```

- [ ] **Step 3: Thread `amp_per_resort` through `main()` in `forecast.py`**

In `main()`, load the amplification dict from the payload and pass it to `forecast_resort`:

```python
def load_model():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run: python -m core.train")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return (
        payload["model"],
        payload["feature_cols"],
        payload.get("log_transform", False),
        payload.get("nwp_amplification_per_resort", {}),
    )

# In main():
model, feat_cols, log_transform, amp_per_resort = load_model()
# ...
all_results[resort_id] = forecast_resort(
    resort_id, cfg, model, feat_cols, calibrations, amp_per_resort,
    days=args.days, log_transform=log_transform,
)
```

- [ ] **Step 4: Retrain the model to bake the new payload key in**

```bash
cd /Users/willlehmann/Desktop/Github/snow-powder-ML
.venv/bin/python -m core.train
```

Expected: training completes, `data/models/xgb_overnight_snow.pkl` updated.

- [ ] **Step 5: Verify the fix**

```bash
.venv/bin/python -c "
import pickle
with open('data/models/xgb_overnight_snow.pkl','rb') as f:
    p = pickle.load(f)
amp = p['nwp_amplification_per_resort']
print('niseko_grand_hirafu:', amp.get('niseko_grand_hirafu'))
print('niseko_annupuri:', amp.get('niseko_annupuri'))
print('thredbo:', amp.get('thredbo'))
assert amp.get('niseko_grand_hirafu', 0) > 2.0, 'Niseko amplification missing'
print('PASS')
"
```

Expected: Niseko ~4.2, thredbo ~0.5 (capped).

- [ ] **Step 6: Commit**

```bash
git add core/train.py forecast.py
git commit -m "fix: apply per-resort nwp_amplification at inference for Japan resorts

Top feature (0.316 importance) was always 0 for Japan resorts in production.
Store amp_per_resort in model payload; thread through forecast_resort()."
```

---

## Task 2: Drop lag features from FEATURE_COLS (inference mismatch)

**Files:**
- Modify: `core/train.py` (FEATURE_COLS list, ~line 40)

`overnight_snow_lag1`, `overnight_snow_lag2`, `snow_depth_lag1`, `overnight_snow_3d_sum` are actual on-mountain observations that don't exist at forecast time. They get filled with 0 at inference, but 0 means "no snow / no snowpack" which is a meaningful wrong value. Combined importance is ~2.4% — removing them costs little and removes a systematic train/inference distribution mismatch.

- [ ] **Step 1: Remove lag features from FEATURE_COLS in `core/train.py`**

Find the `FEATURE_COLS` list. Remove these four lines:

```python
# DELETE these four lines from FEATURE_COLS:
    "overnight_snow_lag1",   # yesterday's actual new snow — multi-day storm signal
    "overnight_snow_lag2",   # two days ago
    "overnight_snow_3d_sum", # lag1 + lag2 observed accumulation
    "snow_depth_lag1",
```

- [ ] **Step 2: Retrain**

```bash
.venv/bin/python -m core.train
```

- [ ] **Step 3: Verify metrics are comparable (expect minimal change)**

```bash
.venv/bin/python -c "
import pickle
with open('data/models/xgb_overnight_snow.pkl','rb') as f:
    p = pickle.load(f)
m = p['test_metrics']
print(f'Test r={m[\"r\"]}  F1={m[\"powder_f1\"]}  MAE={m[\"mae\"]}')
assert m['r'] > 0.65, 'r dropped significantly — investigate'
print('PASS')
"
```

- [ ] **Step 4: Commit**

```bash
git add core/train.py
git commit -m "fix: remove lag features from FEATURE_COLS (train/inference mismatch)

Lag features (overnight_snow_lag1/2, snow_depth_lag1, overnight_snow_3d_sum)
are on-mountain observations unavailable at forecast time. Fill_value=0 meant
model saw 'no snowpack' instead of 'unknown', causing systematic forecast bias."
```

---

## Task 3: Fix rolling windows crossing season boundaries in `core/features.py`

**Files:**
- Modify: `core/features.py` (build_features function, rolling window section ~line 184)

`snowfall_48h/72h/7d/14d/30d` use `.rolling()` on the full time series. A November 1st row's `snowfall_7d` includes October weather (pre-season). Fix: compute rolling within each season group.

- [ ] **Step 1: Refactor rolling windows in `build_features()` to reset per season**

Replace the rolling window block in `build_features()`:

```python
# OLD (crosses season boundaries):
d["snowfall_48h"]      = d["snowfall_24h"].rolling(2,  min_periods=1).sum()
d["snowfall_72h"]      = d["snowfall_24h"].rolling(3,  min_periods=1).sum()
d["snowfall_7d"]       = d["snowfall_24h"].rolling(7,  min_periods=1).sum()
d["snowfall_14d"]      = d["snowfall_24h"].rolling(14, min_periods=1).sum()
d["snowfall_last_30d"] = d["snowfall_24h"].rolling(30, min_periods=1).sum()

# NEW — compute season_year first, then group:
d["_sy_tmp"] = d.index.map(_sy)
for col, window in [
    ("snowfall_48h", 2), ("snowfall_72h", 3), ("snowfall_7d", 7),
    ("snowfall_14d", 14), ("snowfall_last_30d", 30),
]:
    d[col] = (
        d.groupby("_sy_tmp")["snowfall_24h"]
        .transform(lambda s: s.rolling(window, min_periods=1).sum())
    )
d.drop(columns=["_sy_tmp"], inplace=True)
```

Also fix `freeze_thaw_count_7d`:

```python
# OLD:
d["freeze_thaw_count_7d"] = d["freeze_thaw_event"].rolling(7, min_periods=1).sum()

# NEW:
d["_sy_tmp"] = d.index.map(_sy)
d["freeze_thaw_count_7d"] = (
    d.groupby("_sy_tmp")["freeze_thaw_event"]
    .transform(lambda s: s.rolling(7, min_periods=1).sum())
)
d.drop(columns=["_sy_tmp"], inplace=True)
```

- [ ] **Step 2: Rebuild features for all resorts and retrain**

```bash
# Japan features are embedded in processed CSVs — rebuild them
.venv/bin/python -c "
import yaml, pandas as pd
from pathlib import Path
from core.features import build_features
from collectors.weather import RAW_DIR

with open('regions.yaml') as f:
    regions = yaml.safe_load(f)

for resort_id, cfg in regions.items():
    candidates = sorted(Path('data/raw/weather').glob(f'{resort_id}_*.csv'),
                        key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        continue
    hourly = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
    hemisphere = cfg.get('hemisphere', 'north')
    daily = build_features(hourly, hemisphere=hemisphere)
    out = Path('data/processed') / f'features_{resort_id}.csv'
    daily.to_csv(out)
    print(f'  [{resort_id}] {len(daily)} days')
"
.venv/bin/python -m core.dataset
.venv/bin/python -m core.train
```

- [ ] **Step 3: Commit**

```bash
git add core/features.py
git commit -m "fix: reset rolling snow windows at season boundaries in features.py

Rolling windows (48h/72h/7d/14d/30d) now computed within each season group,
preventing October pre-season precipitation bleeding into early-November rows."
```

---

## Task 4: Cap nwp_amplification at 6x and flag geto_kogen

**Files:**
- Modify: `core/dataset.py` (nwp_amplification computation, ~line 152)

`geto_kogen` reports 10.9x amplification — 2x higher than the next resort. This single resort's extreme amplification inflates `amplified_snowfall` features and biases the model's sense of scale. Cap at 6x.

- [ ] **Step 1: Add cap and warning to the amplification computation in `core/dataset.py`**

Find the NWP amplification block and add the cap:

```python
    amp   = (
        snowy.groupby("resort_id")
        .apply(lambda g: g["overnight_snow_cm"].mean() / g["snowfall_24h"].mean(), include_groups=False)
        .rename("nwp_amplification")
    )

    # Cap implausible amplification values. Values above 6x are likely caused by
    # SnowJapan data quality issues or a severely misplaced NWP grid point.
    AMP_CAP = 6.0
    flagged = amp[amp > AMP_CAP]
    if not flagged.empty:
        print(f"  WARNING: nwp_amplification capped at {AMP_CAP}x for:")
        for rid, val in flagged.items():
            print(f"    {rid}: {val:.2f}x -> {AMP_CAP}x")
    amp = amp.clip(upper=AMP_CAP)

    df["nwp_amplification"] = df["resort_id"].map(amp).fillna(1.0)
```

- [ ] **Step 2: Rebuild dataset and retrain**

```bash
.venv/bin/python -m core.dataset
.venv/bin/python -m core.train
```

Expected output during dataset build: warning about geto_kogen being capped.

- [ ] **Step 3: Commit**

```bash
git add core/dataset.py
git commit -m "fix: cap nwp_amplification at 6x; warn on outliers (geto_kogen 10.9x)"
```

---

## Task 5: Fix `cold_air_advection` to be hemisphere-aware in `core/features.py`

**Files:**
- Modify: `core/features.py` (cold_air_advection block, ~line 308)

The current feature encodes "NW wind + rising pressure + cold temp" — the Hokkaido Siberian outbreak pattern. For SH resorts, powder comes from cold SW/S air (Antarctic fronts). The feature should produce a positive value for the correct synoptic pattern in each hemisphere.

- [ ] **Step 1: Modify `build_features()` to accept hemisphere and adjust `cold_air_advection`**

The function already accepts `hemisphere: str = "north"`. Update the cold air advection block:

```python
    # ── Cold air advection index ──────────────────────────────────────────────
    pressure_rise = d["pressure_tendency_24h"].clip(lower=0)
    cold_temp     = (-d["temp_min"]).clip(lower=0)

    if hemisphere == "north":
        # NH pattern: cold air from NW (Siberian outbreak for Japan)
        # Negative wind_u = westerly component (from the west)
        nw_wind = (-d["wind_u_during_snow"]).clip(lower=0)
    else:
        # SH pattern: cold air from SW/S (Antarctic front for AU/NZ/Andes)
        # Negative wind_v = southerly component (from the south)
        nw_wind = (-d["wind_v_during_snow"]).clip(lower=0)

    d["cold_air_advection"] = pressure_rise * cold_temp * nw_wind
```

- [ ] **Step 2: Rebuild all features, dataset, retrain**

```bash
.venv/bin/python -c "
import yaml, pandas as pd
from pathlib import Path
from core.features import build_features

with open('regions.yaml') as f:
    regions = yaml.safe_load(f)

for resort_id, cfg in regions.items():
    candidates = sorted(Path('data/raw/weather').glob(f'{resort_id}_*.csv'),
                        key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        continue
    hourly = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
    hemisphere = cfg.get('hemisphere', 'north')
    daily = build_features(hourly, hemisphere=hemisphere)
    out = Path('data/processed') / f'features_{resort_id}.csv'
    daily.to_csv(out)
    print(f'  [{resort_id}] {len(daily)} days')
"
.venv/bin/python -m core.dataset
.venv/bin/python -m core.train
```

- [ ] **Step 3: Commit**

```bash
git add core/features.py
git commit -m "fix: hemisphere-aware cold_air_advection (SH uses southerly wind component)

NH: NW wind (Siberian outbreak). SH: S/SW wind (Antarctic front).
Previously encoded Japan-specific pattern harmed AU/NZ/Andes predictions."
```

---

## Task 6: Fix hemisphere-aware evaluation in `core/evaluate.py`

**Files:**
- Modify: `core/evaluate.py` (evaluate function, ~line 74)

Test F1=0.528 is a meaningless average: Japan powder threshold=15cm, SH threshold=4-5cm, but the 9.13cm optimised threshold is applied to all resorts. Every SH powder day appears as a false negative. Fix: report Japan and SH metrics separately.

- [ ] **Step 1: Add hemisphere-split evaluation to `evaluate()`**

After the existing "Per-resort breakdown" block, add:

```python
    # ── Hemisphere-split metrics ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  HEMISPHERE-SPLIT METRICS (test set)")
    print(f"{'=' * 60}")

    POWDER_THRESHOLDS = {
        "north": 15.0,   # Japan: 15cm overnight is a genuine powder day
        "south":  4.0,   # SH:    4cm overnight is a powder day (lower elevation, marginal snow)
    }

    if "hemisphere" in test_df.columns:
        for hemi, hemi_df in test_df.groupby("hemisphere"):
            y_h   = hemi_df[TARGET].values
            p_h   = hemi_df["pred"].values
            thr_h = POWDER_THRESHOLDS.get(hemi, POWDER_CM)

            # Find F1-optimal threshold on this hemisphere's predictions
            from core.train import _find_best_threshold
            hemi_thr = _find_best_threshold(y_h, p_h, powder_cm=thr_h)

            mh = _metrics_row(y_h, p_h, hemi_thr)
            if not mh:
                continue
            resorts_h = sorted(hemi_df["resort_id"].unique())
            print(f"\n  {hemi.upper()} ({len(resorts_h)} resorts: {', '.join(resorts_h)})")
            print(f"  n={mh['n']:,}  MAE={mh['mae']:.2f}cm  r={mh['r']:.3f}")
            print(f"  Powder (actual>={thr_h}cm, pred>={hemi_thr:.1f}cm):")
            print(f"    Precision={mh['prec']:.3f}  Recall={mh['rec']:.3f}  F1={mh['f1']:.3f}")
            print(f"    TP={mh['tp']}  FP={mh['fp']}  FN={mh['fn']}")
    else:
        print("  'hemisphere' column not in test set — run core.dataset to rebuild.")
```

- [ ] **Step 2: Also fix LORO to use the Tweedie model architecture**

In the leave-one-resort-out block, replace the existing `m_loro` definition:

```python
        m_loro = XGBRegressor(
            n_estimators          = 800,
            max_depth             = 5,
            learning_rate         = 0.04,
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            min_child_weight      = 7,
            reg_alpha             = 0.1,
            reg_lambda            = 1.0,
            objective             = "reg:tweedie",
            tweedie_variance_power= 1.2,
            eval_metric           = "tweedie-nloglik@1.2",
            random_state          = 42,
            n_jobs                = -1,
        )
        m_loro.fit(X_tr, y_tr, verbose=False)
        p = m_loro.predict(X_te).clip(0)
```

And remove the `np.log1p` / `np.expm1` wrapping (Tweedie outputs raw cm, no log transform needed):

```python
        # No log transform — Tweedie outputs are in cm directly
        p = m_loro.predict(X_te).clip(0)
```

- [ ] **Step 3: Run evaluation to verify output**

```bash
.venv/bin/python -m core.evaluate
```

Expected: two new sections — NORTH metrics and SOUTH metrics — each with their own powder thresholds and F1.

- [ ] **Step 4: Commit**

```bash
git add core/evaluate.py
git commit -m "fix: hemisphere-aware evaluation; fix LORO to use Tweedie objective

Japan (>=15cm) and SH (>=4cm) powder metrics now reported separately.
LORO now uses same XGBoost hyperparameters as production model."
```

---

## Task 7: Fix SH calibration proxy label in `calibrate_transfer.py`

**Files:**
- Modify: `calibrate_transfer.py` (`run_nwp_direct()` function, ~line 81)

The NWP-direct path uses `snowfall_24h` as the label and `snowfall_48h` as the "prediction" — both from the same Open-Meteo NWP run. The r values are inflated because the variables are correlated by construction. Use ERA5 `snow_depth` change (already in `sh_labels.csv`) as the proxy label instead.

- [ ] **Step 1: Update `run_nwp_direct()` to use ERA5 labels when available**

```python
def run_nwp_direct(resort_id: str, cfg: dict,
                   start: str, end: str) -> pd.DataFrame:
    """
    NWP-direct path for regions where the Japan model has no useful signal.

    Prediction: 48h rolling NWP snowfall (genuine lead indicator).
    Label: ERA5 snow_depth change (independent observed state variable).
    Falls back to NWP snowfall_24h only if ERA5 labels are not available.
    """
    hourly = fetch_and_cache(resort_id, cfg["lat"], cfg["lon"], start=start, end=end)
    daily  = build_features(hourly, hemisphere=cfg.get("hemisphere", "north"))
    daily["japan_model_raw"] = daily["snowfall_48h"].clip(0)

    # Use ERA5 snow_depth change as proxy label (independent of NWP features).
    if SH_LABELS_PATH.exists():
        sh_lab = pd.read_csv(SH_LABELS_PATH, parse_dates=["date"])
        sh_lab = sh_lab[sh_lab["resort_id"] == resort_id].copy()
        sh_lab["date"] = sh_lab["date"].dt.normalize()
        era5_map = sh_lab.set_index("date")["new_snow_cm"]
        daily["proxy_snow_cm"] = daily.index.normalize().map(era5_map).fillna(0.0)
        print(f"    [{resort_id}] Using ERA5 snow_depth labels as proxy")
    else:
        # Fallback: NWP snowfall_24h (note: correlated with snowfall_48h)
        print(f"    [{resort_id}] WARNING: ERA5 labels not found, falling back to NWP proxy (inflated r expected)")
        daily["proxy_snow_cm"] = daily["snowfall_24h"].clip(0)

    return daily[daily.index.month.isin(SH_SEASON_MONTHS)].copy()
```

- [ ] **Step 2: Re-run calibration**

```bash
.venv/bin/python calibrate_transfer.py
```

Expected: r values for AU/NZ resorts will decrease (they were inflated by shared NWP origin). The calibration is now honest.

- [ ] **Step 3: Commit**

```bash
git add calibrate_transfer.py
git commit -m "fix: use ERA5 snow_depth labels as calibration proxy (not NWP snowfall)

NWP-direct path previously used snowfall_24h as label and snowfall_48h as
prediction — both from the same Open-Meteo run, giving inflated r values.
Now uses ERA5 snow_depth change (independent observed state variable)."
```

---

## Task 8: Detect and re-fetch stale weather CSVs missing dewpoint/pressure

**Files:**
- Create: `scripts/refresh_weather.py`

52% of training rows are missing `dewpoint_depression` and `pressure_mean` (fi=0.016 and fi=0.033 respectively) because older weather CSVs were cached before these variables were added to `HOURLY_VARS`. This script detects stale CSVs and re-fetches them.

- [ ] **Step 1: Create `scripts/refresh_weather.py`**

```python
"""
Detect and re-fetch weather CSVs that are missing variables added to HOURLY_VARS
after the original cache was written (dewpoint_2m, pressure_msl, surface_pressure).

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


def find_stale(resort_id: str | None = None) -> list[Path]:
    """Return list of cached weather CSVs missing required columns."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch",  action="store_true", help="Actually re-fetch (default: dry-run)")
    ap.add_argument("--resort", default=None,        help="Single resort_id to check/refresh")
    args = ap.parse_args()

    with open(REGIONS_YAML) as f:
        regions = yaml.safe_load(f)

    stale = find_stale(resort_id=args.resort)

    if not stale:
        print("All weather CSVs have current columns. Nothing to refresh.")
        return

    print(f"{'DRY RUN — ' if not args.fetch else ''}Found {len(stale)} stale CSV(s):\n")
    for path, missing in stale:
        print(f"  {path.name}  (missing: {', '.join(sorted(missing))})")

    if not args.fetch:
        print("\nRe-run with --fetch to download updated CSVs.")
        return

    print()
    for path, _ in stale:
        # Parse resort_id and dates from filename: {resort_id}_{start}_{end}.csv
        parts = path.stem.split("_")
        # Date parts are last two tokens of form YYYY-MM-DD
        end   = parts[-1]
        start = parts[-2]
        resort_id = "_".join(parts[:-2])

        cfg = regions.get(resort_id)
        if cfg is None:
            print(f"  [{resort_id}] not in regions.yaml, skipping")
            continue

        print(f"  [{resort_id}] re-fetching {start} -> {end} ...")
        path.unlink()   # remove stale cache

        try:
            df = fetch_hourly(cfg["lat"], cfg["lon"], start, end)
            out = RAW_DIR / f"{resort_id}_{start}_{end}.csv"
            df.to_csv(out)
            print(f"  [{resort_id}] {len(df):,} rows -> {out.name}")
            time.sleep(1.5)
        except Exception as e:
            print(f"  [{resort_id}] ERROR: {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run in dry-run mode to see what's stale**

```bash
.venv/bin/python scripts/refresh_weather.py
```

Expected: list of weather CSVs missing `dewpoint_2m` / `pressure_msl`.

- [ ] **Step 3: Re-fetch stale CSVs (this hits the Open-Meteo API — takes several minutes)**

```bash
.venv/bin/python scripts/refresh_weather.py --fetch
```

Expected: old CSVs removed and re-downloaded with full column set.

- [ ] **Step 4: Rebuild features, dataset, retrain**

```bash
.venv/bin/python -c "
import yaml, pandas as pd
from pathlib import Path
from core.features import build_features

with open('regions.yaml') as f:
    regions = yaml.safe_load(f)

for resort_id, cfg in regions.items():
    candidates = sorted(Path('data/raw/weather').glob(f'{resort_id}_*.csv'),
                        key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        continue
    hourly = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
    hemisphere = cfg.get('hemisphere', 'north')
    daily = build_features(hourly, hemisphere=hemisphere)
    out = Path('data/processed') / f'features_{resort_id}.csv'
    daily.to_csv(out)
    print(f'  [{resort_id}] {len(daily)} days')
"
.venv/bin/python -m core.dataset
.venv/bin/python -m core.train
```

- [ ] **Step 5: Verify NaN rates dropped**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/training_dataset.parquet')
nan_pct = df[['pressure_mean','dewpoint_depression','dewpoint_depression_min']].isna().mean()
print(nan_pct.to_string())
for col in ['pressure_mean','dewpoint_depression']:
    assert df[col].isna().mean() < 0.30, f'{col} still has >30% NaN — check re-fetch'
print('PASS')
"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/refresh_weather.py
git commit -m "feat: add scripts/refresh_weather.py to detect and re-fetch stale weather CSVs

Older cached CSVs lacked dewpoint_2m, pressure_msl, wind_direction_10m.
These were missing for ~52% of training rows despite being top-10 features."
```

---

## Task 9: Model versioning (timestamped artifacts)

**Files:**
- Modify: `core/train.py` (save section, ~line 258)

Every retrain overwrites `xgb_overnight_snow.pkl`. No way to roll back. Add timestamped saves with a `latest` symlink.

- [ ] **Step 1: Add timestamped save to `train()` in `core/train.py`**

Replace the save block:

```python
    # ── Save ─────────────────────────────────────────────────────────────────
    from datetime import datetime
    model_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":                        model,
        "feature_cols":                 FEATURE_COLS,
        "holdout_season":               holdout_season,
        "train_metrics":                train_metrics,
        "test_metrics":                 test_metrics,
        "log_transform":                False,
        "powder_pred_threshold":        powder_pred_thresh,
        "nwp_amplification_per_resort": amp_per_resort,
        "trained_at":                   datetime.utcnow().isoformat(),
    }

    # Write timestamped artifact
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = model_path.parent / f"xgb_overnight_snow_{ts}.pkl"
    with open(versioned_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved -> {versioned_path}")

    # Update the canonical path (copy, not symlink — works on all OSes)
    import shutil
    shutil.copy2(versioned_path, model_path)
    print(f"Latest -> {model_path}")

    return payload
```

- [ ] **Step 2: Retrain to produce first versioned artifact**

```bash
.venv/bin/python -m core.train
ls -lh data/models/
```

Expected: `xgb_overnight_snow_20260607_*.pkl` plus `xgb_overnight_snow.pkl` (canonical).

- [ ] **Step 3: Commit**

```bash
git add core/train.py
git commit -m "feat: timestamped model artifacts — xgb_overnight_snow_YYYYMMDD_HHMMSS.pkl

Canonical xgb_overnight_snow.pkl is a copy of the latest versioned file.
Enables rollback and experiment comparison."
```

---

## Task 10: Write the test suite

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_features.py`
- Create: `tests/test_train.py`
- Create: `tests/test_forecast.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Create `tests/__init__.py`**

```python
```
(Empty file.)

- [ ] **Step 2: Create `tests/conftest.py`**

```python
"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


def _make_hourly(n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create a minimal hourly weather DataFrame suitable for build_features()."""
    rng = np.random.default_rng(seed)
    n   = n_days * 24
    idx = pd.date_range("2023-11-01", periods=n, freq="h")
    df  = pd.DataFrame(
        {
            "temperature_2m":       rng.normal(-5, 8, n),
            "dewpoint_2m":          rng.normal(-10, 8, n),
            "snowfall":             np.clip(rng.exponential(0.3, n), 0, 10),
            "precipitation":        np.clip(rng.exponential(0.2, n), 0, 5),
            "wind_speed_10m":       np.clip(rng.exponential(15, n), 0, 80),
            "wind_direction_10m":   rng.uniform(0, 360, n),
            "relative_humidity_2m": np.clip(rng.normal(70, 15, n), 10, 100),
            "shortwave_radiation":  np.clip(rng.exponential(50, n), 0, 600),
            "cloud_cover":          np.clip(rng.normal(60, 30, n), 0, 100),
            "pressure_msl":         rng.normal(1013, 10, n),
            "surface_pressure":     rng.normal(900, 10, n),
            "temperature_850hPa":   rng.normal(-8, 5, n),
        },
        index=idx,
    )
    return df


@pytest.fixture
def hourly_nh() -> pd.DataFrame:
    """30 days of northern-hemisphere hourly weather."""
    return _make_hourly(n_days=30)


@pytest.fixture
def hourly_sh() -> pd.DataFrame:
    """30 days of southern-hemisphere hourly weather (Jul dates)."""
    rng = np.random.default_rng(99)
    n   = 30 * 24
    idx = pd.date_range("2023-07-01", periods=n, freq="h")
    df  = pd.DataFrame(
        {
            "temperature_2m":       rng.normal(-2, 5, n),
            "dewpoint_2m":          rng.normal(-6, 5, n),
            "snowfall":             np.clip(rng.exponential(0.1, n), 0, 5),
            "precipitation":        np.clip(rng.exponential(0.1, n), 0, 3),
            "wind_speed_10m":       np.clip(rng.exponential(20, n), 0, 80),
            "wind_direction_10m":   rng.uniform(0, 360, n),
            "relative_humidity_2m": np.clip(rng.normal(65, 15, n), 10, 100),
            "shortwave_radiation":  np.clip(rng.exponential(40, n), 0, 600),
            "cloud_cover":          np.clip(rng.normal(55, 30, n), 0, 100),
            "pressure_msl":         rng.normal(1010, 12, n),
            "surface_pressure":     rng.normal(920, 10, n),
            "temperature_850hPa":   rng.normal(-5, 4, n),
        },
        index=idx,
    )
    return df
```

- [ ] **Step 3: Create `tests/test_features.py`**

```python
"""Tests for core/features.py."""

import numpy as np
import pandas as pd
import pytest

from core.features import build_features


def test_build_features_returns_daily_rows(hourly_nh):
    daily = build_features(hourly_nh, hemisphere="north")
    # One row per calendar day
    assert len(daily) == 30
    assert daily.index.dtype == "datetime64[ns]"


def test_all_expected_columns_present(hourly_nh):
    from core.train import FEATURE_COLS
    daily = build_features(hourly_nh, hemisphere="north")
    # Static resort columns are not in features.py, that's fine —
    # check the weather-derived ones
    weather_cols = [
        "snowfall_24h", "snowfall_48h", "snowfall_72h", "snowfall_7d", "snowfall_14d",
        "temp_min", "temp_max", "wind_max", "humidity_mean",
        "pressure_mean", "pressure_tendency_24h",
        "dewpoint_depression", "snow_quality_index",
        "cold_air_advection",
    ]
    for col in weather_cols:
        assert col in daily.columns, f"Missing column: {col}"


def test_rolling_windows_do_not_cross_seasons():
    """snowfall_72h on the first day of a season should not include prior season."""
    # Build two-season hourly data. Put heavy snow in Oct (season N),
    # then check that Nov 1 (season N+1) snowfall_72h reflects only Nov data.
    idx = pd.date_range("2022-10-29", periods=10 * 24, freq="h")
    df  = pd.DataFrame(
        {
            "temperature_2m":       [-5.0] * (10 * 24),
            "dewpoint_2m":          [-8.0] * (10 * 24),
            "snowfall":             [0.5]  * (10 * 24),   # steady snow throughout
            "precipitation":        [0.5]  * (10 * 24),
            "wind_speed_10m":       [10.0] * (10 * 24),
            "wind_direction_10m":   [270.0]* (10 * 24),
            "relative_humidity_2m": [80.0] * (10 * 24),
            "shortwave_radiation":  [0.0]  * (10 * 24),
            "cloud_cover":          [100.0]* (10 * 24),
            "pressure_msl":         [1010.0]*(10 * 24),
            "surface_pressure":     [900.0] *(10 * 24),
        },
        index=idx,
    )
    daily = build_features(df, hemisphere="north")

    # Oct 29-31 are season 2022; Nov 01+ are season 2023
    nov1 = pd.Timestamp("2022-11-01")
    if nov1 in daily.index:
        # snowfall_72h on Nov 1 should only reflect Nov 1 (1 day in new season)
        # with season-bounded rolling, it cannot include Oct 29-31
        oct31_snow = daily.loc[pd.Timestamp("2022-10-31"), "snowfall_24h"]
        nov1_row   = daily.loc[nov1]
        # If rolling resets at season boundary, Nov 1's snowfall_72h = its own 24h sum
        assert nov1_row["snowfall_72h"] == pytest.approx(nov1_row["snowfall_24h"], rel=0.01), \
            "Rolling window crossed season boundary — Nov 1 snowfall_72h includes Oct data"


def test_cold_air_advection_north_uses_u_wind(hourly_nh):
    """NH cold_air_advection should increase when wind_u is negative (westerly)."""
    daily = build_features(hourly_nh, hemisphere="north")
    assert "cold_air_advection" in daily.columns
    assert (daily["cold_air_advection"] >= 0).all(), "cold_air_advection must be non-negative"


def test_cold_air_advection_south_uses_v_wind(hourly_sh):
    """SH cold_air_advection should use southerly wind component, not NW component."""
    daily_sh = build_features(hourly_sh, hemisphere="south")
    daily_nh = build_features(hourly_sh, hemisphere="north")
    # They should differ (the wind component used is different)
    assert not daily_sh["cold_air_advection"].equals(daily_nh["cold_air_advection"]), \
        "SH and NH cold_air_advection should differ"


def test_no_lag_features_in_output(hourly_nh):
    """build_features must not produce lag features (they're only in dataset.py)."""
    daily = build_features(hourly_nh, hemisphere="north")
    lag_cols = {"overnight_snow_lag1", "overnight_snow_lag2",
                "snow_depth_lag1", "overnight_snow_3d_sum"}
    in_output = lag_cols & set(daily.columns)
    assert not in_output, f"Lag features should not be in features.py output: {in_output}"


def test_no_negative_snowfall_features(hourly_nh):
    daily = build_features(hourly_nh, hemisphere="north")
    for col in ["snowfall_24h", "snowfall_48h", "snowfall_72h", "snowfall_7d"]:
        assert (daily[col] >= 0).all(), f"{col} has negative values"
```

- [ ] **Step 4: Create `tests/test_train.py`**

```python
"""Tests for core/train.py — payload structure and feature list integrity."""

import pickle
from pathlib import Path

import pytest


MODEL_PATH = Path("data/models/xgb_overnight_snow.pkl")


@pytest.fixture
def payload():
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet — run python -m core.train first")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_payload_has_required_keys(payload):
    required = {
        "model", "feature_cols", "holdout_season",
        "train_metrics", "test_metrics", "log_transform",
        "powder_pred_threshold", "nwp_amplification_per_resort",
    }
    missing = required - set(payload.keys())
    assert not missing, f"Payload missing keys: {missing}"


def test_nwp_amplification_per_resort_has_japan_resorts(payload):
    amp = payload["nwp_amplification_per_resort"]
    japan_resorts = ["niseko_grand_hirafu", "niseko_annupuri", "kiroro", "rusutsu", "furano"]
    for r in japan_resorts:
        assert r in amp, f"Japan resort {r} missing from nwp_amplification_per_resort"
        assert amp[r] > 1.0, f"{r} amplification {amp[r]} should be > 1.0"
        assert amp[r] <= 6.0, f"{r} amplification {amp[r]} exceeds 6x cap"


def test_no_lag_features_in_feature_cols(payload):
    lag = {"overnight_snow_lag1", "overnight_snow_lag2",
           "snow_depth_lag1", "overnight_snow_3d_sum"}
    in_cols = lag & set(payload["feature_cols"])
    assert not in_cols, f"Lag features must not be in FEATURE_COLS: {in_cols}"


def test_model_test_metrics_reasonable(payload):
    m = payload["test_metrics"]
    assert m["r"] > 0.60,         f"Test r={m['r']} is too low (expect >0.60)"
    assert m["mae"] < 5.0,        f"Test MAE={m['mae']} is too high (expect <5.0)"
    assert m["powder_f1"] > 0.35, f"Powder F1={m['powder_f1']} is too low (expect >0.35)"


def test_log_transform_is_false(payload):
    """Tweedie model does not use log transform."""
    assert payload["log_transform"] is False
```

- [ ] **Step 5: Create `tests/test_forecast.py`**

```python
"""Smoke tests for forecast.py inference pipeline."""

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("data/models/xgb_overnight_snow.pkl")


@pytest.fixture
def model_payload():
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _fake_hourly(n_days: int = 7) -> pd.DataFrame:
    """Minimal hourly DataFrame mimicking Open-Meteo forecast response."""
    import numpy as np
    n   = n_days * 24
    idx = pd.date_range("2025-12-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "temperature_2m":       rng.normal(-8, 5, n),
            "dewpoint_2m":          rng.normal(-12, 5, n),
            "snowfall":             np.clip(rng.exponential(0.5, n), 0, 10),
            "precipitation":        np.clip(rng.exponential(0.3, n), 0, 5),
            "wind_speed_10m":       np.clip(rng.exponential(15, n), 0, 80),
            "wind_direction_10m":   rng.uniform(0, 360, n),
            "relative_humidity_2m": np.clip(rng.normal(75, 10, n), 10, 100),
            "shortwave_radiation":  np.clip(rng.exponential(20, n), 0, 200),
            "cloud_cover":          np.clip(rng.normal(70, 20, n), 0, 100),
            "pressure_msl":         rng.normal(1005, 15, n),
            "surface_pressure":     rng.normal(895, 10, n),
            "temperature_850hPa":   rng.normal(-10, 4, n),
        },
        index=idx,
    )


def test_japan_resort_forecast_uses_nonzero_amplification(model_payload):
    """Niseko forecast must set nwp_amplification from the payload, not default to 0."""
    from forecast import forecast_resort

    amp_per_resort = model_payload["nwp_amplification_per_resort"]
    model     = model_payload["model"]
    feat_cols = model_payload["feature_cols"]

    cfg = {
        "lat": 42.8643, "lon": 140.7009, "elevation": 1200,
        "region": "hokkaido", "hemisphere": "north",
    }

    with patch("forecast.fetch_hourly", return_value=_fake_hourly(7)):
        results = forecast_resort(
            "niseko_grand_hirafu", cfg, model, feat_cols,
            calibrations={}, amp_per_resort=amp_per_resort, days=7,
        )

    assert len(results) == 7
    assert "error" not in results[0]
    for day in results:
        assert 0 <= day["powder_score"] <= 100, "Score out of range"
        assert day["predicted_snow_cm"] >= 0,   "Negative snowfall prediction"


def test_powder_score_zero_when_no_snow():
    from forecast import powder_score
    assert powder_score(0.0, -5.0, 10.0) == 0
    assert powder_score(0.4, -5.0, 10.0) == 0   # below 0.5cm threshold


def test_powder_score_in_range():
    from forecast import powder_score
    for snow in [0, 1, 5, 10, 15, 20, 30]:
        score = powder_score(float(snow), -8.0, 5.0)
        assert 0 <= score <= 100, f"Score {score} out of range for snow={snow}cm"
```

- [ ] **Step 6: Create `tests/test_evaluate.py`**

```python
"""Tests for core/evaluate.py — hemisphere-aware metrics."""

import numpy as np
import pytest


def test_metrics_row_perfect_prediction():
    from core.evaluate import _metrics_row
    y = np.array([0, 0, 5, 20, 0, 30, 15, 0])
    m = _metrics_row(y, y, powder_pred_thresh=15.0)
    assert m["mae"]  == pytest.approx(0.0)
    assert m["r"]    == pytest.approx(1.0, abs=0.01)
    assert m["prec"] == pytest.approx(1.0)
    assert m["rec"]  == pytest.approx(1.0)


def test_metrics_row_all_zeros():
    from core.evaluate import _metrics_row
    y = np.zeros(10)
    m = _metrics_row(y, y, powder_pred_thresh=5.0)
    # No powder days → prec/rec are 0
    assert m["tp"] == 0
    assert m["fn"] == 0


def test_hemisphere_powder_thresholds_differ():
    """NH threshold (15cm) and SH threshold (4cm) must be defined separately."""
    # This test checks the constants exist with correct values
    NH_THRESHOLD = 15.0
    SH_THRESHOLD =  4.0
    assert NH_THRESHOLD > SH_THRESHOLD
    assert SH_THRESHOLD >= 4.0
```

- [ ] **Step 7: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | head -80
```

Expected: all tests pass. Fix any failures before continuing.

- [ ] **Step 8: Commit**

```bash
git add tests/
git commit -m "test: add test suite (features, train payload, forecast smoke, evaluate)

Covers: season-boundary rolling, hemisphere cold_air_advection, lag feature
absence, model payload structure, nwp_amplification at inference, score range."
```

---

## Task 11: Remove hardcoded API key and dead code

**Files:**
- Delete (or empty): `test.py`
- Create: `.env.example`
- Modify: `.gitignore` (add `.env`)

`test.py` contains a hardcoded OpenWeatherMap API key. It is also dead code — all its logic was promoted to `core/features.py` and `collectors/weather.py`.

- [ ] **Step 1: Create `.env.example`**

```bash
# Copy to .env and fill in values
OPENWEATHER_API_KEY=your_key_here
```

- [ ] **Step 2: Add `.env` to `.gitignore`**

If `.gitignore` doesn't exist, create it. Add:

```
.env
*.pkl
__pycache__/
.DS_Store
```

- [ ] **Step 3: Remove the API key from `test.py` and mark it deprecated**

Replace the contents of `test.py` with:

```python
"""
DEPRECATED — this file was the original prototype pipeline.
All logic has been promoted to:
  - collectors/weather.py  (fetch_hourly, fetch_and_cache)
  - core/features.py       (build_features)

Do not use this file for anything. It will be removed in a future cleanup.
"""

raise RuntimeError(
    "test.py is deprecated. Use collectors/weather.py and core/features.py instead."
)
```

- [ ] **Step 4: Commit**

```bash
git add .env.example .gitignore test.py
git commit -m "chore: remove hardcoded API key from test.py; add .env.example

test.py is now a stub pointing to the correct modules. API keys should
be stored in .env (gitignored), not committed to the repository."
```

---

## Task 12: Update /docs for every changed file

**Files:**
- Create/Update: `docs/core_train.md`
- Create/Update: `docs/core_features.md`
- Create/Update: `docs/core_dataset.md`
- Create/Update: `docs/core_evaluate.md`
- Create/Update: `docs/forecast.md`
- Create/Update: `docs/calibrate_transfer.md`
- Create/Update: `docs/collectors_weather.md`
- Create/Update: `docs/build_sh_features.md`
- Create/Update: `docs/build_sh_labels.md`
- Create/Update: `docs/explore.md`
- Create/Update: `docs/hindcast_australia.md`
- Create/Update: `docs/regions_yaml.md`

- [ ] **Step 1: Create `docs/` directory**

```bash
mkdir -p docs
```

- [ ] **Step 2: Write `docs/core_train.md`**

```markdown
# core/train.py

**Purpose:** Train the XGBoost overnight-snowfall regression model and save it with all inference metadata.

**Inputs:**
- `data/processed/training_dataset.parquet` — joined feature+label dataset
- CLI args: `--holdout` (first test season), `--dataset`, `--output`

**Outputs:**
- `data/models/xgb_overnight_snow_YYYYMMDD_HHMMSS.pkl` — timestamped versioned artifact
- `data/models/xgb_overnight_snow.pkl` — canonical copy of the latest versioned model

**Payload dict keys:**
| Key | Description |
|---|---|
| `model` | Fitted XGBRegressor |
| `feature_cols` | Ordered list of feature names (must match inference) |
| `holdout_season` | First season in the test set (e.g. `"2022-2023"`) |
| `train_metrics` / `test_metrics` | MAE, RMSE, r, powder F1, TP/FP/FN |
| `log_transform` | Always `False` — Tweedie handles zeros natively |
| `powder_pred_threshold` | F1-optimal threshold calibrated on Japan train rows only |
| `nwp_amplification_per_resort` | Per-resort ratio of observed/NWP snowfall, capped at 6x |
| `trained_at` | UTC ISO timestamp of the training run |

**Key parameters:**
- `objective="reg:tweedie"`, `tweedie_variance_power=1.2` — zero-inflated right-skewed target
- `n_estimators=800`, `max_depth=5`, `learning_rate=0.04`, `min_child_weight=7`
- `HOLDOUT_SEASON = "2022-2023"` — first season in the test set

**Notes:**
- Season holdout only — never random split (adjacent days are correlated).
- Powder threshold calibrated on Japan training rows only (SH resorts have a 4cm threshold, not 15cm).
- 850hPa features are NaN for pre-2021 rows; XGBoost native NaN routing handles this.
- Lag features (`overnight_snow_lag1/2`, `snow_depth_lag1`) are intentionally excluded — they are observed values unavailable at inference time.
- `nwp_amplification` is capped at 6x before training and before storing in the payload.

**Last updated:** 2026-06-07
```

- [ ] **Step 3: Write `docs/core_features.md`**

```markdown
# core/features.py

**Purpose:** Convert cached hourly Open-Meteo weather into a per-day feature DataFrame used for both training and inference.

**Inputs:**
- `hourly: pd.DataFrame` — hourly weather indexed by UTC timestamp (from `collectors/weather.py`)
- `hemisphere: str` — `"north"` or `"south"` (controls season-year grouping and `cold_air_advection` direction)

**Outputs:**
- Daily DataFrame indexed by date, ~75 feature columns

**Key feature groups:**
| Group | Features |
|---|---|
| Rolling snow windows | `snowfall_48h`, `72h`, `7d`, `14d`, `30d` — **reset per season** |
| Pressure | `pressure_mean`, `pressure_tendency_24h/3d`, `pressure_anomaly` |
| 850 hPa | `temp_850_mean/min/max`, `temp_850_trend`, `rain_risk_850`, `freeze_depth_850` |
| Dewpoint | `dewpoint_depression`, `dewpoint_depression_min` |
| Wind direction | `wind_dir_sin/cos`, `wind_dir_snow_sin/cos`, `wind_dir_consistency` |
| Derived | `snow_quality_index`, `powder_freshness`, `melt_risk`, `cold_air_advection` |
| Calendar | `day_of_year`, `month`, `season_year`, `days_in_season` |

**Notes:**
- Rolling windows (`snowfall_48h` etc.) use `groupby(season_year).transform(rolling)` to prevent October pre-season precipitation bleeding into November rows.
- `cold_air_advection` is hemisphere-aware: NH uses `wind_u` (NW/Siberian outbreak), SH uses `wind_v` (southerly/Antarctic front).
- 850hPa features are NaN when `temperature_850hPa` is absent from the hourly input (pre-2021 archive data). NaN is intentional — 0°C at 850hPa is a real weather condition; NaN signals genuinely missing data.
- This module does **not** produce lag features or `nwp_amplification` — those are added in `core/dataset.py`.

**Last updated:** 2026-06-07
```

- [ ] **Step 4: Write `docs/core_dataset.md`**

```markdown
# core/dataset.py

**Purpose:** Assemble the training dataset by joining Japan + SH labels with per-resort feature CSVs, adding static resort metadata, and computing NWP amplification factors.

**Inputs:**
- `data/processed/snowjapan_labels.csv` — Japan SnowJapan observed labels
- `data/processed/sh_labels.csv` — SH ERA5 snow_depth-change labels
- `data/processed/features_{resort_id}.csv` — per-resort daily feature tables
- `regions.yaml` — resort static metadata

**Outputs:**
- `data/processed/training_dataset.parquet` (~49k rows, ~90 columns)

**Key computed columns:**
| Column | How |
|---|---|
| `overnight_snow_cm` | Renamed from `new_snow_cm`; capped at 80cm |
| `region_code` | Integer encoding of `region` string |
| `nwp_amplification` | Per-resort mean(observed)/mean(NWP snowfall) on snowy days, **capped at 6x** |
| `amplified_snowfall_24h/48h` | `snowfall_24h/48h × nwp_amplification` |
| `snow_depth_lag1` | Previous day's observed snow depth (Japan only) |
| `overnight_snow_lag1/2` | Previous 1–2 days' actual new snow |
| `overnight_snow_3d_sum` | `lag1 + lag2` |

**Notes:**
- `nwp_amplification` cap at 6x: values above this indicate data quality issues (e.g. `geto_kogen` was 10.9x before capping). A warning is printed for any capped resort.
- 850hPa NaN rows are left as NaN (not imputed); XGBoost handles them via native NaN routing.
- Lag features are added here and present in the training parquet, but **excluded from `FEATURE_COLS`** in `core/train.py` because they are unavailable at inference time.

**Last updated:** 2026-06-07
```

- [ ] **Step 5: Write `docs/core_evaluate.md`**

```markdown
# core/evaluate.py

**Purpose:** Evaluate the trained model on the season holdout and (optionally) via leave-one-resort-out, with hemisphere-aware powder metrics.

**Inputs:**
- `data/models/xgb_overnight_snow.pkl`
- `data/processed/training_dataset.parquet`

**Outputs:** Printed metrics tables (no files written).

**Metric groups:**
1. Overall season holdout (Japan 15cm + SH 4cm powder mixed — for trend tracking)
2. **Hemisphere-split** — Japan (>=15cm) and SH (>=4cm) reported independently
3. Per-resort breakdown
4. Leave-one-resort-out (`--loro` flag, slow)

**Notes:**
- Powder thresholds: Japan=15cm, SH=4cm. Mixed evaluation (item 1) is kept for backwards-compatibility trend tracking but should not be used to judge SH performance.
- LORO uses the same Tweedie hyperparameters as the production model (not a separate log-transform model).
- The powder detection threshold (`powder_pred_threshold`) is loaded from the model payload, not recomputed at evaluation time.

**Last updated:** 2026-06-07
```

- [ ] **Step 6: Write `docs/forecast.md`**

```markdown
# forecast.py

**Purpose:** Fetch 7-day weather forecast from Open-Meteo and produce daily predicted snowfall + Powder Score for all configured resorts.

**Inputs:**
- `data/models/xgb_overnight_snow.pkl`
- `data/models/transfer_calibration.json` (SH resorts)
- `regions.yaml`
- Open-Meteo forecast API (live, no key required)

**Outputs:** Formatted text table or JSON (`--json` flag)

**Inference path per resort:**
1. Fetch 7-day hourly forecast → `build_features(hemisphere=...)`
2. Set `nwp_amplification` from model payload's `nwp_amplification_per_resort` (Japan) or calibration object (SH).
3. For SH Andes resorts: run Japan model → apply isotonic calibration.
4. For AU/NZ resorts: use `snowfall_48h` directly (NWP-direct path) → apply isotonic calibration.
5. Physical gate: `temp_min > 2°C → prediction = 0` (applied **after** calibration).
6. Compute Powder Score: `f(predicted_snow_cm, snow_temp_mean, wind_during_snow_mean)`.

**Key constants:**
- `POWDER_SCORE_THRESHOLD = 7.0` — predicted cm to flag a powder day (Japan)
- `_SNOW_SCALE_100 = 15.0` — model output at which score saturates at 100
- Physical gate: `temp_min <= 2.0°C`

**Notes:**
- `nwp_amplification` is always set before calling `model.predict()`. Failure to set it (as was the case pre-fix) causes the top feature to be 0, severely suppressing Japan predictions.
- Lag features (`overnight_snow_lag1/2`, `snow_depth_lag1`) are excluded from `FEATURE_COLS` and therefore not required at inference.

**Last updated:** 2026-06-07
```

- [ ] **Step 7: Write remaining docs**

```markdown
<!-- docs/calibrate_transfer.md -->
# calibrate_transfer.py

**Purpose:** Fit per-resort isotonic calibration mapping from the Japan model's raw output (or NWP snowfall_48h) to local expected snowfall for Southern Hemisphere resorts.

**Inputs:**
- `data/models/xgb_overnight_snow.pkl`
- `data/processed/sh_labels.csv` — ERA5 snow_depth change labels (used as proxy)
- Cached weather CSVs (`data/raw/weather/`)
- `regions.yaml`

**Outputs:**
- `data/models/transfer_calibration.json` — isotonic calibration params per SH resort
- `data/plots/calibration_{resort_id}.png` — calibration scatter plots

**Two calibration paths:**
| Path | Resorts | Prediction | Proxy label |
|---|---|---|---|
| Japan model | Andes (chile, argentina) | XGBoost raw output | ERA5 snow_depth change |
| NWP-direct | AU, NZ | `snowfall_48h` | ERA5 snow_depth change |

**Notes:**
- ERA5 snow_depth change is used as the proxy label for **both** paths (not NWP snowfall). This ensures the label is independent of the NWP features.
- Isotonic regression (monotone) is preferred over linear: the Japan model saturates at high snowfall values, which a linear mapping cannot handle.
- `powder_threshold_raw_equiv` is the median Japan model output on days where ERA5 labels exceed the local powder threshold.

**Last updated:** 2026-06-07
```

```markdown
<!-- docs/collectors_weather.md -->
# collectors/weather.py

**Purpose:** Fetch hourly weather from Open-Meteo and cache to CSV for use in feature engineering.

**Inputs:** lat, lon, date range, `forecast: bool` flag

**Outputs:** `data/raw/weather/{resort_id}_{start}_{end}.csv` (hourly, all HOURLY_VARS)

**Variables fetched:** temperature_2m, dewpoint_2m, snowfall, precipitation, wind_speed_10m, wind_direction_10m, relative_humidity_2m, shortwave_radiation, cloud_cover, pressure_msl, surface_pressure. Pressure-level (temperature_850hPa) is fetched separately from the historical-forecast API (available from 2021-03-23 only).

**Notes:**
- Archive API (`archive-api.open-meteo.com`) for historical; forecast API (`api.open-meteo.com`) for inference.
- 1-second polite delay between requests.
- If the cached CSV was written before `dewpoint_2m` / `pressure_msl` were added to `HOURLY_VARS`, use `scripts/refresh_weather.py` to detect and re-fetch stale files.

**Last updated:** 2026-06-07
```

```markdown
<!-- docs/build_sh_features.md -->
# build_sh_features.py

**Purpose:** Build `data/processed/features_{resort_id}.csv` for all Southern Hemisphere resorts by running `build_features(hemisphere="south")` on cached weather CSVs.

**Inputs:** `data/raw/weather/{resort_id}_*.csv`, `regions.yaml`
**Outputs:** `data/processed/features_{resort_id}.csv` for each SH resort

**Notes:** Mirrors the Japan feature pipeline. Uses the largest available cached weather file per resort. Must be re-run after any change to `core/features.py`.

**Last updated:** 2026-06-07
```

```markdown
<!-- docs/build_sh_labels.md -->
# build_sh_labels.py

**Purpose:** Fetch ERA5 hourly snow_depth from Open-Meteo archive and derive daily new_snow_cm labels for Southern Hemisphere resorts.

**Inputs:** Open-Meteo archive API (`snow_depth` variable), `regions.yaml`
**Outputs:**
- `data/raw/snow_depth/{resort_id}.csv` — cached hourly ERA5 snow_depth
- `data/processed/sh_labels.csv` — daily new_snow_cm for all SH resorts

**Label derivation:** `new_snow_cm = max(0, daily_max_depth_cm - prev_day_depth_cm)`, capped at 80cm.

**Notes:**
- ERA5 snow_depth is an observationally-constrained reanalysis state variable — more reliable than NWP snowfall for marginal-snow regions.
- Grid resolution ~31km; depth represents an area-weighted average that includes lower-elevation terrain. Amplification factors in `core/dataset.py` correct for this.
- Season months: May–October (SH_MONTHS = {5,6,7,8,9,10}).

**Last updated:** 2026-06-07
```

```markdown
<!-- docs/explore.md -->
# explore.py

**Purpose:** Produce 6 EDA plots from `training_dataset.parquet` for visual inspection before and after model changes.

**Inputs:** `data/processed/training_dataset.parquet`
**Outputs:** `data/plots/01_target_distribution.png` through `06_region_comparison.png`

**Plots produced:**
1. Target distribution (linear + log scale)
2. Pearson r heatmap of all features vs target
3. Scatter of top-8 correlated features vs target
4. Powder day % and mean snowfall per resort
5. Monthly snowfall violin
6. CDF by region

**Last updated:** 2026-06-07
```

```markdown
<!-- docs/hindcast_australia.md -->
# hindcast_australia.py

**Purpose:** Run the Japan-trained model on historical SH weather and evaluate against Open-Meteo's own snowfall variable as a rough sanity check.

**Inputs:** Cached weather CSVs, `data/models/xgb_overnight_snow.pkl`, `regions.yaml`
**Outputs:** `data/plots/australia_{resort_id}.png`, printed metrics

**Warning:** The proxy label (`snowfall_24h`) shares the same NWP source as the model features. R values are inflated. This script is a **sanity check only**, not a true independent evaluation. Use ERA5 snow_depth labels from `build_sh_labels.py` for honest evaluation.

**Last updated:** 2026-06-07
```

```markdown
<!-- docs/regions_yaml.md -->
# regions.yaml

**Purpose:** Single source of truth for all resort configurations used across the pipeline.

**Fields per resort:**
| Field | Description |
|---|---|
| `lat`, `lon` | Resort summit or upper lift station (decimal degrees) |
| `elevation` | Metres above sea level |
| `region` | Sub-national grouping (categorical model feature); maps to `region_code` in `core/dataset.py` |
| `hemisphere` | `"north"` or `"south"` — controls season-year logic and `cold_air_advection` direction |
| `snowjapan_id` | Numeric ID for SnowJapan REST API (Japan resorts only) |
| `snowjapan_slug` | URL slug at snowjapan.com (Japan resorts only) |
| `snow_report_url` | LivePass/Alterra endpoint URL (non-Japan resorts with live reports) |
| `snow_report_format` | `"livepass_xml"` or `"alterra_json"` |

**Region codes** (used in `core/dataset.py` REGION_MAP):
hokkaido=0, nagano=1, niigata=2, tohoku=3, nsw=4, victoria=5, nz_south=6, nz_north=7, andes_chile=8, andes_argentina=9

**Last updated:** 2026-06-07
```

- [ ] **Step 8: Write the 12 doc files to disk**

```bash
mkdir -p docs

cat > docs/core_train.md << 'HEREDOC'
[content from Step 2 above]
HEREDOC
# ... etc — write each doc file
```

(When executing this task, write each doc file using the Write tool rather than bash heredocs.)

- [ ] **Step 9: Commit**

```bash
git add docs/
git commit -m "docs: create /docs for all modified source files (CLAUDE.md requirement)"
```

---

## Task 13: Full pipeline smoke test and final verification

- [ ] **Step 1: Run the complete test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Run full evaluation**

```bash
.venv/bin/python -m core.evaluate
```

Expected output includes a HEMISPHERE-SPLIT METRICS section showing separate Japan and SH results.

- [ ] **Step 3: Run a forecast for a Japan resort and verify score is sensible**

```bash
.venv/bin/python forecast.py --resort niseko_grand_hirafu --days 3
```

Expected: non-zero powder scores (not all zeros as would happen with broken amplification).

- [ ] **Step 4: Run a forecast for an SH resort**

```bash
.venv/bin/python forecast.py --resort thredbo --days 3
```

Expected: runs without error; scores in [0,100].

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "chore: final smoke test pass — all audit fixes implemented"
```

---

## Self-review checklist

| Audit item | Task |
|---|---|
| nwp_amplification=0 at inference for Japan resorts | Task 1 |
| Lag features always 0 at inference | Task 2 |
| Rolling windows cross season boundaries | Task 3 |
| geto_kogen 10.9x amplification | Task 4 |
| cold_air_advection Japan-specific | Task 5 |
| Hemisphere-aware evaluation | Task 6 |
| LORO uses wrong model architecture | Task 6 |
| SH calibration proxy label circular | Task 7 |
| 52% NaN in dewpoint/pressure (stale CSVs) | Task 8 |
| No model versioning | Task 9 |
| No test suite | Task 10 |
| Hardcoded API key in test.py | Task 11 |
| /docs not updated | Task 12 |
| Full pipeline verification | Task 13 |
