Project Brief — Powder Forecasting ML System
Paste this whole document into a new Claude conversation as your opening message. It contains the vision, the architecture, the technical findings already validated, and the principles I want you to hold me to.

What I'm building
An ML system that predicts daily snow conditions for ski areas from weather, terrain, and real avalanche/condition reports — and rolls them into a Powder Score (0–100) per resort. The goal is a multi-region system (start: Utah, USA) that I can extend to Japan, Australia, Europe, etc. by adding regions and retraining, and eventually turn into a commercial app (per-resort powder forecasts, scores, and alerts).
I want you to act as a hands-on build partner: concrete code, one step at a time, and honest about data and ML pitfalls even when it's inconvenient. Don't retreat to toy heuristics when I ask for a model.
What the model predicts (and an honest framing)
The model predicts the day's snow conditions, expressed as the avalanche-problem types that professional forecasters report (e.g. New Snow, Wind Slab, Persistent Slab, Wet Snow). The Powder Score is a readout over the model's predicted condition probabilities — high P(new/fresh snow), low P(wet/wind-affected) and low danger ⇒ high score.
Be clear-eyed about this: no one publishes a ground-truth "powder score" to train against. So the score is derived, not scraped. The honest, non-circular path is:

Train a model: features = weather/terrain, target = independently-observed condition reports.
Compute the Powder Score from the model's predicted condition probabilities.
Later, sharpen it with field observations that describe actual surface quality (powder/crust/corn).

Note that avalanche-problem labels are a proxy for surface quality — "Persistent Weak Layer" and "no distinct problem" say little about how the snow skis. New/Wet/Wind-snow problems map cleanly to fresh/slush/wind-affected; the rest is weaker signal. The true surface-quality target comes from field observations (a later phase).
Core ML principles (non-negotiable — hold me to these)

Train on the past, predict on the forecast. Features = weather that actually happened, paired with observed conditions. At inference, feed the weather forecast to predict future conditions/score. Never train on forecast data.
Labels must be independent of the features. Do NOT derive labels from the same weather you train on — that's circular and the model just relearns your formula. Labels come from real condition/avalanche reports.
One row per region per day (the grain labels live at).
Never random-split for evaluation. Adjacent days are nearly identical, so a random split leaks. Use season-based holdout (train on older winters, test on the most recent ones).
Separate fetching from parsing. Network collection is slow/rate-limited — cache raw responses once (resumable), then parse locally as many times as you want.
Be honest about data quality. Drop boilerplate/off-season stubs; flag lossy fields rather than pretending they're clean.

The three data pillars

Weather (features) — Open-Meteo. Free, global, lat/lon-driven. Historical archive endpoint (archive-api.open-meteo.com/v1/archive) for training data; forecast endpoint (api.open-meteo.com/v1/forecast) for inference. Hourly temperature_2m, snowfall, wind_speed_10m, relative_humidity_2m, rolled to daily features.
Terrain / geological — elevation, aspect, slope, treeline, from a global DEM by lat/lon. Constant at a single point (so not useful for a single-location model), but becomes real signal when modeling per aspect × elevation (avalanche danger is already reported on an aspect×elevation "rose") and when pooling multiple regions.
Condition reports (the target / ground truth) — avalanche-center forecasts and field observations. These are the labels. Where no avalanche center exists (e.g. Australia), fall back to resort snow reports or user reports.

Label strategy: canonical schema + region adapters
Label sources differ wildly by country and language, but avalanche problem types are close to internationally standardized. So define one region-agnostic schema and have each region's scraper map into it.
Canonical row:
date, region_id, lat, lon, danger (1–5 | NaN), problem_1, problem_2, problem_3, source
Canonical problem taxonomy:
new_snow, wind_slab, persistent_slab, wet_snow, glide, cornice, loose_dry, no_distinct_problem
Each region = a config entry + an adapter that emits canonical rows. Adding Japan/Australia = a new adapter file, not a new pipeline.
yaml# regions.yaml
utah_slc:  {lat: 40.5884, lon: -111.6386, label_source: uac, uac_region: salt-lake}
# colorado: {lat: ..., lon: ..., label_source: caic}
# niseko:   {lat: ..., lon: ..., label_source: niseko}
# perisher: {lat: ..., lon: ..., label_source: resort_report}   # Australia
Reference implementation: Utah Avalanche Center (region #1, validated)
This region is already working end-to-end; use it as the template.

Current forecast JSON: https://utahavalanchecenter.org/forecast/{region}/json (regions: logan, ogden, salt-lake, provo, uintas, skyline, moab, abajos). Requires a descriptive User-Agent with contact email or it 400s. This endpoint is current-only — appending a date/id is ignored.
Historical forecasts: only as HTML at https://utahavalanchecenter.org/forecast/{region}/{M}/{D}/{YYYY} (no zero-padding). Enumerate them from the archive: https://utahavalanchecenter.org/archives/forecasts/{region}?page=N (50/page, paginate to ~2018). De-dup same-day updates (-0/-1 suffixes) to one per calendar day.
Parse the HTML: avalanche problems are the <h5> after each "Avalanche Problem #N" label (catches "Normal Caution"); danger is the bolded word (Low/Moderate/Considerable/High/Extreme) in the summary (lossy ~26% — treat as secondary). Cache every HTML page; parse separately.
Result: ~1,300 labeled days, 8 seasons. Distribution skews to Persistent Weak Layer + Wind Slab.
Other US/Canada centers use similar platforms; Avalanche Canada has a documented JSON products API (docs.avalanche.ca). The consolidated US danger API (api.avalanche.org/v2/public/products/map-layer/{CENTER}) is current danger only — not a historical label source.

Feature engineering (weather → daily)
From hourly weather, per day: snowfall_24h; rolling snow_48h/72h/7d/14d; temp_min/max/mean; wind_max/mean; humidity_mean; snow_temp_mean (mean temp during snowfall hours = density proxy, colder ⇒ lighter/drier); freeze_thaw (crossed 0 °C both ways); days_since_snow.
Watch the season boundary: Northern-Hemisphere winter crosses Jan 1, so group rolling windows by a "season" = the year the winter started (month ≥ 7 ⇒ that year, else year−1). Southern Hemisphere (Australia) winter sits inside one calendar year — handle per region.
Model design

One pooled model, not one per region. Snow physics is universal, so a single model trained across all regions — with region_id, latitude, elevation, and aspect as features — lets data-poor regions (Australia: ~8 resorts, short season, ~no avalanche forecasts) borrow strength from data-rich ones (Utah, Japan, Canada).
Baseline: XGBoost multiclass on the primary canonical problem (drop off-season/empty rows and singleton classes). Secondary target: danger (ordinal).
Evaluation: season-based holdout plus leave-one-region-out — train on all regions except one, predict the held-out region. That is literally the "will it work in Australia before Australia has its own data" test.
Powder Score: function of predicted class probabilities (↑ new_snow, ↓ wet_snow/wind_slab, ↓ high danger), scaled 0–100.

Repo structure
regions.yaml
collectors/
  weather.py            # shared: lat/lon -> daily weather features (Open-Meteo)
  terrain.py            # shared: lat/lon -> elevation, aspect (DEM)
  labels/
    uac.py              # adapter #1 (done): UAC -> canonical
    caic.py  niseko.py  # later regions
core/
  schema.py             # canonical schema + validation
  features.py           # weather/terrain -> model features
  dataset.py            # join features + canonical labels across regions
  train.py              # pooled, region-aware model
  evaluate.py           # season holdout + leave-one-region-out
data/                   # per-region raw caches + processed
models/                 # versioned artifacts
Repeatable collect → retrain loop
Each region issues a forecast daily. The pipeline is: incremental, cached, resumable collect (per region) → features (shared) → assemble dataset → train → evaluate → version the model artifact. A config-driven entrypoint retrains across all regions on a schedule. Inference: weather forecast → model → predicted conditions → Powder Score, per resort.
Roadmap

Region #1 baseline (Utah): weather → conditions model, season-holdout eval, read per-class F1. (Done/in progress.)
Wrap as adapter + add region #2: Colorado/CAIC (easy, same platform) or a Japan source (stress-tests the adapter). Validate the canonical seam + leave-one-region-out.
Pool regions, add terrain features, move toward aspect×elevation resolution.
Add field observations as a richer condition source → a true surface-quality / powder target.
Productize: inference service, per-resort Powder Score + 7-day outlook, alerts; web/mobile app.

Productization & commercial notes

Build the core as a clean API (region in → daily conditions + Powder Score + outlook out) so the app is a thin client.
Check data licensing / terms of use before selling. Avalanche-center data is public/government-funded, but commercial redistribution and scraping terms vary by source and country — verify per region. Weather (Open-Meteo) has its own terms for commercial use. This is a real to-do for a paid product, not an afterthought. (I'm not asking for legal advice — just flag it when relevant.)
Likely product surface: powder alerts, resort rankings, multi-day Powder Score outlook, "best resort near me this weekend."

How I want you to work with me

Concrete, runnable code, one step at a time; explain the why, not just the how.
Hold the line on the core ML principles above — especially don't let me train on forecast data, train on weather-derived labels, or random-split for evaluation.
Prove each piece on a small slice before scaling (e.g. parse 20 cached files before fetching 1,300).
Be honest about what the data can and can't support; flag lurking issues early.

My immediate next step: [fill in — e.g. "train the region-#1 model and read the per-class scores" or "wire up the canonical schema + add region #2"].