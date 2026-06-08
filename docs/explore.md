# explore.py

**Purpose:** Generates 6 EDA plots from the training dataset parquet for pre-modelling data understanding.

**Inputs:**
- `data/processed/training_dataset.parquet`

**Outputs:**
- `data/plots/01_target_distribution.png` — overnight_snow_cm histogram (linear + log scale)
- `data/plots/02_feature_correlations.png` — Pearson r heatmap with target
- `data/plots/03_scatter_top_features.png` — scatter of top-8 correlated features vs target
- `data/plots/04_powder_by_resort.png` — powder day % and mean snowfall per resort
- `data/plots/05_snowfall_by_month.png` — monthly snowfall violin plots
- `data/plots/06_region_comparison.png` — CDF of overnight snow by region

**Key parameters / constants:**
- None — fully read-only EDA script

**Notes:**
- Uses `Agg` matplotlib backend so it can run headlessly (no display required).
- Run after any major change to the dataset or feature engineering to check that distributions look sensible.

**Last updated:** 2026-06-07
